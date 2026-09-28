"""Initial SAM proposals followed by pixel updates, in the runtime evidence schema."""
from collections import Counter
from copy import deepcopy
from itertools import groupby
import time

from .proposal_tracking import ProposalTracker
from .sam_proposals import default_proposer


class HybridPerception:
    def __init__(self, proposer=None):
        self.proposer = proposer if proposer is not None else default_proposer()
        self.tracker = None
        self.last_id = None
        self.previous_candidates = {}
        self.epoch = self.frame = 0
        self.sam_status = None
        self.bindings = []

    def bind_targets(self, understanding):
        """Pin validated observation-local references to conservative track IDs."""
        by_id = {o['id']: o['track_id'] for o in self.previous_candidates.values()}
        self.bindings = [dict(target_index=i, observation_id=understanding['observation_id'],
                              candidate_refs=list(t.get('candidate_refs', [])),
                              track_ids=[by_id[ref] for ref in t.get('candidate_refs', []) if ref in by_id])
                         for i, t in enumerate(understanding.get('targets', []))]

    def _candidate(self, track, grid):
        from .perception import COLORS
        x1, y1, x2, y2 = track['box']
        patch = [row[x1:x2] for row in grid[y1:y2]]
        colors = sorted(Counter(c for row in patch for c in row))
        detail = patch if (x2-x1)*(y2-y1) <= 100 else {
            'row_runs': [[[c, len(list(items))] for c, items in groupby(row)] for row in patch]}
        return dict(id=f'f{self.frame}t{track["id"]}', track_id=f'e{self.epoch}t{track["id"]}',
                    shape='region_proposal', bbox=[x1, y1, x2-1, y2-1],
                    size=[x2-x1, y2-y1], colors=[COLORS[c] for c in colors],
                    color_ids=colors, pattern=detail, sources=list(track['sources']),
                    support_note='Bounding rectangle pixels include background and may contain parts or multiple objects.')

    @staticmethod
    def _priority(candidate):
        area = candidate['size'][0]*candidate['size'][1]
        # Whole local SAM regions before their fragments; broad background last.
        return (area > 256, 'sam' not in candidate['sources'], -area, candidate['track_id'])

    def measure(self, before, after, *, before_id=None, after_id, action=None, boundary=False,
                seconds_left=None):
        start = time.perf_counter()
        result = dict(before_observation_id=before_id, observation_id=after_id, action=deepcopy(action),
                      status='unavailable', candidates=[], changes=[], unchanged=[], unresolved=[],
                      uncovered_changed_pixels=None, changed_pixels=None, requires_review=True,
                      proposal_mode='sam_initial',
                      identity_note='Unique local exact-pixel ROI correspondence, not physical identity or causal proof.')
        if not after:
            self.tracker = None
            self.previous_candidates = {}
            self.bindings = []
            self.last_id = after_id
            result['seconds'] = time.perf_counter()-start
            return result
        shape = (len(after), len(after[0]))
        initial = (self.tracker is None or boundary or self.last_id != before_id or
                   self.tracker.shape != shape or not before or self.tracker.previous.tolist() != before)
        if initial:
            self.tracker = ProposalTracker()
            self.previous_candidates = {}
            self.bindings = []
            self.epoch += 1
        self.frame += 1
        old = self.previous_candidates
        update = self.tracker.advance(after)
        sam_called = False
        if initial:
            sam_start = time.perf_counter()
            if seconds_left is not None and seconds_left < 6:
                self.sam_status = dict(status='skipped_budget', seconds=0,
                                       reason='Less than 6 seconds left at episode initialization')
            else:
                try:
                    sam_called = True
                    proposed = self.proposer.propose(after)
                    self.tracker.add(proposed['boxes'][:128], 'sam')
                    self.sam_status = {k: v for k, v in proposed.items() if k != 'boxes'}
                    self.sam_status.update(status='ready', proposals=len(proposed['boxes'][:128]),
                                           seconds=time.perf_counter()-sam_start)
                except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
                    self.sam_status = dict(status='unavailable', error=f'{type(exc).__name__}: {exc}'[:500],
                                           seconds=time.perf_counter()-sam_start)
        current = {t['id']: self._candidate(t, after) for t in self.tracker.tracks}
        candidates = sorted(current.values(), key=self._priority)
        result.update(candidates=candidates, extraction_limited=update['extraction_limited'],
                      sam={**self.sam_status, 'called_this_frame': sam_called},
                      proposal_coverage_incomplete=self.sam_status['status'] != 'ready')
        if initial:
            result['status'] = 'initial' if not before else 'boundary'
        else:
            linked = set()
            for link in update['links']:
                key = link['id']
                ob, oa = old[key], current[key]
                linked.add(key)
                delta = [link['after'][i]-link['before'][i] for i in (0, 1)]
                if any(delta):
                    result['changes'].append(dict(before=ob, after=oa,
                        changed=dict(position=True, appearance=False, size=False), delta_xy=delta,
                        correspondence='unique_exact_pixels_within_8px'))
                else:
                    result['unchanged'].append(dict(before_id=ob['id'], after_id=oa['id'], track_id=oa['track_id']))
            result['changes'].sort(key=lambda c: self._priority(c['after']))
            result['unresolved'] = [dict(side='before', candidate=old[u['track']['id']], reason=u['reason'])
                                    for u in update['unresolved']]
            result['unresolved'] += [dict(side='after', candidate=c, reason='new_region_or_changed_appearance')
                                     for key, c in current.items() if key not in linked]
            residual = update['uncovered_positions']
            result.update(status='measured', changed_pixels=update['changed_pixels'],
                          uncovered_changed_pixels=len(residual), uncovered_positions=residual,
                          requires_review=bool(residual or result['unresolved'] or update['extraction_limited']))
        # Keep target grounding separate from hypotheses. Lost IDs are never
        # rebound to a newly detected region just because it has the same color.
        by_track = {c['track_id']: c['id'] for c in candidates}
        result['target_correspondence'] = [dict(b,
            current_candidate_refs=[by_track[t] for t in b['track_ids'] if t in by_track],
            missing_track_ids=[t for t in b['track_ids'] if t not in by_track],
            status='tracked' if b['track_ids'] and all(t in by_track for t in b['track_ids']) else 'unresolved')
            for b in self.bindings]
        self.previous_candidates = current
        self.last_id = after_id
        result['seconds'] = time.perf_counter()-start
        return result
