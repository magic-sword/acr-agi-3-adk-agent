from copy import deepcopy
import unittest

import numpy as np

from agent.cognition import object_memory as objects
from agent.cognition.hybrid_perception import HybridPerception
from agent.cognition.proposal_tracking import ProposalTracker
from agent.cognition.region_masks import encode, decode, native_mask


def region(ident, track, points, color=8, source='component'):
    from agent.cognition.geometry import bounds
    box = bounds(points)
    return dict(id=ident, track_id=track, bbox=box, mask_runs=encode(points),
                color_ids=[color], sources=[source], shape='region_proposal')


def measurement(oid, before, candidates, changes=None, unresolved=None, status='measured'):
    return dict(observation_id=oid, before_observation_id=before, status=status,
                proposal_mode='sam_initial', candidates=candidates, changes=changes or [],
                unchanged=[], unresolved=unresolved or [])


def ledger(obj, **kw):
    return dict(observation_id=obj['observation_id'], invocation_id='attempt-1',
                target_objects=[deepcopy(obj)], target_object_ids=[obj['object_id']],
                verb='activate', target_query='left control', baseline='closed',
                assessment='probe_result', evidence='no visible change', **kw)


class ObjectMemoryTests(unittest.TestCase):
    def setUp(self):
        self.points = {(2, 2), (3, 2), (2, 3), (3, 3)}

    def test_native_mask_keeps_holes_and_is_not_a_filled_box(self):
        native = np.array([[1, 0, 1], [1, 0, 1]], dtype=bool)
        scaled = np.repeat(np.repeat(native, 6, axis=0), 6, axis=1)
        runs = native_mask(scaled, 2, 3)
        self.assertEqual(decode(runs), {(0, 0), (2, 0), (0, 1), (2, 1)})
        scaled[:3, :6] = False
        self.assertIn((0, 0), decode(native_mask(scaled, 2, 3)))

    def test_masks_with_identical_boxes_remain_distinct_and_invalid_input_is_atomic(self):
        tracker = ProposalTracker(); tracker.shape = (8, 8)
        box = [1, 1, 4, 4]
        tracker.add([box, box], 'sam', [[[1, 1, 4], [3, 1, 4]], [[1, 1, 4], [2, 1, 4], [3, 1, 4]]])
        self.assertEqual(len(tracker.tracks), 2)
        old = deepcopy(tracker.tracks)
        with self.assertRaises(ValueError):
            tracker.add([box, box], 'sam', [[[1, 1, 2]], [[2, 0, 4]]])
        self.assertEqual(tracker.tracks, old)

    def test_whole_and_parts_are_retained_with_containment_not_forced_merging(self):
        left = self.points
        right = {(x+2, y) for x, y in left}
        cs = [region('a', 'ta', left), region('b', 'tb', right, 9), region('whole', 'tw', left | right, source='sam')]
        state = objects.update(objects.empty(), measurement('o0', None, cs, status='initial'))
        self.assertEqual(len([o for o in state['objects'] if o['kind'] == 'region']), 3)
        self.assertTrue(any(o['kind'] == 'spatial_group' for o in state['objects']))
        self.assertTrue(all(r['kind'] == 'part_of_hypothesis' for r in state['relations']))
        whole = next(o for o in state['objects'] if o['candidate_refs'] == ['whole'])
        self.assertEqual(objects.allowed_refs(state, whole['object_id']), {'a', 'b', 'whole'})

    def test_tracked_translation_retrieves_old_target_trial_not_same_name_other_object(self):
        a = region('a', 'ta', self.points)
        b = region('b', 'tb', {(x+8, y) for x, y in self.points})
        state = objects.update(objects.empty(), measurement('o0', None, [a, b], status='initial'))
        old = deepcopy(state['objects'][0]); trial = ledger(old)
        newer = [ledger(state['objects'][1], number=i) for i in range(5)]
        moved = region('a1', 'ta', {(x+1, y) for x, y in self.points})
        state = objects.update(state, measurement('o1', 'o0', [moved, b]))
        retrieved = objects.history(state, [trial] + newer, [old['object_id']], 'activate')
        self.assertEqual(len(retrieved['trials']), 1)
        self.assertEqual(retrieved['trials'][0]['correspondence'], 'tracked')
        self.assertEqual(trial['target_objects'][0], old)

    def test_recolor_retains_tentative_identity_and_does_not_launder_history_next_frame(self):
        a = region('a', 'ta', self.points)
        state = objects.update(objects.empty(), measurement('o0', None, [a], status='initial'))
        old = deepcopy(state['objects'][0]); trial = ledger(old)
        b = region('b', 'tb', self.points, 9)
        state = objects.update(state, measurement('o1', 'o0', [b]))
        now = state['objects'][0]
        self.assertEqual(old['object_id'], now['object_id'])
        self.assertEqual(now['identity_status'], 'appearance_hypothesis')
        state = objects.update(state, measurement('o2', 'o1', [b]))
        self.assertEqual(state['objects'][0]['identity_status'], 'tracked')
        self.assertEqual(objects.history(state, [trial], [old['object_id']])['trials'][0]['correspondence'],
                         'tentative_identity_or_lineage')

    def test_repeated_review_of_one_invocation_is_not_multiple_trials(self):
        state = objects.update(objects.empty(), measurement('0', None, [region('a', 'ta', self.points)]))
        obj = state['objects'][0]
        trial = ledger(obj)
        records = [trial, {**trial, 'evidence': 'Later review of the same execution.'}]
        retrieved = objects.history(state, records, [obj['object_id']])['trials']
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]['trial']['evidence'], records[-1]['evidence'])

    def test_unexecuted_plan_is_not_retrieved_as_an_operation_trial(self):
        state = objects.update(objects.empty(), measurement('0', None, [region('a', 'ta', self.points)]))
        obj = state['objects'][0]
        records = [ledger(obj, execution_status='not_executed_or_unacknowledged')]
        self.assertEqual(objects.history(state, records, [obj['object_id']])['trials'], [])

    def test_composite_trial_with_one_uncertain_target_is_not_fully_tracked(self):
        cs = [region('a', 'ta', self.points), region('b', 'tb', {(x+5, y) for x, y in self.points})]
        state = objects.update(objects.empty(), measurement('0', None, cs))
        a, b = state['objects'][:2]
        trial = ledger(a); trial['target_objects'].append(deepcopy(b))
        changed = deepcopy(b); changed['object_id'] = 'new'; changed['predecessors'] = [b['object_id']]
        state['objects'] = [a, changed]
        retrieved = objects.history(state, [trial], [a['object_id'], 'new'])['trials']
        self.assertEqual(retrieved[0]['correspondence'], 'tentative_identity_or_lineage')

    def test_ambiguous_duplicate_cannot_rebind_even_when_one_copy_is_at_old_location(self):
        a = region('a', 'ta', self.points)
        state = objects.update(objects.empty(), measurement('o0', None, [a], status='initial'))
        old = state['objects'][0]['object_id']
        cs = [region('a1', 't1', self.points), region('a2', 't2', {(x+5, y) for x, y in self.points})]
        state = objects.update(state, measurement('o1', 'o0', cs,
            unresolved=[dict(side='before', candidate=a, reason='multiple_exact_matches')]))
        self.assertNotIn(old, {o['object_id'] for o in state['objects']})

    def test_split_and_merge_are_tentative_lineage_not_copied_trials(self):
        left = {(2, 2), (2, 3)}; right = {(3, 2), (3, 3)}
        state = objects.update(objects.empty(), measurement('o0', None, [region('a', 'ta', left | right)], status='initial'))
        trial = ledger(state['objects'][0])
        state = objects.update(state, measurement('o1', 'o0', [region('b', 'tb', left), region('c', 'tc', right)]))
        parts = [o for o in state['objects'] if o['kind'] == 'region']
        self.assertEqual(len(parts), 2)
        for p in parts:
            h = objects.history(state, [trial], [p['object_id']])
            self.assertEqual(h['trials'][0]['correspondence'], 'tentative_identity_or_lineage')
        part_trials = [ledger(o) for o in parts]
        state = objects.update(state, measurement('o2', 'o1', [region('d', 'td', left | right)]))
        merged = state['objects'][0]
        self.assertTrue({p['object_id'] for p in parts} <= set(merged['predecessors']))
        self.assertTrue(all(t['correspondence'] == 'tentative_identity_or_lineage'
                            for t in objects.history(state, part_trials, [merged['object_id']])['trials']))

    def test_motion_group_needs_two_nonzero_translations_and_expires_on_divergence(self):
        state = objects.empty()
        previous = None
        old_cs = None
        for step, dx in enumerate((0, 1, 2)):
            cs = [region('a'+str(step), 'ta', {(x+dx, y) for x, y in self.points}),
                  region('b'+str(step), 'tb', {(x+dx+3, y) for x, y in self.points}, 9)]
            changes = [] if old_cs is None else [dict(before=a, after=b, delta_xy=[1, 0]) for a, b in zip(old_cs, cs)]
            state = objects.update(state, measurement(str(step), previous, cs, changes))
            self.assertEqual(any(o['kind'] == 'co_motion_group' for o in state['objects']), step == 2)
            previous, old_cs = str(step), cs
        cs = [old_cs[0], region('b3', 'tb', {(x+8, y) for x, y in self.points}, 9)]
        state = objects.update(state, measurement('3', '2', cs, [dict(before=old_cs[1], after=cs[1], delta_xy=[4, 0])]))
        self.assertFalse(any(o['kind'] == 'co_motion_group' for o in state['objects']))

    def test_stationary_contact_is_never_motion_evidence(self):
        cs = [region('a', 'ta', self.points), region('b', 'tb', {(x+2, y) for x, y in self.points})]
        state = objects.update(objects.empty(), measurement('0', None, cs))
        for step in range(1, 4):
            state = objects.update(state, measurement(str(step), str(step-1), cs))
        self.assertFalse(any(o['kind'] == 'co_motion_group' for o in state['objects']))

    def test_model_group_is_revised_when_parts_change_relative_positions(self):
        cs = [region('a', 'ta', self.points), region('b', 'tb', {(x+5, y) for x, y in self.points})]
        state = objects.update(objects.empty(), measurement('0', None, cs))
        target = objects.bind_target(state, dict(candidate_refs=['a', 'b']))
        group = next(o for o in state['objects'] if o['object_id'] == target['object_id'])
        trial = ledger(group)
        moved = [cs[0], region('b1', 'tb', {(x+8, y) for x, y in self.points})]
        state = objects.update(state, measurement('1', '0', moved))
        now = next(o for o in state['objects'] if o['object_id'] == group['object_id'])
        self.assertTrue(now['relative_layout_changed'])
        self.assertEqual(objects.history(state, [trial], [now['object_id']])['trials'][0]['correspondence'],
                         'tentative_identity_or_lineage')

    def test_reset_gap_and_missing_frame_expire_identity(self):
        c = region('a', 'ta', self.points)
        for status, before in [('boundary', 'o0'), ('measured', 'gap'), ('unavailable', 'o0')]:
            state = objects.update(objects.empty(), measurement('o0', None, [c]))
            old = state['objects'][0]['object_id']
            state = objects.update(state, measurement('o1', before, [] if status == 'unavailable' else [c], status=status))
            self.assertNotIn(old, {o['object_id'] for o in state['objects']})

    def test_measured_hit_uses_mask_and_effect_target_is_separate(self):
        cs = [region('a', 'ta', self.points), region('b', 'tb', {(6, 6)})]
        old = objects.update(objects.empty(), measurement('o0', None, cs))
        new = objects.update(deepcopy(old), measurement('o1', 'o0', cs))
        before = [[5]*8 for _ in range(8)]; after = deepcopy(before); after[6][6] = 9
        rows = objects.observations(old, new, before, after, dict(action='CLICK', x=2, y=2))
        self.assertTrue(rows[0]['clicked_previous_mask'])
        self.assertEqual(rows[0]['changed_pixels_on_previous_support'], 0)
        self.assertFalse(rows[1]['clicked_previous_mask'])
        self.assertEqual(rows[1]['changed_pixels_on_previous_support'], 1)

    def test_real_hybrid_mask_survives_translation_with_all_original_components(self):
        class Proposer:
            def propose(self, grid):
                return dict(boxes=[[2, 2, 5, 5]], masks=[[[2, 3, 4], [3, 2, 5], [4, 3, 4]]])
        def board(dx):
            g = [[5]*12 for _ in range(12)]
            for x, y in [(3, 2), (3, 4)]: g[y][x+dx] = 9
            for x in range(2, 5): g[3][x+dx] = 8
            return g
        sensor = HybridPerception(Proposer())
        first = sensor.measure(None, board(0), after_id='o0')
        sam = next(c for c in first['candidates'] if 'sam' in c['sources'])
        self.assertEqual(len(decode(sam['mask_runs'])), 5)
        self.assertGreaterEqual(len([c for c in first['candidates'] if 'component' in c['sources']]), 4)
        moved = sensor.measure(board(0), board(1), before_id='o0', after_id='o1')
        current = next(c for c in moved['candidates'] if c['track_id'] == sam['track_id'])
        self.assertEqual(decode(current['mask_runs']), {(x+1, y) for x, y in decode(sam['mask_runs'])})


if __name__ == '__main__':
    unittest.main()
