"""Replay annotated frames through the default runtime; no game/model actions.

Annotations are used only AFTER observation processing. Identity/history scores
use an oracle support match, not the planner's target choice or game success.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.cognition.simple_workflow import SimpleRuntime
from agent.cognition import object_memory as objects
from agent.cognition.region_masks import decode
from agent.cognition.sam_proposals import default_proposer
from scripts.recorded_proposer import RecordedProposer


class CachedMasks:
    def __init__(self, row):
        if row['sam_status'] != 'ready':
            raise ValueError('mask cache must come from a successful detector run')
        self.regions = sorted([o for o in row['objects']['objects'] if 'sam' in o['sources']],
                              key=lambda o: int(o['track_ids'][0].split('t')[-1]))

    def propose(self, grid):
        return dict(boxes=[[o['bbox'][0], o['bbox'][1], o['bbox'][2]+1, o['bbox'][3]+1] for o in self.regions],
                    masks=[o['mask_runs'] for o in self.regions], model='cached_verified_sam_native_masks')


def replay(source, output, sam, mask_cache=None):
    output.mkdir(parents=True, exist_ok=False)
    sequences = json.loads((source/'sequences.json').read_text())
    detections = {r['sequence']: r for r in map(json.loads, (source/'sam-results.jsonl').read_text().splitlines())
                  if r['frame'] == 0}
    cached = {r['sequence']: r for r in map(json.loads, mask_cache.read_text().splitlines()) if r['frame'] == 0} if mask_cache else {}
    code = list(Path('agent/cognition').glob('*.py')) + [Path(__file__)]
    manifest = dict(sam=sam, device=os.getenv('COGNITION_SAM_DEVICE', 'cuda'), source=str(source), source_sha256=hashlib.sha256((source/'sequences.json').read_bytes()).hexdigest(),
        mask_cache=str(mask_cache) if mask_cache else None,
        mask_cache_sha256=hashlib.sha256(mask_cache.read_bytes()).hexdigest() if mask_cache else None,
        code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        support_iou_threshold=.9,
        limitations=['real_partial annotations cannot measure precision or all missed objects',
                    'oracle support matching and history retrieval, not model selection or game progress',
                    'recorded SAM has boxes only; use live to evaluate native SAM masks'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    summaries = []
    with (output/'frames.jsonl').open('x') as stream:
        for sequence in sequences:
            proposer = (default_proposer() if sam == 'live' else CachedMasks(cached[sequence['name']]) if sam == 'cached'
                        else RecordedProposer(detections[sequence['name']]))
            runtime = SimpleRuntime(sequence['name'], proposal_mode='sam_initial', sam_proposer=proposer,
                                     seconds=600, decision_seconds=120, log_dir=None)
            totals = Counter(); elapsed = []; previous = {}; ledger = []; labels = {}
            try:
                for i, frame in enumerate(sequence['frames']):
                    start = time.perf_counter()
                    runtime._receive(dict(game_id=sequence['name'], step=i, grid=frame['grid'],
                        state='NOT_FINISHED', levels_completed=0, available_actions=['ACTION1'], remaining_actions=100-i))
                    if runtime.perception['sam']['status'] != 'ready':
                        raise RuntimeError(f"SAM verification requires a ready detector: {runtime.perception['sam']}")
                    elapsed.append(time.perf_counter()-start)
                    state = runtime.memory.object_memory
                    pixels = {o['object_id']: decode(o['mask_runs']) for o in state['objects']}
                    matched = {}; history_counts = Counter()
                    for gold in frame['gold']:
                        gp = {tuple(p) for p in gold['support']}
                        eligible = [o for o in state['objects'] if pixels[o['object_id']] and
                                    len(gp & pixels[o['object_id']]) / len(gp | pixels[o['object_id']]) >= .9]
                        eligible.sort(key=lambda o: (o['object_id'] != previous.get(gold['id']),
                            -len(gp & pixels[o['object_id']]) / len(gp | pixels[o['object_id']]), o['object_id']))
                        if not eligible:
                            continue
                        chosen = eligible[0]; ident = chosen['object_id']; matched[gold['id']] = ident
                        if gold['id'] in previous:
                            totals['matched_transitions'] += 1
                            totals['selected_id_switches'] += ident != previous[gold['id']]
                        for hit in objects.history(state, ledger, [ident], 'inspect')['trials']:
                            key = hit['trial']['invocation_id']
                            history_counts[hit['correspondence']] += 1
                            if hit['correspondence'] == 'tracked' and labels[key] != gold['id']:
                                history_counts['wrong_gold_tracked_history'] += 1
                        key = f'{i}:{gold["id"]}'
                        ledger.append(dict(observation_id=state['observation_id'], invocation_id=key,
                            verb='inspect', target_objects=[deepcopy(chosen)], target_object_ids=[ident]))
                        labels[key] = gold['id']
                    totals.update(history_counts)
                    totals['gold'] += len(frame['gold']); totals['support_recovered'] += len(matched)
                    totals['sam_calls'] += runtime.perception['sam']['called_this_frame']
                    row = dict(sequence=sequence['name'], frame=i, seconds=elapsed[-1],
                        sam_status=runtime.perception['sam']['status'],
                        matched=matched, gold=len(frame['gold']), history=dict(history_counts),
                        object_kinds=dict(Counter(o['kind'] for o in state['objects'])),
                        act_context_bytes=len(json.dumps(runtime._context('act'))),
                        objects=state)
                    totals['max_context_bytes'] = max(totals['max_context_bytes'], row['act_context_bytes'])
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    previous = matched
                summary = dict(sequence=sequence['name'], kind=sequence['kind'], frames=len(elapsed), **totals,
                               initial_seconds=elapsed[0], update_median_seconds=statistics.median(elapsed[1:]),
                               update_max_seconds=max(elapsed[1:]))
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
            finally:
                runtime.close()
    (output/'summary.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/temporal-proposals-20260928'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sam', choices=['recorded', 'live', 'cached'], default='live')
    parser.add_argument('--mask-cache', type=Path)
    args = parser.parse_args()
    if args.sam == 'cached' and args.mask_cache is None:
        parser.error('--sam cached requires --mask-cache from a successful live run')
    replay(args.source, args.output, args.sam, args.mask_cache)
