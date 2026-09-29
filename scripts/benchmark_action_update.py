"""Counterbalanced four-arm planning input x decision-procedure comparison."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark_planning_comparison import audit_run, read_rows, write

ROOT = Path(__file__).resolve().parents[1]


def analyze_run(output, spec):
    result, reviews = audit_run(output/spec['name'], spec)
    logs = output/spec['name']/result['game_id']/'cognition'
    rows = read_rows(next(logs.glob('*.artifacts.jsonl')))
    plans = [r for r in rows if r['event'] == 'update_plan']
    trials = [r for r in rows if r['event'] == 'update_trial']
    feedback = [r['trial'] for r in rows if r['event'] == 'action_feedback' and not r['trial'].get('boundary')]
    comparisons = []
    for row in plans:
        prior = [r['trial'] for r in trials if r['sequence'] < row['sequence']]
        if not prior:
            continue
        old, new = prior[-1], row['trial']
        same_names = bool(new['target_names']) and new['target_names'] == old['target_names']
        same_query = bool(new['action'].get('target_query')) and new['action'].get('target_query') == old['action'].get('target_query')
        same_control = new['action']['action'] == old['action']['action']
        comparisons.append(dict(
            trial_id=new['trial_id'], previous_trial_id=old['trial_id'],
            no_visible_change=old['measured_result']['changed_cell_count'] == 0,
            same_visible_conditions=new['condition_frame_hash'] == old['condition_frame_hash'],
            repeated_intended_target_control=same_control and
                (new['action']['action'] != 'CLICK' or same_names or same_query),
            target_identity_resolved=bool(new['target_names']) and bool(old['target_names']),
            previous=old, next=new))
    result.update(plans=len(plans), replans=len(comparisons),
        no_change_replans=sum(c['no_visible_change'] for c in comparisons),
        no_change_intended_repeats=sum(c['no_visible_change'] and c['same_visible_conditions']
            and c['repeated_intended_target_control'] for c in comparisons),
        changed_transitions=sum((f.get('changed_cell_count') or 0) > 0 for f in feedback),
        no_change_transitions=sum(f.get('changed_cell_count') == 0 for f in feedback),
        acknowledged_transitions=sum(f.get('acknowledged', False) for f in feedback))
    write(output/spec['name']/'action-update-audit.json', dict(comparisons=comparisons, feedback=feedback))
    return result, reviews


def report(output, results, reviews):
    sums = ('actions', 'levels_completed', 'model_calls', 'plans', 'replans', 'no_change_replans',
            'no_change_intended_repeats', 'identical_frame_action_repeats', 'changed_transitions',
            'no_change_transitions', 'acknowledged_transitions', 'invalid_calls')
    aggregates = {}
    for arm in 'ABCD':
        rows = [r for r in results if r['arm'] == arm]
        aggregates[arm] = dict(runs=len(rows), **{k: sum(r[k] or 0 for r in rows) for k in sums},
                               stops=dict(Counter(r['stop_reason'] for r in rows)))
    write(output/'summary.json', dict(runs=results, aggregates=aggregates))
    write(output/'review-cases.json', reviews)
    lines = ['# Action update 2x2 comparison', '',
             'A: current; B: named regions + trial ledger; C: current + short procedure; D: both.',
             'Same cursor/executor/review/schema; no repeat rejection. Repeated actions are not automatically errors.', '',
             '| Run | Actions | Levels | Calls | Replans after no change | Intended repeats after no change | Changed transitions |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| [{r['name']}]({r['name']}/report.md) | {r['actions']} | {r['levels_completed']} | {r['model_calls']} | "
                     f"{r['no_change_replans']} | {r['no_change_intended_repeats']} | {r['changed_transitions']} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--games', default='ft09,ls20,vc33')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--seconds', type=float, default=90)
    p.add_argument('--hard-seconds', type=float, default=115)
    p.add_argument('--steps', type=int, default=8)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    schedule = []
    for rep in range(args.repeats):
        for gi, game in enumerate(args.games.split(',')):
            order = 'ABCD'[gi%4:]+'ABCD'[:gi%4]
            if rep%2:
                order = order[::-1]
            schedule += [dict(name=f'r{rep+1}-{game}-{arm}', repeat=rep+1, game=game, arm=arm) for arm in order]
    write(output/'plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), schedule=schedule,
        seconds=args.seconds, hard_seconds=args.hard_seconds, steps=args.steps, decision_seconds=45,
        factors=dict(representation='A/C: existing separated context; B/D: all neutral named regions + last 3 trial records and episode counts',
                     procedure='A/B: existing instructions; C/D: same plus 4 short decision rules'),
        shared='Qwen3-VL-4B Q4, temp 0, SAM/tracking, before/current images, schema, review and knowledge writes, fast cursor and execution',
        controls='No host repeat veto, no additional states, no hidden rules or game source inspection',
        limitations='Representation is a package: naming, history window, numeric-location reduction and compact measurements. '
                    'Not a pure name-only ablation. 2 repeats at seed 0 are not independent games. '
                    'Observed repeats are audited separately from unjustified repeats; no optimal-action oracle.',
        metrics='Intended target/control repeats after no change; actual action repeats on same frame; '
                'review/action consistency; justified retries; model/tool errors; changed transitions and levels.'))
    results, reviews, hashes = [], [], None
    for spec in schedule:
        env = {**os.environ, 'COGNITION_PLANNING_COMPARISON':'B', 'COGNITION_MEMORY_COMPARISON':'separated',
               'COGNITION_ACTION_UPDATE_COMPARISON':spec['arm'], 'COGNITION_PROPOSALS':'sam_initial',
               'COGNITION_DECISION_SECONDS':'45', 'COGNITION_REPAIR_ATTEMPTS':'1'}
        print('START '+spec['name'], flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/benchmark_local.py'), '--games', spec['game'],
                        '--steps', str(args.steps), '--seconds', str(args.seconds),
                        '--hard-seconds', str(args.hard_seconds), '--output', str(output/spec['name'])],
                       cwd=ROOT, env=env, check=True)
        manifest = json.loads((output/spec['name']/'manifest.json').read_text())
        if hashes is None:
            hashes = manifest['source_sha256']
        if hashes != manifest['source_sha256']:
            raise RuntimeError('agent source changed during comparison')
        result, cases = analyze_run(output, spec)
        if set(result['sam_status']) != {'ready'}:
            raise RuntimeError(f"SAM unavailable; do not continue the comparison: {result['sam_status']}")
        results.append(result)
        reviews.extend(cases)
        report(output, results, reviews)
    print(str(output/'report.md'), flush=True)


if __name__ == '__main__':
    main()
