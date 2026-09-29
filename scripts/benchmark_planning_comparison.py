"""Counterbalanced bounded A/B runs with common perception and action execution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def audit_run(directory, spec):
    summary = json.loads((directory / 'summary.json').read_text())['games'][0]
    logs = directory / summary['game_id'] / 'cognition'
    artifacts = read_rows(next(logs.glob('*.artifacts.jsonl')))
    models = read_rows(next(logs.glob('*.model.jsonl')))
    requests = read_rows(next(logs.glob('*.requests.jsonl')))
    observations = read_rows(next(logs.glob('*.observations.jsonl')))
    actions = [r for r in artifacts if r['event'] == 'action_selected' and r['pending']['action']['action'] != 'RESET']
    accepted = [r for r in artifacts if r['event'] == 'stage_accepted']
    measured = {r['observation_id']: r['measurement'] for r in artifacts if r['event'] == 'objects_measured'}
    reviews = []
    for r in accepted:
        if r['work'] != 'reconcile': continue
        following = next((a for a in actions if a['sequence'] > r['sequence']), None)
        reviews.append(dict(run=spec['name'], arm=spec['arm'], game=spec['game'],
            observation_id=r['observation_id'], step=r['step'], sequence=r['sequence'],
            result=r['result'], measurement=measured.get(r['observation_id']),
            next_action=following['pending'] if following else None,
            source=str(logs.relative_to(directory))))
    sizes = [len(json.dumps(r.get('context', {}), ensure_ascii=False)) for r in models]
    image_counts = [sum(p.get('type') == 'image_url'
                       for m in r['request']['messages'] if isinstance(m.get('content'), list)
                       for p in m['content']) for r in requests]
    first = ((datetime.fromisoformat(actions[0]['timestamp']) -
              datetime.fromisoformat(observations[0]['timestamp'])).total_seconds() if actions else None)
    result = {**spec, **{key: summary.get(key) for key in (
        'game_id', 'actions', 'levels_completed', 'sdk_score_full_game', 'wall_seconds',
        'stop_reason', 'model_calls', 'model_http_requests', 'tokens', 'usage_recorded_calls',
        'consecutive_action_repeats', 'identical_frame_action_repeats')}}
    result.update(first_action_seconds=first, calls_by_work=dict(Counter(r['work'] for r in models)),
        invalid_calls=sum(not r.get('schema_valid', False) for r in models),
        rejections=[dict(work=r['work'], step=r['step'], reason=r['reason'])
                    for r in artifacts if r['event'] == 'task_rejected'],
        route_mismatches=sum((r['result']['plan'] is not None) != (r['result']['next'] == 'execute')
                             for r in accepted if r['work'] == 'ground'),
        accepted_reviews=len(reviews), max_context_characters=max(sizes, default=0),
        median_context_characters=statistics.median(sizes) if sizes else None,
        input_images=sum(image_counts), action_sequence=[r['pending'] for r in actions],
        sam_calls=sum(bool(m.get('sam', {}).get('called_this_frame')) for m in measured.values()),
        sam_status=dict(Counter(m.get('sam', {}).get('status') for m in measured.values())),
        memory_read_seconds=summary['fast_slow']['memory_read_seconds'])
    return result, reviews


def report(output, results, reviews):
    aggregates = {}
    for arm in ('A', 'B'):
        rows = [r for r in results if r['arm'] == arm]
        first = [r['first_action_seconds'] for r in rows if r['first_action_seconds'] is not None]
        aggregates[arm] = dict(runs=len(rows), runs_with_actions=sum(r['actions'] > 0 for r in rows),
            actions=sum(r['actions'] for r in rows), levels=sum(r['levels_completed'] for r in rows),
            accepted_reviews=sum(r['accepted_reviews'] for r in rows),
            model_calls=sum(r['model_calls'] for r in rows),
            wall_seconds=sum(r['wall_seconds'] for r in rows),
            first_action_seconds_median=statistics.median(first) if first else None,
            stop_reasons=dict(Counter(r['stop_reason'] for r in rows)))
    write(output / 'summary.json', dict(runs=results, aggregates=aggregates))
    write(output / 'review-cases.json', reviews)
    lines = ['# Planning A/B comparison', '',
        'A uses existing stages; B uses one-action planning and a separate review. Both share the experiment input presentation.',
        'Two temperature-zero repeats are not independent statistical samples. No optimal-action accuracy is inferred from action counts.', '',
        '| Run | Arm | Actions | Levels | Seconds | Model calls | Reviews | Stop |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in results:
        lines.append(f"| [{r['name']}]({r['name']}/report.md) | {r['arm']} | {r['actions']} | {r['levels_completed']} | "
                     f"{r['wall_seconds']:.1f} | {r['model_calls']} | {r['accepted_reviews']} | {r['stop_reason']} |")
    lines += ['', 'See summary.json for input volume, timing, source logs and errors; review-cases.json contains unscored review evidence.', '']
    (output / 'report.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--games', default='ls20,vc33,ft09')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--seconds', type=float, default=90)
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--hard-seconds', type=float, default=110)
    args = p.parse_args()
    if args.repeats < 1: p.error('positive repeats required')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    schedule = [dict(name=f'r{repeat+1}-{game}-{arm}', repeat=repeat+1, game=game, arm=arm)
                for repeat in range(args.repeats) for game in args.games.split(',')
                for arm in (('A', 'B') if repeat % 2 == 0 else ('B', 'A'))]
    write(output / 'plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), schedule=schedule,
        seconds=args.seconds, decision_seconds=45, hard_seconds=args.hard_seconds, steps=args.steps,
        perception='sam_initial', image_policy='same previous/current images; current-only for cursor',
        common_context='full candidate index for deliberation; lossless measurement interning',
        A='existing stage decisions, memory navigation and semantic questions',
        B='one-action planning, separate review, last 3 actual trials; no model memory navigation, concept stage, backchain, skill choice, semantic questions',
        shared_execution='same one-token execute_step, cursor, action validation and acknowledgement',
        interpretation='Package comparison, not an isolated causal estimate for each removed stage. No game-rule access.'))
    results = []; reviews = []; source_hashes = None
    for spec in schedule:
        env = {**os.environ, 'COGNITION_PLANNING_COMPARISON': spec['arm'],
               'COGNITION_PROPOSALS': 'sam_initial', 'COGNITION_DECISION_SECONDS': '45',
               'COGNITION_REPAIR_ATTEMPTS': '1'}
        command = [sys.executable, str(ROOT / 'scripts/benchmark_local.py'), '--games', spec['game'],
                   '--steps', str(args.steps), '--seconds', str(args.seconds),
                   '--hard-seconds', str(args.hard_seconds), '--output', str(output / spec['name'])]
        print('START ' + spec['name'], flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        manifest = json.loads((output / spec['name'] / 'manifest.json').read_text())
        if source_hashes is None: source_hashes = manifest['source_sha256']
        if source_hashes != manifest['source_sha256']: raise RuntimeError('source changed during comparison')
        result, cases = audit_run(output / spec['name'], spec)
        results.append(result); reviews.extend(cases)
        report(output, results, reviews)
    print('Report: ' + str(output / 'report.md'), flush=True)


if __name__ == '__main__': main()
