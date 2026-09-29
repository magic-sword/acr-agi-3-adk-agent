"""Compare memory layouts with identical one-action planning/review stages."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark_planning_comparison import audit_run, report, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', default='ls20,vc33,ft09')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--hard-seconds', type=float, default=110)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('positive repeats required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    schedule = [dict(name=f'r{rep+1}-{game}-{layout}', repeat=rep+1, game=game,
                     arm='A' if layout == 'mixed' else 'B', memory_layout=layout)
                for rep in range(args.repeats) for game in args.games.split(',')
                for layout in (('mixed', 'separated') if rep % 2 == 0 else ('separated', 'mixed'))]
    write(output/'plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        schedule=schedule, seconds=args.seconds, steps=args.steps, decision_seconds=45,
        hard_seconds=args.hard_seconds, fixed_planning='B: one-action + separate review',
        A='mixed: last_result, last_review, recent 3 trials, abstract knowledge',
        B='separated: current scene, frozen active trial OR last completed trial, abstract knowledge',
        shared='SAM, pixel tracking, before/current images, stages, prompts, output schema, knowledge writes, execution and cursor',
        knowledge='at most 2 conditional model claims per review; last 8 shown; all retained in archive; no host truth certification',
        interpretation='Memory presentation package comparison, including reduced historical input. Not a test of stage removal or cross-game long-term learning.',
        evaluation='Inspect stale candidate references, stale observation facts, current prediction consistency, reviews without this attempt result, repeated actions, progression and cost. Raw grids are evidence, not a complete optimal-action oracle.'))
    results, reviews = [], []
    source_hashes = None
    for spec in schedule:
        env = {**os.environ, 'COGNITION_PLANNING_COMPARISON':'B',
               'COGNITION_MEMORY_COMPARISON':spec['memory_layout'],
               'COGNITION_PROPOSALS':'sam_initial', 'COGNITION_DECISION_SECONDS':'45',
               'COGNITION_REPAIR_ATTEMPTS':'1'}
        print('START '+spec['name'], flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/benchmark_local.py'),
            '--games', spec['game'], '--steps', str(args.steps), '--seconds', str(args.seconds),
            '--hard-seconds', str(args.hard_seconds), '--output', str(output/spec['name'])],
            cwd=ROOT, env=env, check=True)
        manifest = json.loads((output/spec['name']/'manifest.json').read_text())
        if source_hashes is None:
            source_hashes = manifest['source_sha256']
        if source_hashes != manifest['source_sha256']:
            raise RuntimeError('source changed during comparison')
        result, cases = audit_run(output/spec['name'], spec)
        logs = output/spec['name']/result['game_id']/'cognition'
        artifacts = [json.loads(line) for line in next(logs.glob('*.artifacts.jsonl')).read_text().splitlines()]
        result['knowledge_updates'] = sum(r['event']=='knowledge_updated' for r in artifacts)
        results.append(result)
        reviews.extend(cases)
        report(output, results, reviews)
        report_path = output/'report.md'
        text = report_path.read_text().replace('# Planning A/B comparison', '# Memory layout comparison')
        text = text.replace('A uses existing stages; B uses one-action planning and a separate review. Both share the experiment input presentation.',
            'A = mixed memory; B = separated memory. Both use identical one-action planning and separate review stages, prompts and output contracts.')
        report_path.write_text(text)
    print('Report: '+str(output/'report.md'), flush=True)


if __name__ == '__main__':
    main()
