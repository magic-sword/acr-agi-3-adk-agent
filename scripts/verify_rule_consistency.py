"""Replay SDK recordings through RuleLearner: online prediction hits and final replay consistency.

usage: python scripts/verify_rule_consistency.py RUN_DIR [...]
"""
from collections import defaultdict
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.cognition.rules import RuleLearner  # noqa: E402


def main(runs):
    totals = defaultdict(lambda: defaultdict(float))
    for run in runs:
        for rec in glob.glob(str(Path(run) / '*' / 'recordings' / '*' / '*.jsonl')):
            game = Path(rec).parts[-4][:4]
            rows = [json.loads(l)['data'] for l in open(rec)]
            rules = RuleLearner()
            for prev, cur in zip(rows, rows[1:]):
                act = cur.get('action_input') or {}
                if act.get('id') in (None, 'RESET') or cur.get('levels_completed') != prev.get('levels_completed'):
                    if rules.transitions:
                        c = rules.replay_consistency()
                        if c is not None:
                            totals[game]['replay_sum'] += c; totals[game]['replays'] += 1
                    rules = RuleLearner()
                    continue
                data = act.get('data') or {}
                xy = (data.get('x'), data.get('y'))
                before, after = prev['frame'][-1], cur['frame'][-1]
                p = rules.predict_outcome(before, act['id'], xy)
                if p is not None:
                    totals[game]['predictions'] += 1
                    totals[game]['hits'] += rules.check(p, before, after, act['id'], xy)[0]
                totals[game]['transitions'] += 1
                rules.learn(before, after, act['id'], xy)
            c = rules.replay_consistency()
            if c is not None:
                totals[game]['replay_sum'] += c; totals[game]['replays'] += 1
    print('|game|transitions|predictions made|hits|final replay consistency (mean)|')
    print('|---|---:|---:|---:|---:|')
    for g, t in sorted(totals.items()):
        print(f"|{g}|{int(t['transitions'])}|{int(t['predictions'])}|{int(t['hits'])} ({t['hits']/max(1,t['predictions']):.2f})|"
              f"{t['replay_sum']/max(1,t['replays']):.2f}|")


if __name__ == '__main__':
    main(sys.argv[1:])
