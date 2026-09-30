"""Compare evaluation runs on the memory-verification metrics (docs/memory-architecture-ja.md §5).

usage: python scripts/compare_memory_runs.py LABEL=DIR[,DIR...] [LABEL=DIR...]
"""
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
import re
import statistics
import sys

# Evidence that reports any measured change; "X remained unchanged" alongside moved objects is faithful.
REPORTS_CHANGE = re.compile(r'mov|chang|cells?\b|pixels?|shift|appear|disappear|grew|shr[iu]nk|expand|color', re.I)
CLAIMS_MOVEMENT = re.compile(r'\b(moved|movement|shifted)\b', re.I)
DENIES_MOVEMENT = re.compile(r'\b(no|not|without|nor)\b[^.]{0,30}\b(move|moved|movement|shift)', re.I)
STAGES = ('understand', 'backchain', 'candidates', 'ground', 'reconcile', 'execute_step')


def rows(path, kind):
    for f in glob.glob(str(Path(path) / '*' / 'cognition' / f'*.{kind}.jsonl')):
        game = Path(f).parents[1].name
        for line in open(f):
            yield game, json.loads(line)


def game_metrics(run):
    out = defaultdict(lambda: defaultdict(float))
    no_change = defaultdict(set)
    for game, r in rows(run, 'model'):
        g = out[game]
        work = r.get('work')
        tokens = (r.get('usage') or {}).get('prompt_tokens')
        if tokens:
            g[f'{work}.prompt_tokens'] += tokens
            g[f'{work}.calls_measured'] += 1
        g[f'{work}.seconds'] += r.get('seconds', 0)
        g[f'{work}.calls'] += 1
    for game, r in rows(run, 'artifacts'):
        g = out[game]
        event = r['event']
        if event == 'concrete_action_selected':
            g['planned_actions'] += 1
        elif event == 'fallback_action_selected':
            g['fallback_actions'] += 1
            g[f"fallback.{r['reason']}"] += 1
        elif event == 'understanding_carried':
            g['understanding_carried'] += 1
        elif event == 'action_feedback':
            trial = r['trial']
            key = (trial['action'].get('action'), trial['action'].get('x'), trial['action'].get('y'))
            # Same concrete action again after it already produced no change (negative knowledge unused).
            if trial.get('acknowledged') and trial.get('changed_cell_count') == 0:
                if key in no_change[game]:
                    g['repeated_no_change'] += 1
                no_change[game].add(key)
        elif event == 'stage_accepted' and r['work'] == 'reconcile':
            g['reconciles'] += 1
    # Contradiction: an attempt of that invocation changed cells but the evidence reports no change at all.
    for game, r in rows(run, 'model'):
        if r.get('work') != 'reconcile' or not r.get('schema_valid'):
            continue
        attempts = (r.get('context') or {}).get('attempt_results', [])
        changed = any((a.get('changed_cell_count') or 0) > 0 for a in attempts)
        evidence = json.loads(r.get('response') or '{}').get('evidence', '')
        if changed and not REPORTS_CHANGE.search(evidence.replace('unchanged', '')):
            out[game]['reconcile_contradicts_measurement'] += 1
        # Reverse: nothing changed in any acknowledged attempt, yet the evidence asserts movement.
        if attempts and not changed and CLAIMS_MOVEMENT.search(evidence) and not DENIES_MOVEMENT.search(evidence):
            out[game]['reconcile_claims_unmeasured_change'] += 1
    for game in list(out):
        result = json.load(open(Path(run) / game / 'result.json')) if (Path(run) / game / 'result.json').exists() else {}
        g = out[game]
        g['actions'] = result.get('actions', result.get('action_count', 0))
        g['levels'] = result.get('levels_completed', 0)
        g['wall_seconds'] = result.get('wall_seconds', 0)
        log = Path(run) / game / 'worker.log'
        # llama-server rejects over-long prompts with HTTP 400 (context size exceeded).
        g['context_overflow'] = log.read_text(errors='replace').count('HTTP Error 400') if log.exists() else 0
    return out


def summary(runs):
    per_run = [game_metrics(r) for r in runs]
    games = sorted({g for m in per_run for g in m})
    table = {}
    for game in games:
        keys = sorted({k for m in per_run for k in m.get(game, {})})
        table[game] = {k: [m.get(game, {}).get(k, 0) for m in per_run] for k in keys}
    return table


def main(argv):
    labelled = {}
    for arg in argv:
        label, dirs = arg.split('=', 1)
        labelled[label] = dirs.split(',')
    tables = {label: summary(dirs) for label, dirs in labelled.items()}
    games = sorted({g for t in tables.values() for g in t})
    headline = ['actions', 'levels', 'planned_actions', 'fallback_actions', 'context_overflow',
                'repeated_no_change', 'reconcile_contradicts_measurement', 'reconcile_claims_unmeasured_change', 'understanding_carried']
    for game in games:
        print(f'\n## {game}')
        print('|metric|' + '|'.join(labelled) + '|')
        print('|---|' + '---|' * len(labelled))
        keys = headline + [f'{s}.prompt_tokens/call' for s in STAGES] + [f'{s}.seconds/call' for s in STAGES[:5]]
        for key in keys:
            cells = []
            for label in labelled:
                t = tables[label].get(game, {})
                if key.endswith('/call'):
                    stage, what = key[:-5].split('.')
                    num = t.get(f'{stage}.{what}', [])
                    den = t.get(f'{stage}.calls_measured' if what == 'prompt_tokens' else f'{stage}.calls', [])
                    vals = [n / d for n, d in zip(num, den) if d]
                else:
                    vals = t.get(key, [])
                cells.append(' / '.join(f'{v:.0f}' if isinstance(v, float) and v >= 10 else f'{v:.3g}' for v in vals) or '-')
            print(f'|{key}|' + '|'.join(cells) + '|')


if __name__ == '__main__':
    main(sys.argv[1:])
