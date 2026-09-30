"""V10: does listing the relations the measured rules can produce improve goal hypotheses?

Rules are learned by replaying a benchmark run's recording up to its first hypothesis request
(the same knowledge the runtime had). The program enumerates relations the rules can make true
(Planner.reachable); the V8 format F-CE is asked with and without that list:

  R0  F-CE + measured effects (the runtime's current input)
  R1  R0 + achievable_relations (nearest first) and an instruction to prefer them
  R2  R0 + every achievable relation grouped by object and relation (no distance order),
      with a neutral instruction (the win may also need a rule not found yet)

Scored per sample: a proposed atom matches the ground truth (verify_hypothesis_formats.setting),
and the share of proposed atoms the rules can reach.

usage (dev container): python scripts/verify_reachable_relations.py [samples] [list | R0,R1,R2]
"""
import glob
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_hypothesis_prompting as v  # noqa: E402
import verify_hypothesis_formats as f  # noqa: E402
from agent.controls import ACTION_TO_BUTTON  # noqa: E402
from agent.cognition.goal_planner import Planner  # noqa: E402
from agent.cognition.perception import COLORS  # noqa: E402
from agent.cognition.predicates import describe  # noqa: E402
from agent.cognition.rules import RuleLearner, components, MAX_OBJECT_CELLS  # noqa: E402

RUN = ROOT / 'outputs/evaluations/20260930T181722152636Z'
OUT = ROOT / 'outputs/v10-reachable'
SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LIST_ONLY = len(sys.argv) > 2 and sys.argv[2] == 'list'
CONDS = tuple(sys.argv[2].split(',')) if len(sys.argv) > 2 and not LIST_ONLY else ('R0', 'R1')
GAMES = ('ls20', 'tu93', 'vc33', 'ft09', 'r11l', 'ka59')
LISTED = 40

ACHIEVABLE = ('\nachievable_relations lists relations that the program can already make true with the measured '
              'rules (nearest first, with the number of actions). The win condition is often one of them or a '
              'combination of them: prefer hypotheses built from achievable relations when they fit what the '
              'screen shows, and propose other relations only when the screen clearly suggests them.')

GROUPED = ('\nachievable_relations groups the relations the program can already make true with the measured '
           'rules, by object and relation. The win condition may be one of them or a combination, or it may '
           'need a rule not discovered yet; judge from what the screen shows.')


def grouped(reach):
    out = {}
    for atom, _ in reach:
        head = f"{atom['relation']}({atom['a']}, ...)" if atom.get('b') else f"{atom['relation']}({atom['a']})"
        out.setdefault(head, []).append(atom.get('b') or atom.get('color'))
    return [f"{head}: {', '.join(v)}" for head, v in out.items()]


def learned(game):
    """Rules from the run's transitions before its first hypothesis request."""
    arts = glob.glob(str(RUN / f'{game}*' / 'cognition' / '*.artifacts.jsonl'))[0]
    first = next(json.loads(l)['step'] for l in open(arts) if json.loads(l).get('event') == 'hypothesis_votes')
    rec = glob.glob(str(RUN / f'{game}*' / 'recordings' / '*' / '*.jsonl'))[0]
    rows = [json.loads(l)['data'] for l in open(rec)][:first + 1]
    rules = RuleLearner()
    for prev, cur in zip(rows, rows[1:]):
        act = cur['action_input']
        data = act.get('data') or {}
        rules.learn(prev['frame'][-1], cur['frame'][-1], act['id'], (data.get('x'), data.get('y')))
    available = [f'ACTION{a}' for a in rows[-1]['available_actions']]
    return rules, available, first


def object_cells(grid):
    comps = [c for c in components(grid) if len(c['cells']) <= MAX_OBJECT_CELLS]
    comps.sort(key=lambda c: (c['origin'][1], c['origin'][0]))
    return {f'o{i}': c['cells'] for i, c in enumerate(comps, 1)}


def facts(rules, grid, cells):
    owner = lambda comp: next((i for i, cs in cells.items() if comp['cells'] <= cs), COLORS[comp['color']])
    by_sig = {c['sig']: c for c in components(grid)}
    out = []
    movers = rules.movers()
    ids = sorted({owner(by_sig[s]) for s in movers if s in by_sig})
    for a, d in sorted(rules.direction_deltas(movers).items()):
        out.append(f"{ACTION_TO_BUTTON.get(a, a)} moves {', '.join(ids)} by {d}")
    for sig, outcomes in rules.click_outcomes.items():
        if sig not in by_sig:
            continue
        for _, moved, recolor in outcomes[-2:]:
            parts = [f"moved {owner(by_sig[m]) if m in by_sig else COLORS[m[0]]} by {d}" for m, d in moved.items()]
            if recolor is not None:
                parts.append(f'turned it {COLORS[recolor]}')
            out.append(f"clicking {owner(by_sig[sig])} ({COLORS[sig[0]]}): " + ('; '.join(parts) or 'no visible effect'))
    return out, {i for i, cs in cells.items() for s in movers if s in by_sig and by_sig[s]['cells'] <= cs}


def ask(game, objs, controllable, effects, listed, seed, instruction=ACHIEVABLE):
    payload = f.request(game, objs, controllable, 'F-CE', seed)
    text = json.loads(payload['messages'][1]['content'][1]['text'])
    text['measured_effects'] = effects
    if listed is not None:
        text['achievable_relations'] = listed
        payload['messages'][0]['content'] += instruction
    payload['messages'][1]['content'][1]['text'] = json.dumps(text, separators=(',', ':'))
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        v.API, json.dumps(payload).encode(), {'Content-Type': 'application/json'}), timeout=300).read())
    try:
        return json.loads(r['choices'][0]['message']['content'])
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results, summary = [], []
    for game in GAMES:
        grid = f.frame(game)
        objs = v.objects(grid)
        truth, _ = f.setting(game, objs)
        rules, available, first = learned(game)
        cells = object_cells(grid)
        effects, controllable = facts(rules, grid, cells)
        reach = Planner(rules, grid, cells, available).reachable()
        keys = {describe(a): n for a, n in reach}
        rank = next((k for k, (a, _) in enumerate(reach) if truth(a)), None)
        listed = [f'{describe(a)} in {n}' for a, n in reach[:LISTED]]
        summary.append(dict(game=game, steps_learned=first, effects=effects, reachable=len(reach),
                            truth_rank=rank, listed=listed))
        print(f"{game}: learned from {first} actions; {len(reach)} reachable relations; "
              f"truth at rank {rank}; effects {effects}", flush=True)
        if LIST_ONLY:
            continue
        variants = {'R0': (None, None), 'R1': (listed, ACHIEVABLE), 'R2': (grouped(reach), GROUPED)}
        for cond in CONDS:
            lst, instruction = variants[cond]
            for seed in range(SAMPLES):
                ans = ask(game, objs, controllable, effects, lst, seed, instruction)
                atoms = [a for h in (ans or {}).get('hypotheses', []) for a in h['atoms']]
                results.append(dict(game=game, cond=cond, seed=seed, valid=ans is not None,
                                    hit=any(truth(a) for a in atoms),
                                    reachable=sum(describe(a) in keys for a in atoms), atoms=len(atoms), answer=ans))
            rs = [x for x in results if x['game'] == game and x['cond'] == cond]
            print(f"  {cond}: hit {sum(x['hit'] for x in rs)}/{len(rs)}, reachable atoms "
                  f"{sum(x['reachable'] for x in rs)}/{sum(x['atoms'] for x in rs)}", flush=True)
    (OUT / 'lists.json').write_text(json.dumps(summary, indent=1))
    if LIST_ONLY:
        return
    (OUT / f"trials-{'-'.join(CONDS)}.json").write_text(json.dumps(results, indent=1))
    print('\n|game|' + '|'.join(f'{c} hit' for c in CONDS) + '|' + '|'.join(f'{c} reachable atoms' for c in CONDS) + '|')
    print('|---|' + '---:|' * (2 * len(CONDS)))
    for game in GAMES + ('total',):
        row = []
        for key in ('hit', 'reachable'):
            for cond in CONDS:
                rs = [x for x in results if x['cond'] == cond and (game == 'total' or x['game'] == game)]
                row.append(f"{sum(x['hit'] for x in rs)}/{len(rs)}" if key == 'hit' else
                           f"{sum(x['reachable'] for x in rs)}/{sum(x['atoms'] for x in rs)}")
        print(f'|{game}|' + '|'.join(row) + '|')


if __name__ == '__main__':
    main()
