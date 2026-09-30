"""V3 (docs/backchain-review-ja.md §6): who should propose goal hypotheses?

  A  Qwen proposes 3 win conditions freely from the first screen (no objective text given)
  B  Qwen picks 3 from candidates enumerated by the program
  C  the program's own ranking of the same candidates (control)

Inputs are the level-1 first frame of each game (the agent's rendered PNG plus the colour
grid from the SDK recording). Ground truth comes from the game source and is used only for
scoring (outputs/v3-goal/ground_truth.json); it never reaches the model.

usage:
  python scripts/verify_goal_hypotheses.py candidates   # write candidate lists to review
  python scripts/verify_goal_hypotheses.py run           # query Qwen (A, B), save raw outputs
  python scripts/verify_goal_hypotheses.py score         # score B/C; A needs judged file
"""
import base64
from collections import defaultdict
import glob
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_schema_prediction import components  # noqa: E402

COLORS = ['white', 'light gray', 'gray', 'dark gray', 'very dark gray', 'black', 'magenta', 'pink',
          'red', 'blue', 'light blue', 'yellow', 'orange', 'maroon', 'green', 'purple']
RUN = 'outputs/evaluations/20260930T013243650281Z'
OUT = Path('outputs/v3-goal')
# ls20+moved adds what the program measures after a few arrow presses (V1): the controllable object.
GAMES = {'ls20': dict(base='ls20', controls='UP, DOWN, LEFT, RIGHT', note='', mover=None),
         'ls20+moved': dict(base='ls20', controls='UP, DOWN, LEFT, RIGHT',
                            note=' Observed so far: the orange and blue block moves 5 cells per arrow press; '
                                 'nothing else moves.', mover=(34, 45, 38, 49)),  # bbox of the objects measured to move
         'vc33': dict(base='vc33', controls='CLICK at a cell', note='', mover=None),
         'ft09': dict(base='ft09', controls='CLICK at a cell', note='', mover=None)}
SAMPLES = [(0.0, 0), (0.8, 1), (0.8, 2), (0.8, 3), (0.8, 4)]
API = 'http://127.0.0.1:8080/v1/chat/completions'


def first_frame(game):
    rec = glob.glob(f'{RUN}/{game}*/recordings/*/*.jsonl')[0]
    return json.loads(open(rec).readline())['data']['frame'][-1]


def image_b64(game):
    return base64.b64encode((OUT / f"{GAMES[game]['base']}-level1.png").read_bytes()).decode()


def describe(o):
    xs = [x for x, _ in o['cells']]
    ys = [y for _, y in o['cells']]
    return f"{COLORS[o['color']]} object at x{min(xs)}-{max(xs)}, y{min(ys)}-{max(ys)}"


def centre(o):
    xs = [x for x, _ in o['cells']]
    ys = [y for _, y in o['cells']]
    return (sum(xs)/len(xs), sum(ys)/len(ys))


def dist(a, b):
    (ax, ay), (bx, by) = centre(a), centre(b)
    return abs(ax-bx) + abs(ay-by)


def enumerate_candidates(grid, limit=40, mover=None):
    """Program-only goal templates over measured objects, in the program's own priority order."""
    objs = [o for o in components(grid) if len(o['cells']) <= 600]
    out = []
    if mover:
        # A measured controllable object makes "bring it somewhere" the first template.
        x0, y0, x1, y1 = mover
        movers = [o for o in objs if all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in o['cells'])]
        name = ' and '.join(COLORS[m['color']] for m in movers)
        for b in sorted((o for o in objs if o not in movers and len(o['cells']) <= 150), key=lambda o: dist(movers[0], o))[:12]:
            out.append(f"Move the {name} block onto the {describe(b)}.")
    # 1. same-colour objects brought into line (row or column)
    pairs = sorted(((a, b) for i, a in enumerate(objs) for b in objs[i+1:] if a['color'] == b['color']), key=lambda p: dist(*p))
    for a, b in pairs[:4]:
        for axis in ('column', 'row'):
            out.append(f"Line up the {describe(a)} with the {describe(b)} in the same {axis}.")
    # 2. a salient small object (its colour occurs once) reaches another object, nearest first
    count = defaultdict(int)
    for o in objs:
        count[o['color']] += 1
    salient = [o for o in objs if count[o['color']] == 1 and len(o['cells']) <= 40]
    reach = sorted(((a, b) for a in salient for b in objs if b is not a and len(b['cells']) <= 150),
                   key=lambda p: (len(p[0]['cells']), dist(*p)))
    for a, b in reach[:16]:
        out.append(f"Move the {describe(a)} onto the {describe(b)}.")
    # 3. groups of repeated tiles: copy a pattern / make uniform
    by_shape = defaultdict(list)
    for o in objs:
        by_shape[o['sig'][1]].append(o)
    tiles = max(by_shape.values(), key=len)
    if len(tiles) >= 6:
        groups = []
        for t in sorted(tiles, key=centre):
            g = next((g for g in groups if any(dist(t, u) <= 12 for u in g)), None)
            (g.append(t) if g is not None else groups.append([t]))
        boxes = []
        for g in groups:
            xs = [x for o in g for x, _ in o['cells']]
            ys = [y for o in g for _, y in o['cells']]
            boxes.append(f"x{min(xs)}-{max(xs)}, y{min(ys)}-{max(ys)}")
        for a in boxes:
            for b in boxes:
                if a != b:
                    out.append(f"Recolour the tiles in the group at {a} so their colour pattern matches the group at {b}.")
        for a in boxes:
            out.append(f"Make every tile in the group at {a} the same colour.")
    # 4. clear a colour
    for c, n in sorted(count.items(), key=lambda kv: -kv[1])[:3]:
        if n >= 2:
            out.append(f"Remove every {COLORS[c]} object.")
    return out[:limit]


def ask(game, prompt, schema, temperature, seed):
    payload = {'model': 'qwen3-vl-4b-instruct', 'temperature': temperature, 'seed': seed, 'max_tokens': 400,
               'response_format': {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': schema}},
               'messages': [
                   {'role': 'system', 'content': 'You play an unknown turn-based puzzle game from its first screen. '
                    'Nobody tells you the rules or the goal. Available controls: ' + GAMES[game]['controls'] + '.'
                    + GAMES[game]['note'] + ' Image axes show grid cells 0-63; the bottom/top strips may be status bars.'},
                   {'role': 'user', 'content': [
                       {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image_b64(game)}},
                       {'type': 'text', 'text': prompt}]}]}
    req = urllib.request.Request(API, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    return json.loads(json.loads(urllib.request.urlopen(req, timeout=300).read())['choices'][0]['message']['content'])


PROMPT_A = ('What end state of this screen most likely wins the level? Give the 3 most likely win conditions, '
            'most likely first. Each must be a concrete end state of visible objects (colours, positions, '
            'relations) that could be checked on the screen, not "complete the level" or "explore".')
SCHEMA_A = {'type': 'object', 'properties': {'hypotheses': {'type': 'array', 'minItems': 3, 'maxItems': 3,
            'items': {'type': 'string', 'minLength': 10, 'maxLength': 220}}}, 'required': ['hypotheses']}


def main(mode):
    OUT.mkdir(exist_ok=True)
    cand_path = OUT / 'candidates.json'
    if mode == 'candidates':
        cands = {g: enumerate_candidates(first_frame(v['base']), mover=v['mover']) for g, v in GAMES.items()}
        cand_path.write_text(json.dumps(cands, indent=1))
        for g, c in cands.items():
            print(f'== {g}')
            for i, text in enumerate(c):
                print(f'  {i}: {text}')
        return
    cands = json.loads(cand_path.read_text())
    if mode == 'run':
        raw = {}
        for g in GAMES:
            listing = '\n'.join(f'{i}: {t}' for i, t in enumerate(cands[g]))
            prompt_b = ('Which of these candidate win conditions is most likely for this level? Choose the 3 most '
                        'likely, most likely first.\n' + listing)
            schema_b = {'type': 'object', 'properties': {'choices': {'type': 'array', 'minItems': 3, 'maxItems': 3,
                        'items': {'type': 'integer', 'enum': list(range(len(cands[g])))}}}, 'required': ['choices']}
            raw[g] = dict(A=[ask(g, PROMPT_A, SCHEMA_A, t, s)['hypotheses'] for t, s in SAMPLES],
                          B=[ask(g, prompt_b, schema_b, t, s)['choices'] for t, s in SAMPLES])
            print(g, json.dumps(raw[g], ensure_ascii=False, indent=1))
        (OUT / 'raw.json').write_text(json.dumps(raw, ensure_ascii=False, indent=1))
        return
    truth = json.loads((OUT / 'ground_truth.json').read_text())
    raw = json.loads((OUT / 'raw.json').read_text())
    judged = json.loads((OUT / 'judged_A.json').read_text())
    print('|game|A: correct in top 3 (top 1)|B: correct in top 3 (top 1)|C: correct in top 3 (top 1)|enumerator covers truth|')
    print('|---|---|---|---|---|')
    for g in GAMES:
        ok = set(truth[g]['correct_candidates'])
        a3 = sum(any(j) for j in judged[g]); a1 = sum(j[0] for j in judged[g])
        b3 = sum(bool(ok & set(c)) for c in raw[g]['B']); b1 = sum(c[0] in ok for c in raw[g]['B'])
        c3 = bool(ok & {0, 1, 2}); c1 = 0 in ok
        n = len(SAMPLES)
        print(f'|{g}|{a3}/{n} ({a1}/{n})|{b3}/{n} ({b1}/{n})|{int(c3)}/1 ({int(c1)}/1)|{"yes" if ok else "no"}|')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'candidates')
