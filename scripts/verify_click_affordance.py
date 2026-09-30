"""V9: how to find an effective click in few actions?

Ground truth: each game is reset offline (SDK, no scored actions) and every candidate object is
clicked from the initial state; an object is effective when cells other than the step counter
change. Strategies are compared by the number of clicks until the first effective object:

  P  program salience (the runtime's orientation order: rare colour, small, top-left)
  Q  Qwen ranks objects that look pressable/clickable (affordances; Qwen3-VL was trained with
     labels such as "pressable"), in the V8 format: numbered image, bbox_2d 0-1000, JSON output
     with an analysis field first
  R  random order (expected value)

usage (dev container): python scripts/verify_click_affordance.py [samples]
"""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ['OPERATION_MODE'] = 'offline'
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402
import verify_hypothesis_prompting as v  # noqa: E402
from agent.cognition.rules import components, MAX_OBJECT_CELLS  # noqa: E402

OUT = ROOT / 'outputs/v9-click'
SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 5
GAMES = {'vc33': 'vc33-5430563c', 'ft09': 'ft09-0d8bbf25', 'r11l': 'r11l-495a7899', 'ka59': 'ka59-38d34dbb'}


def status_bar(c):
    xs = [x for x, _ in c['cells']]
    ys = [y for _, y in c['cells']]
    return (max(ys) - min(ys) < 2 and max(xs) - min(xs) >= 15) or (max(xs) - min(xs) < 2 and max(ys) - min(ys) >= 15)


def click(env, x, y):
    env.reset()
    return env.step(GameAction.ACTION6, data={'x': x, 'y': y})


def ground_truth(arcade, game_id):
    env = arcade.make(game_id)
    start = env.reset().frame[-1]
    comps = [c for c in components(start) if len(c['cells']) <= MAX_OBJECT_CELLS and not status_bar(c)]
    # A click on the largest background region measures what changes regardless (the counter).
    background = max(components(start), key=lambda c: len(c['cells']))
    bx, by = min(background['cells'], key=lambda p: (p[1], p[0]))
    after = click(env, bx, by).frame[-1]
    counter = {(x, y) for y in range(64) for x in range(64) if after[y][x] != start[y][x]}
    effective = []
    for c in comps:
        x, y = min(c['cells'], key=lambda p: (p[1], p[0]))
        after = click(env, x, y).frame[-1]
        changed = {(i, j) for j in range(64) for i in range(64) if after[j][i] != start[j][i]} - counter
        effective.append(bool(changed))
    return start, comps, effective


def program_order(comps):
    small = [i for i, c in enumerate(comps) if len(c['cells']) <= 64]
    colors = Counter(comps[i]['color'] for i in small)
    return sorted(small, key=lambda i: (colors[comps[i]['color']], len(comps[i]['cells']),
                                        comps[i]['origin'][1], comps[i]['origin'][0]))


def qwen_order(grid, comps, seed):
    objs = v.objects(grid)   # same components, numbered o1.. in screen order
    by_cells = {frozenset((x, y) for x in range(o['bbox'][0], o['bbox'][2]+1) for y in range(o['bbox'][1], o['bbox'][3]+1)): o['id'] for o in objs}
    index = {}
    for i, c in enumerate(comps):
        xs = [x for x, _ in c['cells']]
        ys = [y for _, y in c['cells']]
        box = [min(xs), min(ys), max(xs), max(ys)]
        ident = next((o['id'] for o in objs if o['bbox'] == box and o['color'] == v.COLORS[c['color']]), None)
        if ident:
            index[ident] = i
    ids = [o['id'] for o in objs if o['id'] in index]
    img, _ = v.image(grid, objs, True)
    rows = [dict(id=o['id'], color=o['color'], bbox_2d=[round(k * 1000 / 63) for k in o['bbox']], cells=o['cells'])
            for o in objs if o['id'] in index]
    system = ('You play an unknown turn-based puzzle game where you can click cells. Nobody tells you the rules. '
              'Object numbers are drawn on the image at each object\'s top-left corner (the number after "o"). '
              'bbox_2d coordinates are normalised to 0-1000 over the board. Rank up to six objects that most likely '
              'do something when clicked: buttons, switches, levers, handles, selectable pieces or tiles. Status bars '
              'and counters along the edges are not controls. Write your reasoning in analysis first.')
    schema = {'type': 'object', 'properties': {
        'analysis': {'type': 'string', 'maxLength': 600},
        'ranking': {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': {'type': 'string', 'enum': ids}}},
        'required': ['analysis', 'ranking']}
    payload = {'model': 'qwen3-vl-4b-instruct', 'temperature': 0.7, 'seed': seed, 'max_tokens': 900,
               'response_format': {'type': 'json_schema', 'json_schema': {'name': 'clicks', 'schema': schema}},
               'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': [
                   {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + img}},
                   {'type': 'text', 'text': json.dumps(dict(objects=rows), separators=(',', ':'))}]}]}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(v.API, json.dumps(payload).encode(),
                                          {'Content-Type': 'application/json'}), timeout=300).read())
    try:
        ranking = json.loads(r['choices'][0]['message']['content'])['ranking']
    except Exception:
        ranking = []
    return [index[i] for i in dict.fromkeys(ranking) if i in index]


def class_order(comps, key):
    """One click per appearance class, most repeated class first (repeated elements are the usual controls)."""
    groups = {}
    for i, c in enumerate(comps):
        groups.setdefault(key(c), []).append(i)
    ranked = sorted(groups.values(), key=lambda g: (-len(g), min(comps[i]['origin'][1] for i in g)))
    firsts = [g[0] for g in ranked]
    rest = [i for g in ranked for i in g[1:]]
    return firsts + rest


def first_hit(order, effective):
    return next((n for n, i in enumerate(order, 1) if effective[i]), None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, arc_api_key='local',
                    environments_dir=str(ROOT / 'environment_files'))
    rows = []
    print('|game|candidates|effective|P: clicks to first effective|Q: first effective within its ranking (per sample)|Q top-3 hit rate|R: expected clicks|D: appearance classes, most repeated first|E: shape classes, most repeated first|')
    print('|---|---:|---:|---:|---|---:|---:|---:|---:|')
    for game, game_id in GAMES.items():
        grid, comps, effective = ground_truth(arcade, game_id)
        n, k = len(comps), sum(effective)
        p = first_hit(program_order(comps), effective)
        q = [first_hit(qwen_order(grid, comps, seed), effective) for seed in range(SAMPLES)]
        d = first_hit(class_order(comps, lambda c: c['sig']), effective)
        e = first_hit(class_order(comps, lambda c: c['sig'][1]), effective)
        top3 = sum(1 for h in q if h is not None and h <= 3)
        expected = (n + 1) / (k + 1) if k else None
        rows.append(dict(game=game, candidates=n, effective=k,
                         effective_objects=[dict(color=int(comps[i]['color']), origin=[int(k) for k in comps[i]['origin']],
                                                 cells=len(comps[i]['cells'])) for i in range(n) if effective[i]],
                         program=p, qwen=q, random_expected=expected, by_appearance_class=d, by_shape_class=e))
        print(f"|{game}|{n}|{k}|{p if p else 'not in its list'}|{q}|{top3}/{SAMPLES}|"
              f"{expected and round(expected, 1)}|{d}|{e}|", flush=True)
    (OUT / 'results.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
