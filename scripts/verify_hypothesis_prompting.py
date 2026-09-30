"""V7: does visual grounding (Set-of-Mark) and a VCGT-derived question checklist improve
Qwen's goal hypotheses? Offline: no game actions are spent.

Conditions (5 samples each at temperature 0.7, the runtime setting):
  C0  runtime prompt: plain image + object list (ids, colours, boxes) + relation vocabulary
  C1  C0 with object IDs drawn on the image (Set-of-Mark)
  C2  C1 plus generic questions from the human VCGT thinking patterns (roles, examples, relations)
  C3  C2 plus a roles field the model fills before its hypotheses

Ground truth comes from the game sources and is used only for scoring. The VCGT annotations of
ls20/vc33/ft09 are not used anywhere (they would leak the answers); only generic questions are.

usage (dev container): python scripts/verify_hypothesis_prompting.py
"""
import base64
import glob
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import ImageDraw  # noqa: E402
from agent.rendering import render_current, ORIGIN, SCALE  # noqa: E402
from agent.cognition.goal_runtime import HYPOTHESIZE  # noqa: E402
from agent.cognition.perception import COLORS  # noqa: E402
from agent.cognition.predicates import RELATIONS, BINARY  # noqa: E402
from agent.cognition.rules import components, MAX_OBJECT_CELLS  # noqa: E402
from scripts.fonts import ui_font  # noqa: E402

RUN = ROOT / 'outputs/evaluations/20260930T013243650281Z'
OUT = ROOT / 'outputs/v7-hypothesis-prompting'
API = os.getenv('VLM_API_BASE', 'http://vlm:8080/v1') + '/chat/completions'
SAMPLES = 5
CONTROLS = {'ls20': 'UP, DOWN, LEFT, RIGHT', 'vc33': 'CLICK at a cell', 'ft09': 'CLICK at a cell'}

CHECKLIST = '''Before proposing, look at the image and answer for yourself:
1. Which object can be controlled or clicked, if known?
2. Which objects look like a target, slot, frame, exit or container?
3. Which objects look like an example, key or pattern to copy or match (for instance a small
   icon showing a shape or colours)?
4. Which objects share a colour or shape but are not yet placed, aligned or matched together?
5. Which single relation between objects of different roles would plausibly complete the level
   (for instance the controllable object inside a target, a piece aligned with its matching slot,
   tiles recoloured to match a shown pattern)?
Status bars and step counters along the edges are not goals.'''

ROLES = ['controllable', 'target_or_container', 'example_or_pattern', 'piece_to_place', 'obstacle_or_wall',
         'status_or_counter', 'background_or_other']


def first_frame(game):
    rec = glob.glob(str(RUN / f'{game}*' / 'recordings' / '*' / '*.jsonl'))[0]
    return json.loads(open(rec).readline())['data']['frame'][-1]


def objects(grid):
    comps = [c for c in components(grid) if len(c['cells']) <= MAX_OBJECT_CELLS]
    comps.sort(key=lambda c: (c['origin'][1], c['origin'][0]))
    out = []
    for i, c in enumerate(comps, 1):
        xs = [x for x, _ in c['cells']]
        ys = [y for _, y in c['cells']]
        out.append(dict(id=f'o{i}', color=COLORS[c['color']], bbox=[min(xs), min(ys), max(xs), max(ys)],
                        cells=len(c['cells'])))
    return out


def image(grid, objs, marked):
    im = render_current(grid)
    if marked:
        draw, font = ImageDraw.Draw(im), ui_font(9)
        for o in objs:
            x, y = ORIGIN[0] + o['bbox'][0]*SCALE, ORIGIN[1] + o['bbox'][1]*SCALE
            text = o['id'][1:]
            box = draw.textbbox((x, y), text, font=font)
            draw.rectangle(box, fill='black')
            draw.text((x, y), text, fill='yellow', font=font)
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode(), im


def truth(game, objs):
    """Accepted atoms per game (from the game source, V2)."""
    def within(o, box):
        return box[0] <= o['bbox'][0] and box[1] <= o['bbox'][1] and o['bbox'][2] <= box[2] and o['bbox'][3] <= box[3]
    if game == 'ls20':
        player = {o['id'] for o in objs if within(o, (34, 45, 38, 49))}
        target = {o['id'] for o in objs if within(o, (33, 9, 39, 15))}
        return lambda a: a['relation'] in ('inside', 'overlaps') and (
            (a['a'] in player and a.get('b') in target) or (a['a'] in target and a.get('b') in player))
    if game == 'vc33':
        pair = {o['id'] for o in objs if o['color'] == 'yellow' and (within(o, (50, 44, 51, 49)) or within(o, (38, 28, 39, 31)))}
        return lambda a: a['relation'] == 'same_column' and {a['a'], a.get('b')} == pair
    # ft09: in the bottom-right group, tiles the clue marks white must become red (the clue's centre colour).
    tiles = sorted((o for o in objs if o['cells'] == 36 and within(o, (36, 36, 59, 59))),
                   key=lambda o: (o['bbox'][1], o['bbox'][0]))
    cols = sorted({o['bbox'][0] for o in tiles})
    rows = sorted({o['bbox'][1] for o in tiles})
    white = {(0, 0), (0, 1), (2, 1), (0, 2)}   # clue pattern WGG / WRW / WGG at (46,46)
    must_red = {o['id'] for o in tiles if (cols.index(o['bbox'][0]), rows.index(o['bbox'][1])) in white}
    centre = {o['id'] for o in objs if o['color'] == 'red' and within(o, (46, 46, 53, 53))}
    return lambda a: (a['relation'] == 'color_is' and a['a'] in must_red and a.get('color') == 'red') or (
        a['relation'] == 'same_color' and ((a['a'] in must_red and a.get('b') in centre) or
                                           (a['a'] in centre and a.get('b') in must_red)))


def schema(ids, roles):
    atom = {'type': 'object', 'properties': {
        'relation': {'type': 'string', 'enum': list(RELATIONS)}, 'a': {'type': 'string', 'enum': ids},
        'b': {'anyOf': [{'type': 'string', 'enum': ids}, {'type': 'null'}]},
        'color': {'anyOf': [{'type': 'string', 'enum': COLORS}, {'type': 'null'}]}},
        'required': ['relation', 'a', 'b', 'color']}
    props = {'hypotheses': {'type': 'array', 'minItems': 1, 'maxItems': 3, 'items': {'type': 'object', 'properties': {
        'atoms': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': atom},
        'rationale': {'type': 'string', 'maxLength': 160}}, 'required': ['atoms', 'rationale']}}}
    if roles:
        props = {'roles': {'type': 'array', 'maxItems': 12, 'items': {'type': 'object', 'properties': {
            'id': {'type': 'string', 'enum': ids}, 'role': {'type': 'string', 'enum': ROLES}},
            'required': ['id', 'role']}}, **props}
    return {'type': 'object', 'properties': props, 'required': list(props)}


def measured_effects(game, objs):
    """Facts the program learns from a few clicks (V1 RuleLearner on the recorded transitions)."""
    at = {(o['color'], o['bbox'][0], o['bbox'][1]): o['id'] for o in objs}
    if game == 'vc33':
        button, marker, block = at[('blue', 60, 32)], at[('yellow', 50, 44)], at[('very dark gray', 46, 44)]
        return [f'clicking {button} (blue) moved {marker} (yellow) and {block} (very dark gray) 4 cells left (2 of 2 clicks)']
    if game == 'ft09':
        a, b = at[('blue', 36, 36)], at[('blue', 36, 44)]
        return [f'clicking {a} (blue tile) turned it red', f'clicking {b} (blue tile) turned it red']
    return []


def ask(game, condition, objs, controllable, sample, effects=()):
    marked = condition != 'C0'
    img, _ = image(first_frame(game), objs, marked)
    rows = [dict(o, controllable=o['id'] in controllable) if controllable else o for o in objs]
    facts = dict(measured_effects=list(effects)) if effects else {}
    text = json.dumps(dict(objects=rows, relations=RELATIONS, controls=CONTROLS[game], **facts), separators=(',', ':'))
    instruction = HYPOTHESIZE
    if marked:
        instruction += '\nObject IDs are drawn on the image at each object\'s top-left corner (the number after "o").'
    if condition in ('C2', 'C3'):
        instruction += '\n' + CHECKLIST
    if condition == 'C3':
        instruction += '\nFirst assign a role to the important objects, then write hypotheses that use those roles.'
    payload = {'model': 'qwen3-vl-4b-instruct', 'temperature': 0.7, 'seed': sample,
               'max_tokens': 1200 if condition == 'C3' else 700,
               'response_format': {'type': 'json_schema', 'json_schema': {
                   'name': 'goals', 'schema': schema([o['id'] for o in objs], condition == 'C3')}},
               'messages': [{'role': 'system', 'content': 'You play an unknown turn-based puzzle game. Nobody tells '
                             'you the rules or the goal. ' + instruction},
                            {'role': 'user', 'content': [
                                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + img}},
                                {'type': 'text', 'text': text}]}]}
    started = time.monotonic()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(API, json.dumps(payload).encode(),
                                          {'Content-Type': 'application/json'}), timeout=300).read())
    try:
        answer = json.loads(r['choices'][0]['message']['content'])
    except json.JSONDecodeError:
        answer = {'hypotheses': [], 'invalid': r['choices'][0].get('finish_reason')}  # truncated output
    return answer, r.get('usage', {}), time.monotonic() - started


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]  # e.g. vc33+effects ft09+effects
    settings = [('ls20', False), ('ls20', True), ('vc33', False), ('ft09', False), ('vc33', 'effects'), ('ft09', 'effects')]
    settings = [s for s in settings if not only or (s[0] + ('+controllable' if s[1] is True else '+effects' if s[1] else '')) in only]
    rows = []
    for game, with_control in settings:
        objs = objects(first_frame(game))
        ok = truth(game, objs)
        controllable = {o['id'] for o in objs if game == 'ls20' and 34 <= o['bbox'][0] and o['bbox'][2] <= 38
                        and 45 <= o['bbox'][1] and o['bbox'][3] <= 49} if with_control is True else set()
        effects = measured_effects(game, objs) if with_control == 'effects' else []
        image(first_frame(game), objs, True)[1].save(OUT / f'{game}-marked.png')
        label = game + ('+controllable' if with_control is True else '+effects' if with_control else '')
        for condition in ('C0', 'C1', 'C2', 'C3'):
            for sample in range(SAMPLES):
                answer, usage, seconds = ask(game, condition, objs, controllable, sample, effects)
                hits = [any(ok(a) for a in h['atoms']) for h in answer['hypotheses']]
                trivial_rel = sum(a['relation'] in ('same_color', 'same_shape', 'gone')
                                  for h in answer['hypotheses'] for a in h['atoms'])
                rows.append(dict(setting=label, condition=condition, sample=sample, hits=hits, answer=answer,
                                 invalid='invalid' in answer,
                                 prompt_tokens=usage.get('prompt_tokens'), seconds=round(seconds, 1),
                                 generic_relations=trivial_rel,
                                 atoms=sum(len(h['atoms']) for h in answer['hypotheses'])))
                print(label, condition, sample, hits, round(seconds, 1))
    (OUT / ('trials.json' if not only else f"trials-{'-'.join(only)}.json")).write_text(json.dumps(rows, indent=1))
    print('\n|setting|C0|C1|C2|C3|')
    print('|---|---|---|---|---|')
    for game, with_control in settings:
        label = game + ('+controllable' if with_control is True else '+effects' if with_control else '')
        cells = []
        for condition in ('C0', 'C1', 'C2', 'C3'):
            rs = [r for r in rows if r['setting'] == label and r['condition'] == condition]
            cells.append(f"{sum(any(r['hits']) for r in rs)}/{len(rs)} (1st {sum(bool(r['hits']) and r['hits'][0] for r in rs)})")
        print(f'|{label}|' + '|'.join(cells) + '|')
    print('\n|condition|share of same_color/same_shape/gone atoms|invalid outputs|prompt tokens|seconds|')
    print('|---|---:|---:|---:|---:|')
    for condition in ('C0', 'C1', 'C2', 'C3'):
        rs = [r for r in rows if r['condition'] == condition]
        print(f"|{condition}|{sum(r['generic_relations'] for r in rs)/max(1, sum(r['atoms'] for r in rs)):.2f}|"
              f"{sum(r['invalid'] for r in rs)}/{len(rs)}|"
              f"{sum(r['prompt_tokens'] or 0 for r in rs)/len(rs):.0f}|{sum(r['seconds'] for r in rs)/len(rs):.1f}|")


if __name__ == '__main__':
    main()
