"""V8: is Qwen's sensitivity to the input format a general trend across games?

Same content (numbered image, VCGT question checklist, required roles, object list), five ways
of writing it (docs/goal-pipeline-ja.md §6):

  F-A  V7 best: colour as a string, 0-63 grid boxes, JSON-schema output (response_format)
  F-B  colours as a list plus a 'kind' field (the runtime's earlier rows)       H1: labels
  F-C  boxes normalised to 0-1000 as bbox_2d (Qwen3-VL grounding convention)   H2: coordinates
  F-D  F-A delivered as a tool call instead of response_format                 H3: output channel
  F-E  F-A with a free-text 'analysis' field before the structured answer      H3: reason first

Six settings over six public games; ground truth from the game source (ls20, vc33, ft09) or the
first frame with the human VCGT description (r11l, tu93, ka59), used only for scoring.

usage (dev container): python scripts/verify_hypothesis_formats.py [samples]
"""
import copy
import glob
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_hypothesis_prompting as v  # noqa: E402

OUT = ROOT / 'outputs/v8-format'
SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NEW_RUN = ROOT / 'outputs/evaluations/20260930T134918595267Z'
CONTROLS = {**v.CONTROLS, 'r11l': 'CLICK at a cell', 'tu93': 'UP, DOWN, LEFT, RIGHT',
            'ka59': 'UP, DOWN, LEFT, RIGHT, CLICK at a cell'}
FORMATS = tuple(a for a in sys.argv[2:]) or ('F-A', 'F-B', 'F-C', 'F-D', 'F-E')   # F-CE combines C and E


def frame(game):
    run = NEW_RUN if game in ('r11l', 'tu93', 'ka59') else v.RUN
    rec = glob.glob(str(run / f'{game}*' / 'recordings' / '*' / '*.jsonl'))[0]
    return json.loads(open(rec).readline())['data']['frame'][-1]


def within(o, box):
    return box[0] <= o['bbox'][0] and box[1] <= o['bbox'][1] and o['bbox'][2] <= box[2] and o['bbox'][3] <= box[3]


def pair_truth(objs, first, second, relations=('inside', 'overlaps')):
    a_ids, b_ids = {o['id'] for o in objs if first(o)}, {o['id'] for o in objs if second(o)}
    assert a_ids and b_ids, (a_ids, b_ids)
    return lambda a: a['relation'] in relations and (
        (a['a'] in a_ids and a.get('b') in b_ids) or (a['a'] in b_ids and a.get('b') in a_ids))


def setting(game, objs):
    if game in ('ls20', 'vc33', 'ft09'):
        truth = v.truth(game, objs)
    elif game == 'r11l':   # purple body onto the dotted purple ring
        truth = pair_truth(objs, lambda o: o['color'] == 'purple' and o['cells'] >= 16,
                           lambda o: o['color'] == 'purple' and within(o, (36, 18, 42, 24)))
    elif game == 'tu93':   # blue player onto the green goal
        truth = pair_truth(objs, lambda o: o['color'] == 'blue' and within(o, (15, 15, 17, 17)),
                           lambda o: o['color'] == 'green' and within(o, (44, 43, 48, 47)))
    else:                  # ka59: a green block into a dark frame
        truth = pair_truth(objs, lambda o: o['color'] == 'green',
                           lambda o: o['color'] == 'very dark gray' and o['cells'] == 16)
    boxes = {'ls20': (34, 45, 38, 49), 'tu93': (15, 15, 17, 17)}
    controllable = {o['id'] for o in objs if game in boxes and within(o, boxes[game])}
    return truth, controllable


def rows(objs, controllable, fmt):
    out = []
    for o in objs:
        if fmt == 'F-B':
            r = dict(id=o['id'], colors=[o['color']], bbox=o['bbox'], cells=o['cells'], kind='region')
        elif fmt in ('F-C', 'F-CE'):
            r = dict(id=o['id'], color=o['color'], bbox_2d=[round(c * 1000 / 63) for c in o['bbox']], cells=o['cells'])
        else:
            r = dict(id=o['id'], color=o['color'], bbox=o['bbox'], cells=o['cells'])
        r['controllable'] = o['id'] in controllable
        out.append(r)
    return out


def request(game, objs, controllable, fmt, seed):
    img, _ = v.image(frame(game), objs, True)
    system = ('You play an unknown turn-based puzzle game. Nobody tells you the rules or the goal. ' + v.HYPOTHESIZE +
              '\nObject IDs are drawn on the image at each object\'s top-left corner (the number after "o").\n' +
              v.CHECKLIST + '\nFirst assign a role to the important objects, then write hypotheses that use those roles.')
    if fmt in ('F-C', 'F-CE'):
        system += '\nbbox_2d coordinates are normalised to 0-1000 over the board.'
    schema = v.schema([o['id'] for o in objs], True)
    if fmt in ('F-E', 'F-CE'):
        schema = {**schema, 'properties': {'analysis': {'type': 'string', 'maxLength': 600}, **schema['properties']},
                  'required': ['analysis'] + schema['required']}
        system += '\nWrite your reasoning in analysis first.'
    text = json.dumps(dict(objects=rows(objs, controllable, fmt), relations=v.RELATIONS, controls=CONTROLS[game]),
                      separators=(',', ':'))
    payload = {'model': 'qwen3-vl-4b-instruct', 'temperature': 0.7, 'seed': seed, 'max_tokens': 1400,
               'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': [
                   {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + img}},
                   {'type': 'text', 'text': text}]}]}
    if fmt == 'F-D':
        payload.update(tools=[{'type': 'function', 'function': {'name': 'submit_goal_hypotheses',
                       'description': 'Submit the goal hypotheses.', 'parameters': schema}}], tool_choice='required')
    else:
        payload['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'goals', 'schema': schema}}
    return payload


def answer_of(response, fmt):
    message = response['choices'][0]['message']
    try:
        if fmt == 'F-D':
            return json.loads(message['tool_calls'][0]['function']['arguments'])
        return json.loads(message['content'])
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    games = ('ls20', 'tu93', 'vc33', 'ft09', 'r11l', 'ka59')
    results = []
    for game in games:
        objs = v.objects(frame(game))
        truth, controllable = setting(game, objs)
        for fmt in FORMATS:
            for seed in range(SAMPLES):
                payload = request(game, objs, controllable, fmt, seed)
                r = json.loads(urllib.request.urlopen(urllib.request.Request(
                    v.API, json.dumps(payload).encode(), {'Content-Type': 'application/json'}), timeout=300).read())
                ans = answer_of(r, fmt)
                hyps = (ans or {}).get('hypotheses', [])
                hits = [any(truth(a) for a in h['atoms']) for h in hyps]
                results.append(dict(game=game, format=fmt, seed=seed, valid=ans is not None, hits=hits,
                                    first=bool(hits) and hits[0], answer=ans))
            rs = [x for x in results if x['game'] == game and x['format'] == fmt]
            print(game, fmt, f"{sum(any(x['hits']) for x in rs)}/{len(rs)}", flush=True)
    (OUT / f"trials-{'-'.join(FORMATS)}.json").write_text(json.dumps(results, indent=1))
    print('\n|game|' + '|'.join(FORMATS) + '|')
    print('|---|' + '---|' * len(FORMATS))
    for game in games + ('total',):
        cells = []
        for fmt in FORMATS:
            rs = [x for x in results if x['format'] == fmt and (game == 'total' or x['game'] == game)]
            cells.append(f"{sum(any(x['hits']) for x in rs)}/{len(rs)}")
        print(f'|{game}|' + '|'.join(cells) + '|')
    print('\n|format|invalid outputs|')
    print('|---|---:|')
    for fmt in FORMATS:
        rs = [x for x in results if x['format'] == fmt]
        print(f"|{fmt}|{sum(not x['valid'] for x in rs)}/{len(rs)}|")


if __name__ == '__main__':
    main()
