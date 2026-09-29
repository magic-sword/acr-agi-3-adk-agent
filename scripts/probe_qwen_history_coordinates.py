"""Controlled history-format x location-representation probes; no environment actions.

Images are deterministic scientific renderings of recorded grid data. Marks are
host annotations and preserve the clicked pixel. No generated image content is used.
"""
import argparse
import base64
from collections import defaultdict
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jsonschema
import numpy as np
from PIL import Image, ImageDraw

from agent.rendering import ORIGIN, SCALE, render_current, png_base64
from probe_incremental_planning_inputs import execute, minimal_payload, read_cases
from probe_repeated_click_inputs import digest

REPRESENTATIONS = ('bare', 'grid', 'normalized', 'marker', 'normalized_marker', 'language')
FORMATS = ('structured', 'official_style')
COLORS = ['white', 'light gray', 'gray', 'dark gray', 'very dark gray', 'black',
          'magenta', 'pink', 'red', 'blue', 'light blue', 'yellow', 'orange', 'maroon', 'green', 'purple']
LOCATION_GOAL = ('Identify the board arrangement containing the previous click, the color of the exact '
                 'game pixel clicked, and whether the recorded action visibly changed the board. '
                 'Report unknown when the evidence is insufficient. Ignore any host marker when naming the game color.')
PLAN_GOAL = 'Explore the unknown rules of this visual game by choosing the next useful click.'


def tools_for(task):
    if task == 'locate':
        properties = {
            'region': {'type': 'string', 'enum': ['upper_left', 'upper_right', 'lower_left', 'lower_right', 'outside', 'unknown']},
            'color': {'type': 'string', 'enum': COLORS + ['unknown']},
            'observed_change': {'type': 'string', 'enum': ['none', 'changed', 'unknown']},
        }
    else:
        properties = {
            'action': {'type': 'string', 'enum': ['CLICK', 'need_information']},
            'coordinate': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 1000},
                           'minItems': 0, 'maxItems': 2,
                           'description': 'For CLICK use [x,y] normalized to the entire supplied image, 0 to 1000. Otherwise use [].'},
            'target': {'type': 'string'},
            'reason': {'type': 'string'},
        }
    return [{'type': 'function', 'function': {
        'name': 'submit_result', 'description': 'Submit the result of the current task.',
        'parameters': {'type': 'object', 'additionalProperties': False,
                       'properties': properties, 'required': list(properties)},
    }}]


def cases_with_geometry():
    cases = read_cases()
    for name, case in cases.items():
        log = next((ROOT / 'outputs/memory-comparison-20260929' / name).glob('*/cognition/*.observations.jsonl'))
        obs = next(r for r in map(json.loads, log.read_text().splitlines()) if r['step'] == 2)
        grid = np.asarray(obs['grid'])
        base = render_current(grid)
        recorded = Image.open(io.BytesIO(base64.b64decode(case['groups']['I']['image_url']['url'].split(',')[1])))
        assert np.array_equal(np.asarray(base), np.asarray(recorded)), 'Rendering must equal recorded image'
        x, y = (case['groups']['F']['action'][k] for k in ('x', 'y'))
        ox, oy = ORIGIN
        center = [ox + (x + .5) * SCALE, oy + (y + .5) * SCALE]
        normalized = [round(1000 * p / size) for p, size in zip(center, base.size)]
        # Convert back by the same coordinate convention; pixel centers avoid edge rounding.
        back = [math.floor((n / 1000 * size - origin) / SCALE)
                for n, size, origin in zip(normalized, base.size, ORIGIN)]
        assert back == [x, y]
        marked = base.copy()
        draw = ImageDraw.Draw(marked)
        px, py = ox + x * SCALE, oy + y * SCALE
        draw.rectangle((px-2, py-2, px+SCALE+1, py+SCALE+1), outline='black', width=1)
        draw.rectangle((px-1, py-1, px+SCALE, py+SCALE), outline='#ffff00', width=1)
        assert np.array_equal(np.asarray(marked)[py:py+SCALE, px:px+SCALE],
                              np.asarray(base)[py:py+SCALE, px:px+SCALE])
        is_left = x < 32
        description = (
            'the right-hand portion of the bottom horizontal strip of the small central patch '
            'in the upper-left 3-by-3 arrangement' if is_left else
            'the bottom end of the leftmost vertical strip of the small central patch '
            'in the upper-right 3-by-3 arrangement')
        # Natural descriptions refer to homogeneous regions, never mention the answer color.
        described_bbox = [14, 14, 17, 15] if is_left else [46, 14, 47, 15]
        l, t, r, b = described_bbox
        assert l <= x <= r and t <= y <= b
        assert np.all(grid[t:b+1, l:r+1] == grid[y, x])
        case.update(grid=grid.tolist(), base=base, marked=marked, language=description,
                    gold=dict(game_xy=[x,y], image_xy=center, normalized_xy=normalized,
                              image_size=list(base.size), board_size=[64,64], origin=list(ORIGIN), scale=SCALE,
                              region='upper_left' if is_left else 'upper_right',
                              color=COLORS[int(grid[y,x])], color_id=int(grid[y,x]), observed_change='none',
                              target_patch=[12,10,17,15] if is_left else [46,10,51,15],
                              language_bbox=described_bbox,
                              observation_source=str(log.relative_to(ROOT))))
    return cases


def location(case, representation):
    gold = case['gold']
    x, y = gold['game_xy']
    nx, ny = gold['normalized_xy']
    marker_note = ('the center of the yellow outline in the current image. '
                   'The outline marks the PREVIOUS click, is not a game object, and is not a request to click again')
    if representation == 'bare':
        return {'x': x, 'y': y}, f'({x},{y})'
    if representation == 'grid':
        convention = ('zero-based game-cell coordinates in the 64x64 board, x=column and y=row; '
                      'not normalized. Use the image axis labels. The board starts at image pixel (32,44), '
                      'with each game cell displayed as 6x6 image pixels')
        return {'coordinate': [x,y], 'coordinate_system': convention}, f'({x},{y}) in {convention}'
    if representation in ('normalized', 'normalized_marker'):
        convention = ('0-1000 normalized coordinates over the ENTIRE supplied image including its margins; '
                      'x from left to right, y from top to bottom')
        data = {'coordinate': [nx,ny], 'coordinate_system': convention}
        text = f'({nx},{ny}) in {convention}'
        if representation == 'normalized_marker':
            data['visual_reference'] = marker_note
            text += '; this point is ' + marker_note
        return data, text
    if representation == 'marker':
        return {'visual_reference': marker_note}, marker_note
    if representation == 'language':
        return {'description': case['language']}, case['language']
    raise ValueError(representation)


def make_payload(case, fmt, representation, task):
    payload = minimal_payload(case['original'])
    payload.update(tools=tools_for(task), tool_choice='required', parallel_tool_calls=False)
    data, text = location(case, representation)
    goal = LOCATION_GOAL if task == 'locate' else PLAN_GOAL
    if fmt == 'structured':
        content = json.dumps({'task': goal, 'last_actual_result': {
            'action': {'action': 'CLICK', 'location': data}, 'acknowledged': True,
            'frame_changed': False, 'changed_cell_count': 0}}, separators=(',', ':'))
    else:
        content = (f'The user query: {goal}\n'
                   'Task progress (You have done the following operation on the current device): '
                   f'Step 1: I clicked at {text}.\n'
                   'Observed result of Step 1: The click was acknowledged. '
                   'The before and after boards were identical: zero cells changed in the observed interval.\n')
    view = case['marked'] if 'marker' in representation else case['base']
    payload['messages'] = [
        {'role': 'system', 'content': (
            'Answer the current task using the screenshot and recorded observations. Submit submit_result.'
            if task == 'locate' else
            'Choose one next step using the screenshot and recorded observations. The available game control is CLICK. '
            'If the supplied information is insufficient, return need_information. '
            'For a next CLICK return its location normalized to the entire supplied image on a 0-1000 scale, '
            'with x increasing rightwards and y downwards. Otherwise return an empty coordinate array. '
            'State the target and a concise reason. Submit submit_result.')},
        {'role': 'user', 'content': [{'type': 'text', 'text': content},
                                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + png_base64(view)}}]},
    ]
    return payload


def build_jobs(cases, repeats, representations=REPRESENTATIONS):
    jobs = []
    for repetition in range(repeats):
        block = []
        for name, case in cases.items():
            for fmt in FORMATS:
                for rep in representations:
                    for task in ('locate', 'plan'):
                        block.append(dict(case=name, variant=f'{fmt}-{rep}-{task}', format=fmt,
                                          representation=rep, task=task, repetition=repetition,
                                          payload=make_payload(case, fmt, rep, task)))
        random.Random(29092028 + repetition).shuffle(block)
        jobs += block
    return jobs


def prepare(out, cases, repeats):
    out.mkdir(parents=True, exist_ok=False)
    (out/'assets').mkdir()
    for name, case in cases.items():
        case['base'].save(out/'assets'/f'{name}-plain.png')
        case['marked'].save(out/'assets'/f'{name}-marked.png')
    (out/'gold.json').write_text(json.dumps({n: c['gold'] for n,c in cases.items()}, indent=2)+'\n')
    jobs = build_jobs(cases, repeats)
    plan = dict(requests=len(jobs), repeats=repeats,
                design='2 formats x 6 representations x 2 tasks x 2 recorded histories x repeats. '
                       'Independent requests with shuffled order, same model/temperature/output schema per task.',
                format_scope='official_style borrows the mobile cookbook goal + completed Step history wrapper. '
                             'Observed result is our extension. Output contract is held fixed, not an exact cookbook stack.',
                geometry='Normalize game-cell center after viewport mapping to the entire supplied 428x452 image. '
                         'Do not normalize a game coordinate as if it were already an image coordinate.',
                metrics='Ground-truth region and clicked pixel color from recorded grid; no-change fact recognition. '
                        'Planning: valid on-board click, exact previous cell / same central patch reuse, need_information. '
                        'A different click is not a known-correct action or task success.',
                limitations='Two selected histories on one unchanged board; no independent-game accuracy or attention claim. '
                            'Natural-language descriptions authored and checked by the experimenter, not automatically generated. '
                            'Marker annotation may change visual salience. Bare coordinates deliberately reproduce ambiguity. '
                            'No old model plans, long-term knowledge or candidate catalog in any condition.',
                sources=['https://arxiv.org/html/2511.21631v1#S3.SS2.SSS4',
                         'https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/mobile_agent.ipynb',
                         'https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/computer_use.ipynb'],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                job_hashes=[digest(j['payload']) for j in jobs])
    (out/'design.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({n:c['gold'] for n,c in cases.items()},indent=2))
    print('Planned requests',len(jobs))


def analyze(out, cases):
    design = json.loads((out/'design.json').read_text())
    jobs = build_jobs(cases, design['repeats'], design.get('representations', REPRESENTATIONS))
    rows = json.loads((out/'run/results.json').read_text())
    evidence = []
    for i, row in enumerate(rows):
        request = json.loads((out/'run'/row['request_file']).read_text())
        assert digest(request) == digest(jobs[i]['payload']) == row['request_sha256'] == design['job_hashes'][i]
        assert all(row[k] == jobs[i][k] for k in ('case','variant','repetition','task','format','representation'))
        answer = row.get('answer', {})
        schema = request['tools'][0]['function']['parameters']
        errors = [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(answer)]
        if row.get('tool') != 'submit_result':
            errors.append('missing or wrong result tool')
        item = {k:v for k,v in row.items() if k not in ('response','answer')}
        item.update(answer=answer, schema_errors=errors)
        gold = cases[row['case']]['gold']
        if row['task'] == 'locate':
            item.update(region_correct=answer.get('region') == gold['region'],
                        color_correct=answer.get('color') == gold['color'],
                        change_correct=answer.get('observed_change') == gold['observed_change'])
            item['region_and_color_correct'] = item['region_correct'] and item['color_correct']
        else:
            coords = answer.get('coordinate', [])
            item.update(click=answer.get('action')=='CLICK', need_information=answer.get('action')=='need_information',
                        valid_click=False, same_cell=False, same_patch=False, game_xy=None)
            if item['click'] and len(coords)==2 and not errors:
                converted = [math.floor((n/1000*size-origin)/gold['scale'])
                             for n,size,origin in zip(coords,gold['image_size'],gold['origin'])]
                item['game_xy'] = converted
                item['valid_click'] = all(0 <= v < 64 for v in converted)
                item['same_cell'] = converted == gold['game_xy']
                l,t,r,b = gold['target_patch']
                item['same_patch'] = l <= converted[0] <= r and t <= converted[1] <= b
        evidence.append(item)
    aggregates = []
    for task in ('locate','plan'):
        for fmt in FORMATS:
            for rep in design.get('representations', REPRESENTATIONS):
                subset = [e for e in evidence if (e['task'],e['format'],e['representation'])==(task,fmt,rep)]
                keys = (['region_correct','color_correct','region_and_color_correct','change_correct'] if task=='locate'
                        else ['click','need_information','valid_click','same_cell','same_patch'])
                aggregates.append(dict(task=task,format=fmt,representation=rep,n=len(subset),
                                       **{key:sum(e[key] for e in subset) for key in keys}))
    (out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(aggregates,indent=2)+'\n')
    verification = dict(planned=len(jobs),completed=len(rows),requests_reconstructed=True,
                        schema_errors=sum(bool(e['schema_errors']) for e in evidence),
                        request_errors=sum('error' in r for r in rows),
                        truncated=sum(r.get('response',{}).get('choices',[{}])[0].get('finish_reason')=='length' for r in rows),
                        mapping_roundtrip='Verified both actual cells', marker_occlusion='Original clicked cells unchanged')
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    lines = ['# Coordinate history comparison', '', '## Grounding', '',
             '| Format | Location | n | Region | Pixel color | Both | No change |',
             '|---|---|---:|---:|---:|---:|---:|']
    for a in aggregates:
        if a['task']=='locate':
            lines.append(f"| {a['format']} | {a['representation']} | {a['n']} | {a['region_correct']} | "
                         f"{a['color_correct']} | {a['region_and_color_correct']} | {a['change_correct']} |")
    lines += ['', '## Planning', '', '| Format | Location | n | CLICK | Need info | In board | Same cell | Same patch |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for a in aggregates:
        if a['task']=='plan':
            lines.append(f"| {a['format']} | {a['representation']} | {a['n']} | {a['click']} | "
                         f"{a['need_information']} | {a['valid_click']} | {a['same_cell']} | {a['same_patch']} |")
    lines += ['', '## All responses', '']
    for e in evidence:
        lines += [f"- {e['case']} {e['variant']} r{e['repetition']}: " + json.dumps(e['answer'],ensure_ascii=False)]
    (out/'responses.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verification,indent=2))
    print(json.dumps(aggregates,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','run','analyze'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--repeats',default=3,type=int)
    args = parser.parse_args()
    cases = cases_with_geometry()
    out = args.output.resolve()
    if args.phase == 'prepare':
        prepare(out,cases,args.repeats)
    elif args.phase == 'run':
        design = json.loads((out/'design.json').read_text())
        jobs = build_jobs(cases,design['repeats'])
        assert [digest(j['payload']) for j in jobs] == design['job_hashes']
        execute(jobs,out/'run',dict(phase='history_coordinates',design_file='../design.json',
                                  probe_script_sha256=design['script_sha256']))
    else:
        analyze(out,cases)


if __name__ == '__main__':
    main()
