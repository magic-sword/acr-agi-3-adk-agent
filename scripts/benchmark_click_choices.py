"""Frozen click-part selection experiment; never sends game actions.

Compare free mask union, coarse location + text choices, and identical choices
with masked thumbnails. Gold is used only for scoring, never for shortlisting.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace
import urllib.request
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.rendering import frame_image, render_current, png_base64
from agent.cognition import object_memory
from agent.cognition.perception import COLORS
from agent.cognition.hybrid_perception import HybridPerception
from agent.cognition.region_masks import decode, encode
from agent.cognition.simple_workflow import click_point

SOURCE = ROOT / 'outputs/evaluations/20260929T132323611961Z'
ARMS = ('free_union', 'choice_text', 'choice_visual')
SEED = 929703
LIMIT = 8
LOCALIZE = '''Locate ONE visibly identifiable target for a click. Choose its location
in the clean board, using a 3 by 3 spatial grid: 1 upper-left, 2 upper-center,
3 upper-right, 4 middle-left, 5 center, 6 middle-right, 7 lower-left,
8 lower-center, 9 lower-right. Choose the cell containing the target's center.
Use 0 for a long/broad target spanning most of the board. Use X if the description
is ambiguous, absent, or depends on an unknown game role. Do not infer game rules.
Return only the offered digit or X. The grid is a location convention, not game UI.'''
SELECT = '''Select the measured mask for ONE intended clickable part described by
the query, using visible appearance and location only. Regions are hypotheses:
some are fragments, background, or entire containers. Select the named part, not
its surrounding panel. Choose X if no offered mask represents the target, several
different targets fit, or the description requires an unknown game role.
An equivalent mask of the same target is acceptable. Do not infer game rules.
Mask thumbnails, labels, and coordinates are host annotations, not game objects.'''


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def rect(box):
    x, y, r, b = box
    return {(xx, yy) for yy in range(y, b+1) for xx in range(x, r+1)}


def pool(record, grid):
    """All measured masks and automatic object groups; exact support dedup only."""
    state = object_memory.update(object_memory.empty(), record)
    result, seen = [], set()
    for obj in state['objects']:
        points = decode(obj['mask_runs'])
        key = frozenset(points)
        if not points or key in seen:
            continue
        seen.add(key)
        xs, ys = zip(*points)
        result.append(dict(id=obj['object_id'], bbox=[min(xs), min(ys), max(xs), max(ys)],
            mask_runs=encode(points), colors=sorted({grid[y][x] for x, y in points}),
            kind=obj['kind'], sources=obj['sources']))
    return result


def build_cases():
    cases = []
    specs = {
        'ls20': [
            ('top-blue', 'The blue angular pattern inside the black display near the top center. Select only the blue part.', (33,9,39,17), (9,)),
            ('bicolor', 'The small orange-above-blue rectangle near the lower center, including both colored halves.', (32,43,40,51), (9,12)),
            ('cross', 'The small white and light gray plus-shaped cross on the gray platform.', (18,29,25,36), (0,1)),
            ('yellow', 'The long yellow horizontal bar along the bottom.', (0,60,63,63), (11,)),
            ('ambiguous', 'The blue angular pattern inside a black square.', None, None),
            ('absent', 'The green triangle.', None, None)],
        'vc33': [
            ('upper-blue', 'The blue square at the right edge just above the long horizontal bar.', (60,24,63,27), (9,)),
            ('lower-blue', 'The blue square at the right edge just below the long horizontal bar.', (60,32,63,35), (9,)),
            ('yellow', 'The thin yellow vertical part beside the dark gray shape in the lower half.', (50,44,51,49), (11,)),
            ('bar', 'The entire long horizontal black bar with a yellow segment inside it.', (20,28,63,31), None),
            ('ambiguous', 'The blue square at the right edge.', None, None),
            ('absent', 'The red circular button.', None, None)],
        'ft09': [
            ('top-left-blue', 'The blue square at the far left of the topmost row.', (4,2,9,7), (9,)),
            ('upper-pattern', 'The white and gray patterned square at the center of the upper-left group of nine squares.', (12,10,17,15), None),
            ('lower-pattern', 'The small white and gray patterned square with a red center, at the center of the lower-right group. Select this small square, not the surrounding panel.', (44,44,49,49), None),
            ('lower-blue', 'The blue square at the bottom-right corner of the lower-right group of nine squares.', (52,52,57,57), (9,)),
            ('ambiguous', 'The red square in the topmost row.', None, None),
            ('absent', 'The green square.', None, None)]}
    for game, questions in specs.items():
        folder = next(SOURCE.glob(game+'-*'))/'cognition'
        op = next(folder.glob('*.observations.jsonl'))
        ap = next(folder.glob('*.artifacts.jsonl'))
        obs = read_rows(op)[0]
        record = next(x['measurement'] for x in read_rows(ap) if x['event']=='objects_measured')
        candidates = pool(record, obs['grid'])
        for name, query, box, colors in questions:
            gold = rect(box) if box is not None else set()
            if colors is not None:
                gold = {(x,y) for x,y in gold if obs['grid'][y][x] in colors}
            if box is not None:
                assert gold, (game, name)
            cases.append(dict(name=game+'-'+name, origin='real', game=game, grid=obs['grid'],
                query=query, gold_runs=encode(gold), reason='unique' if box else name,
                candidates=candidates, source={str(p.relative_to(ROOT)):sha(p) for p in (op,ap)},
                annotation='Assistant-authored visible pixel support, frozen before responses; no game rules.'))
    # Controls distinguish a surrounding object from the clicked part and exercise
    # nonconvex geometry. Candidates still come from the runtime, not gold masks.
    for number in range(8):
        grid = [[5]*64 for _ in range(64)]
        def paint(points, color):
            for x,y in points:
                grid[y][x] = color
        paint(rect((4,4,29,29)), 3)
        paint(rect((35,35,60,60)), 3)
        paint(rect((11,11,20,20)), 9)
        paint(rect((44,44,53,53)), 8)
        gold = set()
        if number < 2:
            gold = rect((14,14,17,17)) if number == 0 else rect((47,47,50,50))
            paint(gold, 0)
            query = 'The small white square inside the '+('upper-left blue' if number==0 else 'lower-right red')+' square. Select the white part only.'
        elif number == 2:
            gold = rect((37,7,55,11)) | rect((37,7,41,25)) | rect((37,21,55,25))
            paint(gold, 11); query = 'The yellow C-shaped object in the upper-right.'
        elif number == 3:
            gold = rect((7,38,26,57)) - rect((11,42,22,53))
            paint(gold, 11); query = 'The yellow hollow square ring in the lower-left, not its empty interior.'
        elif number == 4:
            gold = rect((36,10,58,10)); paint(gold, 11)
            query = 'The thin yellow horizontal line in the upper-right.'
        elif number == 5:
            gold = rect((37,6,52,13)); paint(rect((37,6,52,9)), 11); paint(rect((37,10,52,13)), 0)
            query = 'The rectangle with a yellow top half and white bottom half in the upper-right, including both halves.'
        elif number == 6:
            query = 'The green square.'
        else:
            query = 'The button that wins the game.'
        # Same component + proposal tracker as the current default runtime, with
        # no SAM model inference for synthetic controls. No annotation is used.
        engine = HybridPerception(SimpleNamespace(propose=lambda grid: dict(boxes=[], masks=[])))
        record = engine.measure(None, grid, after_id='synthetic')
        cases.append(dict(name=f'synthetic-{number}', origin='synthetic', game=None, grid=grid,
            query=query, gold_runs=encode(gold), reason='unique' if gold else 'absent' if number==6 else 'role_unknown',
            candidates=pool(record, grid), source={}, annotation='Generated sprite pixels; current runtime components and grouping, no SAM inference.'))
    return cases


def point_for(points, method):
    if not points:
        return None
    if method == 'centroid':
        return click_point({'mask_runs':encode(points)}, {'width':64,'height':64})
    mask = np.zeros((66,66), dtype=bool)
    for x,y in points:
        mask[y+1,x+1] = True
    distances = distance_transform_edt(mask)
    best = max(distances[y+1,x+1] for x,y in points)
    interior = {(x,y) for x,y in points if distances[y+1,x+1] == best}
    sx, sy, n = sum(x for x,y in points), sum(y for x,y in points), len(points)
    return min(interior, key=lambda p: ((p[0]*n-sx)**2+(p[1]*n-sy)**2,p[1],p[0]))


def region_score(points, gold):
    overlap = len(points & gold)
    purity = overlap/len(points) if points else 0
    recall = overlap/len(gold) if gold else 0
    return dict(purity=purity, recall=recall, correct=bool(gold) and purity >= .8 and recall >= .8)


def shortlist(candidates, location):
    """Bounded geometry-only ranking. No query, annotation, or gold argument."""
    if location == 'X':
        return []
    if location == '0':
        ranked = sorted(candidates, key=lambda o: (-len(decode(o['mask_runs'])), o['id']))
    else:
        col, row = (int(location)-1)%3, (int(location)-1)//3
        left, right = col*64/3, (col+1)*64/3
        top, bottom = row*64/3, (row+1)*64/3
        cx, cy = (left+right)/2, (top+bottom)/2
        ranked = []
        for obj in candidates:
            points = decode(obj['mask_runs'])
            coverage = sum(left<=x<right and top<=y<bottom for x,y in points)/len(points)
            if coverage < .5:
                continue
            px,py = point_for(points, 'centroid')
            ranked.append(((-coverage, (px-cx)**2+(py-cy)**2, obj['id']), obj))
        ranked = [obj for _,obj in sorted(ranked, key=lambda p:p[0])]
    # Avoid spending all eight choices on near-identical masks of one object.
    result = []
    for obj in ranked:
        points = decode(obj['mask_runs'])
        if any(len(points & decode(o['mask_runs']))/len(points | decode(o['mask_runs'])) >= .9 for o in result):
            continue
        result.append(obj)
        if len(result) == LIMIT:
            break
    return result


def image_part(image):
    return {'type':'image_url', 'image_url':{'url':'data:image/png;base64,'+png_base64(image)}}


def choices_image(grid, mapping):
    """Each crop uses original colors on the mask; unselected pixels are striped."""
    board = frame_image(grid)
    result = Image.new('RGB', (4*208, ((len(mapping)+3)//4)*224), '#18202b')
    draw = ImageDraw.Draw(result)
    for i,(label,obj) in enumerate(mapping.items()):
        x0,y0 = (i%4)*208, (i//4)*224
        left,top,right,bottom = obj['bbox']
        points = decode(obj['mask_runs'])
        crop = Image.new('RGB',(right-left+1,bottom-top+1))
        for y in range(top,bottom+1):
            for x in range(left,right+1):
                color = board.getpixel((x,y)) if (x,y) in points else ((75,75,75) if (x+y)%2 else (115,115,115))
                crop.putpixel((x-left,y-top), color)
        scale = min(184/crop.width, 174/crop.height)
        crop = crop.resize((max(1,round(crop.width*scale)), max(1,round(crop.height*scale))),Image.Resampling.NEAREST)
        result.paste(crop,(x0+12+(184-crop.width)//2, y0+25+(174-crop.height)//2))
        draw.text((x0+12,y0+6),label+'  MASK',fill='white')
        draw.text((x0+6,y0+204),str(obj['bbox']),fill='white')
    return result


def inventory(mapping):
    return json.dumps([[k,o['bbox'],[COLORS[c] for c in o['colors']],len(decode(o['mask_runs']))]
                       for k,o in mapping.items()], separators=(',',':'))


def payload(system, content, grammar, max_tokens=1):
    return dict(model='qwen3-vl-4b-instruct', messages=[dict(role='system',content=system),
        dict(role='user',content=content)], temperature=0, seed=SEED, max_tokens=max_tokens,
        stream=False, cache_prompt=False, grammar=grammar)


def request_for(grid, query, candidates, arm, permutation, gallery=None):
    ordered = deepcopy(candidates)
    random.Random(SEED+permutation).shuffle(ordered)
    mapping = {str(i+1):obj for i,obj in enumerate(ordered)}
    content = [image_part(render_current(grid))]
    system = SELECT
    if arm == 'choice_visual':
        im = choices_image(grid, mapping)
        if gallery is not None:
            im.save(gallery)
        content += [dict(type='text',text='MASK OPTIONS: crops, not extra game objects. Stripes are outside each mask.'), image_part(im)]
    text = 'Target: '+query+'\nOptions [label, bbox inclusive, color names, mask pixels]:\n'+inventory(mapping)
    text += '\nCoordinates are original pixels, x rightward and y downward, 0..63.'
    labels = ' | '.join(json.dumps(k) for k in mapping)
    if arm == 'free_union':
        system = system.replace('Choose X if no offered mask represents the target',
                                'Return [] if no union of offered masks represents the target')
        system += '\nReturn a JSON list of 1 to 6 option labels whose mask union represents the target. Use [] when unbindable. Do not repeat labels.'
        grammar = 'root ::= "[" (item ("," item)? ("," item)? ("," item)? ("," item)? ("," item)?)? "]"\nitem ::= '+labels
        maximum = 64
    else:
        system += '\nReturn ONE offered option digit, or X if none matches. Never combine masks.'
        grammar = 'root ::= '+labels+' | "X"'
        maximum = 1
    content.append(dict(type='text',text=text))
    return payload(system, content, grammar, maximum), mapping


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    cases = build_cases()
    for case in cases:
        render_current(case['grid']).save(out/(case['name']+'.png'))
        for obj in case['candidates']:
            assert decode(obj['mask_runs'])
    save(out/'cases.json', cases)
    dependencies = [Path(__file__), ROOT/'agent/cognition/object_memory.py', ROOT/'agent/cognition/region_masks.py',
        ROOT/'agent/cognition/perception.py', ROOT/'agent/cognition/geometry.py', ROOT/'agent/cognition/simple_workflow.py',
        ROOT/'agent/cognition/hybrid_perception.py', ROOT/'agent/cognition/proposal_tracking.py',
        ROOT/'agent/rendering.py']
    sources = {}
    for p in dependencies:
        relative = p.relative_to(ROOT)
        dest = out/'sources'/relative
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
        sources[str(relative)] = sha(p)
    save(out/'plan.json', dict(created=datetime.now(timezone.utc).isoformat(), seed=SEED,
        cases_digest=digest(cases), sources=sources, arms=ARMS, permutations=2, shortlist_limit=LIMIT,
        protocol='Freeze all inputs/queries/gold before model calls. Two label orders per case. '
        'free_union uses all masks (max six combined); two choice arms share the same model-localized '
        '3x3 cell and target-independent geometry shortlist (max eight). The visual arm adds masked '
        'thumbnails to identical text and candidate sets. Location X abstains, with no further call. '
        'No cursor loop, retry, or game action. Each choice arm is charged the location call latency. '
        'Two geometric point methods scored offline on every selected mask.',
        scoring='Primary: permutation 0; real/synthetic separate, positives/negatives separate. '
        'Region correct means >=80% target recall and >=80% mask purity. Click correct means '
        'inside annotated intended part. Also shortlist availability, conditional selection, false '
        'clicks on negatives, order sensitivity, API errors, and end-to-end per-arm latency.',
        limitations='Correct visual queries supplied by the assistant, not generated plans. 3 reused '
        'real initial frames plus 8 synthetic scenes. Correlated cases; not statistical proof, '
        'semantic discovery, interactive hit testing, or gameplay. free_union is a mechanism '
        'baseline, not a replay of the entire planner. SAM is frozen; synthetic uses program proposals.'))
    print('Prepared',len(cases),'cases',flush=True)


def verify(out, *, check_current=True):
    plan=json.loads((out/'plan.json').read_text()); cases=json.loads((out/'cases.json').read_text())
    assert digest(cases)==plan['cases_digest']
    for name,h in plan['sources'].items():
        assert sha(out/'sources'/name)==h,name
        if check_current: assert sha(ROOT/name)==h,name
    for c in cases:
        for name,h in c['source'].items(): assert sha(ROOT/name)==h,name
    return plan,cases


def http(base, route, value=None):
    request=urllib.request.Request(base+route, data=json.dumps(value).encode() if value is not None else None,
                                   headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=60) as response:
        return json.load(response)


def run(out, base):
    assert urlparse(base).hostname in ('localhost','127.0.0.1','vlm'), 'local model only'
    _,cases=verify(out)
    assert not (out/'responses.jsonl').exists(), 'Choose a new output; no silent retries'
    save(out/'server-before.json',http(base,'/props'))
    token_audit={v:http(base,'/tokenize',dict(content=v,add_special=False)) for v in '0123456789X'}
    assert all(len(v['tokens'])==1 for v in token_audit.values())
    save(out/'token-audit.json',token_audit)
    jobs=[(c,p) for c in cases for p in range(2)]
    random.Random(SEED).shuffle(jobs)
    start=time.monotonic()
    with (out/'requests.jsonl').open('w') as requests, (out/'responses.jsonl').open('w') as responses:
        def invoke(c,p,arm,body,mapping=None,**extra):
            ident=f'{c["name"]}/{p}/{arm}'
            requests.write(json.dumps(dict(id=ident,payload=body,payload_digest=digest(body)))+'\n');requests.flush()
            t=time.monotonic()
            row=dict(id=ident,case=c['name'],permutation=p,arm=arm,payload_digest=digest(body),
                     mapping={k:o['id'] for k,o in (mapping or {}).items()},**extra)
            try:
                response=http(base,'/v1/chat/completions',body)
                row.update(response=response,answer=response['choices'][0]['message']['content'])
            except Exception as exc:
                row.update(answer='',error=f'{type(exc).__name__}: {exc}')
            row['seconds']=time.monotonic()-t
            responses.write(json.dumps(row)+'\n');responses.flush()
            return row
        for index,(c,p) in enumerate(jobs):
            # No annotations passed to any request/shortlist construction function.
            loc=payload(LOCALIZE,[image_part(render_current(c['grid'])),dict(type='text',text='Target: '+c['query'])],
                        'root ::= '+' | '.join(json.dumps(v) for v in '0123456789X'))
            localized=invoke(c,p,'localize',loc)
            location=localized['answer']
            valid=location in list('0123456789X') and 'error' not in localized
            options=shortlist(c['candidates'],location) if valid else []
            arms=list(ARMS);random.Random(SEED+index).shuffle(arms)
            for arm in arms:
                if arm!='free_union' and (not valid or not options):
                    row=dict(id=f'{c["name"]}/{p}/{arm}',case=c['name'],permutation=p,arm=arm,
                        answer='X' if valid else '',mapping={},seconds=0,location=location,
                        location_seconds=localized['seconds'],skipped=True,location_valid=valid)
                    responses.write(json.dumps(row)+'\n');responses.flush()
                    continue
                candidates=c['candidates'] if arm=='free_union' else options
                body,mapping=request_for(c['grid'],c['query'],candidates,arm,p,
                    out/f'{c["name"]}-{p}-choices.png' if arm=='choice_visual' else None)
                invoke(c,p,arm,body,mapping,location=location,location_valid=valid,
                       location_seconds=localized['seconds'] if arm!='free_union' else 0)
            if (index+1)%4==0:
                print(index+1,'/',len(jobs),'case/orders;',round(time.monotonic()-start,1),'seconds',flush=True)
    save(out/'server-after.json',http(base,'/props'))
    save(out/'health-after.json',http(base,'/health'))


def decode_union(answer, mapping):
    labels = json.loads(answer)
    assert isinstance(labels, list) and len(labels) <= 6
    assert all(type(k) is int for k in labels)
    assert len(set(labels)) == len(labels)
    labels = [str(k) for k in labels]
    assert all(k in mapping for k in labels)
    return labels


def analyze(out):
    # Frozen execution sources remain immutable; analysis can fix decoding bugs.
    _,cases=verify(out, check_current=False); bycase={c['name']:c for c in cases}
    raw=read_rows(out/'responses.jsonl'); rows=[]
    assert len(raw)==len(cases)*2*4 and len({r['id'] for r in raw})==len(raw)
    requests={r['id']:r for r in read_rows(out/'requests.jsonl')}
    for r in raw:
        if not r.get('skipped'):
            assert r['payload_digest']==digest(requests[r['id']]['payload'])
        if r['arm']=='localize': continue
        c=bycase[r['case']]; objects={o['id']:o for o in c['candidates']}
        gold=decode(c['gold_runs']); selected=[]; valid='error' not in r
        try:
            if r['arm']=='free_union':
                labels=decode_union(r['answer'], r['mapping'])
            else:
                assert r['answer']=='X' or r['answer'] in r['mapping']
                labels=[] if r['answer']=='X' else [r['answer']]
            selected=[objects[r['mapping'][k]] for k in labels]
        except (ValueError,AssertionError,TypeError,KeyError): valid=False
        support=set().union(*(decode(o['mask_runs']) for o in selected))
        offered=[objects[ident] for ident in r['mapping'].values()]
        scores=region_score(support,gold)
        row={k:v for k,v in r.items() if k!='response'}
        row.update(origin=c['origin'],reason=c['reason'],valid=valid,selected_ids=[o['id'] for o in selected],
            region_correct=valid and scores['correct'],purity=scores['purity'],recall=scores['recall'],
            selected=bool(selected),positive=bool(gold),
            full_pool_available=any(region_score(decode(o['mask_runs']),gold)['correct'] for o in c['candidates']),
            offered_available=any(region_score(decode(o['mask_runs']),gold)['correct'] for o in offered),
            total_seconds=r['seconds']+r.get('location_seconds',0),
            abstention_correct=valid and not gold and not selected,
            false_click=valid and not gold and bool(selected))
        for method in ('centroid','interior'):
            point=point_for(support,method)
            row[method+'_point']=point
            row[method+'_hit']=valid and point is not None and point in gold
        rows.append(row)
    def stats(group):
        pos=[r for r in group if r['positive']]; neg=[r for r in group if not r['positive']]
        available=[r for r in pos if r['offered_available']]
        return dict(cases=len(group),positive=len(pos),negative=len(neg),
            region_correct=sum(r['region_correct'] for r in pos),
            centroid_hit=sum(r['centroid_hit'] for r in pos),interior_hit=sum(r['interior_hit'] for r in pos),
            full_pool_available=sum(r['full_pool_available'] for r in pos),
            offered_available=len(available),conditional_region_correct=sum(r['region_correct'] for r in available),
            correct_abstentions=sum(r['abstention_correct'] for r in neg),false_clicks=sum(r['false_click'] for r in neg),
            valid=sum(r['valid'] for r in group),
            median_seconds=statistics.median(r['total_seconds'] for r in group),
            median_positive_seconds=statistics.median(r['total_seconds'] for r in pos) if pos else None)
    summary={}
    for arm in ARMS:
        chosen=[r for r in rows if r['arm']==arm]; first=[r for r in chosen if r['permutation']==0]
        summary[arm]=dict(primary=stats(first),real=stats([r for r in first if r['origin']=='real']),
            synthetic=stats([r for r in first if r['origin']=='synthetic']),
            all_orders=stats(chosen),
            both_orders_hit=sum(all(r['centroid_hit'] for r in chosen if r['case']==c['name']) for c in cases if c['reason']=='unique'),
            hit_order_changes=sum(len({r['centroid_hit'] for r in chosen if r['case']==c['name']})>1 for c in cases if c['reason']=='unique'))
    assert json.loads((out/'server-before.json').read_text())==json.loads((out/'server-after.json').read_text())
    assert json.loads((out/'health-after.json').read_text())['status']=='ok'
    save(out/'scored.json',rows);save(out/'summary.json',summary)
    save(out/'verification.json',dict(passed=True,rows=len(raw),requests=len(requests),
        errors=sum('error' in r for r in raw),same_server=True,frozen_inputs=True,
        analyzer_sha256=sha(Path(__file__)), analysis_revision='numeric-label-decoding-v2'))
    lines=['# Click-part selection experiment','',
           '| arm | target masks /18 | centroid hits /18 | interior hits /18 | correct abstentions /8 | median seconds |',
           '|---|---:|---:|---:|---:|---:|']
    for arm,data in summary.items():
        s=data['primary'];lines.append(f'| {arm} | {s["region_correct"]} | {s["centroid_hit"]} | {s["interior_hit"]} | {s["correct_abstentions"]} | {s["median_seconds"]:.3f} |')
    lines+=['','Primary = first label order. Location call is included in both choice-arm times.',
            'See summary.json for real/synthetic split and shortlist losses. No game actions or hitbox proof.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    gallery=['<!doctype html><meta charset="utf-8"><title>Click choices</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto}img{max-width:100%}article{border-top:1px solid #aaa;padding:20px}pre{white-space:pre-wrap}</style><h1>Click-part selection</h1>']
    for c in cases:
        gallery+=['<article><h2>'+html.escape(c['name'])+'</h2><p>'+html.escape(c['query'])+'</p>',
            f'<img width="320" src="{c["name"]}.png">']
        annotated=render_current(c['grid']);draw=ImageDraw.Draw(annotated)
        for x,y in decode(c['gold_runs']):draw.rectangle((32+x*6,44+y*6,32+x*6+5,44+y*6+5),outline='#00ff88')
        annotated.save(out/f'{c["name"]}-gold.png')
        gallery+=[f'<img width="320" src="{c["name"]}-gold.png"><p>Green outlines: evaluation-only target support, never sent to model.</p>']
        for r in rows:
            if r['case']==c['name'] and r['permutation']==0:
                gallery+=['<pre>'+html.escape(json.dumps({k:r[k] for k in ('arm','answer','location','selected_ids','offered_available','region_correct','centroid_point','centroid_hit','interior_point','interior_hit','total_seconds')},indent=2))+'</pre>']
        p=out/f'{c["name"]}-0-choices.png'
        if p.exists():gallery+=[f'<img src="{p.name}">']
        gallery+=['</article>']
    (out/'gallery.html').write_text('\n'.join(gallery))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','analyze'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--base',default='http://vlm:8080')
    args=parser.parse_args()
    if args.command=='run':run(args.output,args.base)
    else:globals()[args.command](args.output)
