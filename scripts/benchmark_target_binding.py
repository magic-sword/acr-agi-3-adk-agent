"""Frozen single-target quizzes against real runtime candidates; no game actions.

Gold comes from synthetic sprite support or explicit annotations of recorded pixels,
never from a model answer. All candidates are offered (no oracle shortlisting).
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
from itertools import groupby
import json
from pathlib import Path
import random
import statistics
import string
import sys
import time
import urllib.request
from urllib.parse import urlparse

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.cognition.geometry import extract
from agent.cognition.perception import brief, COLORS
from scripts.benchmark_recognition import encode

ARMS = ('raw_image', 'compact_text', 'compact_image')
LABELS = string.ascii_uppercase.replace('X', '') + string.ascii_lowercase
SOURCE = Path('outputs/measured-runtime-20260928')
PALETTE_SOURCE = Path('outputs/instance-state-20260928-final/cases.json')
SYSTEM = ('Match ONE target description to ONE complete measured candidate in this observation. '
          'Use visible appearance and spatial relations, not guessed game roles. '
          'Choose a candidate only if it uniquely represents the ENTIRE described target. '
          'Choose X if multiple candidates fit, the target is absent, only fragments are available, '
          'or the description needs an unverified role or action history. '
          'Do not combine candidates or select a fragment. Output only the offered option letter. '
          'Coordinates are original pixels: x increases rightward, y downward; bounds are inclusive. '
          'No past actions or established game roles are supplied.')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(value):
    # Integer color keys become strings on disk; hash the persisted representation.
    canonical = json.loads(json.dumps(value))
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def http(url, value=None, timeout=120):
    data = None if value is None else json.dumps(value).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def support(obj):
    if 'children' in obj:
        return {(x, y) for c in obj['children']
                for y in range(c['bbox'][1], c['bbox'][3] + 1)
                for x in range(c['bbox'][0], c['bbox'][2] + 1)}
    pattern = obj['pattern']
    if isinstance(pattern, dict):
        pattern = [[v for v, n in row for _ in range(n)] for row in pattern['row_runs']]
    x0, y0 = obj['bbox'][:2]
    return {(x0+x, y0+y) for y, row in enumerate(pattern)
            for x, v in enumerate(row) if v is not None}


def rectangle(box):
    return {(x, y) for y in range(box[1], box[3]+1) for x in range(box[0], box[2]+1)}


def sprite(grid, x, y, pattern):
    points = set()
    for yy, row in enumerate(pattern):
        for xx, color in enumerate(row):
            if color is not None:
                assert grid[y+yy][x+xx] == 5
                grid[y+yy][x+xx] = color
                points.add((x+xx, y+yy))
    return points


def case(name, origin, group, grid, query, targets, reason, source=None):
    start = time.perf_counter()
    measured = extract(grid, 'a')
    seconds = time.perf_counter() - start
    candidates = measured['instances']
    assert not measured.get('extraction_limited') and len(candidates) <= len(LABELS)
    matches = [[o['id'] for o in candidates if support(o) == set(target)] for target in targets]
    if reason == 'unique':
        assert len(matches) == 1 and len(matches[0]) == 1, (name, matches)
        gold = matches[0][0]
    else:
        gold = None
        if reason == 'ambiguous':
            assert len(matches) >= 2 and all(len(m) == 1 for m in matches), (name, matches)
        if reason == 'fragmented':
            assert len(matches) == 1 and not matches[0], (name, matches)
            pieces = [support(o) for o in candidates if support(o) <= set(targets[0])]
            assert set().union(*pieces) == set(targets[0]), name
    return dict(name=name, origin=origin, group=group, grid=grid, query=query,
                candidates=candidates, gold_candidate=gold, gold_reason=reason,
                annotated_supports=[sorted(t) for t in targets], measurement_seconds=seconds,
                source=source, annotation='generated sprite support' if origin == 'synthetic'
                else 'manual visible pixel annotation; no game rules inferred')


def build_cases():
    cases = []
    for group in ('color', 'shape', 'relation', 'pattern', 'ambiguous_absent', 'role_fragment'):
        for i in range(8):
            rng = random.Random(928510 + i)
            positions = [(10+rng.randrange(4), 10+rng.randrange(4)),
                         (43+rng.randrange(4), 10+rng.randrange(4)),
                         (10+rng.randrange(4), 43+rng.randrange(4)),
                         (43+rng.randrange(4), 43+rng.randrange(4))]
            rng.shuffle(positions)
            g = [[5]*64 for _ in range(64)]
            targets = []; reason = 'unique'
            if group == 'color':
                colors = [9, 12, 14, 8]
                shapes = [sprite(g, x, y, [[c]*5 for _ in range(5)])
                          for (x, y), c in zip(positions, colors)]
                targets = [shapes[i % 4]]
                query = 'The solid ' + COLORS[colors[i % 4]] + ' square.'
            elif group == 'shape':
                masks = [[[None, None, 9, None, None], [None, None, 9, None, None],
                          [9]*5, [None, None, 9, None, None], [None, None, 9, None, None]],
                         [[9, None, None, None, None]]*4 + [[9]*5], [[9]*7]*2, [[9]*5]*5]
                shapes = [sprite(g, x, y, m) for (x, y), m in zip(positions, masks)]
                targets = [shapes[i % 4]]
                query = ['The blue plus-shaped cross.', 'The blue L-shaped object.',
                         'The thin blue horizontal bar.', 'The solid blue square.'][i % 4]
            elif group == 'relation':
                cx, cy = 29+rng.randrange(4), 29+rng.randrange(4)
                sprite(g, cx, cy, [[8]*4]*4)
                locations = [(cx, cy-16), (cx+16, cy), (cx, cy+16), (cx-16, cy)]
                shapes = [sprite(g, x, y, [[9]*4]*4) for x, y in locations]
                targets = [shapes[i % 4]]
                query = 'The blue square directly ' + ['above', 'to the right of', 'below', 'to the left of'][i % 4] + ' the red square.'
            elif group == 'pattern':
                patterns = [[[9]*4]*2+[[12]*4]*2, [[9, 9, 12, 12]]*4,
                            [[12]*4]*2+[[9]*4]*2, [[12, 12, 9, 9]]*4]
                shapes = [sprite(g, x, y, m) for (x, y), m in zip(positions, patterns)]
                targets = [shapes[i % 4]]
                query = ['The square with a blue top half and orange bottom half.',
                         'The square with a blue left half and orange right half.',
                         'The square with an orange top half and blue bottom half.',
                         'The square with an orange left half and blue right half.'][i % 4]
            elif group == 'ambiguous_absent':
                shapes = [sprite(g, x, y, [[c]*5]*5)
                          for (x, y), c in zip(positions, [9, 9, 12, 8])]
                if i < 4:
                    query = 'The solid blue square.'; targets = shapes[:2]; reason = 'ambiguous'
                else:
                    query = 'The solid green square.'; reason = 'absent'
            else:
                if i < 4:
                    for (x, y), c in zip(positions, [9, 12, 14, 8]):
                        sprite(g, x, y, [[c]*5]*5)
                    query = ['The player controlled by directional buttons.',
                             'The goal that completes the level.'][i % 2]
                    reason = 'role_unknown'
                else:
                    x, y = positions[0]
                    targets = [sprite(g, x, y, [[None, 0, None], [1, 0, 0], [None, 1, None]])]
                    for (x, y), c in zip(positions[1:], [9, 12, 8]):
                        sprite(g, x, y, [[c]*5]*5)
                    query = 'The entire small white and light gray plus-shaped cross.'
                    reason = 'fragmented'
            cases.append(case(f'synthetic-{group}-{i}', 'synthetic', group, g, query, targets, reason))

    definitions = {
        'ls20': [
            ('top-display', 'The black square containing blue pixels near the top of the board.', 'unique', [[33, 9, 39, 15]]),
            ('bicolor', 'The small rectangle with orange above blue near the lower center.', 'unique', [[34, 45, 38, 49]]),
            ('yellow-bar', 'The long yellow horizontal bar along the bottom.', 'unique', [[13, 61, 54, 62]]),
            ('red-ambiguous', 'The small red square along the bottom.', 'ambiguous', [[56, 61, 57, 62], [59, 61, 60, 62], [62, 61, 63, 62]]),
            ('player-role', 'The player controlled by directional buttons.', 'role_unknown', []),
            ('cross-fragment', 'The entire white and light gray cross on the gray platform.', 'fragmented', [])],
        'vc33': [
            ('pink-bar', 'The pink horizontal line across the very top.', 'unique', [[0, 0, 63, 0]]),
            ('lower-blue', 'The blue square at the right edge below the long black horizontal bar.', 'unique', [[60, 32, 63, 35]]),
            ('yellow-vertical', 'The narrow yellow vertical rectangle beside the small dark gray shape in the lower half.', 'unique', [[50, 44, 51, 49]]),
            ('blue-ambiguous', 'The blue square at the right edge.', 'ambiguous', [[60, 24, 63, 27], [60, 32, 63, 35]]),
            ('goal-role', 'The exit that completes the level.', 'role_unknown', []),
            ('bar-fragment', 'The entire long horizontal bar, mostly black with a yellow segment inside it.', 'fragmented', [[20, 28, 63, 31]])],
        'ft09': [
            ('top-left-blue', 'The blue square at the far left of the topmost row.', 'unique', [[4, 2, 9, 7]]),
            ('top-left-red', 'The leftmost red square in the topmost row.', 'unique', [[12, 2, 17, 7]]),
            ('orange-bar', 'The orange horizontal line across the very bottom.', 'unique', [[0, 63, 63, 63]]),
            ('red-ambiguous', 'The red square in the topmost row.', 'ambiguous', [[12, 2, 17, 7], [46, 2, 51, 7]]),
            ('goal-role', 'The goal tile that completes the level.', 'role_unknown', []),
            ('panel-fragment', 'The entire white and gray patterned square in the center of the upper-left group of nine squares.', 'fragmented', [[12, 10, 17, 15]])],
    }
    for game, specs in definitions.items():
        path = next((SOURCE/f'live-{game}-measured').glob('*/cognition/*.observations.jsonl'))
        observation = json.loads(path.read_text().splitlines()[0]); g = observation['grid']
        for name, query, reason, boxes in specs:
            targets = [rectangle(box) for box in boxes]
            if name == 'cross-fragment':
                targets = [{(x, y) for y in range(30, 35) for x in range(19, 24) if g[y][x] in (0, 1)}]
            cases.append(case(game+'-'+name, game, reason, deepcopy(g), query, targets, reason,
                              dict(path=str(path),sha256=sha(path),line=1,observation_id=observation['observation_id'])))
    return cases


def compact(obj):
    """Target-independent lossless local layout; no role or target-derived hints."""
    text = f"box={obj['bbox']}; shape={obj['class_name']}; colors=" + ','.join(COLORS[int(c)] for c in obj['colors'])
    if obj['class_name'] != 'solid_rectangle':
        p = obj['pattern']
        if isinstance(p, dict):
            p = [[v for v, n in row for _ in range(n)] for row in p['row_runs']]
        rows = [','.join(('empty' if v is None else COLORS[int(v)])+'*'+str(len(list(items)))
                         for v, items in groupby(row)) for row in p]
        text += '; rows(top to bottom)=' + ' / '.join(str(len(list(items)))+'x['+row+']' for row, items in groupby(rows))
        if 'children' in obj:
            text += f"; cell_size={obj['cell_size']}; cell_spacing={obj['cell_spacing']}"
    return text


def image_for(grid, palette):
    board = Image.new('RGB', (64,64))
    board.putdata([tuple(int(palette[str(c)][k:k+2],16) for k in (1,3,5)) for row in grid for c in row])
    result = Image.new('RGB', (428,452), '#18202b')
    result.paste(board.resize((384,384), getattr(Image, 'Resampling', Image).NEAREST), (32,44))
    d = ImageDraw.Draw(result); d.text((12,10),'CURRENT: game pixels',fill='white')
    for n in list(range(0,64,8))+[63]:
        d.text((32+n*6,28),str(n),fill='white');d.text((3,44+n*6),str(n),fill='white')
    return result


def request_for(c, arm, permutation, image):
    candidates = deepcopy(c['candidates'])
    random.Random(928800 + permutation*1000 + int(hashlib.sha256(c['name'].encode()).hexdigest()[:6],16)).shuffle(candidates)
    mapping = {label: obj['id'] for label, obj in zip(LABELS, candidates)}
    if arm == 'raw_image':
        records = []
        for label, obj in zip(LABELS, candidates):
            r = brief(obj); r['option'] = label; records.append(r)
        inventory = 'Measured candidates: '+json.dumps(records,separators=(',',':'))
        inventory += '\nPattern color IDs: '+json.dumps(dict(enumerate(COLORS)),separators=(',',':'))
    else:
        inventory = '\n'.join(label+'. '+compact(obj) for label,obj in zip(LABELS,candidates))
    content = [] if arm == 'compact_text' else [deepcopy(image)]
    content.append(dict(type='text',text='Target: '+c['query']+'\n'+inventory+'\nX. Cannot uniquely bind the complete target.'))
    payload = dict(model='qwen3-vl-4b-instruct',messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)],
                   temperature=0,seed=928510,max_tokens=1,stream=False,cache_prompt=False,
                   grammar='root ::= '+' | '.join(json.dumps(x) for x in list(mapping)+['X']))
    return payload, mapping


def prepare(out):
    out.mkdir(parents=True,exist_ok=False); (out/'sources').mkdir()
    palette=json.loads(PALETTE_SOURCE.read_text())[0]['palette']; cases=build_cases();jobs=[]
    for c in cases:
        im=image_for(c['grid'],palette);im.save(out/(c['name']+'.png')); image=encode(im)
        for arm in ARMS:
            for permutation in range(2):
                payload,mapping=request_for(c,arm,permutation,image)
                for repeat in range(2):
                    jobs.append(dict(id=len(jobs),case=c['name'],arm=arm,permutation=permutation,repeat=repeat,
                                     mapping=mapping,payload=payload,payload_digest=digest(payload)))
    save(out/'cases.json',cases);save(out/'jobs.json',jobs)
    paths=[Path(__file__),Path('agent/cognition/geometry.py'),Path('agent/cognition/perception.py'),Path('scripts/benchmark_recognition.py')]
    for p in paths:
        shutil_path=out/'sources'/p.name;shutil_path.write_bytes(p.read_bytes())
    save(out/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),cases=len(cases),requests=len(jobs),
         arms=ARMS,permutations=2,repeats=2,cases_digest=digest(cases),jobs_digest=digest(jobs),
         source_hashes={str(p):sha(p) for p in paths},palette_sha256=sha(PALETTE_SOURCE),
         input_hashes={str(c['source']['path']):c['source']['sha256'] for c in cases if c['source']},
         image_hashes={p.name:sha(p) for p in out.glob('*.png')},
         design='66 fixed quizzes: 48 new synthetic and 18 annotations on 3 previously seen real frames. All runtime candidates offered, including 43 in ft09. No target-dependent shortlist. 3 input formats, 2 independently shuffled label mappings, 2 exact timing repeats. Same single-task instruction in all arms. Frozen before first call; randomized interleaving; greedy one-token grammar; cache disabled.',
         primary='Accuracy on first permutation/repeat per unique quiz; report synthetic/real and positive/abstention separately. Also all-four-correct stability, selection precision, false binding on unbindable cases, repeat and permutation disagreement, time/tokens.',
         limitations='One COMPLETE candidate only; arbitrary candidate sets/group assembly not evaluated. All candidate information is available, unlike current 24-candidate truncation. Real frames reused, manually authored visual queries, correlated questions. Not role learning, target description generation, temporal tracking, action-effect association, end-to-end gameplay, or proof that quiz decomposition outperforms omnibus planning. Small benchmark, no prompt tuning after results.'))
    print('Prepared',len(cases),'cases,',len(jobs),'requests',flush=True)


def verify_inputs(out):
    load=lambda n:json.loads((out/n).read_text())
    plan=load('plan.json');cases=load('cases.json');jobs=load('jobs.json')
    assert digest(cases)==plan['cases_digest'] and digest(jobs)==plan['jobs_digest']
    for p,h in plan['source_hashes'].items():
        assert sha(p)==h and sha(out/'sources'/Path(p).name)==h, p
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    for p,h in plan['image_hashes'].items():assert sha(out/p)==h,p
    assert sha(PALETTE_SOURCE)==plan['palette_sha256']
    for j in jobs:assert digest(j['payload'])==j['payload_digest']
    return plan,cases,jobs


def run(out, base):
    parsed=urlparse(base)
    assert parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','localhost'), 'local server only'
    plan,cases,jobs=verify_inputs(out)
    assert not (out/'responses.jsonl').exists(), 'Use a new output directory; no silent retries'
    save(out/'server-before.json',http(base+'/props'))
    letters=sorted(set('X').union(*(set(j['mapping']) for j in jobs)))
    audit={s:http(base+'/tokenize',dict(content=s,add_special=False)) for s in letters}
    assert all(len(v['tokens'])==1 for v in audit.values())
    save(out/'token-audit.json',audit)
    jobs=jobs.copy();random.Random(928771).shuffle(jobs)
    started=time.monotonic()
    with (out/'responses.jsonl').open('w') as f:
        for i,j in enumerate(jobs):
            r={k:j[k] for k in ('id','case','arm','permutation','repeat','payload_digest')}
            start=time.monotonic()
            try:
                response=http(base+'/v1/chat/completions',j['payload'])
                r.update(response=response,answer=response['choices'][0]['message']['content'])
            except Exception as exc:
                r.update(error=f'{type(exc).__name__}: {exc}',answer='')
            r['seconds']=time.monotonic()-start;f.write(json.dumps(r)+'\n');f.flush()
            if (i+1)%30==0 or 'error' in r:print(i+1,'/',len(jobs),round(time.monotonic()-started,1),'seconds',r.get('error',''),flush=True)
    save(out/'server-after.json',http(base+'/props'));save(out/'health-after.json',http(base+'/health'))


def analyze(out):
    plan,cases,jobs=verify_inputs(out);bycase={c['name']:c for c in cases}
    rows=[json.loads(l) for l in (out/'responses.jsonl').read_text().splitlines()]
    assert Counter(r['id'] for r in rows)==Counter(j['id'] for j in jobs)
    for r in rows:
        j=jobs[r['id']];c=bycase[r['case']]
        assert all(r[k]==j[k] for k in ('case','arm','permutation','repeat','payload_digest'))
        r['valid']=r['answer'] in list(j['mapping'])+['X'] and r.get('response',{}).get('usage',{}).get('completion_tokens')==1
        r['selected']=j['mapping'].get(r['answer']) if r['valid'] else '<invalid>'
        r['correct']=r['valid'] and r['selected']==c['gold_candidate']
        r.update(origin=c['origin'],group=c['group'],gold_reason=c['gold_reason'],gold_candidate=c['gold_candidate'])
    def metrics(rs):
        selected=[r for r in rs if r['valid'] and r['answer']!='X']
        negatives=[r for r in rs if r['gold_candidate'] is None]
        positives=[r for r in rs if r['gold_candidate'] is not None]
        return dict(correct=sum(r['correct'] for r in rs),total=len(rs),
                    bound_correct=sum(r['correct'] for r in positives),bindable=len(positives),
                    abstention_correct=sum(r['correct'] for r in negatives),unbindable=len(negatives),
                    false_bindings=sum(r['valid'] and r['answer']!='X' for r in negatives),
                    selection_correct=sum(r['correct'] for r in selected),selections=len(selected))
    arms=[]
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm];first=[r for r in rr if r['permutation']==r['repeat']==0]
        grouped={c['name']:[r for r in rr if r['case']==c['name']] for c in cases}
        exact=sum(len({r['selected'] for r in rs if r['permutation']==p})>1 for rs in grouped.values() for p in range(2))
        ordering=sum(len({r['selected'] for r in rs if r['repeat']==0})>1 for rs in grouped.values())
        arms.append(dict(arm=arm,primary=metrics(first),
            synthetic=metrics([r for r in first if r['origin']=='synthetic']),real=metrics([r for r in first if r['origin']!='synthetic']),
            by_group={g:metrics([r for r in first if r['group']==g]) for g in sorted({r['group'] for r in first})},
            by_gold_reason={g:metrics([r for r in first if r['gold_reason']==g]) for g in sorted({r['gold_reason'] for r in first})},
            by_origin={g:metrics([r for r in first if r['origin']==g]) for g in sorted({r['origin'] for r in first})},
            all_four_correct=sum(all(r['correct'] for r in rs) for rs in grouped.values()),
            repeat_disagreements=exact,permutation_disagreements=ordering,valid=sum(r['valid'] for r in rr),calls=len(rr),
            median_seconds=statistics.median(r['seconds'] for r in rr),
            median_prompt_tokens=statistics.median(r.get('response',{}).get('usage',{}).get('prompt_tokens',0) for r in rr)))
    save(out/'summary.json',dict(arms=arms,rows=rows,measurement_median_seconds=statistics.median(c['measurement_seconds'] for c in cases)))
    assert json.loads((out/'server-before.json').read_text())==json.loads((out/'server-after.json').read_text())
    assert json.loads((out/'health-after.json').read_text())['status']=='ok'
    save(out/'verification.json',dict(passed=True,requests=len(rows),valid=sum(r['valid'] for r in rows),
                                    frozen_inputs=True,all_jobs_once=True,same_server=True,
                                    all_candidates_offered=True,permutations_scored_by_candidate_id=True))
    gallery=['<!doctype html><meta charset="utf-8"><title>Target binding quizzes</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto}article{border-top:1px solid #aaa;padding:20px}img{width:300px;float:left;margin-right:20px}article:after{content:"";display:block;clear:both}pre{white-space:pre-wrap}td,th{padding:6px;border:1px solid #bbb}table{border-collapse:collapse}</style><h1>Target binding: frozen single-target quizzes</h1><p>Selections are decoded candidate IDs. Four calls per arm: two label orders × two exact repeats. X = cannot bind a complete unique candidate. No game actions.</p><pre>'+html.escape(json.dumps(arms,indent=2))+'</pre>']
    for c in cases:
        gallery.append('<article><h2>'+html.escape(c['name'])+'</h2><img src="'+c['name']+'.png"><p>'+html.escape(c['query'])+'</p><p>Gold: '+html.escape(str(c['gold_candidate'] or 'X')+' / '+c['gold_reason'])+'</p><table><tr><th>Arm</th><th>p0r0</th><th>p0r1</th><th>p1r0</th><th>p1r1</th></tr>')
        for arm in ARMS:
            rr=sorted([r for r in rows if r['case']==c['name'] and r['arm']==arm],key=lambda r:(r['permutation'],r['repeat']))
            gallery.append('<tr><td>'+arm+'</td>'+''.join('<td>'+html.escape(str(r['selected'] or 'X'))+(' ✓' if r['correct'] else ' ✗')+'</td>' for r in rr)+'</tr>')
        gallery.append('</table><details><summary>All candidates and gold annotation</summary><pre>'+html.escape('\n'.join(o['id']+': '+compact(o) for o in c['candidates'])+'\n\n'+json.dumps(dict(annotation=c['annotation'],supports=c['annotated_supports'],source=c['source'])))+'</pre></details></article>')
    (out/'gallery.html').write_text('\n'.join(gallery))
    print(json.dumps(arms,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','analyze'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--base',default='http://127.0.0.1:8080')
    args=parser.parse_args()
    if args.command=='run':run(args.output,args.base)
    else:globals()[args.command](args.output)
