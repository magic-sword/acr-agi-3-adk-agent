"""Compare image-only object proposals with the frozen geometric extractor.

Concept descriptions are reviewed separately from numerical localization. No
target names, candidate lists, game rules or annotations enter model requests.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageColor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.cognition.geometry import extract, bounds
from scripts.benchmark_target_binding import digest, sha, save, http, support
from scripts.benchmark_recognition import encode

SOURCE = Path('outputs/target-binding-quiz-20260928/cases.json')
PALETTE = Path('outputs/instance-state-20260928-final/cases.json')
GROUPS = ('single_color', 'two_color_rectangle', 'small_cross', 'large_cross', 'two_color_L')
ARMS = ('concepts', 'boxes')
SYSTEM = ('List the distinct visible foreground objects in the image. Describe each whole object '
          'once, keeping its visually coherent parts together even when their colors differ. '
          'List separate copies separately. Ignore background regions and image coordinate labels. '
          'Describe appearance, not game roles. Return only JSON.')
PROMPT = ('Return {"objects":[{"description":"short visible appearance",'
          '"location":"approximate place in the board"}]}. '
          'Use one short entry per whole object. Do not include an overall scene summary.')
BOX_PROMPT = ('Return {"objects":[{"description":"short visible appearance",'
              '"bbox":[left,top,right,bottom]}]}. Use one short entry per whole object. '
              'Give tight bounding boxes in ORIGINAL 64 by 64 board pixels, x rightwards and y downwards, '
              'integers 0 through 63, inclusive endpoints. Do not include an overall scene summary.')
RUBRIC = (
    'Primary: concept arm, first repeat, synthetic scenes fully annotated before inference. '
    'Review each output entry as matching one complete gold object or no match; use appearance and coarse location, '
    'not exact coordinates. A cross/plus or L description must refer to the whole coherent shape, not individual '
    'colored strokes. A two-color rectangle must group both colors. A generic colored shape without enough '
    'information to distinguish a whole object from its fragments is insufficient. Contradictory identifying '
    'color/shape or coarse location fails. A group of multiple separated copies is not one object. '
    'Synonyms accepted; duplicate entries match at most once. Unmatched entries count as false proposals only on '
    'fully annotated synthetic scenes. Program full-object grouping uses exact support equality, without requiring '
    'semantic names. This measures completeness of structural candidates versus useful whole-object descriptions; '
    'it is not the same-output segmentation metric. Separate common bounding-box IoU and exact-box matching '
    'for program and boxes arm; names ignored for that spatial metric. Real frames have partial annotations: '
    'report only recall of annotated targets, not overall precision or complete-scene accuracy. '
    'Review by implementing assistant, not independent human ground truth; case order shuffled and arm key hidden '
    'but coordinate output reveals format. Count invalid/truncated replies as failure and report separately.')


def rotate(pattern):
    return [list(row) for row in zip(*pattern[::-1])]


def synthetic_cases():
    result = []
    for group in GROUPS:
        for variation in range(4):
            rng = random.Random(929201 + variation)
            locations = [(9+rng.randrange(5), 9+rng.randrange(5)), (43+rng.randrange(5), 9+rng.randrange(5)),
                         (9+rng.randrange(5), 43+rng.randrange(5)), (43+rng.randrange(5), 43+rng.randrange(5))]
            rng.shuffle(locations)
            grid = [[5]*64 for _ in range(64)]; gold = []
            for i, (x0, y0) in enumerate(locations):
                color = [9, 12, 14, 8][(i+variation) % 4]
                if group == 'single_color':
                    masks = [[[color]*5 for _ in range(5)], [[color]*8 for _ in range(2)],
                             [[color if x == 2 or y == 2 else None for x in range(5)] for y in range(5)],
                             [[color if x == 0 or y == 4 else None for x in range(5)] for y in range(5)]]
                    pattern = masks[i]; kind = ['square', 'bar', 'cross', 'L'][i]
                elif group == 'two_color_rectangle':
                    pattern = [[9]*6 for _ in range(3)]+[[12]*6 for _ in range(3)]
                    for _ in range(i): pattern = rotate(pattern)
                    kind = 'two-color square'
                elif group == 'small_cross':
                    pattern = [[None, 0, None], [1, 0, 0], [None, 1, None]]
                    for _ in range(i): pattern = rotate(pattern)
                    kind = 'white and light gray cross'
                elif group == 'large_cross':
                    pattern = [[9 if x == 3 else 12 if y == 3 else None for x in range(7)] for y in range(7)]
                    for _ in range(i): pattern = rotate(pattern)
                    kind = 'blue and orange cross'
                else:
                    pattern = [[9 if x < 2 else 12 if y >= 5 else None for x in range(7)] for y in range(7)]
                    for _ in range(i): pattern = rotate(pattern)
                    kind = 'blue and orange L'
                points = []
                for y, row in enumerate(pattern):
                    for x, v in enumerate(row):
                        if v is not None:
                            assert grid[y0+y][x0+x] == 5
                            grid[y0+y][x0+x] = v; points.append([x0+x, y0+y])
                colors = sorted({grid[y][x] for x, y in points})
                gold.append(dict(id=f'g{i+1}', kind=kind, colors=colors, support=points, bbox=bounds(points),
                                 location=('upper' if y0 < 32 else 'lower')+' '+('left' if x0 < 32 else 'right')))
            result.append(dict(name=f'{group}-{variation}', origin='synthetic', group=group, grid=grid,
                               gold=gold, fully_annotated=True))
    return result


def real_cases():
    source = json.loads(SOURCE.read_text()); result = []
    for game in ('ls20', 'vc33', 'ft09'):
        cc = [c for c in source if c['origin'] == game]; gold = []; seen = set()
        for c in cc:
            for pts in c['annotated_supports']:
                key = frozenset(tuple(p) for p in pts)
                if key in seen: continue
                seen.add(key)
                gold.append(dict(id=f'g{len(gold)+1}', kind=c['query'], support=pts, bbox=bounds(pts),
                                 colors=sorted({c['grid'][y][x] for x, y in pts}), source_case=c['name']))
        result.append(dict(name=game, origin=game, group='real', grid=cc[0]['grid'], gold=gold,
                           fully_annotated=False, source=cc[0]['source']))
    return result


def render(grid, palette):
    board = Image.new('RGB', (64, 64)); board.putdata([ImageColor.getrgb(palette[str(v)]) for row in grid for v in row])
    image = Image.new('RGB', (428,452), '#18202b')
    image.paste(board.resize((384,384), getattr(Image, 'Resampling', Image).NEAREST), (32,44))
    d = ImageDraw.Draw(image); d.text((12,10), 'CURRENT: game pixels', fill='white')
    for n in list(range(0,64,8))+[63]:
        d.text((32+n*6,28), str(n), fill='white'); d.text((3,44+n*6), str(n), fill='white')
    return image


def prepare(out):
    out.mkdir(parents=True, exist_ok=False); (out/'sources').mkdir()
    cases = synthetic_cases()+real_cases(); palette = json.loads(PALETTE.read_text())[0]['palette']; jobs = []
    for c in cases:
        start = time.perf_counter(); measured = extract(c['grid'], 'a'); c['program_seconds'] = time.perf_counter()-start
        assert not measured.get('extraction_limited')
        c['program'] = measured; c['program_exact_matches'] = [g['id'] for g in c['gold'] if any(support(o) == {tuple(p) for p in g['support']} for o in measured['instances'])]
        im = render(c['grid'], palette); im.save(out/(c['name']+'.png'))
        for arm in ARMS:
            payload = dict(model='qwen3-vl-4b-instruct', temperature=0, seed=929201, cache_prompt=False,
                           max_tokens=2048, stream=False, response_format={'type':'json_object'},
                           messages=[dict(role='system', content=SYSTEM), dict(role='user', content=[encode(im),
                                     dict(type='text', text=PROMPT if arm == 'concepts' else BOX_PROMPT)])])
            for repeat in range(2):
                jobs.append(dict(id=len(jobs), case=c['name'], arm=arm, repeat=repeat, payload=payload,
                                 payload_digest=digest(payload)))
    save(out/'cases.json', cases); save(out/'jobs.json', jobs)
    paths = [Path(__file__), Path('agent/cognition/geometry.py'), Path('scripts/benchmark_target_binding.py'), Path('scripts/benchmark_recognition.py')]
    for path in paths: shutil.copyfile(path, out/'sources'/path.name)
    save(out/'plan.json', dict(created=datetime.now(timezone.utc).isoformat(), cases=len(cases), requests=len(jobs),
         cases_digest=digest(cases), jobs_digest=digest(jobs), image_hashes={p.name:sha(p) for p in out.glob('*.png')},
         source_hashes={str(p):sha(p) for p in paths}, inputs={str(p):sha(p) for p in (SOURCE, PALETTE)},
         rubric=RUBRIC, primary='First repeat concepts, 20 synthetic scenes / 80 objects; all whole-object proposals, without target hints.',
         timing='Cold prompt cache, local HTTP wall time; program extraction on host. Different hardware workloads; not equal compute.',
         design='Two formats, two exact greedy repeats, randomized request order; 3 additional real frames with 17 partially annotated targets. No model response used for gold. Templates developed after known failures; not held-out game evaluation.',
         matching='One-to-one maximum-cardinality matching at IoU >=0.5, >=0.75, or exact box. Inclusive integer coordinates, no clipping/snapping/candidate assistance. Concept matching manually reviewed separately.',
         limits='Synthetic object identity is defined by generated coherent shapes; touching or overlapping ambiguous objects not tested. Program grouping has no learned semantic labels. No temporal or functional role inference. Bounding-box overlap is not exact pixel segmentation.'))
    print('Prepared', len(cases), 'frames,', len(jobs), 'requests;', sum(len(c['gold']) for c in cases), 'annotated objects', flush=True)


def verify(out):
    plan = json.loads((out/'plan.json').read_text()); cases = json.loads((out/'cases.json').read_text()); jobs = json.loads((out/'jobs.json').read_text())
    assert digest(cases) == plan['cases_digest'] and digest(jobs) == plan['jobs_digest']
    for path, expected in plan['source_hashes'].items():
        assert sha(path) == expected and sha(out/'sources'/Path(path).name) == expected
    for path, expected in plan['inputs'].items(): assert sha(path) == expected
    for path, expected in plan['image_hashes'].items(): assert sha(out/path) == expected
    for j in jobs: assert digest(j['payload']) == j['payload_digest']
    return plan, cases, jobs


def parse_answer(answer, arm):
    data = json.loads(answer)
    if not isinstance(data, dict) or not isinstance(data.get('objects'), list): raise ValueError('missing objects array')
    result = data['objects']
    for obj in result:
        if not isinstance(obj, dict) or not isinstance(obj.get('description'), str) or not obj['description'].strip(): raise ValueError('missing description')
        if arm == 'concepts':
            if not isinstance(obj.get('location'), str) or not obj['location'].strip(): raise ValueError('missing location')
        else:
            b = obj.get('bbox')
            if not isinstance(b, list) or len(b) != 4 or any(type(v) != int for v in b): raise ValueError('invalid bbox type')
            if not (0 <= b[0] <= b[2] <= 63 and 0 <= b[1] <= b[3] <= 63): raise ValueError('invalid bbox range')
    return result


def run(out, base):
    assert urlparse(base).hostname in ('127.0.0.1', 'localhost')
    plan, cases, jobs = verify(out); assert not (out/'responses.jsonl').exists()
    save(out/'server-before.json', http(base+'/props')); random.Random(929213).shuffle(jobs)
    with (out/'responses.jsonl').open('x') as log:
        for i, j in enumerate(jobs):
            start = time.monotonic(); row = {k:v for k,v in j.items() if k != 'payload'}
            try:
                r = http(base+'/v1/chat/completions', j['payload']); row['response'] = r
                choice = r['choices'][0]; row['answer'] = choice['message']['content']
                if choice['finish_reason'] != 'stop': raise ValueError('response truncated')
                row['objects'] = parse_answer(row['answer'], j['arm']); row['valid'] = True
            except Exception as exc: row.update(valid=False, error=f'{type(exc).__name__}: {exc}')
            row['seconds'] = time.monotonic()-start; log.write(json.dumps(row)+'\n'); log.flush()
            print(i+1, '/', len(jobs), j['case'], j['arm'], row['valid'], round(row['seconds'],2), flush=True)
    save(out/'server-after.json', http(base+'/props')); save(out/'health-after.json', http(base+'/health'))


def iou(a, b):
    area = lambda z: (z[2]-z[0]+1)*(z[3]-z[1]+1)
    inter = max(0, min(a[2],b[2])-max(a[0],b[0])+1)*max(0,min(a[3],b[3])-max(a[1],b[1])+1)
    return inter/(area(a)+area(b)-inter)


def box_matches(pred, gold, threshold):
    edges = [[j for j,b in enumerate(gold) if (a == b if threshold == 'exact' else iou(a,b) >= threshold)] for a in pred]
    assigned = {}
    def visit(i, seen):
        for j in edges[i]:
            if j in seen: continue
            seen.add(j)
            if j not in assigned or visit(assigned[j], seen): assigned[j] = i; return True
        return False
    for i in range(len(pred)): visit(i, set())
    return [[i,j] for j,i in sorted(assigned.items())]


def stats(tp, predicted, gold):
    return dict(tp=tp, predicted=predicted, gold=gold, precision=tp/predicted if predicted else None,
                recall=tp/gold if gold else None, f1=2*tp/(predicted+gold) if predicted+gold else None)


def analyze(out):
    plan, cases, jobs = verify(out); rows = [json.loads(l) for l in (out/'responses.jsonl').read_text().splitlines()]
    assert Counter(r['id'] for r in rows) == Counter(j['id'] for j in jobs)
    for r in rows:
        assert all(r[k] == jobs[r['id']][k] for k in ('case','arm','repeat','payload_digest'))
    bycase = {c['name']:c for c in cases}; summary = dict(program={}, boxes=[], repeats={}, rows=rows)
    for subset, cc in [('synthetic',[c for c in cases if c['fully_annotated']]), ('real',[c for c in cases if not c['fully_annotated']])]:
        gold = sum(len(c['gold']) for c in cc); predicted = sum(len(c['program']['instances']) for c in cc)
        exact = sum(len(c['program_exact_matches']) for c in cc)
        summary['program'][subset] = dict(support=stats(exact,predicted,gold), median_seconds=statistics.median(c['program_seconds'] for c in cc),
            complete_scenes=sum(len(c['program_exact_matches']) == len(c['gold']) == len(c['program']['instances']) for c in cc),
            by_group={g:dict(gold=sum(len(c['gold']) for c in cc if c['group']==g), tp=sum(len(c['program_exact_matches']) for c in cc if c['group']==g),
                             predicted=sum(len(c['program']['instances']) for c in cc if c['group']==g)) for g in sorted({c['group'] for c in cc})})
        if subset == 'real':
            summary['program'][subset]['support']['precision'] = None; summary['program'][subset]['support']['f1'] = None
            summary['program'][subset]['complete_scenes'] = None
        for method in ('program', 'boxes'):
            for repeat in ([0] if method == 'program' else [0,1]):
                rr = [r for r in rows if r['arm']=='boxes' and r['repeat']==repeat and r['case'] in {c['name'] for c in cc}]
                for threshold in (0.5,0.75,'exact'):
                    records=[]
                    for c in cc:
                        r = next((r for r in rr if r['case']==c['name']),None)
                        pp = [o['bbox'] for o in c['program']['instances']] if method=='program' else [o['bbox'] for o in r.get('objects',[])] if r['valid'] else []
                        pairs=box_matches(pp,[g['bbox'] for g in c['gold']],threshold)
                        records.append(dict(case=c['name'],tp=len(pairs),predicted=len(pp),gold=len(c['gold']),pairs=pairs))
                    score=stats(sum(r['tp'] for r in records),sum(r['predicted'] for r in records),gold)
                    if subset=='real':score['precision']=None;score['f1']=None
                    summary['boxes'].append(dict(subset=subset,method=method,repeat=repeat,threshold=threshold,score=score,records=records))
    for arm in ARMS:
        rr = [r for r in rows if r['arm']==arm]
        summary['repeats'][arm] = dict(valid=sum(r['valid'] for r in rr),total=len(rr),
            different_answers=sum(next(r for r in rr if r['case']==c['name'] and r['repeat']==0).get('answer') != next(r for r in rr if r['case']==c['name'] and r['repeat']==1).get('answer') for c in cases),
            median_seconds=statistics.median(r['seconds'] for r in rr),
            median_output_tokens=statistics.median(r.get('response',{}).get('usage',{}).get('completion_tokens',0) for r in rr))
    save(out/'summary.json', summary)
    review=[]; order=rows.copy();random.Random(929221).shuffle(order)
    for r in order:
        review.append(dict(review_id=len(review),case=r['case'],valid=r['valid'],objects=r.get('objects',[]),answer=r.get('answer','')))
    save(out/'review-key.json',{str(i):r['id'] for i,r in enumerate(order)});save(out/'review-candidates.json',review)
    gallery=['<!doctype html><meta charset="utf-8"><title>Object proposals</title><style>body{font:16px sans-serif;margin:24px}img{width:320px;image-rendering:pixelated}pre{white-space:pre-wrap}article{border-top:1px solid #bbb;padding:16px}</style><h1>Program and Qwen object proposals</h1>']
    for c in cases:
        gallery.append('<article><h2>'+c['name']+'</h2><img src="'+c['name']+'.png"><p>Gold '+str(len(c['gold']))+'; program complete '+str(len(c['program_exact_matches']))+'</p><details><summary>Gold and program</summary><pre>'+html.escape(json.dumps(dict(gold=[{k:v for k,v in g.items() if k!='support'} for g in c['gold']],program=c['program']),ensure_ascii=False,indent=2))+'</pre></details>')
        for r in rows:
            if r['case']==c['name'] and r['repeat']==0:gallery.append('<h3>'+r['arm']+'</h3><pre>'+html.escape(r.get('answer',r.get('error','')) )+'</pre>')
        gallery.append('</article>')
    (out/'gallery.html').write_text('\n'.join(gallery))
    assert json.loads((out/'server-before.json').read_text()) == json.loads((out/'server-after.json').read_text())
    assert json.loads((out/'health-after.json').read_text())['status']=='ok'
    save(out/'verification.json',dict(passed=True,requests=len(rows),valid=sum(r['valid'] for r in rows),inputs_code_frozen=True,all_jobs_once=True,same_server=True))
    print(json.dumps(dict(program=summary['program'],repeats=summary['repeats']),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8080')
    a=p.parse_args()
    if a.command=='run':run(a.output,a.base)
    else:globals()[a.command](a.output)
