"""Frozen comparison of legacy board coordinates and native Qwen grounding.

Only local inference is contacted. Continuous image-edge coordinates are mapped
back to board space before scoring; no clipping, snapping, or box repairs.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
from urllib.parse import urlparse

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_target_binding import save, sha, digest, http
from scripts.benchmark_recognition import encode
from scripts.benchmark_object_proposals import parse_answer as parse_legacy

SOURCE = Path('outputs/object-proposals-20260928')
ARMS = ('legacy', 'normalized_framed', 'normalized_board', 'normalized_cross')
FORMAT = ('Report a JSON array. Each item has "bbox_2d": [x1, y1, x2, y2] and '
          '"label": a short appearance description. Give tight boxes using coordinates normalized '
          'from 0 to 1000 relative to the ENTIRE supplied image, x rightward and y downward. '
          'Return [] when there are no matching objects.')
FREE_PROMPT = ('Locate every distinct visible foreground object, keeping the parts of each whole object '
               'together. List separate copies separately. '+FORMAT)
CROSS_PROMPT = ('Locate every plus-shaped cross in the image. Crosses may contain multiple colors. '
                'Return only plus-shaped crosses, not other kinds of objects. '+FORMAT)
CROSS_SYSTEM = ('Use only the supplied image to locate the requested kind of foreground object. '
                'Keep its visually coherent parts together even when their colors differ. '
                'List separate copies separately. Describe appearance, not game roles. Return only JSON.')


def edges(box):
    """Original integer inclusive pixel support -> continuous outer edges."""
    return [box[0], box[1], box[2]+1, box[3]+1]


def to_board(box, arm):
    if arm == 'legacy': return edges(box)
    if arm == 'normalized_framed':
        return [(box[0]*428/1000-32)/6, (box[1]*452/1000-44)/6,
                (box[2]*428/1000-32)/6, (box[3]*452/1000-44)/6]
    return [v*64/1000 for v in box]


def parse(answer, arm):
    if arm == 'legacy':
        return [dict(label=o['description'], bbox=o['bbox'], board_box=to_board(o['bbox'], arm))
                for o in parse_legacy(answer, 'boxes')]
    text = answer.strip()
    if text.startswith('```') and text.endswith('```'):
        lines = text.splitlines()
        if lines[0].strip() not in ('```', '```json'): raise ValueError('unexpected fence')
        text = '\n'.join(lines[1:-1])
    data = json.loads(text)
    if not isinstance(data, list): raise ValueError('expected JSON array')
    result=[]
    for o in data:
        if not isinstance(o, dict) or not isinstance(o.get('label'), str) or not o['label'].strip(): raise ValueError('missing label')
        b=o.get('bbox_2d')
        if not isinstance(b,list) or len(b)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in b): raise ValueError('invalid box type')
        if not (0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000): raise ValueError('invalid normalized range')
        result.append(dict(label=o['label'],bbox=b,board_box=to_board(b,arm)))
    return result


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+bb-inter)


def match(pred,gold,threshold):
    adj=[[j for j,b in enumerate(gold) if iou(a,b)>=threshold] for a in pred];assigned={}
    def visit(i,seen):
        for j in adj[i]:
            if j in seen:continue
            seen.add(j)
            if j not in assigned or visit(assigned[j],seen):assigned[j]=i;return True
        return False
    for i in range(len(pred)):visit(i,set())
    return [[i,j] for j,i in sorted(assigned.items())]


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    cases=json.loads((SOURCE/'cases.json').read_text());old_jobs=json.loads((SOURCE/'jobs.json').read_text());jobs=[]
    for c in cases:
        c['cross_gold_ids']=[g['id'] for g in c['gold'] if 'cross' in g['kind'].lower()]
        framed=Image.open(SOURCE/(c['name']+'.png')).convert('RGB')
        assert framed.size==(428,452)
        board=framed.crop((32,44,416,428));assert board.size==(384,384)
        framed.save(out/(c['name']+'-framed.png'));board.save(out/(c['name']+'-board.png'))
        old=next(j for j in old_jobs if j['case']==c['name'] and j['arm']=='boxes' and j['repeat']==0)
        for arm in ARMS:
            p=deepcopy(old['payload'])
            if arm != 'legacy':
                p.pop('response_format',None)
                p['messages'][1]['content']=[encode(framed if arm=='normalized_framed' else board),
                    dict(type='text',text=CROSS_PROMPT if arm=='normalized_cross' else FREE_PROMPT)]
                if arm=='normalized_cross':p['messages'][0]['content']=CROSS_SYSTEM
            jobs.append(dict(id=len(jobs),case=c['name'],arm=arm,payload=p,payload_digest=digest(p)))
        assert jobs[-4]['payload_digest']==old['payload_digest']
    assert sum(len(c['cross_gold_ids']) for c in cases if c['fully_annotated'])==36
    save(out/'cases.json',cases);save(out/'jobs.json',jobs)
    paths=[Path(__file__),Path('scripts/benchmark_object_proposals.py'),Path('scripts/benchmark_target_binding.py'),Path('scripts/benchmark_recognition.py')]
    for p in paths:shutil.copyfile(p,out/'sources'/p.name)
    save(out/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),cases=len(cases),requests=len(jobs),arms=ARMS,
        cases_digest=digest(cases),jobs_digest=digest(jobs),source_hashes={str(p):sha(p) for p in paths},
        input_hashes={str(p):sha(p) for p in [SOURCE/'cases.json',SOURCE/'jobs.json',SOURCE/'responses.jsonl']},
        image_hashes={p.name:sha(p) for p in out.glob('*.png')},
        primary='First and only request per case/arm. All-object detection on 20 fully annotated synthetic boards (80 objects); IoU >= 0.5 with one-to-one matching. Report recall, precision, all-object scene exactness, IoU 0.75, and best IoU per gold. Invalid/truncated replies produce no usable boxes and remain in denominators.',
        contrasts='legacy -> normalized_framed changes coordinates, schema, prompt wording and removes json_object grammar as one native-format bundle. normalized_framed -> normalized_board changes only image (same object scale, pixels, prompt and decoding); normalized_board -> normalized_cross changes task to one specified category. This is not an isolated test of coordinate scaling alone.',
        cross='Synthetic gold has 36 crosses across 12 positive boards; 8 fully annotated negative boards. Cross precision/negative control meaningful only for cross-only requests. For all-object arms/program report cross recall only. Real partial annotations include one ls20 cross; do not treat the other two real frames as exhaustively annotated cross negatives.',
        conversion='Gold and legacy inclusive integers become [left,top,right+1,bottom+1] continuous edges. Normalized boxes map linearly through actual image width/height, then subtract (32,44) and divide by 6 for framed images, or multiply by 64/1000 for bare boards. No coordinate rounding, clipping, padding, or snapping before IoU.',
        timing='Same warm local model; temperature 0, seed 929201, cache_prompt false, max_tokens 2048. 92 calls randomized with seed 929301. No repetitions or latency confidence intervals in this comparison. All controls rerun rather than timing historical results.',
        reference='https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb',
        limits='Known synthetic templates and previously seen game boards. Real annotations partial (17 objects): only recall, no precision/full-scene exactness. Box overlap is not pixel segmentation or proof of semantic identity; returned labels are not scored. Neither a reproduction of official RefCOCO/ODinW nor a FCOS speed comparison. Local quantized 4B, not official unquantized model. No game actions/runtime mutations.'))
    print('Prepared',len(jobs),'requests, 80 synthetic objects / 36 crosses, 17 real targets',flush=True)


def verify(out):
    plan=json.loads((out/'plan.json').read_text());cases=json.loads((out/'cases.json').read_text());jobs=json.loads((out/'jobs.json').read_text())
    assert digest(cases)==plan['cases_digest'] and digest(jobs)==plan['jobs_digest']
    for path,h in plan['source_hashes'].items():assert sha(path)==h and sha(out/'sources'/Path(path).name)==h
    for path,h in plan['input_hashes'].items():assert sha(path)==h
    for path,h in plan['image_hashes'].items():assert sha(out/path)==h
    for j in jobs:assert digest(j['payload'])==j['payload_digest']
    return plan,cases,jobs


def run(out,base):
    parsed=urlparse(base);assert parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','localhost')
    plan,cases,jobs=verify(out);assert not (out/'responses.jsonl').exists()
    save(out/'server-before.json',http(base+'/props'));order=jobs.copy();random.Random(929301).shuffle(order)
    save(out/'run-order.json',[j['id'] for j in order])
    with (out/'responses.jsonl').open('x') as f:
        for i,j in enumerate(order):
            start=time.monotonic();r={k:v for k,v in j.items() if k!='payload'}
            try:
                response=http(base+'/v1/chat/completions',j['payload']);r['response']=response
                choice=response['choices'][0];r['answer']=choice['message']['content']
                if choice['finish_reason']!='stop':raise ValueError('response truncated')
                r['objects']=parse(r['answer'],j['arm']);r['valid']=True
            except Exception as exc:r.update(valid=False,error=f'{type(exc).__name__}: {exc}')
            r['seconds']=time.monotonic()-start;f.write(json.dumps(r)+'\n');f.flush()
            print(i+1,'/',len(jobs),j['case'],j['arm'],r['valid'],round(r['seconds'],2),flush=True)
    save(out/'server-after.json',http(base+'/props'));save(out/'health-after.json',http(base+'/health'))


def evaluate(rows,cases,threshold,cross=False,precision_allowed=True):
    tasks=[]
    for c in cases:
        r=next(r for r in rows if r['case']==c['name'])
        pp=[o['board_box'] for o in r.get('objects',[])] if r['valid'] else []
        gg=[g for g in c['gold'] if not cross or g['id'] in c['cross_gold_ids']]
        boxes=[edges(g['bbox']) for g in gg];pairs=match(pp,boxes,threshold)
        tasks.append(dict(case=c['name'],group=c['group'],valid=r['valid'],tp=len(pairs),predicted=len(pp),gold=len(gg),
                          matched=[gg[j]['id'] for i,j in pairs],pairs=pairs,
                          best_ious=[max((iou(a,b) for a in pp),default=0) for b in boxes],
                          complete=bool(r['valid'] and len(pairs)==len(gg)==len(pp))))
    tp=sum(t['tp'] for t in tasks);n=sum(t['predicted'] for t in tasks);g=sum(t['gold'] for t in tasks)
    all_annotated=all(c['fully_annotated'] for c in cases);allow=precision_allowed and all_annotated
    return dict(threshold=threshold,cross=cross,tp=tp,predicted=n,gold=g,recall=tp/g if g else None,
        precision=tp/n if n and allow else None,f1=2*tp/(n+g) if allow and n+g else None,
        complete_scenes=sum(t['complete'] for t in tasks) if allow else None,
        mean_best_iou=statistics.mean(v for t in tasks for v in t['best_ious']) if g else None,
        by_group={group:dict(tp=sum(t['tp'] for t in tasks if t['group']==group),gold=sum(t['gold'] for t in tasks if t['group']==group),predicted=sum(t['predicted'] for t in tasks if t['group']==group)) for group in sorted({c['group'] for c in cases})},
        tasks=tasks)


def analyze(out):
    plan,cases,jobs=verify(out);rows=[json.loads(l) for l in (out/'responses.jsonl').read_text().splitlines()]
    assert Counter(r['id'] for r in rows)==Counter(j['id'] for j in jobs)
    for r in rows:assert all(r[k]==jobs[r['id']][k] for k in ('case','arm','payload_digest'))
    synthetic=[c for c in cases if c['fully_annotated']];real=[c for c in cases if not c['fully_annotated']]
    program=[dict(case=c['name'],valid=True,objects=[dict(board_box=edges(o['bbox'])) for o in c['program']['instances']]) for c in cases]
    result=dict(arms=[],program={})
    for subset,cc in [('synthetic',synthetic),('real',real)]:
        result['program'][subset]={str(t):evaluate(program,cc,t) for t in (.5,.75)}
        result['program'][subset]['cross']=evaluate(program,cc,.5,cross=True,precision_allowed=False)
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm];a=dict(arm=arm,valid=sum(r['valid'] for r in rr),total=len(rr),
            errors=dict(Counter(r['error'] for r in rr if not r['valid'])),
            timing={k:dict(median_seconds=statistics.median(r['seconds'] for r in rr if r['case'] in {c['name'] for c in cc}),
                           max_seconds=max(r['seconds'] for r in rr if r['case'] in {c['name'] for c in cc}),
                           median_output_tokens=statistics.median(r.get('response',{}).get('usage',{}).get('completion_tokens',0) for r in rr if r['case'] in {c['name'] for c in cc})) for k,cc in [('synthetic',synthetic),('real',real)]})
        for subset,cc in [('synthetic',synthetic),('real',real)]:
            a[subset]={str(t):evaluate(rr,cc,t,cross=arm=='normalized_cross') for t in (.5,.75)}
            a[subset]['cross_recall']=evaluate(rr,cc,.5,cross=True,precision_allowed=arm=='normalized_cross')
        if arm=='normalized_cross':
            negatives=[c for c in synthetic if not c['cross_gold_ids']]
            a['negative_control']=dict(total=len(negatives),correct_empty=sum(next(r for r in rr if r['case']==c['name'])['valid'] and not next(r for r in rr if r['case']==c['name'])['objects'] for c in negatives))
        result['arms'].append(a)
    old=[r for r in [json.loads(l) for l in (SOURCE/'responses.jsonl').read_text().splitlines()] if r['arm']=='boxes' and r['repeat']==0]
    result['legacy_historical_differences']=sum(r.get('answer')!=next(x for x in old if x['case']==r['case']).get('answer') for r in rows if r['arm']=='legacy')
    save(out/'summary.json',result)
    gallery=['<!doctype html><meta charset="utf-8"><title>Native grounding comparison</title><style>body{font:15px sans-serif;margin:20px}.row{display:flex;flex-wrap:wrap;gap:12px}.card{width:280px}img{width:280px;image-rendering:pixelated}pre{white-space:pre-wrap;max-height:240px;overflow:auto}article{border-top:1px solid #aaa;padding:16px}</style><h1>Native grounding comparison</h1><p>Green: frozen gold. Magenta: predictions, rendered for review only. Scoring uses continuous unrounded coordinates. Cross-only arm has a narrower task. Real gold is partial.</p>']
    for c in cases:
        gallery.append('<article><h2>'+c['name']+'</h2><div class="row">')
        board=Image.open(out/(c['name']+'-board.png')).convert('RGB')
        items=[('program',[edges(o['bbox']) for o in c['program']['instances']],None)]+[(arm,[o['board_box'] for o in next(r for r in rows if r['case']==c['name'] and r['arm']==arm).get('objects',[])],next(r for r in rows if r['case']==c['name'] and r['arm']==arm)) for arm in ARMS]
        for arm,boxes,r in items:
            im=board.copy();d=ImageDraw.Draw(im)
            for g in c['gold']:
                if arm=='normalized_cross' and g['id'] not in c['cross_gold_ids']:continue
                d.rectangle(tuple(v*6 for v in edges(g['bbox'])),outline='#4fff70',width=1)
            for box in boxes:d.rectangle(tuple(v*6 for v in box),outline='#ff33ff',width=2)
            name=c['name']+'-'+arm+'-overlay.png';im.save(out/name)
            gallery.append('<div class="card"><h3>'+arm+'</h3><img src="'+name+'">')
            if r:gallery.append('<p>'+str(round(r['seconds'],3))+'s | '+str(r['valid'])+'</p><details><summary>Output</summary><pre>'+html.escape(r.get('answer',r.get('error','')))+'</pre></details>')
            gallery.append('</div>')
        gallery.append('</div></article>')
    (out/'gallery.html').write_text('\n'.join(gallery))
    assert json.loads((out/'server-before.json').read_text())==json.loads((out/'server-after.json').read_text())
    assert json.loads((out/'health-after.json').read_text())['status']=='ok'
    save(out/'verification.json',dict(passed=True,requests=len(rows),valid=sum(r['valid'] for r in rows),all_jobs_once=True,frozen_inputs_and_sources=True,same_server=True))
    for a in result['arms']:
        print(a['arm'],'valid',a['valid'],'/23','synthetic',a['synthetic']['0.5']['tp'],'/',a['synthetic']['0.5']['gold'],'real',a['real']['0.5']['tp'],'/',a['real']['0.5']['gold'],'median',round(a['timing']['synthetic']['median_seconds'],3))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8080');a=p.parse_args()
    if a.command=='run':run(a.output,a.base)
    else:globals()[a.command](a.output)
