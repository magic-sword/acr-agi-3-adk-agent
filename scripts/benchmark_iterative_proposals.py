"""Category-free proposal recall: batch versus marked-image iterative search.

Frozen before inference. Gold is used only in scoring and a separately labelled
oracle-overlay gate diagnostic, never in normal proposal search or stopping.
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
from scripts.benchmark_native_grounding import edges, iou, parse as native_parse
from scripts.benchmark_target_binding import save, digest, sha, http
from scripts.benchmark_recognition import encode

SOURCE = Path('outputs/native-grounding-20260928')
ARMS = ('native_batch', 'compact_batch', 'iterative')
CAP = 12
SYSTEM = ('Use only visible appearance to identify distinct foreground objects in this image. '
          'Keep visually coherent parts of one object together even when their colors differ. '
          'Separate copies are separate objects. Do not infer game roles. '
          'Magenta outline rectangles, if present, are annotations marking already proposed objects; '
          'they are not objects. A different object inside an existing rectangle may still need its own proposal. '
          'Do not propose the same object again. No category or object count is provided.')
COORDS = ('Use [x1,y1,x2,y2], normalized to 0..1000 over the entire image, '
          'x rightward, y downward. Enclose the whole object with a reasonably local rectangle; '
          'pixel-perfect edges are unnecessary. No labels or explanations.')
BATCH = 'Propose every distinct foreground object. Return only a JSON array of coordinate arrays, or [] if none. '+COORDS
SINGLE = 'Propose ONE foreground object that has not yet been individually marked. Return only one JSON coordinate array, or [] if none. '+COORDS
GATE = ('Is there any distinct foreground object that has not yet been individually marked by a magenta rectangle? '
        'A different object inside an existing rectangle still counts. Answer Y for yes or N for no. Output exactly one letter.')


def parse_boxes(answer, single=False):
    text = answer.strip()
    if text.startswith('```') and text.endswith('```'):
        lines = text.splitlines()
        if lines[0] not in ('```', '```json'): raise ValueError('unexpected fence')
        text = '\n'.join(lines[1:-1])
    data = json.loads(text)
    if not isinstance(data, list): raise ValueError('expected array')
    boxes = [data] if single and data else data
    for b in boxes:
        if not isinstance(b, list) or len(b) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in b):
            raise ValueError('invalid coordinates')
        if not (0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000):
            raise ValueError('invalid normalized range')
    return boxes


def marked(image, boxes):
    result = image.copy(); draw = ImageDraw.Draw(result)
    for b in boxes:
        draw.rectangle(tuple(v*image.width/1000 for v in b), outline='#ff00ff', width=2)
    return result


def payload(image, mode):
    p = dict(model='qwen3-vl-4b-instruct', temperature=0, seed=930101, stream=False,
             cache_prompt=False, max_tokens=1 if mode=='gate' else 192 if mode=='single' else 1024,
             messages=[dict(role='system', content=SYSTEM),
                       dict(role='user', content=[encode(image), dict(type='text', text={'gate':GATE,'single':SINGLE,'batch':BATCH}[mode])])])
    if mode == 'gate':
        p.update(grammar='root ::= "Y" | "N"', logprobs=True, top_logprobs=5, post_sampling_probs=False)
    return p


def roi(box, padding=1):
    b = [v*.064 for v in box]
    return [max(0,b[0]-padding),max(0,b[1]-padding),min(64,b[2]+padding),min(64,b[3]+padding)]


def coverage(box, gold):
    # Continuous overlap with each annotated unit pixel; no center rounding.
    return sum(max(0,min(box[2],x+1)-max(box[0],x))*max(0,min(box[3],y+1)-max(box[1],y))
               for x,y in gold['support']) / len(gold['support'])


def area(b): return (b[2]-b[0])*(b[3]-b[1])


def assign(adj):
    assigned = {}
    def visit(i, seen):
        for j in adj[i]:
            if j in seen: continue
            seen.add(j)
            if j not in assigned or visit(assigned[j], seen):
                assigned[j] = i
                return True
        return False
    for i in range(len(adj)): visit(i, set())
    return [[i,j] for j,i in sorted(assigned.items())]


def score(boxes, gold, padding=1, cap=9, minimum=.9):
    bb = [roi(b,padding) for b in boxes]
    cov = [[coverage(b,g) for g in gold] for b in bb]
    adj = [[j for j,g in enumerate(gold) if cov[i][j] >= minimum-1e-9
            and area(b) <= cap*area(edges(g['bbox']))+1e-9] for i,b in enumerate(bb)]
    pairs = assign(adj)
    return dict(tp=len(pairs), gold=len(gold), predicted=len(boxes), matched=[gold[j]['id'] for i,j in pairs],
                any_coverage=sum(any(row[j]>=minimum-1e-9 for row in cov) for j in range(len(gold))),
                mixed_boxes=sum(sum(v>=minimum-1e-9 for v in row)>=2 for row in cov),
                no_eligible_target=sum(not a for a in adj),
                duplicate_eligible=max(0,sum(bool(a) for a in adj)-len(pairs)),
                matched_area_ratios=[area(bb[i])/area(edges(gold[j]['bbox'])) for i,j in pairs])


def prepare(out):
    out.mkdir(parents=True, exist_ok=False); (out/'sources').mkdir(); (out/'inputs').mkdir()
    cases = json.loads((SOURCE/'corrected-cases.json').read_text())
    save(out/'cases.json', cases)
    for c in cases: shutil.copyfile(SOURCE/(c['name']+'-board.png'),out/(c['name']+'.png'))
    jobs = json.loads((SOURCE/'jobs.json').read_text())
    controls = {j['case']:j['payload'] for j in jobs if j['arm']=='normalized_board'}
    save(out/'native-payloads.json', controls)
    blocks = [dict(case=c['name'],arm=a) for c in cases for a in ARMS]
    # Independent gate diagnostic uses gold rectangles only, never feeds search.
    for i,c in enumerate(c for c in cases if c['fully_annotated']):
        for variant in ('complete','missing_one'):
            boxes = [[v/64*1000 for v in edges(g['bbox'])] for j,g in enumerate(c['gold'])
                     if variant=='complete' or j!=i%len(c['gold'])]
            blocks.append(dict(case=c['name'],arm='gate_diagnostic',variant=variant,boxes=boxes,
                               expected='N' if variant=='complete' else 'Y'))
    random.Random(930102).shuffle(blocks)
    for i,b in enumerate(blocks): b['id']=i
    save(out/'blocks.json',blocks)
    sources=[Path(__file__),Path('scripts/benchmark_native_grounding.py'),Path('scripts/benchmark_target_binding.py'),
             Path('scripts/benchmark_recognition.py'),Path('scripts/benchmark_object_proposals.py')]
    for p in sources: shutil.copyfile(p,out/'sources'/p.name)
    save(out/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),arms=ARMS,blocks=len(blocks),cap=CAP,
        frozen_files={p.name:sha(p) for p in [out/'cases.json',out/'native-payloads.json',out/'blocks.json',*out.glob('*.png')]},
        source_hashes={str(p):sha(p) for p in sources},
        primary='One request/block run per image/arm, 20 synthetic boards/80 objects, 3 real partial boards/17 targets. Candidate retrieval: >=90% annotated pixel area inside ROI, ROI area <=9x gold bbox area, one-to-one assignment. ROI expands each returned box by 1 board pixel and clips to board. This padding is a fixed downstream search allowance, not a repair inferred from gold.',
        secondary='Raw boxes without padding; area cap 4x; coverage 80%; unrestricted per-object coverage by any single candidate (not a success criterion); mixed-object boxes; duplicates; recall by iteration; total wall time including gates, drawing and image serialization. Strict IoU is not primary.',
        search='First produce one box without category. Draw cumulative magenta outlines (2px on384px image). Ask Y/N with max_tokens1. Y->one more box. N->stop. Up to12 proposals, with final gate; if Y at12, stop as capped, not complete. Empty proposal or invalid reply has a distinct stop reason. Keep duplicate boxes, no ground-truth stopping/deduplication, no chat history, no retries.',
        comparison='Native batch exact previous request, max2048. Compact batch and iterative share system/coordinate rule, max1024 batch/max192 single, no box grammar. Gate Y/N grammar only. Cache disabled all arms. One image per request. No token2 bbox encoding attempted.',
        diagnostic='40 separate oracle-overlay queries: each fully annotated synthetic image with every target marked or one target omitted. No counts/categories supplied. Never used to change online stopping or outputs.',
        limits='Known synthetic scenes; real labels partial, thus no full real recall/precision guarantee. Local Q4_K_M4B, single repeat. Stop errors based on candidate retrieval are not proof of semantic blindness. Overlay may obscure pixels. No runtime/game changes.'))
    print('Prepared',len(blocks),'blocks',flush=True)


def verify(out):
    plan=json.loads((out/'plan.json').read_text())
    for name,h in plan['frozen_files'].items(): assert sha(out/name)==h,name
    for path,h in plan['source_hashes'].items(): assert sha(path)==h and sha(out/'sources'/Path(path).name)==h,path
    return plan


def run(out,base):
    u=urlparse(base); assert u.scheme=='http' and u.hostname in ('localhost','127.0.0.1')
    verify(out); assert not (out/'responses.jsonl').exists()
    save(out/'server-before.json',http(base+'/props'))
    audit={s:http(base+'/tokenize',dict(content=s,add_special=False)) for s in ('Y','N')}
    assert all(len(v['tokens'])==1 for v in audit.values());save(out/'token-audit.json',audit)
    blocks=json.loads((out/'blocks.json').read_text());controls=json.loads((out/'native-payloads.json').read_text())
    with (out/'responses.jsonl').open('x') as log, (out/'requests.jsonl').open('x') as requests, (out/'results.jsonl').open('x') as results:
        request_id=0
        def call(block,mode,im,p=None):
            nonlocal request_id
            start=time.monotonic(); p=payload(im,mode) if p is None else deepcopy(p)
            image_name='inputs/'+str(request_id)+'.png';im.save(out/image_name)
            j=dict(id=request_id,block_id=block['id'],case=block['case'],arm=block['arm'],mode=mode,
                   image=image_name,image_sha256=sha(out/image_name),payload_digest=digest(p),payload=p)
            requests.write(json.dumps(j)+'\n');requests.flush()
            r={k:v for k,v in j.items() if k!='payload'};request_id+=1
            try:
                response=http(base+'/v1/chat/completions',p);r['response']=response
                choice=response['choices'][0];answer=choice['message']['content'];r['answer']=answer
                if mode=='gate':
                    assert answer in ('Y','N') and response['usage']['completion_tokens']==1
                    r['decision']=answer
                else:
                    if choice['finish_reason']!='stop': raise ValueError('truncated')
                    r['boxes']=[o['bbox'] for o in native_parse(answer,'normalized_board')] if mode=='native' else parse_boxes(answer,mode=='single')
                r['valid']=True
            except Exception as exc: r.update(valid=False,error=f'{type(exc).__name__}: {exc}')
            r['seconds']=time.monotonic()-start;log.write(json.dumps(r)+'\n');log.flush()
            return r
        for block in blocks:
            start=time.monotonic();im=Image.open(out/(block['case']+'.png')).convert('RGB');boxes=[];ids=[];trace=[]
            arm=block['arm'];reason='batch'
            if arm=='gate_diagnostic':
                r=call(block,'gate',marked(im,block['boxes']));ids.append(r['id']);reason='diagnostic'
            elif arm in ('native_batch','compact_batch'):
                r=call(block,'native' if arm=='native_batch' else 'batch',im,controls[block['case']] if arm=='native_batch' else None)
                ids.append(r['id']);boxes=r.get('boxes',[]) if r['valid'] else [];reason='batch' if r['valid'] else 'invalid'
            else:
                for step in range(CAP):
                    r=call(block,'single',marked(im,boxes));ids.append(r['id'])
                    if not r['valid']: reason='invalid_box';break
                    if not r['boxes']: reason='empty_proposal';break
                    boxes.extend(r['boxes'])
                    gate=call(block,'gate',marked(im,boxes));ids.append(gate['id'])
                    trace.append(dict(step=step+1,boxes=deepcopy(boxes),box_request=r['id'],gate_request=gate['id'],decision=gate.get('decision'),seconds=time.monotonic()-start))
                    if not gate['valid']: reason='invalid_gate';break
                    if gate['decision']=='N': reason='no';break
                else: reason='cap'
            result=dict(**block,proposals=boxes,request_ids=ids,trace=trace,stop=reason,seconds=time.monotonic()-start)
            if arm=='gate_diagnostic':result.update(decision=r.get('decision'),valid=r['valid'])
            results.write(json.dumps(result)+'\n');results.flush()
            print(block['id']+1,'/',len(blocks),block['case'],arm,len(boxes),reason,round(result['seconds'],2),flush=True)
    save(out/'server-after.json',http(base+'/props'));save(out/'health-after.json',http(base+'/health'))


def analyze(out):
    verify(out)
    cases=json.loads((out/'cases.json').read_text());blocks=json.loads((out/'blocks.json').read_text())
    results=[json.loads(l) for l in (out/'results.jsonl').read_text().splitlines()]
    rows=[json.loads(l) for l in (out/'responses.jsonl').read_text().splitlines()]
    assert Counter(b['id'] for b in blocks)==Counter(r['id'] for r in results)
    assert sorted(r['id'] for r in rows)==sorted(i for r in results for i in r['request_ids'])
    summary=dict(arms=[],diagnostic={})
    for arm in ARMS:
        aa=dict(arm=arm)
        for subset,full in [('synthetic',True),('real',False)]:
            tasks=[]
            for c in cases:
                if c['fully_annotated']!=full:continue
                r=next(r for r in results if r['arm']==arm and r['case']==c['name'])
                s=score(r['proposals'],c['gold']);raw=score(r['proposals'],c['gold'],padding=0)
                curve=[score(t['boxes'],c['gold'])['tp'] for t in r['trace']]
                checks=[dict(step=t['step'],decision=t['decision'],tp=curve[i],gold=len(c['gold']),
                             any_coverage=score(t['boxes'],c['gold'])['any_coverage']) for i,t in enumerate(r['trace'])]
                tasks.append(dict(case=c['name'],group=c['group'],**s,raw_tp=raw['tp'],cap4_tp=score(r['proposals'],c['gold'],cap=4)['tp'],
                                  coverage80_tp=score(r['proposals'],c['gold'],minimum=.8)['tp'],
                                  seconds=r['seconds'],calls=len(r['request_ids']),stop=r['stop'],checks=checks,
                                  output_tokens=sum(rows[i].get('response',{}).get('usage',{}).get('completion_tokens',0) for i in r['request_ids'])))
            totals={key:sum(t[key] for t in tasks) for key in ('tp','gold','predicted','raw_tp','cap4_tp','coverage80_tp','any_coverage','mixed_boxes','duplicate_eligible','no_eligible_target','calls','output_tokens')}
            totals.update(recall=totals['tp']/totals['gold'],all_retrieved=sum(t['tp']==t['gold'] for t in tasks) if full else None,
                          median_seconds=statistics.median(t['seconds'] for t in tasks),max_seconds=max(t['seconds'] for t in tasks),
                          median_output_tokens=statistics.median(t['output_tokens'] for t in tasks),
                          stops=dict(Counter(t['stop'] for t in tasks)),
                          no_with_missing=sum(t['stop']=='no' and t['tp']<t['gold'] for t in tasks),
                          no_with_uncovered=sum(t['stop']=='no' and t['any_coverage']<t['gold'] for t in tasks),
                          by_group={g:dict(tp=sum(t['tp'] for t in tasks if t['group']==g),gold=sum(t['gold'] for t in tasks if t['group']==g)) for g in sorted({t['group'] for t in tasks})},tasks=tasks)
            aa[subset]=totals
        summary['arms'].append(aa)
    for variant in ('complete','missing_one'):
        rr=[r for r in results if r['arm']=='gate_diagnostic' and r['variant']==variant]
        summary['diagnostic'][variant]=dict(total=len(rr),correct=sum(r['decision']==r['expected'] for r in rr),
                                           decisions=dict(Counter(r['decision'] for r in rr)),tasks=rr)
    # Program boxes are unchanged; evaluate using the same relaxed ROI metric.
    summary['program']={}
    for subset,full in [('synthetic',True),('real',False)]:
        ss=[score([[v/64*1000 for v in edges(o['bbox'])] for o in c['program']['instances']],c['gold']) for c in cases if c['fully_annotated']==full]
        summary['program'][subset]={k:sum(s[k] for s in ss) for k in ('tp','gold','predicted','any_coverage','mixed_boxes')}
    save(out/'summary.json',summary)
    gallery=['<!doctype html><meta charset="utf-8"><title>Iterative proposal comparison</title><style>body{font:16px sans-serif}article{border-top:1px solid #aaa}.row{display:flex;flex-wrap:wrap;gap:16px}img{width:384px;image-rendering:pixelated}pre{white-space:pre-wrap;max-width:550px}</style><h1>Category-free proposals</h1><p>Magenta: returned boxes. No gold is drawn. Primary scoring uses a separate fixed 1-board-pixel search margin.</p>']
    for c in cases:
        gallery.append('<article><h2>'+c['name']+'</h2><div class="row">')
        im=Image.open(out/(c['name']+'.png')).convert('RGB')
        for arm in ARMS:
            r=next(r for r in results if r['arm']==arm and r['case']==c['name']);name=c['name']+'-'+arm+'.png'
            marked(im,r['proposals']).save(out/name)
            gallery.append('<div><h3>'+arm+'</h3><img src="'+name+'"><pre>'+html.escape(json.dumps(dict(score=score(r['proposals'],c['gold']),seconds=r['seconds'],stop=r['stop']),indent=2))+'</pre>')
            if arm=='iterative':
                gallery.append('<details><summary>Iteration inputs / answers</summary>')
                for rid in r['request_ids']:
                    row=rows[rid];gallery.append('<p>'+row['mode']+'</p><img src="'+row['image']+'"><pre>'+html.escape(row.get('answer',row.get('error','')))+'</pre>')
                gallery.append('</details>')
            gallery.append('</div>')
        gallery.append('</div></article>')
    (out/'gallery.html').write_text('\n'.join(gallery))
    assert json.loads((out/'server-before.json').read_text())==json.loads((out/'server-after.json').read_text())
    assert json.loads((out/'health-after.json').read_text())['status']=='ok'
    save(out/'verification.json',dict(passed=True,blocks=len(results),requests=len(rows),valid=sum(r['valid'] for r in rows),same_server=True,frozen_inputs=True))
    for a in summary['arms']:print(a['arm'],[(s,a[s]['tp'],a[s]['gold'],round(a[s]['median_seconds'],3),a[s]['stops']) for s in ('synthetic','real')])
    print('Diagnostic',[(k,v['correct'],v['total']) for k,v in summary['diagnostic'].items()])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8080');a=p.parse_args()
    if a.command=='run':run(a.output,a.base)
    else:globals()[a.command](a.output)
