"""Repeated category-free batch proposals with outlines, mosaic or erasure.

Initial inference is shared across branches. Gold never controls requests,
masking or stopping. Final coverage and erased unseen targets are scored offline.
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_iterative_proposals as b

SOURCE=Path('outputs/iterative-proposals-20260928')
ARMS=('outline','mosaic','fill')
CAP=3
RESAMPLE=getattr(Image,'Resampling',Image)
DESCRIPTIONS={
 'outline':'Magenta rectangles mark already proposed objects. A different object inside a rectangle may still need its own proposal.',
 'mosaic':'Mosaic regions inside magenta rectangles cover already proposed objects. Ignore the mosaic regions and the rectangles.',
 'fill':'Regions painted with the background color inside magenta rectangles cover already proposed objects. Ignore these painted regions and the rectangles.'}
QUESTIONS={
 'outline':'Are there any objects without their own magenta rectangle?',
 'mosaic':'Are there any objects in the non-mosaic parts of the image?',
 'fill':'Are there any objects remaining outside the painted regions?'}
FORMAT=('Return only a JSON array with "bbox_2d": [x1,y1,x2,y2] and "label": a short appearance description. '
        'Use coordinates normalized to 0..1000 over the entire supplied image, x rightward and y downward. '
        'Return [] if there are no additional objects.')


def bounds(box,size=384):
    return (max(0,math.floor(box[0]*size/1000)),max(0,math.floor(box[1]*size/1000)),
            min(size,math.ceil(box[2]*size/1000)),min(size,math.ceil(box[3]*size/1000)))


def mask_image(boxes,size=384):
    mask=Image.new('L',(size,size),0);d=ImageDraw.Draw(mask)
    for box in boxes:
        l,t,r,bot=bounds(box,size)
        if r>l and bot>t:d.rectangle((l,t,r-1,bot-1),fill=255)
    return mask


def background(im): return Counter(im.getdata()).most_common(1)[0][0]


def render(im,boxes,arm):
    result=im.copy()
    if arm!='outline':
        mask=mask_image(boxes,im.width)
        replacement=(im.resize((16,16),RESAMPLE.BOX).resize(im.size,RESAMPLE.NEAREST)
                     if arm=='mosaic' else Image.new('RGB',im.size,background(im)))
        result=Image.composite(replacement,result,mask)
    # Same outline color/width for all branches; no cumulative raster degradation.
    return b.marked(result,boxes)


def payload(im,arm,mode):
    if mode=='gate':
        system='Answer the visual question. '+DESCRIPTIONS[arm]
        question=QUESTIONS[arm]+' Y = yes. N = no. Answer one letter.'
    else:
        system=('Use only visible appearance to identify distinct foreground objects. '
                'Keep visually coherent parts together even when colors differ. Separate copies are separate objects. '
                'Do not infer game roles. '+DESCRIPTIONS[arm])
        question='Locate ALL additional objects not yet proposed. Do not repeat already proposed objects. '+FORMAT
    p=dict(model='qwen3-vl-4b-instruct',temperature=0,seed=931101,cache_prompt=False,stream=False,
           max_tokens=1 if mode=='gate' else 2048,
           messages=[dict(role='system',content=system),dict(role='user',content=[b.encode(im),dict(type='text',text=question)])])
    if mode=='gate':p.update(grammar='root ::= "Y" | "N"',logprobs=True,top_logprobs=5,post_sampling_probs=False)
    return p


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir();(out/'inputs').mkdir()
    cases=json.loads((SOURCE/'cases.json').read_text())
    for name in ['cases.json','native-payloads.json',*[c['name']+'.png' for c in cases]]:shutil.copyfile(SOURCE/name,out/name)
    initial=[dict(case=c['name'],arm='initial') for c in cases];random.Random(931102).shuffle(initial)
    branches=[dict(case=c['name'],arm=arm) for c in cases for arm in ARMS];random.Random(931103).shuffle(branches)
    blocks=initial+branches
    for i,j in enumerate(blocks):j['id']=i
    b.save(out/'blocks.json',blocks)
    sources=[Path(__file__),Path('scripts/benchmark_iterative_proposals.py'),Path('scripts/benchmark_native_grounding.py'),
             Path('scripts/benchmark_target_binding.py'),Path('scripts/benchmark_recognition.py'),Path('scripts/benchmark_object_proposals.py')]
    for p in sources:shutil.copyfile(p,out/'sources'/p.name)
    b.save(out/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),arms=ARMS,cap=CAP,
      frozen_files={p.name:b.sha(p) for p in [out/'cases.json',out/'native-payloads.json',out/'blocks.json',*out.glob('*.png')]},
      source_hashes={str(p):b.sha(p) for p in sources},
      initial='23 fresh requests identical to prior native_batch, shared as initial boxes across all three arms. Each branch total time includes the same measured initial inference time plus its own actual wall time; initial not charged three times in experiment cost.',
      loop='After initial batch, transform original image with all retained boxes; Y/N max_tokens1; Y -> ALL additional boxes, accumulate raw valid boxes; N -> stop. At most3 batches including initial; final gate after batch3. Empty additional batch and invalid reply are distinct stops. No retries, no gold stopping, no NMS or box repair.',
      masks='384 square. Normalized rectangle floor left/top, ceil right/bottom, half-open clipped bounds, no padding. Union mask over all proposed boxes. Mosaic: full ORIGINAL image downsample16x16 BOX then upsample384 NEAREST, pasted only inside union mask. Fill: most frequent ORIGINAL RGB color. Magenta2px outlines on all branches. Every image rebuilt from original.',
      comparison='Same initial boxes, same later standard bbox_2d schema and generation settings. Later prompts differ only in annotation description and gate phrasing appropriate to treatment. One trial, known20 synthetic80objects +3 partial real17targets. No categories/counts supplied.',
      scoring='Unchanged previous primary:1 board pixel search margin,90% target pixel area,ROI area<=9x target bbox,1-to-1 matching. Also margin0/cap4/coverage80. Raw accumulative proposals retained so recall cannot drop. Track duplicate eligible boxes and mixed boxes.',
      masking_risk='Offline, for each newly transformed image, count prior-unretrieved gold objects with >=90% support area under mask (outline reports hypothetical fill). Report any exposure and initial exposure. This measures evidence loss, not proven causal loss of a detection. Gold never used to construct mask.',
      shadow='If FIRST gate is N, make ONE shadow additional-batch request on that exact transformed image after branch ends. It is diagnostic only: not added to branch proposals, not included in branch time or recall. Tests potential additions suppressed by immediate gate. Frozen before any inference.',
      timing='cache_prompt false; common initial uses exact old seed929201; later seed931101 temperature0 max2048 batch/1 gate. Sequential randomized blocks, no repetitions. Model config checked before/after. No runtime/game changes.'))
    print('Prepared',len(blocks),'blocks')


def run(out,base):
    u=urlparse(base);assert u.scheme=='http' and u.hostname in ('localhost','127.0.0.1')
    b.verify(out);assert not (out/'responses.jsonl').exists()
    b.save(out/'server-before.json',b.http(base+'/props'))
    audit={s:b.http(base+'/tokenize',dict(content=s,add_special=False)) for s in ('Y','N')}
    assert all(len(v['tokens'])==1 for v in audit.values());b.save(out/'token-audit.json',audit)
    blocks=json.loads((out/'blocks.json').read_text());controls=json.loads((out/'native-payloads.json').read_text());initial={};rid=0
    with (out/'requests.jsonl').open('x') as reqs,(out/'responses.jsonl').open('x') as log,(out/'results.jsonl').open('x') as results:
        def call(block,mode,im,boxes,p=None):
            nonlocal rid
            start=time.monotonic();p=payload(im,block['arm'],mode) if p is None else deepcopy(p)
            name='inputs/'+str(rid)+'.png';im.save(out/name)
            j=dict(id=rid,block_id=block['id'],case=block['case'],arm=block['arm'],mode=mode,prior_boxes=deepcopy(boxes),
                   image=name,image_sha256=b.sha(out/name),payload_digest=b.digest(p),payload=p)
            reqs.write(json.dumps(j)+'\n');reqs.flush();r={k:v for k,v in j.items() if k!='payload'};rid+=1
            try:
                response=b.http(base+'/v1/chat/completions',p);r['response']=response
                ch=response['choices'][0];r['answer']=ch['message']['content']
                if mode=='gate':
                    assert r['answer'] in ('Y','N') and response['usage']['completion_tokens']==1
                    r['decision']=r['answer']
                else:
                    if ch['finish_reason']!='stop':raise ValueError('truncated')
                    r['boxes']=[o['bbox'] for o in b.native_parse(r['answer'],'normalized_board')]
                r['valid']=True
            except Exception as exc:r.update(valid=False,error=f'{type(exc).__name__}: {exc}')
            r['seconds']=time.monotonic()-start;log.write(json.dumps(r)+'\n');log.flush();return r
        for block in blocks:
            start=time.monotonic();im=Image.open(out/(block['case']+'.png')).convert('RGB');arm=block['arm'];ids=[];trace=[];shadow=None
            if arm=='initial':
                r=call(block,'initial',im,[],controls[block['case']]);ids.append(r['id']);boxes=r.get('boxes',[]) if r['valid'] else []
                result=dict(**block,proposals=boxes,request_ids=ids,trace=[],stop='initial' if r['valid'] else 'invalid_initial',seconds=time.monotonic()-start)
                initial[block['case']]=result
            else:
                first=initial[block['case']];boxes=deepcopy(first['proposals']);reason='invalid_initial';batch_count=1
                if first['stop']=='initial':
                    for batch_count in range(1,CAP+1):
                        transformed=render(im,boxes,arm)
                        gate=call(block,'gate',transformed,boxes);ids.append(gate['id'])
                        trace.append(dict(batch=batch_count,boxes=deepcopy(boxes),gate_request=gate['id'],decision=gate.get('decision'),elapsed=first['seconds']+time.monotonic()-start))
                        if not gate['valid']:reason='invalid_gate';break
                        if gate['decision']=='N':reason='no';break
                        if batch_count==CAP:reason='cap';break
                        r=call(block,'additional',transformed,boxes);ids.append(r['id'])
                        if not r['valid']:reason='invalid_batch';break
                        if not r['boxes']:reason='empty_batch';break
                        boxes.extend(r['boxes'])
                seconds=first['seconds']+time.monotonic()-start
                if trace and len(trace)==1 and trace[0]['decision']=='N':
                    shadow=call(block,'shadow',render(im,boxes,arm),boxes)
                result=dict(**block,proposals=boxes,initial_request=first['request_ids'][0],request_ids=ids,trace=trace,stop=reason,
                            seconds=seconds,shadow_request=shadow['id'] if shadow else None)
            results.write(json.dumps(result)+'\n');results.flush()
            print(block['id']+1,'/',len(blocks),block['case'],arm,len(boxes),result['stop'],round(result['seconds'],2),flush=True)
    b.save(out/'server-after.json',b.http(base+'/props'));b.save(out/'health-after.json',b.http(base+'/health'))


def masked_fraction(boxes,gold):
    mask=mask_image(boxes);pixels=mask.load()
    return sum(pixels[xx,yy]>0 for x,y in gold['support'] for xx in range(x*6,(x+1)*6) for yy in range(y*6,(y+1)*6))/(36*len(gold['support']))


def analyze(out):
    b.verify(out)
    cases=json.loads((out/'cases.json').read_text());blocks=json.loads((out/'blocks.json').read_text())
    rows=[json.loads(l) for l in (out/'responses.jsonl').read_text().splitlines()]
    results=[json.loads(l) for l in (out/'results.jsonl').read_text().splitlines()]
    assert Counter(r['id'] for r in results)==Counter(j['id'] for j in blocks)
    actual_ids=[i for r in results for i in r['request_ids']]+[r['shadow_request'] for r in results if r.get('shadow_request') is not None]
    assert sorted(actual_ids)==list(range(len(rows)))
    summary=dict(arms=[],requests=len(rows),gates=sum(r['mode']=='gate' for r in rows))
    for arm in ('initial',)+ARMS:
        aa=dict(arm=arm)
        for subset,full in [('synthetic',True),('real',False)]:
            tasks=[]
            for c in cases:
                if c['fully_annotated']!=full:continue
                r=next(r for r in results if r['case']==c['name'] and r['arm']==arm)
                first=next(r for r in results if r['case']==c['name'] and r['arm']=='initial')
                score=b.score(r['proposals'],c['gold']);base=b.score(first['proposals'],c['gold']);erased=set();initial_erased=[];trace=[]
                for idx,t in enumerate(r['trace']):
                    sc=b.score(t['boxes'],c['gold'])
                    missing=[g for g in c['gold'] if g['id'] not in sc['matched']]
                    loss=[g['id'] for g in missing if masked_fraction(t['boxes'],g)>=.9]
                    erased.update(loss)
                    if idx==0:initial_erased=loss
                    trace.append(dict(batch=t['batch'],tp=sc['tp'],decision=t['decision'],hidden_unretrieved=loss))
                shadow=rows[r['shadow_request']] if r.get('shadow_request') is not None else None
                shadow_gain=(b.score(r['proposals']+shadow['boxes'],c['gold'])['tp']-score['tp']) if shadow and shadow['valid'] else 0
                tasks.append(dict(case=c['name'],group=c['group'],**score,initial_tp=base['tp'],gain=score['tp']-base['tp'],
                  raw_tp=b.score(r['proposals'],c['gold'],padding=0)['tp'],cap4_tp=b.score(r['proposals'],c['gold'],cap=4)['tp'],
                  coverage80_tp=b.score(r['proposals'],c['gold'],minimum=.8)['tp'],
                  stop=r['stop'],seconds=r['seconds'],calls=len(r['request_ids'])+(arm!='initial'),
                  initial_hidden=initial_erased,ever_hidden=sorted(erased),trace=trace,shadow=shadow is not None,shadow_gain=shadow_gain,
                  tokens=sum(rows[i].get('response',{}).get('usage',{}).get('completion_tokens',0) for i in r['request_ids']+([r['initial_request']] if arm!='initial' else []))))
            totals={k:sum(t[k] for t in tasks) for k in ('tp','gold','predicted','initial_tp','gain','raw_tp','cap4_tp','coverage80_tp','any_coverage','mixed_boxes','duplicate_eligible','no_eligible_target','calls','tokens','shadow_gain')}
            totals.update(recall=totals['tp']/totals['gold'],median_seconds=statistics.median(t['seconds'] for t in tasks),
              max_seconds=max(t['seconds'] for t in tasks),all_retrieved=sum(t['tp']==t['gold'] for t in tasks) if full else None,
              stops=dict(Counter(t['stop'] for t in tasks)),no_with_missing=sum(t['stop']=='no' and t['tp']<t['gold'] for t in tasks),
              no_with_uncovered=sum(t['stop']=='no' and t['any_coverage']<t['gold'] for t in tasks),
              initial_hidden=sum(len(t['initial_hidden']) for t in tasks),ever_hidden=sum(len(t['ever_hidden']) for t in tasks),
              shadow_calls=sum(t['shadow'] for t in tasks),tasks=tasks)
            aa[subset]=totals
        summary['arms'].append(aa)
    b.save(out/'summary.json',summary)
    gallery=['<!doctype html><meta charset="utf-8"><title>Repeated batch proposals</title><style>body{font:16px sans-serif;margin:20px}.row{display:flex;gap:16px;flex-wrap:wrap}.card{width:384px}img{width:384px;image-rendering:pixelated}pre{white-space:pre-wrap}article{border-top:1px solid gray}</style><h1>Repeated batch proposals</h1><p>Final boxes over ORIGINAL images; details show actual transformed inputs. No gold is drawn.</p>']
    for c in cases:
        gallery.append('<article><h2>'+c['name']+'</h2><div class="row">');im=Image.open(out/(c['name']+'.png')).convert('RGB')
        for arm in ('initial',)+ARMS:
            r=next(r for r in results if r['case']==c['name'] and r['arm']==arm);name=c['name']+'-'+arm+'-final.png';b.marked(im,r['proposals']).save(out/name)
            gallery.append('<div class="card"><h3>'+arm+'</h3><img src="'+name+'"><p>'+r['stop']+' | '+str(round(r['seconds'],3))+'s | '+str(b.score(r['proposals'],c['gold'])['tp'])+'/'+str(len(c['gold']))+'</p><details><summary>Actual inputs and outputs</summary>')
            for rid in r['request_ids']+([r['shadow_request']] if r.get('shadow_request') is not None else []):
                row=rows[rid];gallery.append('<h4>'+row['mode']+'</h4><img src="'+row['image']+'"><pre>'+html.escape(row.get('answer',row.get('error','')))+'</pre>')
            gallery.append('</details></div>')
        gallery.append('</div></article>')
    (out/'gallery.html').write_text('\n'.join(gallery))
    assert json.loads((out/'server-before.json').read_text())==json.loads((out/'server-after.json').read_text())
    assert json.loads((out/'health-after.json').read_text())['status']=='ok'
    b.save(out/'verification.json',dict(passed=True,requests=len(rows),valid=sum(r['valid'] for r in rows),blocks=len(results),same_server=True))
    for a in summary['arms']:print(a['arm'],[(s,a[s]['tp'],a[s]['gold'],round(a[s]['median_seconds'],3),a[s]['stops'],a[s]['ever_hidden']) for s in ('synthetic','real')])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8080');a=p.parse_args()
    if a.command=='run':run(a.output,a.base)
    else:globals()[a.command](a.output)
