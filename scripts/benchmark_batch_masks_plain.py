"""Adaptive follow-up: remove annotation outlines, share actual initial batches."""
import argparse,json,random,shutil,sys,time
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_batch_masks as m
b=m.b
SOURCE=Path('outputs/batch-masks-20260928')
DESCRIPTIONS={
 'mosaic':'Mosaic regions cover already proposed objects. Ignore the mosaic regions.',
 'fill':'Regions painted with the background color cover already proposed objects. Ignore these painted regions.'}

def render(im,boxes,arm):
 mask=m.mask_image(boxes,im.width)
 replacement=(im.resize((16,16),m.RESAMPLE.BOX).resize(im.size,m.RESAMPLE.NEAREST) if arm=='mosaic' else Image.new('RGB',im.size,m.background(im)))
 return Image.composite(replacement,im,mask)

def prepare(out):
 out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir();(out/'inputs').mkdir()
 cases=json.loads((SOURCE/'cases.json').read_text())
 for name in ['cases.json',*[c['name']+'.png' for c in cases]]:shutil.copyfile(SOURCE/name,out/name)
 initial=[json.loads(l) for l in (SOURCE/'results.jsonl').read_text().splitlines() if json.loads(l)['arm']=='initial']
 assert len(initial)==len(cases)
 b.save(out/'shared-initial.json',initial)
 initial_responses=[json.loads(l) for l in (SOURCE/'responses.jsonl').read_text().splitlines() if json.loads(l)['arm']=='initial']
 b.save(out/'shared-initial-responses.json',initial_responses)
 blocks=[dict(case=c['name'],arm=a) for c in cases for a in ('mosaic','fill')];random.Random(931104).shuffle(blocks)
 for i,j in enumerate(blocks):j['id']=i
 b.save(out/'blocks.json',blocks)
 sources=[Path(__file__),Path('scripts/benchmark_batch_masks.py'),Path('scripts/benchmark_iterative_proposals.py'),Path('scripts/benchmark_native_grounding.py'),Path('scripts/benchmark_target_binding.py'),Path('scripts/benchmark_recognition.py'),Path('scripts/benchmark_object_proposals.py')]
 for p in sources:shutil.copyfile(p,out/'sources'/p.name)
 b.save(out/'plan.json',dict(frozen_files={p.name:b.sha(p) for p in [out/'cases.json',out/'blocks.json',out/'shared-initial.json',out/'shared-initial-responses.json',*out.glob('*.png')]},source_hashes={str(p):b.sha(p) for p in sources},
  reason='Adaptive follow-up after shadow batches in the outlined-mask study described magenta rectangles as objects. Frozen before any no-outline inference.',
  changes='Use identical mask union, mosaic strength/background color, batch schema and decoding. Remove magenta outlines from both mask arms and remove reference to rectangles from system text. Same gate question. Initial boxes/results/time reused exactly from primary, not re-inferred.',
  loop='At most3 total batches including shared initial. Gate every stage. Initial-N shadow batch retained only as diagnostic, excluded from online result/time. Same scoring as primary. One run each.',
  descriptions=DESCRIPTIONS))
 print('Prepared',len(blocks),'plain-mask blocks')

def run(out,base):
 u=urlparse(base);assert u.scheme=='http' and u.hostname in ('localhost','127.0.0.1')
 b.verify(out);assert not (out/'responses.jsonl').exists();b.save(out/'server-before.json',b.http(base+'/props'))
 blocks=json.loads((out/'blocks.json').read_text());initial={r['case']:r for r in json.loads((out/'shared-initial.json').read_text())};rid=0
 with (out/'requests.jsonl').open('x') as reqs,(out/'responses.jsonl').open('x') as log,(out/'results.jsonl').open('x') as results:
  def call(block,mode,im,boxes):
   nonlocal rid
   start=time.monotonic();p=m.payload(im,block['arm'],mode)
   p['messages'][0]['content']=p['messages'][0]['content'].replace(m.DESCRIPTIONS[block['arm']],DESCRIPTIONS[block['arm']])
   name='inputs/'+str(rid)+'.png';im.save(out/name)
   j=dict(id=rid,block_id=block['id'],case=block['case'],arm=block['arm'],mode=mode,prior_boxes=deepcopy(boxes),image=name,image_sha256=b.sha(out/name),payload_digest=b.digest(p),payload=p)
   reqs.write(json.dumps(j)+'\n');reqs.flush();r={k:v for k,v in j.items() if k!='payload'};rid+=1
   try:
    response=b.http(base+'/v1/chat/completions',p);r['response']=response;ch=response['choices'][0];r['answer']=ch['message']['content']
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
   start=time.monotonic();im=Image.open(out/(block['case']+'.png')).convert('RGB');first=initial[block['case']];boxes=deepcopy(first['proposals']);ids=[];trace=[];shadow=None;reason='invalid_initial'
   if first['stop']=='initial':
    for count in range(1,4):
     transformed=render(im,boxes,block['arm']);gate=call(block,'gate',transformed,boxes);ids.append(gate['id'])
     trace.append(dict(batch=count,boxes=deepcopy(boxes),gate_request=gate['id'],decision=gate.get('decision'),elapsed=first['seconds']+time.monotonic()-start))
     if not gate['valid']:reason='invalid_gate';break
     if gate['decision']=='N':reason='no';break
     if count==3:reason='cap';break
     r=call(block,'additional',transformed,boxes);ids.append(r['id'])
     if not r['valid']:reason='invalid_batch';break
     if not r['boxes']:reason='empty_batch';break
     boxes.extend(r['boxes'])
   seconds=first['seconds']+time.monotonic()-start
   if trace and len(trace)==1 and trace[0]['decision']=='N':shadow=call(block,'shadow',render(im,boxes,block['arm']),boxes)
   result=dict(**block,proposals=boxes,initial_request=first['request_ids'][0],request_ids=ids,trace=trace,stop=reason,seconds=seconds,shadow_request=shadow['id'] if shadow else None)
   results.write(json.dumps(result)+'\n');results.flush();print(block['id']+1,'/',len(blocks),block['case'],block['arm'],len(boxes),reason,round(seconds,2),flush=True)
 b.save(out/'server-after.json',b.http(base+'/props'));b.save(out/'health-after.json',b.http(base+'/health'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8080');a=p.parse_args()
 if a.command=='run':run(a.output,a.base)
 else:prepare(a.output)
