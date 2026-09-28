"""Predeclared follow-up after early all-N behavior; preserve the first experiment.

Compare a minimal gate prompt and an unconditional six-proposal budget. The
latter is diagnostic and must not be presented as a learned stopping policy.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import sys
import time
from urllib.parse import urlparse
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_iterative_proposals as b

GATE_SYSTEM='Answer the visual question. Magenta rectangle outlines are annotations, not part of the scene.'
GATE_QUESTION='Are there any objects without their own magenta rectangle? Y = yes. N = no. Answer one letter.'
SOURCE=Path('outputs/iterative-proposals-20260928')

def prepare(out):
 out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir();(out/'inputs').mkdir()
 cases=json.loads((SOURCE/'cases.json').read_text())
 for name in ['cases.json','native-payloads.json',*[c['name']+'.png' for c in cases]]:shutil.copyfile(SOURCE/name,out/name)
 blocks=[dict(case=c['name'],arm=a) for c in cases for a in ('fixed_six','simple_gate')]
 blocks += [dict(case=j['case'],arm='gate_diagnostic',variant=j['variant'],boxes=j['boxes'],expected=j['expected']) for j in json.loads((SOURCE/'blocks.json').read_text()) if j['arm']=='gate_diagnostic']
 random.Random(930103).shuffle(blocks)
 for i,j in enumerate(blocks):j['id']=i
 b.save(out/'blocks.json',blocks)
 sources=[Path(__file__),Path('scripts/benchmark_iterative_proposals.py'),Path('scripts/benchmark_native_grounding.py'),Path('scripts/benchmark_target_binding.py'),Path('scripts/benchmark_recognition.py'),Path('scripts/benchmark_object_proposals.py')]
 for p in sources:shutil.copyfile(p,out/'sources'/p.name)
 b.save(out/'plan.json',dict(frozen_files={p.name:b.sha(p) for p in [out/'cases.json',out/'native-payloads.json',out/'blocks.json',*out.glob('*.png')]},source_hashes={str(p):b.sha(p) for p in sources},
  reason='Adaptive follow-up motivated by original gate returning N after one proposal. New conditions frozen before their inference. No prompt search or selective reruns.',
  changes='simple_gate uses the identical one-box prompt and overlay, replacing ONLY the gate system/user question by minimal text. Same Y/N grammar and one-token limit. fixed_six uses identical one-box prompt/overlay, omits all gates, and attempts six boxes. Six is a fixed compute budget, not a claimed object count; never supplied to the model. This control is not evidence of correct termination.',
  gate_system=GATE_SYSTEM,gate_question=GATE_QUESTION,cap=12,
  evaluation='Exactly the primary experiment score, cases, decoding and diagnostic overlays. One run each. Do not pool or replace original results.'))
 print('Prepared',len(blocks),'follow-up blocks')

def run(out,base):
 u=urlparse(base);assert u.scheme=='http' and u.hostname in ('localhost','127.0.0.1')
 b.verify(out);assert not (out/'responses.jsonl').exists()
 b.save(out/'server-before.json',b.http(base+'/props'))
 blocks=json.loads((out/'blocks.json').read_text());rid=0
 with (out/'requests.jsonl').open('x') as requests,(out/'responses.jsonl').open('x') as responses,(out/'results.jsonl').open('x') as results:
  def call(block,mode,im):
   nonlocal rid
   start=time.monotonic();p=b.payload(im,mode)
   if mode=='gate':
    p['messages'][0]['content']=GATE_SYSTEM;p['messages'][1]['content'][1]['text']=GATE_QUESTION
   name='inputs/'+str(rid)+'.png';im.save(out/name)
   j=dict(id=rid,block_id=block['id'],case=block['case'],arm=block['arm'],mode=mode,image=name,image_sha256=b.sha(out/name),payload_digest=b.digest(p),payload=p)
   requests.write(json.dumps(j)+'\n');requests.flush();r={k:v for k,v in j.items() if k!='payload'};rid+=1
   try:
    response=b.http(base+'/v1/chat/completions',p);r['response']=response
    ch=response['choices'][0];r['answer']=ch['message']['content']
    if mode=='gate':
     assert r['answer'] in ('Y','N') and response['usage']['completion_tokens']==1
     r['decision']=r['answer']
    else:
     if ch['finish_reason']!='stop':raise ValueError('truncated')
     r['boxes']=b.parse_boxes(r['answer'],single=True)
    r['valid']=True
   except Exception as exc:r.update(valid=False,error=f'{type(exc).__name__}: {exc}')
   r['seconds']=time.monotonic()-start;responses.write(json.dumps(r)+'\n');responses.flush();return r
  for block in blocks:
   start=time.monotonic();im=Image.open(out/(block['case']+'.png')).convert('RGB');boxes=[];trace=[];ids=[]
   if block['arm']=='gate_diagnostic':
    r=call(block,'gate',b.marked(im,block['boxes']));ids.append(r['id']);reason='diagnostic'
   else:
    limit=6 if block['arm']=='fixed_six' else 12
    for step in range(limit):
     r=call(block,'single',b.marked(im,boxes));ids.append(r['id'])
     if not r['valid']:reason='invalid_box';break
     if not r['boxes']:reason='empty_proposal';break
     boxes.extend(r['boxes']);decision=None;gid=None
     if block['arm']=='simple_gate':
      gate=call(block,'gate',b.marked(im,boxes));ids.append(gate['id']);decision=gate.get('decision');gid=gate['id']
     trace.append(dict(step=step+1,boxes=deepcopy(boxes),box_request=r['id'],gate_request=gid,decision=decision,seconds=time.monotonic()-start))
     if block['arm']=='simple_gate':
      if not gate['valid']:reason='invalid_gate';break
      if decision=='N':reason='no';break
    else:reason='budget' if block['arm']=='fixed_six' else 'cap'
   result=dict(**block,proposals=boxes,request_ids=ids,trace=trace,stop=reason,seconds=time.monotonic()-start)
   if block['arm']=='gate_diagnostic':result.update(decision=r.get('decision'),valid=r['valid'])
   results.write(json.dumps(result)+'\n');results.flush()
   print(block['id']+1,'/',len(blocks),block['case'],block['arm'],len(boxes),reason,round(result['seconds'],2),flush=True)
 b.save(out/'server-after.json',b.http(base+'/props'));b.save(out/'health-after.json',b.http(base+'/health'))

def analyze(out):
 # Only report arm names differ; scoring/verification are exactly the frozen original.
 b.ARMS=('fixed_six','simple_gate')
 b.analyze(out)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8080');a=p.parse_args()
 if a.command=='run':run(a.output,a.base)
 else:globals()[a.command](a.output)
