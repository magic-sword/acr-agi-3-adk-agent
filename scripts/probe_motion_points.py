"""Separate single-image localization diagnostic; not a motion-prompt replacement."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import encode, save_json
from scripts.benchmark_motion_inputs import raw_image, NEAREST
from scripts.benchmark_visual_emphasis import references

LABELS={'orange_blue':'the adjacent orange-and-blue group', 'white_cross':'the white cross', 'yellow_strip':'the long yellow strip'}


def to_grid(point, mode):
    return [x*64/1000-.5 for x in point] if mode=='normalized' else point


def parse(response, mode):
    c=response['choices'][0]
    if c['finish_reason']=='length':raise ValueError('truncated')
    v=json.loads(c['message']['content']);limit=1000 if mode=='normalized' else 63
    if not isinstance(v,list) or len(v)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=limit for x in v):
        raise ValueError('invalid point')
    return v


def prepare(source, output):
    output.mkdir(parents=True,exist_ok=False)
    c=next(c for c in json.loads(source.read_text()) if c['name']=='forward');jobs=[]
    for frame,boxes in zip(('before','after'),references('forward')):
        im=raw_image(c[frame],c['palette']).resize((384,384),NEAREST)
        for target,label in LABELS.items():
            b=boxes[target];truth=[(b[0]+b[2])/2,(b[1]+b[3])/2]
            for mode in ('grid','normalized'):
                coord=('Use a 64 by 64 grid: column x and row y run from 0 to 63 at cell centers.' if mode=='grid' else
                       'Use normalized image coordinates: x and y run from 0 to 1000 from the left/top boundary to the right/bottom boundary.')
                p={'model':'qwen3-vl-4b-instruct','messages':[{'role':'user','content':[encode(im),
                    {'type':'text','text':f'Locate the center of {label}. {coord} Return only a JSON array [x,y].'}]}],
                    'temperature':0,'seed':123,'max_tokens':40,'cache_prompt':False,'id_slot':0,'stream':False}
                for rep in range(2):jobs.append({'frame':frame,'target':target,'mode':mode,'repeat':rep,'truth':truth,'payload':p,'payload_digest':digest(p)})
    save_json(output/'jobs.json',jobs);save_json(output/'plan.json',{'created_at':now(),'jobs_digest':digest(jobs),
        'scope':'Single image, original board only; 0..63 versus 0..1000 coordinates. Two repeats of identical input, not independent examples.',
        'correct':'Euclidean center error <=2 original cells; invalid/timeout counts as failure',
        'host_motion':'Paired orange-blue centers: dx within 1 cell of 0 and dy within 1 cell of -5',
        'wall_budget_seconds':20,'max_tokens':40,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('Prepared',len(jobs),'point requests')


def run(output,base):
    assert not (output/'measurements.jsonl').exists()
    jobs=json.loads((output/'jobs.json').read_text());plan=json.loads((output/'plan.json').read_text())
    assert digest(jobs)==plan['jobs_digest'] and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['source_sha256']
    save_json(output/'server-before.json',http(base.removesuffix('/v1')+'/props'))
    random.Random(27000).shuffle(jobs)
    with (output/'measurements.jsonl').open('w') as log:
        for j in jobs:
            r={k:v for k,v in j.items() if k!='payload'};r['started_at']=now();start=time.monotonic()
            try:
                response=http(base+'/chat/completions',j['payload'],20);r['response']=response;r['point']=parse(response,j['mode']);r['grid_point']=to_grid(r['point'],j['mode']);r['valid']=True
                r['error_cells']=math.hypot(*(a-b for a,b in zip(r['grid_point'],j['truth'])))
            except Exception as exc:r.update(valid=False,error=str(exc))
            r['seconds']=time.monotonic()-start;r['correct']=r['valid'] and r['seconds']<=20 and r['error_cells']<=2
            log.write(json.dumps(r)+'\n');log.flush();print(j['frame'],j['target'],j['mode'],r.get('point'),r.get('error_cells'),flush=True)
    save_json(output/'server-after.json',http(base.removesuffix('/v1')+'/props'))
    rows=read_lines(output/'measurements.jsonl');summary=[]
    for mode in ('grid','normalized'):
        rs=[r for r in rows if r['mode']==mode];motion=[]
        for rep in range(2):
            pair=[next(r for r in rs if r['repeat']==rep and r['target']=='orange_blue' and r['frame']==f) for f in ('before','after')]
            if all(r['valid'] for r in pair):
                dx,dy=[a-b for a,b in zip(pair[1]['grid_point'],pair[0]['grid_point'])]
                motion.append({'repeat':rep,'dx':dx,'dy':dy,'within_one_cell':abs(dx)<=1 and abs(dy+5)<=1})
        summary.append({'mode':mode,'n':len(rs),'correct':sum(r['correct'] for r in rs),'motion':motion})
    save_json(output/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('outputs/motion-inputs-20260927/cases.json'));p.add_argument('--base',default='http://127.0.0.1:8080/v1');a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.output)
    else:run(a.output,a.base)
