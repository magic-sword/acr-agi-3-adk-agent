"""Post-hoc transfer check of single-image coordinate reading on synthetic cases."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from copy import deepcopy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import encode, save_json
from scripts.benchmark_motion_inputs import raw_image, NEAREST
from scripts.probe_motion_points import parse, to_grid


def prepare(source, output):
    output.mkdir(parents=True,exist_ok=False);cases=json.loads((source/'cases.json').read_text())
    templates=json.loads((source/'points/jobs.json').read_text());jobs=[]
    for c in cases:
        if c['kind']!='synthetic':continue
        for frame in ('before','after'):
            # Ground truth includes adjacent orange AND blue, excluding the isolated blue distractor.
            orange=[(x,y) for y,row in enumerate(c[frame]) for x,v in enumerate(row) if v==12]
            xs=[x for x,y in orange];ys=[y for x,y in orange];l,r=min(xs),max(xs);t=min(ys);height=(max(ys)-t+1)*2
            truth=[(l+r)/2,t+(height-1)/2]
            im=raw_image(c[frame],c['palette']).resize((384,384),NEAREST)
            for mode in ('grid','normalized'):
                p=deepcopy(next(j['payload'] for j in templates if j['target']=='orange_blue' and j['mode']==mode))
                p['messages'][0]['content'][0]=encode(im)
                jobs.append({'case':c['name'],'frame':frame,'mode':mode,'truth':truth,'expected_direction':c['truth']['motion'],
                             'payload':p,'payload_digest':digest(p)})
    save_json(output/'jobs.json',jobs);save_json(output/'plan.json',{'created_at':now(),'exploratory':True,
        'reason':'Check transfer after normalized coordinates recovered real-pair displacement.',
        'jobs_digest':digest(jobs),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rubric':'Center error <=2 cells. Vector within 1 cell per axis. Direction from dominant displacement axis; unchanged if both abs components <=1.',
        'scope':'10 synthetic cases, two layouts, one response per image/mode; before images repeat across directions.'})
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('Prepared',len(jobs))


def run(output,base):
    assert not (output/'measurements.jsonl').exists()
    jobs=json.loads((output/'jobs.json').read_text());plan=json.loads((output/'plan.json').read_text())
    assert digest(jobs)==plan['jobs_digest'] and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['source_sha256']
    save_json(output/'server-before.json',http(base.removesuffix('/v1')+'/props'));random.Random(28000).shuffle(jobs)
    with (output/'measurements.jsonl').open('w') as log:
        for j in jobs:
            r={k:v for k,v in j.items() if k!='payload'};r['started_at']=now();start=time.monotonic()
            try:
                response=http(base+'/chat/completions',j['payload'],20);r['response']=response;r['point']=parse(response,j['mode']);r['grid_point']=to_grid(r['point'],j['mode']);r['valid']=True
                r['error_cells']=math.hypot(*(a-b for a,b in zip(r['grid_point'],j['truth'])))
            except Exception as exc:r.update(valid=False,error=str(exc))
            r['seconds']=time.monotonic()-start;r['correct']=r['valid'] and r['seconds']<=20 and r['error_cells']<=2
            log.write(json.dumps(r)+'\n');log.flush()
    save_json(output/'server-after.json',http(base.removesuffix('/v1')+'/props'))
    rows=read_lines(output/'measurements.jsonl');pairs=[]
    for name in sorted({r['case'] for r in rows}):
        for mode in ('grid','normalized'):
            b,a=[next(r for r in rows if r['case']==name and r['mode']==mode and r['frame']==f) for f in ('before','after')]
            p={'case':name,'mode':mode,'expected':a['expected_direction'],'direction_correct':False,'vector_correct':False}
            if all(r['valid'] and r['seconds']<=20 for r in (a,b)):
                dx,dy=[x-y for x,y in zip(a['grid_point'],b['grid_point'])];tx,ty=[x-y for x,y in zip(a['truth'],b['truth'])]
                direction='unchanged' if max(abs(dx),abs(dy))<=1 else ('right' if dx>0 else 'left') if abs(dx)>abs(dy) else ('down' if dy>0 else 'up')
                p.update(dx=dx,dy=dy,direction=direction,direction_correct=direction==p['expected'],vector_correct=abs(dx-tx)<=1 and abs(dy-ty)<=1)
            pairs.append(p)
    save_json(output/'paired-results.json',pairs);print(json.dumps(pairs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('outputs/motion-inputs-20260927'));p.add_argument('--base',default='http://127.0.0.1:8080/v1');a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.output)
    else:run(a.output,a.base)
