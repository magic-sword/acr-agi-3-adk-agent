"""Paired prompt ablation: tell Qwen the scene is a game made of geometric shapes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import save_json
from scripts.benchmark_object_recognition import run_jobs, RUBRIC

CONTEXT = 'These observations show a two-dimensional game composed of flat geometric shapes.'
ARMS = ('baseline_greedy', 'geometry_greedy', 'baseline_official', 'geometry_official')


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    old=json.loads((source/'jobs.json').read_text());jobs=[]
    for j in old:
        if not (j['arm'].startswith('plain_') or j['arm'].startswith('no_image_')): continue
        for context in (False,True):
            p=deepcopy(j['payload'])
            if context:
                p['messages'][0]['content'] += '\n'+CONTEXT
            label='geometry' if context else 'baseline'
            sampler='official' if j['arm'].endswith('official') else 'greedy'
            jobs.append({'id':len(jobs),'source_id':j['id'],'case':j['case'],'kind':j['kind'],
                         'arm':label+'_'+sampler,'context':context,'repeat':j['repeat'],
                         'payload':p,'payload_digest':digest(p)})
    assert len(jobs)==96
    for a,b in zip(jobs[::2],jobs[1::2]):
        check=deepcopy(b['payload']);check['messages'][0]['content']=check['messages'][0]['content'].removesuffix('\n'+CONTEXT)
        assert check==a['payload']
        assert a['payload']==next(j['payload'] for j in old if j['id']==a['source_id'])
    cases=json.loads((source/'cases.json').read_text())
    save_json(output/'jobs.json',jobs);save_json(output/'cases.json',cases)
    save_json(output/'plan.json',{'created_at':now(),'context_added_verbatim':CONTEXT,'main_requests':96,
        'arms':ARMS,'jobs_digest':digest(jobs),'cases_digest':digest(cases),'wall_budget_seconds':20,'max_tokens':256,
        'scope':'Same seven cases as prior object-recognition study: one real transition, reversed/repeated controls, four synthetic cases. 84 image requests and 12 no-image controls, three repeats per combination.',
        'design':'Rerun the identical original-image baseline in shuffled order with context added. No auxiliary images, no role/target/action hints. Same seeds, generation settings and output/time budgets. Only one added system sentence differs within each pair.',
        'primary':RUBRIC,'secondary':'Whether the entire real scene is positively described as one robot/person instead of separate geometry. Score semantics, not keyword occurrence. Description change alone is not motion success.',
        'controls':'No-image responses must acknowledge insufficient visual evidence; geometric context does not establish motion or lack of motion.',
        'warmup':'one forward request per arm, excluded from main',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path('scripts/benchmark_object_recognition.py').read_bytes()).hexdigest()})
    for path in [Path(__file__),Path('scripts/benchmark_object_recognition.py')]:
        (output/path.name).write_bytes(path.read_bytes())
    print('Prepared 96 main requests; baseline equality and one-sentence-only difference checked')


def run(output,base):
    jobs=json.loads((output/'jobs.json').read_text());plan=json.loads((output/'plan.json').read_text())
    assert digest(jobs)==plan['jobs_digest']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['source_sha256']
    assert hashlib.sha256(Path('scripts/benchmark_object_recognition.py').read_bytes()).hexdigest()==plan['runner_sha256']
    save_json(output/'server-before.json',http(base.removesuffix('/v1')+'/props'))
    warm=[next(j for j in jobs if j['case']=='forward' and j['arm']==a) for a in ARMS]
    run_jobs(output,base,jobs,'measurements.jsonl',warm)
    save_json(output/'server-after.json',http(base.removesuffix('/v1')+'/props'))
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']]
    random.Random(71321).shuffle(rows)
    save_json(output/'review-key.json',{str(i):r['id'] for i,r in enumerate(rows)})
    groups={}
    for i,r in enumerate(rows):
        key=(r['case'],r.get('answer',''),r['valid'] and r['within_budget'])
        groups.setdefault(key,[]).append(i)
    save_json(output/'review-groups.json',[{'group':i,'case':k[0],'answer':k[1],'valid':k[2],'review_ids':v}
                                         for i,(k,v) in enumerate(groups.items())])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--source',type=Path,default=Path('outputs/object-recognition-20260927'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8080/v1');a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.output)
    else:run(a.output,a.base)
