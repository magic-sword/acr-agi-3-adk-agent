"""Supplemental image-removal ablation of the frozen raw-delta requests."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_delta_recognition import verify,save,sha,digest,http


def prepare(source,out):
    cases,jobs=verify(source);out.mkdir(parents=True,exist_ok=False)
    selected=[]
    for j in jobs:
        if j['arm']!='delta':continue
        r=deepcopy(j);r['arm']='delta_text';content=r['payload']['messages'][1]['content']
        r['payload']['messages'][1]['content']=[part for part in content if part['type']!='image_url']
        r['payload_digest']=digest(r['payload']);selected.append(r)
    save(out/'jobs.json',selected);save(out/'cases.json',cases)
    save(out/'plan.json',dict(source=str(source),jobs_digest=digest(selected),cases_digest=digest(cases),
        code_sha256=sha(Path(__file__)),source_jobs_sha256=sha(source/'jobs.json'),
        design='Supplemental ablation prepared during four-arm execution, before its complete scores. Remove images only from every delta request; retain all BEFORE/AFTER inventories, raw changed-pixel runs, question, system, settings. No tracking. Not an independent preregistered confirmation.'))
    print('Prepared',len(selected),'text-only requests')


def load(out):
    p=json.loads((out/'plan.json').read_text());j=json.loads((out/'jobs.json').read_text());c=json.loads((out/'cases.json').read_text())
    assert digest(j)==p['jobs_digest'] and digest(c)==p['cases_digest'] and sha(Path(__file__))==p['code_sha256']
    source=Path(p['source']);verify(source);assert sha(source/'jobs.json')==p['source_jobs_sha256']
    return c,j


def run(out):
    _,jobs=load(out);base='http://127.0.0.1:8080';assert not (out/'responses.jsonl').exists()
    save(out/'server-before.json',http(base,'/props'));random.Random(929831).shuffle(jobs)
    with (out/'responses.jsonl').open('x') as f:
        for i,j in enumerate(jobs):
            r={k:v for k,v in j.items() if k!='payload'};start=time.perf_counter()
            try:
                response=http(base,'/v1/chat/completions',j['payload']);r.update(response=response,answer=response['choices'][0]['message']['content'])
            except Exception as exc:r.update(error=str(exc),answer='')
            r['seconds']=time.perf_counter()-start;f.write(json.dumps(r)+'\n');f.flush()
            if i%10==0:print(i+1,'/',len(jobs),r['case'],r['answer'],flush=True)
    save(out/'server-after.json',http(base,'/props'))


def analyze(out):
    cases,jobs=load(out);bycase={c['name']:c for c in cases};rows=list(map(json.loads,(out/'responses.jsonl').read_text().splitlines()))
    assert sorted(r['id'] for r in rows)==sorted(j['id'] for j in jobs)
    for r in rows:
        c=bycase[r['case']];r['valid']=not r.get('error') and r['answer'] in [o['letter'] for o in c['options']]+['X']
        r.update(correct=bool(r['valid'] and r['answer']==c['truth']),group=c['group'],origin=c['origin'],truth=c['truth'])
    groups={}
    for key in sorted({r['group'] for r in rows}|{r['origin'] for r in rows}|{'all'}):
        rr=[r for r in rows if key=='all' or r['group']==key or r['origin']==key]
        groups[key]=dict(correct=sum(r['correct'] for r in rr),total=len(rr),abstentions=sum(r['answer']=='X' for r in rr))
    summary=dict(groups=groups,calls=len(rows),seconds=sum(r['seconds'] for r in rows),median_seconds=statistics.median(r['seconds'] for r in rows),
        invalid=sum(not r['valid'] for r in rows),server_unchanged=json.loads((out/'server-before.json').read_text())==json.loads((out/'server-after.json').read_text()))
    save(out/'scores.json',rows);save(out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path);a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.out)
    else:globals()[a.command](a.out)
