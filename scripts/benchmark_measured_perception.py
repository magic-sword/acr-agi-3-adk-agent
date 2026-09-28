"""Frozen factual questions: image-only versus the runtime's measured evidence.

This isolates recognition/readout on identical frames; live gameplay is separate.
"""
import argparse
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.cognition.perception import measure,context_record
from scripts.benchmark_attention_selection import http,now
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_official_relations import sha
from scripts.benchmark_recognition import save_json

SOURCE=Path('outputs/simple-action-20260928')


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    cases=json.loads((SOURCE/'cases.json').read_text());jobs=[];measurements={}
    originals={j['case']:j['payload'] for j in json.loads((SOURCE/'jobs.json').read_text()) if j['arm']=='images'}
    for c in cases:
        measured=[]
        for t in c['trials']:
            r=measure(None if c['after_only'] else t['before'],t['after'],
                      after_id=t['id']+'after',before_id=t['id']+'before',action=t['action'])
            measured.append(r)
            for f in ['before','after']:
                name=f"{c['name']}-{t['id']}-{f}.png";shutil.copyfile(SOURCE/name,out/name)
        measurements[c['name']]=measured
        for arm in ['images','measured']:
            payload=deepcopy(originals[c['name']]);payload.update(max_tokens=1,
                grammar='root ::= '+' | '.join(json.dumps(x) for x in [o['letter'] for o in c['options']]+['X']))
            if arm=='measured':
                # Same system/question/options. No expected label, rule or role added.
                content=[dict(type='text',text='These are separate recorded trials. WAIT means no button was pressed.')]
                for t,r in zip(c['trials'],measured):
                    content.append(dict(type='text',text=t['id']+': '+json.dumps(context_record(r,inventory=True),separators=(',',':'))))
                content.append(deepcopy(originals[c['name']]['messages'][1]['content'][-1]))
                payload['messages'][1]['content']=content
            for repeat in range(2):
                jobs.append(dict(id=len(jobs),case=c['name'],arm=arm,repeat=repeat,payload=payload,payload_digest=digest(payload)))
    for name,data in [('cases',cases),('jobs',jobs),('measurements',measurements)]:save_json(out/(name+'.json'),data)
    paths=[Path(__file__),Path('agent/cognition/perception.py'),Path('agent/cognition/geometry.py')]
    for p in paths:shutil.copyfile(p,out/'sources'/p.name)
    save_json(out/'plan.json',dict(created=now(),requests=len(jobs),repeats=2,
        cases_digest=digest(cases),jobs_digest=digest(jobs),sources={str(p):sha(p) for p in paths},
        design='29 reused frozen factual questions x 2 evidence sources x 2 timing repeats; identical system and final question, greedy single-token grammar, cache disabled. Runtime measurement adapter, not handwritten change descriptions. Recognition/readout test, not a rule/semantic reasoning benchmark. Live runs are reported separately.',
        limitation='Small previously studied synthetic cases; not held-out. Measured evidence includes computed changes. Scores must not be interpreted as improved image-only perception. No prompt tuning after execution.'))
    print('Prepared',len(jobs),'requests')


def run(out):
    load=lambda n:json.loads((out/n).read_text());jobs=load('jobs.json');plan=load('plan.json')
    assert digest(jobs)==plan['jobs_digest']
    for p,h in plan['sources'].items():assert sha(p)==h
    assert not (out/'responses.jsonl').exists()
    save_json(out/'server-before.json',http('http://127.0.0.1:8080/props'))
    random.Random(928041).shuffle(jobs)
    with (out/'responses.jsonl').open('w') as f:
        for i,j in enumerate(jobs):
            start=time.monotonic();r={k:j[k] for k in ['id','case','arm','repeat','payload_digest']}
            try:
                response=http('http://127.0.0.1:8080/v1/chat/completions',j['payload'],120)
                r.update(response=response,answer=response['choices'][0]['message']['content'])
            except Exception as e:r.update(error=str(e),answer='')
            r['seconds']=time.monotonic()-start;f.write(json.dumps(r)+'\n');f.flush()
            if i%20==0:print(i+1,j['arm'],r['answer'],flush=True)
    save_json(out/'server-after.json',http('http://127.0.0.1:8080/props'))
    save_json(out/'health-after.json',http('http://127.0.0.1:8080/health'))


def analyze(out):
    load=lambda n:json.loads((out/n).read_text());cases=load('cases.json');jobs=load('jobs.json');plan=load('plan.json')
    rows=read_lines(out/'responses.jsonl');bycase={c['name']:c for c in cases}
    assert digest(cases)==plan['cases_digest'] and digest(jobs)==plan['jobs_digest']
    assert Counter(r['id'] for r in rows)==Counter(j['id'] for j in jobs)
    for r in rows:
        c=bycase[r['case']];j=jobs[r['id']]
        assert all(r[k]==j[k] for k in ['case','arm','repeat','payload_digest'])
        assert digest(j['payload'])==j['payload_digest']
        r['valid']=r['answer'] in [o['letter'] for o in c['options']]+['X'] and r.get('response',{}).get('usage',{}).get('completion_tokens')==1
        r.update(correct=bool(r['valid'] and r['answer']==c['truth']),truth=c['truth'],group=c['group'])
    arms=[]
    for arm in ['images','measured']:
        rr=[r for r in rows if r['arm']==arm]
        correct=lambda cc:sum(all(r['correct'] for r in rr if r['case']==c['name']) for c in cc)
        arms.append(dict(arm=arm,correct=correct(cases),total=len(cases),valid=sum(r['valid'] for r in rr),
            groups={g:dict(correct=correct([c for c in cases if c['group']==g]),total=sum(c['group']==g for c in cases)) for g in ['color','motion','panel','binding']},
            actual_movement=correct([c for c in cases if c['group']=='motion' and c['movement_class']!=3]),
            no_movement=correct([c for c in cases if c['group']=='motion' and c['movement_class']==3]),
            disagreements=sum(len({r['answer'] for r in rr if r['case']==c['name']})>1 for c in cases),
            median_seconds=statistics.median(r['seconds'] for r in rr)))
    save_json(out/'summary.json',dict(arms=arms,rows=rows))
    assert load('server-before.json')==load('server-after.json')
    assert load('health-after.json')['status']=='ok'
    for p,h in plan['sources'].items():assert sha(p)==h and sha(out/'sources'/Path(p).name)==h
    for j in jobs:
        p=j['payload'];content=p['messages'][1]['content']
        assert p['max_tokens']==1 and not p['cache_prompt']
        if j['arm']=='measured':assert all(x['type']=='text' for x in content)
    save_json(out/'verification.json',dict(passed=True,requests=len(rows),valid_outputs=sum(r['valid'] for r in rows),
        input_code_hashes=True,same_server=True,all_jobs_once=True))
    print(json.dumps(arms,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();globals()[a.command](a.output)
