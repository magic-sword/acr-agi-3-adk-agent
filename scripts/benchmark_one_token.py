"""Paired output-cap/grammar test on frozen simple factual questions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.observation_questions import grammar, build_questions
from scripts.report_simple_action import reported_choice
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import save_json
from scripts.benchmark_official_relations import sha

SOURCE=Path('outputs/simple-action-20260928')
MODES=['baseline16','limit1','grammar1']
REPEATS=3


def variant(payload,case,mode):
    result=deepcopy(payload)
    if mode!='baseline16':result['max_tokens']=1
    if mode=='grammar1':result['grammar']=grammar([o['letter'] for o in case['options']]+['X'])
    return result


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    cases=json.loads((SOURCE/'cases.json').read_text());bycase={c['name']:c for c in cases};jobs=[]
    for src in json.loads((SOURCE/'jobs.json').read_text()):
        if src['arm']=='images':continue
        for mode in MODES:
            p=variant(src['payload'],bycase[src['case']],mode)
            for repeat in range(REPEATS):jobs.append(dict(id=len(jobs),case=src['case'],source=src['arm'],mode=mode,repeat=repeat,payload=p,payload_digest=digest(p)))
    generated=[]
    for c in cases:
        for t in c['trials']:
            start=time.perf_counter();q=build_questions(t['before'],t['after'],t['action'])
            generated.append(dict(case=c['name'],trial=t['id'],seconds=time.perf_counter()-start,**q))
    save_json(out/'cases.json',cases);save_json(out/'jobs.json',jobs);save_json(out/'generated-questions.json',generated)
    paths=[Path(__file__),Path('scripts/observation_questions.py'),Path('scripts/instance_state.py')]
    for p in paths:shutil.copyfile(p,out/'sources'/p.name)
    save_json(out/'plan.json',dict(created_at=now(),requests=len(jobs),warmups=6,repeats=REPEATS,modes=MODES,
        cases_digest=digest(cases),jobs_digest=digest(jobs),sources={str(p):sha(p) for p in paths},source=str(SOURCE),source_jobs_hash=sha(SOURCE/'jobs.json'),
        design='29 frozen questions x 2 text sources x 3 output modes x 3 repeats. Re-run 16-token baseline interleaved with plain max_tokens=1 and max_tokens=1 plus one-letter GBNF. Same messages, temperature and cache_prompt=false. No hidden reasoning mode switch. No extra images or facts. Tokenizer endpoints audit labels. No grammar-based answer correction.',
        metrics='Factual accuracy, exact allowed-letter output, actual completion-token count, stop reason, wall time, paired per-question median speedup. Timing excludes question generation; reported separately.',
        limitations='Small repeated synthetic tasks; repeated calls are timing repeats, not independent accuracy examples. One output token still requires prompt processing. Grammar enforces syntax, never truth. Candidate-question generator is tested separately; its generated questions are not the benchmark questions. No production integration.'))
    print('Prepared',len(jobs),'model calls and',sum(len(x['questions']) for x in generated),'automatically generated questions')


def run(out):
    load=lambda n:json.loads((out/n).read_text())
    plan,jobs=load('plan.json'),load('jobs.json');cases={c['name']:c for c in load('cases.json')}
    assert digest(jobs)==plan['jobs_digest']
    for p,h in plan['sources'].items():assert sha(p)==h
    assert sha(SOURCE/'jobs.json')==plan['source_jobs_hash']
    assert not (out/'responses.jsonl').exists()
    save_json(out/'server-before.json',http('http://127.0.0.1:8080/props'))
    tokens={s:http('http://127.0.0.1:8080/tokenize',dict(content=s,add_special=False)) for s in 'ABCDEX'}
    assert all(len(x['tokens'])==1 for x in tokens.values());save_json(out/'token-audit.json',tokens)
    order=jobs.copy();random.Random(293320).shuffle(order)
    warm=[next(j for j in jobs if j['source']==s and j['mode']==m) for s in ['states','facts'] for m in MODES]
    with (out/'responses.jsonl').open('w') as log:
        for index,(j,iswarm) in enumerate([(j,True) for j in warm]+[(j,False) for j in order]):
            r={k:j[k] for k in ['id','case','source','mode','repeat','payload_digest']};r.update(warmup=iswarm,started_at=now());start=time.monotonic()
            try:
                response=http('http://127.0.0.1:8080/v1/chat/completions',j['payload'],120)
                choice=response['choices'][0];answer=choice['message']['content'];selection,form=reported_choice(answer,cases[j['case']])
                allowed=[o['letter'] for o in cases[j['case']]['options']]+['X']
                r.update(response=response,answer=answer,choice=selection,format=form,finish_reason=choice['finish_reason'],
                    completion_tokens=response['usage']['completion_tokens'],strict_valid=answer in allowed,
                    correct=selection==cases[j['case']]['truth'])
            except Exception as e:r.update(error=f'{type(e).__name__}: {e}',correct=False,strict_valid=False)
            r['seconds']=time.monotonic()-start;log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
            if index%25==0 or 'error' in r:print(index+1,j['source'],j['mode'],r.get('answer'),r.get('completion_tokens'),round(r['seconds'],3),r.get('error',''),flush=True)
    save_json(out/'server-after.json',http('http://127.0.0.1:8080/props'));save_json(out/'health-after.json',http('http://127.0.0.1:8080/health'))


def analyze(out):
    cases={c['name']:c for c in json.loads((out/'cases.json').read_text())};rows=[r for r in read_lines(out/'responses.jsonl') if not r['warmup']]
    assert len(rows)==29*2*3*REPEATS
    summary=[]
    for source in ['states','facts']:
        for mode in MODES:
            rr=[r for r in rows if r['source']==source and r['mode']==mode]
            groups={c:[r for r in rr if r['case']==c] for c in cases}
            summary.append(dict(source=source,mode=mode,requests=len(rr),correct_calls=sum(r['correct'] for r in rr),
                correct_all_repeats=sum(all(r['correct'] for r in rs) for rs in groups.values()),cases=len(groups),
                exact_letter=sum(r['strict_valid'] for r in rr),one_token=sum(r.get('completion_tokens')==1 for r in rr),
                median_seconds=statistics.median(r['seconds'] for r in rr),median_completion_tokens=statistics.median(r.get('completion_tokens',0) for r in rr),
                disagreements=sum(len({r.get('choice') for r in rs})>1 for rs in groups.values()),
                per_group={g:dict(correct=sum(all(r['correct'] for r in rs) for c,rs in groups.items() if cases[c]['group']==g),total=sum(cases[c]['group']==g for c in groups)) for g in ['color','motion','panel','binding']}))
    paired=[]
    for source in ['states','facts']:
        for c in cases:
            rr=[r for r in rows if r['source']==source and r['case']==c]
            bymode={m:[r for r in rr if r['mode']==m] for m in MODES}
            timings={m:statistics.median(r['seconds'] for r in rs) for m,rs in bymode.items()}
            paired.append(dict(case=c,source=source,timings=timings,
                grammar_speedup=timings['baseline16']/timings['grammar1'],
                choices={m:sorted(set(r.get('answer','<error>') for r in rs)) for m,rs in bymode.items()},
                correct={m:all(r['correct'] for r in rs) for m,rs in bymode.items()}))
    result=dict(arms=summary,paired=paired,paired_median_speedup={s:statistics.median(r['grammar_speedup'] for r in paired if r['source']==s) for s in ['states','facts']})
    save_json(out/'summary.json',result);print(json.dumps(result['arms'],ensure_ascii=False,indent=2));print(result['paired_median_speedup'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();globals()[a.command](a.output)
