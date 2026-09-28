"""Actual adaptive calls: flat, model gate, direct combination, program gate."""
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
from scripts.observation_questions import build_questions
from scripts.hierarchical_questions import KINDS,LABEL_FLAGS,measured_flags,encode_flags,flat_result,needs_detail,make_request
from scripts.benchmark_attention_selection import http,now
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_recognition import save_json
from scripts.benchmark_official_relations import sha
from scripts.benchmark_simple_action import render

SOURCE=Path('outputs/one-token-20260928/generated-questions.json')
MODES=['flat','model_gate','direct','program_gate']
REPEATS=2


def fixtures():
    def rect(x=20,y=20,w=3,h=3,color=9):
        g=[[5]*64 for _ in range(64)]
        for yy in range(y,y+h):
            for xx in range(x,x+w):g[yy][xx]=color
        return g
    result=[]
    for name,b,a in [
        ('grow',rect(),rect(w=5)),('move_grow',rect(),rect(x=21,w=5)),
        ('shrink',rect(w=5),rect()),('color',rect(),rect(color=12)),
        ('move_color_unresolved',rect(),rect(x=21,color=12)),
        ('new_unresolved',[[5]*64 for _ in range(64)],rect()),
        ('large_uncovered',[[5]*64 for _ in range(64)],rect(w=20,h=20)),
    ]:result.append(dict(name=name,before=b,after=a,action='A'))
    from scripts.benchmark_rule_inference import objects
    result.append(dict(name='ambiguous_copies',before=objects([[16,20],[40,20]],[0,0]),after=objects([[24,20],[32,20]],[0,0]),action='A'))
    return result


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    old=json.loads(SOURCE.read_text());source_cases={c['name']:c for c in json.loads(Path('outputs/simple-action-20260928/cases.json').read_text())}
    scenes=[]
    for g in old:
        t=next(t for t in source_cases[g['case']]['trials'] if t['id']==g['trial'])
        scenes.append(dict(name=g['case']+'_'+g['trial'],origin='previous',before=t['before'],after=t['after'],action=t['action'],generated={k:g[k] for k in ['questions','coverage','independent_states']}))
    for f in fixtures():scenes.append(dict(**f,origin='diagnostic',generated=build_questions(f['before'],f['after'],f['action'])))
    palette=json.loads(Path('outputs/instance-state-20260928-final/cases.json').read_text())[0]['palette']
    subjects=[]
    for scene in scenes:
        for frame in ['before','after']:render(scene[frame],palette).save(out/f"{scene['name']}-{frame}.png")
        for q in scene['generated']['questions']:
            if q['property']!='position':continue
            context=q['context'];subjects.append(dict(id=len(subjects),scene=scene['name'],subject=q['subject_id'],context=context,
                truth=encode_flags(measured_flags(context)),origin=scene['origin']))
    jobs=[]
    for s in subjects:
        for mode in MODES:
            for repeat in range(REPEATS):jobs.append(dict(id=len(jobs),subject_id=s['id'],mode=mode,repeat=repeat))
    save_json(out/'scenes.json',scenes);save_json(out/'subjects.json',subjects);save_json(out/'jobs.json',jobs)
    sources=[Path(__file__),Path('scripts/hierarchical_questions.py'),Path('scripts/observation_questions.py'),Path('scripts/instance_state.py')]
    for p in sources:shutil.copyfile(p,out/'sources'/p.name)
    save_json(out/'plan.json',dict(created_at=now(),modes=MODES,repeats=REPEATS,pipelines=len(jobs),warmup_calls=2,
        subjects_digest=digest(subjects),jobs_digest=digest(jobs),scenes_digest=digest(json.loads((out/'scenes.json').read_text())),source_data_hash=sha(SOURCE),sources={str(p):sha(p) for p in sources},
        design='Previous 76 generated candidate pairs plus resizing/multiple-change/unresolved/uncovered diagnostics. 3 binary property questions; learned any-change gate then one 9-way combination; direct 9-way combination; deterministic measured-state gate then same combination question. All text-only, one token, grammar, identical subject context/system, greedy, no prompt cache, 2 timing repeats. Actual gated calls executed; No skips detail, Yes/unknown runs detail. No oracle used for model routing.',
        metrics='Actual calls and summed request time per full pass; exact property-set accuracy; known-change detection; false-negative gate skips; unresolved cases discarded as unchanged; residual changed pixels separately.',
        limitations='Small synthetic candidates; appearance includes shape/dimension changes. Ground truth concerns measured candidate states, not proof of physical identity. Program gate uses deterministic facts and is not model capability. Previous 27/29 scores are different questions. Timing repeats are not independent tasks. No production integration.'))
    print('Prepared',len(subjects),'subjects;',sum(s['truth'] not in ['A','X'] for s in subjects),'known changed;',sum(s['truth']=='X' for s in subjects),'unresolved;',len(jobs),'pipelines')


def run(out):
    load=lambda n:json.loads((out/n).read_text())
    subjects,jobs,plan=load('subjects.json'),load('jobs.json'),load('plan.json')
    assert digest(subjects)==plan['subjects_digest'] and digest(jobs)==plan['jobs_digest']
    for p,h in plan['sources'].items():assert sha(p)==h
    assert not (out/'calls.jsonl').exists()
    save_json(out/'server-before.json',http('http://127.0.0.1:8080/props'))
    tokens={s:http('http://127.0.0.1:8080/tokenize',dict(content=s,add_special=False)) for s in 'ABCDEFGHX'}
    assert all(len(t['tokens'])==1 for t in tokens.values());save_json(out/'token-audit.json',tokens)
    order=jobs.copy();random.Random(294010).shuffle(order)
    with (out/'calls.jsonl').open('w') as log,(out/'pipelines.jsonl').open('w') as results:
        def call(job,context,stage,warmup=False):
            payload=make_request(context,stage);r=dict(job_id=job['id'],subject_id=job['subject_id'],mode=job['mode'],repeat=job['repeat'],stage=stage,warmup=warmup,payload_digest=digest(payload),started_at=now());start=time.monotonic()
            try:
                response=http('http://127.0.0.1:8080/v1/chat/completions',payload,120);ch=response['choices'][0];answer=ch['message']['content']
                allowed=list(LABEL_FLAGS) if stage=='detail' else ['A','B','X']
                r.update(response=response,answer=answer,valid=answer in allowed and response['usage']['completion_tokens']==1)
            except Exception as e:r.update(valid=False,error=f'{type(e).__name__}: {e}',answer='X')
            r['seconds']=time.monotonic()-start;log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush();return r
        for stage in ['gate','detail']:call(jobs[0],subjects[0]['context'],stage,True)
        for index,j in enumerate(order):
            s=subjects[j['subject_id']];ctx=s['context'];calls=[];gate=None;host_time=0
            if j['mode']=='flat':
                for stage in KINDS:calls.append(call(j,ctx,stage))
                prediction=flat_result({c['stage']:c['answer'] for c in calls})
            elif j['mode']=='direct':
                calls.append(call(j,ctx,'detail'));prediction=calls[-1]['answer']
            else:
                if j['mode']=='model_gate':
                    calls.append(call(j,ctx,'gate'));gate=calls[-1]['answer'] if calls[-1]['valid'] else 'X'
                else:
                    start=time.perf_counter();flags=measured_flags(ctx);gate='X' if flags is None else 'B' if any(flags) else 'A';host_time=time.perf_counter()-start
                if needs_detail(gate):calls.append(call(j,ctx,'detail'));prediction=calls[-1]['answer']
                else:prediction='A'
            result=dict(**j,prediction=prediction,gate=gate,calls=len(calls),seconds=sum(c['seconds'] for c in calls)+host_time,host_gate_seconds=host_time,valid=all(c['valid'] for c in calls),stages=[c['stage'] for c in calls])
            results.write(json.dumps(result,ensure_ascii=False)+'\n');results.flush()
            if index%50==0:print(index+1,j['mode'],'calls',len(calls),'prediction',prediction,flush=True)
    save_json(out/'server-after.json',http('http://127.0.0.1:8080/props'));save_json(out/'health-after.json',http('http://127.0.0.1:8080/health'))


def analyze(out):
    subjects=json.loads((out/'subjects.json').read_text());rows=read_lines(out/'pipelines.jsonl')
    assert len(rows)==len(subjects)*len(MODES)*REPEATS
    for r in rows:
        s=subjects[r['subject_id']];r.update(truth=s['truth'],scene=s['scene'],subject=s['subject'],origin=s['origin'])
        r['correct']=bool(r['valid'] and r['prediction']==s['truth'])
    summary=[]
    for mode in MODES:
        rr=[r for r in rows if r['mode']==mode];first=[r for r in rr if r['repeat']==0]
        passes=[dict(repeat=k,calls=sum(r['calls'] for r in rr if r['repeat']==k),seconds=sum(r['seconds'] for r in rr if r['repeat']==k)) for k in range(REPEATS)]
        known_changed=[r for r in first if r['truth'] not in ['A','X']];unknown=[r for r in first if r['truth']=='X'];static=[r for r in first if r['truth']=='A']
        summary.append(dict(mode=mode,subjects=len(subjects),correct_all_repeats=sum(all(r['correct'] for r in rr if r['subject_id']==s['id']) for s in subjects),
            detected_changes=sum(r['valid'] and r['prediction'] not in ['A','X'] for r in known_changed),known_changed=len(known_changed),
            exact_changed=sum(r['correct'] for r in known_changed),false_unchanged=sum(r['prediction']=='A' for r in known_changed),
            gate_missed_changes=sum(r['gate']=='A' for r in known_changed) if mode.endswith('gate') else None,
            static_correct=sum(r['correct'] for r in static),static=len(static),unknown_retained=sum(r['prediction']=='X' for r in unknown),unknown=len(unknown),unknown_discarded=sum(r['prediction']=='A' for r in unknown),
            prediction_disagreements=sum(len({r['prediction'] for r in rr if r['subject_id']==s['id']})>1 for s in subjects),
            median_calls_per_pass=statistics.median(p['calls'] for p in passes),median_seconds_per_pass=statistics.median(p['seconds'] for p in passes),passes=passes))
    save_json(out/'summary.json',dict(arms=summary,rows=rows));print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();globals()[a.command](a.output)
