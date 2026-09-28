"""Exploratory one-question calls after noticing action-choice/query echoing."""
import argparse
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rule_brief import request_for as brief_request, SOURCE
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import save_json
from scripts.benchmark_official_relations import sha


def request_for(c, m, task):
    p = brief_request(c, m)
    context = json.loads(p['messages'][1]['content'])
    instruction = p['messages'][0]['content']
    instruction = instruction[:instruction.index('prediction_choice:')]
    instruction = instruction.replace('then answer two questions.', 'then answer ONE question.')
    if task == 'prediction':
        context.pop('target'); context.pop('available_actions')
        instruction += '''Choose the option ID whose state best predicts CURRENT after the query_action, or uncertain. You are not selecting an action. Return only JSON: {"prediction_choice":"option ID or uncertain","rule_hypothesis":"brief rule or why not identifiable","evidence_trials":["trial IDs"],"limitation":"what remains unproven","next_test":"one discriminating intervention"}. Keep text fields short. No success feedback is given.'''
    else:
        context.pop('prediction_options'); context.pop('query_action')
        instruction += '''Choose A, B, or WAIT that best achieves TARGET from CURRENT, or uncertain. You are selecting the NEXT action; no action has been chosen for you. Return only JSON: {"action_choice":"A/B/WAIT/uncertain","rule_hypothesis":"brief rule or why not identifiable","evidence_trials":["trial IDs"],"limitation":"what remains unproven","next_test":"one discriminating intervention"}. Keep text fields short. No success feedback is given.'''
    p['messages'] = [dict(role='system', content=instruction), dict(role='user', content=json.dumps(context, separators=(',', ':')))]
    return p


def prepare(out):
    out.mkdir(parents=True, exist_ok=False); (out/'sources').mkdir()
    cases = json.loads((SOURCE/'cases.json').read_text()); measured = json.loads((SOURCE/'measured.json').read_text())
    jobs = []
    for c in cases:
        for task in ['prediction', 'action']:
            p = request_for(c, measured[c['name']], task)
            jobs.append(dict(id=len(jobs), case=c['name'], task=task, arm='split', payload=p, payload_digest=digest(p)))
    save_json(out/'cases.json', cases); save_json(out/'jobs.json', jobs)
    paths = [Path(__file__), Path('scripts/benchmark_rule_brief.py'), Path('scripts/benchmark_rule_inference.py')]
    for p in paths: shutil.copyfile(p, out/'sources'/p.name)
    save_json(out/'plan.json', dict(created_at=now(), requests=len(jobs), warmups=1, arms=['split'],
        jobs_digest=digest(jobs), cases_digest=digest(cases), sources={str(p): sha(p) for p in paths},
        source=str(SOURCE), source_measurements_digest=digest(measured), exploratory=True,
        rationale='Added after finding all earlier action_choice responses echoed query_action. Two separate calls per episode: prediction has no target; action has no query_action or prediction options. Same brief measurements, trials and truth. Changed task packaging, instruction and compute; not evidence for one variable alone. No further tuning.'))
    print('Prepared exploratory split', len(jobs), 'requests')


def run(out):
    load = lambda n: json.loads((out/n).read_text())
    plan, jobs = load('plan.json'), load('jobs.json'); cases = {c['name']: c for c in load('cases.json')}
    assert digest(jobs) == plan['jobs_digest']
    for p, h in plan['sources'].items(): assert sha(p) == h
    assert not (out/'responses.jsonl').exists()
    save_json(out/'server-before.json', http('http://127.0.0.1:8080/props'))
    order = jobs.copy(); random.Random(8132).shuffle(order)
    with (out/'responses.jsonl').open('w') as log:
        for j, warm in [(jobs[0], True)] + [(j, False) for j in order]:
            r = {k: j[k] for k in ['id','case','task','arm','payload_digest']}; r.update(warmup=warm, started_at=now()); start=time.monotonic()
            try:
                response = http('http://127.0.0.1:8080/v1/chat/completions', j['payload'], 240)
                choice = response['choices'][0]; r.update(response=response, answer=choice['message']['content'])
                p = json.loads(r['answer']); r['parsed'] = p
                key = j['task']+'_choice'; allowed = [o['id'] for o in cases[j['case']]['options']] if j['task']=='prediction' else ['A','B','WAIT']
                assert p[key] in allowed+['uncertain']
                assert all(isinstance(p[k],str) for k in ['rule_hypothesis','limitation','next_test'])
                assert isinstance(p['evidence_trials'],list) and all(t in [t['id'] for t in cases[j['case']]['trials']] for t in p['evidence_trials'])
                r['valid'] = choice['finish_reason']=='stop'
            except Exception as e: r.update(valid=False,error=f'{type(e).__name__}: {e}')
            r['seconds']=time.monotonic()-start;log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
            print(j['id'],j['case'],j['task'],r['valid'],round(r['seconds'],2),r.get('error',''),flush=True)
    save_json(out/'server-after.json',http('http://127.0.0.1:8080/props'));save_json(out/'health-after.json',http('http://127.0.0.1:8080/health'))


def analyze(out):
    cases=json.loads((out/'cases.json').read_text());rows=[r for r in read_lines(out/'responses.jsonl') if not r['warmup']]
    assert len(rows)==len(cases)*2
    scored=[]
    for c in cases:
        rr={r['task']:r for r in rows if r['case']==c['name']}
        scores={task:bool(r['valid'] and r.get('parsed',{}).get(task+'_choice')==c['truth'][task+'_choice']) for task,r in rr.items()}
        scored.append(dict(case=c['name'],arm='split',family=c['family'],valid=all(r['valid'] for r in rr.values()),**scores,
            joint=all(scores.values()),truth=c['truth'],parsed={task:r.get('parsed',r.get('answer')) for task,r in rr.items()},
            seconds=sum(r['seconds'] for r in rr.values()),prompt_tokens=max(r.get('response',{}).get('usage',{}).get('prompt_tokens',0) for r in rr.values())))
    main=[r for r in scored if r['family']!='uncertain'];controls=[r for r in scored if r['family']=='uncertain']
    arm=dict(arm='split',valid=sum(r['valid'] for r in scored),total=len(scored),main_cases=len(main),prediction=sum(r['prediction'] for r in main),action=sum(r['action'] for r in main),joint=sum(r['joint'] for r in main),uncertainty_joint=sum(r['joint'] for r in controls),uncertainty_cases=len(controls),median_seconds=statistics.median(r['seconds'] for r in scored),max_prompt_tokens=max(r['prompt_tokens'] for r in scored),families={f:{k:sum(r[k] for r in main if r['family']==f) for k in ['prediction','action','joint']} for f in ['counter','orbit','clock','panels']})
    save_json(out/'summary.json',dict(arms=[arm],rows=scored));print(json.dumps(arm,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();globals()[a.command](a.output)
