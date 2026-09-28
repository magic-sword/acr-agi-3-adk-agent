"""Verify source hashes, controlled interventions, coverage and restoration."""
import json,sys
from pathlib import Path
from PIL import Image,ImageChops
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_official_relations import sha
from scripts.benchmark_explanation_xai import SOURCE,POSITION,WRONG

out=Path(sys.argv[1]);load=lambda n:json.loads((out/n).read_text());plan=load('plan.json');jobs=load('jobs.json');parents={j['id']:j for j in json.loads((SOURCE/'jobs.json').read_text())}
assert sha('scripts/benchmark_explanation_xai.py')==plan['script_sha256'] and digest(jobs)==plan['jobs_digest']
for name,h in load('input-hashes.json').items():assert sha(out/name)==h
for j in jobs:
    assert digest(j['payload'])==j['payload_digest'];base=next(b for b in jobs if b['case']==j['case'] and b['arm']=='base')
    if j['arm']=='base':assert j['payload']==parents[j['parent_id']]['payload']
    a=json.loads(json.dumps(j['payload']));b=base['payload']
    if j['arm'].startswith('inventory_'):
        expected=b['messages'][1]['content'][-1]['text'].replace(POSITION[j['case']],'' if j['arm']=='inventory_no_position' else WRONG[j['case']])
        if j['arm']=='inventory_no_position':expected=expected.replace(' \n','\n')
        assert a['messages'][1]['content'][-1]['text']==expected
        a['messages'][1]['content'][-1]=b['messages'][1]['content'][-1]
    for index,path,bpath in zip([1,3,5],j['paths'],base['paths']):
        im=Image.open(out/path);assert im.size==(640,820)
        bbox=ImageChops.difference(im,Image.open(out/bpath)).getbbox()
        if j['arm'] in ['target_black','control_black']:assert bbox==tuple(j['roi'] if j['arm']=='target_black' else j['control'])
        else:assert bbox is None
        a['messages'][1]['content'][index]=b['messages'][1]['content'][index]
    assert a==b
rows=read_lines(out/'measurements.jsonl');main=[r for r in rows if not r['warmup']]
assert len(rows)==11 and sorted(r['id'] for r in main)==list(range(10))
for r in rows:
    assert r['payload_digest']==jobs[r['id']]['payload_digest'] and r['valid']
    assert r['image_grid_thw']==[[1,52,40]]*3 and r['output_tokens']<=1024
    if not r['warmup'] and r['arm']=='base':assert r['baseline_reproduced']
for case in plan['cases']:
    trace=load(case+'-trace-refined.json');assert trace['max_logit_difference']==0
    assert len(trace['attention'])==32
    for r in trace['attention']:assert abs(sum(r['mass'].values())-1)<1e-6
assert len(load('prefix-scores.json'))==4
assert sorted(r['id'] for r in load('semantic-review.json')['rows'])==list(range(10))
before=load('server-before.json');after=load('server-restored.json')
assert {k:v for k,v in before.items() if k!='media_marker'}=={k:v for k,v in after.items() if k!='media_marker'}
assert load('health-restored.json')['status']=='ok' and load('browser-verification.json')['passed']
result=dict(passed=True,requests=10,warmups=1,prefix_interventions=4,baseline_reproduced=2,max_attention_logit_difference=0,checks=['frozen source/input hashes','exact previous baseline payloads and answers','only declared image/text changes','equal-area control mask','same image token grids','future attention weights zero (causal mask)','attention partitions sum to one','all outputs reviewed','all HTML selectors tested','VLM settings and health restored'],artifacts_sha256={n:sha(out/n) for n in ['jobs.json','measurements.jsonl','prefix-scores.json','summary.json','semantic-review.json']})
(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,indent=2))
