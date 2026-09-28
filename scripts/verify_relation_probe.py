"""Verify the bounded reference-field diagnostic against its parent inputs."""
import json,sys
from pathlib import Path
from copy import deepcopy
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_relation_overlay as core
from scripts.probe_relation_fields import SOURCE,INSTRUCTION
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_official_relations import sha
from scripts.benchmark_recognition import encode
from scripts.summarize_temporal_overlay import audit_key

out=Path(sys.argv[1]);load=lambda n:json.loads((out/n).read_text())
jobs=core.validate(out);plan=load('plan.json')
assert sha('scripts/probe_relation_fields.py')==plan['probe_script_sha256']
assert len(jobs)==8
parents=json.loads((SOURCE/'jobs.json').read_text())
for j in jobs:
    parent=next(p for p in parents if p['case']==j['case'] and p['backend']==j['backend'] and p['variant']==j['variant'].replace('reference_','relation_'))
    expected=deepcopy(parent['payload']);t=expected['messages'][1]['content'][-1]['text']
    t=t.replace(core.EVIDENCE['relation'],INSTRUCTION).replace('id, before, after, description, kind','id, reference, before, after, description, kind')
    expected['messages'][1]['content'][-1]['text']=t.replace('"id":"A","before":','"id":"A","reference":"specific visible reference or none","before":')
    assert expected==j['payload'] and j['catalog_key']==parent['catalog_key']
    assert digest(j['payload'])==j['payload_digest']
    for index,path in zip([1,3,5],j['paths']):
        im=Image.open(out/path).convert('RGB');assert im.size==(640,820)
        assert sha(out/path)==sha(SOURCE/path)
        assert encode(im)==j['payload']['messages'][1]['content'][index]
    counterpart=next(p for p in jobs if p['case']==j['case'] and p['variant']==j['variant'] and p['backend']!=j['backend'])
    assert counterpart['payload']==j['payload']
rows=read_lines(out/'measurements.jsonl');main=[r for r in rows if not r['warmup']]
assert len(rows)==12 and sorted(r['id'] for r in main)==list(range(8))
for r in rows:
    assert r['payload_digest']==jobs[r['id']]['payload_digest'] and 'error' not in r
    assert r['output_tokens']<=1024
    if r['backend']=='official':assert r['image_grid_thw']==[[1,52,40]]*3
assert {r['key'] for r in load('semantic-review.json')['rows']}=={audit_key(r) for r in main}
summary=load('summary.json');assert summary['main_requests']==8 and summary['warmups']==4
assert all(a['counts']['moving_targets']==2 for a in summary['arms'])
assert load('browser-verification.json')['passed']
before=load('server-before.json');assert before==load('server-after.json')
assert {k:v for k,v in before.items() if k!='media_marker'}=={k:v for k,v in load('server-restored.json').items() if k!='media_marker'}
result=dict(passed=True,requests=8,warmups=4,valid=sum(r['valid'] for r in main),checks=['frozen source and input hashes','only declared prompt/schema changes from parent','same image inputs and backend payloads','official image grid','measurement and review coverage','interactive HTML','VLM restored'],artifacts_sha256={n:sha(out/n) for n in ['jobs.json','measurements.jsonl','semantic-review.json','summary.json']})
(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,indent=2))
