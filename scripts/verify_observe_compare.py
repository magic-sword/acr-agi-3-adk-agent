"""Verify paired inputs, recorded outputs, review coverage and server restoration."""
import base64,json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_observe_compare import validate,EVIDENCE,MAX_TOKENS
from scripts.benchmark_official_relations import sha
from scripts.benchmark_parallel import digest,read_lines
from scripts.summarize_temporal_overlay import audit_key

out=Path(sys.argv[1]);load=lambda n:json.loads((out/n).read_text())
jobs=validate(out);plan=load('plan.json');cats=load('catalogs.json')
assert len(jobs)==plan['requests'] and digest(cats)==plan['catalog_digest'] and digest(load('cases.json'))==plan['cases_digest']
for cat in cats:
    paired={j['arm']:j for j in jobs if j['catalog_key']==cat['key']};assert set(paired)==set(plan['arms'])
    for m in ['direct','evidence']:assert paired['llama_'+m]['payload']==paired['official_'+m]['payload']
    direct=paired['llama_direct']['payload'];evidence=json.loads(json.dumps(paired['llama_evidence']['payload']))
    text=evidence['messages'][1]['content'][-1]['text']
    text=text.replace(EVIDENCE,'').replace('{"id":"A","before":"visible first-frame state","after":"visible last-frame state","description":"observed agreement or difference","kind":"unchanged"}','{"id":"A","kind":"unchanged","description":"brief visual evidence"}')
    evidence['messages'][1]['content'][-1]['text']=text;assert evidence==direct
    assert direct['max_tokens']==MAX_TOKENS and direct['temperature']==0 and 'response_format' not in direct
    for j in paired.values():
        assert digest(j['payload'])==j['payload_digest']
        clip=base64.b64decode(j['payload']['messages'][1]['content'][1]['input_video']['data'].split(',',1)[1])
        assert clip==(Path('outputs/video-comparison-20260928')/f'{cat["case"]}-grouped.mkv').read_bytes()
rows=read_lines(out/'measurements.jsonl');main=[r for r in rows if not r['warmup']]
assert len(rows)==plan['requests']+4 and len(main)==plan['requests']
assert sorted(r['id'] for r in main)==list(range(plan['requests']))
assert Counter(r['arm'] for r in main)==dict.fromkeys(plan['arms'],len(cats))
for r in rows:
    assert r['payload_digest']==jobs[r['id']]['payload_digest']
    assert r['output_tokens']<=MAX_TOKENS and 'error' not in r
    if r['backend']=='official':assert r['video_grid_thw']==[[2,52,40]]
review=load('semantic-review.json')['rows'];assert len(review)==len({r['key'] for r in review})
assert {r['key'] for r in review}=={audit_key(r) for r in main}
summary=load('summary.json');assert summary['main_requests']==plan['requests']
assert sum(a['counts']['valid'] for a in summary['arms'])==sum(r['valid'] and r['within_budget'] for r in main)
assert load('browser-verification.json')['passed']
before=load('server-before.json');after=load('server-after.json');restored=load('server-restored.json');assert before==after
assert {k:v for k,v in before.items() if k!='media_marker'}=={k:v for k,v in restored.items() if k!='media_marker'}
result=dict(passed=True,requests=len(main),valid=sum(r['valid'] for r in main),warmups=4,post_hoc=plan.get('post_hoc',False),
    checks=['source and input hashes','paired image, inventory, generation parameters and prompts','identical payload across backends','complete request and semantic review coverage','official video grid','equal maximum output tokens','interactive report','original VLM settings restored'],
    artifacts_sha256={n:sha(out/n) for n in ['jobs.json','measurements.jsonl','semantic-review.json','summary.json']})
(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,indent=2))
