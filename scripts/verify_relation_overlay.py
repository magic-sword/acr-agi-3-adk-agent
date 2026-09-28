"""Verify fixed inputs and the two independent factors of the experiment."""
import json,sys
from pathlib import Path
from collections import Counter
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_relation_overlay import validate,METHODS,EVIDENCE
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_recognition import encode
from scripts.benchmark_official_relations import sha
from scripts.summarize_temporal_overlay import audit_key

out=Path(sys.argv[1]);load=lambda n:json.loads((out/n).read_text());jobs=validate(out);plan=load('plan.json');cats=load('catalogs.json')
assert len(jobs)==56 and digest(cats)==plan['catalog_digest'] and digest(load('cases.json'))==plan['cases_digest']
for cat in cats:
    pair={j['arm']:j for j in jobs if j['catalog_key']==cat['key']}
    for v in METHODS:assert pair['llama_'+v]['payload']==pair['official_'+v]['payload']
    for image in ['repeat','edges']:
        a=json.loads(json.dumps(pair['llama_relation_'+image]['payload']))
        a['messages'][1]['content'][-1]['text']=a['messages'][1]['content'][-1]['text'].replace(EVIDENCE['relation'],EVIDENCE['appearance'])
        assert a==pair['llama_appearance_'+image]['payload']
    for prompt in ['appearance','relation']:
        a=json.loads(json.dumps(pair['llama_'+prompt+'_edges']['payload']))
        a['messages'][1]['content'][5]=pair['llama_'+prompt+'_repeat']['payload']['messages'][1]['content'][5]
        assert a==pair['llama_'+prompt+'_repeat']['payload']
    for j in pair.values():
        assert digest(j['payload'])==j['payload_digest'] and j['payload']['max_tokens']==1024
        assert j['payload']['temperature']==0 and 'response_format' not in j['payload']
        for index,path in zip([1,3,5],j['paths']):
            im=Image.open(out/path).convert('RGB');assert im.size==(640,820)
            assert encode(im)==j['payload']['messages'][1]['content'][index]
    if cat['case'] in ['synthetic_static','color_only']:
        assert Image.open(out/f'{cat["case"]}-after.png').tobytes()==Image.open(out/f'{cat["case"]}-after_edges.png').tobytes()
        for prompt in ['appearance','relation']:assert pair['llama_'+prompt+'_repeat']['payload']==pair['llama_'+prompt+'_edges']['payload']
rows=read_lines(out/'measurements.jsonl');main=[r for r in rows if not r['warmup']]
assert len(rows)==64 and len(main)==56 and sorted(r['id'] for r in main)==list(range(56))
assert Counter(r['arm'] for r in main)==dict.fromkeys(plan['arms'],7)
for r in rows:
    assert r['payload_digest']==jobs[r['id']]['payload_digest'] and 'error' not in r
    assert r['output_tokens']<=1024
    if r['backend']=='official':assert r['image_grid_thw']==[[1,52,40]]*3
for backend in ['llama','official']:
    for case in ['synthetic_static','color_only']:
        for prompt in ['appearance','relation']:
            rs=[r for r in main if r['backend']==backend and r['case']==case and r['variant'].startswith(prompt+'_')]
            assert len(rs)==2 and rs[0]['answer']==rs[1]['answer']
reviews=load('semantic-review.json')['rows'];assert {r['key'] for r in reviews}=={audit_key(r) for r in main}
s=load('summary.json');assert s['main_requests']==56 and s['warmups']==8
for a in s['arms']:assert a['counts']['moving_targets']==5
assert load('browser-verification.json')['passed']
before=load('server-before.json');after=load('server-after.json');restored=load('server-restored.json');assert before==after
assert {k:v for k,v in before.items() if k!='media_marker'}=={k:v for k,v in restored.items() if k!='media_marker'}
result=dict(passed=True,requests=56,warmups=8,valid=sum(r['valid'] for r in main),checks=['input and source hashes','independent prompt/image factors','same image count and dimensions','identical requests across backends','static/color-only no-op image and output controls','official image grid','full measurement and semantic review coverage','interactive HTML','original VLM settings restored'],artifacts_sha256={n:sha(out/n) for n in ['jobs.json','measurements.jsonl','semantic-review.json','summary.json']})
(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,indent=2))
