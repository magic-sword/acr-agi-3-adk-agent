import json,sys
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_official_relations import sha
from scripts.benchmark_instance_state import request_for
from scripts.instance_state import extract,compare

out=Path(sys.argv[1]);load=lambda n:json.loads((out/n).read_text());plan=load('plan.json');jobs=load('jobs.json');cases=load('cases.json');states=load('states.json')
assert digest(jobs)==plan['jobs_digest'] and digest(cases)==plan['cases_digest'] and digest(states)==plan['states_digest']
for p,h in plan['source_hashes'].items():assert sha(p)==h
for p,h in load('input-hashes.json').items():assert sha(out/p)==h
for c in cases:
    for frame,label in [('before','b'),('after','a')]:
        computed=json.loads(json.dumps(extract(c[frame],label)));assert computed==states[c['name']][frame]
        im=Image.open(out/f'{c["name"]}-{frame}.png').convert('RGB');assert im.size==(592,592)
        for y,row in enumerate(c[frame]):
            for x,color in enumerate(row):
                rgb=tuple(int(c['palette'][str(color)][i:i+2],16) for i in [1,3,5]);assert im.getpixel((48+x*8+4,48+y*8+4))==rgb
for p in load('program-results.json'):assert p['result']==compare(states[p['case']]['before'],states[p['case']]['after'])
rows=read_lines(out/'measurements.jsonl');reqs=read_lines(out/'requests.jsonl');main=[r for r in rows if not r['warmup']]
assert len(rows)==len(reqs)==62 and sorted(r['id'] for r in main)==list(range(60))
observations={(r['case'],r['frame']):r for r in main if r['stage']=='observe'}
for r,q in zip(rows,reqs):
    j=jobs[r['id']];assert r['id']==q['id'] and r['warmup']==q['warmup']
    expected=request_for(j,out,states,observations);assert expected==q['payload'] and digest(expected)==q['payload_digest']==r['payload_digest']
    assert 'response' in r and r['output_tokens']<=expected['max_tokens']
    images=[c for c in expected['messages'][1]['content'] if c['type']=='image_url']
    assert len(images)==(1 if r['stage']=='observe' else 2 if r['arm'] in ['direct_pair','measured_images'] else 0)
for c in cases:
    pair={r['arm']:q['payload'] for r,q in zip(rows,reqs) if not r['warmup'] and r['case']==c['name'] and r['stage']=='judge'}
    assert pair['measured_text']['messages'][1]['content'][-1]==pair['measured_images']['messages'][1]['content'][-1]
review=load('semantic-review.json');assert len(review['rows'])==50 and len({(r['case'],r['arm']) for r in review['rows']})==50
assert load('server-before.json')==load('server-after.json') and load('health-after.json')['status']=='ok'
assert load('browser-verification.json')['passed']
result=dict(passed=True,model_requests=60,warmups=2,independent_observations=20,comparisons=40,program_comparisons=10,checks=['frozen data/code hashes','independent per-frame extraction reproduced','original colors preserved at all rendered pixel centers','reconstructed exact requests and generated-observation provenance','text-only arms have no images','measured text identical with/without images','same inference settings','all comparisons reviewed','interactive HTML','server settings unchanged and health OK'],artifacts_sha256={n:sha(out/n) for n in ['cases.json','states.json','measurements.jsonl','semantic-review.json','summary.json']})
(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,indent=2))
