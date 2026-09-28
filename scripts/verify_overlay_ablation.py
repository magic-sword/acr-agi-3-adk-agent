"""Check paired inputs and completed output integrity for the overlay ablation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image,ImageColor

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import digest,read_lines
from scripts.summarize_temporal_overlay import audit_key

def verify(out):
    load=lambda n:json.loads((out/n).read_text())
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    plan=load('plan.json');jobs=load('jobs.json');cases=load('cases.json');cats=load('catalogs.json')
    assert digest(jobs)==plan['jobs_digest'] and digest(cases)==plan['cases_digest'] and digest(cats)==plan['catalog_digest']
    assert len(jobs)==132 and len(cases)==11 and len(cats)==33
    for f,h in plan['source_hashes'].items():assert sha(Path(f))==h==sha(out/'sources'/Path(f).name)
    for f,h in load('input-manifest.json').items():assert sha(Path(f))==h
    old=Path(plan['comparison_source']);previous=json.loads((old/'jobs.json').read_text())
    before=json.loads((old/'server-before.json').read_text())
    assert before==load('server-before.json')==load('server-after.json')
    template=Image.open('outputs/action-cue-20260927/ui-template.png').convert('RGB')
    for c in cases:
        for mode in ['before','after','outline','outline_bright','after_edges']:
            im=Image.open(out/f'{c["name"]}-{mode}.png').convert('RGB');assert im.size==(640,820)
            ui=im.copy();ui.paste(template.crop((59,80,571,592)),(59,80));assert ui.tobytes()==template.tobytes()
            if mode in ('before','after'):
                raw=Image.new('RGB',(64,64));raw.putdata([ImageColor.getrgb(c['palette'][str(x)]) for row in c[mode] for x in row])
                assert raw.resize((512,512),getattr(Image,'Resampling',Image).NEAREST).tobytes()==im.crop((59,80,571,592)).tobytes()
    for arm in ('outline_bright','after_edges'):
        assert Image.open(out/f'synthetic_static-{arm}.png').tobytes()==Image.open(out/'synthetic_static-after.png').tobytes()
    for j in jobs:
        p=j['payload'];assert digest(p)==j['payload_digest']
        ref=next(x for x in previous if x['catalog_key']==j['catalog_key'] and x['arm']==('repeat' if j['arm']=='repeat' else 'outline'))['payload']
        if j['arm'] in ('repeat','outline'):assert p==ref
        else:
            content=p['messages'][1]['content'];common=ref['messages'][1]['content']
            assert content[:4]==common[:4] and content[6:]==common[6:]
            assert sum(x['type']=='image_url' for x in content)==3
            pp=json.loads(json.dumps(p));pp['messages'][1]['content']=common;assert pp==ref
    rows=read_lines(out/'measurements.jsonl');main=[r for r in rows if not r['warmup']]
    assert len(rows)==136 and len(main)==132 and sorted(r['id'] for r in main)==list(range(132))
    assert Counter(r['arm'] for r in main)==dict.fromkeys(plan['arms'],33)
    for r in rows:assert r['payload_digest']==jobs[r['id']]['payload_digest']
    reviews=load('semantic-review.json')['rows']
    assert len(reviews)==len({r['key'] for r in reviews})
    assert {r['key'] for r in reviews}=={audit_key(r) for r in main}
    result=load('summary.json');assert result['main_requests']==132
    for a in result['arms']:assert a['counts']['motion_n']==18 and a['counts']['moving_targets']==21
    oldrows={(r['case'],r['repeat'],r['arm']):r for r in read_lines(old/'measurements.jsonl') if not r['warmup']}
    replay=[r for r in main if r['arm'] in ('repeat','outline')]
    match=sum(r.get('answer')==oldrows[r['case'],r['repeat'],r['arm']].get('answer') for r in replay)
    result={'passed':True,'requests':len(main),'valid':sum(r['valid'] and r['within_budget'] for r in main),
        'max_seconds':max(r['seconds'] for r in rows),'baseline_replay_identical_answers':match,'baseline_replay_requests':len(replay),
        'checks':['frozen input/source hashes','identical raw board RGB and UI','same originals, main prompt, inventories and generation','no-change overlays equal raw AFTER','all requests present exactly once','semantic review coverage','server settings match previous experiment'],
        'artifacts_sha256':{n:sha(out/n) for n in ['jobs.json','measurements.jsonl','semantic-review.json','summary.json']}}
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();verify(a.output)
