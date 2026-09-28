"""Validate diagnostic interventions and recorded model measurements."""
import json,sys
from pathlib import Path
from PIL import Image,ImageChops
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_xai_motion import CASES,ARMS,IMAGES,sha
from scripts.benchmark_parallel import digest

out=Path(sys.argv[1]);load=lambda n:json.loads((out/n).read_text())
jobs=load('jobs.json');plan=load('plan.json');assert len(jobs)==55 and digest(jobs)==plan['jobs_digest']
assert sha('scripts/benchmark_xai_motion.py')==plan['script_sha256']==sha(out/'sources'/'benchmark_xai_motion.py')
for name,h in load('input-hashes.json').items():assert sha(out/name)==h
for m in load('video-manifest.json'):
    assert m['lossless'] and m['frames']==4 and m['fps']==4
    assert sha(out/f'{m["case"]}-{m["arm"]}.mkv')==m['sha256']
for j in jobs:
    assert digest(j['payload'])==j['payload_digest']
    case,arm=j['case'],j['arm'];roi=j['roi'];control=j['control']
    assert roi[2]-roi[0]==control[2]-control[0] and roi[3]-roi[1]==control[3]-control[1]
    images=[Image.open(out/f'{case}-{arm}-{fr}.png').convert('RGB') for fr in ['before','after']]
    original=[Image.open(IMAGES/f'{case}-{fr}.png').convert('RGB') for fr in ['before','after']]
    for i,image in enumerate(images):
        expected=original[i].copy()
        if arm.startswith(('target_','control_')):
            box=roi if arm.startswith('target_') else control;fill=102 if arm.endswith('gray') else 0
            expected.paste((fill,fill,fill),box)
        elif arm=='ui_gray':
            board=expected.crop((59,80,571,592));expected.paste((51,51,51),(0,0,640,820));expected.paste(board,(59,80))
        elif arm=='restore_target' and i==1:expected.paste(original[0].crop(roi),roi[:2])
        elif arm=='before_only':expected=original[0]
        elif arm=='after_only':expected=original[1]
        assert ImageChops.difference(image,expected).getbbox() is None
    if arm in ['restore_target','before_only','after_only']:assert images[0].tobytes()==images[1].tobytes()
for backend in ['llama','official']:
    rs=[json.loads(s) for s in (out/f'{backend}.jsonl').read_text().splitlines()]
    assert len(rs)==56 and sum(r['warmup'] for r in rs)==1
    assert sorted(r['id'] for r in rs if not r['warmup'])==list(range(55))
    for r in rs:
        assert r['payload_digest']==jobs[r['id']]['payload_digest'] and r['valid']
        obs=r['parsed']['observations']
        assert all(o['kind'] in ['unchanged','moved','shape_changed','separated','resized','color_changed','appeared','disappeared','uncertain'] and isinstance(o['description'],str) for o in obs)
        if not r['arm'].startswith('free_'):assert sorted(o['id'] for o in obs)==list('ABCDEX')
    for c in CASES:
        a=next(r for r in rs if not r['warmup'] and r['case']==c and r['arm']=='restore_target')
        b=next(r for r in rs if not r['warmup'] and r['case']==c and r['arm']=='before_only')
        assert a['answer']==b['answer'],('duplicate control not reproduced',backend,c)
        if backend=='official':assert a['candidate_scores']==b['candidate_scores']
summary=load('summary.json');assert len(summary['outcomes'])==110 and len(summary['attention'])==20 and len(summary['features'])==45
assert load('browser-verification.json')['passed']
assert summary['observed_attention_logit_max_difference']==0
assert all(x['same_answer'] for x in summary['baseline_reproduction'])
assert all(x['max']==0 for x in summary['features'] if x['case']=='synthetic_static')
review=load('semantic-review.json');assert len(review['rows'])==110
assert {(r['backend'],r['id']) for r in review['rows']}=={(r['backend'],r['id']) for r in summary['outcomes']}
before=load('server-before.json');after=load('server-after.json');restored=load('server-restored.json')
assert before==after
assert {k:v for k,v in before.items() if k!='media_marker'}=={k:v for k,v in restored.items() if k!='media_marker'}
result=dict(passed=True,main_requests=110,warmups=2,cases=5,
    checks=['all image interventions and equal-area controls','55 lossless videos','request coverage and input hashes',
        'original ten baseline answers reproduced','identical-image controls reproduced','unchanged visual features exactly equal',
        'attention recording did not alter measured logits','all110 semantic reviews','original VLM settings restored'],
    hashes={n:sha(out/n) for n in ['summary.json','semantic-review.json','llama.jsonl','official.jsonl']})
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
