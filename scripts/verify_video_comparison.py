"""Verify frame sequences, matched requests, and the completed video comparison."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import digest,read_lines
from scripts.summarize_temporal_overlay import audit_key

def verify(out):
    load=lambda n:json.loads((out/n).read_text())
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    plan=load('plan.json');jobs=load('jobs.json');cats=load('catalogs.json');cases=load('cases.json');clips=load('video-manifest.json')
    assert len(jobs)==99 and len(cats)==33 and len(cases)==11 and len(clips)==33
    assert digest(jobs)==plan['jobs_digest'] and digest(cats)==plan['catalog_digest'] and digest(cases)==plan['cases_digest']
    for f,h in plan['source_hashes'].items():assert sha(Path(f))==h==sha(out/'sources'/Path(f).name)
    for f,h in load('input-manifest.json').items():assert sha(Path(f))==h
    source=Path(plan['comparison_source'])
    assert json.loads((source/'server-before.json').read_text())==load('server-before.json')==load('server-after.json')
    for clip in clips:
        assert clip['frames']==4 and clip['fps']==4 and clip['size']==[640,820] and clip['lossless_decode_equal']
        name=clip['case'];assert sha(out/f'{name}-{clip["arm"]}.mkv')==clip['video_sha256']
        raw=[Image.open(out/f'{name}-{fr}.png').convert('RGB').tobytes() for fr in clip['sequence']]
        assert [hashlib.sha256(x).hexdigest() for x in raw]==clip['frame_sha256']
        assert hashlib.sha256(b''.join(raw)).hexdigest()==clip['decoded_rgb_sha256']
        assert clip['sequence'][0]=='before' and clip['sequence'][-1]=='after'
    for c in cases:
        for fr in ['before','after']:assert sha(out/f'{c["name"]}-{fr}.png')==sha(source/f'{c["name"]}-{fr}.png')
        before=Image.open(out/f'{c["name"]}-before.png').convert('RGB');after=Image.open(out/f'{c["name"]}-after.png').convert('RGB')
        for fr,alpha in [('mix1',1/3),('mix2',2/3)]:
            mid=Image.open(out/f'{c["name"]}-{fr}.png').convert('RGB')
            expected=bytes(int(x*(1-alpha)+y*alpha+0.5) for x,y in zip(before.tobytes(),after.tobytes()))
            assert mid.tobytes()==expected
    for cat in cats:
        paired=[j for j in jobs if j['catalog_key']==cat['key']];assert len(paired)==3
        common=paired[0]['payload']
        for j in paired:
            p=j['payload'];assert digest(p)==j['payload_digest']
            content=p['messages'][1]['content'];assert len(content)==3 and content[1]['type']=='input_video'
            assert content[2]==common['messages'][1]['content'][2]
            clone=json.loads(json.dumps(p));clone['messages'][1]['content']=common['messages'][1]['content'];assert clone==common
            assert base64.b64decode(content[1]['input_video']['data'].split(',',1)[1])==(out/f'{j["case"]}-{j["arm"]}.mkv').read_bytes()
    rows=read_lines(out/'measurements.jsonl');main=[r for r in rows if not r['warmup']]
    assert len(rows)==102 and len(main)==99 and sorted(r['id'] for r in main)==list(range(99))
    assert Counter(r['arm'] for r in main)==dict.fromkeys(plan['arms'],33)
    for r in rows:assert r['payload_digest']==jobs[r['id']]['payload_digest']
    review=load('semantic-review.json')['rows'];assert len(review)==len({r['key'] for r in review})
    assert {r['key'] for r in review}=={audit_key(r) for r in main}
    summary=load('summary.json');assert summary['warmups']==3 and summary['main_requests']==99
    for arm in summary['arms']:assert arm['counts']['moving_targets']==21 and arm['counts']['motion_n']==18
    result={'passed':True,'clips':33,'decoded_frames':132,'main_requests':99,'warmups':3,
        'main_valid':sum(r['valid'] and r['within_budget'] for r in main),'max_seconds':max(r['seconds'] for r in rows),
        'checks':['input/source hashes','exact endpoints from previous benchmark','exact intermediate RGB mixture','four lossless decoded RGB frames per video','same main question and generation across arms','video payload matches verified file','request and semantic-review coverage','server settings match previous experiment'],
        'limitations':'Decoder validation uses the runtime-equivalent ffmpeg filter; internal vision features are not directly captured. Pinned llama.cpp video processing is not claimed equivalent to official Transformers.',
        'artifacts_sha256':{n:sha(out/n) for n in ['jobs.json','video-manifest.json','measurements.jsonl','semantic-review.json','summary.json']}}
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();verify(a.output)
