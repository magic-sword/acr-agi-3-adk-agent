"""Check immutable inputs, paired prompts, official grid, and score coverage."""
import base64
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_official_relations import ARMS, RELATION, SOURCE, sha, validate
from scripts.benchmark_parallel import digest, read_lines
from scripts.summarize_temporal_overlay import audit_key

def verify(out):
    validate(out)
    load=lambda n:json.loads((out/n).read_text())
    plan=load('plan.json');jobs=load('jobs.json');cats=load('catalogs.json')
    assert len(jobs)==132 and len(cats)==33
    assert digest(cats)==plan['catalog_digest'] and digest(load('cases.json'))==plan['cases_digest']
    for cat in cats:
        paired={j['arm']:j for j in jobs if j['catalog_key']==cat['key']}
        assert set(paired)==set(ARMS)
        for suffix in ['base','relation']:
            assert paired['llama_'+suffix]['payload']==paired['official_'+suffix]['payload']
        plain=paired['llama_base']['payload'];relation=json.loads(json.dumps(paired['llama_relation']['payload']))
        relation['messages'][1]['content'][-1]['text']=relation['messages'][1]['content'][-1]['text'].replace(RELATION,'')
        assert relation==plain and 'response_format' not in plain
        assert plain['max_tokens']==512 and plain['temperature']==0
        for j in paired.values():
            assert digest(j['payload'])==j['payload_digest']
            encoded=j['payload']['messages'][1]['content'][1]['input_video']['data'].split(',',1)[1]
            assert base64.b64decode(encoded)==(SOURCE/f'{cat["case"]}-grouped.mkv').read_bytes()
    audit=load('official-processor-audit.json')
    assert audit['grid_thw']==[[2,52,40]] and audit['visual_tokens']==1040
    assert audit['resized_hw']==[832,640] and audit['sampled_frames']==[0,1,2,3]
    assert audit['timestamps']==['<0.1 seconds>','<0.6 seconds>']
    for f,h in audit['source_hashes'].items():assert sha(out/'sources'/f)==h
    native=load('native-processor-audit.json')
    assert native['group_frame_counts']==[1,2,1] and native['grid_xy']==[[20,26]]*3
    assert native['visual_tokens']==1560
    tensor=load('official-tensor-audit.json')
    assert tensor['passed'] and len(tensor['rows'])==11
    assert tensor['script_sha256']==sha('scripts/audit_qwen_video_tensor.py')
    for r in tensor['rows']:
        assert r['first_pair_equal'] and r['last_pair_equal']
        assert (r['changed_spatial_pixels_after_preprocessing']>0)==(r['case']!='synthetic_static')
    rows=read_lines(out/'measurements.jsonl');main=[r for r in rows if not r['warmup']]
    assert len(rows)==136 and len(main)==132
    assert sorted(r['id'] for r in main)==list(range(132))
    assert Counter(r['arm'] for r in main)==dict.fromkeys(ARMS,33)
    for r in rows:
        assert r['payload_digest']==jobs[r['id']]['payload_digest']
        if r['arm'].startswith('official'):assert r['video_grid_thw']==[[2,52,40]]
    reviewed=load('semantic-review.json')['rows']
    assert len(reviewed)==len({r['key'] for r in reviewed})
    assert {r['key'] for r in reviewed}=={audit_key(r) for r in main}
    summary=load('summary.json')
    assert summary['main_requests']==132 and summary['warmups']==4
    for a in summary['arms']:assert a['counts']['moving_targets']==21 and a['counts']['motion_n']==18
    before=load('server-before.json');after=load('server-after.json');restored=load('server-restored.json')
    assert before==after
    # The process generates a fresh media placeholder nonce at startup.
    # Every other property, including all model/generation settings, must match.
    assert {k:v for k,v in before.items() if k!='media_marker'}=={k:v for k,v in restored.items() if k!='media_marker'}
    result={'passed':True,'main_requests':132,'warmups':4,'valid':sum(r['valid'] for r in main),
        'checks':['input and source hashes','paired prompts and identical payload across backends',
            'lossless video from previous verified experiment','official tensor grid and timestamp tokens',
            'native frame grouping in execution log','all frame differences survive official preprocessing',
            'request and semantic-review coverage','original VLM configuration restored except startup media-marker nonce'],
        'limitations':'Backend contrast includes BF16 versus quantized weights, resize, temporal packing, tokenizer and runtime. Not an isolated preprocessing intervention.',
        'artifacts_sha256':{n:sha(out/n) for n in ['jobs.json','measurements.jsonl','semantic-review.json','summary.json']}}
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':verify(Path(sys.argv[1]))
