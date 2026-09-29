"""Frozen-request input interventions; no game actions and no attention-weight claims."""
import argparse
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('baseline_start', 'remove_plan_prose', 'remove_knowledge',
            'factual_negative_knowledge', 'remove_images', 'remove_measurements',
            'remove_plan_prose_and_factual_knowledge', 'baseline_end')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def context_part(payload):
    for message in reversed(payload['messages']):
        if not isinstance(message['content'],list):
            continue
        for part in reversed(message['content']):
            if part['type']=='text' and part['text'].startswith('{'):
                value=json.loads(part['text'])
                if value.get('work')=='ground':
                    return part,value
    raise ValueError('ground context missing')


def perturb(original, variant):
    payload=deepcopy(original)
    if variant.startswith('baseline'):
        return payload
    part,c=context_part(payload)
    if variant in ('remove_plan_prose','remove_plan_prose_and_factual_knowledge'):
        trial=c['last_completed_trial']
        for key in ('baseline','expected_effect','targets','planned_action'):
            trial.pop(key,None)
        trial['actual_result'].pop('prediction',None)
    if variant=='remove_knowledge':
        c['long_term_knowledge']=[]
    if variant in ('factual_negative_knowledge','remove_plan_prose_and_factual_knowledge'):
        for k in c['long_term_knowledge']:
            k.update(claim='The tested click produced no visible pixel change in the observed interval. '
                           'This does not establish that this target can never respond.',
                     conditions='The same observed board and tested click; other conditions are untested.',
                     status='supported')
    if variant=='remove_measurements':
        c['current_observation'].pop('measured_objects',None)
        c['measurements']={}
    part['text']=json.dumps(c,separators=(',',':'))
    if variant=='remove_images':
        for m in payload['messages']:
            if isinstance(m['content'],list):
                m['content']=[p for p in m['content'] if p['type']!='image_url']
    assert payload['tools']==original['tools']
    assert payload['messages'][0]==original['messages'][0]
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/memory-comparison-20260929'
    cases=[]
    for name in ('r1-ft09-separated','r2-ft09-separated'):
        log=next((source/name).glob('*/cognition/*.requests.jsonl'))
        row=next(r for r in map(json.loads,log.read_text().splitlines()) if r['work']=='ground' and r['step']==2)
        payload=deepcopy(row['request'])
        for m in payload['messages']:
            if not isinstance(m['content'],list):
                continue
            for p in m['content']:
                if p['type']!='image_url':
                    continue
                info=p['image_url'];data=(log.parent/info['path']).read_bytes()
                assert hashlib.sha256(data).hexdigest()==info['sha256']
                p['image_url']={'url':f"data:{info['mime_type']};base64,"+base64.b64encode(data).decode()}
                if info.get('detail') is not None:
                    p['image_url']['detail']=info['detail']
        assert digest(payload)==row['request_sha256'], 'baseline must reproduce exact original request'
        cases.append((name,payload,row['request_sha256']))
    jobs=[(name,v,perturb(payload,v)) for name,payload,_ in cases for v in VARIANTS]
    (out/'plan.json').write_text(json.dumps(dict(
        cases=[dict(name=n,source_request_sha256=h) for n,_,h in cases],
        variants=list(VARIANTS),requests=len(jobs),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixed='Original model, temperature, prompt, schema, observation ID, remaining budget text; baseline byte-equivalent JSON digest.',
        measurement='Generated target query, candidate references, expected effect and rationale. No click execution or spatial attention weights.',
        limitations='Two selected failure contexts, not independent samples. Removing evidence can be out of distribution. A changed output shows local input sensitivity, not a uniquely identified internal cause. Negative rewrite is a representation intervention, not a new game observation.'),indent=2)+'\n')
    api=os.getenv('VLM_API_BASE','http://vlm:8080/v1').rstrip('/')
    with urllib.request.urlopen(api.removesuffix('/v1')+'/props',timeout=15) as response:
        (out/'server-props.json').write_text(json.dumps(json.load(response),indent=2)+'\n')
    results=[]
    for name,variant,payload in jobs:
        print('START',name,variant,flush=True)
        request_path=out/f'{name}-{variant}.request.json'
        request_path.write_text(json.dumps(payload,ensure_ascii=False)+'\n')
        started=time.monotonic()
        result=dict(case=name,variant=variant,request_sha256=digest(payload))
        try:
            request=urllib.request.Request(api+'/chat/completions',data=json.dumps(payload).encode(),
                headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=60) as response:
                answer=json.load(response)
            result['response']=answer
            tool=answer['choices'][0]['message']['tool_calls'][0]
            result['plan']=json.loads(tool['function']['arguments'])
            print('TARGET',result['plan'].get('action'),flush=True)
        except Exception as exc:
            result['error']=f'{type(exc).__name__}: {exc}'
            print(result['error'],flush=True)
        result['seconds']=time.monotonic()-started
        results.append(result)
        (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('Results:',out/'results.json',flush=True)


if __name__=='__main__':
    main()
