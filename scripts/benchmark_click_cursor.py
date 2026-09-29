"""Isolated historical cursor replay on fixed visual queries; no game calls."""
import ast
import importlib.util
import json
import random
import time
import statistics
from pathlib import Path
from scripts import benchmark_click_choices as b
OUT=Path('outputs/click-cursor-20260930')

def legacy():
    spec=importlib.util.spec_from_file_location('historic_cursor',OUT/'sources/cursor.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tree=ast.parse((OUT/'sources/tasks.py').read_text())
    instructions=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INSTRUCTIONS' for t in n.targets))
    return module,instructions


def run():
    assert not (OUT/'responses.jsonl').exists()
    cursor,instructions=legacy();base='http://vlm:8080'
    cases=json.loads(Path('outputs/click-choices-20260929-v2/cases.json').read_text())
    b.save(OUT/'cases.json',cases)
    (OUT/'sources/benchmark_click_cursor.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'sources/rendering.py').write_bytes(Path('agent/rendering.py').read_bytes())
    b.save(OUT/'plan.json',dict(commit='46ac1b2',cases_hash=b.digest(cases),sources={p.name:b.sha(p) for p in (OUT/'sources').iterdir() if p.is_file()},
        seconds_limit=45,call_limit=64,orders=1,
        protocol='Historical cursor functions and aim instructions unchanged. Isolated context with correct target query; no planner, scene hypotheses, measured inventory or memory. '
        'Quadrant locate then movement/stride/confirmation. Label 8 ends as reconsider, without replanning. '
        '45 seconds or 64 calls experimental cap; no early repeated-state stop. cache_prompt=True and logprobs as historical fast_choice. '
        'One run per case, fixed seed schedule. Only confirmed pixel inside frozen annotation counts hit; no region-mask score.'))
    b.save(OUT/'server-before.json',b.http(base,'/props'))
    audit={s:b.http(base,'/tokenize',dict(content=s,add_special=False)) for s in '12345678'}
    assert all(len(x['tokens'])==1 for x in audit.values());b.save(OUT/'token-audit.json',audit)
    random.Random(b.SEED).shuffle(cases);results=[];started=time.monotonic()
    with (OUT/'requests.jsonl').open('w') as rq,(OUT/'responses.jsonl').open('w') as rp:
        for index,c in enumerate(cases):
            obs=dict(width=64,height=64,observation_id=c['name'],grid=c['grid'])
            cur=cursor.start_cursor(obs,c['query']);begin=time.monotonic();seen=set();repeats=0;trace=[];reason='call_limit';confirmed=False
            while len(trace)<64:
                left=45-(time.monotonic()-begin)
                if left<=0:reason='time_limit';break
                state=tuple(cur[k] for k in ('mode','x','y','stride'))
                repeats+=state in seen;seen.add(state)
                options=cursor.cursor_options(cur,obs);options['8']=dict(kind='reconsider',meaning='Unexpected result or uncertainty: reconcile')
                context=dict(work='aim',observation_id=c['name'],target_query=c['query'],cursor=dict(cur),
                             image_size=dict(width=64,height=64),choices=options)
                parts=[dict(type='text',text='CURRENT board'),b.image_part(b.render_current(c['grid']))]
                if cur['mode']=='adjust':parts+=cursor.cursor_parts(obs,cur)
                parts.append(dict(type='text',text=json.dumps(context,separators=(',',':'))))
                body=dict(model='qwen3-vl-4b-instruct',messages=[dict(role='system',content=instructions['aim_locate' if cur['mode']=='locate' else 'aim']),dict(role='user',content=parts)],
                          temperature=0,max_tokens=1,stream=False,cache_prompt=True,logprobs=True,top_logprobs=20,
                          grammar='root ::= '+' | '.join(json.dumps(k) for k in options))
                ident=c['name']+'/'+str(len(trace));rq.write(json.dumps(dict(id=ident,payload=body,digest=b.digest(body)))+'\n');rq.flush()
                row=dict(id=ident,cursor=dict(cur),digest=b.digest(body));t=time.monotonic()
                try:
                    response=b.http(base,'/v1/chat/completions',body);label=response['choices'][0]['message']['content']
                    row.update(response=response,answer=label);assert label in options
                except Exception as exc:row['error']=str(exc);label=None
                row['seconds']=time.monotonic()-t;trace.append(row);rp.write(json.dumps(row)+'\n');rp.flush()
                if time.monotonic()-begin>=45:reason='time_limit';break
                if label is None:reason='invalid';break
                if label=='8':reason='reconsider';break
                option=options[label]
                if option['kind']=='click_cursor':confirmed=True;cur['confirmed']=True;reason='confirmed';break
                cursor.adjust_cursor(cur,option)
            gold=b.decode(c['gold_runs']);point=(cur['x'],cur['y'])
            results.append(dict(case=c['name'],origin=c['origin'],positive=bool(gold),reason=reason,confirmed=confirmed,
                point=point,hit=confirmed and point in gold,false_click=confirmed and not gold,
                correct_reconsider=not gold and reason=='reconsider',calls=len(trace),adjustments=cur['adjustments'],
                repeated_states=repeats,seconds=time.monotonic()-begin))
            b.save(OUT/'scored.json',results)
            print(index+1,'/26',c['name'],reason,point,'hit',results[-1]['hit'],'calls',len(trace),round(time.monotonic()-started,1),flush=True)
    b.save(OUT/'server-after.json',b.http(base,'/props'))
    assert json.loads((OUT/'server-before.json').read_text())==json.loads((OUT/'server-after.json').read_text())
    summary={}
    for origin in ('all','real','synthetic'):
        rs=[r for r in results if origin=='all' or r['origin']==origin];pos=[r for r in rs if r['positive']];neg=[r for r in rs if not r['positive']]
        summary[origin]=dict(positive=len(pos),negative=len(neg),hit=sum(r['hit'] for r in pos),confirmed=sum(r['confirmed'] for r in pos),
            correct_reconsider=sum(r['correct_reconsider'] for r in neg),false_click=sum(r['false_click'] for r in neg),
            median_positive_seconds=statistics.median(r['seconds'] for r in pos),median_calls=statistics.median(r['calls'] for r in rs),
            repeated_cases=sum(r['repeated_states']>0 for r in rs),limits=sum(r['reason'] in ('call_limit','time_limit') for r in rs),
            calls=sum(r['calls'] for r in rs))
    b.save(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
