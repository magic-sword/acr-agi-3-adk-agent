"""Bounded, exhaustive page selection experiment; no gameplay or oracle retrieval."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse
from scripts import benchmark_click_choices as b

ARMS = ('free_union', 'pages_text', 'pages_visual')


def pages(candidates, order):
    ordered = list(candidates)
    random.Random(b.SEED + order).shuffle(ordered)
    return [ordered[i:i+8] for i in range(0, len(ordered), 8)]


def prepare(out, source):
    cases = json.loads((source/'cases.json').read_text())
    assert b.digest(cases) == json.loads((source/'plan.json').read_text())['cases_digest']
    out.mkdir(parents=True, exist_ok=False)
    b.save(out/'cases.json', cases)
    sources = {}
    for path in (Path(__file__).resolve(), Path(b.__file__).resolve()):
        (out/path.name).write_bytes(path.read_bytes())
        sources[path.name] = b.sha(path)
    b.save(out/'plan.json', dict(source=str(source), cases_digest=b.digest(cases), sources=sources,
        arms=ARMS, orders=2, page_size=8, seed=b.SEED,
        protocol='All candidates offered exactly once at first round; one or X per page. '
        'Page winners enter further pages until at most one winner. No retries, localization, '
        'gold filtering or cursor refinement. Even singleton pages call the model. '
        'Same prompts as previous experiment. Primary order=0; order=1 sensitivity. '
        'Baseline rerun contemporaneously. Failure of any call invalidates that case/arm.',
        scoring='Same frozen gold and 80% recall/purity rule; centroid and interior hit; negatives separate.'))


def run(out, base):
    assert urlparse(base).hostname in ('localhost', '127.0.0.1', 'vlm')
    plan=json.loads((out/'plan.json').read_text()); cases=json.loads((out/'cases.json').read_text())
    assert b.digest(cases)==plan['cases_digest']
    for path in (Path(__file__).resolve(), Path(b.__file__).resolve()):
        assert b.sha(path)==plan['sources'][path.name]
    assert not (out/'responses.jsonl').exists()
    b.save(out/'server-before.json', b.http(base,'/props'))
    audit={v:b.http(base,'/tokenize',dict(content=v,add_special=False)) for v in '12345678X'}
    assert all(len(x['tokens'])==1 for x in audit.values())
    b.save(out/'token-audit.json',audit)
    jobs=[(c,p) for c in cases for p in range(2)]; random.Random(b.SEED).shuffle(jobs)
    results=[]; start=time.monotonic()
    with (out/'requests.jsonl').open('w') as req, (out/'responses.jsonl').open('w') as resp:
        for index,(c,p) in enumerate(jobs):
            arms=list(ARMS);random.Random(b.SEED+index).shuffle(arms)
            for arm in arms:
                calls=[]; rounds=[]; valid=True; selected=[]; begin=time.monotonic()
                def invoke(options, round_id, page_id):
                    nonlocal valid
                    kind='free_union' if arm=='free_union' else ('choice_visual' if arm=='pages_visual' else 'choice_text')
                    body,mapping=b.request_for(c['grid'],c['query'],options,kind,p)
                    ident=f'{c["name"]}/{p}/{arm}/{round_id}/{page_id}'
                    req.write(json.dumps(dict(id=ident,payload=body,digest=b.digest(body)))+'\n');req.flush()
                    row=dict(id=ident,mapping={k:o['id'] for k,o in mapping.items()},digest=b.digest(body))
                    t=time.monotonic(); chosen=[]
                    try:
                        response=b.http(base,'/v1/chat/completions',body)
                        answer=response['choices'][0]['message']['content'];row.update(response=response,answer=answer)
                        if kind=='free_union': labels=b.decode_union(answer,mapping)
                        else:
                            assert answer=='X' or answer in mapping
                            labels=[] if answer=='X' else [answer]
                        chosen=[mapping[k] for k in labels]
                    except Exception as exc:
                        valid=False;row['error']=f'{type(exc).__name__}: {exc}'
                    row['seconds']=time.monotonic()-t;calls.append(row)
                    resp.write(json.dumps(row)+'\n');resp.flush()
                    return chosen
                if arm=='free_union': selected=invoke(c['candidates'],0,0)
                else:
                    current=c['candidates'];round_id=0
                    while current:
                        batches=pages(current,p+round_id);winners=[]
                        for page_id,options in enumerate(batches): winners+=invoke(options,round_id,page_id)
                        rounds.append(dict(offered=[o['id'] for o in current],winners=[o['id'] for o in winners]))
                        if len(winners)<=1: selected=winners;break
                        assert len(winners)<len(current)
                        current=winners;round_id+=1
                gold=b.decode(c['gold_runs']);support=set().union(*(b.decode(o['mask_runs']) for o in selected))
                score=b.region_score(support,gold)
                results.append(dict(case=c['name'],origin=c['origin'],order=p,arm=arm,positive=bool(gold),valid=valid,
                    selected=[o['id'] for o in selected],region_correct=valid and score['correct'],
                    centroid_hit=valid and b.point_for(support,'centroid') in gold,
                    interior_hit=valid and b.point_for(support,'interior') in gold,
                    abstention_correct=valid and not gold and not selected,false_click=valid and not gold and bool(selected),
                    rounds=rounds,calls=len(calls),api_seconds=sum(r['seconds'] for r in calls),
                    wall_seconds=time.monotonic()-begin,
                    first_round_survives=any(b.region_score(b.decode(o['mask_runs']),gold)['correct'] for o in c['candidates'] if rounds and o['id'] in rounds[0]['winners'])))
                b.save(out/'scored.json',results)
            print(index+1,'/',len(jobs),round(time.monotonic()-start,1),'seconds',flush=True)
    b.save(out/'server-after.json',b.http(base,'/props'))
    assert json.loads((out/'server-before.json').read_text())==json.loads((out/'server-after.json').read_text())
    analyze(out)


def analyze(out):
    rows=json.loads((out/'scored.json').read_text());assert len(rows)==156
    requests={r['id']:r for r in b.read_rows(out/'requests.jsonl')}
    responses=b.read_rows(out/'responses.jsonl')
    assert len(requests)==len(responses)==sum(r['calls'] for r in rows)
    for r in responses: assert r['digest']==b.digest(requests[r['id']]['payload'])
    def stats(rs):
        pos=[r for r in rs if r['positive']];neg=[r for r in rs if not r['positive']]
        return dict(positive=len(pos),negative=len(neg),region=sum(r['region_correct'] for r in pos),
            centroid=sum(r['centroid_hit'] for r in pos),interior=sum(r['interior_hit'] for r in pos),
            abstentions=sum(r['abstention_correct'] for r in neg),false_clicks=sum(r['false_click'] for r in neg),
            valid=sum(r['valid'] for r in rs),first_round_survives=sum(r['first_round_survives'] for r in pos),
            median_seconds=statistics.median(r['wall_seconds'] for r in rs),
            median_positive_seconds=statistics.median(r['wall_seconds'] for r in pos),
            mean_calls=statistics.mean(r['calls'] for r in rs),max_calls=max(r['calls'] for r in rs))
    summary={a:{name:stats([r for r in rows if r['arm']==a and pred(r)]) for name,pred in
        [('primary',lambda r:r['order']==0),('real',lambda r:r['order']==0 and r['origin']=='real'),
         ('synthetic',lambda r:r['order']==0 and r['origin']=='synthetic'),('all_orders',lambda r:True)]} for a in ARMS}
    b.save(out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','analyze'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('outputs/click-choices-20260929-v2'))
    parser.add_argument('--base',default='http://vlm:8080');args=parser.parse_args()
    if args.command=='prepare':prepare(args.output,args.source)
    elif args.command=='run':run(args.output,args.base)
    else:analyze(args.output)
