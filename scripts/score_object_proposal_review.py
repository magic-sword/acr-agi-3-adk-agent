"""Score explicit assistant annotations; never infer semantic matches from gold."""
import argparse
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_target_binding import save, sha
from scripts.benchmark_object_proposals import stats


def score(out):
    cases={c['name']:c for c in json.loads((out/'cases.json').read_text())}
    rows=[json.loads(l) for l in (out/'responses.jsonl').read_text().splitlines()]
    review=json.loads((out/'semantic-review.json').read_text())
    assert review['responses_sha256']==sha(out/'responses.jsonl')
    annotations={r['id']:r for r in review['annotations']}
    rr=[r for r in rows if r['arm']=='concepts']
    assert set(annotations)=={r['id'] for r in rr}
    tasks=[]
    for r in rr:
        a=annotations[r['id']];c=cases[r['case']]
        assert len(a['matches'])==(len(r['objects']) if r['valid'] else 0)
        assert set(a['matches'])-{None}<={g['id'] for g in c['gold']}
        matched=set(a['matches'])-{None}
        tp=len(matched);predicted=len(a['matches']);gold=len(c['gold'])
        tasks.append(dict(id=r['id'],case=c['name'],repeat=r['repeat'],group=c['group'],
                          fully_annotated=c['fully_annotated'],valid=r['valid'],
                          tp=tp,predicted=predicted,gold=gold,matched=sorted(matched),
                          complete=bool(r['valid'] and tp==gold==predicted)))
    results=[]
    for rep in (0,1):
        for subset in ('synthetic','real'):
            tt=[t for t in tasks if t['repeat']==rep and t['fully_annotated']==(subset=='synthetic')]
            s=stats(sum(t['tp'] for t in tt),sum(t['predicted'] for t in tt),sum(t['gold'] for t in tt))
            if subset=='real':s['precision']=None;s['f1']=None
            results.append(dict(repeat=rep,subset=subset,score=s,valid=sum(t['valid'] for t in tt),total=len(tt),
                 complete_scenes=sum(t['complete'] for t in tt) if subset=='synthetic' else None,
                 by_group={g:stats(sum(t['tp'] for t in tt if t['group']==g),sum(t['predicted'] for t in tt if t['group']==g),sum(t['gold'] for t in tt if t['group']==g)) for g in sorted({t['group'] for t in tt})} ))
    # Partial real annotations never support precision, including the group view.
    for r in results:
        if r['subset']=='real':
            for v in r['by_group'].values():v['precision']=None;v['f1']=None
    save(out/'semantic-summary.json',dict(results=results,tasks=tasks,review_sha256=sha(out/'semantic-review.json')))
    page=['<!doctype html><meta charset="utf-8"><title>Concept proposal review</title><style>body{font:16px sans-serif;margin:24px;max-width:1200px}img{width:300px;image-rendering:pixelated}article{border-top:1px solid #aaa;padding:16px}pre{white-space:pre-wrap}.ok{color:#176631}.bad{color:#a32020}</style><h1>Concept proposals: explicit review</h1><p>First repeat only. Coarse position and whole-object appearance; numerical box accuracy is evaluated separately. Implementing-assistant review, not independent or fully blinded. Real frames are partially annotated, so unmatched entries need not be false objects.</p>']
    for name,c in cases.items():
        r=next(r for r in rr if r['case']==name and r['repeat']==0);a=annotations[r['id']]
        t=next(t for t in tasks if t['id']==r['id'])
        page.append('<article><h2>'+name+'</h2><img src="'+name+'.png"><p>Program whole-object recall: '+str(len(c['program_exact_matches']))+'/'+str(len(c['gold']))+'; Qwen concept recall: '+str(t['tp'])+'/'+str(t['gold'])+'</p><ol>')
        seen=set()
        for obj,match in zip(r.get('objects',[]),a['matches']):
            good=match is not None and match not in seen;seen.add(match)
            page.append('<li class="'+('ok' if good else 'bad')+'">'+html.escape(obj['description']+' | '+obj['location']+' → '+str(match)+(' (duplicate)' if match and not good else ''))+'</li>')
        page.append('</ol><p>'+html.escape(a['reason'])+'</p><details><summary>Frozen gold</summary><pre>'+html.escape(json.dumps([{k:v for k,v in g.items() if k!='support'} for g in c['gold']],indent=2))+'</pre></details></article>')
    (out/'semantic-review.html').write_text('\n'.join(page))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();score(a.output)
