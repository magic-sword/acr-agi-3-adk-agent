"""Analyze preserved EdgeBoxes outputs allowing observed repeat variation.

The original frozen analyzer rejected nonidentical repeats. This correction
retains its predeclared repeat0 primary result and reports all five repeats.
Inference code, parameters and outputs are unchanged.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_edgeboxes import *

def analyze(out):
    plan=verify(out);cases={c['name']:c for c in json.loads((out/'cases.json').read_text())}
    rows=read_rows(out,'results.jsonl');jobs=json.loads((out/'jobs.json').read_text())
    assert len(rows)==len(jobs)
    assert [{k:r[k] for k in ('case','arm','budget','repeat')} for r in rows]==jobs
    selected=[];repeat_agreement=0
    for name in cases:
        for arm in CONFIGS:
            for budget in BUDGETS:
                rr=[r for r in rows if (r['case'],r['arm'],r['budget'])==(name,arm,budget)]
                assert len(rr)==REPEATS and {r['repeat'] for r in rr}==set(range(REPEATS))
                first=next(r for r in rr if r['repeat']==0)
                for r in rr:repeat_agreement+=int(r['records']==first['records'])
                selected.append(dict(case=name,arm=arm,budget=budget,boxes=first['boxes'],
                    seconds=statistics.median(r['seconds'] for r in rr),
                    stage_medians={k:statistics.median(r['stages'][k] for r in rr) for k in first['stages']},
                    min_seconds=min(r['seconds'] for r in rr),max_seconds=max(r['seconds'] for r in rr)))
    for file in ('sam-results.jsonl','qwen-results.jsonl'):
        for r in read_rows(out,file):selected.append(dict(case=r['case'],arm=r['arm'],budget=128,boxes=r['boxes'],seconds=r['seconds']))
    scored=[]
    for r in selected:
        c=cases[r['case']]
        for cap in ([16,32,64,128] if r['budget']==128 else [1000]):
            boxes=r['boxes'][:cap]
            scored.append(dict(case=r['case'],arm=r['arm'],budget=r['budget'],cap=cap,group=c['group'],
                fully_annotated=c['fully_annotated'],pool=len(r['boxes']),seconds=r['seconds'],
                main=score(boxes,c['gold']),raw=score(boxes,c['gold'],padding=0),cap4=score(boxes,c['gold'],cap=4)))
    summary=[]
    for arm,budget,cap in sorted({(r['arm'],r['budget'],r['cap']) for r in scored}):
        for full in (True,False):
            rr=[r for r in scored if (r['arm'],r['budget'],r['cap'],r['fully_annotated'])==(arm,budget,cap,full)]
            summary.append(dict(arm=arm,budget=budget,cap=cap,dataset='synthetic' if full else 'real_partial',
                tp=sum(r['main']['tp'] for r in rr),gold=sum(r['main']['gold'] for r in rr),
                raw_tp=sum(r['raw']['tp'] for r in rr),cap4_tp=sum(r['cap4']['tp'] for r in rr),
                proposals=sum(r['main']['predicted'] for r in rr),median_seconds=statistics.median(r['seconds'] for r in rr),
                duplicate=sum(r['main']['duplicate_eligible'] for r in rr),mixed=sum(r['main']['mixed_boxes'] for r in rr)))
    save(out/'selected.json',selected);save(out/'scored.json',scored);save(out/'summary.json',summary)
    save(out/'repeat-audit.json',dict(equal_records=repeat_agreement,expected=len(jobs)))
    (out/'review').mkdir()
    page=['<meta charset="utf-8"><title>EdgeBoxes comparison</title><style>body{font:16px sans-serif}section{display:flex;flex-wrap:wrap}figure{margin:6px}img{width:320px}</style>',
          '<h1>Top 128 boxes; edges and gold shown separately</h1>']
    for name,c in cases.items():
        page.append('<h2>'+html.escape(name)+'</h2><section>')
        entries=[('gold',[[v/64*1000 for v in (g['bbox'][0],g['bbox'][1],g['bbox'][2]+1,g['bbox'][3]+1)] for g in c['gold']])]
        entries += [(r['arm'],r['boxes'][:128]) for r in selected if r['case']==name and r['budget']==128]
        for arm,boxes in entries:
            im=Image.open(out/(name+'.png')).convert('RGB');draw=ImageDraw.Draw(im)
            for i,b in enumerate(boxes):
                xy=[v*384/1000 for v in b];draw.rectangle(xy,outline='#ff00ff',width=1)
                draw.text((xy[0],xy[1]),str(i+1),fill='#ff00ff')
            filename='review/'+name+'-'+arm+'.png';im.save(out/filename)
            page.append(f'<figure><figcaption>{arm}: {len(boxes)} boxes</figcaption><img src="{filename}"></figure>')
        page.append(f'<figure><figcaption>Structured edges (fixed scale)</figcaption><img src="edges/{name}.png"></figure></section>')
    (out/'gallery.html').write_text('\n'.join(page))
    print('Analyzed',len(rows),'requests;',len(scored),'scores',flush=True)


def fast_tp(boxes,gold,padding=1,area_cap=9):
    import numpy as np
    from scripts.benchmark_iterative_proposals import assign
    if not boxes:return 0
    bb=np.asarray(boxes,dtype=float)*.064
    bb[:,:2]=np.maximum(0,bb[:,:2]-padding);bb[:,2:]=np.minimum(64,bb[:,2:]+padding)
    areas=np.prod(bb[:,2:]-bb[:,:2],axis=1)
    adj=[[] for _ in boxes]
    for j,g in enumerate(gold):
        pixels=np.asarray(g['support'],dtype=float)
        overlap=np.maximum(0,np.minimum(bb[:,None,2:],pixels[None,:,:]+1)-np.maximum(bb[:,None,:2],pixels[None,:,:]))
        cov=np.prod(overlap,axis=2).sum(axis=1)/len(pixels)
        x1,y1,x2,y2=g['bbox']
        for i in np.flatnonzero((cov>=.9-1e-9)&(areas<=(x2-x1+1)*(y2-y1+1)*area_cap+1e-9)):adj[i].append(j)
    return len(assign(adj))


def repeat_sensitivity(out):
    cases={c['name']:c for c in json.loads((out/'cases.json').read_text())}
    rr=[]
    for r in read_rows(out,'results.jsonl'):
        c=cases[r['case']]
        for cap in ([16,32,64,128] if r['budget']==128 else [1000]):
            boxes=r['boxes'][:cap]
            rr.append(dict(case=r['case'],arm=r['arm'],budget=r['budget'],cap=cap,repeat=r['repeat'],
                fully_annotated=c['fully_annotated'],tp=fast_tp(boxes,c['gold']),
                raw_tp=fast_tp(boxes,c['gold'],padding=0),cap4_tp=fast_tp(boxes,c['gold'],area_cap=4)))
    # Confirm vectorized implementation against the original scorer for all primary rows.
    primary={(r['case'],r['arm'],r['budget'],r['cap']):r for r in json.loads((out/'scored.json').read_text())}
    for r in rr:
        if r['repeat']!=0:continue
        p=primary[r['case'],r['arm'],r['budget'],r['cap']]
        assert (r['tp'],r['raw_tp'],r['cap4_tp'])==(p['main']['tp'],p['raw']['tp'],p['cap4']['tp'])
    summary=[]
    for a,b,cap in sorted({(r['arm'],r['budget'],r['cap']) for r in rr}):
        for full in (True,False):
            per_repeat=[sum(r['tp'] for r in rr if (r['arm'],r['budget'],r['cap'],r['fully_annotated'],r['repeat'])==(a,b,cap,full,k)) for k in range(REPEATS)]
            summary.append(dict(arm=a,budget=b,cap=cap,dataset='synthetic' if full else 'real_partial',
                per_repeat=per_repeat,min_tp=min(per_repeat),max_tp=max(per_repeat),mean_tp=statistics.mean(per_repeat)))
    save(out/'repeat-scores.json',rr);save(out/'repeat-summary.json',summary)
    save(out/'analysis-correction.json',dict(reason='Frozen analyzer assumed byte-identical repeat outputs; observed variation invalidates that assumption. Keep predeclared repeat0 primary; additionally score every repeat. No inference or parameter changes.',
        script=str(Path(__file__)),sha256=sha(Path(__file__))))
    print('Scored repeat sensitivity',len(rr),flush=True)


if __name__=='__main__':
    out=Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/edgeboxes-20260928')
    analyze(out);repeat_sensitivity(out)
