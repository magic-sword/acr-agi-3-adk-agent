"""Offline report for shared-initial, no-outline mask branches."""
import argparse,html,json,statistics,sys
from collections import Counter
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_batch_masks as m
b=m.b

def analyze(out):
 b.verify(out)
 cases=json.loads((out/'cases.json').read_text());first={r['case']:r for r in json.loads((out/'shared-initial.json').read_text())}
 rows=[json.loads(l) for l in (out/'responses.jsonl').read_text().splitlines()]
 results=[json.loads(l) for l in (out/'results.jsonl').read_text().splitlines()]
 blocks=json.loads((out/'blocks.json').read_text());assert Counter(r['id'] for r in results)==Counter(j['id'] for j in blocks)
 ids=[i for r in results for i in r['request_ids']]+[r['shadow_request'] for r in results if r.get('shadow_request') is not None]
 assert sorted(ids)==list(range(len(rows)))
 summary=dict(arms=[],requests=len(rows),gates=sum(r['mode']=='gate' for r in rows))
 for arm in ('mosaic','fill'):
  aa=dict(arm=arm)
  for subset,full in [('synthetic',True),('real',False)]:
   tasks=[]
   for c in cases:
    if c['fully_annotated']!=full:continue
    r=next(r for r in results if r['case']==c['name'] and r['arm']==arm)
    sc=b.score(r['proposals'],c['gold']);base=b.score(first[c['name']]['proposals'],c['gold']);hidden=set();trace=[]
    for t in r['trace']:
     step=b.score(t['boxes'],c['gold']);loss=[g['id'] for g in c['gold'] if g['id'] not in step['matched'] and m.masked_fraction(t['boxes'],g)>=.9];hidden.update(loss)
     trace.append(dict(batch=t['batch'],tp=step['tp'],decision=t['decision'],hidden_unretrieved=loss))
    shadow=rows[r['shadow_request']] if r.get('shadow_request') is not None else None
    gain=b.score(r['proposals']+shadow['boxes'],c['gold'])['tp']-sc['tp'] if shadow and shadow['valid'] else 0
    tasks.append(dict(case=c['name'],**sc,initial_tp=base['tp'],gain=sc['tp']-base['tp'],
     raw_tp=b.score(r['proposals'],c['gold'],padding=0)['tp'],cap4_tp=b.score(r['proposals'],c['gold'],cap=4)['tp'],coverage80_tp=b.score(r['proposals'],c['gold'],minimum=.8)['tp'],
     stop=r['stop'],seconds=r['seconds'],calls=len(r['request_ids'])+1,trace=trace,ever_hidden=sorted(hidden),shadow=shadow is not None,shadow_gain=gain))
   totals={k:sum(t[k] for t in tasks) for k in ('tp','gold','predicted','initial_tp','gain','raw_tp','cap4_tp','coverage80_tp','any_coverage','mixed_boxes','duplicate_eligible','no_eligible_target','calls','shadow_gain')}
   totals.update(median_seconds=statistics.median(t['seconds'] for t in tasks),max_seconds=max(t['seconds'] for t in tasks),
     all_retrieved=sum(t['tp']==t['gold'] for t in tasks) if full else None,stops=dict(Counter(t['stop'] for t in tasks)),
     no_with_missing=sum(t['stop']=='no' and t['tp']<t['gold'] for t in tasks),ever_hidden=sum(len(t['ever_hidden']) for t in tasks),shadow_calls=sum(t['shadow'] for t in tasks),tasks=tasks)
   aa[subset]=totals
  summary['arms'].append(aa)
 b.save(out/'summary.json',summary)
 gallery=['<!doctype html><meta charset="utf-8"><title>Plain masks</title><style>body{font:16px sans-serif}.row{display:flex;gap:20px;flex-wrap:wrap}img{width:384px;image-rendering:pixelated}pre{white-space:pre-wrap;max-width:384px}</style><h1>Plain masks: actual inputs</h1><p>Final proposal outlines are for review only; model inputs have no outlines.</p>']
 for c in cases:
  gallery.append('<h2>'+c['name']+'</h2><div class="row">');im=Image.open(out/(c['name']+'.png')).convert('RGB')
  for arm in ('mosaic','fill'):
   r=next(r for r in results if r['case']==c['name'] and r['arm']==arm);name=c['name']+'-'+arm+'-final.png';b.marked(im,r['proposals']).save(out/name)
   gallery.append('<div><h3>'+arm+'</h3><img src="'+name+'"><p>'+r['stop']+' | '+str(round(r['seconds'],3))+'s</p><details><summary>Inputs and responses</summary>')
   for rid in r['request_ids']+([r['shadow_request']] if r.get('shadow_request') is not None else []):
    row=rows[rid];gallery.append('<h4>'+row['mode']+'</h4><img src="'+row['image']+'"><pre>'+html.escape(row.get('answer',row.get('error','')))+'</pre>')
   gallery.append('</details></div>')
  gallery.append('</div>')
 (out/'gallery.html').write_text('\n'.join(gallery))
 assert json.loads((out/'server-before.json').read_text())==json.loads((out/'server-after.json').read_text())
 assert json.loads((out/'health-after.json').read_text())['status']=='ok'
 b.save(out/'verification.json',dict(passed=True,requests=len(rows),valid=sum(r['valid'] for r in rows),blocks=len(results),same_server=True,initial_reused=True))
 for a in summary['arms']:print(a['arm'],[(s,a[s]['tp'],a[s]['gold'],round(a[s]['median_seconds'],3),a[s]['stops']) for s in ('synthetic','real')])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);analyze(p.parse_args().output)
