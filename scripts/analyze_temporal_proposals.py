"""Read-only evaluation of frozen temporal proposals; gold never fed to tracker."""
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_temporal_proposals import ARMS,verify,save
from scripts.benchmark_iterative_proposals import score,roi,coverage,area


def norm(box):return [v/64*1000 for v in box]


def eligible(box,gold):
    box=roi(norm(box))
    return {g['id'] for g in gold if coverage(box,g)>=.9-1e-9 and area(box)<=9*(g['bbox'][2]-g['bbox'][0]+1)*(g['bbox'][3]-g['bbox'][1]+1)+1e-9}


def kind(before,after,gb,ga):
    dx=ga['bbox'][0]-gb['bbox'][0];dy=ga['bbox'][1]-gb['bbox'][1]
    if {(x+dx,y+dy) for x,y in gb['support']}!={tuple(p) for p in ga['support']}:return 'shape',dx,dy
    if any(before['grid'][y][x]!=after['grid'][y+dy][x+dx] for x,y in gb['support']):return 'appearance',dx,dy
    return ('translation' if dx or dy else 'static'),dx,dy


def analyze(out):
    plan=verify(out);seqs=json.loads((out/'sequences.json').read_text())
    rows=[json.loads(x) for x in (out/'results.jsonl').read_text().splitlines()]
    assert len(rows)==plan['frames']*len(ARMS)*plan['cpu_repeats']
    lookup={(r['sequence'],r['arm'],r['frame'],r['repeat']):r for r in rows}
    assert len(lookup)==len(rows)
    for r in rows:
        ref=lookup[r['sequence'],r['arm'],r['frame'],0]
        for k in ('events','tracks','sam_called','sam_seconds'):assert r[k]==ref[k]
    frames=[];sequence_results=[];link_audit=[];continuity=[]
    for s in seqs:
        for arm in ARMS:
            scored=[];maps=[];motion_found=set();motion_gold=set();wrong_id=wrong_displacement=correct_links=ghost=ambiguous_links=unannotated_links=0
            states=[lookup[s['name'],arm,f['index'],0] for f in s['frames']]
            for f,r in zip(s['frames'],states):
                boxes=[norm(t['box']) for t in r['tracks']]
                emap={t['id']:eligible(t['box'],f['gold']) for t in r['tracks']};maps.append(emap)
                row=dict(sequence=s['name'],kind=s['kind'],arm=arm,frame=f['index'],
                    main=score(boxes,f['gold']),raw=score(boxes,f['gold'],padding=0),cap4=score(boxes,f['gold'],cap=4),
                    sam_called=r['sam_called'],refresh_requested=r['events']['refresh_requested'],
                    cpu_seconds=statistics.median(lookup[s['name'],arm,f['index'],k]['cpu_seconds'] for k in range(plan['cpu_repeats'])),sam_seconds=r['sam_seconds'])
                frames.append(row);scored.append(row)
            for i in range(1,len(states)):
                gb={g['id']:g for g in s['frames'][i-1]['gold']};ga={g['id']:g for g in s['frames'][i]['gold']}
                for ident in gb.keys()&ga.keys():
                    event,dx,dy=kind(s['frames'][i-1],s['frames'][i],gb[ident],ga[ident])
                    if event=='translation':motion_gold.add((i,ident))
                for link in states[i]['events']['links']:
                    prev=maps[i-1].get(link['id'],set());now=maps[i].get(link['id'],set())
                    category='unannotated'
                    if len(prev)>1 or len(now)>1:ambiguous_links+=1;category='multiple_gold_regions'
                    elif len(prev)==1 and len(now)==1:
                        a=next(iter(prev));b=next(iter(now))
                        if a!=b:wrong_id+=1;category='wrong_identity'
                        else:
                            event,dx,dy=kind(s['frames'][i-1],s['frames'][i],gb[a],ga[a])
                            observed=[link['after'][k]-link['before'][k] for k in (0,1)]
                            if event in ('translation','static') and observed!=[dx,dy]:wrong_displacement+=1;category='wrong_displacement'
                            else:
                                correct_links+=1;category='correct'
                                if event=='translation':motion_found.add((i,a))
                    elif len(prev)==1 and next(iter(prev)) not in ga:ghost+=1;category='departed_target_link'
                    else:unannotated_links+=1
                    link_audit.append(dict(sequence=s['name'],arm=arm,frame=i,track_id=link['id'],before_gold=sorted(prev),after_gold=sorted(now),category=category))
            continuous=initialized=initial_total=0
            for g in s['frames'][0]['gold']:
                ident=g['id'];initial_total+=1
                ids={tid for tid,gg in maps[0].items() if gg=={ident}};start=bool(ids);initialized+=start
                present=0
                for f,mp in zip(s['frames'],maps):
                    if ident not in {x['id'] for x in f['gold']}:break
                    present+=1;ids &= {tid for tid,gg in mp.items() if gg=={ident}}
                continuous+=bool(ids)
                continuity.append(dict(sequence=s['name'],arm=arm,gold_id=ident,initialized=start,visible_frames=present,continuous=bool(ids),surviving_ids=sorted(ids)))
            totals=[sum(lookup[s['name'],arm,f['index'],k]['cpu_seconds'] for f in s['frames']) for k in range(plan['cpu_repeats'])]
            cpu=statistics.median(totals);sam=sum(r['sam_seconds'] for r in states)
            first=statistics.median(lookup[s['name'],arm,0,k]['cpu_seconds'] for k in range(plan['cpu_repeats']))+states[0]['sam_seconds']
            after=sum(r['cpu_seconds']+r['sam_seconds'] for r in scored[1:])
            sequence_results.append(dict(sequence=s['name'],kind=s['kind'],arm=arm,frames=len(states),gold=sum(r['main']['gold'] for r in scored),
                tp=sum(r['main']['tp'] for r in scored),raw_tp=sum(r['raw']['tp'] for r in scored),cap4_tp=sum(r['cap4']['tp'] for r in scored),
                initial_tp=scored[0]['main']['tp'],after_tp=sum(r['main']['tp'] for r in scored[1:]),after_gold=sum(r['main']['gold'] for r in scored[1:]),
                motion_tp=len(motion_found),motion_gold=len(motion_gold),initial_targets=initial_total,initialized=initialized,continuous=continuous,
                correct_links=correct_links,wrong_identity=wrong_id,wrong_displacement=wrong_displacement,departed_links=ghost,
                ambiguous_links=ambiguous_links,unannotated_links=unannotated_links,sam_calls=sum(r['sam_called'] for r in states),
                refresh_requests_after_initial=sum(r['events']['refresh_requested'] for r in states[1:]),
                proposals=sum(len(r['tracks']) for r in states),cpu_seconds=cpu,sam_seconds=sam,total_seconds=cpu+sam,
                first_frame_seconds=first,after_mean_seconds=after/(len(states)-1),cpu_min_seconds=min(totals),cpu_max_seconds=max(totals)))
    summary=[]
    additive=['frames','gold','tp','raw_tp','cap4_tp','initial_tp','after_tp','after_gold','motion_tp','motion_gold','initial_targets','initialized','continuous','correct_links','wrong_identity','wrong_displacement','departed_links','ambiguous_links','unannotated_links','sam_calls','refresh_requests_after_initial','proposals','cpu_seconds','sam_seconds','total_seconds']
    for k in ('synthetic','real_partial'):
        for a in ARMS:
            rr=[r for r in sequence_results if r['kind']==k and r['arm']==a]
            summary.append(dict(kind=k,arm=a,sequences=len(rr),**{key:sum(r[key] for r in rr) for key in additive},
                mean_frame_seconds=sum(r['total_seconds'] for r in rr)/sum(r['frames'] for r in rr),
                median_first_frame_seconds=statistics.median(r['first_frame_seconds'] for r in rr),
                mean_update_seconds=sum(r['after_mean_seconds']*(r['frames']-1) for r in rr)/sum(r['frames']-1 for r in rr)))
    for name,value in [('frame-scores',frames),('sequence-summary',sequence_results),('summary',summary),('link-audit',link_audit),('continuity',continuity)]:save(out/(name+'.json'),value)
    save(out/'repeat-audit.json',dict(status='passed',equal_tracking_outputs=len(rows),repeats=plan['cpu_repeats']))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':analyze(Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/temporal-proposals-20260928'))
