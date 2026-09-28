"""Independent accounting, pixel-link and region-recall audit."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_temporal_proposals import verify,save,ARMS
from scripts.audit_sam_proposals import independent_count
from scripts.temporal_proposal_tracker import program_boxes


def eligible_integer(box,gold):
    l,t,r,b=box;l=max(0,l-1);t=max(0,t-1);r=min(64,r+1);b=min(64,b+1)
    found=set()
    for g in gold:
        x1,y1,x2,y2=g['bbox']
        hits=sum(l<=x<r and t<=y<b for x,y in g['support'])
        if hits>=.9*len(g['support'])-1e-9 and (r-l)*(b-t)<=9*(x2-x1+1)*(y2-y1+1):found.add(g['id'])
    return found


def patch_signature(frame,g):
    x0=min(p[0] for p in g['support']);y0=min(p[1] for p in g['support'])
    return {(x-x0,y-y0,frame['grid'][y][x]) for x,y in g['support']}


def run(out):
    plan=verify(out)
    seqs={s['name']:s for s in json.loads((out/'sequences.json').read_text())}
    rows=[json.loads(x) for x in (out/'results.jsonl').read_text().splitlines()]
    lookup={(r['sequence'],r['arm'],r['frame'],r['repeat']):r for r in rows}
    sam={(r['sequence'],r['frame']):r for r in [json.loads(x) for x in (out/'sam-results.jsonl').read_text().splitlines()]}
    scores=json.loads((out/'frame-scores.json').read_text());checks=links=triggers=0
    for r in scores:
        state=lookup[r['sequence'],r['arm'],r['frame'],0]
        frame=seqs[r['sequence']]['frames'][r['frame']]
        boxes=[[v*1000/64 for v in t['box']] for t in state['tracks']]
        for key,pad,cap in [('main',1,9),('raw',0,9),('cap4',1,4)]:
            assert independent_count(boxes,frame['gold'],pad,cap)==r[key]['tp'];checks+=1
    for name,s in seqs.items():
        for arm in ARMS:
            for i,f in enumerate(s['frames']):
                r=lookup[name,arm,i,0];current=np.asarray(f['grid']);old=lookup.get((name,arm,i-1,0))
                assert len({t['id'] for t in r['tracks']})==len(r['tracks'])
                assert len({tuple(t['box']) for t in r['tracks']})==len(r['tracks'])
                if old:
                    previous=np.asarray(s['frames'][i-1]['grid']);old_by={t['id']:t for t in old['tracks']};new_by={t['id']:t for t in r['tracks']}
                    linked=set()
                    for link in r['events']['links']:
                        ident=link['id'];assert ident in old_by and ident in new_by
                        assert old_by[ident]['box']==link['before'] and new_by[ident]['box']==link['after']
                        x1,y1,x2,y2=link['before'];a,b,c,d=link['after']
                        assert abs(a-x1)<=8 and abs(b-y1)<=8
                        assert np.array_equal(previous[y1:y2,x1:x2],current[b:d,a:c])
                        linked.add(ident);links+=1
                    # Exclude SAM candidates just added AFTER refresh decision.
                    pre_sam=[t for t in r['tracks'] if t['source']=='program' or t['id'] in old_by]
                    covered=set()
                    for t in old['tracks']+pre_sam:
                        x1,y1,x2,y2=t['box']
                        if (x2-x1)*(y2-y1)<=256:covered.update((x,y) for x in range(x1,x2) for y in range(y1,y2))
                    yy,xx=np.where(previous!=current);changed=set(zip(xx.tolist(),yy.tolist()))
                    residual=len(changed-covered)
                    local=sum(t['id'] not in linked and (t['box'][2]-t['box'][0])*(t['box'][3]-t['box'][1])<=256 for t in old['tracks'])
                    _,limited=program_boxes(f['grid'])
                    assert residual==r['events']['uncovered_changed_pixels']
                    assert local==r['events']['local_unresolved']
                    assert r['events']['refresh_requested']==bool(residual or local or limited);triggers+=1
                expected=arm=='sam_every' or (arm!='program' and i==0) or (arm=='sam_adaptive' and r['events']['refresh_requested'])
                assert r['sam_called']==expected
                assert r['sam_seconds']==(sam[name,i]['seconds'] if expected else 0)
    motion_checks=0
    summary=json.loads((out/'sequence-summary.json').read_text())
    for s in seqs.values():
        for arm in ARMS:
            true_events=set();detected=set()
            for i in range(1,len(s['frames'])):
                bf=s['frames'][i-1];af=s['frames'][i]
                bg={g['id']:g for g in bf['gold']};ag={g['id']:g for g in af['gold']}
                movement={}
                for ident in bg.keys()&ag.keys():
                    a=bg[ident];b=ag[ident]
                    if patch_signature(bf,a)!=patch_signature(af,b):continue
                    delta=(b['bbox'][0]-a['bbox'][0],b['bbox'][1]-a['bbox'][1])
                    if delta!=(0,0):true_events.add((i,ident));movement[ident]=delta
                for link in lookup[s['name'],arm,i,0]['events']['links']:
                    old=eligible_integer(link['before'],bf['gold']);new=eligible_integer(link['after'],af['gold'])
                    if len(old)!=1 or old!=new:continue
                    ident=next(iter(old));delta=tuple(link['after'][k]-link['before'][k] for k in (0,1))
                    if movement.get(ident)==delta:detected.add((i,ident))
            expected=next(r for r in summary if r['sequence']==s['name'] and r['arm']==arm)
            assert (len(true_events),len(detected))==(expected['motion_gold'],expected['motion_tp'])
            motion_checks+=1
    result=dict(status='passed',independent_frame_scores=checks,exact_pixel_links=links,refresh_decisions=triggers,independent_motion_summaries=motion_checks,
                frames=plan['frames'],cpu_runs=len(rows),sam_calls_measured=len(sam),source_and_model_hashes='passed')
    save(out/'independent-audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':run(Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/temporal-proposals-20260928'))
