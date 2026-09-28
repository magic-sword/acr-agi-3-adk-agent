"""Independent rectangle/score audit of EdgeBoxes; no inference parameters changed."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_edgeboxes import verify,read_rows,save
from scripts.audit_sam_proposals import independent_count


def run(out):
    verify(out)
    cases={c['name']:c for c in json.loads((out/'cases.json').read_text())}
    rows=read_rows(out,'results.jsonl');rectangles=0
    for r in rows:
        assert len(r['records'])==len(r['boxes'])<=r['budget']
        assert r['seconds']>=sum(r['stages'].values())>=0
        for record,box in zip(r['records'],r['boxes']):
            x,y,w,h=record['rect']
            assert w>0 and h>0 and x>=0 and y>=0 and x+w<=384 and y+h<=384
            assert np.allclose(np.array([x,y,x+w,y+h])*1000/384,box,rtol=0,atol=1e-10)
            rectangles+=1
        assert r['records']==sorted(r['records'],key=lambda v:(-v['score'],v['index']))
    selected=json.loads((out/'selected.json').read_text())
    lookup={(r['case'],r['arm'],r['budget']):r for r in selected}
    checks=0
    for r in json.loads((out/'scored.json').read_text()):
        boxes=lookup[(r['case'],r['arm'],r['budget'])]['boxes'][:r['cap']]
        for key,pad,cap in [('main',1,9),('raw',0,9),('cap4',1,4)]:
            assert independent_count(boxes,cases[r['case']]['gold'],pad,cap)==r[key]['tp'],r
            checks+=1
    repeated={(r['case'],r['arm'],r['budget'],r['repeat']):r for r in rows}
    repeat_checks=0
    for r in json.loads((out/'repeat-scores.json').read_text()):
        if r['cap']!=r['budget']:continue
        boxes=repeated[r['case'],r['arm'],r['budget'],r['repeat']]['boxes']
        assert independent_count(boxes,cases[r['case']]['gold'],1,9)==r['tp']
        repeat_checks+=1
    result=dict(status='passed',rectangle_checks=rectangles,score_checks=checks,repeat_checks=repeat_checks,
                requests=len(rows),frozen_hashes='passed')
    save(out/'independent-audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':run(Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/edgeboxes-20260928'))
