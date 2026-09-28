"""Post-run integrity checks and grouped cost summaries, without inference."""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_delta_recognition import verify,ARMS,save


def audit(out):
    cases,jobs=verify(out);bycase={c['name']:c for c in cases}
    evidence=json.loads((out/'evidence.json').read_text());e={(r['case'],r['trial']):r for r in evidence}
    nlinks=0
    for c in cases:
        for t in c['trials']:
            record=e[c['name'],t['id']];reconstructed=deepcopy(t['before']);seen=set()
            for y,x0,x1,old,new in record['delta']['runs']:
                assert old!=new
                for x in range(x0,x1+1):
                    assert (x,y) not in seen and reconstructed[y][x]==old
                    seen.add((x,y));reconstructed[y][x]=new
            assert reconstructed==t['after'] and len(seen)==record['delta']['changed_pixels']
            for link in record['tracking']['links']:
                x,y,r,b=link['before'];xx,yy,rr,bb=link['after']
                assert [row[x:r+1] for row in t['before'][y:b+1]]==[row[xx:rr+1] for row in t['after'][yy:bb+1]]
                assert link['delta_xy']==[xx-x,yy-y];nlinks+=1
        byarm={j['arm']:j for j in jobs if j['case']==c['name']}
        content=lambda arm:byarm[arm]['payload']['messages'][1]['content']
        for arm in ARMS:
            txt=' '.join(k.get('text','') for k in content(arm))
            assert ('Program rectangle correspondence:' in txt)==(arm=='tracked' and not c['after_only'])
            assert ('Raw pixel delta:' in txt)==(arm in ('delta','tracked') and not c['after_only'])
            # Exact static inventories must appear in every corresponding arm.
            for t in c['trials']:
                assert json.dumps(e[c['name'],t['id']]['after'],separators=(',',':')) in txt
        stripped=[k for k in content('tracked') if not k.get('text','').startswith('Program rectangle correspondence:')]
        assert stripped==content('delta')
        stripped=[k for k in content('delta') if not k.get('text','').startswith('Raw pixel delta:')]
        assert stripped==content('frames')
    rows=json.loads((out/'scores.json').read_text());assert Counter(r['id'] for r in rows)==Counter(j['id'] for j in jobs)
    assert all(r['valid'] and not r.get('error') for r in rows)
    assert all(r['response']['usage']['completion_tokens']==1 for r in rows)
    assert json.loads((out/'server-before.json').read_text())==json.loads((out/'server-after.json').read_text())
    images=re.findall(r'<img src="([^"]+)"',(out/'gallery.html').read_text());assert all((out/p).is_file() for p in images)
    cost={}
    for origin in sorted({c['origin'] for c in cases}):
        ee=[r for r in evidence if bycase[r['case']]['origin']==origin]
        cost[origin]=dict(trials=len(ee),milliseconds={k:dict(median=1000*statistics.median(r['timing'][k] for r in ee),
            total=1000*sum(r['timing'][k] for r in ee)) for k in ee[0]['timing']})
    factual={}
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm and r['group']!='identity_limit']
        controls=[r for r in rows if r['arm']==arm and r['group']=='identity_limit']
        factual[arm]=dict(factual_correct=sum(r['correct'] for r in rr),factual_total=len(rr),
            identity_limit_correct=sum(r['correct'] for r in controls),identity_limit_total=len(controls),
            examples=sum(len(bycase[r['case']]['trials']) for r in rr))
    result=dict(passed=True,lossless_delta_trials=len(evidence),exact_link_patches=nlinks,all_static_inventories_retained=True,
                arms_differ_only_by_specified_evidence=True,same_server_settings=True,single_token_answers=len(rows),
                html_image_links=len(images),timing=cost,factual_separate_from_identity_controls=factual,
                limitations='One pass timing, no SAM detector time, no live gameplay. Static inventory is shared but current-only has no BEFORE. Ground truth is generated or manually specified independently of matcher output.')
    save(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();audit(a.out)
