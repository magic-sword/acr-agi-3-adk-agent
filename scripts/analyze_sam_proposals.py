"""Post-measurement rank/group diagnostics; never changes frozen inference."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_sam_proposals import verify, save, score


def run(out):
    verify(out)
    cases={c['name']:c for c in json.loads((out/'cases.json').read_text())}
    rows=[json.loads(line) for name in ('sam-results.jsonl','qwen-results.jsonl')
          for line in (out/name).read_text().splitlines()]
    diagnostics=[]
    for r in rows:
        gold=cases[r['case']]['gold']
        complete=next((k for k in range(len(r['boxes'])+1)
                       if score(r['boxes'][:k],gold)['tp']==len(gold)),None)
        ranks={g['id']:next((i+1 for i,b in enumerate(r['boxes']) if score([b],[g])['tp']),None)
               for g in gold}
        diagnostics.append(dict(case=r['case'],arm=r['arm'],pool=len(r['boxes']),
            minimum_prefix_for_all_annotated=complete,first_eligible_rank_by_target=ranks,
            pool_score=score(r['boxes'],gold)))
    scored=json.loads((out/'scored.json').read_text())
    groups=[]
    for arm in sorted({r['arm'] for r in scored}):
        for group in sorted({r['group'] for r in scored}):
            rr=[r for r in scored if r['arm']==arm and r['group']==group and r['cap']==128]
            groups.append(dict(arm=arm,group=group,tp=sum(r['main']['tp'] for r in rr),
                               gold=sum(r['main']['gold'] for r in rr),
                               raw_tp=sum(r['raw']['tp'] for r in rr),
                               cap4_tp=sum(r['cap4']['tp'] for r in rr)))
    # Prior Qwen outputs serve only as a reproducibility check, not new timing data.
    historical=Path('outputs/batch-masks-20260928/responses.jsonl')
    agreement=None
    if historical.exists():
        old=[json.loads(line) for line in historical.read_text().splitlines()]
        # Saved fields vary by experiment; choose explicitly marked initial calls.
        old=[r for r in old if r.get('mode')=='initial']
        if len(old)==23:
            mapping={r['case']:r for r in old}
            agreement=sum(r['boxes']==mapping[r['case']].get('boxes') for r in rows if r['arm']=='qwen')
    save(out/'diagnostics.json',dict(note='Post-measurement diagnostics; no parameters selected from gold.',
        per_case=diagnostics,by_group_top128=groups,historical_qwen_box_agreement=agreement))
    print(json.dumps(groups,indent=2))


if __name__=='__main__':run(Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/sam-proposals-20260928'))
