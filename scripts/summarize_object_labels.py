"""Aggregate explicit semantic reviews, retaining excluded diagnostic responses."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import read_lines
from scripts.benchmark_recognition import save_json
from scripts.benchmark_object_labels import ARMS


def summarize(out):
    original=[r for r in read_lines(out/'measurements.jsonl') if not r['warmup']]
    corrected=read_lines(out/'oracle-correction-measurements.jsonl')
    replacement={r['id']:r for r in corrected}
    rows=[replacement.get(r['id'],r) for r in original]
    reviews=json.loads((out/'assistant-review.json').read_text())['rows']
    scores={r['measurement_id']:r for r in reviews}
    cases={c['name']:c for c in json.loads((out/'cases.json').read_text())}
    assert len(rows)==len(scores)==len(reviews)==153
    assert set(scores)=={r['id'] for r in rows}
    valid=lambda r:r['valid'] and r['within_budget']
    success=lambda r:valid(r) and scores[r['id']]['primary']
    arms=[];by_case=[]
    for arm in ARMS:
        rs=[r for r in rows if r['arm']==arm];entry={'arm':arm,'n':len(rs)}
        for category in ['motion','multi_motion','deformation','split','static','ambiguous']:
            selected=[r for r in rs if cases[r['case']]['truth']['category']==category]
            entry[category]={'correct':sum(bool(success(r)) for r in selected),
                             'target_change':sum(valid(r) and scores[r['id']]['target_change'] for r in selected),
                             'n':len(selected)}
        entry.update(seconds_median=statistics.median(r['seconds'] for r in rs),
            prompt_tokens=sorted({r['response']['usage']['prompt_tokens'] for r in rs if r.get('response')}),
            valid=sum(valid(r) for r in rs),
            spurious_change=sum(scores[r['id']]['spurious_change'] for r in rs),
            identity_overcommit=sum(scores[r['id']]['identity_overcommit'] for r in rs),
            ls20_strip=sum(scores[r['id']]['ls20_strip'] for r in rs if r['case']=='ls20_up'))
        arms.append(entry)
        for name in cases:
            selected=[r for r in rs if r['case']==name]
            by_case.append({'arm':arm,'case':name,'n':len(selected),'correct':sum(bool(success(r)) for r in selected)})
    controls=[r for r in rows if r['case']=='no_images']
    inventory=[r for r in read_lines(out/'inventory-measurements.jsonl') if not r['warmup']]
    result={'arms':arms,'by_case':by_case,'no_images':{'proper_abstention':sum(bool(success(r)) for r in controls),'n':len(controls)},
        'valid':sum(valid(r) for r in rows),'main_including_controls':len(rows),
        'static':{'n':len(inventory),'valid':sum(valid(r) for r in inventory),'seconds_median':statistics.median(r['seconds'] for r in inventory)},
        'total_requests':len(read_lines(out/'measurements.jsonl'))+len(read_lines(out/'inventory-measurements.jsonl'))+len(corrected),
        'excluded_oracle_ids':sorted(replacement),'seconds_max':max(r['seconds'] for r in rows+inventory)}
    save_json(out/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='by_case'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();summarize(a.output)
