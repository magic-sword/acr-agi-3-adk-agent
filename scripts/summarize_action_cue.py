"""Aggregate explicit semantic reviews for action-cue experiments."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_action_cue import ARMS
from scripts.benchmark_parallel import read_lines
from scripts.benchmark_recognition import save_json


def summarize(output):
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']]
    reviews=json.loads((output/'assistant-review.json').read_text())['rows'];by_id={r['measurement_id']:r for r in reviews}
    cases={c['name']:c for c in json.loads((output/'cases.json').read_text())}
    assert len(rows)==len(reviews)==len(by_id)==87 and set(by_id)=={r['id'] for r in rows}
    good=lambda r:r['valid'] and r['within_budget']
    correct=lambda r:good(r) and bool(by_id[r['id']]['strict'])
    summary=[];by_case=[]
    for arm in ARMS:
        rs=[r for r in rows if r['arm']==arm]
        moving=[r for r in rs if cases[r['case']]['truth']['change'].startswith('moved')]
        stationary=[r for r in rs if cases[r['case']]['truth']['change']=='unchanged']
        strip=[r for r in rs if r['case']=='vc33_click_remote']
        summary.append({'arm':arm,'n':len(rs),'moving_correct':sum(correct(r) for r in moving),'moving_n':len(moving),
            'unchanged_correct':sum(correct(r) for r in stationary),'unchanged_n':len(stationary),
            'remote_click_correct':sum(correct(r) for r in strip),'remote_click_n':len(strip),
            'false_changes':sum(by_id[r['id']]['false_change'] for r in rs),
            'annotation_confusion':sum(by_id[r['id']]['annotation_confusion'] for r in rs),
            'ls20_strip_correct':sum(bool(by_id[r['id']]['ls20_strip_correct']) and good(r) for r in rs if r['case']=='ls20_up'),
            'seconds_median':statistics.median(r['seconds'] for r in rs),
            'prompt_tokens':sorted({r['response']['usage']['prompt_tokens'] for r in rs if 'response' in r})})
        for name in cases:
            selected=[r for r in rs if r['case']==name]
            by_case.append({'arm':arm,'case':name,'kind':cases[name]['kind'],'n':len(selected),'correct':sum(correct(r) for r in selected)})
    controls=[r for r in rows if r['kind']=='control']
    result={'arms':summary,'by_case':by_case,'no_image':{'n':len(controls),'proper_abstentions':sum(bool(by_id[r['id']]['proper_abstention']) and good(r) for r in controls)},
            'main_requests':len(rows),'valid_main':sum(good(r) for r in rows),'seconds_max':max(r['seconds'] for r in rows),
            'cue_reading':[{'case':r['case'],'answer':r.get('answer'),'valid':good(r)} for r in read_lines(output/'cue-reading.jsonl')]}
    save_json(output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='by_case'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);summarize(p.parse_args().output)
