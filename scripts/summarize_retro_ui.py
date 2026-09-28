"""Aggregate saved semantic assessments of the retro UI comparison."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_retro_ui import ARMS
from scripts.benchmark_parallel import read_lines
from scripts.benchmark_recognition import save_json


def summarize(output):
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']]
    reviews=json.loads((output/'assistant-review.json').read_text())['rows']
    review={r['measurement_id']:r for r in reviews}
    assert len(rows)==len(reviews)==len(review)==138
    assert set(review)=={r['id'] for r in rows}
    def good(r):return r['valid'] and r['within_budget']
    def correct(r):return good(r) and bool(review[r['id']]['strict'])
    summary=[];by_case=[]
    for arm in ARMS:
        rs=[r for r in rows if r['arm']==arm]
        moving=[r for r in rs if r['case'] not in ('same_frame','synthetic_static')]
        static=[r for r in rs if r['case'] in ('same_frame','synthetic_static')]
        real=[r for r in rs if r['case'] in ('forward','reverse','same_frame')]
        summary.append({'arm':arm,'n':len(rs),'valid':sum(good(r) for r in rs),
            'moving_n':len(moving),'moving_correct':sum(correct(r) for r in moving),
            'static_n':len(static),'static_correct':sum(correct(r) for r in static),
            'real_robot_descriptions':sum(review[r['id']]['scene_as_robot'] for r in real),
            'real_bare_none':sum(review[r['id']]['bare_none'] for r in real),
            'forward_strip_correct':sum(bool(review[r['id']]['strip_correct']) and good(r) for r in rs if r['case']=='forward'),
            'false_mover':sum(review[r['id']]['false_mover'] for r in rs),
            'seconds_median':statistics.median(r['seconds'] for r in rs),
            'prompt_tokens':sorted({r.get('response',{}).get('usage',{}).get('prompt_tokens') for r in rs}),
            'completion_tokens_median':statistics.median(r.get('response',{}).get('usage',{}).get('completion_tokens',0) for r in rs)})
        for case in sorted({r['case'] for r in rs}):
            selected=[r for r in rs if r['case']==case]
            by_case.append({'arm':arm,'case':case,'n':len(selected),'correct':sum(correct(r) for r in selected)})
    controls=[]
    for arm in sorted({r['arm'] for r in rows if r['kind']=='control'}):
        rs=[r for r in rows if r['arm']==arm]
        controls.append({'arm':arm,'n':len(rs),'correct':sum(correct(r) for r in rs),
            'proper_no_image_abstention':sum(bool(review[r['id']]['proper_no_image_abstention']) for r in rs),
            'unsupported_scene':sum(review[r['id']]['unsupported_scene'] for r in rs)})
    result={'arms':summary,'by_case':by_case,'controls':controls,'main_requests':len(rows),'valid_main':sum(good(r) for r in rows),
            'seconds_max':max(r['seconds'] for r in rows),'seconds_sum':sum(r['seconds'] for r in rows)}
    save_json(output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='by_case'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);summarize(p.parse_args().output)
