"""Aggregate explicit semantic reviews, never infer correctness from keywords."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import read_lines
from scripts.benchmark_recognition import save_json
from scripts.benchmark_object_recognition import ARMS


def summarize(output):
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']]
    reviews=json.loads((output/'assistant-review.json').read_text())['rows']
    assert len(rows)==90 and len(reviews)==90
    review={r['measurement_id']:r for r in reviews}
    assert len(review)==len(rows) and set(review)=={r['id'] for r in rows}
    results=[]
    subsets={'forward':lambda r:r['case']=='forward', 'reverse':lambda r:r['case']=='reverse',
             'real_unchanged':lambda r:r['case']=='same_frame',
             'synthetic_moving':lambda r:r['kind']=='synthetic' and r['case']!='synthetic_static',
             'synthetic_unchanged':lambda r:r['case']=='synthetic_static',
             'all_moving':lambda r:r['case'] not in ('same_frame','synthetic_static','no_image'),
             'all_images':lambda r:r['case']!='no_image'}
    for name,predicate in subsets.items():
        for arm in ARMS:
            rs=[r for r in rows if r['arm']==arm and predicate(r)]
            valid=lambda r:r['valid'] and r['within_budget']
            results.append({'subset':name,'arm':arm,'n':len(rs),
                **{metric:sum(bool(review[r['id']][metric]) and valid(r) for r in rs)
                   for metric in ('identity','strict','false_mover','relation','strip','annotation_as_new_object')},
                'valid':sum(valid(r) for r in rs),
                'seconds_median':statistics.median(r['seconds'] for r in rs),
                'seconds_max':max(r['seconds'] for r in rs),
                'prompt_tokens_median':statistics.median(r['response']['usage']['prompt_tokens'] for r in rs),
                'output_tokens_median':statistics.median(r['response']['usage']['completion_tokens'] for r in rs)})
    controls=[]
    for arm in ('no_image_greedy','no_image_official'):
        rs=[r for r in rows if r['arm']==arm]
        controls.append({'arm':arm,'n':len(rs),'proper_abstentions':sum(review[r['id']]['abstains_without_image'] for r in rs),
                         'unsupported_scene_descriptions':sum(review[r['id']]['unsupported_scene_description'] for r in rs)})
    save_json(output/'summary.json',{'primary':results,'no_image':controls,'main_requests':len(rows),
         'valid':sum(r['valid'] and r['within_budget'] for r in rows),'total_measured_seconds':sum(r['seconds'] for r in rows)})
    if (output/'stage2-review.json').exists():
        rs=read_lines(output/'stage2-measurements.jsonl');reviews=json.loads((output/'stage2-review.json').read_text())['rows']
        assert len(rs)==12 and len(reviews)==12
        by_id={r['measurement_id']:r for r in reviews}
        assert len(by_id)==12 and set(by_id)=={r['id'] for r in rs}
        stage=[]
        for arm in ARMS:
            selected=[r for r in rs if r['arm']==arm]
            stage.append({'arm':arm,'n':len(selected),**{metric:sum(bool(by_id[r['id']][metric]) and r['valid'] and r['within_budget'] for r in selected)
                          for metric in ('target','tentative','grounded_motion','next_check','handoff')},
                          'seconds_median':statistics.median(r['seconds'] for r in selected)})
        save_json(output/'stage2-summary.json',stage)
    print(json.dumps({'primary':[r for r in results if r['subset'] in ('all_moving','forward','reverse')],'controls':controls},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    summarize(p.parse_args().output)
