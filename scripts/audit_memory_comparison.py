"""Extract raw evidence and current-attempt provenance; does not grade semantics."""
import argparse
from collections import Counter
import json
from pathlib import Path

from benchmark_planning_comparison import read_rows, write


def audit(root):
    summary = json.loads((root/'summary.json').read_text())
    cases = json.loads((root/'review-cases.json').read_text())
    evidence = []
    for case in cases:
        logs = root/case['run']/case['source']
        models = read_rows(next(logs.glob('*.model.jsonl')))
        observations = read_rows(next(logs.glob('*.observations.jsonl')))
        artifacts = read_rows(next(logs.glob('*.artifacts.jsonl')))
        model = next(m for m in models if m['work']=='reconcile' and m['sequence']>case['sequence']
                     and m['observation_id']==case['observation_id'] and m['schema_valid'])
        c = model['context']
        actual = (c.get('active_trial') or {}).get('actual_result') if 'active_trial' in c else c.get('current_invocation_result')
        original = next(r['result'] for r in artifacts if r['event']=='comparison_stage_accepted'
                        and r.get('work')=='reconcile' and r['sequence']>case['sequence'])
        index = next(i for i,o in enumerate(observations) if o['observation_id']==case['observation_id'])
        before, after = observations[index-1], observations[index]
        changes = [dict(x=x,y=y,before=before['grid'][y][x],after=v)
                   for y,row in enumerate(after['grid']) for x,v in enumerate(row) if before['grid'][y][x]!=v]
        facts = dict(raw_changed_count=len(changes), raw_changed_pixels=changes,
            measured_moves=[dict(before=c['before']['bbox'],after=c['after']['bbox'],delta=c['delta_xy'])
                            for c in (case['measurement'] or {}).get('changes',[])])
        if case['game']=='ls20':
            facts['white_cross_roi_changed_count'] = sum(20<=c['x']<=22 and 31<=c['y']<=33 for c in changes)
            facts['bottom_bar_roi_changes'] = [c for c in changes if c['y'] in (61,62)]
        evidence.append(dict(run=case['run'],arm=case['arm'],step=case['step'],sequence=case['sequence'],
            observation_id=case['observation_id'],review=original,facts=facts,
            current_attempt_result=actual, active_trial=c.get('active_trial'),
            prior_understanding=c.get('understanding'), plan=c.get('plan'),
            knowledge_input=c.get('long_term_knowledge'), next_action=case['next_action'],
            before_image=str((logs/before['image_path']).relative_to(root)),
            after_image=str((logs/after['image_path']).relative_to(root))))
    write(root/'review-evidence.json', evidence)
    lines = ['# Memory comparison review evidence', '',
             'Raw grid facts and submitted interpretation. Not an automatic accuracy score.', '']
    for e in evidence:
        prediction = (e['active_trial'] or {}).get('expected_effect') or (e['current_attempt_result'] or {}).get('prediction')
        lines.extend([f"## {e['run']} step {e['step']} sequence {e['sequence']}", '',
            f"![Before]({e['before_image']}) ![After]({e['after_image']})", '',
            '**Current attempt has result:** '+str(e['current_attempt_result'] is not None), '',
            '**Prediction:** '+str(prediction), '',
            '**Review:** '+json.dumps(e['review'],ensure_ascii=False), '',
            '**Facts:** '+json.dumps({k:v for k,v in e['facts'].items() if k!='raw_changed_pixels'},ensure_ascii=False), ''])
    (root/'review-evidence.md').write_text('\n'.join(lines))
    metrics = {}
    for arm in ('A','B'):
        rows = [r for r in summary['runs'] if r['arm']==arm]
        reviews = [e for e in evidence if e['arm']==arm]
        calls = Counter()
        for r in rows:
            calls.update(r['calls_by_work'])
        metrics[arm] = dict(**summary['aggregates'][arm], calls_by_work=dict(calls),
            unique_review_observations=len({e['observation_id'] for e in reviews}),
            reviews_without_current_attempt_result=sum(e['current_attempt_result'] is None for e in reviews),
            stale_candidate_rejections=sum('unknown current measured candidate' in e['reason'] for r in rows for e in r['rejections']),
            knowledge_updates=sum(r['knowledge_updates'] for r in rows),
            prompt_tokens_recorded=sum(r['tokens']['prompt_tokens'] for r in rows),
            usage_recorded_calls=sum(r['usage_recorded_calls'] for r in rows),
            input_images=sum(r['input_images'] for r in rows))
    write(root/'audit-summary.json',metrics)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    audit(parser.parse_args().output.resolve())
