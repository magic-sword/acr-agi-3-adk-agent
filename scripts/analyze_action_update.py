"""Reproducible input/decision audit, without assigning optimal game actions."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics

import jsonschema

from probe_repeated_click_inputs import context_part, digest


def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', type=Path, required=True)
    parser.add_argument('--frozen', type=Path)
    args = parser.parse_args()
    root = args.live.resolve()
    summary = json.loads((root/'summary.json').read_text())
    inputs, decisions, model_errors = [], [], []
    source_hashes = []
    for run in summary['runs']:
        directory = root/run['name']
        source_hashes.append(json.loads((directory/'manifest.json').read_text())['source_sha256'])
        logs = directory/run['game_id']/'cognition'
        artifacts = read_rows(next(logs.glob('*.artifacts.jsonl')))
        observations = {r['observation_id']:r for r in read_rows(next(logs.glob('*.observations.jsonl')))}
        for m in read_rows(next(logs.glob('*.model.jsonl'))):
            inputs.append(dict(run=run['name'], arm=run['arm'], work=m['work'],
                               characters=len(json.dumps(m.get('context', {}), ensure_ascii=False)),
                               prompt_tokens=(m.get('usage') or {}).get('prompt_tokens')))
        for r in artifacts:
            if r['event'] == 'task_rejected':
                model_errors.append(dict(run=run['name'], arm=run['arm'], work=r['work'], reason=r['reason']))
            if r['event'] != 'update_trial':
                continue
            t = r['trial']
            feedback = next(x['trial'] for x in artifacts if x['event'] == 'action_feedback'
                            and x['trial']['decision_id'] == t['decision_id'])
            before, after = (observations[feedback[k]] for k in ('before_observation_id','after_observation_id'))
            changes = [dict(x=x, y=y, before=before['grid'][y][x], after=value)
                       for y, row in enumerate(after['grid']) for x, value in enumerate(row)
                       if before['grid'][y][x] != value]
            assert len(changes) == feedback['changed_cell_count'] == t['measured_result']['changed_cell_count']
            decisions.append(dict(run=run['name'], arm=run['arm'], trial=t, raw_changes=changes,
                                  before_image=str((logs/before['image_path']).relative_to(root)),
                                  after_image=str((logs/after['image_path']).relative_to(root))))
    assert all(h == source_hashes[0] for h in source_hashes)
    for rel, expected in source_hashes[0].items():
        assert hashlib.sha256((root/summary['runs'][0]['name']/'package'/rel).read_bytes()).hexdigest() == expected
    metrics = {}
    for arm in 'ABCD':
        selected = [r for r in inputs if r['arm'] == arm]
        ground = [r for r in selected if r['work'] == 'ground']
        metrics[arm] = dict(**summary['aggregates'][arm],
                            ground_context_characters_median=statistics.median(r['characters'] for r in ground),
                            calls_by_work=dict(Counter(r['work'] for r in selected)),
                            task_rejections=dict(Counter(r['reason'] for r in model_errors if r['arm'] == arm)))
    (root/'audit-summary.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n')
    (root/'review-evidence.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2)+'\n')
    print('Live:', json.dumps(metrics, ensure_ascii=False))
    if not args.frozen:
        return
    frozen = args.frozen.resolve()
    plan = json.loads((frozen/'plan.json').read_text())
    rows = json.loads((frozen/'results.json').read_text())
    annotated = []
    blocks = defaultdict(dict)
    repetition_hashes = defaultdict(set)
    for row in rows:
        payload = json.loads((frozen/row['request_file']).read_text())
        assert digest(payload) == row['request_sha256']
        blocks[row['case'], row['repetition']][row['variant']] = payload
        repetition_hashes[row['case'], row['variant']].add(digest(payload))
        _, context = context_part(payload)
        previous = plan['sources'][row['case']]['previous_trial']
        answer = row.get('answer')
        errors = []
        if isinstance(answer, dict):
            validator = jsonschema.Draft202012Validator(payload['tools'][0]['function']['parameters'])
            errors = [e.message for e in validator.iter_errors(answer)]
        else:
            errors = [row.get('error', 'No structured answer')]
        if 'objects' in context['current_observation']:
            by_ref = {c['candidate_ref']:c['name'] for c in context['current_observation']['objects']}
        else:
            measurement = next(iter(context['measurements'].values()))
            by_ref = {c[0]:'R-'+c[4] for c in measurement['candidate_index']}
        answer = answer if isinstance(answer, dict) else {}
        refs = sorted({ref for t in answer.get('targets', []) for ref in t.get('candidate_refs', [])})
        unknown = [ref for ref in refs if ref not in by_ref]
        names = sorted({by_ref[ref] for ref in refs if ref in by_ref})
        action = answer.get('action', {})
        same_control = action.get('action') == previous['action']['action']
        same_names = bool(names) and names == previous['target_names'] and not unknown
        same_query = bool(action.get('target_query')) and action.get('target_query') == previous['action'].get('target_query')
        annotated.append(dict(case=row['case'], arm=row['variant'], repetition=row['repetition'],
                              schema_errors=errors, unknown_candidate_refs=unknown,
                              finish_reason=(row.get('response', {}).get('choices') or [{}])[0].get('finish_reason'),
                              previous_control=previous['action']['action'], previous_names=previous['target_names'],
                              previous_no_change=previous['measured_result']['changed_cell_count'] == 0,
                              current_names=names, same_control=same_control, same_names=same_names,
                              same_exact_query=same_query, repeated_target_control=same_control and
                              (action.get('action') != 'CLICK' or same_names or same_query),
                              answer=answer, request_file=row['request_file']))
    matched_blocks = 0
    for arms in blocks.values():
        if set(arms) != set('ABCD'):
            continue
        a, b, c, d = (arms[arm] for arm in 'ABCD')
        assert a['messages'][0] == b['messages'][0]
        assert c['messages'][0] == d['messages'][0]
        assert c['messages'][0]['content'].startswith(a['messages'][0]['content'])
        assert a['messages'][1:] == c['messages'][1:]
        assert b['messages'][1:] == d['messages'][1:]
        for p in (b, c, d):
            assert {k:v for k,v in p.items() if k != 'messages'} == {k:v for k,v in a.items() if k != 'messages'}
        masked = []
        for p in (a, b):
            p = deepcopy(p)
            part, _ = context_part(p)
            part['text'] = 'CONTEXT'
            masked.append(p)
        assert masked[0] == masked[1], 'Representation changed something outside context'
        matched_blocks += 1
    assert all(len(hashes) == 1 for hashes in repetition_hashes.values())
    aggregates = {}
    for arm in 'ABCD':
        selected = [r for r in annotated if r['arm'] == arm]
        aggregates[arm] = dict(requests=len(selected), schema_valid=sum(not r['schema_errors'] for r in selected),
                               known_refs=sum(not r['unknown_candidate_refs'] for r in selected),
                               exact_identity_repeats=sum(r['repeated_target_control'] for r in selected),
                               no_change_requests=sum(r['previous_no_change'] for r in selected),
                               no_change_exact_identity_repeats=sum(r['previous_no_change'] and r['repeated_target_control'] for r in selected))
    output = dict(complete=len(rows)==plan['requests'], expected=plan['requests'], requests=len(rows),
                  matched_four_arm_blocks=matched_blocks, repeated_input_hashes_identical=True,
                  aggregates=aggregates, cases=annotated,
                  caveat='Unknown identity is not a successful target change. Different candidate IDs may overlap. '
                         'Semantics and game progress require separate evidence; no game action is executed here.')
    (frozen/'audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print('Frozen:', json.dumps({k:v for k,v in output.items() if k!='cases'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
