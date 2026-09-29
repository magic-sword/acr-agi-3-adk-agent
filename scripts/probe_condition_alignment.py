"""Matched condition/current-value inputs: local backward links and full plans."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from probe_abstract_skill_planning import build as planning_build, task_for, holds, solve
from probe_skill_backchain_component import build as link_build
from probe_incremental_planning_inputs import execute
from probe_repeated_click_inputs import digest

SCENARIOS = ('closed_gate', 'gate_open', 'at_switch', 'other_goal', 'already_done',
             'unreachable', 'needs_recharge', 'composite_available',
             'false_required_ready', 'false_required_missing', 'conjunction_ready', 'conjunction_missing')
VARIANTS = ('separate', 'local_maps', 'paired_rows')


def world(scenario):
    if scenario in SCENARIOS[:8]:
        return task_for(scenario, 'descriptive')
    task = task_for('closed_gate', 'descriptive')
    exit_skill = next(s for s in task['available_skills'] if s['target'] == 'exit_A')
    if scenario.startswith('false_required'):
        exit_skill['requires']['gate_A_open'] = False
        task['current_state']['gate_A_open'] = scenario.endswith('missing')
        switch = next(s for s in task['available_skills'] if s['verb'] == 'activate' and s['target'] == 'left_switch')
        switch['verb'] = 'deactivate'
        switch['effects']['gate_A_open'] = False
    else:
        exit_skill['requires']['charged'] = True
        task['current_state'].update(gate_A_open=True, charged=scenario.endswith('ready'))
    return task


def represent(task, variant):
    result = deepcopy(task)
    result['rules'] += (' Any current_values or condition_values annotations refer only to the initial current_state '
                        'of this request. They are not permanent conditions; when planning later steps, '
                        'use the state after preceding skill effects.')
    for skill in result['available_skills']:
        if variant == 'local_maps':
            skill['current_values'] = {k: task['current_state'][k] for k in skill['requires']}
        elif variant == 'paired_rows':
            skill['condition_values'] = [dict(condition=k, required=v, current=task['current_state'][k])
                                         for k, v in skill['requires'].items()]
    return result


def build(phase):
    templates = dict(plan=next(j['payload'] for j in planning_build() if j['variant'] == 'direct'),
                     component=link_build()[0]['payload'])
    jobs = []
    for scenario in SCENARIOS:
        for reverse in (False, True):
            task = world(scenario)
            if reverse:
                task['available_skills'].reverse()
                # Also reverse conjunction rows so no predicate always appears first.
                for s in task['available_skills']:
                    s['requires'] = dict(reversed(list(s['requires'].items())))
            goal_holds = holds(task['current_state'], task['goal'])
            matches = [s for s in task['available_skills'] if holds(s['effects'], task['goal'])]
            assert len(matches) == 1
            last = matches[0]
            expected = dict(goal_holds_now=goal_holds,
                            last_skill=None if goal_holds else {k: last[k] for k in ('verb', 'target')},
                            unmet_requirements={} if goal_holds else
                            {k: v for k, v in last['requires'].items() if task['current_state'][k] != v})
            for variant in VARIANTS:
                payload = deepcopy(templates[phase])
                payload['messages'][0]['content'][1]['text'] = json.dumps(represent(task, variant), separators=(',', ':'))
                jobs.append(dict(case=scenario, scenario=scenario, phase=phase, variant=variant,
                                 repetition=int(reverse), task=task, expected=expected,
                                 oracle_plan=solve(task), payload=payload))
    random.Random(2909202680 + (phase == 'plan')).shuffle(jobs)
    assert len(jobs) == 72
    return jobs


def check_design(jobs):
    for job in jobs:
        rep = json.loads(job['payload']['messages'][0]['content'][1]['text'])
        for original, skill in zip(job['task']['available_skills'], rep['available_skills']):
            if 'current_values' in skill:
                assert skill.pop('current_values') == {k: job['task']['current_state'][k] for k in original['requires']}
            if 'condition_values' in skill:
                rows = skill.pop('condition_values')
                assert {r['condition']: r['required'] for r in rows} == original['requires']
                assert all(r['current'] == job['task']['current_state'][r['condition']] for r in rows)
        rep['rules'] = job['task']['rules']
        assert rep == job['task']
        assert not any(k in json.dumps(job['payload']) for k in ('oracle_plan', 'goal_holds_now":true', 'is_satisfied'))
    by = {(j['scenario'], j['variant'], j['repetition']): j for j in jobs}
    for job in jobs:
        p = deepcopy(job['payload'])
        other = by[job['scenario'], 'separate', job['repetition']]['payload']
        p['messages'][0]['content'][1] = other['messages'][0]['content'][1]
        assert p == other
    expected_lengths = dict(closed_gate=3, gate_open=1, at_switch=2, other_goal=3, already_done=0,
                            unreachable=None, needs_recharge=5, composite_available=2,
                            false_required_ready=1, false_required_missing=3, conjunction_ready=1, conjunction_missing=3)
    for job in jobs:
        assert (None if job['oracle_plan'] is None else len(job['oracle_plan'])) == expected_lengths[job['scenario']]


def audit(out, phase):
    import jsonschema
    from analyze_abstract_skill_planning import evaluate, content_diagnostic, self_check
    self_check()
    jobs = build(phase)
    check_design(jobs)
    rows = json.loads((out / 'results.json').read_text())
    plan = json.loads((out / 'plan.json').read_text())
    assert len(rows) == len(jobs) == plan['requests']
    assert plan['probe_script_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = []
    for i, (job, row) in enumerate(zip(jobs, rows)):
        request = json.loads((out / row['request_file']).read_text())
        assert digest(request) == digest(job['payload']) == row['request_sha256'] == plan['jobs'][i]['request_sha256']
        answer = row.get('answer')
        schema = request['tools'][0]['function']['parameters']
        errors = [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(answer)]
        choices = row.get('response', {}).get('choices', [{}])
        calls = choices[0].get('message', {}).get('tool_calls', [])
        valid = not errors and len(calls) == 1 and row.get('tool') == request['tools'][0]['function']['name']
        score = dict(correct=valid and answer == job['expected'])
        if phase == 'plan':
            score = evaluate(job['task'], answer)
            score['correct'] = valid and score['correct']
            score['optimal'] = valid and score['optimal']
            d = content_diagnostic(job['task'], calls)
            score.update(diagnostic=d, diagnostic_correct=bool(d and d['correct']),
                         diagnostic_optimal=bool(d and d['optimal']))
        else:
            score['field_correct'] = {k: isinstance(answer, dict) and answer.get(k) == v for k, v in job['expected'].items()}
        records.append({k: job[k] for k in ('case', 'scenario', 'variant', 'repetition', 'expected')} |
                       score | dict(answer=answer, schema_valid=valid, request_error=row.get('error'),
                                    tool_call_count=len(calls), finish_reason=choices[0].get('finish_reason'),
                                    usage=row.get('response', {}).get('usage'), request_file=row['request_file']))
    aggregates = []
    groups = defaultdict(list)
    for r in records:
        groups['variant', r['variant']].append(r)
        groups['scenario_variant', r['scenario'] + '/' + r['variant']].append(r)
        groups['order_variant', str(r['repetition']) + '/' + r['variant']].append(r)
    for (dimension, value), rs in groups.items():
        a = dict(dimension=dimension, value=value, n=len(rs), correct=sum(r['correct'] for r in rs),
                 schema_valid=sum(r['schema_valid'] for r in rs), errors=sum(bool(r['request_error']) for r in rs))
        if phase == 'plan':
            a.update({k: sum(r[k] for r in rs) for k in ('optimal', 'diagnostic_correct', 'diagnostic_optimal')})
        else:
            a['field_correct'] = {k: sum(r['field_correct'][k] for r in rs) for k in rs[0]['field_correct']}
        aggregates.append(a)
    data = dict(phase=phase, requests=len(rows), hashes_match=True, design_checked=True, aggregates=aggregates, records=records)
    (out / 'audit.json').write_text(json.dumps(data, indent=2) + '\n')
    lines = [f'# Condition alignment: {phase}', '', '```json', json.dumps([a for a in aggregates if a['dimension'] == 'variant'], indent=2), '```', '']
    for r in records:
        lines += [f"## {r['case']} / {r['variant']} / order {r['repetition']}", '', '```json', json.dumps(r, indent=2), '```', '']
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps([a for a in aggregates if a['dimension'] == 'variant'], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('component', 'plan'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    jobs = build(args.phase)
    check_design(jobs)
    if args.dry_run:
        print(f'{args.phase}: {len(jobs)} matched requests; annotations contain original values only; shortest paths checked.')
    elif args.audit:
        audit(args.output, args.phase)
    else:
        execute(jobs, args.output.resolve(), dict(
            design='12 tasks x 2 skill/condition orders x 3 input representations. Separate current state and requirements; '
                   'add skill-local current-value maps; add rows pairing condition, required value, and current value. '
                   'Both augmented forms retain original fields. No satisfaction label, missing-condition answer, '
                   'candidate filtering, or oracle plan enters inputs. First 8 task worlds reproduce the previous probe; '
                   '4 add required-false and conjunction controls. All representations use the same initial-snapshot note.',
            phase=args.phase,
            limitations='Known deterministic rules and exact state. Two order variants, no independent repeat trials. '
                        'Value annotations are mechanically joined by exact state key; perception and identity resolution '
                        'are outside scope. Plan goal attainment is primary, shortest length secondary.',
            probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
