"""Replay returned macro plans and audit frozen requests without repairing outputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics

import jsonschema

from probe_abstract_skill_planning import TOOL, build, holds, solve, task_for, transition
from probe_repeated_click_inputs import digest


def evaluate(task, answer):
    errors = [e.message for e in jsonschema.Draft202012Validator(
        TOOL[0]['function']['parameters']).iter_errors(answer)]
    gold = solve(task)
    expected_status = 'unreachable' if gold is None else 'already_satisfied' if not gold else 'plan'
    state = dict(task['current_state'])
    trace = []
    execution_error = None
    steps = answer.get('steps', []) if isinstance(answer, dict) else []
    if not isinstance(steps, list):
        steps = []
    for index, step in enumerate(steps):
        skill = next((s for s in task['available_skills']
                      if isinstance(step, dict) and all(step.get(k) == s[k] for k in ('verb', 'target'))), None)
        if skill is None:
            execution_error = dict(step=index, kind='unknown_skill', proposed=step)
            break
        after = transition(state, skill)
        if after is None:
            execution_error = dict(step=index, kind='unsatisfied_precondition', proposed=step,
                                   unmet={k: dict(required=v, actual=state.get(k))
                                          for k, v in skill['requires'].items() if state.get(k) != v})
            break
        trace.append(dict(step=step, before=state, after=after))
        state = after
    status = answer.get('status') if isinstance(answer, dict) else None
    executable = execution_error is None
    reached = executable and holds(state, task['goal'])
    content_correct = ((not steps) if gold is None else reached)
    correct = not errors and status == expected_status and content_correct and (not steps if expected_status != 'plan' else True)
    optimal = correct and (gold is None or len(steps) == len(gold))
    return dict(schema_errors=errors, expected_status=expected_status, status=status,
                oracle_length=None if gold is None else len(gold), returned_length=len(steps),
                executable=executable, goal_reached=reached, correct=correct, optimal=optimal,
                execution_error=execution_error, trace=trace, final_state=state)


def self_check():
    for scenario in ('closed_gate', 'gate_open', 'at_switch', 'other_goal', 'already_done',
                     'unreachable', 'needs_recharge', 'composite_available'):
        task = task_for(scenario, 'descriptive')
        gold = solve(task)
        status = 'unreachable' if gold is None else 'already_satisfied' if not gold else 'plan'
        assert evaluate(task, dict(status=status, steps=gold or []))['optimal']
    task = task_for('closed_gate', 'descriptive')
    gold = solve(task)
    assert not evaluate(task, dict(status='already_satisfied', steps=gold))['correct']
    assert not evaluate(task, dict(status='plan', steps=gold[:1]))['correct']
    assert evaluate(task, dict(status='plan', steps=gold[1:]))['execution_error']['kind'] == 'unsatisfied_precondition'
    assert evaluate(task, dict(status='plan', steps=[dict(verb='teleport', target='exit_A')]))['execution_error']['kind'] == 'unknown_skill'
    assert not evaluate(task, dict(status='unreachable', steps=[]))['correct']
    assert not evaluate(task_for('unreachable', 'descriptive'), dict(status='plan', steps=[]))['correct']
    depleted = task_for('needs_recharge', 'descriptive')
    assert evaluate(depleted, dict(status='plan', steps=gold))['execution_error']['unmet']['charged']['actual'] is False
    redundant = [gold[0]] + gold
    score = evaluate(task, dict(status='plan', steps=redundant))
    assert score['correct'] and not score['optimal']
    calls = [dict(function=dict(name='submit_plan', arguments=json.dumps(dict(status='plan', steps=[s])))) for s in gold]
    assert content_diagnostic(task, calls)['optimal']
    truncated = json.dumps(dict(status='plan', steps=gold))[:-1] + ', "reason": "unfinished'
    diagnostic = content_diagnostic(task, [dict(function=dict(name='submit_plan', arguments=truncated))])
    assert diagnostic['optimal'] and diagnostic['mode'] == 'closed_fields_before_truncation'
    missing_steps = '{"status":"plan","steps":['
    assert content_diagnostic(task, [dict(function=dict(name='submit_plan', arguments=missing_steps))]) is None


def content_diagnostic(task, calls):
    """Auxiliary interpretation only, never change the strict response score.

    Concatenate reported step lists in their emitted order. If a trailing reason
    was truncated, read only fully closed status/steps JSON values; do not infer
    missing actions or edit any target. Multiple calls are NOT valid API output.
    """
    answers = []
    extracted = False
    try:
        for call in calls:
            if call['function']['name'] != 'submit_plan':
                return None
            raw = call['function']['arguments']
            try:
                answer = json.loads(raw)
            except json.JSONDecodeError:
                answer = {}
                for key in ('status', 'steps'):
                    matches = list(re.finditer('"' + key + r'"\s*:', raw))
                    if len(matches) != 1:
                        return None
                    answer[key] = json.JSONDecoder().raw_decode(raw[matches[0].end():].lstrip())[0]
                extracted = True
            answers.append(answer)
        if not answers:
            return None
        if len(answers) > 1 and any(a.get('status') != 'plan' for a in answers):
            return None
        combined = dict(status=answers[0]['status'], steps=[s for a in answers for s in a['steps']])
        score = evaluate(task, combined)
        return dict(mode='concatenated_calls' if len(calls) > 1 else 'closed_fields_before_truncation' if extracted else 'single_call',
                    answer=combined, **score)
    except (KeyError, TypeError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--followup', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.output is None:
        assert args.self_check
        print('Evaluator self-check passed: gold paths, status mismatch, partial plans, preconditions, unknown skills, resource depletion and nonoptimal success.')
        return
    out = args.output
    if args.followup:
        from probe_abstract_skill_inputs import build as followup_build
        jobs = followup_build()
    else:
        jobs = build()
    plan = json.loads((out / 'plan.json').read_text())
    rows = json.loads((out / 'results.json').read_text())
    assert len(rows) == len(jobs) == plan['requests']
    records = []
    for i, (job, row) in enumerate(zip(jobs, rows)):
        request = json.loads((out / row['request_file']).read_text())
        assert digest(request) == digest(job['payload']) == row['request_sha256'] == plan['jobs'][i]['request_sha256']
        for k in ('case', 'scenario', 'naming', 'variant', 'repetition', 'oracle_plan'):
            assert row[k] == job[k]
        task = job['task'] if args.followup else json.loads(request['messages'][0]['content'][1]['text'])
        score = evaluate(task, row.get('answer'))
        choices = row.get('response', {}).get('choices', [])
        calls = choices[0]['message'].get('tool_calls', []) if choices else []
        tool_valid = len(calls) == 1 and row.get('tool') == 'submit_plan'
        if not tool_valid:
            score.update(correct=False, optimal=False)
        records.append({k: row[k] for k in ('case', 'scenario', 'naming', 'variant', 'repetition', 'request_file', 'seconds')} |
                       score | dict(answer=row.get('answer'), request_error=row.get('error'), tool_valid=tool_valid,
                                    tool_call_count=len(calls), content_diagnostic=content_diagnostic(task, calls),
                                    finish_reason=choices[0].get('finish_reason') if choices else None,
                                    usage=row.get('response', {}).get('usage')))
    groups = defaultdict(list)
    for r in records:
        groups['overall', 'all'].append(r)
        for dim in ('variant', 'scenario', 'naming'):
            groups[dim, r[dim]].append(r)
        groups['scenario_variant', r['scenario'] + '/' + r['variant']].append(r)
        groups['order', str(r['repetition'])].append(r)
    aggregates = []
    for (dimension, value), rs in groups.items():
        aggregates.append(dict(dimension=dimension, value=value, n=len(rs),
                               correct=sum(r['correct'] for r in rs), optimal=sum(r['optimal'] for r in rs),
                               diagnostic_correct=sum(bool(r['content_diagnostic'] and r['content_diagnostic']['correct']) for r in rs),
                               diagnostic_optimal=sum(bool(r['content_diagnostic'] and r['content_diagnostic']['optimal']) for r in rs),
                               schema_valid=sum(not r['schema_errors'] and r['tool_valid'] for r in rs),
                               errors=sum(r['request_error'] is not None for r in rs),
                               median_seconds=statistics.median(r['seconds'] for r in rs)))
    pairs = []
    for scenario in sorted({r['scenario'] for r in records}):
        for naming in ('descriptive', 'opaque'):
            for order in (0, 1):
                rs = {r['variant']: r for r in records if (r['scenario'], r['naming'], r['repetition']) == (scenario, naming, order)}
                if not {'direct', 'backward_hint'} <= rs.keys():
                    continue
                pairs.append(dict(scenario=scenario, naming=naming, order=order,
                                  direct_correct=rs['direct']['correct'], backward_correct=rs['backward_hint']['correct'],
                                  direct_optimal=rs['direct']['optimal'], backward_optimal=rs['backward_hint']['optimal']))
    audit = dict(requests=len(records), reconstructed_hashes_match=True, aggregates=aggregates,
                 prompt_pairs=pairs, records=records)
    (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Known-rule abstract-skill planning', '',
             'Strict scores require one valid tool call. Diagnostic scores concatenate emitted calls in order or read fully closed status/steps before a truncated reason; they are not accepted runtime plans.', '',
             '| Dimension | Condition | N | Strict correct | Strict optimal | Diagnostic correct | Diagnostic optimal | Schema |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for a in aggregates:
        lines.append(f"| {a['dimension']} | {a['value']} | {a['n']} | {a['correct']} | {a['optimal']} | {a['diagnostic_correct']} | {a['diagnostic_optimal']} | {a['schema_valid']} |")
    lines += ['', '## Returned plans', '']
    for r in records:
        lines += [f"### {r['case']} / {r['variant']} / order {r['repetition']}", '',
                  f"Correct={r['correct']}; optimal={r['optimal']}; oracle length={r['oracle_length']}", '',
                  '```json', json.dumps(dict(answer=r['answer'], execution_error=r['execution_error']), ensure_ascii=False, indent=2), '```', '']
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(aggregates, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
