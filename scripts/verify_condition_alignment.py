"""Verify condition-value interventions and summarize matched case pairs."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from probe_condition_alignment import build, check_design, VARIANTS, world
from probe_abstract_skill_planning import solve
from probe_repeated_click_inputs import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    summaries = {}
    all_rows = []
    for phase in ('component', 'plan'):
        out = ROOT / 'outputs' / f'condition-alignment-{phase}-20260929'
        jobs = build(phase)
        check_design(jobs)
        rows = json.loads((out / 'results.json').read_text())
        plan = json.loads((out / 'plan.json').read_text())
        audit = json.loads((out / 'audit.json').read_text())
        assert len(jobs) == len(rows) == len(audit['records']) == plan['requests'] == 72
        assert plan['probe_script_sha256'] == hashlib.sha256((ROOT / 'scripts/probe_condition_alignment.py').read_bytes()).hexdigest()
        assert plan['script_sha256'] == hashlib.sha256((ROOT / 'scripts/probe_incremental_planning_inputs.py').read_bytes()).hexdigest()
        for i, (job, row) in enumerate(zip(jobs, rows)):
            assert digest(job['payload']) == row['request_sha256'] == plan['jobs'][i]['request_sha256']
            assert digest(json.loads((out / row['request_file']).read_text())) == row['request_sha256']
        rs = {(r['scenario'], r['variant'], r['repetition']): r for r in audit['records']}
        pairs = []
        for first, second, key in (
            ('closed_gate', 'gate_open', 'current_state'),
            ('false_required_missing', 'false_required_ready', 'current_state'),
            ('conjunction_missing', 'conjunction_ready', 'current_state'),
            ('closed_gate', 'other_goal', 'goal')):
            a, b = world(first), world(second)
            assert a[key] != b[key]
            a[key] = b[key]
            assert a == b
            for variant in VARIANTS:
                for order in (0, 1):
                    left, right = rs[first, variant, order], rs[second, variant, order]
                    pairs.append(dict(pair=first + '/' + second, variant=variant, order=order,
                                      both_correct=left['correct'] and right['correct'],
                                      both_diagnostic_correct=None if phase == 'component' else
                                      left['diagnostic_correct'] and right['diagnostic_correct']))
        summaries[phase] = dict(aggregates=[a for a in audit['aggregates'] if a['dimension'] == 'variant'],
                               counterfactual_pairs=pairs)
        if phase == 'plan':
            incomplete = []
            for r in audit['records']:
                d = r['diagnostic']
                if not d or d['correct'] or d['execution_error'] is not None:
                    continue
                task = world(r['scenario'])
                before = solve(task)
                task['current_state'] = d['final_state']
                after = solve(task)
                if before is not None and after is not None and 0 < len(after) < len(before):
                    incomplete.append(dict(scenario=r['scenario'], variant=r['variant'], order=r['repetition'],
                                           remaining_before=len(before), remaining_after=len(after), answer=d['answer']))
            summaries[phase]['posthoc_incomplete_but_progressing'] = incomplete
        all_rows += rows
    initial = json.loads((ROOT / 'outputs/condition-alignment-component-20260929/runtime-snapshot.json').read_text())
    assert all(hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == sha for f, sha in initial.items())
    data = dict(requests=len(all_rows), request_hashes_match=True, matched_values_checked=True,
                runtime_and_notebook_unchanged=True,
                finish_reasons=dict(Counter(r['response']['choices'][0]['finish_reason'] for r in all_rows)),
                max_prompt_tokens=max(r['response']['usage']['prompt_tokens'] for r in all_rows),
                parse_or_request_errors=sum('error' in r for r in all_rows),
                tool_call_counts=dict(Counter(len(r['response']['choices'][0]['message'].get('tool_calls', [])) for r in all_rows)),
                results=summaries)
    (ROOT / 'outputs/condition-alignment-component-20260929/verification.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({k: v for k, v in data.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
