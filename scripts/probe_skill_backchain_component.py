"""Isolate one level of backward planning; no host-computed answer in inputs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from probe_abstract_skill_planning import build as baseline_build, holds
from probe_incremental_planning_inputs import execute

TOOL = [{"type": "function", "function": {
    "name": "submit_goal_link", "description": "Report one backward link from the goal to a skill and its unmet conditions.",
    "parameters": {"type": "object", "additionalProperties": False,
        "properties": {
            "goal_holds_now": {"type": "boolean"},
            "last_skill": {"anyOf": [{"type": "null"}, {"type": "object", "additionalProperties": False,
                "properties": {"verb": {"type": "string"}, "target": {"type": "string"}},
                "required": ["verb", "target"]}]},
            "unmet_requirements": {"type": "object", "additionalProperties": {"type": ["string", "boolean"]}}},
        "required": ["goal_holds_now", "last_skill", "unmet_requirements"]}}}]


def build():
    jobs = []
    for original in baseline_build():
        if original['variant'] != 'direct' or original['naming'] != 'descriptive':
            continue
        task = json.loads(original['payload']['messages'][0]['content'][1]['text'])
        payload = deepcopy(original['payload'])
        payload['messages'][0]['content'][0]['text'] = (
            'Do only one level of backward planning. Check whether the goal holds in the current state. '
            'If it does, set goal_holds_now=true, last_skill=null, and unmet_requirements={}. '
            'Otherwise choose an available skill whose effects directly establish the goal. '
            'Report its exact verb and target as last_skill, and only the entries in its requires '
            'that do not already hold in the current state as unmet_requirements. '
            'This is the final skill of a possible plan, not necessarily the next executable skill. '
            'Do not construct the rest of the plan.')
        payload['tools'] = deepcopy(TOOL)
        goal_holds = holds(task['current_state'], task['goal'])
        matches = [s for s in task['available_skills'] if holds(s['effects'], task['goal'])]
        assert len(matches) == 1
        last = matches[0]
        expected = dict(goal_holds_now=goal_holds,
                        last_skill=None if goal_holds else {k: last[k] for k in ('verb', 'target')},
                        unmet_requirements={} if goal_holds else
                        {k: v for k, v in last['requires'].items() if task['current_state'].get(k) != v})
        jobs.append({k: original[k] for k in ('case', 'scenario', 'naming', 'repetition', 'candidate_order')} |
                    dict(variant='one_link', expected=expected, payload=payload))
    random.Random(2909202673).shuffle(jobs)
    assert len(jobs) == 16
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit', action='store_true')
    args = p.parse_args()
    jobs = build()
    if args.audit:
        import jsonschema
        from probe_repeated_click_inputs import digest
        rows = json.loads((args.output / 'results.json').read_text())
        plan = json.loads((args.output / 'plan.json').read_text())
        assert len(rows) == len(jobs) == plan['requests']
        records = []
        for i, (job, row) in enumerate(zip(jobs, rows)):
            payload = json.loads((args.output / row['request_file']).read_text())
            assert digest(payload) == digest(job['payload']) == row['request_sha256'] == plan['jobs'][i]['request_sha256']
            answer = row.get('answer')
            errors = [e.message for e in jsonschema.Draft202012Validator(TOOL[0]['function']['parameters']).iter_errors(answer)]
            calls = row.get('response', {}).get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
            valid = not errors and len(calls) == 1 and row.get('tool') == 'submit_goal_link'
            records.append({k: job[k] for k in ('case', 'scenario', 'repetition', 'expected')} |
                           dict(answer=answer, schema_valid=valid, correct=valid and answer == job['expected'],
                                field_correct={k: isinstance(answer, dict) and answer.get(k) == v for k, v in job['expected'].items()},
                                request_file=row['request_file'], error=row.get('error')))
        audit = dict(requests=len(records), reconstructed_hashes_match=True, correct=sum(r['correct'] for r in records),
                     schema_valid=sum(r['schema_valid'] for r in records), records=records)
        (args.output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        print(json.dumps({k: v for k, v in audit.items() if k != 'records'}))
        return
    execute(jobs, args.output.resolve(), dict(
        design='Adaptive diagnostic: same 8 descriptive baseline tasks x 2 orders; replace full-plan question '
               'with one-level backward link and current unmet-condition extraction. Exact same world information.',
        limitations='This does not test recursive search, autonomous decomposition, or multi-call execution. '
                    'The final action has only one immediate prerequisite in these tasks. '
                    'The unreachable task still has a final action; missing earlier links are outside this local question.',
        probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
