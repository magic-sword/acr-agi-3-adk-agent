"""Follow-up: equivalent prose versus JSON, irrelevant candidates, and chain depth."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from probe_abstract_skill_planning import TOOL, build as baseline_build, solve, task_for
from probe_incremental_planning_inputs import execute


def prose(task):
    def facts(values):
        return ', '.join(f'{k} = {json.dumps(v)}' for k, v in values.items()) or 'none'
    lines = [task['rules'], 'Current state: ' + facts(task['current_state']),
             'Goal: ' + facts(task['goal']), 'Available skills:']
    for skill in task['available_skills']:
        lines.append(f"- verb={skill['verb']}, target={skill['target']}. "
                     f"Requires: {facts(skill['requires'])}. Effects: {facts(skill['effects'])}.")
    return '\n'.join(lines)


def chain(depth):
    names = ['part_Q', 'part_A', 'part_M', 'part_B', 'part_Z'][:depth]
    return dict(rules=task_for('closed_gate', 'descriptive')['rules'],
                current_state={n + '_ready': False for n in names},
                goal={names[-1] + '_ready': True},
                available_skills=[dict(verb='prepare', target=n,
                                       requires={} if i == 0 else {names[i-1] + '_ready': True},
                                       effects={n + '_ready': True}) for i, n in enumerate(names)])


def build():
    baselines = baseline_build()
    template = next(j['payload'] for j in baselines if j['variant'] == 'direct')
    jobs = []
    tasks = []
    for scenario in ('closed_gate', 'gate_open', 'at_switch', 'other_goal'):
        for scope in ('full', 'relevant_only'):
            task = task_for(scenario, 'descriptive')
            if scope == 'relevant_only':
                target, destination = ('right_switch', 'exit_B') if scenario == 'other_goal' else ('left_switch', 'exit_A')
                task['available_skills'] = [s for s in task['available_skills'] if s['target'] in (target, destination)]
            tasks.append((scenario, scope, task))
    for depth in (1, 2, 3, 5):
        tasks.append((f'chain_{depth}', 'chain', chain(depth)))
    for scenario, scope, original in tasks:
        for reverse in (False, True):
            task = deepcopy(original)
            if reverse:
                task['available_skills'].reverse()
            for representation in ('json', 'prose'):
                payload = deepcopy(template)
                payload['messages'][0]['content'][1]['text'] = (
                    json.dumps(task, separators=(',', ':')) if representation == 'json' else prose(task))
                jobs.append(dict(case=f'{scenario}-{scope}', scenario=scenario, naming='descriptive',
                                 variant=scope + '_' + representation, representation=representation,
                                 scope=scope, repetition=int(reverse), candidate_order='reverse' if reverse else 'forward',
                                 oracle_plan=solve(task), task=task, payload=payload))
    random.Random(2909202672).shuffle(jobs)
    assert len(jobs) == 48
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    jobs = build()
    if args.dry_run:
        print(json.dumps(dict(requests=len(jobs), shortest_lengths=sorted({len(j['oracle_plan']) for j in jobs}))))
        return
    execute(jobs, args.output.resolve(), dict(
        design='Adaptive follow-up after baseline errors. 4 original tasks x full/relevant-only skills x JSON/prose '
               'x 2 orders = 32. Simple monotone prerequisite chains of depths 1,2,3,5 x JSON/prose x 2 orders = 16. '
               'Original direct question, schema, model, temperature and token budget unchanged. '
               'Eight full-JSON requests exactly repeat the descriptive direct baseline to check reproducibility.',
        limitations='Relevant-only candidates are filtered by the experimenter and do not demonstrate autonomous '
                    'retrieval. Monotone chains omit location overwrite, resource consumption and alternatives. '
                    'Prose preserves every rule, state, goal, skill condition and effect; it changes serialization only.',
        probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
