"""Known-rule macro-skill planning: frozen inputs, counterfactuals and BFS oracle.

No game actions, image recognition, model repair, or runtime changes. The oracle
is an evaluator only: neither its paths nor its decisions enter model requests.
"""
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from probe_incremental_planning_inputs import execute, minimal_payload, read_cases

TOOL = [{"type": "function", "function": {
    "name": "submit_plan", "description": "Return a complete sequence of available skills, or report that the goal is unreachable.",
    "parameters": {"type": "object", "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["plan", "already_satisfied", "unreachable"]},
            "steps": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "properties": {"verb": {"type": "string"}, "target": {"type": "string"}},
                "required": ["verb", "target"]}},
            "reason": {"type": "string"}},
        "required": ["status", "steps"]}}}]


def holds(state, conditions):
    return all(state.get(k) == v for k, v in conditions.items())


def transition(state, skill):
    if not holds(state, skill['requires']):
        return None
    return state | skill['effects']


def solve(task):
    """Exhaustive shortest-path search on the finite, explicitly given world."""
    queue = deque([(task['current_state'], [])])
    seen = set()
    while queue:
        state, path = queue.popleft()
        key = json.dumps(state, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if holds(state, task['goal']):
            return path
        for skill in task['available_skills']:
            after = transition(state, skill)
            if after is not None:
                queue.append((after, path + [{k: skill[k] for k in ('verb', 'target')}]))
    return None


def task_for(scenario, naming):
    # Abstract names deliberately reverse the lexical order of paired targets.
    names = (dict(start='start', left='left_switch', right='right_switch',
                  station='charging_station', exit_a='exit_A', exit_b='exit_B')
             if naming == 'descriptive' else
             dict(start='site_M', left='object_Z', right='object_A',
                  station='object_K', exit_a='site_T', exit_b='site_B'))
    n = names
    skills = []

    def add(verb, target, requires, effects):
        skills.append(dict(verb=verb, target=n[target], requires=requires, effects=effects))

    for target in ('left', 'right', 'station'):
        add('move_to', target, {}, dict(location=n[target]))
    for target, gate in (('exit_a', 'gate_A_open'), ('exit_b', 'gate_B_open')):
        add('move_to', target, {gate: True}, dict(location=n[target]))
    for target, gate in (('left', 'gate_A_open'), ('right', 'gate_B_open')):
        add('activate', target, dict(location=n[target], charged=True), {gate: True, 'charged': False})
    add('recharge_at', 'station', dict(location=n['station']), dict(charged=True))
    state = dict(location=n['start'], gate_A_open=False, gate_B_open=False, charged=True)
    goal = dict(location=n['exit_a'])
    if scenario == 'gate_open':
        state['gate_A_open'] = True
    elif scenario == 'at_switch':
        state['location'] = n['left']
    elif scenario == 'other_goal':
        goal['location'] = n['exit_b']
    elif scenario == 'already_done':
        state.update(location=n['exit_a'], gate_A_open=True)
    elif scenario == 'unreachable':
        skills = [s for s in skills if not (s['verb'] == 'activate' and s['target'] == n['left'])]
    elif scenario == 'needs_recharge':
        state['charged'] = False
    elif scenario == 'composite_available':
        # A provided, verified macro wraps move_to(left) + activate(left).
        add('go_and_activate', 'left', dict(charged=True),
            dict(location=n['left'], gate_A_open=True, charged=False))
    elif scenario != 'closed_gate':
        raise ValueError(scenario)
    return dict(
        rules='This is a fully observed deterministic planning world. All available skills and their complete '
              'conditions and effects are listed. A skill can run only when every requires entry equals the '
              'current value. On completion its effects overwrite those state entries; every other entry stays '
              'unchanged. There are no hidden actions, effects, obstacles, or delays. Each skill is a complete '
              'abstract behavior executed by a fast controller, not a single button press. Each invocation costs one.',
        current_state=state, goal=goal, available_skills=skills)


def build():
    original = next(iter(read_cases().values()))['original']
    base = minimal_payload(original)
    base['max_tokens'] = 1200
    scenarios = ('closed_gate', 'gate_open', 'at_switch', 'other_goal',
                 'already_done', 'unreachable', 'needs_recharge', 'composite_available')
    jobs = []
    for scenario in scenarios:
        for naming in ('descriptive', 'opaque'):
            for reverse in (False, True):
                task = task_for(scenario, naming)
                if reverse:
                    task['available_skills'].reverse()
                gold = solve(task)
                for variant in ('direct', 'backward_hint'):
                    question = ('Plan how to satisfy the goal using the fewest available skill invocations. '
                                'Return the full sequence. Use the exact verb and target of each available skill. '
                                'If the goal already holds, return already_satisfied and no steps. '
                                'If no sequence can satisfy it, return unreachable and no steps.')
                    if variant == 'backward_hint':
                        question += (' Work backward from the goal through skill effects and their required '
                                     'conditions, then return the executable sequence in forward order.')
                    payload = dict(**deepcopy(base), messages=[dict(role='user', content=[
                        dict(type='text', text=question),
                        dict(type='text', text=json.dumps(task, separators=(',', ':')))])],
                        tools=deepcopy(TOOL), tool_choice='required', parallel_tool_calls=False)
                    jobs.append(dict(case=f'{scenario}-{naming}', scenario=scenario, naming=naming,
                                     variant=variant, repetition=int(reverse),
                                     candidate_order='reverse' if reverse else 'forward',
                                     oracle_plan=gold, payload=payload))
    random.Random(2909202671).shuffle(jobs)
    assert len(jobs) == 64
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    jobs = build()
    if args.dry_run:
        expected = dict(closed_gate=3, gate_open=1, at_switch=2, other_goal=3,
                        already_done=0, unreachable=None, needs_recharge=5, composite_available=2)
        for job in jobs:
            gold = job['oracle_plan']
            assert (len(gold) if gold is not None else None) == expected[job['scenario']]
        print(json.dumps(dict(requests=len(jobs), optimal_lengths=expected), indent=2))
        return
    execute(jobs, args.output.resolve(), dict(
        design='8 known-rule macro-planning tasks x 2 naming schemes x 2 candidate orders x 2 prompts. '
               'Same one-call interface; direct vs one backward-planning hint. Temperature zero. '
               'No identical-request repetitions; these are paired sensitivity probes, not independent game trials.',
        limitations='All skills, complete effects, target bindings and current state are supplied accurately. '
                    'No causal discovery, unknown effects, perception, runtime state split, or live execution tested. '
                    'Exact verb-target copying and structured output scored separately from executable goal attainment.',
        probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
