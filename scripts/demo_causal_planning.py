"""Synthetic symbolic demonstration; no LLM, real ARC game, or network required.

Records three controlled action outcomes, reopens persistent memory, finds a plan,
then removes one rule to expose an experiment gap. Not a game-performance benchmark.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.cognition.workflow import CognitiveRuntime
from agent.cognition.causal import CausalMemoryTool, BackwardPlanTool


def state(*true, false=()):
    return {'true': list(true), 'false': list(false)}


def obs(step):
    return {'game_id': 'symbolic-demo-v1', 'state': 'WIN' if step == 3 else 'NOT_FINISHED',
            'step': step, 'levels_completed': 1 if step == 3 else 0,
            'available_actions': [f'ACTION{min(step + 1, 3)}'], 'remaining_actions': 20,
            'grid': [[step, 0], [0, 0]]}


def invoke(tool, args):
    result = asyncio.run(tool.run_async(args=args, tool_context=None))
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result


def run(directory):
    runtime = CognitiveRuntime('symbolic-demo-v1', knowledge_dir=str(directory))
    try:
        ids, actions = [], []
        for step in range(4):
            actions.append(runtime.decide(obs(step)))
            ids.append(runtime.memory.last_observation['observation_id'])
        transitions = []
        for index, (name, before, after) in enumerate([
            ('pickup_key', state('at_key', false=['has_key']), state('has_key')),
            ('unlock_door', state('has_key', false=['door_open']), state('door_open', false=['has_key'])),
            ('enter_exit', state('door_open'), state('stage_clear')),
        ]):
            transitions.append({'before_id': ids[index], 'after_id': ids[index + 1],
                                'action_name': name, 'before': before, 'after': after})
        invoke(CausalMemoryTool(runtime), {'operation': 'update', 'transitions': transitions})
    finally:
        runtime.close()
    # Reopening is intentional: symbolic rules outlive a game session.
    runtime = CognitiveRuntime('symbolic-demo-v1', knowledge_dir=str(directory))
    try:
        runtime.decide(obs(0))
        query = {'observation_id': runtime.memory.last_observation['observation_id'],
                 'current': state('at_key', false=['has_key', 'door_open'])}
        plan = invoke(BackwardPlanTool(runtime), query)
        assert plan['status'] == 'plan', plan
        assert [s['action_name'] for s in plan['plan']] == ['pickup_key', 'unlock_door', 'enter_exit']
        rules = invoke(CausalMemoryTool(runtime), {})['rules']
        key_rule = next(r['id'] for r in rules if r['action_name'] == 'pickup_key')
        invoke(CausalMemoryTool(runtime), {'operation': 'update', 'delete_rule_ids': [key_rule]})
        gap = invoke(BackwardPlanTool(runtime), query)
        assert gap['status'] == 'knowledge_gap', gap
        assert any('has_key' in f['no_known_producer']['true'] for f in gap['frontier']), gap
        return {'kind': 'synthetic_symbolic_demo', 'observed_actions': actions,
                'plan_after_reopen': plan, 'gap_after_rule_deletion': gap}
    finally:
        runtime.close()


def main():
    import tempfile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/verification/causal-planning/demo.json'))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='arc-causal-demo-') as directory:
        result = run(directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Plan: pickup_key -> unlock_door -> enter_exit')
    print('After deleting pickup_key: missing producer for has_key')
    print(f'Evidence: {args.output}')


if __name__ == '__main__':
    main()
