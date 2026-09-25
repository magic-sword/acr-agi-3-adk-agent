"""Bounded real-Qwen tool-use diagnostic with PROVIDED synthetic symbolic labels.

Checks write/induce -> backward search -> action submission, not visual perception
or ARC performance. The normal runtime and three-request limit are used.
"""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.cognition.workflow import CognitiveRuntime
from scripts.eval_reporting import read_jsonl, tool_requests


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/verification/causal-planning/model-check'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='causal-model-check-') as directory:
        runtime = CognitiveRuntime('causal-diagnostic-v1', max_calls=1, seconds=45,
                                   knowledge_dir=directory, log_dir=str(args.output))
        def obs(step, color):
            return dict(game_id='causal-diagnostic-v1', state='NOT_FINISHED', step=step,
                        grid=[[color, 0], [0, 0]], levels_completed=0,
                        available_actions=['ACTION1'], remaining_actions=5)
        try:
            runtime.decide(obs(0, 0))
            before = runtime.memory.last_observation['observation_id']
            runtime.decide(obs(1, 1))
            after = runtime.memory.last_observation['observation_id']
            runtime.memory.notebook = (
                'Synthetic diagnostic with provided symbolic labels (not a perception test). '
                f'At observation {before}, powered=true and gate_open=false. '
                f'The UP action produced observation {after}, with gate_open=true. '
                'The abstract action name is activate_switch. The CURRENT state is powered=true, gate_open=false. '
                'Diagnostic task: first use causal_memory(operation=update, transitions=[the above trial], '
                'induce=true); then use plan_backward with current powered=true/gate_open=false and '
                'goal gate_open=true; finally submit the resulting next input. Use exactly those symbols. '
                'Source action and stage_clear are supplied by the host. No extra skill loading or observation '
                'calls are needed because the example labels are provided.'
            )
            runtime.model = 'local/qwen3-vl-4b-instruct'
            result = runtime.decide(obs(2, 0))
            records = read_jsonl(args.output / f'{runtime.session_id}.model.jsonl')
            names = [c['function']['name'] for record in records for e in record.get('exchanges', [])
                     for c in tool_requests(e)]
            tool_results = [json.loads(m['content']) for record in records for e in record.get('exchanges', [])
                            for m in e.get('tool_results', [])]
            plans = [r for r in tool_results if r.get('status') == 'plan' and r.get('plan')]
            report = {'scope': __doc__, 'result': result, 'tools': names,
                      'confirmed_plan_results': plans,
                      'model_calls': len(records), 'http_requests': sum(r['http_requests'] for r in records),
                      'passed': (result.get('action') == 'ACTION1' and 'causal_memory' in names
                                 and 'plan_backward' in names and bool(plans)
                                 and bool(runtime.causal.read()['transitions']))}
            (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps(report, ensure_ascii=False))
            return 0 if report['passed'] else 1
        finally:
            runtime.close()


if __name__ == '__main__':
    raise SystemExit(main())
