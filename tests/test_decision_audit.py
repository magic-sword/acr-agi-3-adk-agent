"""Audit actual ADK execution, including final results and rejected skill loads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from scripts.analyze_agent import read_records, render
from scripts.eval_reporting import diagnostics
from test_completion_tools import call
from test_evidence_reasoning import observation


class DecisionAuditTests(unittest.TestCase):
    def test_skill_success_failure_final_submission_and_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct',
                                       log_dir=directory, max_calls=1)
            self.addCleanup(runtime.close)
            responses = iter([
                call('load_skill', {'skill_name': 'effect-revaluation'}, 'bad'),
                call('load_skill', {'skill_name': 'goal-inference'}, 'good'),
                call('submit_decision', {'action': {'action': 'UP', 'reason': '<evidence> explore UP'},
                     'prediction': 'Object moves up', 'notebook': 'Goal uncertain.'}, 'final'),
            ])
            with patch.object(LocalVisionLlm, '_complete', lambda model, payload:
                              {'choices': [{'message': {'tool_calls': [next(responses)]}}]}):
                first = runtime.decide(observation())
                self.assertEqual(runtime.decide(observation()), first)
            runtime.decide(observation(1, 1, state='WIN'))
            turns = read_records(root / f'{runtime.session_id}.jsonl')
            self.assertEqual(len(turns), 2)
            self.assertEqual(turns[0]['trace'], ['OBSERVE', 'DECIDE', 'COMMIT'])
            self.assertEqual(turns[1]['trace'], ['OBSERVE', 'COMMIT'])
            self.assertIn('VERIFY', turns[0]['internal_trace'])
            executions = turns[0]['tool_executions']
            self.assertEqual([t['status'] for t in executions], ['error', 'success', 'success'])
            self.assertTrue(executions[-1]['result']['accepted'])
            self.assertEqual(executions[0]['result']['error_code'], 'SKILL_NOT_FOUND')
            self.assertEqual(turns[1]['previous_outcome']['decision_id'], first['decision_id'])
            self.assertTrue(turns[1]['previous_outcome']['frame_changed'])
            self.assertIsNone(turns[1]['previous_outcome']['prediction_verified'])
            events = read_records(root / f'{runtime.session_id}.tools.jsonl')
            self.assertEqual(len(events), 6)
            self.assertEqual(events[0]['status'], 'started')
            self.assertEqual(events[-1]['status'], 'success')
            self.assertTrue(all(e['observation_id'] == turns[0]['observation_id'] for e in events))
            model = read_records(root / f'{runtime.session_id}.model.jsonl')[0]
            self.assertEqual(model['tool_executions'], executions)
            metrics = diagnostics(root)
            self.assertEqual(metrics['loaded_skills'], {'goal-inference': 1})
            self.assertEqual(metrics['tool_execution_errors'], 1)
            self.assertEqual(metrics['committed_actions'], 1)
            self.assertEqual(metrics['decisions_missing_reason'], 0)
            html = render(root)
            self.assertIn('&lt;evidence&gt;', html)
            self.assertNotIn('<evidence>', html)

    def test_exception_and_retry_preserve_executions_even_with_reused_call_id(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct',
                                       log_dir=directory, max_calls=2)
            self.addCleanup(runtime.close)
            def observe_current():
                raise RuntimeError('evidence unavailable')
            runtime.observe_current = observe_current
            responses = iter([
                call('observe_current', {}, 'reused'),
                call('submit_decision', {'action': {'action': 'UP'}, 'prediction': 'move'}, 'reused'),
            ])
            with patch.object(LocalVisionLlm, '_complete', lambda model, payload:
                              {'choices': [{'message': {'tool_calls': [next(responses)]}}]}):
                result = runtime.decide(observation())
            self.assertEqual(result['status'], 'action')
            tools = runtime.memory.history[-1]['tool_executions']
            self.assertEqual([t['call_index'] for t in tools], [1, 2])
            self.assertEqual([t['status'] for t in tools], ['error', 'success'])
            self.assertIn('evidence unavailable', tools[0]['result']['error'])

    def test_boundary_is_not_a_successful_prediction_and_host_has_no_skill_loads(self):
        runtime = CognitiveRuntime('test')
        self.addCleanup(runtime.close)
        runtime.decide(observation())
        runtime.decide(observation(1, 1, levels_completed=1))
        row = runtime.memory.history[-1]
        self.assertTrue(row['previous_outcome']['boundary'])
        self.assertIsNone(row['previous_outcome']['frame_changed'])
        self.assertEqual(row['tool_executions'], [])

    def test_interrupted_tool_is_visible_without_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run.tools.jsonl').write_text(json.dumps({
                'event': 'tool_started', 'status': 'started', 'tool': 'load_skill',
                'arguments': {'skill_name': 'goal-inference'}}) + '\n{"incomplete":')
            html = render(root)
            self.assertIn('tool_started', html)
            self.assertIn('goal-inference', html)
            self.assertEqual(diagnostics(root)['committed_actions'], 0)
