"""Check noncommuting effects without executing or trusting hypothetical actions."""
import json
import unittest
from unittest.mock import patch

from agent.cognition.planning import OrderQuery, check_order
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm


class PlanOrderTests(unittest.TestCase):
    def steps(self):
        return [
            {'id': 'place', 'requires': ['block_access'], 'adds': ['pad_occupied'],
             'removes': ['switch_access']},
            {'id': 'activate', 'requires': ['switch_access'], 'adds': ['gate_open']},
        ]

    def query(self, steps):
        return OrderQuery(initial_conditions=['block_access', 'switch_access'], steps=steps,
                          required_final_conditions=['gate_open', 'pad_occupied'])

    def test_order_that_removes_access_is_blocked_and_reverse_is_feasible(self):
        steps = self.steps()
        query = self.query(steps)
        before = query.model_dump()
        result = check_order(query)
        self.assertFalse(result['feasible_under_assumptions'])
        self.assertEqual(result['blocked_step'], 'activate')
        self.assertEqual(result['missing_conditions'], ['switch_access'])
        self.assertEqual(query.model_dump(), before)
        self.assertTrue(check_order(self.query(list(reversed(steps))))['feasible_under_assumptions'])

    def test_mutual_blocking_has_no_feasible_order(self):
        steps = self.steps()
        steps[1]['removes'] = ['block_access']
        for order in (steps, list(reversed(steps))):
            self.assertFalse(check_order(self.query(order))['feasible_under_assumptions'])

    def test_later_effect_can_destroy_a_required_final_condition(self):
        steps = self.steps()
        steps[0]['removes'].append('gate_open')
        result = check_order(self.query(list(reversed(steps))))
        self.assertFalse(result['feasible_under_assumptions'])
        self.assertIsNone(result['blocked_step'])
        self.assertEqual(result['missing_conditions'], ['gate_open'])

    def test_ambiguous_effects_and_step_ids_are_rejected(self):
        steps = self.steps()
        steps[0]['adds'].append('switch_access')
        with self.assertRaisesRegex(ValueError, 'both add and remove'):
            self.query(steps)
        with self.assertRaisesRegex(ValueError, 'IDs must be unique'):
            self.query([self.steps()[0], self.steps()[0]])

    def test_adk_returns_check_result_before_validated_submission(self):
        runtime = CognitiveRuntime('order-check', 'local/qwen3-vl-4b-instruct', max_calls=1)
        self.addCleanup(runtime.close)
        payloads = []

        def respond(model, payload):
            payloads.append(payload)
            if len(payloads) == 1:
                name, args = 'check_plan_order', self.query(self.steps()).model_dump()
            else:
                result = next(json.loads(m['content']) for m in payload['messages'] if m['role'] == 'tool')
                self.assertFalse(result['feasible_under_assumptions'])
                self.assertEqual(result['missing_conditions'], ['switch_access'])
                self.assertIsNone(runtime.turn.selected)
                self.assertEqual(runtime.turn.memory.plan, [])
                context = next(json.loads(p['text']) for m in payload['messages']
                               if m['role'] == 'user' and isinstance(m['content'], list)
                               for p in m['content'] if p.get('type') == 'text')
                name = 'submit_decision'
                args = {'action': {'action': 'UP'}, 'prediction': 'Test whether the input moves the target.'}
            return {'choices': [{'message': {'tool_calls': [{'id': str(len(payloads)), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}

        with patch.object(LocalVisionLlm, '_complete', respond):
            result = runtime.decide({'game_id': 'order-check', 'state': 'NOT_FINISHED', 'step': 0,
                                     'available_actions': ['ACTION1'], 'remaining_actions': 3, 'grid': [[0]]})
        self.assertEqual(result['action'], 'ACTION1')
        self.assertEqual(len(payloads), 2)
