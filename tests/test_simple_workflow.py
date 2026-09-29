import unittest
from unittest.mock import patch

from agent.cognition.simple_workflow import SimpleRuntime, NextAction, click_point
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs, ack, context, call


class SimpleWorkflowTests(unittest.TestCase):
    def runtime(self):
        r = SimpleRuntime('test', 'local/qwen3-vl-4b-instruct', proposal_mode='program')
        self.addCleanup(r.close)
        return r

    def test_one_call_per_action_and_acknowledged_feedback(self):
        r = self.runtime(); contexts = []
        def respond(model, payload):
            c = context(payload); contexts.append(c)
            return call('submit_next_action', dict(observation_id=c['observation_id'],
                interpretation='No movement observed.' if c['recent_trials'] else '',
                question='Does UP move the actor?', action='UP', object_id=None,
                expected_effect='The actor may move upwards.'))
        with patch.object(LocalVisionLlm, '_complete', respond):
            self.assertEqual(r.decide(obs())['action'], 'ACTION1')
            self.assertEqual(r.decide(obs())['action'], 'ACTION1')
            self.assertEqual(len(contexts), 1)
            ack(r)
            self.assertEqual(r.decide(obs(1))['action'], 'ACTION1')
        self.assertEqual([c['work'] for c in contexts], ['act', 'act'])
        self.assertTrue(contexts[1]['recent_trials'][0]['results'][0]['acknowledged'])
        self.assertEqual(contexts[1]['recent_trials'][0]['results'][0]['changed_cell_count'], 0)
        self.assertEqual(r.memory.trial_ledger[-1]['interpretation'], 'No movement observed.')

    def test_click_uses_current_mask_without_fast_model(self):
        r = self.runtime()
        def respond(model, payload):
            c = context(payload)
            return call('submit_next_action', dict(observation_id=c['observation_id'],
                interpretation='', question='Does this object react?', action='CLICK',
                object_id=r._clickable_ids()[0], expected_effect='It may move.'))
        with patch.object(LocalVisionLlm, '_complete', respond):
            result = r.decide(obs(grid=[[0, 2, 2, 0], [0, 2, 0, 0]]))
        self.assertEqual(result['action'], 'ACTION6', r.errors)
        self.assertEqual(r.memory.model_calls, 1)
        binding = r.memory.pending['binding']
        obj = next(o for o in r.memory.object_memory['objects'] if o['object_id'] == binding['object_id'])
        self.assertEqual((result['x'], result['y']), click_point(obj, r.obs))

    def test_centroid_hole_and_missing_mask(self):
        obj = dict(mask_runs=[[0, 0, 3], [1, 0, 1], [1, 2, 3], [2, 0, 3]])
        self.assertNotEqual(click_point(obj, dict(width=3, height=3)), (1, 1))
        with self.assertRaises(ValueError): click_point(dict(mask_runs=[]), dict(width=3, height=3))

    def test_unknown_target_and_unavailable_control_rejected(self):
        r = self.runtime(); r._receive(obs())
        base = dict(observation_id=r.obs['observation_id'], interpretation='', question='Test?',
                    action='CLICK', object_id='missing', expected_effect='May change.')
        with self.assertRaises(ValueError): r.validate_stage('act', NextAction(**base))
        with self.assertRaises(ValueError):
            r.validate_stage('act', NextAction(**{**base, 'action':'DOWN', 'object_id':None}))

    def test_no_click_mask_stops_without_model_call(self):
        r = self.runtime(); observation = obs(); observation['available_actions'] = ['ACTION6']
        with patch.object(r, '_clickable_ids', return_value=[]), patch.object(LocalVisionLlm, '_complete') as model:
            result = r.decide(observation)
        self.assertEqual(result['reason'], 'no_grounded_action')
        model.assert_not_called()

    def test_unacknowledged_action_stops_before_next_model_call(self):
        r = self.runtime()
        def respond(model, payload):
            c = context(payload)
            return call('submit_next_action', dict(observation_id=c['observation_id'],
                interpretation='', question='Move?', action='UP', object_id=None, expected_effect='Move up.'))
        with patch.object(LocalVisionLlm, '_complete', respond):
            r.decide(obs())
            result = r.decide(obs(1))
        self.assertEqual(result['reason'], 'execution_outcome_unknown')
        self.assertEqual(r.memory.model_calls, 1)

    def test_default_factory_uses_simple_runtime(self):
        from agent.adk_policy import create_runtime
        with patch.dict('os.environ', {'COGNITION_PLANNING_COMPARISON':'',
                'COGNITION_MEMORY_COMPARISON':'', 'COGNITION_ACTION_UPDATE_COMPARISON':''}):
            r = create_runtime('test'); self.addCleanup(r.close)
        self.assertIsInstance(r, SimpleRuntime)
