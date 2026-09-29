from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from agent.cognition.memory_comparison import (
    MemoryComparisonRuntime, SeparatedMemoryRuntime, MemoryReview)
from agent.cognition.planning_comparison import OneActionPlan
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs, context, call, token, ack
from test_planning_comparison import EmptyProposer, plan


class MemoryComparisonTests(unittest.TestCase):
    def runtime(self, cls=SeparatedMemoryRuntime):
        runtime = cls('test', 'local/qwen3-vl-4b-instruct', sam_proposer=EmptyProposer())
        self.addCleanup(runtime.close)
        return runtime

    def review(self, runtime, **kwargs):
        return MemoryReview(observation_id=runtime.obs['observation_id'], assessment='unclear',
            evidence='No visible change.', causal_notes='Effect remains uncertain.',
            next_question='Does another control affect this object?', **kwargs)

    def execute(self, runtime):
        def respond(model, payload):
            c = context(payload)
            if c['work']=='ground':
                return call('submit_one_action', plan(c))
            return token('1')
        with patch.object(LocalVisionLlm, '_complete', respond):
            self.assertEqual(runtime.decide(obs())['action'], 'ACTION1')
        ack(runtime)
        runtime._receive(obs(1))

    def test_prompts_contracts_images_and_fast_execution_are_identical(self):
        a, b = (self.runtime(cls) for cls in (MemoryComparisonRuntime, SeparatedMemoryRuntime))
        for r in (a, b):
            r._receive(obs())
            r.work = 'ground'
        for work in ('ground', 'reconcile'):
            self.assertEqual(a._agent(work).instruction, b._agent(work).instruction)
            schemas = [r._agent(work).tools[0]._get_declaration().parameters_json_schema for r in (a,b)]
            for schema in schemas:
                schema['properties']['observation_id']['enum'] = ['CURRENT']
            self.assertEqual(*schemas)
        self.assertEqual(a._visual_parts(), b._visual_parts())
        self.assertIs(a._fast.__func__, b._fast.__func__)

    def test_separated_review_keeps_exact_current_prediction_and_result(self):
        r = self.runtime(); self.execute(r)
        r.recent_trials.insert(0, {'prediction':'OLD WRONG PREDICTION'})
        r.memory.reconciliations.append({'evidence':'OLD WRONG OUTCOME'})
        c = r._context('reconcile')
        self.assertEqual(c['active_trial']['expected_effect'], 'The square moves upward.')
        self.assertEqual(c['active_trial']['actual_result']['prediction'], 'The square moves upward.')
        self.assertTrue(c['active_trial']['actual_result']['acknowledged'])
        self.assertNotIn('OLD WRONG', json.dumps(c))
        self.assertNotIn('recent_trials', c)

    def test_new_unexecuted_attempt_does_not_receive_old_result(self):
        r = self.runtime(); self.execute(r)
        r._accept_stage('reconcile', self.review(r))
        completed = deepcopy(r.last_completed_trial)
        value = plan({'observation_id':r.obs['observation_id']})
        value['expected_effect'] = 'A different prediction.'
        r._accept_stage('ground', OneActionPlan.model_validate(value))
        r.memory.active_skill = {'invocation_id':'new-attempt', 'index':0}
        r.memory.review = {'trigger':'test'}
        c = r._context('reconcile')
        self.assertIsNone(c['active_trial']['actual_result'])
        self.assertEqual(c['active_trial']['expected_effect'], 'A different prediction.')
        self.assertNotIn('last_result', c)
        self.assertNotIn('last_completed_trial', c)
        r._accept_stage('reconcile', self.review(r))
        self.assertEqual(r.last_completed_trial, completed)

    def test_knowledge_survives_boundary_with_provenance_but_trial_does_not(self):
        r = self.runtime(); self.execute(r)
        claim = dict(kind='role', subject='Small square', claim='May be controlled by direction keys.',
                     conditions='When the game accepts direction controls.', status='hypothesis')
        r._accept_stage('reconcile', self.review(r, knowledge_updates=[claim]))
        record = deepcopy(r.knowledge['k1'])
        self.assertFalse(record['verified_by_host'])
        self.assertTrue(record['sources'][0]['acknowledged'])
        r._receive({**obs(2), 'levels_completed':1})
        self.assertIsNone(r.last_completed_trial)
        self.assertIsNone(r.frozen_trial)
        self.assertEqual(r.knowledge['k1'], record)
        c = r._context('ground')
        self.assertEqual(c['long_term_knowledge'][0], record)
        self.assertEqual(c['current_observation']['observation_id'], r.obs['observation_id'])

    def test_invalid_knowledge_write_is_atomic_and_not_promoted_without_action(self):
        r = self.runtime(); self.execute(r)
        claim = dict(kind='causal', subject='Square', claim='UP moves it.', conditions='During play.', status='supported')
        for bad in ({**claim, 'knowledge_id':'missing'}, {**claim, 'subject':'f1t23'},
                    {**claim, 'claim':'It is at (34,40).'}):
            before = r.memory.model_dump()
            with self.assertRaises(ValueError):
                r._accept_stage('reconcile', self.review(r, knowledge_updates=[bad]))
            self.assertEqual(before, r.memory.model_dump())
        r.memory.active_skill['invocation_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'acknowledgement'):
            r._accept_stage('reconcile', self.review(r, knowledge_updates=[claim]))
        self.assertEqual(r.knowledge, {})

    def test_factory_requires_explicit_fixed_planning_arm(self):
        from agent.adk_policy import create_runtime
        for layout, expected in [('mixed', MemoryComparisonRuntime), ('separated', SeparatedMemoryRuntime)]:
            with patch.dict('os.environ', {'COGNITION_PLANNING_COMPARISON':'B', 'COGNITION_MEMORY_COMPARISON':layout}):
                r = create_runtime('test'); self.addCleanup(r.close)
                self.assertIs(type(r), expected)
        with patch.dict('os.environ', {'COGNITION_PLANNING_COMPARISON':'A', 'COGNITION_MEMORY_COMPARISON':'mixed'}):
            with self.assertRaises(ValueError):
                create_runtime('test')
