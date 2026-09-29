from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from agent.cognition.planning_comparison import (
    ComparisonRuntime, OneActionRuntime, OneActionPlan, ActionReview, OneActionTool, evidence_context)
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs, context, call, token, ack


class EmptyProposer:
    def propose(self, grid):
        return {'boxes': [], 'model': 'test'}


def plan(c):
    return dict(observation_id=c['observation_id'], goal_hypothesis='Reach the marked area.',
        question='Does UP move the square?',
        targets=[dict(appearance='The square', role_hypothesis='Possible actor', candidate_refs=[])],
        baseline='The square is visible.', action={'action': 'UP'},
        expected_effect='The square moves upward.', rationale='Test the control before inferring its role.')


class PlanningComparisonTests(unittest.TestCase):
    def runtime(self, cls=OneActionRuntime):
        r = cls('test', 'local/qwen3-vl-4b-instruct', sam_proposer=EmptyProposer())
        self.addCleanup(r.close)
        return r

    def test_controller_output_contract_is_identical_in_both_arms(self):
        from agent.cognition.deliberation import StageTool
        for actions in (['ACTION1'], ['ACTION6'], ['ACTION1', 'ACTION6']):
            with self.subTest(actions=actions):
                r = self.runtime(); r._receive({**obs(), 'available_actions': actions})
                baseline = StageTool(r, 'ground')._get_declaration().parameters_json_schema
                simplified = OneActionTool(r, 'ground')._get_declaration().parameters_json_schema
                self.assertEqual(baseline['$defs']['ActionIntent'], simplified['$defs']['ActionIntent'])

    def test_measurement_interning_is_lossless_and_observation_scoped(self):
        first = {'observation_id': 'old', 'candidates': [{'id': 'a', 'pattern': [[1, 2]]}]}
        second = {**first, 'observation_id': 'new'}
        source = {'measured_objects': first, 'last_result': {'measured_objects': first},
                  'recent_trials': [{'measured_objects': second}]}
        original = deepcopy(source)
        compact = evidence_context(source)
        self.assertEqual(len(compact['measurements']), 2)
        self.assertEqual(compact['measured_objects'], compact['last_result']['measured_objects'])
        self.assertNotEqual(compact['measured_objects'], compact['recent_trials'][0]['measured_objects'])
        def restore(value):
            if isinstance(value, list): return [restore(v) for v in value]
            if not isinstance(value, dict): return value
            if set(value) == {'measurement_ref'}: return compact['measurements'][value['measurement_ref']]
            return {k: restore(v) for k, v in value.items()}
        self.assertEqual(restore({k: compact[k] for k in source}), original)
        self.assertEqual(source, original)

    def test_b_plans_executes_acknowledges_and_reviews_even_without_change(self):
        r = self.runtime(); works = []
        def respond(model, payload):
            c = context(payload); works.append(c['work'])
            if c['work'] == 'ground':
                return call('submit_one_action', plan(c))
            if c['work'] == 'reconcile':
                self.assertTrue(c['last_result']['acknowledged'])
                self.assertFalse(c['last_result']['frame_changed'])
                return call('submit_action_review', dict(observation_id=c['observation_id'],
                    assessment='contradicted', evidence='The visible square did not move.',
                    causal_notes='This interval does not show the expected movement.',
                    next_question='Would another direction distinguish the control?'))
            return token('1')
        with patch.object(LocalVisionLlm, '_complete', respond):
            self.assertEqual(r.decide(obs())['action'], 'ACTION1'); ack(r)
            self.assertEqual(r.decide(obs(1))['action'], 'ACTION1')
        self.assertEqual(works, ['ground', 'execute_step', 'reconcile', 'ground', 'execute_step'])
        self.assertEqual(r.memory.reconciliations[-1]['effect_assessment'], 'contradicted')
        self.assertIsNone(r._current_result())
        self.assertEqual(r.memory.phase, 'execute_step')

    def test_b_click_uses_the_existing_cursor_and_not_model_coordinates(self):
        r = self.runtime(); works = []
        def respond(model, payload):
            c = context(payload); works.append(c['work'])
            if c['work'] == 'ground':
                value = plan(c); value['action'] = {'action': 'CLICK', 'target_query': 'The visible square.'}
                return call('submit_one_action', value)
            if c['work'] == 'aim': return token('1' if c['cursor']['mode'] == 'locate' else '7')
            return token('1')
        with patch.object(LocalVisionLlm, '_complete', respond): result = r.decide(obs())
        self.assertEqual(result['action'], 'ACTION6')
        self.assertEqual(works, ['ground', 'execute_step', 'aim', 'aim'])
        self.assertTrue(r.memory.pending['binding']['confirmed'])

    def test_invalid_binding_is_rejected_without_mutation(self):
        r = self.runtime(); r._receive(obs())
        value = plan({'observation_id': r.obs['observation_id']})
        value['targets'][0]['candidate_refs'] = ['missing']
        before = r.memory.model_dump()
        with self.assertRaisesRegex(ValueError, 'unknown current measured candidate'):
            r._accept_stage('ground', OneActionPlan.model_validate(value))
        self.assertEqual(before, r.memory.model_dump())

    def test_valid_max_length_outputs_fit_the_existing_execution_adapter(self):
        r = self.runtime(); r._receive(obs())
        value = plan({'observation_id': r.obs['observation_id']})
        value.update(goal_hypothesis='g'*200, question='q'*180, baseline='b'*180,
                     expected_effect='e'*180, rationale='r'*200)
        value['targets'] *= 6
        for target in value['targets']:
            target.update(appearance='a'*160, role_hypothesis='h'*180)
        r._accept_stage('ground', OneActionPlan.model_validate(value))
        self.assertEqual(r.memory.plan['question'], value['question'])
        self.assertEqual(r.memory.plan['baseline'], value['baseline'])
        r.memory.review = {'trigger': 'test'}
        review = ActionReview(observation_id=r.obs['observation_id'], assessment='unclear',
            evidence='e'*300, causal_notes='c'*600, next_question='q'*240)
        r._accept_stage('reconcile', review)
        self.assertEqual(r.memory.handoff_question, review.next_question)
        self.assertEqual(r.memory.reconciliations[-1]['causal_notes'], review.causal_notes)

    def test_boundary_clears_recent_trials_and_returns_to_single_action_plan(self):
        r = self.runtime()
        def respond(model, payload):
            c = context(payload)
            return call('submit_one_action', plan(c)) if c['work'] == 'ground' else token('1')
        with patch.object(LocalVisionLlm, '_complete', respond):
            r.decide(obs()); ack(r)
            r.decide({**obs(1), 'levels_completed': 1})
        self.assertEqual(r.recent_trials, [])
        self.assertEqual(r.memory.reconciliations, [])
        self.assertEqual(r.memory.active_skill['action_count'], 0)

    def test_common_images_and_measurements_match_at_same_observation(self):
        runtimes = [self.runtime(cls) for cls in (ComparisonRuntime, OneActionRuntime)]
        for r in runtimes:
            r._receive(obs()); r.work = 'ground'
        self.assertEqual(runtimes[0]._visual_parts(slow=True), runtimes[1]._visual_parts(slow=True))
        records = []
        for r in runtimes:
            c = r._context('ground')
            records.append(c['measurements'][c['measured_objects']['measurement_ref']])
        # Sensor timings differ, but the complete candidate index is identical.
        self.assertEqual(records[0]['candidate_index'], records[1]['candidate_index'])
        self.assertEqual(records[0]['changes'], records[1]['changes'])

    def test_factory_is_opt_in(self):
        from agent.adk_policy import create_runtime
        from agent.cognition.simple_workflow import SimpleRuntime
        for arm, expected in [('', SimpleRuntime), ('A', ComparisonRuntime), ('B', OneActionRuntime)]:
            with patch.dict('os.environ', {'COGNITION_PLANNING_COMPARISON': arm, 'ADK_MODEL': ''}):
                r = create_runtime('test'); self.addCleanup(r.close)
                self.assertIs(type(r), expected)
