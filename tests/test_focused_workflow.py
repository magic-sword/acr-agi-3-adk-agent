from copy import deepcopy
import unittest
from unittest.mock import patch
import jsonschema

from agent.cognition.candidates import CandidateBatch
from agent.cognition.focused_state import Scene, Subgoal, PlanChoice
from agent.cognition.focused_workflow import FocusedRuntime, FocusedTool
from agent.cognition.state import Reconciliation
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs, context, call, token, skill, ack


def scene(c):
    return dict(observation_id=c['observation_id'], observed='Actor below the exit.',
                goal_hypothesis='Reach the exit.', targets=[dict(concept='exit', appearance='Marked cell',
                    role_hypothesis='Possible exit', relations='Above actor', candidate_refs=[])])


def subgoal(c, intent='achieve'):
    return dict(observation_id=c['observation_id'], desired_state='Actor at the exit.',
                target_query='The marked exit.', intent=intent,
                question='Does moving toward the exit work?' if intent == 'probe' else '',
                expected_observation='Actor reaches the exit.', rationale='Entry may finish the board.')


def batch(c):
    return dict(observation_id=c['observation_id'], goal_id=c['goal_id'], candidates=[
        dict(id='move-exit', verb='move_to', target_query='The marked exit.',
             expected_effect='Actor reaches the exit.', rationale='Advance toward the goal.',
             candidate_refs=[], source='proposed', skill_name=None)], missing_info='')


def plan(c):
    return dict(observation_id=c['observation_id'], candidate_id='move-exit', procedure=skill(),
                probe_action_limit=2 if c['purpose']['intent'] == 'probe' else None,
                next='execute', reason='Approach the exit until the actor enters it.')


class FixedGenerator:
    def __init__(self): self.requests = []
    async def generate(self, request, propose):
        self.requests.append(deepcopy(request))
        return CandidateBatch.model_validate(batch(request))


class FocusedWorkflowTests(unittest.TestCase):
    def runtime(self, **kwargs):
        r = FocusedRuntime('test', 'local/qwen3-vl-4b-instruct', proposal_mode='program', **kwargs)
        self.addCleanup(r.close)
        return r

    def prepare(self, r, intent='achieve'):
        r._receive(obs())
        c = r._context('understand'); r._accept_stage('understand', Scene.model_validate(scene(c)))
        c = r._context('backchain'); r._accept_stage('backchain', Subgoal.model_validate(subgoal(c, intent)))
        c = r._context('candidates'); r._accept_stage('candidates', CandidateBatch.model_validate(batch(c)))

    def response(self, works, intent='achieve', click=False):
        def respond(model, payload):
            c = context(payload); works.append(c['work'])
            if c['work'] == 'understand': return call('submit_scene', scene(c))
            if c['work'] == 'backchain': return call('submit_subgoal', subgoal(c, intent))
            if c['work'] == 'candidates': return call('submit_candidates', batch(c))
            if c['work'] == 'ground':
                p = plan(c)
                if click: p['procedure'] = skill(x=1)
                return call('submit_plan_choice', p)
            if c['work'] == 'aim': return token('1' if c['cursor']['mode'] == 'locate' else '7')
            return token('1')
        return respond

    def test_default_model_provider_and_replacement_execute_without_notebook_or_skill_selector(self):
        for provider in (None, FixedGenerator()):
            with self.subTest(provider=provider):
                r = self.runtime(candidate_generator=provider); works = []
                with patch.object(LocalVisionLlm, '_complete', self.response(works)):
                    result = r.decide(obs())
                self.assertEqual(result['action'], 'ACTION1', r.errors)
                expected = ['understand', 'backchain'] + ([] if provider else ['candidates']) + ['ground', 'execute_step']
                self.assertEqual(works, expected)
                self.assertEqual(r.memory.schema_version, 13)
                self.assertEqual(r.memory.supported_skills, {})
                self.assertEqual(r.memory.memory_brief, {})

    def test_click_still_uses_cursor(self):
        r = self.runtime(); works = []
        with patch.object(LocalVisionLlm, '_complete', self.response(works, click=True)):
            result = r.decide(obs())
        self.assertEqual(result['action'], 'ACTION6', r.errors)
        self.assertIn('aim', works)
        self.assertTrue(r.memory.cursor['confirmed'])

    def test_ground_has_only_three_inputs_and_no_raw_history_or_coordinates(self):
        r = self.runtime(); self.prepare(r)
        r.memory.notes['noise'] = {'text': 'Click old white square at (15,15)'}
        r.memory.memory_brief = {'selected': ['noise']}
        c = r._context('ground')
        self.assertEqual(set(c), {'work', 'observation_id', 'purpose', 'candidates', 'evidence'})
        self.assertNotIn('candidate_refs', c['candidates'][0])
        self.assertNotIn('white square', str(c))
        self.assertNotIn('x', r._trial(dict(action=dict(action='ACTION6', x=15, y=15)))['action'])
        r.work = 'ground'; self.assertEqual(r._visual_parts(True), [])
        schema = FocusedTool(r, 'ground')._get_declaration().parameters_json_schema
        self.assertEqual(schema['anyOf'][0]['properties']['candidate_id']['enum'], ['move-exit'])
        self.assertEqual(schema['anyOf'][0]['properties']['procedure'], {'$ref': '#/$defs/Procedure'})
        jsonschema.validate(plan(c), schema)
        for change in (dict(procedure=None), dict(next='understand')):
            with self.assertRaises(jsonschema.ValidationError): jsonschema.validate({**plan(c), **change}, schema)

    def test_provider_output_validation_is_atomic(self):
        r = self.runtime(); self.prepare(r)
        good = batch(r._context('candidates'))
        mutations = [dict(observation_id='old'), dict(goal_id='old')]
        for update in mutations:
            with self.assertRaises(ValueError):
                r._accept_stage('candidates', CandidateBatch.model_validate({**good, **update}))
        for update in (dict(candidate_refs=['old-region']), dict(source='reuse', skill_name='unknown'),
                       dict(skill_name='invented-known-skill')):
            bad = deepcopy(good); bad['candidates'][0].update(update)
            with self.assertRaises(ValueError): r._accept_stage('candidates', CandidateBatch.model_validate(bad))
        bad = deepcopy(good); bad['candidates'] *= 2
        with self.assertRaises(ValueError): r._accept_stage('candidates', CandidateBatch.model_validate(bad))
        self.assertEqual(r.memory.candidate_batch, good)

    def test_contradictory_plan_and_stale_batch_rejected(self):
        r = self.runtime(); self.prepare(r)
        p = plan(r._context('ground'))
        with self.assertRaises(ValueError):
            r._accept_stage('ground', PlanChoice.model_validate({**p, 'next': 'understand'}))
        r.memory.candidate_batch['observation_id'] = 'old'
        with self.assertRaises(ValueError): r._accept_stage('ground', PlanChoice.model_validate(p))
        self.assertIsNone(r.memory.active_skill)

    def test_empty_candidates_request_missing_information(self):
        r = self.runtime(); self.prepare(r)
        value = dict(observation_id=r.obs['observation_id'], goal_id=r.memory.selected_goal_id,
                     candidates=[], missing_info='Which region is the exit?')
        r._accept_stage('candidates', CandidateBatch.model_validate(value))
        self.assertEqual(r.memory.phase, 'understand')
        self.assertEqual(r.memory.handoff_question, value['missing_info'])

    def test_probe_can_repeat_fast_actions_before_review(self):
        r = self.runtime(); works = []
        with patch.object(LocalVisionLlm, '_complete', self.response(works, intent='probe')):
            self.assertEqual(r.decide(obs())['action'], 'ACTION1'); ack(r)
            self.assertEqual(r.decide(obs(1))['action'], 'ACTION1', r.errors)
        self.assertEqual(works[-1], 'execute_step')
        self.assertNotIn('reconcile', works)
        self.assertEqual(r.memory.active_skill['action_count'], 1)

    def test_probe_limit_reviews_no_change_then_refreshes_targets_before_replanning(self):
        r = self.runtime(); works = []
        ordinary = self.response(works, intent='probe')
        def respond(model, payload):
            c = context(payload)
            if c['work'] != 'reconcile': return ordinary(model, payload)
            works.append('reconcile')
            self.assertEqual(len(c['attempt_results']), 2)
            self.assertTrue(all(t['acknowledged'] and t['changed_cell_count'] == 0 for t in c['attempt_results']))
            return call('submit_reconciliation', dict(observation_id=c['observation_id'], goal_id=c['plan']['goal_id'],
                assessment='probe_result', evidence='Two actions produced no change.', goal_status='unknown',
                causal_notes='UP did not move the actor in this context.', next='ground',
                reason='Test another operation.', next_question='Which operation could move the actor?'))
        with patch.object(LocalVisionLlm, '_complete', respond):
            r.decide(obs()); ack(r)
            r.decide(obs(1)); ack(r)
            result = r.decide(obs(2))
        self.assertEqual(result['action'], 'ACTION1', r.errors)
        self.assertEqual(works[-5:], ['reconcile', 'understand', 'candidates', 'ground', 'execute_step'])
        self.assertEqual(r.memory.supported_skills, {})
        self.assertEqual(r.memory.causal_knowledge[-1]['status'], 'conditional_model_interpretation')
        self.assertEqual(r.memory.candidate_batch['observation_id'], r.obs['observation_id'])
        self.assertIsNone(r._current_result())

    def test_factory_injects_provider_into_default_workflow(self):
        from agent.adk_policy import create_runtime
        provider = FixedGenerator()
        with patch.dict('os.environ', {'ADK_MODEL': '', 'COGNITION_PROPOSALS': 'program',
            'COGNITION_PLANNING_COMPARISON': '', 'COGNITION_MEMORY_COMPARISON': '', 'COGNITION_ACTION_UPDATE_COMPARISON': ''}):
            r = create_runtime('test', candidate_generator=provider); self.addCleanup(r.close)
        self.assertIs(r.candidate_generator, provider)

    def review(self, r, **changes):
        return Reconciliation.model_validate(dict(observation_id=r.obs['observation_id'],
            goal_id=r.memory.plan['goal_id'], assessment='matched', evidence='Actor reached the exit.',
            goal_status='confirmed', causal_notes='UP moved this actor under these conditions.',
            next='backchain', reason='Choose the next useful goal.', next_question='What remains?', **changes))

    def test_confirmation_needs_acknowledged_same_invocation_and_support_is_scoped(self):
        r = self.runtime(); self.prepare(r)
        r._accept_stage('ground', PlanChoice.model_validate(plan(r._context('ground'))))
        r._queue_review('completion_candidate', 'Test completion', 'procedure_done')
        with self.assertRaises(ValueError): r._accept_stage('reconcile', self.review(r))
        r.recent_trials = [dict(observation_id='result', action={'action': 'ACTION1'}, acknowledged=True,
                               skill={'invocation_id': 'other'})]
        with self.assertRaises(ValueError): r._accept_stage('reconcile', self.review(r))
        r.recent_trials[0]['skill']['invocation_id'] = r.memory.active_skill['invocation_id']
        r._accept_stage('reconcile', self.review(r))
        self.assertIn('move', r.memory.supported_skills)
        self.assertEqual(r.memory.phase, 'understand')
        self.assertIsNone(r.memory.candidate_batch)
        self.assertEqual(r.memory.trial_ledger[-1]['actual_trials'][0]['action'], 'ACTION1')
        self.assertEqual(r.memory.final_goal, 'Reach the exit.')

    def test_custom_provider_failure_stops_before_execution(self):
        class Bad:
            async def generate(self, request, propose): return {**batch(request), 'observation_id': 'old'}
        r = self.runtime(candidate_generator=Bad()); works = []
        with patch.object(LocalVisionLlm, '_complete', self.response(works)):
            r.decide(obs())
        self.assertEqual(r.memory.stop_reason, 'candidate_provider_invalid')
        self.assertNotIn('execute_step', works)

    def test_boundary_expires_short_term_state(self):
        r = self.runtime(); self.prepare(r)
        r.memory.trial_ledger = [{'old': True}]
        r._on_observation('level')
        self.assertIsNone(r.memory.candidate_batch)
        self.assertIsNone(r.memory.subgoal)
        self.assertEqual(r.memory.trial_ledger, [])
        self.assertEqual(r.memory.final_goal, '')


if __name__ == '__main__': unittest.main()
