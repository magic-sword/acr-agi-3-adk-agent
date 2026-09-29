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
             candidate_refs=[], object_refs=[], source='proposed', skill_name=None)], missing_info='')


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
            if c['work'] == 'understand':
                v = scene(c)
                if click:
                    row = c['object_index']['rows'][0]
                    v['targets'][0].update(object_id=row[0], candidate_refs=row[1])
                return call('submit_scene', v)
            if c['work'] == 'backchain': return call('submit_subgoal', subgoal(c, intent))
            if c['work'] == 'candidates':
                v = batch(c)
                if click: v['candidates'][0]['object_refs'] = [c['targets'][0]['object_id']]
                return call('submit_candidates', v)
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
                self.assertEqual(r.memory.schema_version, 16)
                self.assertEqual(r.memory.supported_skills, {})

    def test_click_uses_mask_without_cursor_model(self):
        r = self.runtime(); works = []
        with patch.object(LocalVisionLlm, '_complete', self.response(works, click=True)):
            result = r.decide(obs())
        self.assertEqual(result['action'], 'ACTION6', r.errors)
        self.assertNotIn('aim', works)
        self.assertTrue(r.memory.pending['binding']['confirmed'])

    def test_abstract_procedure_keeps_steps_and_fast_selects_concrete_action(self):
        r = self.runtime(); works = []
        ordinary = self.response(works)
        def respond(model, payload):
            c = context(payload)
            if c['work'] == 'ground':
                works.append('ground')
                p = plan(c)
                first = p['procedure']['steps'][0]
                second = deepcopy(first)
                second['purpose'] = 'Move down after reaching the intermediate position.'
                second['options'].append(dict(action={'action':'DOWN'}, expected_effect='Move down.'))
                p['procedure']['steps'].append(second)
                return call('submit_plan_choice', p)
            if c['work'] == 'execute_step':
                works.append('execute_step')
                self.assertEqual(c['abstract_action']['verb'], 'move_to')
                if c['active_skill']['action_count'] == 0:
                    return token('1')
                return token('7' if c['active_skill']['index'] == 0 else '2')
            return ordinary(model, payload)
        a = obs(); a['available_actions'].append('ACTION2')
        b = obs(1); b['available_actions'].append('ACTION2')
        with patch.object(LocalVisionLlm, '_complete', respond):
            self.assertEqual(r.decide(a)['action'], 'ACTION1')
            invocation = r.memory.active_skill['invocation_id']
            ack(r)
            self.assertEqual(r.decide(b)['action'], 'ACTION2')
        self.assertEqual(works, ['understand','backchain','candidates','ground',
                                 'execute_step','execute_step','execute_step'])
        self.assertEqual(r.memory.active_skill['invocation_id'], invocation)
        self.assertEqual(r.memory.active_skill['index'], 1)

    def test_ground_has_only_three_inputs_and_no_raw_history_or_coordinates(self):
        r = self.runtime(); self.prepare(r)
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

    def test_scene_schema_prevents_stale_refs_and_duplicate_id_binding(self):
        r = self.runtime(); r._receive(obs())
        value = scene(r._context('understand'))
        schema = FocusedTool(r, 'understand')._get_declaration().parameters_json_schema
        target = value['targets'][0]
        target.update(object_id=None, candidate_refs=['old-frame-ref'])
        with self.assertRaises(jsonschema.ValidationError): jsonschema.validate(value, schema)
        obj = r.memory.object_memory['objects'][0]
        target.update(object_id=obj['object_id'], candidate_refs=[])
        jsonschema.validate(value, schema)
        r._accept_stage('understand', Scene.model_validate(value))
        self.assertEqual(r.memory.understanding['targets'][0]['candidate_refs'], obj['candidate_refs'])

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

    def test_ground_reroute_requires_execution_obstacle(self):
        r = self.runtime(); self.prepare(r, intent='probe')
        value = dict(observation_id=r.obs['observation_id'], candidate_id=None,
                     procedure=None, probe_action_limit=None, next='understand',
                     reason='The effect is unknown on this static board.')
        schema = FocusedTool(r, 'ground')._get_declaration().parameters_json_schema
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(value, schema)
        with self.assertRaises(ValueError):
            r._accept_stage('ground', PlanChoice.model_validate(value))
        self.assertIsNotNone(r.memory.candidate_batch)
        for blocker in ('missing_target', 'unavailable_control'):
            blocked = {**value, 'blocked_by': blocker, 'reason': 'Cannot locate the target or use its control.'}
            jsonschema.validate(blocked, schema)
            r.validate_stage('ground', PlanChoice.model_validate(blocked))
        executable = {**plan(r._context('ground')), 'blocked_by': 'missing_target'}
        with self.assertRaises(ValueError):
            r.validate_stage('ground', PlanChoice.model_validate(executable))
        r._accept_stage('ground', PlanChoice.model_validate(blocked))
        self.assertEqual(r.memory.phase, 'understand')

    def test_probe_can_repeat_fast_actions_before_review(self):
        r = self.runtime(); works = []
        with patch.object(LocalVisionLlm, '_complete', self.response(works, intent='probe')):
            self.assertEqual(r.decide(obs())['action'], 'ACTION1'); ack(r)
            self.assertEqual(r.decide(obs(1))['action'], 'ACTION1', r.errors)
        self.assertEqual(works[-1], 'execute_step')
        self.assertNotIn('reconcile', works)
        self.assertEqual(r.memory.active_skill['action_count'], 1)

    def test_probe_completion_requires_acknowledged_observation(self):
        for acknowledged in (False, True):
            with self.subTest(acknowledged=acknowledged):
                r = self.runtime(); works = []; choices = []
                ordinary = self.response(works, intent='probe')
                def respond(model, payload):
                    c = context(payload)
                    if c['work'] == 'execute_step':
                        choices.append(c['choices'])
                    return ordinary(model, payload)
                with patch.object(LocalVisionLlm, '_complete', respond):
                    r.decide(obs())
                    self.assertNotIn('7', choices[-1])
                    self.assertIn('8', choices[-1])
                    if acknowledged:
                        ack(r)
                        # Receipt alone does not yet supply the post-action observation.
                        self.assertEqual(r.memory.active_skill['action_count'], 0)
                    r.decide(obs(1))  # Identical pixels are still a valid result.
                self.assertEqual('7' in choices[-1], acknowledged)

    def test_achieve_offers_completion_without_action(self):
        r = self.runtime(); works = []
        ordinary = self.response(works)
        def respond(model, payload):
            c = context(payload)
            if c['work'] == 'execute_step':
                self.assertIn('7', c['choices'])
                self.assertEqual(r.memory.active_skill['action_count'], 0)
            return ordinary(model, payload)
        with patch.object(LocalVisionLlm, '_complete', respond):
            r.decide(obs())
        self.assertIn('execute_step', works)

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

    def test_object_identity_flows_from_scene_to_trial_and_candidate_history(self):
        r = self.runtime(); r._receive(obs())
        ident = r.memory.object_memory['objects'][0]['object_id']
        data = scene(r._context('understand')); data['targets'][0]['object_id'] = ident
        r._accept_stage('understand', Scene.model_validate(data))
        self.assertTrue(r.memory.understanding['targets'][0]['candidate_refs'])
        r._accept_stage('backchain', Subgoal.model_validate(subgoal(r._context('backchain'), 'probe')))
        data = batch(r._context('candidates')); data['candidates'][0]['object_refs'] = [ident]
        schema = FocusedTool(r, 'candidates')._get_declaration().parameters_json_schema
        jsonschema.validate(data, schema)
        missing = deepcopy(data); missing['candidates'][0].pop('object_refs')
        with self.assertRaises(jsonschema.ValidationError): jsonschema.validate(missing, schema)
        r._accept_stage('candidates', CandidateBatch.model_validate(data))
        r._accept_stage('ground', PlanChoice.model_validate(plan(r._context('ground'))))
        saved = deepcopy(r.memory.active_target_binding)
        fast = r._context('execute_step')['current_target_objects']
        self.assertEqual(fast['targets'][0]['object_id'], ident)
        self.assertEqual(fast['observation_id'], r.obs['observation_id'])
        self.assertEqual(fast['missing_object_ids'], [])
        r._select({'action': 'ACTION6', 'x': 1, 'y': 0}, 'Observe the target.')
        ack(r); r._receive(obs(1))
        trial = r.recent_trials[-1]
        self.assertTrue(trial['object_observations'][0]['clicked_previous_mask'])
        self.assertEqual(trial['object_observations'][0]['changed_pixels_on_previous_support'], 0)
        r._queue_review('probe_result', 'Observe the result.', 'probe_observed')
        review = dict(observation_id=r.obs['observation_id'], goal_id=r.memory.plan['goal_id'],
            assessment='probe_result', evidence='No visible change on the clicked target.', goal_status='unknown',
            causal_notes='', next='ground', reason='Choose another test.', next_question='What is untested?')
        r._accept_stage('reconcile', Reconciliation.model_validate(review))
        self.assertEqual(r.memory.trial_ledger[-1]['target_objects'], saved['target_objects'])
        self.assertEqual(r.memory.trial_ledger[-1]['target_object_ids'], [ident])
        # Force the relevant trial outside the old last-three window.
        r.memory.trial_ledger.extend([dict(observation_id='other', target_objects=[], target_object_ids=[],
            verb='activate', evidence='Unrelated object trial.') for _ in range(4)])
        data = scene(r._context('understand')); data['targets'][0]['object_id'] = ident
        r._accept_stage('understand', Scene.model_validate(data))
        target = r._context('candidates')['targets'][0]
        self.assertEqual(target['related_history']['trials'][0]['correspondence'], 'tracked')
        data = batch(r._context('candidates')); data['candidates'][0]['object_refs'] = [ident]
        r._accept_stage('candidates', CandidateBatch.model_validate(data))
        card = r._context('ground')['candidates'][0]
        self.assertEqual(card['related_history']['trials'][0]['trial']['target_object_ids'], [ident])
        self.assertNotIn('mask_runs', str(r._context('ground')))
        self.assertNotIn('mask_runs', str(r._context('understand')))

    def test_invented_object_and_incompatible_part_are_rejected_without_mutation(self):
        grid = [[5]*12 for _ in range(12)]
        for x, color in [(2, 8), (8, 9)]:
            for y in (2, 3):
                grid[y][x:x+2] = [color, color]
        r = self.runtime(); r._receive(obs(grid=grid))
        current = r.memory.object_memory['objects']
        self.assertGreaterEqual(len(current), 2)
        left, right = current[0], current[-1]
        data = scene(r._context('understand')); data['targets'][0].update(object_id='invented')
        with self.assertRaisesRegex(ValueError, 'unknown object'):
            r._accept_stage('understand', Scene.model_validate(data))
        self.assertIsNone(r.memory.understanding)
        data['targets'][0].update(object_id=left['object_id'], candidate_refs=right['candidate_refs'])
        with self.assertRaises(ValueError): r._accept_stage('understand', Scene.model_validate(data))
        data['targets'][0].update(candidate_refs=[])
        r._accept_stage('understand', Scene.model_validate(data))
        r._accept_stage('backchain', Subgoal.model_validate(subgoal(r._context('backchain'))))
        candidate = batch(r._context('candidates')); candidate['candidates'][0]['object_refs'] = [right['object_id']]
        with self.assertRaisesRegex(ValueError, 'current targets'):
            r._accept_stage('candidates', CandidateBatch.model_validate(candidate))
        self.assertIsNone(r.memory.candidate_batch)

    def test_supported_skill_cannot_reuse_same_description_for_a_different_object(self):
        r = self.runtime(); self.prepare(r)
        current = r.memory.object_memory['objects'][0]
        ident = current['object_id']
        r.memory.understanding['targets'][0]['object_id'] = ident
        known = dict(verb='move_to', target_query='The marked exit.', procedure=skill(),
                     target_objects=[{**deepcopy(current), 'identity_segment': current['identity_segment']+1}],
                     evidence='A past test.', observation_id='old')
        r.memory.supported_skills['move'] = known
        candidate = batch(r._context('candidates'))
        candidate['candidates'][0].update(source='reuse', skill_name='move', object_refs=[ident],
                                         expected_effect=known['procedure']['effect'])
        with self.assertRaisesRegex(ValueError, 'supported object identity'):
            r._accept_stage('candidates', CandidateBatch.model_validate(candidate))
        self.assertEqual(r._skill_cards(), [])


if __name__ == '__main__': unittest.main()
