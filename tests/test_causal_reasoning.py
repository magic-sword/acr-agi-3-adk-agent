"""Learn from interventions, regress conjunctions, retain provenance and expose typed tools."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.cognition.causal import (CausalStore, CausalMemoryTool, BackwardPlanTool, Symbols,
                                    learning, planning)
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from test_evidence_reasoning import observation


def state(*true, false=()):
    return {'true': list(true), 'false': list(false)}


def rule(identifier, pre=(), adds=(), removes=(), action='ACTION1', negative_pre=(), status='supported'):
    return dict(id=identifier, action_name=identifier, actions=[{'action': action}],
                preconditions=state(*pre, false=negative_pre), effects=state(*adds, false=removes),
                scope='level', level=0, status=status)


def example(identifier, before, after, *, action_name='switch', confounded=False):
    return dict(id=identifier, action_name=action_name, actions=[{'action': 'ACTION1'}],
                scope='level', level=0, before=before, after=after, confounded=confounded)


class InductionTests(unittest.TestCase):
    def test_positive_and_negative_examples_learn_a_discriminating_precondition(self):
        examples = [example('yes', state('powered', false=['open']), state('open')),
                    example('no', state(false=['powered', 'open']), state(false=['open']))]
        result = learning.induce(examples)
        self.assertEqual(len(result['rules']), 1)
        r = result['rules'][0]
        self.assertEqual(r['preconditions'], state('powered'))
        self.assertEqual(r['effects'], state('open'))
        self.assertEqual(learning.assess(r, examples)['status'], 'supported')
        counter = example('counter', state('powered', false=['open']), state(false=['open']))
        assessed = learning.assess(r, examples + [counter])
        self.assertEqual(assessed['status'], 'contradicted')
        self.assertEqual(assessed['evidence']['negative'], ['counter'])

    def test_unknown_is_not_a_negative_and_unchanged_is_not_causal_support(self):
        r = rule('r', adds=['open']); r['action_name'] = 'switch'
        examples = [example('hidden', state(false=['open']), state()),
                    example('already', state('open'), state('open'))]
        self.assertEqual(learning.assess(r, examples)['status'], 'candidate')
        self.assertEqual(learning.assess(r, examples)['evidence']['unknown'], ['hidden', 'already'])
        self.assertEqual(learning.induce(examples)['rules'], [])

    def test_positive_only_keeps_context_and_confounded_trials_do_not_train(self):
        examples = [example('yes', state('powered', false=['open']), state('open'))]
        r = learning.induce(examples)['rules'][0]
        self.assertEqual(r['preconditions'], state('powered', false=['open']))
        examples[0]['confounded'] = True
        self.assertEqual(learning.induce(examples)['rules'], [])
        self.assertEqual(learning.assess(r, examples)['status'], 'candidate')

    def test_inseparable_examples_report_missing_condition_instead_of_false_rule(self):
        examples = [example('yes', state('powered', false=['open']), state('open')),
                    example('no', state('powered', false=['open']), state(false=['open']))]
        result = learning.induce(examples)
        self.assertEqual(result['rules'], [])
        self.assertEqual(result['unresolved'][0]['negative_examples'], ['no'])
        self.assertIn('no_separating', result['unresolved'][0]['reason'])

    def test_time_budget_and_deleted_effects(self):
        e = example('remove', state('blocked'), state(false=['blocked']))
        self.assertEqual(learning.induce([e])['rules'][0]['effects'], state(false=['blocked']))
        self.assertFalse(learning.induce([e], seconds=0)['search_complete'])
        with self.assertRaises(ValueError):
            Symbols(true=['x'], false=['x'])


class RegressionTests(unittest.TestCase):
    def test_small_domains_match_independent_forward_reachability(self):
        import random
        from collections import deque
        rng = random.Random(872)
        atoms = ['a', 'b', 'c', 'd']
        for case in range(60):
            initial = frozenset(p for p in atoms if rng.randrange(2))
            goal_true = set(rng.sample(atoms, 2))
            rules = []
            for i in range(5):
                pre_atoms = rng.sample(atoms, rng.randrange(3))
                pre_true = [p for p in pre_atoms if rng.randrange(2)]
                effect_atoms = rng.sample(atoms, rng.randrange(1, 3))
                adds = [p for p in effect_atoms if rng.randrange(2)]
                rules.append(rule(str(i), pre_true, adds, set(effect_atoms) - set(adds),
                                  action=f'ACTION{i+1}', negative_pre=set(pre_atoms) - set(pre_true)))
            queue, seen = deque([initial]), {initial}
            possible = False
            while queue:
                value = queue.popleft()
                if goal_true <= value:
                    possible = True
                    break
                for r in rules:
                    if set(r['preconditions']['true']) <= value and not set(r['preconditions']['false']) & value:
                        after = frozenset((set(value) - set(r['effects']['false'])) | set(r['effects']['true']))
                        if after not in seen:
                            seen.add(after)
                            queue.append(after)
            result = planning.backward_plan(state(*initial, false=set(atoms) - initial), state(*goal_true),
                                            rules, max_depth=16, max_nodes=256, seconds=2)
            with self.subTest(case=case):
                self.assertEqual(result['status'] == 'plan', possible, result)

    def test_backward_chain_and_forward_proof(self):
        rules = [rule('key', ['at_key'], ['has_key'], action='ACTION1'),
                 rule('door', ['has_key'], ['door_open'], action='ACTION2'),
                 rule('finish', ['door_open'], ['stage_clear'], action='ACTION3')]
        result = planning.backward_plan(state('at_key'), state('stage_clear'), rules)
        self.assertEqual(result['status'], 'plan')
        self.assertEqual([s['rule_id'] for s in result['plan']], ['key', 'door', 'finish'])
        self.assertIn('stage_clear', result['plan'][-1]['after']['true'])
        self.assertEqual(result['next_action'], {'action': 'ACTION1'})

    def test_deleting_access_requires_reverse_order(self):
        rules = [rule('place', ['block_access'], ['pad_occupied'], ['switch_access']),
                 rule('activate', ['switch_access'], ['gate_open'], action='ACTION2')]
        result = planning.backward_plan(state('block_access', 'switch_access'),
                                        state('pad_occupied', 'gate_open'), rules)
        self.assertEqual([s['rule_id'] for s in result['plan']], ['activate', 'place'])
        rules[1]['effects']['false'] = ['block_access']
        self.assertEqual(planning.backward_plan(state('block_access', 'switch_access'),
                                                state('pad_occupied', 'gate_open'), rules)['status'], 'knowledge_gap')

    def test_missing_link_becomes_experiment_frontier(self):
        rules = [rule('finish', ['door_open'], ['stage_clear'])]
        result = planning.backward_plan(state('at_door', false=['door_open']), state('stage_clear'), rules)
        self.assertEqual(result['status'], 'knowledge_gap')
        self.assertTrue(any('door_open' in f['no_known_producer']['true'] for f in result['frontier']))
        self.assertTrue(any(f['suffix_rule_ids'] == ['finish'] for f in result['frontier']))

    def test_unknown_false_literals_cycles_and_limits(self):
        self.assertEqual(planning.backward_plan(state(), state(false=['blocked']), [])['status'], 'knowledge_gap')
        r = rule('clear', adds=(), removes=['blocked'])
        self.assertEqual(planning.backward_plan(state('blocked'), state(false=['blocked']), [r])['status'], 'plan')
        self.assertEqual(planning.backward_plan(state('blocked'), state(false=['blocked']), [r], max_depth=0)['status'], 'search_limit')
        self.assertEqual(planning.backward_plan(state('blocked'), state(false=['blocked']), [r], max_nodes=1)['status'], 'search_limit')
        cycle = [rule('a', ['b'], ['a']), rule('b', ['a'], ['b'], action='ACTION2')]
        self.assertEqual(planning.backward_plan(state(), state('a'), cycle)['status'], 'knowledge_gap')

    def test_candidate_and_competing_outcomes_are_not_silently_trusted(self):
        candidate = rule('r', adds=['open'], status='candidate')
        self.assertEqual(planning.backward_plan(state(), state('open'), [candidate])['status'], 'knowledge_gap')
        self.assertEqual(planning.backward_plan(state(), state('open'), [candidate], allow_candidates=True)['status'], 'tentative_plan')
        candidates = [rule('open', adds=['open']), rule('close', removes=['open'])]
        result = planning.backward_plan(state(), state('open'), candidates)
        self.assertEqual(result['status'], 'knowledge_gap')
        self.assertEqual({e['reason'] for e in result['excluded_rules']}, {'overlapping_action_models'})


class PersistentToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = CognitiveRuntime('test', knowledge_dir=self.temp.name)
        self.addCleanup(self.runtime.close)

    def invoke(self, tool, args):
        return asyncio.run(tool.run_async(args=args, tool_context=None))

    def archive_pair(self, *, cleared=False):
        self.runtime.decide(observation())
        before = self.runtime.memory.last_observation['observation_id']
        self.runtime.decide(observation(1, 1, levels_completed=1 if cleared else 0))
        after = self.runtime.memory.last_observation['observation_id']
        return before, after

    def transition(self, before, after, identifier='e'):
        return dict(id=identifier, before_id=before, after_id=after, action_name='switch',
                    before=state('powered', false=['open']), after=state('open'))

    def test_persistent_crud_provenance_and_deleted_rules_do_not_resurrect(self):
        before, after = self.archive_pair()
        tool = CausalMemoryTool(self.runtime)
        result = self.invoke(tool, {'operation': 'update', 'transitions': [self.transition(before, after)]})
        self.assertNotIn('error', result)
        self.assertEqual(result['rule_count'], 1)
        self.assertEqual(result['rules'][0]['status'], 'supported')
        identifier = result['rules'][0]['id']
        reopened = CausalStore('test', self.temp.name)
        self.assertEqual(reopened.read()['transitions']['e']['actions'], [{'action': 'ACTION1'}])
        self.assertIn(before, reopened.read()['transitions']['e']['source_observations'])
        self.assertEqual(CausalStore('another-version', self.temp.name).read()['rules'], {})
        self.invoke(tool, {'operation': 'update', 'delete_rule_ids': [identifier]})
        self.assertEqual(self.invoke(tool, {'operation': 'update'})['rule_count'], 0)
        self.invoke(tool, {'operation': 'update', 'delete_transition_ids': ['e']})
        self.assertEqual(reopened.read()['transitions'], {})

    def test_invalid_batch_is_atomic_and_manual_rule_is_not_verified(self):
        before, after = self.archive_pair()
        tool = CausalMemoryTool(self.runtime)
        r = rule('speculative', ['powered'], ['open'])
        for key in ('level', 'status'):
            del r[key]
        result = self.invoke(tool, {'operation': 'update', 'rules': [r],
            'transitions': [self.transition(before, 'invented')]})
        self.assertIn('error', result)
        self.assertEqual(self.runtime.causal.read()['rules'], {})
        result = self.invoke(tool, {'operation': 'update', 'rules': [r], 'induce': False})
        self.assertEqual(result['rules'][0]['status'], 'candidate')

    def test_final_outcome_survives_runtime_close_and_cannot_be_invented(self):
        before, after = self.archive_pair(cleared=True)
        tool = CausalMemoryTool(self.runtime)
        item = self.transition(before, after)
        item['after'] = state('stage_clear')
        result = self.invoke(tool, {'operation': 'update', 'transitions': [item]})
        self.assertNotIn('error', result)
        self.assertEqual(self.runtime.causal.read()['transitions']['e']['after'], state('stage_clear'))
        self.runtime.close()
        next_run = CognitiveRuntime('test', knowledge_dir=self.temp.name)
        self.addCleanup(next_run.close)
        self.assertEqual(len(next_run.causal.read()['transitions']), 1)
        self.assertEqual(next_run.causal.read()['observations'][after]['levels_completed'], 1)

    def test_false_stage_clear_stale_plan_and_scope_are_rejected(self):
        before, after = self.archive_pair()
        item = self.transition(before, after); item['after'] = state('stage_clear')
        result = self.invoke(CausalMemoryTool(self.runtime), {'operation': 'update', 'transitions': [item]})
        self.assertIn('stage_clear', result['error'])
        tool = BackwardPlanTool(self.runtime)
        self.assertIn('error', self.invoke(tool, {'observation_id': before, 'current': state()}))
        self.assertIn('error', self.invoke(tool, {'observation_id': after, 'current': state('stage_clear')}))
        result = self.invoke(tool, {'observation_id': after, 'current': state()})
        self.assertEqual(result['status'], 'knowledge_gap')

    def test_deleting_only_support_downgrades_a_rule(self):
        before, after = self.archive_pair()
        tool = CausalMemoryTool(self.runtime)
        self.invoke(tool, {'operation': 'update', 'transitions': [self.transition(before, after)]})
        result = self.invoke(tool, {'operation': 'update', 'delete_transition_ids': ['e'], 'induce': False})
        self.assertEqual(result['rules'][0]['status'], 'candidate')

    def test_correction_and_duplicate_trial_handling(self):
        before, after = self.archive_pair()
        tool = CausalMemoryTool(self.runtime)
        item = self.transition(before, after)
        self.invoke(tool, {'operation': 'update', 'transitions': [item]})
        duplicate = {**item, 'id': 'duplicate'}
        self.assertIn('already recorded', self.invoke(tool, {'operation': 'update', 'transitions': [duplicate]})['error'])
        corrected = {**item, 'after': state(false=['open'])}
        result = self.invoke(tool, {'operation': 'update', 'transitions': [corrected]})
        self.assertEqual(result['rules'][0]['status'], 'contradicted')
        self.assertEqual(result['transition_count'], 1)

    def test_saved_source_evidence_survives_raw_archive_eviction(self):
        before, after = self.archive_pair()
        tool = CausalMemoryTool(self.runtime)
        self.invoke(tool, {'operation': 'update', 'transitions': [self.transition(before, after)]})
        self.runtime.causal.mutate(lambda data: data['observations'].clear())
        result = self.invoke(tool, {'observation_ids': [before, after]})
        self.assertNotIn('error', result)
        self.assertTrue(result['_image_png_base64'])
        self.assertEqual([o['observation_id'] for o in result['observations']], [before, after])
        self.assertTrue(all('grid' not in o and 'image_png_base64' not in o for o in result['observations']))
        corrected = self.transition(before, after); corrected['confounded'] = True
        self.assertNotIn('error', self.invoke(tool, {'operation': 'update', 'transitions': [corrected]}))

    def test_invalid_read_selector_cannot_commit_an_update(self):
        before, after = self.archive_pair()
        tool = CausalMemoryTool(self.runtime)
        result = self.invoke(tool, {'operation': 'update', 'transitions': [self.transition(before, after)],
                                    'observation_ids': ['unknown']})
        self.assertIn('error', result)
        self.assertEqual(self.runtime.causal.read()['transitions'], {})

    def test_macro_preserves_every_input_and_reset_is_rejected(self):
        before, _ = self.archive_pair()
        self.runtime.decide(observation(2, 2))
        after = self.runtime.memory.last_observation['observation_id']
        result = self.invoke(CausalMemoryTool(self.runtime), {'operation': 'update',
                             'transitions': [self.transition(before, after)]})
        self.assertEqual(len(result['transitions'][0]['actions']), 2)
        self.assertTrue(result['transitions'][0]['abstract_macro'])
        self.runtime.decide(observation(3, state='GAME_OVER'))
        reset = self.runtime.memory.last_observation['observation_id']
        self.runtime.decide(observation(4))
        result = self.invoke(CausalMemoryTool(self.runtime), {'operation': 'update', 'transitions': [
            self.transition(reset, self.runtime.memory.last_observation['observation_id'], 'reset')]})
        self.assertIn('RESET', result['error'])

    def test_rule_scope_does_not_leak_into_another_level(self):
        before, after = self.archive_pair()
        tool = CausalMemoryTool(self.runtime)
        self.invoke(tool, {'operation': 'update', 'transitions': [self.transition(before, after)]})
        self.runtime.decide(observation(2, 2, levels_completed=1))
        self.assertEqual(self.invoke(tool, {})['rule_count'], 0)

    def test_win_is_archived_without_an_extra_model_call(self):
        self.runtime.decide(observation())
        self.runtime.decide(observation(1, state='WIN'))
        records = list(self.runtime.causal.read()['observations'].values())
        self.assertEqual(records[-1]['state'], 'WIN')
        self.assertEqual(records[-1]['source_action']['action']['action'], 'ACTION1')
        self.assertEqual(self.runtime.memory.model_calls, 0)

    def test_actual_adk_can_record_learn_plan_then_submit_in_three_requests(self):
        before, after = self.archive_pair()
        # The next real observation arrives after the preceding recorded action.
        self.runtime.model = 'local/qwen3-vl-4b-instruct'
        calls = []
        def respond(model, payload):
            calls.append(payload)
            context = next(json.loads(p['text']) for m in payload['messages']
                           if m['role'] == 'user' and isinstance(m['content'], list)
                           for p in m['content'] if p.get('type') == 'text')
            if len(calls) == 1:
                name, args = 'causal_memory', {'operation': 'update', 'transitions': [self.transition(before, after)]}
            elif len(calls) == 2:
                response = next(json.loads(m['content']) for m in payload['messages'] if m['role'] == 'tool')
                self.assertEqual(response['rules'][0]['status'], 'supported')
                name, args = 'plan_backward', {'observation_id': context['observation_id'],
                    'current': state('powered', false=['open']), 'goal': state('open')}
            else:
                response = [json.loads(m['content']) for m in payload['messages'] if m['role'] == 'tool'][-1]
                self.assertEqual(response['status'], 'plan')
                self.assertEqual(response['next_action'], {'action': 'ACTION1'})
                self.assertEqual(payload['tool_choice'], 'required')
                name, args = 'submit_decision', {'action': {'action': 'UP'}, 'prediction': 'The gate may open.'}
            return {'choices': [{'message': {'tool_calls': [{'id': str(len(calls)), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}
        with patch.object(LocalVisionLlm, '_complete', respond):
            result = self.runtime.decide(observation(2, 0))
        self.assertEqual(result['action'], 'ACTION1')
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.runtime.memory.model_calls, 1)

    def test_dependent_causal_calls_in_same_response_execute_neither(self):
        before, after = self.archive_pair()
        self.runtime.model = 'local/qwen3-vl-4b-instruct'
        self.runtime.max_calls = 1
        def respond(model, payload):
            calls = [('causal_memory', {'operation': 'update', 'transitions': [self.transition(before, after)]}),
                     ('plan_backward', {'observation_id': after, 'current': state()})]
            return {'choices': [{'message': {'tool_calls': [
                {'id': str(i), 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
                for i, (name, args) in enumerate(calls)]}}]}
        with patch.object(LocalVisionLlm, '_complete', respond):
            result = self.runtime.decide(observation(2))
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(self.runtime.causal.read()['transitions'], {})
