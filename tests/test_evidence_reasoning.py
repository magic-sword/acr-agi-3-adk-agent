"""Evidence survives turns; interpretation belongs to reasoning and commits atomically."""
import json
import unittest
from unittest.mock import patch

from agent.cognition.evidence import EvidenceStore
from agent.cognition.engine import CognitiveTurn, new_memory
from agent.cognition.state import Interpretation, Proposal, Fact
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from agent.observation import attach_visuals


def observation(step=0, color=0, **extra):
    obs = dict(game_id='test', state='NOT_FINISHED', step=step, levels_completed=0,
               available_actions=['ACTION1'], remaining_actions=10,
               grid=[[color]*8 for _ in range(8)])
    obs.update(extra)
    attach_visuals(obs, [obs['grid']])
    return obs


class EvidenceContracts(unittest.TestCase):
    def test_archive_is_immutable_paginated_and_bounded(self):
        store = EvidenceStore(capacity=2)
        self.addCleanup(store.close)
        a, b = observation(), observation(1, 1)
        a['observation_id'], b['observation_id'] = 'a', 'b'
        store.add(a); store.add(b)
        a['grid'][0][0] = 15
        self.assertEqual(store.get('a')['grid'][0][0], 0)
        comparison = store.compare('a', 'b')
        self.assertEqual(comparison['changed_cell_count'], 64)
        self.assertFalse(comparison['time_advanced'])
        self.assertIn('_image_png_base64', comparison)
        self.assertEqual(comparison, store.compare('a', 'b'))
        c = observation(2); c['observation_id'] = 'c'
        store.add(c, boundary=True)
        with self.assertRaisesRegex(ValueError, 'evicted'):
            store.get('a')
        with self.assertRaisesRegex(ValueError, 'boundary'):
            store.compare('b', 'c')
        with self.assertRaises(ValueError):
            store.compare('b', 'b')

    def test_full_diff_pagination_and_multiple_actions(self):
        store = EvidenceStore()
        self.addCleanup(store.close)
        for step in range(3):
            o = observation(step)
            o['grid'] = [[step]*64 for _ in range(64)]
            o['observation_id'] = str(step)
            store.add(o, source_action={'decision_id': str(step), 'action': {'action': 'ACTION1'}})
        page = store.compare('0', '2')
        self.assertEqual(page['changed_cell_count'], 4096)
        self.assertEqual(len(page['transitions']), 2)
        total = len(page['changed_cells'])
        while page['next_offset'] is not None:
            page = store.compare('0', '2', page['next_offset'])
            total += len(page['changed_cells'])
        self.assertEqual(total, 4096)

    def test_history_is_available_without_logging_and_cursor_is_not_rewritten(self):
        runtime = CognitiveRuntime('test')
        self.addCleanup(runtime.close)
        runtime.decide(observation())
        old = runtime.memory.last_observation['observation_id']
        runtime.decide(observation(1, 1))
        new = runtime.memory.last_observation['observation_id']
        archived = runtime.get_observation(old)
        runtime.move_cursor(2, 3)
        self.assertEqual(archived, runtime.get_observation(old))
        self.assertEqual(runtime.compare_observations(old, new)['changed_cell_count'], 64)
        event = runtime.evidence.get(old)['animation']['event_id']
        self.assertEqual(runtime.observe_animation(event)['observation_id'], old)
        self.assertEqual(runtime.memory.revision, 2)
        runtime.decide(observation(2, 2, levels_completed=1))
        boundary = runtime.memory.last_observation['observation_id']
        self.assertIn('error', runtime.compare_observations(new, boundary))

    def test_unknown_goal_and_unknowns_do_not_block_plan(self):
        turn = CognitiveTurn(new_memory('test'), observation())
        turn.observe(); turn.verify(); turn.update()
        turn.memory.unknowns = ['control mapping']
        self.assertEqual(turn.route(), 'PLAN')
        self.assertEqual(turn.memory.goal, '')
        self.assertEqual(turn.calls, 0)

    def test_rejected_proposal_cannot_publish_its_interpretation(self):
        turn = CognitiveTurn(new_memory('test'), observation())
        turn.observe()
        oid = turn.obs['observation_id']
        before = turn.memory.model_dump()
        p = Proposal(observation_id=oid, memory_revision=0, purpose='probe',
                     interpretation=Interpretation(observation_id=oid, memory_revision=0,
                                                    goal='align objects', evidence_refs=[oid]))
        with self.assertRaisesRegex(ValueError, 'experiment'):
            turn.accept(p)
        self.assertEqual(turn.memory.model_dump(), before)
        self.assertFalse(turn.revise)

    def test_stale_and_invented_evidence_rejected_before_update(self):
        turn = CognitiveTurn(new_memory('test'), observation())
        turn.observe()
        for oid, evidence in [('stale', []), (turn.obs['observation_id'], ['invented'])]:
            with self.assertRaises(ValueError):
                turn.stage_review(Interpretation(observation_id=oid, memory_revision=0,
                                                 summary='changed', evidence_refs=evidence))
        self.assertIsNone(turn.review)

    def test_goal_can_be_withdrawn_without_losing_causal_memory(self):
        turn = CognitiveTurn(new_memory('test'), observation())
        turn.observe()
        turn.memory.goal = 'reach an object'
        turn.memory.unknowns = ['control mapping']
        turn.apply_interpretation(Interpretation(observation_id=turn.obs['observation_id'], memory_revision=0,
                                  goal='', evidence_refs=[turn.obs['observation_id']]), 'REVISE')
        self.assertEqual(turn.memory.goal, '')
        self.assertEqual(turn.memory.unknowns, ['control mapping'])


class ReasoningRoundTrip(unittest.TestCase):
    def test_initial_plan_probe_then_visual_verify_updates_memory(self):
        requests = []

        def respond(model, payload):
            context = next(json.loads(p['text']) for m in payload['messages']
                           if m['role'] == 'user' and isinstance(m['content'], list)
                           for p in m['content'] if p.get('type') == 'text')
            state, step = context['reasoning_state'], context['observation']['step']
            requests.append((state, step, payload))
            oid = context['observation_id']
            base = dict(observation_id=oid, memory_revision=context['memory_revision'])
            if state == 'PLAN' and step == 0:
                reply = dict(base, status='need_evidence', purpose='plan', inquiry='Does UP affect the target?',
                             interpretation=dict(base, unknowns=['control mapping']))
            elif state == 'PROBE':
                self.assertIn('UP', context['inquiry'])
                reply = dict(base, purpose='probe', action={'action': 'UP'},
                    effects=[{'kind': 'fact', 'key': 'target_changed', 'value': True}],
                    experiment={'question': context['inquiry'], 'risk': 'unknown', 'discriminator': 'target change',
                                'alternatives': {'changed': [{'kind': 'frame_changed', 'value': True}],
                                                 'unchanged': [{'kind': 'frame_changed', 'value': False}]}})
            elif state == 'VERIFY':
                if not any(m['role'] == 'tool' for m in payload['messages']):
                    return {'choices': [{'message': {'tool_calls': [{'id': 'compare', 'type': 'function',
                        'function': {'name': 'compare_observations', 'arguments': json.dumps({
                            'before_id': context['pending']['observation_id'], 'after_id': oid})}}]}}]}
                tool = next(json.loads(m['content']) for m in payload['messages'] if m['role'] == 'tool')
                self.assertEqual(tool['changed_cell_count'], 64)
                self.assertNotIn('_image_png_base64', tool)
                self.assertTrue(any(p.get('type') == 'image_url' for m in payload['messages']
                                    if isinstance(m['content'], list) for p in m['content']))
                reply = dict(base, summary='The tested target changed.', evidence_refs=[oid], unknowns=[],
                             facts=[{'key': 'target_changed', 'value': True, 'evidence': oid}])
            else:
                self.assertEqual(context['verification'][0]['result'], 'supported')
                self.assertEqual(context['verification'][0]['matched_alternatives'], ['changed'])
                self.assertEqual(context['facts'][0]['key'], 'target_changed')
                reply = dict(base, purpose='plan', interpretation=dict(base, goal='satisfy alignment conditions',
                             evidence_refs=[oid]), plan=[{'id': 'align', 'subgoal': 'test alignment',
                             'action': {'action': 'UP'}, 'completion': [{'kind': 'state', 'value': 'WIN'}],
                             'effects': [{'kind': 'state', 'value': 'WIN'}]}])
            name = {'PLAN': 'submit_plan', 'PROBE': 'submit_experiment', 'VERIFY': 'submit_interpretation'}[state]
            return {'choices': [{'message': {'tool_calls': [{'id': 'submission', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(reply)}}]}}]}

        with patch.object(LocalVisionLlm, '_complete', respond):
            runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct')
            self.addCleanup(runtime.close)
            self.assertEqual(runtime.decide(observation())['action'], 'ACTION1')
            self.assertEqual(runtime.memory.goal, '')
            self.assertEqual(runtime.decide(observation(1, 1))['action'], 'ACTION1')
            self.assertEqual(runtime.memory.goal, 'satisfy alignment conditions')
            self.assertEqual([s for s, _, _ in requests], ['PLAN', 'PROBE', 'VERIFY', 'VERIFY', 'PLAN'])
            self.assertTrue(any(x['state'] == 'VERIFY' for x in runtime.memory.interpretations))
            self.assertEqual(runtime.memory.model_calls, 4)  # Tool roundtrip is within VERIFY.
