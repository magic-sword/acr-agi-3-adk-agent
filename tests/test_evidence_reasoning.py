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
    def test_reflection_and_next_action_share_one_call_with_before_after_evidence(self):
        requests = []

        def respond(model, payload):
            context = next(json.loads(p['text']) for m in payload['messages']
                           if m['role'] == 'user' and isinstance(m['content'], list)
                           for p in m['content'] if p.get('type') == 'text')
            requests.append(context)
            step = context['observation']['step']
            self.assertEqual(context['reasoning_state'], 'DECIDE')
            images = [p for m in payload['messages'] if isinstance(m['content'], list)
                      for p in m['content'] if p.get('type') == 'image_url']
            self.assertEqual(len(images), 1 if step == 0 else 2)
            if step == 1:
                self.assertEqual(context['notebook'], 'Control mapping unknown; test UP.')
                self.assertEqual(context['previous_action']['prediction'], 'UP may change the target.')
                self.assertEqual(context['observation']['changed_cell_count'], 64)
            reply = {'action': {'action': 'UP'}, 'prediction': 'UP may change the target.',
                     'reflection': '' if step == 0 else 'The target changed; causation still uncertain.',
                     'notebook': 'Control mapping unknown; test UP.' if step == 0 else 'UP changed the board once.'}
            return {'choices': [{'message': {'tool_calls': [{'id': 'submission', 'type': 'function',
                'function': {'name': 'submit_decision', 'arguments': json.dumps(reply)}}]}}]}

        with patch.object(LocalVisionLlm, '_complete', respond):
            runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct', max_calls=1)
            self.addCleanup(runtime.close)
            self.assertEqual(runtime.decide(observation())['action'], 'ACTION1')
            self.assertEqual(runtime.decide(observation(1, 1))['action'], 'ACTION1')
            self.assertEqual(runtime.memory.notebook, 'UP changed the board once.')
            self.assertEqual(len(requests), 2)
            self.assertEqual(runtime.memory.model_calls, 2)
            self.assertEqual(runtime.memory.facts, {})  # Model notes never become verified facts.
            self.assertEqual(runtime.memory.plan, [])
            self.assertEqual(set(runtime.agents), {'DECIDE'})
