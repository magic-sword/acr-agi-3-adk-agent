"""Action-first decisions retain learning without requiring a plan or experiment DAG."""
import json
import unittest
from unittest.mock import patch

from agent.cognition.state import Decision
from agent.cognition.engine import CognitiveTurn, new_memory
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from test_evidence_reasoning import observation


class SimpleLoopTests(unittest.TestCase):
    def test_trial_feedback_uses_frame_after_each_action_and_marks_boundaries(self):
        runtime = CognitiveRuntime('test')
        self.addCleanup(runtime.close)
        runtime.decide(observation())
        runtime.decide(observation(1, 1))
        runtime.decide(observation(2, 1))
        # Build the next decision's context before commit replaces runtime.memory.
        turn = CognitiveTurn(runtime.memory, observation(3, 1, levels_completed=1))
        turn.observe()
        runtime.turn = turn
        trials = runtime._context()['recent_trials']
        self.assertEqual([t['observed_frame_changed'] for t in trials], [True, False, None])
        self.assertTrue(all('notebook' not in t for t in trials))

    def test_rejected_action_cannot_change_notes_or_selection(self):
        turn = CognitiveTurn(new_memory('test'), observation())
        turn.observe()
        turn.memory.notebook = 'Prior useful hypothesis'
        with self.assertRaisesRegex(ValueError, 'Invalid model action'):
            turn.accept_decision(Decision(action={'action': 'DOWN'}, prediction='move', notebook='bad'))
        self.assertEqual(turn.memory.notebook, 'Prior useful hypothesis')
        self.assertIsNone(turn.selected)
        self.assertIsNone(turn.decision)

    def test_missing_prediction_is_rejected(self):
        for reply in ({'action': {'action': 'UP'}}, {'action': {'action': 'UP'}, 'prediction': ''}):
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                Decision.model_validate(reply)

    def test_notes_can_be_retained_and_cleared(self):
        turn = CognitiveTurn(new_memory('test'), observation())
        turn.observe()
        turn.memory.notebook = 'Hidden object last seen at step 0'
        turn.accept_decision(Decision(action={'action': 'UP'}, prediction='move'))
        self.assertEqual(turn.memory.notebook, 'Hidden object last seen at step 0')
        turn.accept_decision(Decision(action={'action': 'UP'}, prediction='move', notebook=''))
        self.assertEqual(turn.memory.notebook, '')

    def test_boundary_keeps_notes_but_does_not_present_old_board_as_before(self):
        runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct', max_calls=1)
        self.addCleanup(runtime.close)
        seen = []

        def respond(model, payload):
            parts = next(m['content'] for m in payload['messages'] if m['role'] == 'user')
            context = json.loads(parts[0]['text'])
            seen.append(context)
            if len(seen) == 2:
                self.assertTrue(context['boundary'])
                self.assertEqual(context['notebook'], 'UP may move the object; recheck next level.')
                self.assertEqual(sum(p['type'] == 'image_url' for p in parts), 1)
            args = {'action': {'action': 'UP'}, 'prediction': 'Object may move.',
                    'notebook': 'UP may move the object; recheck next level.'}
            return {'choices': [{'message': {'tool_calls': [{'id': 'submit', 'type': 'function',
                    'function': {'name': 'submit_decision', 'arguments': json.dumps(args)}}]}}]}

        with patch.object(LocalVisionLlm, '_complete', respond):
            first = runtime.decide(observation())
            self.assertEqual(runtime.decide(observation()), first)
            runtime.decide(observation(1, 1, levels_completed=1))
            self.assertEqual(runtime.decide(observation(2, 2, state='WIN'))['status'], 'done')
        self.assertEqual(len(seen), 2)  # Neither duplicate input nor WIN calls the model.

    def test_expired_budget_prevents_note_changes(self):
        turn = CognitiveTurn(new_memory('test'), observation(), deadline=0)
        turn.observe()
        with self.assertRaisesRegex(ValueError, 'budget'):
            turn.accept_decision(Decision(action={'action': 'UP'}, prediction='move', notebook='bad'))
        self.assertEqual(turn.memory.notebook, '')
        self.assertIsNone(turn.selected)
