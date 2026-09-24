"""Completion validates before publishing and ends the actual ADK tool loop."""
import json
import unittest
from unittest.mock import patch

from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm


def context(payload):
    return next(json.loads(p['text']) for m in payload['messages']
                if m['role'] == 'user' and isinstance(m['content'], list)
                for p in m['content'] if p.get('type') == 'text')


def call(name, args, identifier='submit'):
    return {'id': identifier, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)}}


def plan(c):
    return {'action': {'action': 'UP'}, 'prediction': 'The controlled target may move up.'}


class CompletionTests(unittest.TestCase):
    def runtime(self, max_calls=1):
        r = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct', max_calls=max_calls)
        self.addCleanup(r.close)
        return r

    def decide(self, runtime):
        return runtime.decide({'game_id': 'test', 'state': 'NOT_FINISHED', 'step': 0,
                               'available_actions': ['ACTION1'], 'remaining_actions': 3, 'grid': [[0]]})

    def test_rejected_submission_does_not_publish_interpretation_and_can_be_corrected(self):
        r = self.runtime()
        requests = []
        def respond(model, payload):
            requests.append(payload)
            c = context(payload)
            declaration = next(t['function'] for t in payload['tools'] if t['function']['name'] == 'submit_decision')
            self.assertEqual(declaration['parameters']['$defs']['Action']['properties']['action']['enum'], ['UP'])
            result = plan(c)
            if len(requests) == 1:
                result['action'] = {'action': 'DOWN'}
                result['notebook'] = 'uncommitted'
            else:
                self.assertEqual(r.turn.memory.notebook, '')
                self.assertEqual(r.turn.memory.interpretations, [])
                response = next(json.loads(m['content']) for m in payload['messages'] if m['role'] == 'tool')
                self.assertFalse(response['accepted'])
                self.assertIn('Invalid model action', response['error'])
            return {'choices': [{'message': {'tool_calls': [call('submit_decision', result)]}}]}
        with patch.object(LocalVisionLlm, '_complete', respond):
            self.assertEqual(self.decide(r)['action'], 'ACTION1')
        self.assertEqual(len(requests), 2)
        self.assertEqual(r.memory.model_calls, 1)
        self.assertEqual(r.memory.notebook, '')

    def test_text_json_is_not_completion(self):
        r = self.runtime()
        with patch.object(LocalVisionLlm, '_complete', lambda model, p: {
                'choices': [{'message': {'content': json.dumps(plan(context(p)))}}]}):
            self.assertEqual(self.decide(r)['reason'], 'proposal_unavailable')
        self.assertIsNone(r.memory.pending)

    def test_native_completion_in_final_request_reaches_act_without_followup(self):
        r = self.runtime()
        r._agent('DECIDE').model.max_requests = 1
        def respond(model, p):
            self.assertEqual(p['tool_choice'], 'required')
            return {'choices': [{'message': {'content': '<tool_call>' + json.dumps({
                'name': 'submit_decision', 'arguments': plan(context(p))}) + '</tool_call>'}}]}
        with patch.object(LocalVisionLlm, '_complete', respond):
            self.assertEqual(self.decide(r)['action'], 'ACTION1')
        self.assertEqual(len(r._agent('DECIDE').model._exchanges), 1)

    def test_mixed_completion_and_evidence_calls_execute_neither(self):
        r = self.runtime()
        def respond(model, p):
            return {'choices': [{'message': {'tool_calls': [call('submit_decision', plan(context(p))),
                call('move_cursor', {'x': 0, 'y': 0}, 'cursor')]}}]}
        with patch.object(LocalVisionLlm, '_complete', respond):
            self.assertEqual(self.decide(r)['reason'], 'proposal_unavailable')
        self.assertIsNone(r.cursor)
        self.assertIsNone(r.memory.pending)

    def test_final_request_rejects_information_tool_and_does_not_execute_action(self):
        r = self.runtime()
        r._agent('DECIDE').model.max_requests = 1
        def respond(model, p):
            self.assertEqual(p['tool_choice'], 'required')
            self.assertEqual([t['function']['name'] for t in p['tools']], ['submit_decision'])
            return {'choices': [{'message': {'content': '<tool_call>' + json.dumps({
                'name': 'list_observations', 'arguments': {}}) + '</tool_call>'}}]}
        with patch.object(LocalVisionLlm, '_complete', respond):
            self.assertEqual(self.decide(r)['reason'], 'proposal_unavailable')
        self.assertIsNone(r.memory.pending)
