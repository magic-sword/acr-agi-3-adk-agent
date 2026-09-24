"""Exercise bounded failure recovery through a real ADK/HTTP model round trip."""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from agent.cognition.workflow import CognitiveRuntime


class ModelRecoveryTests(unittest.TestCase):
    def run_server(self, responder):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(body)
                result = responder(body, len(requests))
                response = json.dumps({'choices': [{'message': {'content': result}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with patch.dict(os.environ, {'VLM_API_BASE': f'http://127.0.0.1:{server.server_port}/v1'}):
                runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct')
                try:
                    result = runtime.decide({'game_id': 'test', 'state': 'NOT_FINISHED', 'step': 0,
                                             'available_actions': ['ACTION1'], 'remaining_actions': 5})
                    return result, runtime.memory, requests
                finally:
                    runtime.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_malformed_plan_stops_after_bounded_retry(self):
        result, memory, requests = self.run_server(lambda body, n: 'not valid JSON')
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['reason'], 'proposal_unavailable')
        self.assertEqual(len(requests), 3)
        self.assertIsNone(memory.pending)

    def test_illegal_proposal_retries_without_sending_it_to_game(self):
        def respond(body, n):
            text = next(part['text'] for msg in body['messages'] if isinstance(msg['content'], list)
                        for part in msg['content'] if part.get('type') == 'text')
            context = json.loads(text)
            reply = {'observation_id': context['observation_id'], 'memory_revision': context['memory_revision']}
            reply.update(purpose='plan', plan=[{
                'id': 'go', 'subgoal': 'finish',
                'action': {'action': 'ACTION2' if n == 1 else 'ACTION1'},
                'completion': [{'kind': 'state', 'value': 'WIN'}],
                'effects': [{'kind': 'state', 'value': 'WIN'}],
            }])
            return json.dumps(reply)
        result, memory, requests = self.run_server(respond)
        self.assertEqual(result['action'], 'ACTION1')
        self.assertEqual(len(requests), 2)
        self.assertEqual(memory.revision, 1)
        self.assertEqual(memory.pending.action.action, 'ACTION1')
        self.assertTrue(memory.history[-1]['errors'])
