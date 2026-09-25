"""Tool protocol and real ADK progressive skill loading against a fake HTTP server."""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.skills import load_skill_from_dir
from agent.cognition.skills import ReasoningSkillToolset
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from agent.local_vlm import LocalVisionLlm
from scripts.build_notebook import SOURCES


class SkillAdapterTests(unittest.TestCase):
    def test_inline_discovery_and_native_on_demand_loading(self):
        received = []
        names = ['load_skill', 'load_skill_resource']
        arguments = [{'skill_name': 'test-skill'},
                     {'skill_name': 'test-skill', 'file_path': 'references/detail.md'}]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append(payload)
                index = len(received) - 1
                message = {'content': 'REFERENCE_CONFIRMED'} if index >= 2 else {
                    'content': None, 'tool_calls': [{'id': f'call_{index}', 'type': 'function',
                    'function': {'name': names[index], 'arguments': json.dumps(arguments[index])}}]}
                data = json.dumps({'choices': [{'message': message}]}).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data)
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)/'test-skill'; (root/'references').mkdir(parents=True)
                (root/'SKILL.md').write_text('---\nname: test-skill\ndescription: Test progressive discovery.\n---\nBODY_ONLY_AFTER_LOAD\nRead references/detail.md.')
                (root/'references/detail.md').write_text('RESOURCE_ONLY_AFTER_LOAD')
                model = LocalVisionLlm(model='test', api_base=f'http://127.0.0.1:{server.server_port}/v1')
                async def run():
                    service = InMemorySessionService()
                    await service.create_session(app_name='test', user_id='u', session_id='s')
                    agent = LlmAgent(name='test', model=model, include_contents='none',
                                     tools=[ReasoningSkillToolset(skills=[load_skill_from_dir(root)])])
                    runner = Runner(agent=agent, app_name='test', session_service=service)
                    replies = []
                    try:
                        async for event in runner.run_async(user_id='u', session_id='s',
                                new_message=types.Content(role='user', parts=[types.Part(text='Use test-skill.')])):
                            if event.content:
                                replies += [p.text for p in event.content.parts or [] if p.text]
                    finally:
                        await runner.close()
                    return replies
                self.assertIn('REFERENCE_CONFIRMED', asyncio.run(run()))
                self.assertEqual(len(received), 3)
                serialized = [json.dumps(p) for p in received]
                self.assertNotIn('BODY_ONLY_AFTER_LOAD', serialized[0])
                self.assertIn('BODY_ONLY_AFTER_LOAD', serialized[1])
                self.assertNotIn('RESOURCE_ONLY_AFTER_LOAD', serialized[1])
                self.assertIn('RESOURCE_ONLY_AFTER_LOAD', serialized[2])
                self.assertIn('test-skill', serialized[0])
                for payload in received:
                    names = {t['function']['name'] for t in payload['tools']}
                    self.assertEqual(names, {'load_skill', 'load_skill_resource'})
                    self.assertNotIn('run_skill_script', payload['messages'][0]['content'])
                for i in range(1, 3):
                    results = [m for m in received[i]['messages'] if m['role'] == 'tool']
                    self.assertEqual(results[-1]['tool_call_id'], f'call_{i-1}')
                self.assertEqual(len(model._exchanges), 3)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)

    def test_function_schema_and_parallel_result_ids(self):
        model = LocalVisionLlm(model='test', api_base='http://unused')
        request = LlmRequest(contents=[
            types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(id=i, name='read', args={'x': 1})) for i in ['a', 'b']]),
            types.Content(role='user', parts=[types.Part(function_response=types.FunctionResponse(id=i, name='read', response={'ok': True})) for i in ['b', 'a']]),
        ], config=types.GenerateContentConfig(tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
            name='read', parameters=types.Schema(type='OBJECT', properties={'x': types.Schema(type='INTEGER')}))])]))
        payload = model._payload(request)
        self.assertEqual(payload['tools'][0]['function']['parameters']['properties']['x']['type'], 'integer')
        self.assertEqual([m['tool_call_id'] for m in payload['messages'][1:]], ['b', 'a'])

    def test_invalid_calls_never_reach_adk_execution(self):
        model = LocalVisionLlm(model='test', api_base='http://unused')
        request = LlmRequest(contents=[types.Content(role='user', parts=[types.Part(text='test')])],
            config=types.GenerateContentConfig(tools=[types.Tool(function_declarations=[types.FunctionDeclaration(name='read')])]))
        async def consume():
            return [r async for r in model.generate_content_async(request)]
        for name, args in [('unknown', '{}'), ('read', '[]'), ('read', '{broken')]:
            with self.subTest(name=name, args=args):
                model.begin_invocation()
                message = {'tool_calls': [{'type': 'function', 'id': 'a', 'function': {'name': name, 'arguments': args}}]}
                with patch.object(LocalVisionLlm, '_complete', return_value={'choices': [{'message': message}]}):
                    with self.assertRaises(ValueError):
                        asyncio.run(consume())

    def test_tool_loop_has_request_limit(self):
        model = LocalVisionLlm(model='test', api_base='http://unused', max_requests=1)
        request = LlmRequest(contents=[types.Content(role='user', parts=[types.Part(text='test')])])
        async def consume():
            return [r async for r in model.generate_content_async(request)]
        with patch.object(LocalVisionLlm, '_complete', return_value={'choices': [{'message': {'content': 'ok'}}]}) as http:
            asyncio.run(consume())
            with self.assertRaisesRegex(ValueError, 'budget exhausted'):
                asyncio.run(consume())
            self.assertEqual(http.call_count, 1)
            model.begin_invocation()
            asyncio.run(consume())

