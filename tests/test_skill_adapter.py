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
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from agent.local_vlm import LocalVisionLlm
from scripts.build_notebook import SOURCES


class SkillAdapterTests(unittest.TestCase):
    def test_state_toolsets_expose_only_relevant_skills(self):
        from agent.cognition.workflow import CognitiveRuntime
        runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct')
        try:
            observe = {'visual-observation', 'occlusion-memory', 'prediction-verification',
                       'hypothesis-maintenance', 'goal-inference'}
            probe = {'discriminating-experiment', 'temporal-reasoning', 'action-grounding', 'budget-recovery'}
            plan = {'backward-planning', 'plan-repair', 'temporal-reasoning',
                    'effect-revaluation', 'action-grounding', 'procedure-reuse'}
            revision = {'hypothesis-maintenance', 'goal-inference', 'cause-diagnosis',
                        'plan-repair', 'effect-revaluation'}
            expected = {
                ('OBSERVE', None): observe,
                ('PROBE', None): probe,
                ('PLAN', None): plan,
                ('REVISE', 'PROBE'): revision | probe,
                ('REVISE', 'PLAN'): revision | plan,
            }
            for (state, destination), names in expected.items():
                agent = runtime._agent(state, destination)
                registered = [s.name for s in agent.tools[0]._list_skills()]
                self.assertEqual(set(registered), names)
                self.assertEqual(len(registered), len(names))
            from agent.cognition.skills import SKILL_NAMES
            self.assertEqual(set().union(*expected.values()),
                             set(SKILL_NAMES.values()) - {'decision-commit'})
            self.assertIsNot(runtime._agent('REVISE', 'PROBE'), runtime._agent('REVISE', 'PLAN'))
            self.assertIs(runtime._agent('REVISE', 'PROBE'), runtime._agent('REVISE', 'PROBE'))
        finally:
            runtime.close()

    def test_host_states_cannot_create_model_agents(self):
        from agent.cognition.workflow import CognitiveRuntime
        runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct')
        try:
            for state in ('VERIFY', 'UPDATE', 'ACT', 'COMMIT', 'CONSOLIDATE', 'RECOVER'):
                with self.subTest(state=state), self.assertRaisesRegex(ValueError, 'host-only'):
                    runtime._agent(state)
            with self.assertRaisesRegex(ValueError, 'destination'):
                runtime._agent('REVISE')
            self.assertFalse(runtime.agents)
        finally:
            runtime.close()

    def test_adk_three_stage_loading_and_tool_results(self):
        received = []
        names = ['list_skills', 'load_skill', 'load_skill_resource']
        arguments = [{}, {'skill_name': 'test-skill'},
                     {'skill_name': 'test-skill', 'file_path': 'references/detail.md'}]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append(payload)
                index = len(received) - 1
                message = {'content': 'REFERENCE_CONFIRMED'} if index >= 3 else {
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
                                     tools=[SkillToolset(skills=[load_skill_from_dir(root)])])
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
                self.assertEqual(len(received), 4)
                serialized = [json.dumps(p) for p in received]
                self.assertNotIn('BODY_ONLY_AFTER_LOAD', serialized[0])
                self.assertNotIn('BODY_ONLY_AFTER_LOAD', serialized[1])
                self.assertIn('BODY_ONLY_AFTER_LOAD', serialized[2])
                self.assertNotIn('RESOURCE_ONLY_AFTER_LOAD', serialized[2])
                self.assertIn('RESOURCE_ONLY_AFTER_LOAD', serialized[3])
                for i in range(1, 4):
                    results = [m for m in received[i]['messages'] if m['role'] == 'tool']
                    self.assertEqual(results[-1]['tool_call_id'], f'call_{i-1}')
                self.assertEqual(len(model._exchanges), 4)
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

    def test_notebook_packages_native_skills_and_references(self):
        self.assertIn('agent/skills/visual-observation/SKILL.md', SOURCES)
        self.assertIn('agent/skills/visual-observation/references/evidence-contract.md', SOURCES)

    def test_workflow_budget_allows_loading_before_final_decision(self):
        from agent.cognition.workflow import CognitiveRuntime
        payloads = []
        def respond(model, payload):
            payloads.append(payload)
            n = len(payloads)
            if n <= 3:
                name, args = [
                    ('list_skills', {}),
                    ('load_skill', {'skill_name': 'visual-observation'}),
                    ('load_skill_resource', {'skill_name': 'visual-observation',
                                             'file_path': 'references/evidence-contract.md'}),
                ][n-1]
                message = {'tool_calls': [{'id': f'c{n}', 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(args)}}]}
            else:
                context = next(json.loads(p['text']) for m in payload['messages']
                               if m['role'] == 'user' and isinstance(m['content'], list)
                               for p in m['content'] if p.get('type') == 'text')
                reply = {'observation_id': context['observation_id'], 'memory_revision': context['memory_revision']}
                if n == 4:
                    reply.update(goal='finish', facts=[], unknowns=[])
                else:
                    reply.update(purpose='plan', plan=[{'id': 'go', 'subgoal': 'finish',
                        'action': {'action': 'ACTION1'}, 'completion': [{'kind': 'state', 'value': 'WIN'}],
                        'effects': [{'kind': 'state', 'value': 'WIN'}]}])
                message = {'content': json.dumps(reply)}
            return {'choices': [{'message': message}]}
        with patch.object(LocalVisionLlm, '_complete', respond):
            runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct')
            try:
                result = runtime.decide({'game_id': 'test', 'state': 'NOT_FINISHED', 'step': 0,
                                         'available_actions': ['ACTION1'], 'remaining_actions': 1})
                self.assertEqual(result['action'], 'ACTION1')
                self.assertEqual(len(payloads), 5)
            finally:
                runtime.close()
