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
    def test_knowledge_toolset_rejects_unconnected_execution_capabilities(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'test-skill'
            root.mkdir()
            definition = '---\nname: test-skill\ndescription: A test method.\n'
            (root / 'SKILL.md').write_text(definition + '---\nUse evidence.')
            (root / 'scripts').mkdir()
            script = root / 'scripts' / 'helper.py'
            script.write_text('print("unconnected")')
            with self.assertRaisesRegex(ValueError, 'scripts or dynamic tools'):
                ReasoningSkillToolset(skills=[load_skill_from_dir(root)])
            script.unlink()
            (root / 'SKILL.md').write_text(definition
                + 'metadata:\n  adk_additional_tools: [unconnected_tool]\n---\nUse evidence.')
            with self.assertRaisesRegex(ValueError, 'scripts or dynamic tools'):
                ReasoningSkillToolset(skills=[load_skill_from_dir(root)])

    def test_out_of_scope_skill_cannot_be_loaded(self):
        from types import SimpleNamespace
        from agent.cognition.skills import skill_toolset

        async def run():
            toolset = skill_toolset('VERIFY')
            try:
                tools = await toolset.get_tools()
                loader = next(t for t in tools if t.name == 'load_skill')
                context = SimpleNamespace(invocation_id='scope', agent_name='verify', state={})
                result = await loader.run_async(args={'skill_name': 'backward-planning'},
                                               tool_context=context)
                self.assertEqual(result['error_code'], 'SKILL_NOT_FOUND')
                self.assertEqual(context.state, {})
            finally:
                await toolset.close()
        asyncio.run(run())

    def test_state_toolsets_expose_only_relevant_skills(self):
        from agent.cognition.workflow import CognitiveRuntime
        runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct')
        try:
            shared = {'visual-observation', 'occlusion-memory', 'hypothesis-maintenance', 'temporal-reasoning'}
            expected = {
                'VERIFY': shared | {'prediction-verification'},
                'PLAN': shared | {'goal-inference', 'backward-planning', 'plan-repair',
                                  'effect-revaluation', 'action-grounding', 'procedure-reuse'},
                'PROBE': shared | {'discriminating-experiment', 'action-grounding'},
                'REVISE': shared | {'goal-inference', 'cause-diagnosis', 'discriminating-experiment',
                                    'backward-planning', 'plan-repair', 'effect-revaluation',
                                    'action-grounding'},
            }
            for state, names in expected.items():
                agent = runtime._agent(state)
                registered = [s.name for s in agent.tools[1]._list_skills()]
                self.assertEqual(set(registered), names)
                self.assertEqual(len(registered), len(names))
                tools = {getattr(tool, 'name', None) or tool.__name__ for tool in agent.tools[2:]}
                self.assertTrue({'get_observation', 'compare_observations', 'list_observations'} <= tools)
                self.assertEqual('check_plan_order' in tools, state in ('PLAN', 'REVISE'))
            from agent.cognition.skills import SKILL_NAMES
            self.assertEqual(set().union(*expected.values()),
                             set(SKILL_NAMES.values()))
            self.assertIs(runtime._agent('REVISE'), runtime._agent('REVISE'))
        finally:
            runtime.close()

    def test_host_states_cannot_create_model_agents(self):
        from agent.cognition.workflow import CognitiveRuntime
        runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct')
        try:
            for state in ('OBSERVE', 'UPDATE', 'ACT', 'COMMIT', 'CONSOLIDATE', 'RECOVER'):
                with self.subTest(state=state), self.assertRaisesRegex(ValueError, 'host-only'):
                    runtime._agent(state)
            self.assertFalse(runtime.agents)
        finally:
            runtime.close()

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

    def test_notebook_packages_native_skills_and_references(self):
        self.assertIn('agent/skills/visual-observation/SKILL.md', SOURCES)
        self.assertIn('agent/skills/visual-observation/references/views.md', SOURCES)
        self.assertIn('agent/cognition/instructions.py', SOURCES)
        self.assertIn('agent/rendering.py', SOURCES)
        self.assertFalse(any('/scripts/' in path or '/decision-commit/' in path
                             or '/budget-recovery/' in path
                             for path in SOURCES if path.startswith('agent/skills/')))

    def test_workflow_budget_allows_loading_before_final_decision(self):
        from agent.cognition.workflow import CognitiveRuntime
        payloads = []
        def respond(model, payload):
            payloads.append(payload)
            systems = [m for m in payload['messages'] if m['role'] == 'system']
            self.assertEqual(len(systems), 1)
            self.assertIn('JSON schema:', systems[0]['content'])
            self.assertIn('Current reasoning state: PLAN', systems[0]['content'])
            self.assertIn('HTTP requests remain', systems[0]['content'])
            n = len(payloads)
            if n <= 2:
                name, args = [
                    ('load_skill', {'skill_name': 'visual-observation'}),
                    ('load_skill_resource', {'skill_name': 'visual-observation',
                                             'file_path': 'references/views.md'}),
                ][n-1]
                message = {'tool_calls': [{'id': f'c{n}', 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(args)}}]}
            else:
                context = next(json.loads(p['text']) for m in payload['messages']
                               if m['role'] == 'user' and isinstance(m['content'], list)
                               for p in m['content'] if p.get('type') == 'text')
                reply = {'observation_id': context['observation_id'], 'memory_revision': context['memory_revision']}
                self.assertEqual(payload['tool_choice'], 'required')
                self.assertEqual([t['function']['name'] for t in payload['tools']], ['submit_plan'])
                reply.update(purpose='plan', plan=[{'id': 'go', 'subgoal': 'finish',
                    'action': {'action': 'ACTION1'}, 'completion': [{'kind': 'state', 'value': 'WIN'}],
                    'effects': [{'kind': 'state', 'value': 'WIN'}]}])
                message = {'tool_calls': [{'id': 'submit', 'type': 'function', 'function': {'name': 'submit_plan', 'arguments': json.dumps(reply)}}]}
            return {'choices': [{'message': message}]}
        with patch.object(LocalVisionLlm, '_complete', respond):
            runtime = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct')
            runtime._agent('PLAN').model.max_requests = 3
            try:
                result = runtime.decide({'game_id': 'test', 'state': 'NOT_FINISHED', 'step': 0,
                                         'available_actions': ['ACTION1'], 'remaining_actions': 1})
                self.assertEqual(result['action'], 'ACTION1')
                self.assertEqual(len(payloads), 3)
            finally:
                runtime.close()
