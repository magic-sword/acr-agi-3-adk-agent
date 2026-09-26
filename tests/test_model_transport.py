"""The local ADK transport preserves schemas, tool IDs and protocol validation."""
import asyncio
import unittest
from unittest.mock import patch

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from agent.local_vlm import LocalVisionLlm


class ModelTransportTests(unittest.TestCase):
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

