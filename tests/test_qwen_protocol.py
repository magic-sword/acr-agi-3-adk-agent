"""Protocol and actual ADK dispatch tests for Qwen3-VL Hermes output."""
import json
import unittest
from unittest.mock import patch

from agent.qwen_protocol import QwenToolCallAdapter, ToolProtocolError, ToolCallNotAllowedError
from agent.local_vlm import LocalVisionLlm
from agent.cognition.workflow import CognitiveRuntime


def native(name='observe_current', arguments=None):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments or {}}) + '\n</tool_call>'


class QwenProtocolTests(unittest.TestCase):
    def setUp(self):
        self.adapter = QwenToolCallAdapter()

    def test_native_nested_json_and_delimiter_inside_string(self):
        text = native(arguments={'nested': {'text': '</tool_call>', 'items': [1, 2]}})
        parts, protocol = self.adapter.decode({'content': text}, allowed={'observe_current'})
        self.assertEqual(protocol, 'qwen_hermes_text')
        self.assertEqual(parts[0].function_call.args['nested']['text'], '</tool_call>')
        self.assertTrue(parts[0].function_call.id)

    def test_multiple_calls_and_duplicate_representation_execute_once(self):
        content = native() + '\n' + native()
        parts, _ = self.adapter.decode({'content': content}, allowed={'observe_current'})
        self.assertEqual(len({p.function_call.id for p in parts}), 2)
        structured = [{'id': 'x', 'type': 'function', 'function': {'name': 'observe_current', 'arguments': '{}'}}]
        parts, protocol = self.adapter.decode({'content': native(), 'tool_calls': structured}, allowed={'observe_current'})
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].function_call.id, 'x')
        self.assertEqual(protocol, 'structured_with_duplicate_native')
        with self.assertRaisesRegex(ToolProtocolError, 'Conflicting'):
            self.adapter.decode({'content': native(arguments={'x': 1}), 'tool_calls': structured}, allowed={'observe_current'})

    def test_policy_truncation_unregistered_and_malformed_calls_rejected(self):
        with self.assertRaises(ToolCallNotAllowedError):
            self.adapter.decode({'content': native()}, allowed={'observe_current'}, tool_choice='none')
        cases = [native('CLICK'), native()[:-3], native()+' trailing prose',
                 '<tool_call>{"name":"observe_current","arguments":[],"name":"x"}</tool_call>',
                 '<tool_call>{"name":"observe_current","arguments":[]}</tool_call>',
                 '<tool_call>{"name":"observe_current","arguments":{"x":NaN}}</tool_call>']
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ToolProtocolError):
                self.adapter.decode({'content': content}, allowed={'observe_current'})
        with self.assertRaisesRegex(ToolProtocolError, 'Truncated'):
            self.adapter.decode({'content': native()}, allowed={'observe_current'}, finish_reason='length')

    def test_quoted_examples_and_json_remain_text(self):
        for text in ('Example: '+native(), json.dumps({'summary': native()}), '```\n'+native()+'\n```'):
            parts, protocol = self.adapter.decode({'content': text}, allowed={'observe_current'})
            self.assertEqual(protocol, 'text')
            self.assertEqual(parts[0].text, text)

    def test_native_call_reaches_adk_and_result_returns_with_same_id(self):
        requests = []
        def respond(model, payload):
            requests.append(payload)
            if len(requests) == 1:
                message = {'content': native('list_observations')}
            else:
                tool = next(m for m in payload['messages'] if m['role']=='tool')
                assistant = next(m for m in payload['messages'] if m.get('tool_calls'))
                self.assertEqual(tool['tool_call_id'], assistant['tool_calls'][0]['id'])
                self.assertEqual(len(json.loads(tool['content'])['observations']), 1)
                context = next(json.loads(p['text']) for m in payload['messages']
                    if m['role']=='user' and isinstance(m['content'], list)
                    for p in m['content'] if p.get('type')=='text')
                message = {'content': json.dumps({'observation_id':context['observation_id'],
                    'memory_revision':context['memory_revision'], 'purpose':'plan', 'plan':[{
                        'id':'test', 'subgoal':'test input', 'action':{'action':'UP'},
                        'completion':[{'kind':'frame_changed','value':True}],
                        'effects':[{'kind':'frame_changed','value':True}]}]})}
            return {'choices':[{'message':message,'finish_reason':'stop'}]}
        with patch.object(LocalVisionLlm, '_complete', respond):
            runtime = CognitiveRuntime('test','local/qwen3-vl-4b-instruct')
            try:
                result = runtime.decide({'game_id':'test','state':'NOT_FINISHED','step':0,
                    'available_actions':['ACTION1'],'remaining_actions':2,'grid':[[0]]})
                self.assertEqual(result['action'], 'ACTION1')
                self.assertEqual(len(requests), 2)
                self.assertEqual(runtime._agent('PLAN').model._exchanges[0]['decoded_protocol'], 'qwen_hermes_text')
            finally:
                runtime.close()
