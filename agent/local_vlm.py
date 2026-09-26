"""ADK connector for local OpenAI-compatible vision and function calling.

ADK executes tools; this adapter only translates declarations, calls and results.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
import urllib.request
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr

from agent.qwen_protocol import QwenToolCallAdapter

MAX_REQUESTS_PER_INVOCATION = 8

def json_schema(value):
    """Normalize google.genai Schema types without changing property names."""
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json', exclude_none=True, by_alias=True)
    if isinstance(value, list):
        return [json_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == 'type' and isinstance(item, str):
            result[key] = item.lower()
        elif key == 'properties':
            result[key] = {name: json_schema(schema) for name, schema in item.items()}
        elif key != 'nullable':
            result[key] = json_schema(item)
    if value.get('nullable'):
        return {'anyOf': [result, {'type': 'null'}]}
    return result


class LocalVisionLlm(BaseLlm):
    api_base: str
    timeout_seconds: int = 180
    max_output_tokens: int = 1600
    max_requests: int = MAX_REQUESTS_PER_INVOCATION
    completion_tools: tuple[str, ...] = ()
    _last_metrics: dict = PrivateAttr(default_factory=dict)
    _exchanges: list = PrivateAttr(default_factory=list)
    _request_count: int = PrivateAttr(default=0)
    _request_observer: object = PrivateAttr(default=None)

    def begin_invocation(self):
        self._request_count = 0
        self._last_metrics = {}
        self._exchanges = []

    def _complete(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.api_base.rstrip('/') + '/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.load(response)

    def _payload(self, request: LlmRequest) -> dict:
        messages = []
        instruction = request.config.system_instruction
        if instruction:
            if isinstance(instruction, types.Content):
                instruction = '\n'.join(p.text for p in instruction.parts or [] if p.text)
            messages.append({'role': 'system', 'content': str(instruction)})
        pending = {}
        for content in request.contents:
            parts, calls, responses, visual_results = [], [], [], []
            for part in content.parts or []:
                if part.function_call:
                    call = part.function_call
                    if content.role != 'model' or not call.id or call.id in pending:
                        raise ValueError('Tool calls require unique IDs and model role')
                    pending[call.id] = call.name
                    calls.append({'id': call.id, 'type': 'function', 'function': {
                        'name': call.name, 'arguments': json.dumps(call.args or {})}})
                elif part.function_response:
                    response = part.function_response
                    call_id = response.id
                    if not call_id:
                        matches = [key for key, name in pending.items() if name == response.name]
                        if len(matches) == 1:
                            call_id = matches[0]
                    if pending.get(call_id) != response.name:
                        raise ValueError('Unmatched tool response')
                    del pending[call_id]
                    if response.parts:
                        raise ValueError('Multimodal function responses are not supported')
                    result = dict(response.response or {})
                    if response.name in {'observe_animation', 'get_observation', 'compare_observations'}:
                        encoded = result.pop('_image_png_base64', None)
                        if encoded:
                            visual_results.extend([
                                {'type': 'text', 'text': 'Visual tool result: ' + response.name + ' ' + json.dumps(result)},
                                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}}])
                    responses.append({'role': 'tool', 'tool_call_id': call_id,
                                      'content': json.dumps(result, ensure_ascii=False)})
                elif part.text:
                    parts.append({'type': 'text', 'text': part.text})
                elif part.inline_data:
                    blob = part.inline_data
                    if not blob.mime_type.startswith('image/'):
                        raise ValueError(f'Unsupported model input: {blob.mime_type}')
                    encoded = base64.b64encode(blob.data).decode('ascii')
                    parts.append({'type': 'image_url', 'image_url': {
                        'url': f'data:{blob.mime_type};base64,{encoded}'}})
                else:
                    raise ValueError('Unsupported ADK part in local vision model input')
            # Tool results must immediately follow the corresponding assistant call.
            messages.extend(responses)
            if visual_results:
                messages.append({'role': 'user', 'content': visual_results})
            if parts or calls:
                message = {'role': 'assistant' if content.role == 'model' else 'user',
                           'content': parts or None}
                if calls:
                    message['tool_calls'] = calls
                messages.append(message)
        if pending:
            raise ValueError('Tool call has no result')
        if not messages or all(m['role'] == 'system' for m in messages):
            raise ValueError('The vision model needs an observation')
        declarations = {}
        for tool in request.config.tools or []:
            if not isinstance(tool, types.Tool) or not tool.function_declarations:
                raise ValueError('Only function tools are supported')
            for declaration in tool.function_declarations:
                declarations[declaration.name] = declaration
        for name, tool in request.tools_dict.items():
            if name not in declarations:
                declaration = tool._get_declaration()
                if declaration is None:
                    raise ValueError(f'Tool has no function declaration: {name}')
                declarations[name] = declaration
        payload = {'model': self.model, 'messages': messages, 'temperature': 0,
                   'max_tokens': self.max_output_tokens, 'stream': False}
        if declarations:
            payload['tools'] = [{'type': 'function', 'function': {
                'name': d.name, 'description': d.description or '',
                'parameters': json_schema(d.parameters_json_schema if d.parameters_json_schema is not None
                                          else d.parameters) or {'type': 'object', 'properties': {}}}}
                for d in declarations.values()]
            payload.update(tool_choice='auto', parallel_tool_calls=False)
        return payload

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        if self._request_count >= self.max_requests:
            raise ValueError('Local model tool-round budget exhausted')
        payload = self._payload(llm_request)
        remaining = self.max_requests - self._request_count
        if payload.get('tools'):
            if remaining == 1 and self.completion_tools:
                payload['tools'] = [tool for tool in payload['tools']
                                    if tool['function']['name'] in self.completion_tools]
                payload['tool_choice'] = 'required'
            budget_instruction = (f'{remaining} HTTP requests remain in this reasoning state. '
                + ('Submit the completed result using the completion tool now. No further evidence lookups.'
                   if remaining == 1 and self.completion_tools else
                   'Load specialist guidance only if needed; preserve requests for evidence and task completion.'))
            # Qwen's template reads only the first system message. Keep the full state contract there.
            if payload['messages'][0]['role'] == 'system':
                payload['messages'][0]['content'] += '\n' + budget_instruction
            else:
                payload['messages'].insert(0, {'role': 'system', 'content': budget_instruction})
        self._request_count += 1
        started = time.monotonic()
        record = {'request_index': self._request_count,
                  'tool_choice': payload.get('tool_choice', 'auto'),
                  'available_tools': [t['function']['name'] for t in payload.get('tools', [])],
                  'tool_results': [m for m in payload['messages'] if m['role'] == 'tool']}
        self._exchanges.append(record)
        try:
            if self._request_observer is not None:
                record['request_sha256'] = self._request_observer(payload)
            result = await asyncio.to_thread(self._complete, payload)
            choice = result['choices'][0]
            message = choice['message']
            record.update(response=message, usage=result.get('usage'), timings=result.get('timings'),
                          finish_reason=choice.get('finish_reason'))
            allowed = {t['function']['name'] for t in payload.get('tools', [])}
            parts, protocol = QwenToolCallAdapter().decode(
                message, allowed=allowed, tool_choice=payload.get('tool_choice', 'auto'),
                finish_reason=choice.get('finish_reason'))
            calls = [part.function_call for part in parts if part.function_call]
            if any(call.name in self.completion_tools for call in calls) and len(calls) != 1:
                raise ValueError('Completion must be the only tool call in a response')
            record['decoded_protocol'] = protocol
            record['normalized_tool_calls'] = [
                {'id': p.function_call.id, 'name': p.function_call.name, 'arguments': p.function_call.args}
                for p in parts if p.function_call]
            self._last_metrics = {'usage': {
                key: sum((r.get('usage') or {}).get(key, 0) for r in self._exchanges)
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
                if any(r.get('usage') is not None for r in self._exchanges) else None,
                'finish_reason': choice.get('finish_reason'), 'server_model': result.get('model'),
                'timings': result.get('timings')}
            yield LlmResponse(content=types.Content(role='model', parts=parts))
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            record['seconds'] = time.monotonic() - started
