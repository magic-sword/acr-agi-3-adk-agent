"""Qwen3-VL Hermes tool-call framing → ADK function calls.

Transport normalization only: ADK owns execution, and host tool policy still applies.
"""
import json
import uuid

from google.genai import types


class ToolProtocolError(ValueError):
    pass


class ToolCallNotAllowedError(ToolProtocolError):
    pass


def strict_json():
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ToolProtocolError('Duplicate JSON key in tool call')
            result[key] = value
        return result
    def constant(value):
        raise ToolProtocolError('Non-finite JSON value in tool call')
    return json.JSONDecoder(object_pairs_hook=pairs, parse_constant=constant)


class QwenToolCallAdapter:
    """Accept structured API calls and complete standalone Hermes JSON blocks.

    Quoted examples/ordinary text are never scanned for executable embedded calls.
    Qwen3-Coder's different <function=...> grammar is intentionally unsupported.
    """

    @staticmethod
    def _native(content):
        if not isinstance(content, str) or not content.lstrip().startswith('<tool_call>'):
            return None
        decoder = strict_json()
        remaining = content.strip()
        calls = []
        while remaining:
            if not remaining.startswith('<tool_call>'):
                raise ToolProtocolError('Expected standalone Qwen tool-call blocks without trailing prose')
            remaining = remaining[len('<tool_call>'):].lstrip()
            try:
                value, end = decoder.raw_decode(remaining)
            except ValueError as exc:
                raise ToolProtocolError(f'Invalid Qwen tool-call JSON: {exc}') from exc
            remaining = remaining[end:].lstrip()
            if not remaining.startswith('</tool_call>'):
                raise ToolProtocolError('Incomplete Qwen tool-call closing tag')
            remaining = remaining[len('</tool_call>'):].strip()
            if not isinstance(value, dict) or set(value) != {'name', 'arguments'}:
                raise ToolProtocolError('Qwen tool call requires exactly name and arguments')
            calls.append({'type': 'function', 'function': value})
        return calls

    @staticmethod
    def _validated(calls, allowed):
        result, ids = [], set()
        if not isinstance(calls, list):
            raise ToolProtocolError('tool_calls must be a list')
        for call in calls:
            if not isinstance(call, dict) or call.get('type') != 'function':
                raise ToolProtocolError('Expected a function tool call')
            fn = call.get('function')
            if not isinstance(fn, dict) or not isinstance(fn.get('name'), str) or fn['name'] not in allowed:
                raise ToolProtocolError('Model requested an unregistered function')
            args = fn.get('arguments')
            if isinstance(args, str):
                try:
                    args = strict_json().decode(args)
                except ValueError as exc:
                    raise ToolProtocolError(f'Invalid tool arguments: {exc}') from exc
            if not isinstance(args, dict):
                raise ToolProtocolError('Tool arguments must be a JSON object')
            call_id = call.get('id') or 'call_' + uuid.uuid4().hex
            if not isinstance(call_id, str) or call_id in ids:
                raise ToolProtocolError('Invalid or duplicate tool call ID')
            ids.add(call_id)
            result.append(types.Part(function_call=types.FunctionCall(id=call_id, name=fn['name'], args=args)))
        return result

    def decode(self, message, *, allowed, tool_choice='auto', finish_reason=None):
        content = message.get('content')
        native = self._native(content)
        structured = message.get('tool_calls') or []
        if native is not None or structured:
            if finish_reason == 'length':
                raise ToolProtocolError('Truncated generation must not execute tool calls')
            if tool_choice == 'none':
                raise ToolCallNotAllowedError('Tool call returned in final-answer-only request (tool_choice=none)')
            calls = self._validated(structured or native, allowed)
            source = 'openai_tool_calls' if structured else 'qwen_hermes_text'
            if structured and native is not None:
                other = self._validated(native, allowed)
                def signature(parts):
                    return [(p.function_call.name, p.function_call.args) for p in parts]
                if signature(calls) != signature(other):
                    raise ToolProtocolError('Conflicting structured and native tool calls')
                source = 'structured_with_duplicate_native'
            if native is None and isinstance(content, str) and content.strip():
                calls.insert(0, types.Part(text=content))
            return calls, source
        if isinstance(content, str) and content.strip():
            return [types.Part(text=content)], 'text'
        raise ToolProtocolError('Local model returned neither text nor tool calls')
