"""Single-token decisions using the existing local vision transport.

Token probabilities are diagnostics, not calibrated probabilities of success.
"""
import asyncio
import json
import time


async def choose(model, *, instruction, parts, options, observe_request, timeout):
    labels = list(options)
    if not labels or any(len(k) != 1 or k not in '12345678' for k in labels):
        raise ValueError('choices require distinct single-digit labels 1 through 8')
    payload = {'model': model.model, 'messages': [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': parts}],
        'temperature': 0, 'max_tokens': 1, 'stream': False,
        'cache_prompt': True, 'logprobs': True, 'top_logprobs': 20,
        'grammar': 'root ::= ' + ' | '.join(json.dumps(k) for k in labels)}
    record = {'request_sha256': observe_request(payload), 'http_requests': 1}
    started = time.monotonic()
    try:
        model.timeout_seconds = max(.001, timeout)
        async with asyncio.timeout(timeout):
            response = await asyncio.to_thread(model._complete, payload)
        choice = response['choices'][0]
        label = (choice.get('message', {}).get('content') or '')
        record.update(usage=response.get('usage'), timings=response.get('timings'),
                      finish_reason=choice.get('finish_reason'), raw_response=choice,
                      server_model=response.get('model'))
        if label not in options or choice.get('message', {}).get('tool_calls'):
            raise ValueError('expected exactly one offered choice token')
        record.update(schema_valid=True, response=json.dumps({'label': label}), label=label)
    except Exception as exc:
        record.update(schema_valid=False, error=f'{type(exc).__name__}: {exc}'[:500])
    finally:
        record['seconds'] = time.monotonic()-started
    return record
