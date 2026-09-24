"""Bounded real-model check of ADK L1/L2/L3 loading; no game or submission."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from google.genai import types
from agent.local_vlm import LocalVisionLlm


async def check():
    output = Path('outputs/skill-checks') / uuid.uuid4().hex
    output.mkdir(parents=True)
    model = LocalVisionLlm(model='qwen3-vl-4b-instruct',
                           api_base=os.getenv('VLM_API_BASE', 'http://vlm:8080/v1'),
                           timeout_seconds=20, max_output_tokens=512, max_requests=8)
    marker = 'verified-' + uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)/'connection-check'
        (root/'references').mkdir(parents=True)
        (root/'SKILL.md').write_text('---\nname: connection-check\ndescription: Verify the local skill connection.\n---\nLoad references/result.md with load_skill_resource. Reply with only the verification value found there.')
        (root/'references/result.md').write_text('Verification value: ' + marker)
        service = InMemorySessionService()
        await service.create_session(app_name='skill_check', user_id='u', session_id='s')
        agent = LlmAgent(name='skill_check', model=model, include_contents='none',
            instruction='First discover skills using list_skills. Select and load the relevant skill using load_skill. Follow its instructions. Do not guess verification values.',
            tools=[SkillToolset(skills=[load_skill_from_dir(root)])])
        runner = Runner(agent=agent, app_name='skill_check', session_service=service)
        events, texts = [], []
        failure = None
        try:
            async with asyncio.timeout(60):
                async for event in runner.run_async(user_id='u', session_id='s',
                    new_message=types.Content(role='user', parts=[types.Part(text='Verify the local skill connection.')])):
                    events.append(event.model_dump(mode='json', exclude_none=True))
                    if event.content:
                        texts += [p.text for p in event.content.parts or [] if p.text]
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
        finally:
            await runner.close()
            calls = [call['function']['name'] for r in model._exchanges
                     for call in r.get('response', {}).get('tool_calls', [])]
            passed = (failure is None and marker in '\n'.join(texts)
                      and all(name in calls for name in ('list_skills', 'load_skill', 'load_skill_resource')))
            report = {'passed': passed, 'error': failure, 'calls': calls, 'texts': texts,
                      'http_requests': len(model._exchanges), 'metrics': model._last_metrics,
                      'exchanges': model._exchanges, 'events': events}
            (output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps({'passed': passed, 'calls': calls, 'log': str(output/'result.json'), 'error': failure}))
        return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(check()))
