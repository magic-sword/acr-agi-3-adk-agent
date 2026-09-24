"""Bounded synthetic decision checks for specialist methods, using the local VLM.

These text cases check method application with an explicitly selected skill.
They do not measure visual recognition, autonomous skill selection or game scores.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.skills import load_skill_from_dir
from google.genai import types

from agent.cognition.skills import ReasoningSkillToolset
from agent.cognition.planning import PlanOrderTool
from agent.local_vlm import LocalVisionLlm

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    {
        'name': 'hidden-token', 'skill': 'occlusion-memory',
        'question': 'Observation o1 shows a token at (4,2). In o2 an opaque panel covers '
                    'that location. Between them a collection input was executed; its effect '
                    'is unknown. Which statement is justified about the token now?',
        'choices': {'A': 'It is still at (4,2), visible=false is enough to prove it.',
                    'B': 'It has been collected because it disappeared.',
                    'C': 'Its last visible position is known; still present and collected remain alternatives.'},
        'expected': 'C',
    },
    {
        'name': 'blocked-discriminator', 'skill': 'discriminating-experiment',
        'question': 'Two control hypotheses predict that UP moves a token north or west. '
                    'At location P both paths are blocked and both hypotheses predict no change. '
                    'A known safe relocation puts it at Q, where north and west are open and '
                    'the resulting positions are visibly distinct. Which test can distinguish them?',
        'choices': {'A': 'Relocate to Q, then test UP and compare the resulting positions.',
                    'B': 'Press UP repeatedly at P and compare changed versus unchanged frames.',
                    'C': 'Replay the same initial frame several times and count no-change results.'},
        'expected': 'A',
    },
    {
        'name': 'preserve-access', 'skill': 'backward-planning',
        'question': 'The goal requires an open gate and a block on a pad. The pad currently '
                    'lies on the only route to the switch. Placing the block there blocks that '
                    'route. It is known that activating the switch opens the gate permanently '
                    'and leaves the block reachable. Which ordering has justified prerequisites?',
        'choices': {'A': 'Place the block on the pad, then try to reach the switch.',
                    'B': 'Reach and activate the switch, then move the block onto the pad.',
                    'C': 'Ignore the gate because occupying the pad is visually sufficient.'},
        'expected': 'B',
    },
    {
        'name': 'repair-replacement', 'skill': 'plan-repair',
        'question': 'The old plan has A done, B invalid and C still required after B. Current '
                    'evidence confirms A remains achieved. An alternative B2 has known effects '
                    'that enable C. The runtime replaces the stored plan with the submitted '
                    'plan and forbids model-asserted completion. What should the replacement contain?',
        'choices': {'A': 'Only B2; the runtime will merge C automatically.',
                    'B': 'A marked done and the original invalid B marked done.',
                    'C': 'B2 and C as remaining todo nodes, C depending on B2, with A supported by current evidence.'},
        'expected': 'C',
    },
    {
        'name': 'reverse-access', 'skill': 'backward-planning',
        'question': 'The goal needs a block on a pad and a gate open. Activating the switch '
                    'opens the gate permanently but seals the only access to the block. '
                    'Moving the block onto the pad leaves the switch reachable. '
                    'Both operations are initially possible. Choose a feasible order.',
        'choices': {'A': 'Activate the switch, then fetch and place the block.',
                    'B': 'Place the block first, then activate the switch.',
                    'C': 'Only activate the switch; the pad requirement will resolve itself.'},
        'expected': 'B',
    },
    {
        'name': 'mutual-blocking', 'skill': 'backward-planning',
        'question': 'The goal requires A and B completed. A permanently removes the only '
                    'access required for B. B permanently removes the only access required '
                    'for A. Both are initially available; no staging, reversal or alternate '
                    'route is known. Which conclusion follows from the supplied effects?',
        'choices': {'A': 'A then B is feasible because A is listed first in the goal.',
                    'B': 'B then A is feasible because it reverses that ordering.',
                    'C': 'Neither order is feasible with these effects; another operation or assumption needs investigation.'},
        'expected': 'C',
    },
]


async def check_case(case):
    model = LocalVisionLlm(model='qwen3-vl-4b-instruct',
        api_base=os.getenv('VLM_API_BASE', 'http://vlm:8080/v1'),
        timeout_seconds=30, max_output_tokens=512, max_requests=4)
    service = InMemorySessionService()
    await service.create_session(app_name='method_check', user_id='u', session_id='s')
    agent = LlmAgent(name='method_check', model=model, include_contents='none',
        instruction='Read the requested skill using load_skill and apply it to the supplied '
                    'scenario. Treat the scenario as the available evidence. Return only a '
                    'JSON object with choice (A, B or C) and a short reason.',
        tools=[ReasoningSkillToolset(skills=[load_skill_from_dir(
            ROOT / 'agent/skills' / case['skill'])]), PlanOrderTool()])
    runner = Runner(agent=agent, app_name='method_check', session_service=service)
    texts, error = [], None
    try:
        prompt = {k: case[k] for k in ('skill', 'question', 'choices')}
        async with asyncio.timeout(60):
            async for event in runner.run_async(user_id='u', session_id='s',
                    new_message=types.Content(role='user', parts=[types.Part(text=json.dumps(prompt))])):
                if event.content:
                    texts.extend(p.text for p in event.content.parts or [] if p.text)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        await runner.close()
    calls = [c['function']['name'] for e in model._exchanges
             for c in e.get('response', {}).get('tool_calls', [])]
    try:
        answer = json.loads(texts[-1]) if texts else {}
        if not isinstance(answer, dict):
            answer = {}
    except ValueError:
        answer = {}
    return {**case, 'answer': answer, 'error': error, 'texts': texts, 'calls': calls,
            'passed': (error is None and 'load_skill' in calls and answer.get('choice') == case['expected']
                       and (case['skill'] != 'backward-planning' or 'check_plan_order' in calls)),
            'http_requests': len(model._exchanges), 'exchanges': model._exchanges}


async def main():
    output = ROOT / 'outputs/skill-checks' / ('methods-' + uuid.uuid4().hex)
    output.mkdir(parents=True)
    results = []
    for case in CASES:
        result = await check_case(case)
        results.append(result)
        (output / 'result.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps({k: result[k] for k in ('name', 'passed', 'answer', 'error', 'http_requests')}), flush=True)
    print(f'Results: {output / "result.json"}', flush=True)
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
