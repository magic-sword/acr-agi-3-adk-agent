"""V5: can Qwen use ADK skills (SKILL.md structure) and program tools for exact planning?

Task: bring the ls20 player block into the framed box at the top from a start where going
straight up is blocked. Ground truth is the shortest path under the rules V1 learned.

  skills     SkillToolset with three skills built from measured rules; loading the movement
             skill exposes plan_path / predict_effect (ADK additional_tools, in-process, so
             they read the current frame). list_skills is removed so the L1 list sits in the
             prompt and no request is spent on discovery.
  no-skills  the same rules as plain text; Qwen must work out the moves itself.
  protocol   skills, but the first request offers only load_skill (tool_choice=required):
             Qwen still chooses which skill to load; it cannot skip loading.

Runs inside the dev container (google-adk, VLM at $VLM_API_BASE).
usage: python scripts/verify_skill_tools.py
"""
import asyncio
from collections import deque
import glob
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.adk.skills import models  # noqa: E402
from google.adk.tools import FunctionTool, ToolContext  # noqa: E402
from google.adk.tools.skill_toolset import ListSkillsTool, SkillToolset  # noqa: E402
from google.genai import types  # noqa: E402

from agent.local_vlm import LocalVisionLlm  # noqa: E402
from agent.rendering import render_current, png_base64  # noqa: E402

RUN = ROOT / 'outputs/evaluations/20260930T013243650281Z'
OUT = ROOT / 'outputs/v5-skills'
PASSABLE, BLOCKING = {3}, {4}          # learned in V1: entered dark gray, stopped by very dark gray
STEP, SIZE = 5, 5
PLAYER0 = (34, 45)                     # measured origin of the orange+blue block in level 1
TARGET_BOX = (33, 9, 39, 15)           # framed black box at the top (x0, y0, x1, y1)
DELTA = {'UP': (0, -STEP), 'DOWN': (0, STEP), 'LEFT': (-STEP, 0), 'RIGHT': (STEP, 0)}
SAMPLES = [0.0, 0.8, 0.8, 0.8, 0.8]
CONDITIONS = tuple(sys.argv[1:]) or ('no skills', 'skills', 'protocol')


def level1():
    rec = glob.glob(str(RUN / 'ls20*' / 'recordings' / '*' / '*.jsonl'))[0]
    return json.loads(open(rec).readline())['data']['frame'][-1]


def place_player(grid, origin):
    """Move the measured 5x5 block to another corridor cell (controlled start)."""
    g = [row[:] for row in grid]
    x0, y0 = PLAYER0
    block = [row[x0:x0+SIZE] for row in grid[y0:y0+SIZE]]
    for y in range(SIZE):
        for x in range(SIZE):
            g[y0+y][x0+x] = 3
    nx, ny = origin
    for y in range(SIZE):
        for x in range(SIZE):
            g[ny+y][nx+x] = block[y][x]
    return g


class Board:
    def __init__(self, grid, origin):
        self.grid, self.origin = grid, origin

    def free(self, x, y):
        own = {(self.origin[0]+i, self.origin[1]+j) for i in range(SIZE) for j in range(SIZE)}
        for j in range(SIZE):
            for i in range(SIZE):
                cx, cy = x+i, y+j
                if not (0 <= cx < 64 and 0 <= cy < 64):
                    return False
                if (cx, cy) not in own and self.grid[cy][cx] in BLOCKING:
                    return False  # unknown colours are optimistic (WorldCoder): only learned blockers stop
        return True

    def inside(self, pos, box):
        x, y = pos
        return box[0] <= x and box[1] <= y and x+SIZE-1 <= box[2] and y+SIZE-1 <= box[3]

    def shortest(self, box):
        start = self.origin
        prev, queue = {start: None}, deque([start])
        while queue:
            p = queue.popleft()
            if self.inside(p, box):
                path = []
                while prev[p] is not None:
                    p, a = prev[p]
                    path.append(a)
                return path[::-1]
            for a, (dx, dy) in DELTA.items():
                q = (p[0]+dx, p[1]+dy)
                if q not in prev and self.free(*q):
                    prev[q] = (p, a)
                    queue.append(q)
        return None


OBJECTS = {'o1': ('the orange and blue block (player)', None),
           'o2': ('the framed black box at the top with a blue mark', TARGET_BOX),
           'o3': ('the white cross', (20, 31, 22, 33)),
           'o4': ('the yellow bar at the bottom', (13, 61, 54, 62))}
RULES = ('Measured rules: UP/DOWN/LEFT/RIGHT move the orange and blue block (the player, 5x5 cells) by 5 cells. '
         'It entered dark gray (colour 3) cells and was stopped by very dark gray (colour 4) walls. '
         'Every action shortens the yellow bar at the bottom (a step budget). The white cross is untested.')


def build(board, temperature, with_skills, protocol=False):
    result = {}

    def plan_path(target_object_id: str) -> dict:
        """Shortest arrow sequence that brings the player block inside the target object, using learned walls."""
        box = OBJECTS.get(target_object_id, (None, None))[1]
        if box is None:
            return {'error': f'unknown or non-target object {target_object_id}'}
        path = board.shortest(box)
        return {'actions': path, 'note': 'Exact under the learned rules.'} if path is not None else {'error': 'unreachable'}

    def predict_effect(control: str) -> dict:
        """Predict what one arrow press does from the current position."""
        d = DELTA.get(control.upper())
        if d is None:
            return {'error': 'unknown control'}
        x, y = board.origin[0]+d[0], board.origin[1]+d[1]
        return {'moves': board.free(x, y), 'new_origin': [x, y] if board.free(x, y) else list(board.origin)}

    def submit_moves(actions: list[str], skill_used: str, tool_context: ToolContext) -> dict:
        """Submit the planned arrow presses (at most 12) in order. skill_used is the loaded skill name or 'none'."""
        result.update(actions=[a.upper() for a in actions], skill_used=skill_used)
        tool_context.actions.skip_summarization = True  # submission ends the turn, as in the runtime
        return {'accepted': True}

    tools = [FunctionTool(submit_moves)]
    if with_skills:
        move = models.Skill(
            frontmatter=models.Frontmatter(
                name='move-orange-blue-block',
                description='Arrow keys move the orange and blue block (the player) 5 cells; walls stop it. '
                            'Load before planning any movement of the player.',
                metadata={'adk_additional_tools': ['plan_path', 'predict_effect']}),
            instructions=('Measured in this game: UP/DOWN/LEFT/RIGHT move the 5x5 player block by 5 cells. '
                          'It entered dark gray (3) and was stopped by very dark gray (4) walls in every try.\n'
                          'Do not work out directions yourself: call plan_path(target_object_id) and submit its '
                          'actions in order. Use predict_effect(control) to check a single press.'))
        budget = models.Skill(frontmatter=models.Frontmatter(
            name='step-budget-bar', description='Every action shortens the yellow bar at the bottom: a step budget, not a goal.'),
            instructions='Prefer short plans. The bar is not a target.')
        cross = models.Skill(frontmatter=models.Frontmatter(
            name='white-cross-tile', description='The white cross has not been tested; its effect is unknown.'),
            instructions='Stepping on the white cross is a probe with unknown effect.')

        class PromptListedSkills(SkillToolset):
            # Put the L1 list in the prompt instead of spending a request on list_skills.
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._tools = [t for t in self._tools if not isinstance(t, ListSkillsTool)]

        tools.append(PromptListedSkills(skills=[move, budget, cross],
                                        additional_tools=[FunctionTool(plan_path), FunctionTool(predict_effect)]))

    class Model(LocalVisionLlm):
        def _payload(self, request):
            payload = super()._payload(request)
            payload['temperature'] = temperature
            if protocol:
                # Every round must be a tool call; the first can only load a skill.
                if self._request_count == 0:
                    payload['tools'] = [t for t in payload.get('tools', []) if t['function']['name'] == 'load_skill']
                payload['tool_choice'] = 'required'
            return payload

    model = Model(model='qwen3-vl-4b-instruct', api_base=os.getenv('VLM_API_BASE', 'http://vlm:8080/v1'),
                  max_output_tokens=400, max_requests=6, completion_tools=('submit_moves',))
    instruction = ('Plan the arrow presses that achieve the goal. Submit them with submit_moves.'
                   + (' Reply only with a tool call, without explanatory text.' if protocol else '')
                   + ('' if with_skills else ' ' + RULES + ' Work out the moves from the image.'))
    agent = LlmAgent(name='planner', model=model, instruction=instruction, tools=tools)
    return agent, model, result


async def trial(start, temperature, condition):
    with_skills = condition != 'no skills'
    board = Board(place_player(level1(), start), start)
    agent, model, result = build(board, temperature, with_skills, protocol=condition == 'protocol')
    service = InMemorySessionService()
    await service.create_session(app_name='v5', user_id='u', session_id='s')
    runner = Runner(agent=agent, app_name='v5', session_service=service)
    objects = '\n'.join(f'{k}: {v[0]}' + (f' at x{v[1][0]}-{v[1][2]}, y{v[1][1]}-{v[1][3]}' if v[1] else
                        f' at x{start[0]}-{start[0]+4}, y{start[1]}-{start[1]+4}') for k, v in OBJECTS.items())
    text = f'Goal hypothesis: move the player block o1 inside o2.\nObjects:\n{objects}'
    image = png_base64(render_current(board.grid))
    message = types.Content(role='user', parts=[types.Part.from_bytes(data=__import__('base64').b64decode(image), mime_type='image/png'),
                                                types.Part(text=text)])
    model.begin_invocation()
    started = time.monotonic()
    error = None
    try:
        async for _ in runner.run_async(user_id='u', session_id='s', new_message=message):
            pass
    except Exception as exc:  # budget exhausted, malformed call
        error = f'{type(exc).__name__}: {exc}'[:200]
    finally:
        await runner.close()
    calls = [c for r in model._exchanges for c in r.get('normalized_tool_calls', [])]
    return dict(start=start, temperature=temperature, condition=condition, truth=board.shortest(TARGET_BOX),
                submitted=result.get('actions'), skill_used=result.get('skill_used'), error=error,
                requests=len(model._exchanges), seconds=round(time.monotonic()-started, 1),
                calls=[(c['name'], c['arguments']) for c in calls])


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in ((29, 45), (24, 45)):
        for condition in CONDITIONS:
            for t in SAMPLES:
                r = await trial(start, t, condition)
                rows.append(r)
                print(json.dumps(r))
    (OUT / f"trials-{'-'.join(c.replace(' ', '_') for c in CONDITIONS)}.json").write_text(json.dumps(rows, indent=1))
    print('|condition|exact path|first move correct|requests (mean)|seconds (mean)|loaded movement skill|called plan_path|')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for condition in CONDITIONS:
        rs = [r for r in rows if r['condition'] == condition]
        n = len(rs)
        exact = sum(r['submitted'] == r['truth'] for r in rs)
        first = sum(bool(r['submitted']) and r['submitted'][0] == r['truth'][0] for r in rs)
        loaded = sum(any(c[0] == 'load_skill' and (c[1] or {}).get('skill_name') == 'move-orange-blue-block' for c in r['calls']) for r in rs)
        planned = sum(any(c[0] == 'plan_path' for c in r['calls']) for r in rs)
        print(f"|{condition}|{exact}/{n}|{first}/{n}|"
              f"{sum(r['requests'] for r in rs)/n:.1f}|{sum(r['seconds'] for r in rs)/n:.1f}|{loaded}/{n}|{planned}/{n}|")


if __name__ == '__main__':
    asyncio.run(main())
