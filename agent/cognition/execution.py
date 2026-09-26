"""Game boundary, receipts, evidence and journals shared by the two decision speeds."""
import asyncio
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent.observation import VISUAL_PAYLOAD_KEYS
from .audit import append_record, request_snapshot, compact, tool_status
from .evidence import EvidenceStore
from .state import Memory
from .validation import validate_action
from .machine import TRANSITIONS

APP_NAME = 'arc_fast_slow'


class ExecutionRuntime:
    def __init__(self, game_id, model=None, *, seconds=600, decision_seconds=45,
                 max_resets=2, log_dir=None):
        if model not in (None, 'local/qwen3-vl-4b-instruct'):
            raise ValueError('unsupported model')
        if (not math.isfinite(seconds) or not math.isfinite(decision_seconds) or
                seconds <= 0 or decision_seconds <= 0 or type(max_resets) is not int or max_resets < 0):
            raise ValueError('invalid runtime budget')
        self.model, self.max_resets = model, max_resets
        self.deadline = time.monotonic()+seconds
        self.decision_seconds = decision_seconds
        self.turn_deadline = self.deadline
        self.memory = Memory(run_id=uuid.uuid4().hex, game_id=game_id)
        self.session_id = self.memory.run_id
        self.log_dir = Path(log_dir) if log_dir else None
        self.loop, self.lock = asyncio.new_event_loop(), threading.Lock()
        self.service = InMemorySessionService()
        self.evidence = EvidenceStore()
        self.obs, self.previous, self.outcome = {}, {}, None
        self.calls = self.http_requests = self.event_sequence = 0
        self.trace, self.errors, self.tool_executions = [], [], []
        self.job = self.result = self.submission = None
        self.initialized = self.closed = False
        self.work = 'understand'
        self.workflow = self._build_graph()
        self.runner = Runner(agent=self.workflow, app_name=APP_NAME, session_service=self.service)

    def time_left(self):
        return max(0, min(self.deadline, self.turn_deadline)-time.monotonic())

    def _record(self, kind, event, **data):
        self.event_sequence += 1
        append_record(self.log_dir, self.session_id, kind, {
            'sequence': self.event_sequence, 'event': event,
            'timestamp': datetime.now(timezone.utc).isoformat(), 'run_id': self.session_id,
            'game_id': self.memory.game_id, 'observation_id': self.obs.get('observation_id'),
            'step': self.obs.get('step'), **data})

    def _request_record(self, payload):
        record = request_snapshot(payload, self.log_dir)
        self._record('requests', 'model_request', state='DECIDE', work=self.work,
                     call_index=self.calls, **record)
        return record['request_sha256']

    def _before_tool(self, tool, args, tool_context):
        record = {'state': 'DECIDE', 'work': self.work, 'call_index': self.calls,
                  'tool_call_id': tool_context.function_call_id, 'tool': tool.name,
                  'arguments': compact(args), 'status': 'started'}
        self.tool_executions.append(record)
        self._record('tools', 'tool_started', **record)

    def _after_tool(self, tool, args, tool_context, tool_response):
        record = next(r for r in reversed(self.tool_executions)
                      if r['tool_call_id'] == tool_context.function_call_id and r['status'] == 'started')
        record.update(status=tool_status(tool_response), result=compact(tool_response))
        self._record('tools', 'tool_finished', **record)

    def _tool_error(self, tool, args, tool_context, error):
        self._after_tool(tool, args, tool_context, {'error': str(error)[:500]})

    def _machine_transition(self, event):
        source, target, condition, kind = TRANSITIONS[event]
        self._record('artifacts', 'machine_transition', transition=event, source=source,
                     target=target, condition=condition, category=kind)

    def _stop(self, reason):
        self.memory.lifecycle = 'DONE' if reason in ('win', 'level_limit') else 'STOPPED'
        self.memory.stop_reason = reason
        self.result = {'status': 'stop', 'reason': reason}

    def _select(self, action, prediction):
        action = validate_action(action, self.obs, reset=action.get('action') == 'RESET')
        did = f'{self.session_id}:{self.obs["step"]}'
        self.result = {'status': 'action', 'decision_id': did, **action.model_dump(exclude_none=True)}
        self.memory.pending = {'decision_id': did, 'observation_id': self.obs['observation_id'],
            'step': self.obs['step'], 'action': action.model_dump(exclude_none=True),
            'prediction': prediction, 'execution': 'selected',
            'skill': deepcopy(self.memory.active_skill)}
        self.memory.lifecycle = 'AWAIT_FRAME'
        self._record('artifacts', 'action_selected', state='RUN', pending=self.memory.pending)

    def _receive(self, observation):
        obs = deepcopy(observation)
        if obs.get('game_id') != self.memory.game_id:
            raise ValueError('wrong game')
        for key in ('step', 'remaining_actions'):
            if type(obs.get(key)) is not int or obs[key] < 0:
                raise ValueError(f'invalid {key}')
        if obs.get('state') not in ('NOT_PLAYED', 'NOT_FINISHED', 'GAME_OVER', 'WIN'):
            raise ValueError('invalid game state')
        if obs.get('grid') is not None:
            g = obs['grid']
            if not g or len(g)>64 or not g[0] or len(g[0])>64 or any(
                    len(row)!=len(g[0]) or any(type(v) is not int or not 0<=v<=15 for v in row) for row in g):
                raise ValueError('invalid color grid')
            obs['width'], obs['height'] = len(g[0]), len(g)
            obs['frame_hash'] = hashlib.sha256(json.dumps(g).encode()).hexdigest()
        obs.setdefault('available_actions', [])
        obs['observation_id'] = f'{self.session_id}:{obs["step"]}:{obs.get("frame_hash", "none")[:12]}'
        last = self.memory.last_observation
        if last and obs['step'] <= last['step']:
            if obs['step'] == last['step'] and all(obs.get(k) == last.get(k) for k in (
                    'frame_hash', 'state', 'levels_completed', 'available_actions', 'remaining_actions')):
                self.result = deepcopy(self.memory.last_result)
                return False
            raise ValueError('stale or conflicting observation')
        self.obs = obs
        self.previous = self.evidence.get(last['observation_id']) if last else {}
        pending = self.memory.pending
        boundary = ('reset' if obs.get('full_reset') or pending and pending['action']['action']=='RESET' else
                    'level' if last and obs.get('levels_completed',0)!=last.get('levels_completed',0) else '')
        self.outcome = None
        if pending:
            acknowledged = pending['execution']=='action_acknowledged' and obs['step']==pending['step']+1
            self.outcome = {'decision_id':pending['decision_id'], 'before_observation_id':pending['observation_id'],
                            'after_observation_id':obs['observation_id'], 'action': pending['action'], 'acknowledged': acknowledged,
                            'prediction': pending['prediction'], 'skill': pending.get('skill'),
                            'boundary': boundary, 'frame_changed': None if boundary else
                            obs.get('frame_hash') != last.get('frame_hash')}
            a, b = self.previous.get('grid'), obs.get('grid')
            if not boundary and a and b and len(a)==len(b) and all(len(x)==len(y) for x,y in zip(a,b)):
                changes = [{'x':x,'y':y,'before':a[y][x],'after':v}
                           for y,row in enumerate(b) for x,v in enumerate(row) if a[y][x]!=v]
                self.outcome.update(changed_cell_count=len(changes),
                    changed_cells=changes if len(changes)<=12 else [changes[i*(len(changes)-1)//11] for i in range(12)],
                    change_bounds=None if not changes else {
                        'x':min(c['x'] for c in changes), 'y':min(c['y'] for c in changes),
                        'width':max(c['x'] for c in changes)-min(c['x'] for c in changes)+1,
                        'height':max(c['y'] for c in changes)-min(c['y'] for c in changes)+1})
            self.memory.pending = None
        self.evidence.add(obs, boundary=bool(boundary), source_action=pending)
        self._log_observation()
        self.memory.last_observation = {k:v for k,v in obs.items() if k not in VISUAL_PAYLOAD_KEYS | {'grid'}}
        self.memory.revision += 1
        self.memory.lifecycle = 'ACTIVE'
        self._on_observation(boundary)
        if self.outcome and not self.outcome['acknowledged']:
            self._stop('execution_outcome_unknown')
        return True

    def _log_observation(self):
        if not self.log_dir:
            return
        obs = self.obs
        record = {k:v for k,v in obs.items() if k not in VISUAL_PAYLOAD_KEYS}
        frames = self.log_dir/'frames'
        frames.mkdir(parents=True, exist_ok=True)
        if obs.get('image_png_base64'):
            name = f'{self.session_id}-{obs["step"]:05d}.png'
            (frames/name).write_bytes(base64.b64decode(obs['image_png_base64']))
            record['image_path'] = 'frames/'+name
        if obs.get('_visual_frames'):
            name = f'{self.session_id}-{obs["step"]:05d}.json'
            (frames/name).write_text(json.dumps({'frames':obs['_visual_frames'],
                'available_actions':obs['available_actions'], **obs.get('animation', {})}))
            record['animation_archive_path'] = 'frames/'+name
        self._record('observations', 'observation_received', **{k:v for k,v in record.items()
                     if k not in ('game_id','step','observation_id')})

    async def _decide(self, observation):
        self.calls = self.http_requests = 0
        self.trace, self.errors, self.tool_executions = [], [], []
        self.job = self.result = None
        if self.memory.lifecycle in ('DONE','STOPPED'):
            return self.memory.last_result
        self.turn_deadline = min(self.deadline, time.monotonic()+self.decision_seconds)
        started = time.monotonic()
        if not self._receive(observation):
            return self.result
        if not self.initialized:
            await self.service.create_session(app_name=APP_NAME, user_id='player', session_id=self.session_id)
            self.initialized = True
        message = types.Content(role='user', parts=[types.Part(text=f'Observation {self.obs["observation_id"]}')])
        try:
            async for _ in self.runner.run_async(user_id='player', session_id=self.session_id, new_message=message):
                pass
        except Exception as exc:
            self.errors.append(f'{type(exc).__name__}: {exc}'[:500])
            self._stop('runtime_error')
        if self.result is None:
            self._stop('missing_result')
        self.memory.last_result = deepcopy(self.result)
        row = {'step':self.obs['step'], 'observation_id':self.obs['observation_id'],
               'frame_hash':self.obs.get('frame_hash'), 'trace':self.trace, 'action':self.result,
               'outcome':self.outcome, 'errors':self.errors, 'decision_seconds':time.monotonic()-started,
               'lifecycle':self.memory.lifecycle, 'stop_reason':self.memory.stop_reason}
        self.memory.history = (self.memory.history+[row])[-128:]
        if self.log_dir:
            self.log_dir.mkdir(parents=True,exist_ok=True)
            (self.log_dir/f'{self.session_id}.json').write_text(self.memory.model_dump_json(indent=2))
            with (self.log_dir/f'{self.session_id}.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
        return self.result

    def decide(self, observation):
        with self.lock:
            if self.closed:
                raise RuntimeError('runtime closed')
            return self.loop.run_until_complete(self._decide(observation))

    def record_execution(self, event, **details):
        pending = self.memory.pending
        allowed = {'selected': {'action_dispatched'}, 'action_dispatched': {'action_acknowledged','action_outcome_unknown'}}
        if pending is None or event not in allowed.get(pending['execution'],set()):
            raise ValueError('invalid execution acknowledgement sequence')
        pending['execution'] = event
        self._record('execution', event, state='RUN', decision_id=pending['decision_id'], action=pending['action'], **details)

    def close(self):
        with self.lock:
            if not self.closed:
                self._record('states', 'runtime_closed', lifecycle=self.memory.lifecycle, stop_reason=self.memory.stop_reason)
                self.loop.run_until_complete(self.runner.close())
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.run_until_complete(self.loop.shutdown_default_executor())
                self.loop.close()
                self.evidence.close()
                self.closed = True
