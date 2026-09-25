"""Work-scoped ADK judgments share a notebook; RUN owns effects and skill gating."""
from __future__ import annotations
import asyncio
import base64
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
import uuid

from google.adk import Context, Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import node
from google.genai import types

from agent.controls import controller_context, ACTION_TO_BUTTON
from agent.observation import VISUAL_PAYLOAD_KEYS
from agent.local_vlm import LocalVisionLlm
from .audit import append_record, compact, tool_status, request_snapshot
from .completion import CompletionTool
from .evidence import EvidenceStore
from .instructions import COMMON, DESIGN, REVIEW, BUILD
from .experiments import Experiments
from .notebook import Notebook
from .library import SkillLibrary, check_all, step_action, digest
from .skills import skill_toolset, skill_instructions
from .state import Decision, Draft, Memory, ExperimentPlan, ExperimentReview, SkillDeferral, ExperimentRedesign, DesignRevision
from .validation import validate_action

APP_NAME = 'arc_skill_learning'

class CognitiveRuntime:
    def __init__(self, game_id, model=None, *, max_calls=4, max_resets=2, seconds=600,
                 log_dir=None, max_http_requests=8, learning=True, skill_library=None):
        if model not in (None, 'local/qwen3-vl-4b-instruct'):
            raise ValueError('unsupported model')
        if seconds <= 0 or not 1 <= max_calls <= 12 or not 1 <= max_http_requests <= 32 or max_resets < 0:
            raise ValueError('invalid runtime budget')
        self.model, self.learning = model, learning
        self.max_calls, self.max_http_requests, self.max_resets = max_calls, max_http_requests, max_resets
        self.deadline = time.monotonic() + seconds
        self.memory = Memory(run_id=uuid.uuid4().hex, game_id=game_id)
        self.session_id = self.memory.run_id
        self.log_dir = Path(log_dir) if log_dir else None
        self.loop, self.lock = asyncio.new_event_loop(), threading.Lock()
        self.service = InMemorySessionService()
        self.evidence = EvidenceStore()
        self.obs, self.previous, self.outcome = {}, {}, None
        self.calls = self.http_requests = 0
        self.errors, self.trace, self.tool_executions = [], [], []
        self.submission = self.job = self.result = None
        self.proposed_job = None
        self.initialized = self.closed = False
        self.event_sequence = 0
        directory = self.log_dir / self.session_id / 'skills' if self.log_dir else None
        self.library = SkillLibrary(game_id, directory, source=skill_library, emit=self._learning_event)
        self.notebook = Notebook(game_id, self.session_id,
            self.log_dir / self.session_id / 'notebook' if self.log_dir else None,
            emit=self._notebook_event)
        self.experiments = Experiments(self.notebook, self._experiment_event)
        self.boundary = ''
        self.controllers = {}
        self.work, self.learning_request = 'experiment_design', None
        self.workflow = self._build_graph()
        self.runner = Runner(agent=self.workflow, app_name=APP_NAME, session_service=self.service)

    def time_left(self):
        return max(0, self.deadline - time.monotonic())

    def _record(self, kind, event, **data):
        self.event_sequence += 1
        append_record(self.log_dir, self.session_id, kind, {
            'sequence': self.event_sequence,
            'event': event, 'timestamp': datetime.now(timezone.utc).isoformat(),
            'run_id': self.session_id, 'game_id': self.memory.game_id,
            'observation_id': self.obs.get('observation_id'), 'step': self.obs.get('step'),
            **data})

    def _learning_event(self, event, **data):
        self._record('learning', event, **data)

    def _experiment_event(self, event, **data):
        self._record('experiments', event, work=self.work, **data)

    def _notebook_event(self, event, **data):
        self._record('notebook', event, work=getattr(self, 'work', 'experiment_design'), **data)

    def _note_operation(self, operation, *args, **kwargs):
        if self.submission is not None:
            return {'error': 'a job has already been submitted'}
        try:
            return controller_context(operation(*args, **kwargs))
        except ValueError as exc:
            return {'error': str(exc)}

    def read_notebook(self, reference: str = '', query: str = '', offset: int = 0,
                      include_previous: bool = False) -> dict:
        """Open a note/bookmark; without reference search six index entries. Never advances time."""
        return self._note_operation(self.notebook.read, reference, query, offset, include_previous)

    def write_note(self, kind: str, title: str, text: str, evidence_ids: list[str],
                   note_id: str = '', expected_revision: int = 0) -> dict:
        """Write goal/hypothesis/plan/interpretation. Cite observation/experience/note IDs.

        New notes use empty note_id and revision 0. Updates require the read revision.
        For goal, update note_id='goal' with the revision from the opening page.
        Title max 80; text max 400 for goal, otherwise 1000. Host result pages are read-only.
        """
        return self._note_operation(self.notebook.write, kind, title, text, evidence_ids,
                                    note_id, expected_revision)

    def erase_note(self, note_id: str, expected_revision: int, reason: str) -> dict:
        """Withdraw an obsolete note; keep its past versions. Goal and results cannot be erased."""
        return self._note_operation(self.notebook.erase, note_id, expected_revision, reason)

    def set_bookmark(self, name: str, note_id: str) -> dict:
        """Set a named note reference (max six custom bookmarks); empty note_id removes it."""
        return self._note_operation(self.notebook.bookmark, name, note_id)

    def _agent(self):
        if self.work not in self.controllers:
            completion = [CompletionTool(self, 'submit_decision', Decision)]
            instruction = COMMON + DESIGN + skill_instructions('design-experiment')
            note_tools = [self.write_note, self.erase_note, self.set_bookmark]
            if self.work == 'experiment_review':
                completion = [CompletionTool(self, 'submit_review', ExperimentReview)]
                instruction = COMMON + REVIEW + skill_instructions('review-experiment')
                note_tools = []
            elif self.work == 'skill_creation':
                completion = [CompletionTool(self, 'propose_skill', Draft),
                              CompletionTool(self, 'defer_skill', SkillDeferral)]
                instruction = COMMON + BUILD + skill_instructions('skill-creator')
            self.controllers[self.work] = LlmAgent(
                name=self.work,
                include_contents='none',
                model=LocalVisionLlm(model='qwen3-vl-4b-instruct',
                    api_base=os.getenv('VLM_API_BASE', 'http://vlm:8080/v1'), max_output_tokens=1800,
                    completion_tools=tuple(t.name for t in completion)),
                instruction=instruction,
                tools=[*completion, skill_toolset(), self.read_notebook, *note_tools,
                       self.list_observations, self.get_observation, self.compare_observations,
                       self.observe_animation, self.read_skill],
                before_tool_callback=self._before_tool, after_tool_callback=self._after_tool,
                on_tool_error_callback=self._tool_error)
        return self.controllers[self.work]

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

    def list_observations(self) -> dict:
        """List actual retained observation references; does not advance the game."""
        return {'observations': self.evidence.list()}

    def get_observation(self, observation_id: str) -> dict:
        """Read a recorded observation, not a new game state."""
        try:
            return controller_context(self.evidence.view(observation_id))
        except ValueError as e:
            return {'error': str(e)}

    def compare_observations(self, before_id: str, after_id: str, offset: int = 0) -> dict:
        """Read recorded before/after images and measured pixel differences."""
        try:
            return controller_context(self.evidence.compare(before_id, after_id, offset))
        except ValueError as e:
            return {'error': str(e)}

    def observe_animation(self, event_id: str, start_frame: int = 0) -> dict:
        """Inspect historical intermediate frames. This never advances game time."""
        try:
            return controller_context(self.evidence.animation(event_id, start_frame))
        except ValueError as e:
            return {'error': str(e)}

    def read_skill(self, skill_id: str) -> dict:
        """Read an executable version, its observed evidence and evaluation status."""
        try:
            r = self.library.get(skill_id)
            return {'skill_id': skill_id, 'spec': r['spec'], 'status': r['status'],
                    'evaluation': r['evaluation'], 'trials': [
                        {'arguments': t['arguments'], 'outcome': t['outcome']} for t in r['trials']]}
        except ValueError as e:
            return {'error': str(e)}

    def validate_job(self, job):
        if isinstance(job, ExperimentReview):
            if self.work != 'experiment_review':
                raise ValueError('reviews are only accepted in the result review job')
            self.experiments.validate_review(job)
            return
        if self.time_left() <= 0:
            raise ValueError('time budget exhausted')
        if isinstance(job, ExperimentRedesign):
            if self.work != 'experiment_design':
                raise ValueError('redesign belongs to experiment design')
            return
        if isinstance(job, SkillDeferral):
            if self.work != 'skill_creation':
                raise ValueError('only a skill builder can defer construction')
            return
        if isinstance(job, Draft):
            if not self.learning:
                raise ValueError('learning disabled')
            if self.work != 'skill_creation':
                raise ValueError('choose learn before constructing a skill')
            # Draft validation performs the full evidence check on an isolated library.
            trial = SkillLibrary(self.memory.game_id)
            trial.records, trial.experiences = deepcopy(self.library.records), deepcopy(self.library.experiences)
            trial.sequence = self.library.sequence
            trial.draft(job)
            return
        if self.work != 'experiment_design':
            raise ValueError('actions and skill selection belong to experiment design')
        if not set(job.evidence_ids) <= self.library.experiences.keys():
            raise ValueError('unknown or expired experience reference')
        if job.kind == 'learn':
            if not self.learning:
                raise ValueError('learning disabled')
            if self.work != 'experiment_design':
                raise ValueError('already constructing a skill')
            if not all(self.library.experiences[k]['acknowledged'] for k in job.evidence_ids):
                raise ValueError('learning requires acknowledged experiences')
        if job.kind == 'act':
            action = validate_action(job.action, self.obs)
            self.experiments.validate(job.experiment, action.model_dump(exclude_none=True), self.obs)
        if job.kind in ('trial', 'evaluate') and not self.learning:
            raise ValueError('learning disabled')
        if job.skill_id:
            r = self.library.get(job.skill_id)
            expected = 'active' if job.kind == 'invoke' else 'candidate'
            if r['status'] != expected:
                raise ValueError(f'{job.kind} requires a {expected} skill')
            if job.kind != 'evaluate':
                if job.kind == 'trial' and len(r['trials']) >= 8:
                    raise ValueError('candidate trial limit reached; revise from evidence')
                step_action(self.library.spec(job.skill_id), 0, job.arguments, self.obs)

    def _context(self):
        obs = {k: v for k, v in self.obs.items() if k not in VISUAL_PAYLOAD_KEYS | {'grid', 'cursor'}}
        context = {'work': self.work, 'observation': obs, 'notebook': self.notebook.opening(),
            'skills': self.library.catalog(), 'learning_enabled': self.learning,
            'last_tool_result': getattr(self, 'job_result', None), 'errors': self.errors[-1:],
            'remaining_seconds': round(self.time_left(), 2)}
        if self.work == 'skill_creation':
            context['learning_request'] = self.learning_request
        return controller_context(context)

    def _request_record(self, payload):
        record = request_snapshot(payload, self.log_dir)
        self._record('requests', 'model_request', state='DECIDE', work=self.work, call_index=self.calls, **record)
        return record['request_sha256']

    async def _ask(self, ctx):
        self.calls += 1
        self.memory.model_calls += 1
        self.submission = None
        self.proposed_job = None
        context = self._context()
        self._notebook_event('notebook_opened', view=context['notebook'])
        method = {'experiment_design': 'design-experiment', 'experiment_review': 'review-experiment',
                  'skill_creation': 'skill-creator'}[self.work]
        self._record('tools', 'skill_instructions_loaded', work=self.work,
                     skill=method, source='host', instruction=skill_instructions(method))
        parts = [types.Part(text=json.dumps(context, separators=(',', ':')))]
        views = [('BEFORE the issued action', self.previous), ('CURRENT original game pixels', self.obs)]
        if self.work == 'experiment_review':
            data = self.experiments.active['data']
            views = [(label, self.evidence.get(data[key]) if data[key] in self.evidence.index else {})
                     for label, key in [('EXPERIMENT BEFORE', 'before_id'), ('EXPERIMENT AFTER', 'after_id')]]
        for label, obs in views:
            if not obs:
                parts.append(types.Part(text=label + ': original observation no longer retained; use only recorded measurements.'))
                continue
            parts.append(types.Part(text=label + ': ' + str(obs.get('observation_id'))))
            if obs.get('image_png_base64'):
                parts.append(types.Part.from_bytes(data=base64.b64decode(obs['image_png_base64']), mime_type='image/png'))
            elif obs.get('grid') is not None:
                parts.append(types.Part(text=json.dumps(obs['grid'])))
        parts.append(types.Part(text='Complete only the assigned work: ' + self.work + '.'))
        agent = self._agent()
        agent.model.begin_invocation()
        agent.model.max_requests = min(4, self.max_http_requests - self.http_requests)
        agent.model.timeout_seconds = max(1, min(90, int(self.time_left())))
        agent.model._request_observer = self._request_record
        started = time.monotonic()
        record = {'state': 'DECIDE', 'work': self.work, 'call_index': self.calls, 'context': context}
        try:
            async with asyncio.timeout(self.time_left()):
                await ctx.run_node(agent, node_input=types.Content(role='user', parts=parts))
            if self.submission is None:
                raise ValueError('model ended without an accepted submission')
            record.update(schema_valid=True, response=self.proposed_job.model_dump_json())
            if isinstance(self.submission, ExperimentRedesign):
                record['host_route'] = self.submission.model_dump()
            self.job = self.submission
        except Exception as e:
            self.errors.append(f'{type(e).__name__}: {e}'[:500])
            record['error'] = self.errors[-1]
        finally:
            record.update(agent.model._last_metrics)
            record.update(exchanges=agent.model._exchanges, http_requests=len(agent.model._exchanges),
                          seconds=time.monotonic()-started)
            self.http_requests += record['http_requests']
            self._record('model', 'model_decision', **record)

    def _stop(self, reason):
        self.experiments.interrupt(reason)
        self.memory.lifecycle = 'DONE' if reason in ('win', 'level_limit') else 'STOPPED'
        self.memory.stop_reason = reason
        self.result = {'status': 'stop', 'reason': reason}

    def _select(self, action, prediction, skill=None, experiment_id=None):
        action = validate_action(action, self.obs, reset=action.get('action') == 'RESET')
        did = f'{self.session_id}:{self.obs["step"]}'
        self.result = {'status': 'action', 'decision_id': did, **action.model_dump(exclude_none=True)}
        self.memory.pending = {'decision_id': did, 'observation_id': self.obs['observation_id'],
            'step': self.obs['step'], 'action': action.model_dump(exclude_none=True),
            'prediction': prediction, 'execution': 'selected', 'skill': deepcopy(skill),
            'experiment_id': experiment_id}
        self.memory.lifecycle = 'AWAIT_FRAME'
        self._record('artifacts', 'action_selected', state='RUN', pending=self.memory.pending)

    def _advance_skill(self):
        active = self.memory.active_skill
        if not active:
            return False
        try:
            spec = self.library.spec(active['id'])
            expected = 'candidate' if active['trial'] else 'active'
            if self.library.get(active['id'])['status'] != expected:
                raise ValueError('skill version no longer executable')
            action = step_action(spec, active['index'], active['arguments'], self.obs)
            self._select(action, 'Verify the procedure step effects against the next real observation.', active)
            return True
        except ValueError as e:
            self._end_skill('unknown', str(e))
            return False

    def _end_skill(self, outcome, reason):
        active = self.memory.active_skill
        if not active:
            return
        if active['trial']:
            self.library.finish_trial(active['id'], active['arguments'], active['trace'], outcome)
        elif outcome != 'pass':
            self.library.suspend(active['id'], reason)
        self._record('learning', 'skill_execution_finished', skill_id=active['id'],
                     outcome=outcome, reason=reason, trial=active['trial'])
        self.job_result = {'skill_id': active['id'], 'outcome': outcome, 'reason': reason}
        self.memory.active_skill = None

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
            obs.setdefault('frame_hash', digest(g))
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
        self.previous = self.evidence.get(last['observation_id']) if last and last['observation_id'] in self.evidence.index else {}
        pending = self.memory.pending
        boundary = ('reset' if obs.get('full_reset') or pending and pending['action']['action']=='RESET' else
                    'level' if last and obs.get('levels_completed',0)!=last.get('levels_completed',0) else '')
        self.outcome = None
        if pending:
            consecutive = obs['step'] == pending['step']+1
            acknowledged = pending['execution']=='action_acknowledged' and consecutive
            self.outcome = {'action': pending['action'], 'acknowledged': acknowledged,
                            'prediction': pending['prediction'], 'experiment_id': pending.get('experiment_id'),
                            'boundary': boundary,
                            'frame_changed': None if boundary else obs.get('frame_hash')!=last.get('frame_hash')}
            exp = self.library.add_experience(self.previous, pending['action'], obs,
                acknowledged=acknowledged, boundary_kind=boundary)
            self.outcome['experience_id'] = exp['id']
            a,b = self.previous.get('grid'),obs.get('grid')
            if not boundary and a and b and len(a)==len(b) and all(len(x)==len(y) for x,y in zip(a,b)):
                changes = [{'x':x,'y':y,'before':a[y][x],'after':v}
                           for y,row in enumerate(b) for x,v in enumerate(row) if a[y][x]!=v]
                self.outcome.update(changed_cell_count=len(changes), changed_cells=changes[:24])
            active = self.memory.active_skill
            if active:
                active['trace'].append(exp)
                spec = self.library.spec(active['id'])
                check = check_all(spec.steps[active['index']].after, self.previous, obs, active['arguments']) if acknowledged else None
                allowed_boundary = boundary == 'level' and active['index']==len(spec.steps)-1 and any(
                    c.kind=='level_increased' for c in spec.steps[active['index']].after)
                if boundary and not allowed_boundary:
                    self._end_skill('unknown', 'environment boundary')
                elif check is not True:
                    self._end_skill('fail' if check is False else 'unknown', 'effect contradicted or unavailable')
                else:
                    active['index'] += 1
                    if active['index'] >= len(spec.steps):
                        self._end_skill('pass', 'all observed effects passed')
            self.memory.pending = None

        self.evidence.add(obs, boundary=bool(boundary), source_action=pending)
        self._log_observation()
        self.memory.last_observation = {k:v for k,v in obs.items() if k not in VISUAL_PAYLOAD_KEYS | {'grid'}}
        self.memory.revision += 1
        if self.memory.lifecycle not in ('STOPPED','DONE'):
            self.memory.lifecycle = 'ACTIVE'
        self.boundary = boundary
        # Preserve the old segment until its experiment has a verdict.
        self.notebook.record_observation(obs, self.outcome)
        if self.experiments.active:
            self.experiments.observe(obs, self.outcome)
            self.work = 'experiment_review'
            if not self.outcome['acknowledged']:
                self._finish_review(self.experiments.automatic_review(), source='host')
        if self.outcome and not self.outcome['acknowledged']:
            self._stop('execution_outcome_unknown')
        if not self.experiments.active:
            self._apply_boundary()
        return True

    def _log_observation(self):
        if not self.log_dir:
            return
        obs = self.obs
        record = {k:v for k,v in obs.items() if k not in VISUAL_PAYLOAD_KEYS}
        frames = self.log_dir / 'frames'
        frames.mkdir(parents=True, exist_ok=True)
        if obs.get('image_png_base64'):
            name = f'{self.session_id}-{obs["step"]:05d}.png'
            (frames/name).write_bytes(base64.b64decode(obs['image_png_base64']))
            record['image_path'] = 'frames/'+name
        if obs.get('_visual_frames'):
            name = f'{self.session_id}-{obs["step"]:05d}.json'
            (frames/name).write_text(json.dumps({'frames': obs['_visual_frames'],
                'available_actions': obs['available_actions'], **obs.get('animation',{})}))
            record['animation_archive_path'] = 'frames/'+name
        self._record('observations', 'observation_received', **{k:v for k,v in record.items() if k not in ('game_id','step','observation_id')})

    def _apply_boundary(self):
        if self.boundary:
            self.notebook.begin_segment(self.boundary)
            self.notebook.record_observation(self.obs, self.outcome)
            self.boundary = ''

    def _finish_review(self, review, *, source):
        self.experiments.finish(review, source=source)
        self.job_result = None  # The verdict has one opening-view source: latest_review.
        if self.outcome is not None and self.outcome.get('experiment_id') == review.experiment_id:
            self.outcome['review'] = review.model_dump()
        self.work, self.learning_request = 'experiment_design', None
        self._apply_boundary()

    def _offline_job(self):
        allowed = [a for a in self.obs['available_actions'] if a.startswith('ACTION')]
        if not allowed:
            self._stop('no_legal_action')
            return
        action = allowed[self.obs['step'] % len(allowed)]
        data = {'action': action, 'reason': 'deterministic control probe'}
        x = self.obs['step']*11 % self.obs.get('width', 64)
        y = self.obs['step']*7 % self.obs.get('height', 64)
        if action == 'ACTION6':
            data.update(x=x, y=y)
        plan = ExperimentPlan.model_validate({
            'subgoal': {'text': 'Find an observable control effect', 'done_when': 'A tested cell changes'},
            'question': 'Does this probe change the selected cell?',
            'hypothesis': 'The selected cell may change after one probe', 'conditions': 'Current observed state',
            'expected': {'kind': 'region_changed', 'description': 'The selected cell changes',
                         'region': {'x': x, 'y': y, 'width': 1, 'height': 1}}})
        handoff = self.notebook.handoff()
        if handoff and handoff['revision_required']:
            old = self.notebook.pages[handoff['experiment_id']]['data']
            plan.revision = DesignRevision(experiment_id=handoff['experiment_id'],
                review_revision=handoff['review_revision'],
                change='action' if data['action'] == 'ACTION6' or old['action']['action'] != action else 'observation_scope',
                reconsidered_assumption='The prior probe did not establish a reactive cell.',
                reason='The deterministic smoke probe samples a different cell for observable effects.')
        self.job = Decision(kind='act', action=data, purpose='Deterministic smoke probe', experiment=plan)
        try:
            self.validate_job(self.job)
        except ValueError:
            self.job = None
            self._stop('offline_probe_exhausted')

    def _build_graph(self):
        @node(rerun_on_resume=True)
        async def decide(ctx: Context):
            self._record('states', 'state_entered', state='DECIDE', work=self.work, input=self._context())
            try:
                self.trace.append('DECIDE')
                if self.result is not None:
                    return
                self.job = None
                if self.work == 'experiment_review':
                    reconsider = bool(self.experiments.active['data'].get('review_request'))
                    self.job = None if reconsider else self.experiments.automatic_review()
                    if self.job is None:
                        while (self.job is None and self.calls < self.max_calls and
                               self.http_requests < self.max_http_requests and self.time_left() > 0):
                            await self._ask(ctx)
                        if self.job is None:
                            self._stop('review_budget_exhausted')
                    return
                if self.obs['state']=='WIN':
                    self._stop('win')
                elif self.obs.get('evaluation_stop_reason'):
                    self._stop(self.obs['evaluation_stop_reason'])
                elif self.time_left()<=0 or self.obs['remaining_actions']<=0:
                    self._stop('budget_exhausted')
                elif self.obs['state'] in ('NOT_PLAYED','GAME_OVER'):
                    if self.memory.resets >= self.max_resets:
                        self._stop('reset_budget')
                    else:
                        self.memory.resets += 1
                        self._select({'action':'RESET','reason':'start or restart game'}, 'New episode boundary.')
                elif self._advance_skill():
                    pass
                elif not self.model:
                    self._offline_job()
                else:
                    self.job = None
                    while (self.job is None and self.calls < self.max_calls and
                           self.http_requests < self.max_http_requests and self.time_left()>0):
                        await self._ask(ctx)
                    if self.job is None:
                        self._stop('budget_exhausted' if self.time_left()<=0 else 'model_budget')

            finally:
                self._record('states', 'state_exited', state='DECIDE', work=self.work, output={
                    'job': self.job.model_dump() if self.job is not None else None,
                    'result': self.result, 'errors': self.errors})

        @node(rerun_on_resume=True)
        async def run(ctx: Context):
            self._record('states', 'state_entered', state='RUN', work=self.work, input={
                'job': self.job.model_dump() if self.job is not None else None,
                'selected_result': self.result,
                'active_skill': compact(self.memory.active_skill)})
            try:
                self.trace.append('RUN')
                if self.result is None:
                    try:
                        self.validate_job(self.job)
                        if isinstance(self.job, ExperimentReview):
                            self._finish_review(self.job, source='agent' if self.submission is self.job else 'host')
                        elif isinstance(self.job, ExperimentRedesign):
                            self.job_result = {'redesign_required': self.job.model_dump()}
                            self._record('artifacts', 'experiment_redesign_requested', state='RUN',
                                         **self.job.model_dump())
                            if self.experiments.reconsider(self.job.experiment_id):
                                self.work = 'experiment_review'
                            else:
                                self._stop('experiment_redesign_stalled')
                        elif isinstance(self.job, SkillDeferral):
                            self.job_result = {'missing_evidence': self.job.reason}
                            self.work, self.learning_request = 'experiment_design', None
                        elif isinstance(self.job, Draft):
                            self.job_result = self.library.draft(self.job)
                            self.work, self.learning_request = 'experiment_design', None
                        else:
                            job = self.job
                            if job.kind == 'learn':
                                self.work = 'skill_creation'
                                self.learning_request = {'evidence_ids': job.evidence_ids,
                                                         'purpose': job.purpose}
                            else:
                                self.work, self.learning_request = 'experiment_design', None
                            self._record('artifacts', 'decision_accepted', state='RUN', decision=job.model_dump())
                            if job.kind=='act':
                                action = validate_action(job.action, self.obs).model_dump(exclude_none=True)
                                experiment_id = self.experiments.start(job.experiment, action, self.obs)
                                self._select(action, job.experiment.expected.description, experiment_id=experiment_id)
                            elif job.kind=='stop':
                                self._stop('model_stopped')
                            elif job.kind=='learn':
                                self.job_result = {'work': self.work, **self.learning_request}
                            elif job.kind=='evaluate':
                                self.job_result = self.library.evaluate(job.skill_id)
                            else:
                                self.memory.active_skill = {'id':job.skill_id, 'arguments':job.arguments,
                                    'trial':job.kind=='trial', 'index':0, 'trace':[]}
                                self._record('learning','skill_execution_started', skill_id=job.skill_id,
                                             trial=job.kind=='trial', arguments=job.arguments)
                                self._advance_skill()
                    except ValueError as e:
                        self.errors.append(str(e)[:500])
                        self.job_result = {'error': str(e)}
                if self.result is None:
                    return Event(route='DECIDE')
                return Event(output=self.result, state={'cognition': self.memory.model_dump(mode='json')})

            finally:
                self._record('states', 'state_exited', state='RUN', work=self.work, output={
                    'result': self.result, 'tool_result': getattr(self, 'job_result', None),
                    'notebook': self.notebook.opening(),
                    'lifecycle': self.memory.lifecycle, 'errors': self.errors})

        return Workflow(name='skill_learning_loop', edges=[('START',decide), (decide,run), (run,{'DECIDE':decide})])

    async def _decide(self, observation):
        self.calls = self.http_requests = 0
        self.trace, self.errors, self.tool_executions = [], [], []
        self.job = self.result = None
        if self.memory.lifecycle in ('DONE','STOPPED'):
            return self.memory.last_result
        started = time.monotonic()
        if not self._receive(observation):
            return self.result
        if not self.initialized:
            await self.service.create_session(app_name=APP_NAME, user_id='player', session_id=self.session_id)
            self.initialized = True
        message = types.Content(role='user', parts=[types.Part(text=f'Observation {self.obs["observation_id"]}')])
        try:
            async for _ in self.runner.run_async(user_id='player', session_id=self.session_id,
                    new_message=message, run_config=RunConfig(max_llm_calls=self.max_http_requests)):
                pass
        except Exception as e:
            self.errors.append(f'{type(e).__name__}: {e}'[:500])
            self._stop('runtime_error')
        if self.result is None:
            self._stop('missing_result')
        if self.result['status']=='stop' and self.memory.active_skill:
            self._end_skill('unknown', self.result['reason'])
        self.memory.last_result = deepcopy(self.result)
        row = {'step':self.obs['step'], 'observation_id':self.obs['observation_id'],
               'frame_hash':self.obs.get('frame_hash'), 'trace':self.trace, 'action':self.result,
               'goal_ref': {'id':'goal', 'revision':self.notebook.pages['goal']['revision']},
               'outcome':self.outcome,
               'experiment_id': (self.experiments.active or {}).get('id'),
               'errors':self.errors, 'decision_seconds':time.monotonic()-started,
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
        self._record('execution', event, state='RUN', decision_id=pending['decision_id'],
                     action=pending['action'], **details)

    def close(self):
        with self.lock:
            if not self.closed:
                if self.memory.active_skill:
                    self._end_skill('unknown', 'runtime closed before trial completed')
                self.experiments.interrupt('runtime closed before experiment completed')
                self.library.save()
                self._record('states', 'runtime_closed', lifecycle=self.memory.lifecycle,
                             stop_reason=self.memory.stop_reason)
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.run_until_complete(self.loop.shutdown_default_executor())
                self.loop.close()
                self.evidence.close()
                self.closed = True
