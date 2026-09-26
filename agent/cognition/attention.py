"""Image-led attention: one judgment selects a focus and the next real action.

The host remembers receipts and pixel changes, never an inventory of semantic
objects. A no-effect trial constrains only the same action in the same state.
"""
import asyncio
import base64
import json
import os
import time

from google.adk import Context, Workflow
from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool
from google.adk.workflow import node
from google.genai import types
from pydantic import Field, model_validator

from agent.controls import ACTION_TO_BUTTON, controller_context
from agent.local_vlm import LocalVisionLlm
from .state import Contract, Action
from .validation import validate_action


class AttentionDecision(Contract):
    observation_id: str
    interpretation: str = Field(min_length=1, max_length=240,
        description='Interpret the last actual result; distinguish observation from hypothesis. Initial view: state what is uncertain.')
    focus: str = Field(min_length=1, max_length=160,
        description='One visually described object, place, or relationship to attend to. No object IDs or scene inventory.')
    action: Action
    expectation: str = Field(min_length=1, max_length=160,
        description='What this action could reveal, including remote effects or no visible effect.')
    notes: str = Field(max_length=400,
        description='Replace your brief working memory. Keep useful evidence and tentative rules, correcting old mistakes.')
    repeat_limit: int = Field(default=1, ge=1, le=3,
        description='Total unchanged-state attempts, declared BEFORE the first trial. Normally one.')
    repeat_reason: str = Field(default='', max_length=160,
        description='Only for a bounded delay, hold, or stochastic hypothesis declared before the first trial.')

    @model_validator(mode='after')
    def repeat_requires_reason(self):
        if self.repeat_limit > 1 and not self.repeat_reason.strip():
            raise ValueError('bounded repeats need a reason before the first trial')
        return self


TASKS = {'attend': ('submit_attention', AttentionDecision)}
INSTRUCTIONS = {'attend': '''Look at the game image and choose ONE informative or useful next action.
In this single answer, interpret the previous result, choose your visual focus,
and act. You may notice any object, empty place, group, or remote relationship;
do not enumerate the scene or restrict attention to previously named objects.
Use original board coordinates (x=column, y=row), not the image ruler or margins.
The host's feedback describes measured changes, NOT their meaning or causation.
Interpret actual changed positions before reusing a claim from your notes.
An edge pixel or counter changing does NOT establish that your target responded.
When repeated actions only produce unrelated changes, try a different focus or control.
No visible effect normally means shift to a different place, target, or control.
Change elsewhere means look there; progress may justify staying with a target.
Do not repeat an exhausted action in unchanged_state_trials, even with new prose.
Those trials do not prove an object useless or non-interactive in other states.
Only predeclared repeat allowances permit an unchanged-state retry. Otherwise
choose something else now. Unknown rules are a reason to experiment, not stop.
Treat your notes and prior interpretations as revisable hypotheses. Replace
notes with a short useful memory, not a transcript or a list of all objects.
Use the remaining time efficiently. Return concise fields via submit_attention.
Do not claim success unless the observed game reports it.'''}


def state_key(obs):
    return (obs.get('frame_hash'), obs.get('state'), obs.get('levels_completed', 0),
            tuple(sorted(obs.get('available_actions', []))))


def action_key(action):
    return tuple(action.get(k) for k in ('action', 'x', 'y'))


class AttentionTool(BaseTool):
    def __init__(self, runtime):
        super().__init__(name='submit_attention', description='Submit one visual judgment and next action. Call once, alone.')
        self.runtime = runtime

    def _get_declaration(self):
        schema = AttentionDecision.model_json_schema()
        schema['properties']['observation_id']['enum'] = [self.runtime.obs['observation_id']]
        action = schema['$defs']['Action']
        allowed = [ACTION_TO_BUTTON[a] for a in self.runtime.obs['available_actions']
                   if a in ACTION_TO_BUTTON and a != 'RESET']
        action['properties']['action']['enum'] = allowed
        action['properties'].pop('reason')  # focus and expectation already explain the action
        for axis, dimension in [('x', 'width'), ('y', 'height')]:
            if 'CLICK' in allowed:
                action['properties'][axis] = {'type': 'integer', 'minimum': 0,
                    'maximum': self.runtime.obs.get(dimension, 64)-1}
            else:
                action['properties'].pop(axis)
        if allowed == ['CLICK']:
            action['required'] = ['action', 'x', 'y']
        return types.FunctionDeclaration(name=self.name, description=self.description,
                                         parameters_json_schema=schema)

    async def run_async(self, *, args, tool_context):
        # End this invocation even on rejection. Any repair is a fresh, bounded
        # decision, not an opaque chain of tool-response correction requests.
        tool_context.actions.skip_summarization = True
        try:
            if self.runtime.submission is not None:
                raise ValueError('only one decision may be submitted')
            proposal = AttentionDecision.model_validate(args)
            self.runtime._machine_transition('proposed')
            self.runtime.validate_job(proposal)
        except ValueError as exc:
            self.runtime.rejection = str(exc)[:500]
            return {'accepted': False, 'error': self.runtime.rejection}
        self.runtime.submission = proposal
        return {'accepted': True}


class AttentionPolicy:
    """Uses the shared receipt/archive engine without the staged task pipeline."""
    def __init__(self, game_id, model=None, *, repair_attempts=1, max_resets=2,
                 seconds=600, log_dir=None):
        if type(repair_attempts) is not int or repair_attempts not in (0, 1):
            raise ValueError('repair_attempts must be 0 or 1')
        self.repair_attempts = repair_attempts
        self.recent_trials, self.unchanged_trials = [], []
        self.notes = ''
        self.selected_attention = None
        self.rejection = None
        super().__init__(game_id, model, max_calls=1+repair_attempts,
                         max_http_requests=1+repair_attempts, max_resets=max_resets,
                         seconds=seconds, log_dir=log_dir, learning=False)
        self.work = 'attend'

    def _receive(self, observation):
        selected = self.selected_attention
        received = super()._receive(observation)
        if not received:
            return False
        self._machine_transition('next_observation' if self.outcome else 'received')
        if self.outcome:
            outcome = self.outcome
            if outcome.get('boundary'):
                self.recent_trials.clear()
                self.unchanged_trials.clear()
                self.notes = ''
            elif selected and outcome['acknowledged']:
                changes = []
                a, b = self.previous.get('grid'), self.obs.get('grid')
                if a and b and len(a) == len(b) and all(len(x) == len(y) for x, y in zip(a, b)):
                    changes = [{'x': x, 'y': y, 'before': a[y][x], 'after': v}
                               for y, row in enumerate(b) for x, v in enumerate(row) if a[y][x] != v]
                outcome['change_bounds'] = None if not changes else {
                    'x': min(c['x'] for c in changes), 'y': min(c['y'] for c in changes),
                    'width': max(c['x'] for c in changes)-min(c['x'] for c in changes)+1,
                    'height': max(c['y'] for c in changes)-min(c['y'] for c in changes)+1}
                # Spread examples across the entire screen; don't feed only the top-left changes.
                outcome['changed_cells'] = changes if len(changes) <= 12 else [
                    changes[i*(len(changes)-1)//11] for i in range(12)]
                trial = {'step': self.previous['step'], 'focus': selected['focus'],
                    'action': outcome['action'], 'expectation': selected['expectation'],
                    'frame_changed': outcome['frame_changed'],
                    'changed_cell_count': outcome.get('changed_cell_count'),
                    'change_bounds': outcome['change_bounds']}
                self.recent_trials = (self.recent_trials + [trial])[-8:]
                if outcome['frame_changed'] is False:
                    key = state_key(self.previous)
                    entry = next((t for t in self.unchanged_trials if t['state'] == key
                                  and action_key(t['action']) == action_key(outcome['action'])), None)
                    if entry is None:
                        entry = {**trial, 'state': key, 'attempts': 0,
                                 'limit': selected['repeat_limit'], 'reason': selected['repeat_reason']}
                        self.unchanged_trials.append(entry)
                    entry['attempts'] += 1
                    self.unchanged_trials = self.unchanged_trials[-128:]
                self._record('artifacts', 'attention_feedback', trial=trial, measured=outcome)
        elif self.obs.get('full_reset'):
            self.recent_trials.clear()
            self.unchanged_trials.clear()
            self.notes = ''
        self.selected_attention = None
        self.work = 'attend'
        return True

    def _context(self):
        feedback = None if not self.outcome else {k: self.outcome.get(k) for k in (
            'action', 'acknowledged', 'boundary', 'frame_changed', 'changed_cell_count',
            'change_bounds', 'changed_cells', 'prediction')}
        trials = [{k: v for k, v in t.items() if k in ('action', 'attempts', 'limit', 'reason')} for t in self.unchanged_trials
                  if t['state'] == state_key(self.obs)]
        recent = [{k: v for k, v in t.items() if k not in ('expectation', 'focus')} for t in self.recent_trials]
        return controller_context({'work': 'attend', 'observation_id': self.obs['observation_id'],
            'available_actions': self.obs['available_actions'],
            'image_size': {'width': self.obs.get('width', 64), 'height': self.obs.get('height', 64)},
            'levels_completed': self.obs.get('levels_completed', 0),
            'remaining_actions': self.obs['remaining_actions'], 'seconds_left': round(self.time_left(), 1),
            'last_result': feedback, 'recent_trials': recent,
            'unchanged_state_trials': trials, 'working_notes_hypotheses': self.notes,
            'correction': self.rejection})

    def validate_job(self, job):
        if not isinstance(job, AttentionDecision):
            raise ValueError('expected one attention decision')
        if job.observation_id != self.obs['observation_id']:
            raise ValueError('stale observation_id')
        action = validate_action(job.action, self.obs).model_dump(exclude_none=True)
        for trial in self.unchanged_trials:
            if trial['state'] == state_key(self.obs) and action_key(trial['action']) == action_key(action):
                if trial['attempts'] >= trial['limit']:
                    raise ValueError('This action already had no visible effect in this state and its '
                                     'predeclared attempts are exhausted. Shift focus or control now.')

    def _agent(self):
        if 'attend' not in self.controllers:
            self.controllers['attend'] = LlmAgent(name='attend', include_contents='none',
                model=LocalVisionLlm(model='qwen3-vl-4b-instruct',
                    api_base=os.getenv('VLM_API_BASE', 'http://vlm:8080/v1'),
                    max_output_tokens=600, max_requests=1, completion_tools=('submit_attention',)),
                instruction=INSTRUCTIONS['attend'], tools=[AttentionTool(self)],
                before_tool_callback=self._before_tool, after_tool_callback=self._after_tool,
                on_tool_error_callback=self._tool_error)
        return self.controllers['attend']

    async def _ask(self, ctx):
        self.calls += 1
        self.memory.model_calls += 1
        self.submission = self.job = None
        context = self._context()
        parts = [types.Part(text=json.dumps(context, separators=(',', ':')))]
        views = [('CURRENT board', self.obs)]
        if self.previous and self.outcome and not self.outcome.get('boundary') and self.outcome['frame_changed']:
            views.insert(0, ('BEFORE the previous action', self.previous))
        for label, obs in views:
            parts.append(types.Part(text=label))
            if obs.get('image_png_base64'):
                parts.append(types.Part.from_bytes(data=base64.b64decode(obs['image_png_base64']), mime_type='image/png'))
            elif obs.get('grid') is not None:
                parts.append(types.Part(text=json.dumps(obs['grid'])))
        exhausted = [t['action'] for t in context['unchanged_state_trials'] if t['attempts'] >= t['limit']]
        # Put measured novelty constraints next to the requested answer, after
        # the image. Past predictions are not a template for the next action.
        guidance = {'already_tried_without_effect_do_not_repeat': exhausted}
        if self.outcome:
            guidance['last_actual_result'] = {k: self.outcome.get(k) for k in (
                'action', 'frame_changed', 'changed_cell_count', 'change_bounds', 'changed_cells')}
        if self.rejection:
            guidance['repair_this_error'] = self.rejection
        parts.append(types.Part(text='Choose your next focus from the image. '+
            json.dumps(controller_context(guidance), separators=(',', ':'))))
        agent = self._agent()
        agent.model.begin_invocation()
        agent.model.timeout_seconds = max(1, min(60, int(self.time_left())))
        agent.model._request_observer = self._request_record
        started = time.monotonic()
        record = {'state': 'DECIDE', 'work': 'attend', 'call_index': self.calls, 'context': context}
        self.rejection = None
        try:
            async with asyncio.timeout(self.time_left()):
                await ctx.run_node(agent, node_input=types.Content(role='user', parts=parts))
            if self.submission is None:
                raise ValueError(self.rejection or 'no attention decision submitted')
            self.job = self.submission
            record.update(schema_valid=True, response=self.job.model_dump_json())
        except Exception as exc:
            self.rejection = self.rejection or f'{type(exc).__name__}: {exc}'[:500]
            self.errors.append(self.rejection)
            record.update(schema_valid=False, error=self.rejection)
            self._record('artifacts', 'task_rejected', work='attend', reason=self.rejection)
        finally:
            self.http_requests += len(agent.model._exchanges)
            record.update(agent.model._last_metrics)
            record.update(http_requests=len(agent.model._exchanges), exchanges=agent.model._exchanges,
                          seconds=time.monotonic()-started)
            self._record('model', 'model_decision', **record)

    def _build_graph(self):
        @node(rerun_on_resume=True)
        async def decide(ctx: Context):
            self.trace.append('DECIDE')
            self._record('states', 'state_entered', state='DECIDE', work='attend', input=self._context())
            try:
                if self.result is not None:
                    return
                if self.obs['state'] == 'WIN':
                    self._stop('win')
                elif self.obs.get('evaluation_stop_reason'):
                    self._stop(self.obs['evaluation_stop_reason'])
                elif self.time_left() <= 0 or self.obs['remaining_actions'] <= 0:
                    self._stop('budget_exhausted')
                elif self.obs['state'] in ('NOT_PLAYED', 'GAME_OVER'):
                    if self.memory.resets >= self.max_resets:
                        self._stop('reset_budget')
                    else:
                        self.memory.resets += 1
                        self._select({'action': 'RESET', 'reason': 'start or restart game'}, '')
                elif not set(self.obs['available_actions']) & (set(ACTION_TO_BUTTON)-{'RESET'}):
                    self._stop('no_legal_action')
                elif self.model and not (self.obs.get('grid') or self.obs.get('image_png_base64')):
                    self._stop('observation_unavailable')
                elif not self.model:
                    allowed = [a for a in self.obs['available_actions'] if a != 'RESET']
                    action = {'action': allowed[self.obs['step'] % len(allowed)], 'reason': 'offline smoke probe'}
                    if action['action'] == 'ACTION6':
                        action.update(x=self.obs['step'] % self.obs.get('width', 64), y=0)
                    self._select(action, '')
                else:
                    self._machine_transition('ready')
                    self.rejection = None
                    for _ in range(1 + self.repair_attempts):
                        if self.time_left() <= 0:
                            break
                        await self._ask(ctx)
                        if self.job is not None:
                            break
                        if self.calls <= self.repair_attempts:
                            self._machine_transition('repair')
                    if self.job is None:
                        self._machine_transition('repair_exhausted')
                        self._stop('budget_exhausted' if self.time_left() <= 0 else 'attention_output_invalid')
            finally:
                self._record('states', 'state_exited', state='DECIDE', work='attend', output={
                    'job': self.job.model_dump() if self.job else None, 'result': self.result, 'errors': self.errors})

        @node(rerun_on_resume=True)
        async def run(ctx: Context):
            self.trace.append('RUN')
            self._record('states', 'state_entered', state='RUN', work='attend', input={
                'job': self.job.model_dump() if self.job else None})
            try:
                if self.result is None:
                    if self.time_left() <= 0:
                        self._stop('budget_exhausted')
                    else:
                        self.validate_job(self.job)
                        action = validate_action(self.job.action, self.obs).model_dump(exclude_none=True)
                        action['reason'] = self.job.focus
                        self.notes = self.job.notes
                        self.selected_attention = self.job.model_dump()
                        self._record('artifacts', 'attention_selected', decision=self.selected_attention)
                        self._select(action, self.job.expectation)
                        self._machine_transition('accepted')
            finally:
                self._record('states', 'state_exited', state='RUN', work='attend', output={'result': self.result})

        return Workflow(name='visual_attention', edges=[('START', decide), (decide, run)])
