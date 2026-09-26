"""ADK fast/slow workflow: deliberation builds procedures, one-token decisions execute them."""
import asyncio
import base64
from copy import deepcopy
import json
import os
import time

from google.adk import Context, Workflow
from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool
from google.adk.workflow import node
from google.genai import types

from agent.controls import ACTION_TO_BUTTON, controller_context
from agent.fast_choice import choose
from agent.local_vlm import LocalVisionLlm
from .execution import ExecutionRuntime
from .state import Deliberation
from .tasks import INSTRUCTIONS
from .validation import validate_action


class PlanTool(BaseTool):
    def __init__(self, runtime):
        super().__init__(name='submit_plan', description='Submit one grounded plan and its reusable procedures.')
        self.runtime = runtime

    def _get_declaration(self):
        schema = Deliberation.model_json_schema()
        schema['properties']['observation_id']['enum'] = [self.runtime.obs['observation_id']]
        action = schema['$defs']['Action']
        allowed = [ACTION_TO_BUTTON[a] for a in self.runtime.obs['available_actions']
                   if a in ACTION_TO_BUTTON and a != 'RESET']
        action['properties']['action']['enum'] = allowed
        action['properties'].pop('reason')
        for axis, dimension in [('x', 'width'), ('y', 'height')]:
            if 'CLICK' in allowed:
                action['properties'][axis] = {'type':'integer', 'minimum':0,
                    'maximum':self.runtime.obs.get(dimension,64)-1}
            else:
                action['properties'].pop(axis)
        if allowed == ['CLICK']:
            action['required'] = ['action','x','y']
        return types.FunctionDeclaration(name=self.name, description=self.description, parameters_json_schema=schema)

    async def run_async(self, *, args, tool_context):
        tool_context.actions.skip_summarization = True
        try:
            if self.runtime.submission is not None:
                raise ValueError('submit exactly one plan')
            proposal = Deliberation.model_validate(args)
            self.runtime.validate_plan(proposal)
        except ValueError as exc:
            self.runtime.rejection = str(exc)[:500]
            return {'accepted':False, 'error':self.runtime.rejection}
        self.runtime.submission = proposal
        return {'accepted':True}


class CognitiveRuntime(ExecutionRuntime):
    def __init__(self, game_id, model=None, *, repair_attempts=1, max_resets=2,
                 seconds=600, decision_seconds=45, log_dir=None):
        if type(repair_attempts) is not int or repair_attempts not in (0,1):
            raise ValueError('repair_attempts must be 0 or 1')
        self.repair_attempts = repair_attempts
        self.recent_trials, self.routing_history = [], []
        self.completion_history = []
        self.completed = set()
        self.rejection = None
        self.replan_reason = 'Initial board: form a useful goal or informative probe.'
        self.plan_anchor = None
        self.planner = None
        self.choice_model = LocalVisionLlm(model='qwen3-vl-4b-instruct',
            api_base=os.getenv('VLM_API_BASE','http://vlm:8080/v1'), max_output_tokens=1)
        super().__init__(game_id, model, max_resets=max_resets, seconds=seconds,
                         decision_seconds=decision_seconds, log_dir=log_dir)

    def _snapshot(self):
        self._record('artifacts','cognition_updated', cognition={
            'plan':self.memory.plan, 'causal_notes':self.memory.causal_notes, 'skills':self.memory.skills,
            'active_skill':self.memory.active_skill, 'completed':sorted(self.completed),
            'replan_reason':self.replan_reason, 'recent_trials':self.recent_trials,
            'routing_history':self.routing_history, 'completion_history':self.completion_history})

    def _on_observation(self, boundary):
        if self.outcome:
            trial = {'observation_id':self.obs['observation_id'], 'step':self.obs['step'], **self.outcome}
            self._record('artifacts','action_feedback', trial=trial)
            if self.outcome['acknowledged'] and not boundary:
                self.recent_trials = (self.recent_trials+[trial])[-8:]
        if boundary:
            self.memory.plan = self.memory.active_skill = None
            self.plan_anchor = None
            self.completed.clear()
            self.recent_trials.clear()
            self.routing_history.clear()
            self.completion_history.clear()
            self.replan_reason = f'New {boundary} boundary; reinterpret the board and re-ground retained procedures.'
        self._machine_transition('next_observation' if self.outcome else 'received')
        self._snapshot()

    def validate_plan(self, plan):
        if plan.observation_id != self.obs['observation_id']:
            raise ValueError('stale observation_id')
        names = [s.name for s in plan.skills]
        if len(set(names)) != len(names) or len(set(plan.candidates)) != len(plan.candidates):
            raise ValueError('procedure names and candidates must be unique')
        skills = {**self.memory.skills, **{s.name:s.model_dump() for s in plan.skills}}
        if any(name not in skills for name in plan.candidates):
            raise ValueError('candidate procedure does not exist')
        for name in set(names) | set(plan.candidates):
            for step in skills[name]['steps']:
                for option in step['options']:
                    validate_action(option['action'], self.obs)

    def _accept_plan(self, plan):
        self.validate_plan(plan)
        for skill in plan.skills:
            self.memory.skills.pop(skill.name, None)
            self.memory.skills[skill.name] = skill.model_dump()
        while len(self.memory.skills)>16:
            victim = next(k for k in self.memory.skills if k not in plan.candidates)
            self.memory.skills.pop(victim)
        self.memory.plan = plan.model_dump(exclude={'skills'})
        self.memory.causal_notes = plan.causal_notes
        self.memory.active_skill = None
        self.completed.clear()
        self.replan_reason = ''
        self.plan_anchor = deepcopy(self.obs)
        self._record('artifacts','plan_created', plan=plan.model_dump())
        self._machine_transition('planned')
        self._snapshot()

    def _reconsider(self, reason, source):
        self.replan_reason = reason
        event = {'step':self.obs['step'], 'source':source, 'reason':reason,
                 'skill':deepcopy(self.memory.active_skill)}
        self.routing_history = (self.routing_history+[event])[-6:]
        self._record('artifacts','reconsider_requested', **event)
        self.memory.active_skill = None
        self._machine_transition('skill_reconsider' if source=='choose_skill' else 'step_reconsider')
        self._snapshot()

    def _retained_skills(self):
        # Bound the slow input independently of the library's storage capacity.
        retained, size = {}, 0
        for name, skill in reversed(list(self.memory.skills.items())):
            length = len(json.dumps(skill,ensure_ascii=False))
            if len(retained)<4 and size+length<=10000:
                retained[name] = skill
                size += length
        return retained

    def _context(self, work=None):
        work = work or self.work
        common = {'work':work, 'observation_id':self.obs['observation_id'],
                  'available_actions':self.obs['available_actions'],
                  'remaining_actions':self.obs['remaining_actions'], 'seconds_left':round(self.time_left(),2)}
        if work == 'deliberate':
            common.update(image_size={'width':self.obs.get('width',64),'height':self.obs.get('height',64)},
                last_result=self.outcome, recent_trials=self.recent_trials,
                previous_plan=self.memory.plan, retained_skills=self._retained_skills(),
                causal_notes_hypotheses=self.memory.causal_notes,
                completed_candidates=sorted(self.completed), completion_judgments=self.completion_history,
                return_to_deliberation=self.replan_reason, routing_history=self.routing_history,
                correction=self.rejection)
        else:
            common.pop('available_actions')
            common.update(goal=self.memory.plan['goal'], last_result=self.outcome)
            if work == 'execute_step':
                active = self.memory.active_skill
                skill = self.memory.skills[active['name']]
                common.update(active_skill=active, procedure_effect=skill['effect'],
                              current_step={k:v for k,v in skill['steps'][active['index']].items() if k!='options'})
        return controller_context(common)

    def _views(self, slow=False):
        views = []
        if slow and self.plan_anchor and self.plan_anchor['observation_id'] not in (
                self.obs['observation_id'], self.previous.get('observation_id')):
            views.append(('AT THE LAST PLAN', self.plan_anchor))
        if self.previous and self.outcome and not self.outcome.get('boundary') and self.outcome['frame_changed']:
            views.append(('BEFORE the last action', self.previous))
        views.append(('CURRENT board',self.obs))
        return views

    def _visual_parts(self, slow=False):
        parts = []
        for label, obs in self._views(slow):
            parts.append({'type':'text', 'text':label})
            if obs.get('image_png_base64'):
                parts.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+obs['image_png_base64']}})
            else:
                parts.append({'type':'text','text':json.dumps(obs.get('grid'))})
        return parts

    def _agent(self):
        if self.planner is None:
            self.planner = LlmAgent(name='deliberate', include_contents='none',
                model=LocalVisionLlm(model='qwen3-vl-4b-instruct',
                    api_base=os.getenv('VLM_API_BASE','http://vlm:8080/v1'),
                    max_output_tokens=2200, max_requests=1, completion_tools=('submit_plan',)),
                instruction=INSTRUCTIONS['deliberate'], tools=[PlanTool(self)],
                before_tool_callback=self._before_tool, after_tool_callback=self._after_tool,
                on_tool_error_callback=self._tool_error)
        return self.planner

    async def _deliberate(self, ctx):
        self.work = 'deliberate'
        self.rejection = None
        for attempt in range(1+self.repair_attempts):
            if self.time_left()<=0:
                return
            self.calls += 1
            self.memory.model_calls += 1
            self.submission = None
            context = self._context()
            self._record('states','state_entered',state='DECIDE',work=self.work,input=context)
            parts = []
            for part in self._visual_parts(slow=True):
                if part['type']=='text':
                    parts.append(types.Part(text=part['text']))
                else:
                    parts.append(types.Part.from_bytes(data=base64.b64decode(part['image_url']['url'].split(',',1)[1]),mime_type='image/png'))
            parts.append(types.Part(text=json.dumps(context,separators=(',',':'))))
            agent = self._agent()
            agent.model.begin_invocation()
            agent.model.timeout_seconds = max(.001, self.time_left())
            agent.model._request_observer = self._request_record
            started = time.monotonic()
            record = {'state':'DECIDE','work':self.work,'call_index':self.calls,'attempt':attempt,'context':context}
            self.rejection = None
            try:
                async with asyncio.timeout(self.time_left()):
                    await ctx.run_node(agent,node_input=types.Content(role='user',parts=parts))
                if self.submission is None:
                    raise ValueError(self.rejection or 'no plan submitted')
                if self.time_left()<=0:
                    return
                self._accept_plan(self.submission)
                record.update(schema_valid=True,response=self.submission.model_dump_json())
            except Exception as exc:
                self.rejection = self.rejection or f'{type(exc).__name__}: {exc}'[:500]
                self.errors.append(self.rejection)
                record.update(schema_valid=False,error=self.rejection)
                self._record('artifacts','task_rejected',work=self.work,reason=self.rejection)
            finally:
                self.http_requests += len(agent.model._exchanges)
                record.update(agent.model._last_metrics)
                record.update(http_requests=len(agent.model._exchanges),exchanges=agent.model._exchanges,
                              seconds=time.monotonic()-started)
                self._record('model','model_decision',**record)
                self._record('states','state_exited',state='DECIDE',work=self.work,output=record)
            if record.get('schema_valid'):
                return
            if attempt<self.repair_attempts:
                self._machine_transition('repair')
        if self.time_left()>0:
            self._machine_transition('repair_exhausted')
            self._stop('plan_output_invalid')

    def _skill_options(self):
        names = [n for n in self.memory.plan['candidates'] if n not in self.completed]
        return {str(i+1):{'kind':'skill','name':name,
                'when_to_use':self.memory.skills[name]['when_to_use'],
                'effect':self.memory.skills[name]['effect']} for i,name in enumerate(names)}

    async def _fast(self, work, options):
        self.work = work
        options = {**options, '8':{'kind':'reconsider','meaning':'Unexpected result, cannot judge, or need a different plan'}}
        context = self._context(work)
        context['choices'] = controller_context(options)
        context['question'] = ('Which numbered procedure fits the board and goal?' if work=='choose_skill'
                               else 'Which numbered option follows from the current step and observed result?')
        self.calls += 1
        self.memory.model_calls += 1
        self._record('states','state_entered',state='DECIDE',work=work,input=context)
        parts = self._visual_parts()+[{'type':'text','text':json.dumps(context,separators=(',',':'))}]
        record = await choose(self.choice_model, instruction=INSTRUCTIONS[work], parts=parts,
                              options=options, observe_request=self._request_record, timeout=self.time_left())
        self.http_requests += record['http_requests']
        self._record('model','model_decision',state='DECIDE',work=work,call_index=self.calls,context=context,**record)
        self._record('states','state_exited',state='DECIDE',work=work,output=record)
        if not record['schema_valid']:
            self.errors.append(record['error'])
            self._reconsider('Fast selection could not be read: '+record['error'],work)
            return None
        option = options[record['label']]
        self._record('artifacts','fast_selected',work=work,label=record['label'],selection=option)
        if option['kind']=='reconsider':
            self._reconsider(
                'Fast selector chose 8 (reconsider): none of the candidate procedures fits; reconsider their applicability. No action was sent.'
                if work=='choose_skill' else
                'Fast executor chose 8 (reconsider): interpret the current result against the expected effect. No action was sent.',work)
            return None
        return option

    async def _think(self, ctx):
        while self.result is None and self.job is None and self.time_left()>0:
            if self.memory.plan is None or self.replan_reason:
                self._machine_transition('need_plan')
                await self._deliberate(ctx)
                if self.result is not None or self.time_left()<=0:
                    break
            if self.memory.active_skill is None:
                options = self._skill_options()
                if not options:
                    self.replan_reason = 'All candidate procedures were judged complete; choose the next goal from the actual board.'
                    continue
                self._machine_transition('choose')
                choice = await self._fast('choose_skill',options)
                if choice is None:
                    continue
                self.memory.active_skill = {'name':choice['name'],'index':0}
                self._record('artifacts','skill_started',skill=self.memory.active_skill)
                self._snapshot()
            active = self.memory.active_skill
            step = self.memory.skills[active['name']]['steps'][active['index']]
            options = {str(i+1):{'kind':'action', **option} for i,option in enumerate(step['options'])}
            try:
                for option in options.values():
                    validate_action(option['action'],self.obs)
            except ValueError as exc:
                self._reconsider('The controls or coordinates need grounding again: '+str(exc),'execute_step')
                continue
            options['7'] = {'kind':'advance','meaning':'The current step done_when is observed; advance without an action'}
            self._machine_transition('execute')
            choice = await self._fast('execute_step',options)
            if choice is None:
                continue
            if choice['kind']=='action':
                self.job = choice
                break
            completion = {'observation_id':self.obs['observation_id'],
                          'skill':deepcopy(active), 'done_when':step['done_when'],
                          'source':'fast_model_judgment'}
            self.completion_history = (self.completion_history+[completion])[-6:]
            self._record('artifacts','step_completed',**completion)
            active['index'] += 1
            if active['index'] == len(self.memory.skills[active['name']]['steps']):
                self.completed.add(active['name'])
                self._record('artifacts','skill_completed',skill=active,source='fast_model_judgment')
                self.memory.active_skill = None
                self._machine_transition('procedure_done')
            else:
                self._machine_transition('step_done')
            self._snapshot()
        if self.result is None and self.time_left()<=0:
            self._stop('budget_exhausted' if time.monotonic()>=self.deadline else 'decision_time_exhausted')

    def _build_graph(self):
        @node(rerun_on_resume=True)
        async def decide(ctx: Context):
            self.trace.append('DECIDE')
            if self.result is not None:
                return
            if self.obs['state']=='WIN':
                self._stop('win')
            elif self.obs.get('evaluation_stop_reason'):
                self._stop(self.obs['evaluation_stop_reason'])
            elif self.time_left()<=0 or self.obs['remaining_actions']<=0:
                self._stop('budget_exhausted')
            elif self.obs['state'] in ('NOT_PLAYED','GAME_OVER'):
                if self.memory.resets>=self.max_resets:
                    self._stop('reset_budget')
                else:
                    self.memory.resets += 1
                    self._select({'action':'RESET','reason':'start or restart game'},'')
            elif not set(self.obs['available_actions']) & (set(ACTION_TO_BUTTON)-{'RESET'}):
                self._stop('no_legal_action')
            elif self.model and not (self.obs.get('grid') or self.obs.get('image_png_base64')):
                self._stop('observation_unavailable')
            elif not self.model:
                allowed = [a for a in self.obs['available_actions'] if a!='RESET']
                action = {'action':allowed[self.obs['step']%len(allowed)],'reason':'offline smoke probe'}
                if action['action']=='ACTION6':
                    action.update(x=self.obs['step']%self.obs.get('width',64),y=0)
                self._select(action,'')
            else:
                await self._think(ctx)

        @node(rerun_on_resume=True)
        async def run(ctx: Context):
            self.trace.append('RUN')
            self._record('states','state_entered',state='RUN',work=self.work,input={'job':self.job})
            if self.result is None:
                if self.time_left()<=0:
                    self._stop('budget_exhausted')
                else:
                    self._select(self.job['action'],self.job['expected_effect'])
                    self._machine_transition('accepted')
            self._record('states','state_exited',state='RUN',work=self.work,output={'result':self.result})
        return Workflow(name='fast_slow',edges=[('START',decide),(decide,run)])
