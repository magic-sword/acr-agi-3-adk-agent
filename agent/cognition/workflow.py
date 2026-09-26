"""Persistent goal-directed deliberation with single-token procedure execution."""
from copy import deepcopy
import json
import os
import time

from google.adk import Context, Workflow
from google.adk.workflow import node

from agent.controls import ACTION_TO_BUTTON, controller_context
from agent.fast_choice import choose
from agent.local_vlm import LocalVisionLlm
from .deliberation import DeliberationStages
from .execution import ExecutionRuntime
from .tasks import INSTRUCTIONS, SLOW_TASKS
from .validation import validate_action


class CognitiveRuntime(DeliberationStages, ExecutionRuntime):
    def __init__(self, game_id, model=None, *, repair_attempts=1, max_resets=2,
                 seconds=600, decision_seconds=45, log_dir=None):
        if type(repair_attempts) is not int or repair_attempts not in (0,1):
            raise ValueError('repair_attempts must be 0 or 1')
        self.repair_attempts=repair_attempts
        self.recent_trials=[]
        self.replan_reason='Initial board: identify a useful small goal or a specific uncertainty.'
        self.rejection=None
        self.plan_anchor=None
        self.plan_sequence=self.invocation_sequence=0
        self.planners={}
        self.choice_model=LocalVisionLlm(model='qwen3-vl-4b-instruct',
            api_base=os.getenv('VLM_API_BASE','http://vlm:8080/v1'),max_output_tokens=1)
        super().__init__(game_id,model,max_resets=max_resets,seconds=seconds,
                         decision_seconds=decision_seconds,log_dir=log_dir)

    def _snapshot(self):
        keys=('phase','understanding','targets','goals','goal_status','selected_goal_id',
              'backchain','plan','causal_notes','skills','active_skill','review','reconciliations')
        self._record('artifacts','cognition_updated',cognition={
            **{k:getattr(self.memory,k) for k in keys},'recent_trials':self.recent_trials,
            'replan_reason':self.replan_reason})

    def _on_observation(self, boundary):
        m=self.memory
        if self.outcome:
            trial={'observation_id':self.obs['observation_id'],'step':self.obs['step'],**self.outcome}
            self._record('artifacts','action_feedback',trial=trial)
            if self.outcome['acknowledged'] and not boundary:
                self.recent_trials=(self.recent_trials+[trial])[-8:]
                if self._current_result() is not None:
                    m.active_skill['action_count']+=1
                    m.active_skill['step_action_count']+=1
        if boundary:
            m.understanding=m.backchain=m.plan=m.active_skill=m.review=None
            m.targets.clear();m.goals.clear();m.goal_status.clear()
            m.selected_goal_id=None
            m.reconciliations.clear()
            m.phase='understand'
            self.plan_anchor=None
            self.recent_trials.clear()
            self.replan_reason=f'New {boundary} boundary; re-ground retained causal knowledge and procedures.'
        self._machine_transition('next_observation' if self.outcome else 'received')
        if (not boundary and m.plan and m.plan['intent']=='probe' and self._current_result()
                and self.outcome['acknowledged']):
            self._queue_review('probe_result','One probe action was acknowledged; interpret its observed result.',
                               'probe_observed')
        elif m.phase=='execute_step':
            self._machine_transition('continue_execution')
        self._snapshot()

    def _current_result(self):
        active=self.memory.active_skill
        previous=(self.outcome or {}).get('skill') or {}
        if (active and previous.get('invocation_id')==active['invocation_id']
                and previous.get('index')==active['index']):
            return self.outcome
        return None

    def _queue_review(self, trigger, reason, transition):
        m=self.memory
        m.review={'trigger':trigger,'reason':reason,'observation_id':self.obs['observation_id'],
                  'invocation':deepcopy(m.active_skill)}
        self.replan_reason=reason
        if trigger=='completion_candidate' and m.plan['intent']=='achieve':
            m.goal_status[m.plan['goal_id']]={'status':'candidate','observation_id':self.obs['observation_id'],
                                             'evidence':'Fast model claim; awaiting reconciliation.'}
        m.phase='reconcile'
        self._record('artifacts','reconciliation_requested',**m.review)
        self._machine_transition(transition)
        self._snapshot()

    def _retained_skills(self):
        retained={};size=0
        for name,skill in reversed(list(self.memory.skills.items())):
            length=len(json.dumps(skill,ensure_ascii=False))
            if len(retained)<4 and size+length<=10000:
                retained[name]=skill;size+=length
        return retained

    def _goal_context(self):
        m=self.memory
        # Retain the selected branch and its prerequisites; other completed branches stay in memory.
        needed=set()
        def add(key):
            if key is None or key in needed or key not in m.goals:return
            needed.add(key)
            g=m.goals[key];add(g['parent_id'])
            for ref in g['requires']:add(ref)
        add(m.selected_goal_id)
        return {key:{**m.goals[key],'assessment':m.goal_status.get(key)} for key in m.goals if key in needed}

    def _context(self, work=None):
        work=work or self.work
        m=self.memory
        goal=m.goals.get(m.selected_goal_id)
        common={'work':work,'observation_id':self.obs['observation_id'],
                'remaining_actions':self.obs['remaining_actions'],'seconds_left':round(self.time_left(),2)}
        if work in SLOW_TASKS:
            common.update(available_actions=self.obs['available_actions'],
                image_size={'width':self.obs.get('width',64),'height':self.obs.get('height',64)},
                current_goal=goal,goal_status=m.goal_status.get(m.selected_goal_id),
                causal_notes=m.causal_notes,last_result=self.outcome,correction=self.rejection,
                reason=self.replan_reason)
            if work=='understand':
                common.update(previous_understanding=m.understanding,recent_trials=self.recent_trials[-4:])
            elif work=='backchain':
                common.update(understanding=m.understanding,goals=self._goal_context(),
                              last_reconciliation=m.reconciliations[-1:] )
            elif work=='ground':
                common.update(understanding=m.understanding,goals=self._goal_context(),
                    previous_plan=m.plan,retained_skills=self._retained_skills(),
                    last_reconciliation=m.reconciliations[-1:])
            else:
                active=m.active_skill
                common.update(plan=m.plan,review=m.review,active_skill=active,
                    procedure=m.skills[active['name']] if active else None,
                    current_invocation_result=self._current_result(),
                    attempt_results=[t for t in self.recent_trials if active and
                        (t.get('skill') or {}).get('invocation_id')==active['invocation_id']],
                    prior_reconciliations=m.reconciliations[-2:])
            ids=set((m.plan or {}).get('target_ids',[])) if work=='reconcile' else set()
            if work in ('backchain','ground'):
                ids.update(t['id'] for t in (m.understanding or {}).get('targets',[]))
                for g in self._goal_context().values():ids.update(g['target_ids'])
            if ids:common['targets']={k:m.targets[k] for k in ids if k in m.targets}
        else:
            common.update(goal=m.goals.get(m.plan['goal_id']),intent=m.plan['intent'],
                question_to_test=m.plan['question'],baseline=m.plan['baseline'],
                baseline_observation_id=m.plan['observation_id'],
                targets={k:m.targets[k] for k in m.plan['target_ids']},
                last_result=self._current_result())
            if work=='execute_step':
                active=m.active_skill;skill=m.skills[active['name']]
                common.update(active_skill=active,procedure_effect=skill['effect'],
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

    def _skill_options(self):
        return {str(i+1):{'kind':'skill','name':name,
                'when_to_use':self.memory.skills[name]['when_to_use'],
                'effect':self.memory.skills[name]['effect']}
                for i,name in enumerate(self.memory.plan['candidates'])}

    async def _fast(self, work, options):
        self.work=work
        options={**options,'8':{'kind':'reconsider','meaning':'Unexpected result or uncertainty: reconcile'}}
        context=self._context(work)
        context['choices']=controller_context(options)
        self.calls+=1;self.memory.model_calls+=1
        self._record('states','state_entered',state='DECIDE',work=work,input=context)
        parts=self._visual_parts()+[{'type':'text','text':json.dumps(context,separators=(',',':'))}]
        record=await choose(self.choice_model,instruction=INSTRUCTIONS[work],parts=parts,
                            options=options,observe_request=self._request_record,timeout=self.time_left())
        self.http_requests+=record['http_requests']
        self._record('model','model_decision',state='DECIDE',work=work,call_index=self.calls,context=context,**record)
        self._record('states','state_exited',state='DECIDE',work=work,output=record)
        transition='skill_reconsider' if work=='choose_skill' else 'step_reconsider'
        if not record['schema_valid']:
            self.errors.append(record['error'])
            self._queue_review('invalid_fast_output','Fast selection could not be read: '+record['error'],transition)
            return None
        option=options[record['label']]
        self._record('artifacts','fast_selected',work=work,label=record['label'],selection=option)
        if option['kind']=='reconsider':
            self._queue_review('unexpected','Fast model chose 8: compare the plan and actual evidence.',transition)
            return None
        return option

    async def _think(self, ctx):
        m=self.memory
        while self.result is None and self.job is None and self.time_left()>0:
            if m.phase in SLOW_TASKS:
                await self._deliberate(ctx,m.phase)
                continue
            if m.phase=='choose_skill':
                choice=await self._fast('choose_skill',self._skill_options())
                if choice is None:continue
                self.invocation_sequence+=1
                m.active_skill={'name':choice['name'],'index':0,
                    'invocation_id':f'{m.run_id}:attempt:{self.invocation_sequence}',
                    'started_at':self.obs['observation_id'],'step_started_at':self.obs['observation_id'],
                    'action_count':0,'step_action_count':0}
                m.phase='execute_step'
                self._record('artifacts','skill_started',skill=m.active_skill)
                self._machine_transition('execute')
                self._snapshot()
            active=m.active_skill
            step=m.skills[active['name']]['steps'][active['index']]
            options={str(i+1):{'kind':'action',**option} for i,option in enumerate(step['options'])}
            try:
                for option in options.values():validate_action(option['action'],self.obs)
            except ValueError as exc:
                self._queue_review('controls_changed','Re-ground controls or coordinates: '+str(exc),'step_reconsider')
                continue
            # A one-action probe finishes on its acknowledged observation, not on
            # a visual completion guess before it has been executed.
            if m.plan['intent']=='achieve':
                options['7']={'kind':'advance','meaning':'The specified done_when relation is visible'}
            choice=await self._fast('execute_step',options)
            if choice is None:continue
            if choice['kind']=='action':
                self.job=choice
                break
            completion={'skill':deepcopy(active),'done_when':step['done_when'],'source':'fast_model_judgment'}
            self._record('artifacts','completion_candidate',**completion)
            if active['index']+1==len(m.skills[active['name']]['steps']):
                self._queue_review('completion_candidate','Assess the fast completion claim against the target relation.',
                                   'procedure_done')
            else:
                active['index']+=1
                active['step_action_count']=0
                active['step_started_at']=self.obs['observation_id']
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
