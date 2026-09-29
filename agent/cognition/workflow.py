"""Shared observation, model transport and receipt-driven action lifecycle."""
from copy import deepcopy
import json
import os
from google.adk import Context, Workflow
from google.adk.workflow import node
from agent.controls import ACTION_TO_BUTTON
from .deliberation import DeliberationStages
from .execution import ExecutionRuntime
from .perception import measure, context_record
from . import object_memory as objects


class CognitiveRuntime(DeliberationStages, ExecutionRuntime):
    def __init__(self, game_id, model=None, *, repair_attempts=1, max_resets=2,
                 seconds=600, decision_seconds=45, log_dir=None,
                 proposal_mode=None, sam_proposer=None):
        proposal_mode=proposal_mode or os.getenv('COGNITION_PROPOSALS','sam_initial')
        if proposal_mode not in ('program','sam_initial'):
            raise ValueError('COGNITION_PROPOSALS must be program or sam_initial')
        from .hybrid_perception import HybridPerception
        self.proposal_perception=HybridPerception(sam_proposer) if proposal_mode=='sam_initial' else None
        self.perception=None
        if type(repair_attempts) is not int or repair_attempts not in (0,1):
            raise ValueError('repair_attempts must be 0 or 1')
        self.repair_attempts=repair_attempts
        self.rejection=None
        self.invocation_sequence=0
        self.planners={}
        super().__init__(game_id,model,max_resets=max_resets,seconds=seconds,
                         decision_seconds=decision_seconds,log_dir=log_dir)

    def _measure_observation(self, boundary):
        previous = deepcopy(self.memory.object_memory)
        engine = self.proposal_perception
        measure_frame = engine.measure if engine is not None else measure
        self.perception = measure_frame(self.previous.get('grid'), self.obs.get('grid'),
            before_id=self.previous.get('observation_id'), after_id=self.obs['observation_id'],
            action=(self.outcome or {}).get('action'), boundary=bool(boundary),
            **({'seconds_left': self.time_left()} if engine is not None else {}))
        self.memory.object_memory = objects.update(deepcopy(previous), self.perception)
        if self.outcome:
            self.outcome['measured_objects'] = context_record(self.perception)
            if not boundary:
                self.outcome['object_observations'] = objects.observations(previous, self.memory.object_memory,
                    self.previous.get('grid'), self.obs.get('grid'), self.outcome.get('action') or {})
        self._record('artifacts', 'objects_measured', measurement=self.perception)
        self._record('artifacts', 'object_hypotheses_updated', objects=self.memory.object_memory)

    def _snapshot(self):
        self._record('artifacts', 'cognition_updated', cognition=dict(
            phase=self.memory.phase, plan=self.memory.plan, active_skill=self.memory.active_skill,
            episode=self.memory.episode, measured_objects=context_record(self.perception, inventory=True)
            if self.perception else None, **self._snapshot_extra()))

    def _visual_parts(self, slow=False):
        if self.perception and self.perception['status']=='measured' and not self.perception['requires_review']:
            return []
        parts = [{'type':'text', 'text':'CURRENT board'}]
        if self.obs.get('image_png_base64'):
            parts.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+self.obs['image_png_base64']}})
        else:
            parts.append({'type':'text','text':json.dumps(self.obs.get('grid'))})
        return parts

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
                    self._machine_transition('direct_accepted')
            self._record('states','state_exited',state='RUN',work=self.work,output={'result':self.result})
        return Workflow(name='single_action',edges=[('START',decide),(decide,run)])
