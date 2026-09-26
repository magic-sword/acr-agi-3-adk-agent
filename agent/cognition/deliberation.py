"""ADK stage calls and atomic updates to persistent goal/evidence memory."""
import asyncio
import base64
from copy import deepcopy
import json
import os
import time

from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool
from google.genai import types

from agent.controls import ACTION_TO_BUTTON
from agent.local_vlm import LocalVisionLlm
from .tasks import TASKS, INSTRUCTIONS, OUTPUT_TOKENS
from .validation import validate_intent


class StageTool(BaseTool):
    def __init__(self, runtime, work):
        name, self.contract = TASKS[work]
        super().__init__(name=name, description=f'Submit the {work} result for the current observation.')
        self.runtime, self.work = runtime, work

    def _get_declaration(self):
        schema = self.contract.model_json_schema()
        schema['properties']['observation_id']['enum'] = [self.runtime.obs['observation_id']]
        definitions = schema.get('$defs', {})
        # Expose the IDs and routes that this stage can actually use, rather than
        # asking the model to discover structural restrictions through rejections.
        m = self.runtime.memory
        if 'GroundedPlan' in definitions:
            p = definitions['GroundedPlan']['properties']
            p['goal_id']['enum'] = [None] if m.selected_goal_id is None else [None, m.selected_goal_id]
            if m.skills:
                p['reuse']['items']['enum'] = list(self.runtime._retained_skills())
            else:
                p['reuse']['maxItems'] = 0
                p['skills']['minItems'] = 1
            if m.selected_goal_id is None:
                p['intent']['enum'] = ['probe']
        if self.work == 'reconcile':
            schema['properties']['goal_id']['enum'] = [m.plan['goal_id']]
            if m.plan['intent'] == 'probe' or m.plan['goal_id'] is None:
                schema['properties']['goal_status']['enum'] = ['active', 'unknown']
            if m.plan['intent'] == 'probe' or m.active_skill is None:
                schema['properties']['next']['enum'] = ['understand', 'backchain', 'ground']
        if 'ActionIntent' in definitions:
            action = definitions['ActionIntent']
            allowed = [ACTION_TO_BUTTON[a] for a in self.runtime.obs['available_actions']
                       if a in ACTION_TO_BUTTON and a != 'RESET']
            action['properties']['action']['enum'] = allowed
            if 'CLICK' not in allowed:
                action['properties'].pop('target_query')
            elif allowed == ['CLICK']:
                action['properties']['target_query'] = {'type':'string', 'minLength':1, 'maxLength':300,
                    'description':'Visible instance and part to click, described by appearance and relations; no coordinates.'}
                action['required'] = ['action', 'target_query']
        return types.FunctionDeclaration(name=self.name, description=self.description, parameters_json_schema=schema)

    async def run_async(self, *, args, tool_context):
        tool_context.actions.skip_summarization = True
        try:
            if self.runtime.submission is not None:
                raise ValueError('submit exactly one stage result')
            proposal = self.contract.model_validate(args)
            self.runtime.validate_stage(self.work, proposal)
        except ValueError as exc:
            self.runtime.rejection = str(exc)[:500]
            return {'accepted':False, 'error':self.runtime.rejection}
        self.runtime.submission = proposal
        return {'accepted':True}


def unique(values, name):
    if len(values) != len(set(values)):
        raise ValueError(f'duplicate {name}')


def acyclic(goals, field):
    visiting, visited = set(), set()
    def visit(key):
        if key in visiting:
            raise ValueError(f'cyclic goal {field}')
        if key in visited:
            return
        visiting.add(key)
        links = goals[key][field]
        links = [links] if field=='parent_id' and links else [] if field=='parent_id' else links
        for other in links:
            if other not in goals:
                raise ValueError('unknown goal reference: '+other)
            visit(other)
        visiting.remove(key)
        visited.add(key)
    for key in goals:
        visit(key)


class DeliberationStages:
    def validate_stage(self, work, value):
        if value.observation_id != self.obs['observation_id']:
            raise ValueError('stale observation_id')
        m = self.memory
        if work=='understand':
            unique([c.name for c in value.concepts], 'concept names')
            names = {c.name for c in value.concepts}
            if any(t.concept not in names for t in value.targets):
                raise ValueError('target concept must be described in concepts')
        elif work=='backchain':
            unique([g.id for g in value.goals],'goal IDs')
            merged = {**m.goals, **{g.id:g.model_dump() for g in value.goals}}
            if value.selected_goal_id not in merged:
                raise ValueError('unknown selected goal')
            for g in value.goals:
                unique(g.requires,'prerequisites')
            for field in ('parent_id','requires'):
                acyclic(merged,field)
        elif work=='ground':
            if value.next!='execute':
                if not value.reason:
                    raise ValueError('rerouting requires a reason')
                return
            p = value.plan
            if p is None:
                raise ValueError('execution requires a grounded plan')
            if p.goal_id is not None and p.goal_id!=m.selected_goal_id:
                raise ValueError('ground the selected goal; use backchain to select another')
            if p.intent=='achieve' and (p.goal_id is None or p.goal_id not in m.goals):
                raise ValueError('achievement requires a selected goal')
            if p.intent=='probe' and not p.question:
                raise ValueError('probe requires a question')
            unique([s.name for s in p.skills],'procedure names')
            unique(p.reuse,'reuse names')
            if not p.skills and not p.reuse:
                raise ValueError('provide a new skill or reuse an existing skill')
            if any(n not in m.skills for n in p.reuse):
                raise ValueError('reused procedure does not exist')
            skills = {**m.skills, **{s.name:s.model_dump() for s in p.skills}}
            for name in set(p.reuse) | {s.name for s in p.skills}:
                if p.intent=='probe' and len(skills[name]['steps'])!=1:
                    raise ValueError('a probe has one action step, then observation and reconciliation')
                for step in skills[name]['steps']:
                    for option in step['options']:
                        validate_intent(option['action'],self.obs)
        elif work=='reconcile':
            if m.review is None or m.plan is None or value.goal_id!=m.plan['goal_id']:
                raise ValueError('reconcile the reviewed goal')
            if value.goal_status=='confirmed' and (value.goal_id is None or m.plan['intent']=='probe'):
                raise ValueError('probe completion does not confirm a goal')
            if value.next=='resume':
                if m.plan['intent']!='achieve' or m.active_skill is None or value.goal_status=='confirmed':
                    raise ValueError('resume requires an unfinished achievement procedure')
                active=m.active_skill
                for option in m.skills[active['name']]['steps'][active['index']]['options']:
                    validate_intent(option['action'],self.obs)

    def _accept_stage(self, work, value):
        self.validate_stage(work,value)
        m=self.memory
        data=value.model_dump()
        if work=='understand':
            m.understanding=data
            for concept in value.concepts:
                m.concepts.pop(concept.name, None)
                m.concepts[concept.name]={**concept.model_dump(),'observed_at':value.observation_id}
            while len(m.concepts)>16:
                m.concepts.pop(next(iter(m.concepts)))
            m.causal_notes=value.causal_hypotheses
            m.active_skill=None
            m.phase=value.next
            self._machine_transition('understood' if value.next=='backchain' else 'direct_ground')
        elif work=='backchain':
            for goal in value.goals:
                old=m.goals.get(goal.id)
                updated=goal.model_dump()
                if old!=updated:
                    m.goal_status[goal.id]={'status':'active','observation_id':value.observation_id,
                                            'evidence':'New or revised goal definition.'}
                m.goals[goal.id]=updated
            m.backchain=data
            m.selected_goal_id=value.selected_goal_id
            m.phase='ground'
            self._machine_transition('backchained')
        elif work=='ground':
            if value.next!='execute':
                m.phase=value.next
                self.replan_reason=value.reason
                self._machine_transition('ground_'+value.next)
            else:
                p=value.plan
                for skill in p.skills:
                    m.skills.pop(skill.name,None)
                    m.skills[skill.name]=skill.model_dump()
                candidates=list(dict.fromkeys([s.name for s in p.skills]+p.reuse))
                while len(m.skills)>16:
                    m.skills.pop(next(n for n in m.skills if n not in candidates))
                self.plan_sequence+=1
                m.plan={**p.model_dump(exclude={'skills','reuse'}),'candidates':candidates,
                        'observation_id':value.observation_id,
                        'plan_id':f'{m.run_id}:plan:{self.plan_sequence}'}
                m.active_skill=None
                m.review=None
                m.phase='choose_skill'
                self.replan_reason=''
                self.plan_anchor=deepcopy(self.obs)
                self._record('artifacts','plan_created',plan=m.plan,skills={n:m.skills[n] for n in candidates})
                self._machine_transition('grounded')
        elif work=='reconcile':
            m.reconciliations=(m.reconciliations+[{
                **data,'trigger':m.review,'invocation':deepcopy(m.active_skill)}])[-6:]
            m.causal_notes=value.causal_notes
            if value.goal_id is not None:
                m.goal_status[value.goal_id]={'status':value.goal_status,'observation_id':value.observation_id,
                                              'evidence':value.evidence}
            m.review=None
            if value.next=='resume':
                m.phase='execute_step'
            else:
                m.active_skill=None
                m.phase=value.next
            self.replan_reason=value.reason
            self._machine_transition('review_'+value.next)
        self._record('artifacts','stage_accepted',work=work,result=data)
        self._snapshot()

    def _agent(self, work):
        if work not in self.planners:
            tool_name=TASKS[work][0]
            self.planners[work]=LlmAgent(name=work,include_contents='none',
                model=LocalVisionLlm(model='qwen3-vl-4b-instruct',
                    api_base=os.getenv('VLM_API_BASE','http://vlm:8080/v1'),
                    max_output_tokens=OUTPUT_TOKENS[work],max_requests=1,completion_tools=(tool_name,)),
                instruction=INSTRUCTIONS[work],tools=[StageTool(self,work)],
                before_tool_callback=self._before_tool,after_tool_callback=self._after_tool,
                on_tool_error_callback=self._tool_error)
        return self.planners[work]

    async def _deliberate(self, ctx, work):
        self.work=work
        self.rejection=None
        for attempt in range(1+self.repair_attempts):
            if self.time_left()<=0:
                return
            self.calls+=1
            self.memory.model_calls+=1
            self.submission=None
            context=self._context(work)
            self._record('states','state_entered',state='DECIDE',work=work,input=context)
            parts=[]
            for part in self._visual_parts(slow=True):
                if part['type']=='text':
                    parts.append(types.Part(text=part['text']))
                else:
                    parts.append(types.Part.from_bytes(data=base64.b64decode(part['image_url']['url'].split(',',1)[1]),mime_type='image/png'))
            parts.append(types.Part(text=json.dumps(context,separators=(',',':'))))
            agent=self._agent(work)
            agent.model.begin_invocation()
            agent.model.timeout_seconds=max(.001,self.time_left())
            agent.model._request_observer=self._request_record
            started=time.monotonic()
            record={'state':'DECIDE','work':work,'call_index':self.calls,'attempt':attempt,'context':context}
            self.rejection=None
            try:
                async with asyncio.timeout(self.time_left()):
                    await ctx.run_node(agent,node_input=types.Content(role='user',parts=parts))
                if self.submission is None:
                    raise ValueError(self.rejection or 'no stage result submitted')
                if self.time_left()<=0:
                    return
                self._accept_stage(work,self.submission)
                record.update(schema_valid=True,response=self.submission.model_dump_json())
            except Exception as exc:
                self.rejection=self.rejection or f'{type(exc).__name__}: {exc}'[:500]
                self.errors.append(self.rejection)
                record.update(schema_valid=False,error=self.rejection)
                self._record('artifacts','task_rejected',work=work,reason=self.rejection)
            finally:
                self.http_requests+=len(agent.model._exchanges)
                record.update(agent.model._last_metrics)
                record.update(http_requests=len(agent.model._exchanges),exchanges=agent.model._exchanges,
                              seconds=time.monotonic()-started)
                self._record('model','model_decision',**record)
                self._record('states','state_exited',state='DECIDE',work=work,output=record)
            if record.get('schema_valid'):
                return
            if attempt<self.repair_attempts:
                self._machine_transition('repair_'+work)
        if self.time_left()>0:
            self._machine_transition('invalid_'+work)
            self._stop('stage_output_invalid')
