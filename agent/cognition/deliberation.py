"""One typed model submission with a bounded schema-repair attempt."""
import asyncio
import base64
import json
import os
import time
from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool
from google.genai import types
from agent.local_vlm import LocalVisionLlm

MEASUREMENT_INSTRUCTION = ('\nMeasured changes are host observations. Do not replace them with '
    'a no-change guess. Roles and causation remain hypotheses. When understanding targets, '
    'fill candidate_refs from supplied current candidates or candidate_index rows where they match the image; use [] '
    'when unresolved. Candidate references expire with that observation; target_correspondence '
    'supplies conservative links to later observations, not proof of identity. A region may include '
    'background or several objects.')


class StageTool(BaseTool):
    def __init__(self, runtime, work):
        name, self.contract = runtime.stage_tasks[work]
        super().__init__(name=name, description=f'Submit the {work} result for the current observation.')
        self.runtime, self.work = runtime, work

    def _get_declaration(self):
        schema = self.contract.model_json_schema()
        schema['properties']['observation_id']['enum'] = [self.runtime.obs['observation_id']]
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


class DeliberationStages:
    accept_late_results = False

    def _agent(self, work):
        if work not in self.planners:
            tool_name=self.stage_tasks[work][0]
            self.planners[work]=LlmAgent(name=work,include_contents='none',
                model=LocalVisionLlm(model='qwen3-vl-4b-instruct',
                    api_base=os.getenv('VLM_API_BASE','http://vlm:8080/v1'),
                    max_output_tokens=self.stage_tokens[work],max_requests=1,completion_tools=(tool_name,)),
                instruction=self.stage_instructions[work]+MEASUREMENT_INSTRUCTION,tools=[self.stage_tool(self,work)],
                before_tool_callback=self._before_tool,after_tool_callback=self._after_tool,
                on_tool_error_callback=self._tool_error)
        return self.planners[work]

    async def _deliberate(self, ctx, work, *, accept_result=True):
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
                # A late result is kept only by runtimes whose fallback can execute it.
                if self.time_left()<=0 and not self.accept_late_results:
                    return
                if accept_result:
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
                return self.submission
            if attempt<self.repair_attempts:
                self._machine_transition('repair_'+work)
        if self.time_left()>0:
            self._machine_transition('invalid_'+work)
            self._stage_invalid(work)

    def _stage_invalid(self, work):
        self._stop('stage_output_invalid')


def unique(values, name):
    if len(values) != len(set(values)):
        raise ValueError("duplicate " + name)
