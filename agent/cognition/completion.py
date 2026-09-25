"""Model submissions propose work. Only RUN applies changes or selects actions."""
from google.adk.tools import BaseTool
from google.genai import types

class CompletionTool(BaseTool):
    def __init__(self, runtime, name, contract):
        super().__init__(name=name, description=(
            'Submit one next job. Call alone. Invalid submissions can be corrected. '
            'propose_skill creates a candidate only; the host owns testing and promotion.'
            if name=='propose_skill' else
            'Choose one of the currently offered jobs. Call alone. '
            'CLICK requires explicit original-pixel x,y. trial spends real game actions.'))
        self.runtime, self.contract = runtime, contract

    def _get_declaration(self):
        schema = self.contract.model_json_schema()
        if self.name == 'submit_decision':
            catalog = self.runtime.library.catalog()
            kinds = ['act', 'stop']
            if self.runtime.learning and self.runtime.library.experiences and self.runtime.work == 'action':
                kinds.append('learn')
            if any(s['status']=='active' for s in catalog):
                kinds.append('invoke')
            if self.runtime.learning and any(s['status']=='candidate' for s in catalog):
                kinds.extend(['trial', 'evaluate'])
            schema['properties']['kind']['enum'] = kinds
            schema['properties']['kind']['description'] = 'act includes exploring unknown rules. trial is ONLY for an existing candidate skill ID.'
            # Keep common fields at the root: the local tool grammar can lose them
            # when a root oneOf branch is selected. Null is explicit for non-act jobs.
            schema['required'].append('action')
            schema['properties']['action']['description'] = 'An object with action set to one legal button when kind=act; null for every other kind.'
            if not any(k in kinds for k in ('invoke', 'trial', 'evaluate')):
                schema['properties'].pop('skill_id')
                schema['properties'].pop('arguments')
            if not self.runtime.library.experiences:
                schema['properties'].pop('evidence_ids')
            from agent.controls import ACTION_TO_BUTTON
            allowed = [ACTION_TO_BUTTON[a] for a in self.runtime.obs.get('available_actions', []) if a in ACTION_TO_BUTTON and a!='RESET']
            action = schema['$defs']['Action']
            if allowed:
                action['properties']['action']['enum'] = allowed
            if allowed and 'CLICK' not in allowed:
                action['properties'].pop('x', None)
                action['properties'].pop('y', None)
            if allowed == ['CLICK']:
                action['required'] = ['action', 'x', 'y']
                for axis, dimension in [('x','width'),('y','height')]:
                    action['properties'][axis] = {'type':'integer', 'minimum':0,
                        'maximum':self.runtime.obs.get(dimension,64)-1}
        return types.FunctionDeclaration(name=self.name, description=self.description,
                                         parameters_json_schema=schema)

    async def run_async(self, *, args, tool_context):
        if self.runtime.submission is not None:
            return {'accepted':False,'error':'a job has already been submitted'}
        try:
            proposal = self.contract.model_validate(args)
            self.runtime.validate_job(proposal)
        except ValueError as e:
            correction = 'Correct the reported field error. Every decision needs kind, prediction, and action (null for a non-act job).'
            if (self.name == 'submit_decision' and args.get('kind') == 'act'
                    and not isinstance(args.get('action'), dict)):
                correction = ('kind=act requires action={"action":"<one legal button>"}. '
                              'CLICK additionally requires action.x and action.y; other buttons omit both coordinates.')
            return {'accepted':False,'error':str(e)[:1500],
                    'correction':correction}
        self.runtime.submission = proposal
        tool_context.actions.skip_summarization = True
        return {'accepted':True,'state':'DECIDE','next':'RUN'}
