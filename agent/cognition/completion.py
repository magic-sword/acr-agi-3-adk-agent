"""Model submissions propose work. Only RUN applies changes or selects actions."""
from google.adk.tools import BaseTool
from google.genai import types
from .experiments import RepeatedExperiment
from .state import ExperimentRedesign

class CompletionTool(BaseTool):
    def __init__(self, runtime, name, contract):
        super().__init__(name=name, description=(
            'Submit the verdict for the active observed experiment; no actions are allowed.'
            if name == 'submit_review' else
            'Return to experiment design with missing evidence.' if name == 'defer_skill' else
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
            if self.runtime.learning and self.runtime.library.experiences and self.runtime.work == 'experiment_design':
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
            schema['required'].append('experiment')
            schema['properties']['experiment']['description'] = 'Frozen experiment plan for act; null for every other kind.'
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
            self.runtime.proposed_job = proposal
            self.runtime.validate_job(proposal)
        except RepeatedExperiment as exc:
            # End this invocation and return through the graph with a compact
            # redesign input. Do not spend all tool rounds repeating a field error.
            self.runtime.submission = ExperimentRedesign(experiment_id=exc.page['id'],
                blocked_action=exc.page['data']['action'], reason=str(exc))
            tool_context.actions.skip_summarization = True
            return {'accepted': False, 'redesign_required': True, 'next': 'experiment_design',
                    'experiment_id': exc.page['id'], 'reason': str(exc)}
        except ValueError as e:
            correction = 'Correct the reported field error using this tool schema.'
            if self.name == 'submit_decision':
                correction += ' Every decision needs kind, purpose, experiment (for act), and action (null for a non-act job).'
            if (self.name == 'submit_decision' and args.get('kind') == 'act'
                    and not isinstance(args.get('action'), dict)):
                correction = ('kind=act requires action={"action":"<one legal button>"}. '
                              'CLICK additionally requires action.x and action.y; other buttons omit both coordinates.')
            return {'accepted':False,'error':str(e)[:1500],
                    'correction':correction}
        self.runtime.submission = proposal
        tool_context.actions.skip_summarization = True
        return {'accepted':True,'state':'DECIDE','next':'RUN'}
