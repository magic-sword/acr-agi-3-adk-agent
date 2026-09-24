"""State-scoped result submission; ADK ends the tool loop, Workflow owns routing."""
from copy import copy

from google.adk.tools import BaseTool
from google.genai import types

from agent.controls import controller_context
from .state import Interpretation, Proposal

COMPLETION_TOOLS = {
    'PLAN': 'submit_plan', 'REVISE': 'submit_revision',
    'PROBE': 'submit_experiment', 'VERIFY': 'submit_interpretation',
}


def submission_schema(state, buttons=None):
    schema = (Interpretation if state == 'VERIFY' else Proposal).model_json_schema()
    action = schema.get('$defs', {}).get('Action', {})
    for field in ('x', 'y'):
        action.get('properties', {}).pop(field, None)
    if 'action' in action.get('properties', {}):
        action['properties']['action']['enum'] = (buttons if buttons is not None else
            ['UP', 'DOWN', 'LEFT', 'RIGHT', 'ACT', 'CLICK', 'UNDO'])
    if state == 'PROBE':
        schema['properties']['purpose'] = {'type': 'string', 'enum': ['probe']}
        schema['properties']['status'] = {'type': 'string', 'enum': ['ok'], 'default': 'ok'}
        schema['required'] = list(dict.fromkeys(schema['required'] + ['action', 'experiment']))
    return schema


class CompletionTool(BaseTool):
    def __init__(self, runtime, state):
        super().__init__(name=COMPLETION_TOOLS[state], description=(
            f'Submit the completed {state} result. On validation error, correct and retry. '
            'Acceptance ends this reasoning state; Workflow handles routing and game execution. '
            'Call alone, never alongside another tool.'))
        self.runtime, self.state = runtime, state

    def _get_declaration(self):
        turn = self.runtime.turn
        buttons = (controller_context({'available_actions': turn.obs.get('available_actions', [])})
                   ['available_actions'] if turn else None)
        return types.FunctionDeclaration(name=self.name, description=self.description,
                                         parameters_json_schema=submission_schema(self.state, buttons))

    async def run_async(self, *, args, tool_context):
        runtime, state = self.runtime, self.state
        if runtime._submission is not None:
            return {'accepted': False, 'error': 'A result has already been submitted'}
        try:
            if runtime.turn.time_left() <= 0:
                raise ValueError('reasoning time budget exhausted')
            result = (Interpretation if state == 'VERIFY' else Proposal).model_validate(args)
            # Check against an isolated transaction; no memory/cursor/game side effects.
            trial = copy(runtime.turn)
            trial.memory = runtime.turn.memory.model_copy(deep=True)
            if state == 'VERIFY':
                trial.stage_review(result)
            else:
                if state == 'PROBE' and (result.status != 'ok' or result.purpose != 'probe'):
                    raise ValueError('PROBE requires a complete discriminating experiment')
                trial.accept(result, state)
        except ValueError as exc:
            response = {'accepted': False, 'error': str(exc)[:2000],
                        'instruction': 'Correct the submission and call this tool again.'}
            runtime._submission_attempts.append({'tool': self.name, **response})
            return response
        runtime._submission = result
        tool_context.actions.skip_summarization = True
        response = {'accepted': True, 'state': state, 'observation_id': result.observation_id}
        runtime._submission_attempts.append({'tool': self.name, **response})
        return response
