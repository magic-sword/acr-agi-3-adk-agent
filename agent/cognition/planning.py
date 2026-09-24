"""Pure forward checking of a proposed order under explicit symbolic assumptions."""
from google.adk.tools import BaseTool
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OrderStep(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    id: str = Field(min_length=1, max_length=100)
    requires: list[str] = Field(default_factory=list, max_length=32)
    adds: list[str] = Field(default_factory=list, max_length=32)
    removes: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode='after')
    def consistent_effects(self):
        if set(self.adds) & set(self.removes):
            raise ValueError('one step cannot both add and remove the same condition')
        return self


class OrderQuery(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    initial_conditions: list[str] = Field(max_length=64)
    steps: list[OrderStep] = Field(min_length=1, max_length=12)
    required_final_conditions: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode='after')
    def unique_steps(self):
        if len({step.id for step in self.steps}) != len(self.steps):
            raise ValueError('step IDs must be unique')
        conditions = (self.initial_conditions + self.required_final_conditions
                      + [c for s in self.steps for c in s.requires + s.adds + s.removes])
        if any(not c or len(c) > 100 for c in conditions):
            raise ValueError('conditions must be nonempty names of at most 100 characters')
        return self


def check_order(query: OrderQuery) -> dict:
    conditions = set(query.initial_conditions)
    trace = []
    result = {'scope': 'consistency of supplied assumptions only; not evidence of game effects',
              'trace': trace}
    for step in query.steps:
        missing = sorted(set(step.requires) - conditions)
        trace.append({'step': step.id, 'before': sorted(conditions), 'missing': missing})
        if missing:
            return {**result, 'feasible_under_assumptions': False,
                    'blocked_step': step.id, 'missing_conditions': missing}
        conditions.difference_update(step.removes)
        conditions.update(step.adds)
        trace[-1]['after'] = sorted(conditions)
    missing = sorted(set(query.required_final_conditions) - conditions)
    return {**result, 'feasible_under_assumptions': not missing,
            'blocked_step': None, 'missing_conditions': missing,
            'final_conditions': sorted(conditions)}


class PlanOrderTool(BaseTool):
    def __init__(self):
        super().__init__(name='check_plan_order', description=(
            'Check a proposed sequence by applying explicit added/removed conditions. '
            'Returns the first unmet prerequisite or missing final condition. '
            'Supply current conditions and known effects; this checks your assumptions, '
            'not the truth of game rules. Never executes actions or updates memory.'))

    def _get_declaration(self):
        return types.FunctionDeclaration(name=self.name, description=self.description,
                                         parameters_json_schema=OrderQuery.model_json_schema())

    async def run_async(self, *, args, tool_context):
        try:
            return check_order(OrderQuery.model_validate(args))
        except ValueError as exc:
            return {'error': str(exc)[:2000]}
