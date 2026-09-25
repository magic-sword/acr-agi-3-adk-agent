"""Execution contracts; puzzle knowledge lives in the shared notebook."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class Action(Contract):
    action: str
    x: int | None = Field(default=None, ge=0, le=63)
    y: int | None = Field(default=None, ge=0, le=63)
    reason: str = Field(default="", max_length=300)

class Region(Contract):
    x: int = Field(ge=0, le=63)
    y: int = Field(ge=0, le=63)
    width: int = Field(ge=1, le=64)
    height: int = Field(ge=1, le=64)

class Expectation(Contract):
    kind: Literal['region_changed', 'level_increased', 'semantic']
    description: str = Field(min_length=1, max_length=300)
    region: Region | None = None

    @model_validator(mode='after')
    def scoped(self):
        if (self.kind != 'level_increased') != (self.region is not None):
            raise ValueError('region_changed and semantic require an observed region; level_increased omits it')
        return self

class Subgoal(Contract):
    id: str = Field(default='', max_length=80)
    parent_id: str = Field(default='goal', max_length=80)
    text: str = Field(min_length=1, max_length=200)
    done_when: str = Field(min_length=1, max_length=200)

class DesignRevision(Contract):
    experiment_id: str = Field(min_length=1, max_length=80)
    review_revision: int = Field(ge=1)
    change: Literal['action', 'observation_scope', 'context']
    reconsidered_assumption: str = Field(min_length=1, max_length=250)
    reason: str = Field(min_length=1, max_length=300,
        description='Why this concrete change helps answer the remaining question; not just a different click.')

class ExperimentPlan(Contract):
    subgoal: Subgoal
    question: str = Field(min_length=1, max_length=200)
    hypothesis: str = Field(min_length=1, max_length=300)
    conditions: str = Field(min_length=1, max_length=200)
    expected: Expectation
    context_region: Region | None = Field(default=None,
        description='Optional separate region containing relevant prerequisites, not unrelated status pixels.')
    max_attempts: int = Field(default=1, ge=1, le=3,
        description='Predeclare a bounded repeat only for delay or stochastic effects.')
    repeat_reason: str = Field(default='', max_length=200)
    retry_of: str = Field(default='', max_length=80,
        description='Previous experiment ID; preserves its frozen plan and total attempt limit.')
    revision: DesignRevision | None = Field(default=None,
        description='Required after an unsupported/inconclusive review, except a predeclared retry. Cite notebook.handoff.')

    @model_validator(mode='after')
    def bounded_repeat(self):
        if self.max_attempts > 1 and not self.repeat_reason.strip():
            raise ValueError('multiple attempts require a reason declared before the first action')
        return self

class ReviewUnderstanding(Contract):
    reconsider_assumption: str = Field(min_length=1, max_length=250)
    open_question: str = Field(min_length=1, max_length=250)
    subgoal_reason: str = Field(min_length=1, max_length=250)

class EffectFact(Contract):
    experiment_id: str
    verdict: Literal["supported", "unsupported", "inconclusive"]
    finding: str = Field(min_length=1, max_length=400)
    evidence_ids: list[str] = Field(min_length=1, max_length=4)


class ExperimentReview(Contract):
    experiment_id: str
    verdict: Literal['supported', 'unsupported', 'inconclusive']
    finding: str = Field(min_length=1, max_length=400)
    evidence_ids: list[str] = Field(min_length=1, max_length=4)
    subgoal_status: Literal['active', 'completed', 'abandoned'] = 'active'
    next_step: Literal['continue', 'revise', 'learn', 'stop'] = 'revise'
    update: str = Field(min_length=1, max_length=300)
    understanding: ReviewUnderstanding

class Decision(Contract):
    kind: Literal["act", "invoke", "trial", "evaluate", "learn", "stop"]
    action: Action | None = None
    skill_id: str | None = None
    arguments: dict[str, int] = Field(default_factory=dict)
    purpose: str = Field(min_length=1, max_length=300)
    experiment: ExperimentPlan | None = None
    evidence_ids: list[str] = Field(default_factory=list, max_length=8,
        description='Optional retained experience references for this decision; required for learn.')

    @model_validator(mode="after")
    def selected_work(self):
        if (self.kind == 'act') != (self.experiment is not None):
            raise ValueError('act requires an experiment plan; other jobs must omit experiment')
        if (self.kind == "act") != (self.action is not None):
            raise ValueError("kind=act requires an action object; other kinds must omit action")
        if (self.kind in ("invoke", "trial", "evaluate")) != (self.skill_id is not None):
            raise ValueError("invoke, trial and evaluate require a skill_id")
        if self.kind not in ("invoke", "trial") and self.arguments:
            raise ValueError("arguments are only for invoke or trial")
        if self.kind == 'learn' and not self.evidence_ids:
            raise ValueError('learn requires retained experience IDs')
        return self

# A coordinate is an original pixel or the name of an integer argument ($x).
Coordinate = int | str

class Condition(Contract):
    kind: Literal["cell_is", "cell_changed", "color_count_delta", "level_increased"]
    x: Coordinate | None = None
    y: Coordinate | None = None
    value: int | None = None

    @model_validator(mode="after")
    def valid_condition(self):
        if self.kind.startswith("cell_"):
            for c in (self.x, self.y):
                if not ((type(c) is int and 0 <= c <= 63) or
                        (isinstance(c, str) and c.startswith("$") and c[1:].isidentifier())):
                    raise ValueError("cell condition needs pixel coordinates or $argument names")
        elif self.x is not None or self.y is not None:
            raise ValueError("non-cell condition cannot have coordinates")
        if self.kind in ("cell_is", "color_count_delta"):
            if type(self.value) is not int or not 0 <= self.value <= 15:
                raise ValueError("value is a color ID 0..15")
        elif self.value is not None:
            raise ValueError("this condition has no value")
        return self

class SkillStep(Contract):
    action: Literal["UP", "DOWN", "LEFT", "RIGHT", "ACT", "CLICK", "UNDO"]
    x: Coordinate | None = None
    y: Coordinate | None = None
    before: list[Condition] = Field(min_length=1, max_length=4)
    after: list[Condition] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def valid_step(self):
        if self.action == "CLICK":
            Condition(kind="cell_changed", x=self.x, y=self.y)
        elif self.x is not None or self.y is not None:
            raise ValueError("only CLICK has coordinates")
        if any(c.kind != "cell_is" for c in self.before):
            raise ValueError("start conditions must be current observed cell colors")
        if not any(c.kind in ("cell_changed", "color_count_delta", "level_increased") for c in self.after):
            raise ValueError("each step requires an observable change, not an already-true claim")
        return self

class SkillSpec(Contract):
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,47}$")
    description: str = Field(min_length=1, max_length=300)
    game_id: str = Field(min_length=1, max_length=100)
    parameters: list[str] = Field(default_factory=list, max_length=4)
    steps: list[SkillStep] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def parameters_used(self):
        if len(set(self.parameters)) != len(self.parameters) or any(not p.isidentifier() for p in self.parameters):
            raise ValueError("parameter names must be unique identifiers")
        refs=set()
        for s in self.steps:
            for c in [s, *s.before, *s.after]:
                for v in (c.x, c.y):
                    if isinstance(v, str):
                        refs.add(v[1:])
        if refs != set(self.parameters):
            raise ValueError("all and only declared parameters must be used as $names")
        return self

class Draft(Contract):
    spec: SkillSpec
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    examples: list[dict[str, int]] = Field(min_length=1, max_length=8)
    parent_id: str | None = None

class SkillDeferral(Contract):
    reason: str = Field(min_length=1, max_length=300)

class ExperimentRedesign(Contract):
    """Host routing signal; never exposed as a model submission schema."""
    experiment_id: str
    blocked_action: dict
    reason: str

class Memory(Contract):
    schema_version: int = 7
    revision: int = 0
    run_id: str
    game_id: str
    lifecycle: Literal["BOOT", "ACTIVE", "AWAIT_FRAME", "DONE", "STOPPED"] = "BOOT"
    stop_reason: str = ""
    pending: dict | None = None
    active_skill: dict | None = None
    history: list[dict] = Field(default_factory=list)
    last_observation: dict = Field(default_factory=dict)
    last_result: dict = Field(default_factory=dict)
    model_calls: int = 0
    resets: int = 0
