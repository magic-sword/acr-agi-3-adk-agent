"""Small, stage-specific contracts for grounded goals, procedures and evidence."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Action(Contract):
    action: str
    x: int | None = Field(default=None, ge=0, le=63)
    y: int | None = Field(default=None, ge=0, le=63)
    reason: str = Field(default='', max_length=300)


class Concept(Contract):
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(min_length=1, max_length=240)


class Target(Contract):
    concept: str = Field(min_length=1, max_length=60)
    appearance: str = Field(min_length=1, max_length=160)
    role_hypothesis: str = Field(max_length=180)
    relations: str = Field(max_length=240)


class ActionIntent(Contract):
    action: str
    target_query: str = Field(default='', max_length=300,
        description='For CLICK: describe the visible instance and part to click by appearance, role and relations.')


class Understanding(Contract):
    observation_id: str
    concepts: list[Concept] = Field(min_length=1, max_length=4)
    targets: list[Target] = Field(min_length=1, max_length=6)
    observed: str = Field(min_length=1, max_length=400)
    goal_hypothesis: str = Field(min_length=1, max_length=200)
    causal_hypotheses: str = Field(max_length=400)
    question: str = Field(max_length=200)
    next: Literal['backchain', 'ground']


class Goal(Contract):
    id: str = Field(min_length=1, max_length=40)
    parent_id: str | None
    desired_state: str = Field(min_length=1, max_length=200)
    target_query: str = Field(min_length=1, max_length=300)
    requires: list[str] = Field(max_length=6,
        description='IDs of prerequisite goals declared in goals or retained goals. Empty if none; not action names.')


class Backchain(Contract):
    observation_id: str
    goals: list[Goal] = Field(min_length=1, max_length=6)
    selected_goal_id: str
    rationale: str = Field(min_length=1, max_length=250)


class ActionOption(Contract):
    action: ActionIntent
    expected_effect: str = Field(min_length=1, max_length=180)


class ProcedureStep(Contract):
    purpose: str = Field(min_length=1, max_length=180)
    continue_when: str = Field(min_length=1, max_length=180)
    done_when: str = Field(min_length=1, max_length=180)
    reconsider_when: str = Field(min_length=1, max_length=180)
    options: list[ActionOption] = Field(min_length=1, max_length=6)


class Procedure(Contract):
    name: str = Field(pattern=r'^[a-z][a-z0-9-]{0,47}$')
    when_to_use: str = Field(min_length=1, max_length=180)
    effect: str = Field(min_length=1, max_length=180)
    steps: list[ProcedureStep] = Field(min_length=1, max_length=4)


class GroundedPlan(Contract):
    goal_id: str | None
    intent: Literal['achieve', 'probe']
    question: str = Field(max_length=200)
    target_query: str = Field(min_length=1, max_length=300)
    baseline: str = Field(min_length=1, max_length=200)
    skills: list[Procedure] = Field(max_length=2)
    reuse: list[str] = Field(default_factory=list, max_length=2,
        description='Names of existing procedures to reuse. New skills automatically become candidates.')


class Grounding(Contract):
    observation_id: str
    next: Literal['execute', 'understand', 'backchain']
    reason: str = Field(max_length=200)
    plan: GroundedPlan | None


class Reconciliation(Contract):
    observation_id: str
    goal_id: str | None
    assessment: Literal['matched', 'unexpected', 'unclear', 'probe_result']
    evidence: str = Field(min_length=1, max_length=300)
    goal_status: Literal['active', 'confirmed', 'unknown']
    causal_notes: str = Field(max_length=600)
    next: Literal['understand', 'backchain', 'ground', 'resume']
    reason: str = Field(min_length=1, max_length=200)


class FastSelection(Contract):
    label: str


class Memory(Contract):
    schema_version: int = 11
    revision: int = 0
    run_id: str
    game_id: str
    lifecycle: Literal['BOOT', 'ACTIVE', 'AWAIT_FRAME', 'DONE', 'STOPPED'] = 'BOOT'
    stop_reason: str = ''
    phase: str = 'understand'
    understanding: dict | None = None
    concepts: dict = Field(default_factory=dict)
    cursor: dict | None = None
    goals: dict = Field(default_factory=dict)
    goal_status: dict = Field(default_factory=dict)
    selected_goal_id: str | None = None
    backchain: dict | None = None
    review: dict | None = None
    reconciliations: list[dict] = Field(default_factory=list)
    pending: dict | None = None
    active_skill: dict | None = None
    plan: dict | None = None
    causal_notes: str = ''
    skills: dict = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)
    last_observation: dict = Field(default_factory=dict)
    last_result: dict = Field(default_factory=dict)
    model_calls: int = 0
    resets: int = 0
