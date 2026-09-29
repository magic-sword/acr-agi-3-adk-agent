"""Contracts for the current single-action runtime. Schema 15 is not an old-memory migration."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from .object_memory import empty

class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

class Action(Contract):
    action: str
    x: int | None = Field(default=None, ge=0, le=63)
    y: int | None = Field(default=None, ge=0, le=63)
    reason: str = Field(default='', max_length=300)

class ActionIntent(Contract):
    action: str
    target_query: str = Field(default='', max_length=300,
        description='For CLICK: describe the visible instance and part to click by appearance, role and relations.')

class Memory(Contract):
    schema_version: int = 15
    revision: int = 0
    run_id: str
    game_id: str
    lifecycle: Literal['BOOT', 'ACTIVE', 'AWAIT_FRAME', 'DONE', 'STOPPED'] = 'BOOT'
    stop_reason: str = ''
    phase: str = 'act'
    pending: dict | None = None
    active_skill: dict | None = None
    plan: dict | None = None
    episode: int = 0
    history: list[dict] = Field(default_factory=list)
    trial_ledger: list[dict] = Field(default_factory=list)
    object_memory: dict = Field(default_factory=empty)
    last_observation: dict = Field(default_factory=dict)
    last_result: dict = Field(default_factory=dict)
    model_calls: int = 0
    resets: int = 0

class Target(Contract):
    concept: str = Field(min_length=1, max_length=60)
    appearance: str = Field(min_length=1, max_length=160)
    role_hypothesis: str = Field(max_length=180)
    relations: str = Field(max_length=240)
    candidate_refs: list[str] = Field(default_factory=list, max_length=12,
        description='Current measured candidate IDs supporting this target/group, if provided. Empty if unresolved.')

class ActionOption(Contract):
    target_object_id: str | None = Field(default=None, description="CLICK: current object ID belonging to the selected candidate; required if multiple targets.")
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

class Reconciliation(Contract):
    observation_id: str
    goal_id: str | None
    assessment: Literal['matched', 'unexpected', 'unclear', 'probe_result']
    evidence: str = Field(min_length=1, max_length=300)
    goal_status: Literal['active', 'confirmed', 'unknown']
    causal_notes: str = Field(max_length=600)
    next: Literal['understand', 'backchain', 'ground', 'resume']
    reason: str = Field(min_length=1, max_length=200)
    next_question: str = Field(min_length=1, max_length=240)
