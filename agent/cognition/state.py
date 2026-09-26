"""Contracts for deliberation, grounded procedures and acknowledged execution."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Action(Contract):
    action: str
    x: int | None = Field(default=None, ge=0, le=63)
    y: int | None = Field(default=None, ge=0, le=63)
    reason: str = Field(default='', max_length=300)


class ActionOption(Contract):
    action: Action
    expected_effect: str = Field(min_length=1, max_length=200)


class ProcedureStep(Contract):
    purpose: str = Field(min_length=1, max_length=200)
    done_when: str = Field(min_length=1, max_length=200)
    reconsider_when: str = Field(min_length=1, max_length=200)
    options: list[ActionOption] = Field(min_length=1, max_length=6)


class Procedure(Contract):
    name: str = Field(pattern=r'^[a-z][a-z0-9-]{0,47}$')
    when_to_use: str = Field(min_length=1, max_length=200)
    effect: str = Field(min_length=1, max_length=200)
    steps: list[ProcedureStep] = Field(min_length=1, max_length=6)


class Deliberation(Contract):
    observation_id: str
    interpretation: str = Field(min_length=1, max_length=600)
    goal: str = Field(min_length=1, max_length=200)
    backward_plan: list[str] = Field(min_length=1, max_length=6)
    causal_notes: str = Field(max_length=800)
    skills: list[Procedure] = Field(max_length=4,
        description='Create or replace named procedures. Reuse existing procedures by name when appropriate.')
    candidates: list[str] = Field(min_length=1, max_length=4,
        description='Names of procedures suitable for this goal and current board; include new or retained skills.')


class FastSelection(Contract):
    label: str


class Memory(Contract):
    schema_version: int = 9
    revision: int = 0
    run_id: str
    game_id: str
    lifecycle: Literal['BOOT', 'ACTIVE', 'AWAIT_FRAME', 'DONE', 'STOPPED'] = 'BOOT'
    stop_reason: str = ''
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
