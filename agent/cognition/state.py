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
