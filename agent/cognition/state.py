"""Bounded, serializable contracts. Model proposals never mutate these directly."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Predicate(Contract):
    kind: Literal["cell", "fact", "state", "levels_min", "frame_changed"]
    key: str = ""
    value: str | int | bool
    x: int | None = Field(default=None, ge=0, le=63)
    y: int | None = Field(default=None, ge=0, le=63)

    @model_validator(mode="after")
    def coordinates(self):
        if self.kind == "cell" and (self.x is None or self.y is None):
            raise ValueError("cell predicates require x and y")
        if self.kind == "cell" and (type(self.value) is not int or not 0 <= self.value <= 15):
            raise ValueError("cell requires an integer color ID in [0,15]")
        if self.kind == "state" and not isinstance(self.value, str):
            raise ValueError("state requires a string")
        if self.kind == "fact" and not self.key:
            raise ValueError("fact predicates require a key")
        if self.kind == "levels_min" and type(self.value) is not int:
            raise ValueError("levels_min requires an integer")
        if self.kind == "frame_changed" and type(self.value) is not bool:
            raise ValueError("frame_changed requires a boolean")
        return self


class Action(Contract):
    action: str
    x: int | None = Field(default=None, ge=0, le=63)
    y: int | None = Field(default=None, ge=0, le=63)
    reason: str = Field(default="", max_length=300)


class Fact(Contract):
    key: str = Field(min_length=1, max_length=100)
    value: str | int | bool
    evidence: str
    visible: bool = True


class Hypothesis(Contract):
    id: str
    kind: Literal["goal", "control", "dynamics", "mode"]
    claim: str = Field(max_length=500)
    scope: Literal["level", "game"] = "level"
    status: Literal["candidate", "supported", "refuted", "suspended"] = "candidate"
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)


class Perception(Contract):
    observation_id: str
    memory_revision: int
    facts: list[Fact] = Field(default_factory=list, max_length=24)
    unknowns: list[str] = Field(default_factory=list, max_length=8)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=8)
    goal: str = Field(default="", max_length=500)
    change: Literal["none", "layout", "mode", "dynamics", "goal", "uncertain"] = "none"
    change_evidence: list[str] = Field(default_factory=list, max_length=8)


class PlanNode(Contract):
    id: str
    subgoal: str = Field(max_length=300)
    depends_on: list[str] = Field(default_factory=list, max_length=12)
    hypothesis_ids: list[str] = Field(default_factory=list, max_length=8)
    preconditions: list[Predicate] = Field(default_factory=list, max_length=8)
    completion: list[Predicate] = Field(min_length=1, max_length=8)
    invariants: list[Predicate] = Field(default_factory=list, max_length=8)
    action: Action | None = None
    effects: list[Predicate] = Field(default_factory=list, max_length=8)
    delay_steps: int = Field(default=1, ge=1, le=8)
    earliest_step: int | None = Field(default=None, ge=0)
    latest_step: int | None = Field(default=None, ge=0)
    status: Literal["todo", "active", "done", "invalid"] = "todo"
    started_hash: str = ""
    started_step: int | None = None


class Experiment(Contract):
    question: str = Field(min_length=1, max_length=400)
    alternatives: dict[str, list[Predicate]] = Field(min_length=2, max_length=4)
    discriminator: str = Field(min_length=1, max_length=300)
    risk: str = Field(min_length=1, max_length=300)


class Proposal(Contract):
    observation_id: str
    memory_revision: int
    status: Literal["ok", "need_evidence", "exhausted"] = "ok"
    purpose: Literal["probe", "plan", "recover"]
    action: Action | None = None
    effects: list[Predicate] = Field(default_factory=list, max_length=8)
    invariants: list[Predicate] = Field(default_factory=list, max_length=8)
    delay_steps: int = Field(default=1, ge=1, le=8)
    plan: list[PlanNode] = Field(default_factory=list, max_length=12)
    node_id: str | None = None
    experiment: Experiment | None = None
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)
    diagnosis: str = Field(default="", max_length=500)
    invalidated_hypotheses: list[str] = Field(default_factory=list, max_length=8)


class Pending(Contract):
    decision_id: str
    observation_id: str
    step: int
    action: Action
    effects: list[Predicate] = Field(default_factory=list)
    invariants: list[Predicate] = Field(default_factory=list)
    deadline_step: int
    baseline_hash: str
    node_id: str | None = None
    experiment: Experiment | None = None
    intervening_actions: list[str] = Field(default_factory=list)


class Memory(Contract):
    schema_version: int = 1
    revision: int = 0
    run_id: str
    game_id: str
    lifecycle: Literal["BOOT", "ACTIVE", "AWAIT_FRAME", "RECOVER", "DONE", "STOPPED"] = "BOOT"
    stop_reason: str = ""
    attempt: int = 0
    level: int = 0
    model_revision: int = 0
    goal: str = ""
    facts: dict[str, Fact] = Field(default_factory=dict)
    hypotheses: dict[str, Hypothesis] = Field(default_factory=dict)
    unknowns: list[str] = Field(default_factory=list)
    plan: list[PlanNode] = Field(default_factory=list)
    pending: Pending | None = None
    deferred: list[Pending] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)
    procedures: list[dict] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    last_observation: dict = Field(default_factory=dict)
    last_result: dict = Field(default_factory=dict)
    model_calls: int = 0
    resets: int = 0
