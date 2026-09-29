"""Contracts for the default compact planning workflow (memory schema 13)."""
from typing import Literal
from pydantic import Field
from .state import Contract, Memory, Target, Procedure


class Scene(Contract):
    observation_id: str
    observed: str = Field(min_length=1, max_length=400)
    goal_hypothesis: str = Field(min_length=1, max_length=200)
    targets: list[Target] = Field(max_length=4)


class Subgoal(Contract):
    observation_id: str
    desired_state: str = Field(min_length=1, max_length=200)
    target_query: str = Field(min_length=1, max_length=300)
    intent: Literal['achieve', 'probe']
    question: str = Field(max_length=200)
    expected_observation: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=250)


class PlanChoice(Contract):
    observation_id: str
    candidate_id: str | None
    procedure: Procedure | None
    probe_action_limit: int | None = Field(ge=1, le=16,
        description='For a probe: maximum acknowledged controller actions before reviewing even no change. Otherwise null.')
    next: Literal['execute', 'understand', 'backchain']
    reason: str = Field(min_length=1, max_length=240)


class FocusedMemory(Memory):
    schema_version: int = 13
    final_goal: str = ''
    subgoal: dict | None = None
    candidate_batch: dict | None = None
    selected_candidate: dict | None = None
    # These are attributed past results, never current observations.
    trial_ledger: list[dict] = Field(default_factory=list)
    supported_skills: dict = Field(default_factory=dict)
    causal_knowledge: list[dict] = Field(default_factory=list)
