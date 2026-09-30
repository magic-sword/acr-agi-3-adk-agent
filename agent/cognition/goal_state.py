"""Contracts for goal-directed play: model-proposed goal predicates and probe choices."""
from typing import Literal
from pydantic import Field
from .state import Contract
from .predicates import RELATIONS


class Atom(Contract):
    relation: Literal[tuple(RELATIONS)]
    a: str = Field(description='Object ID from the object list.')
    b: str | None = Field(default=None, description='Second object ID for a relation between two objects, else null.')
    color: str | None = Field(default=None, description='Colour name for color_is, else null.')


class GoalHypothesis(Contract):
    atoms: list[Atom] = Field(min_length=1, max_length=4,
        description='Conditions that must all hold on the screen when the level is won.')
    rationale: str = Field(min_length=1, max_length=160)


ROLES = ('controllable', 'target_or_container', 'example_or_pattern', 'piece_to_place', 'obstacle_or_wall',
         'status_or_counter', 'background_or_other')


class RoleAssignment(Contract):
    id: str = Field(description='Object ID from the object list.')
    role: Literal[ROLES]


class GoalHypotheses(Contract):
    observation_id: str
    # Reasoning before the structured answer: V8 found reason-first output better (19/60 vs 11/60).
    analysis: str = Field(default='', max_length=600, description='Your reasoning about the screen, written first.')
    # Declared before hypotheses so the model assigns roles first (V7: roles + measured facts
    # turned 0/5 into 5/5 correct first hypotheses on ls20).
    # Required: an optional field was left empty, and hypotheses then ignored roles (V7 bridge test).
    roles: list[RoleAssignment] = Field(min_length=1, max_length=12,
        description='Roles of the important objects, decided before the hypotheses.')
    hypotheses: list[GoalHypothesis] = Field(min_length=1, max_length=6,  # up to 3 per sample; votes merge them
        description='Distinct win-condition hypotheses, most likely first.')


class ProbeChoice(Contract):
    observation_id: str
    item: str = Field(description='ID of the untested item to try.')
    rationale: str = Field(min_length=1, max_length=160)
