"""Replaceable, provisional candidate generation. No runtime or notebook access.

Providers receive a detached JSON request and may use the supplied asynchronous
model callback, or return their own CandidateBatch. All output is host-validated.
Names describe semantic actions, not controller buttons or pixel coordinates.
"""
from typing import Awaitable, Callable, Literal, Protocol
from pydantic import Field
from .state import Contract

# A provisional vocabulary for the default model provider only. Custom providers
# can supply other semantic verbs; supported new verbs can subsequently be reused.
DEFAULT_VERBS = ('move_to', 'activate', 'attack', 'push', 'pull', 'manipulate', 'inspect', 'combine')


class Candidate(Contract):
    id: str = Field(min_length=1, max_length=40)
    verb: str = Field(min_length=1, max_length=60)
    target_query: str = Field(min_length=1, max_length=300)
    expected_effect: str = Field(min_length=1, max_length=180)
    rationale: str = Field(min_length=1, max_length=180)
    candidate_refs: list[str] = Field(default_factory=list, max_length=6)
    object_refs: list[str] = Field(default_factory=list, max_length=4,
        description='Current object IDs selected from targets. Region references may specify the acted part.')
    source: Literal['proposed', 'reuse']
    skill_name: str | None = None


class CandidateBatch(Contract):
    observation_id: str
    goal_id: str
    candidates: list[Candidate] = Field(max_length=3)
    missing_info: str = Field(default='', max_length=240)


ModelProposal = Callable[[], Awaitable[CandidateBatch | None]]


class CandidateGenerator(Protocol):
    async def generate(self, request: dict, propose: ModelProposal) -> CandidateBatch | None: ...


class ModelCandidateGenerator:
    """Initial implementation; its proposals are hypotheses, never learned skills."""
    async def generate(self, request: dict, propose: ModelProposal) -> CandidateBatch | None:
        return await propose()
