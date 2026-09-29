"""Memory presentation ablation on a fixed one-action/review state machine.

Both arms have identical prompts, output contracts and knowledge-writing rules.
Only the slow-stage input context changes. Raw notes remain an audit archive.
"""
from copy import deepcopy
import re
from typing import Literal

from pydantic import Field

from agent.controls import controller_context
from .perception import context_record
from .planning_comparison import (
    OneActionRuntime, OneActionTool, OneActionPlan, ActionReview, evidence_context)
from .state import Contract


class KnowledgeUpdate(Contract):
    knowledge_id: str = Field(default='', max_length=40)
    kind: Literal['role', 'causal', 'procedure']
    subject: str = Field(min_length=1, max_length=120)
    claim: str = Field(min_length=1, max_length=240)
    conditions: str = Field(min_length=1, max_length=180)
    status: Literal['hypothesis', 'supported', 'contradicted']


class MemoryReview(ActionReview):
    knowledge_updates: list[KnowledgeUpdate] = Field(default_factory=list, max_length=2)


MEMORY_INSTRUCTION = '''
Memory experiment: both input layouts use the same reasoning and output contract.
With a separated layout, current_observation is the only current scene;
active_trial freezes this attempt's pre-action baseline, targets and expected_effect.
Compare active_trial.actual_result ONLY with that expected_effect. If actual_result
is null, this attempt has no result: do not fill it using another trial's result.
last_completed_trial is explicitly historical; it can guide the next experiment,
but never establishes an effect in a new unexecuted attempt. Long-term knowledge
contains attributed, conditional claims, not current locations or verified truth.
Bind any remembered role to CURRENT measured candidates again. A former position,
candidate reference or region change is not a current object or an actual click.
With a mixed layout, use recent_trials, last_result and last_review, checking their
attempt and observation IDs against current_invocation_result in the same way.
In review, optionally return at most two knowledge_updates: reusable role,
conditional causal relation or procedure. State appearance/role, claim and applicable
conditions, not current coordinates, candidate IDs, observation IDs or frame-specific
state. Use [] if nothing reusable was learned. Keep uncertainty; one unchanged frame
does not prove a control never works. status is your assessment, not host verification.
Use an existing knowledge_id to revise/refute a claim; empty creates a new claim.
Never replace a past claim merely to make the current result fit a prediction.
'''


def trial_facts(result):
    """One historical transition; exclude embedded older measurements/interpretations."""
    if result is None:
        return None
    keys = ('decision_id', 'before_observation_id', 'after_observation_id', 'action',
            'acknowledged', 'prediction', 'boundary', 'frame_changed',
            'changed_cell_count', 'changed_cells', 'change_bounds')
    return {k: deepcopy(result[k]) for k in keys if k in result}


class MemoryComparisonRuntime(OneActionRuntime):
    memory_layout = 'mixed'

    def __init__(self, *args, **kwargs):
        self.knowledge = {}
        self.knowledge_sequence = 0
        self.frozen_trial = None
        self.last_completed_trial = None
        super().__init__(*args, **kwargs)

    def _on_observation(self, boundary):
        if boundary:
            self.frozen_trial = None
            self.last_completed_trial = None
        super()._on_observation(boundary)

    def _agent(self, work):
        fresh = work not in self.planners
        agent = super()._agent(work)
        if fresh:
            agent.instruction += MEMORY_INSTRUCTION
            if work == 'reconcile':
                tool = OneActionTool(self, work)
                tool.contract = MemoryReview
                agent.tools = [tool]
        return agent

    def validate_stage(self, work, value):
        super().validate_stage(work, value)
        if not isinstance(value, MemoryReview):
            return
        seen = set()
        for update in value.knowledge_updates:
            if update.knowledge_id:
                if update.knowledge_id not in self.knowledge:
                    raise ValueError('unknown knowledge_id')
                if update.knowledge_id in seen:
                    raise ValueError('update each knowledge_id at most once')
                seen.add(update.knowledge_id)
            text = ' '.join((update.subject, update.claim, update.conditions))
            if re.search(r'\bf\d+t\d+\b|\(\s*\d+\s*,\s*\d+\s*\)', text):
                raise ValueError('knowledge must not store candidate IDs or coordinate pairs')
            if self.obs['observation_id'] in text:
                raise ValueError('observation provenance is attached by the host, not part of a claim')
            if update.status != 'hypothesis' and not (self._current_result() or {}).get('acknowledged'):
                raise ValueError('supported/contradicted knowledge requires this attempt acknowledgement')

    def _accept_stage(self, work, value):
        self.validate_stage(work, value)
        actual = deepcopy(self._current_result())
        active = deepcopy(self.memory.active_skill)
        super()._accept_stage(work, value)
        if isinstance(value, OneActionPlan):
            self.frozen_trial = dict(
                plan_id=self.memory.plan['plan_id'],
                before_observation_id=value.observation_id,
                baseline=value.baseline, expected_effect=value.expected_effect,
                targets=[dict(appearance=t.appearance, role_hypothesis=t.role_hypothesis)
                         for t in value.targets],
                planned_action=value.action.model_dump())
        elif isinstance(value, MemoryReview):
            if actual is not None:
                self.last_completed_trial = dict(
                    **(deepcopy(self.frozen_trial) or {}),
                    invocation_id=(active or {}).get('invocation_id'),
                    actual_result=trial_facts(actual),
                    model_assessment=value.assessment, next_question=value.next_question,
                    interpretation_is_verified=False)
            for update in value.knowledge_updates:
                key = update.knowledge_id
                if not key:
                    self.knowledge_sequence += 1
                    key = f'k{self.knowledge_sequence}'
                prior = self.knowledge.get(key, {})
                provenance = dict(observation_id=value.observation_id,
                    invocation_id=(active or {}).get('invocation_id'),
                    decision_id=(actual or {}).get('decision_id'),
                    acknowledged=(actual or {}).get('acknowledged', False))
                record = dict(**update.model_dump(exclude={'knowledge_id'}), knowledge_id=key,
                    game_id=self.memory.game_id, episode=self.memory.episode,
                    author='reconcile', verified_by_host=False,
                    sources=prior.get('sources', []) + [provenance])
                self.knowledge[key] = record
                self._write_note('knowledge', update.subject, record, author='reconcile')
                self._record('artifacts', 'knowledge_updated', record=record,
                             memory_layout=self.memory_layout)

    def _knowledge_context(self):
        # All claims remain retained in knowledge and append-only notes. The small
        # working set is bounded; this experiment does not add a retrieval state.
        return [deepcopy(v) for v in list(self.knowledge.values())[-8:]]

    def _context(self, work=None):
        work = work or self.work
        if work not in ('ground', 'reconcile'):
            return super()._context(work)
        if self.memory_layout == 'mixed':
            context = super()._context(work)
            context.update(memory_layout='mixed', long_term_knowledge=self._knowledge_context())
            return context
        m = self.memory
        context = dict(work=work, memory_layout='separated',
            observation_id=self.obs['observation_id'],
            remaining_actions=self.obs['remaining_actions'], seconds_left=round(self.time_left(), 2),
            available_actions=self.obs['available_actions'], correction=self.rejection,
            current_observation=dict(observation_id=self.obs['observation_id'],
                measured_objects=context_record(self.perception, inventory=True)),
            long_term_knowledge=self._knowledge_context(), question=m.handoff_question)
        if work == 'reconcile':
            context['active_trial'] = dict(**(deepcopy(self.frozen_trial) or {}),
                invocation_id=(m.active_skill or {}).get('invocation_id'),
                actual_result=trial_facts(self._current_result()),
                review_trigger=deepcopy(m.review))
            # Old outcomes and old reviews are deliberately not injected here.
        else:
            context['last_completed_trial'] = deepcopy(self.last_completed_trial)
        return evidence_context(controller_context(context))


class SeparatedMemoryRuntime(MemoryComparisonRuntime):
    memory_layout = 'separated'
