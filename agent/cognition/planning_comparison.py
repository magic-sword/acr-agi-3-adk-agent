"""Opt-in planning experiment; the default runtime is unchanged.

A: existing stages with the common evidence/image presentation below.
B: one-action planning and a separate review, sharing execution and perception.
"""
from copy import deepcopy
import hashlib
import json
import os

from google.adk.agents import LlmAgent
from google.genai import types
from pydantic import Field

from agent.controls import ACTION_TO_BUTTON, controller_context
from agent.local_vlm import LocalVisionLlm
from .deliberation import MEASUREMENT_INSTRUCTION, StageTool
from .perception import context_record
from .state import Contract, ActionIntent, Grounding, Reconciliation
from .validation import validate_intent
from .workflow import CognitiveRuntime


class FocusTarget(Contract):
    appearance: str = Field(min_length=1, max_length=160)
    role_hypothesis: str = Field(max_length=180)
    candidate_refs: list[str] = Field(default_factory=list, max_length=12)


class OneActionPlan(Contract):
    observation_id: str
    goal_hypothesis: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=180)
    targets: list[FocusTarget] = Field(min_length=1, max_length=6)
    baseline: str = Field(min_length=1, max_length=180)
    action: ActionIntent
    expected_effect: str = Field(min_length=1, max_length=180)
    rationale: str = Field(min_length=1, max_length=200)


class ActionReview(Contract):
    observation_id: str
    assessment: str = Field(pattern='^(supported|contradicted|unclear)$')
    evidence: str = Field(min_length=1, max_length=300)
    causal_notes: str = Field(max_length=600)
    next_question: str = Field(min_length=1, max_length=240)


PLAN_INSTRUCTION = '''Plan ONE next action using the measured candidates, before/current images,
available controls, recent_trials and last_review. Interpret relevant targets and
prerequisites within this call; their roles and the game goal remain hypotheses.
You do not need a complete object taxonomy or known game rules before a probe.
State one concrete uncertainty, the visible baseline and an expected visible effect.
Prefer an action whose outcome helps distinguish the remaining explanations or advances
a supported small goal. A display change alone does not prove the target changed.
Use prior failures when selecting what to test next; explain a repeated test if needed.
For CLICK describe the visible instance and part in target_query, without coordinates;
the existing cursor will locate it. Other actions need no target_query.
Bind targets to current candidate_refs where supported, otherwise use [].
Submit submit_one_action. This call chooses the action, not its observed result.'''

REVIEW_INSTRUCTION = '''Compare the actual result with the previous action's expected effect
and baseline. Use the before/current images and measurements. Distinguish target change,
target unchanged and uncertain correspondence. A change elsewhere is not target progress.
An unexecuted or unacknowledged action cannot support an effect claim. Prior invocation
results do not belong to a new attempt. Keep observation facts separate from causal
explanations; temporal proximity or co-motion alone is not causal proof. No visible
change in this interval does not prove the control never has an effect.
Return supported, contradicted or unclear, concise evidence, remaining causal hypotheses
and the specific next question. Do not plan an action or assert game victory.
Submit submit_action_review.'''


def evidence_context(context):
    """Lossless interning of repeated measurement dictionaries in this request.

The original logs are not modified. Hash references are scoped to this request;
different observations or summaries are never merged merely by candidate ID.
"""
    measurements = {}

    def walk(value):
        if isinstance(value, list):
            return [walk(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key == 'measured_objects' and isinstance(item, dict):
                encoded = json.dumps(item, sort_keys=True, separators=(',', ':'))
                ref = hashlib.sha256(encoded.encode()).hexdigest()[:16]
                measurements[ref] = deepcopy(item)
                result[key] = {'measurement_ref': ref}
            else:
                result[key] = walk(item)
        return result

    result = walk(context)
    result['measurements'] = measurements
    result['evidence_format'] = 'measurement_ref refers to measurements in THIS request; facts are stored once.'
    return result


class ComparisonRuntime(CognitiveRuntime):
    """A control: unchanged decisions, common input presentation for both arms."""
    comparison_arm = 'A'

    def _context(self, work=None):
        work = work or self.work
        context = super()._context(work)
        if self.perception and work in ('understand', 'backchain', 'ground', 'reconcile'):
            # Both planners can access the full candidate index, not a ranked prefix.
            context['measured_objects'] = context_record(self.perception, inventory=True)
        return evidence_context(context)

    def _visual_parts(self, slow=False):
        # Identical image policy, independent of stage count and review heuristics.
        views = []
        if self.work != 'aim' and self.previous and self.outcome and not self.outcome.get('boundary'):
            views.append(('BEFORE the last action', self.previous))
        views.append(('CURRENT board', self.obs))
        parts = []
        for label, obs in views:
            parts.append({'type': 'text', 'text': label})
            if obs.get('image_png_base64'):
                parts.append({'type': 'image_url', 'image_url': {
                    'url': 'data:image/png;base64,' + obs['image_png_base64']}})
            else:
                parts.append({'type': 'text', 'text': json.dumps(obs.get('grid'))})
        return parts


class OneActionTool(StageTool):
    def __init__(self, runtime, work):
        super().__init__(runtime, work)
        self.contract = OneActionPlan if work == 'ground' else ActionReview
        self.name = 'submit_one_action' if work == 'ground' else 'submit_action_review'

    def _get_declaration(self):
        schema = self.contract.model_json_schema()
        schema['properties']['observation_id']['enum'] = [self.runtime.obs['observation_id']]
        if self.work == 'ground':
            action = schema['$defs']['ActionIntent']
            allowed = [ACTION_TO_BUTTON[a] for a in self.runtime.obs['available_actions']
                       if a in ACTION_TO_BUTTON and a != 'RESET']
            action['properties']['action']['enum'] = allowed
            if 'CLICK' not in allowed:
                action['properties'].pop('target_query')
            elif allowed == ['CLICK']:
                action['properties']['target_query'] = {'type': 'string', 'minLength': 1, 'maxLength': 300,
                    'description': 'Visible instance and part to click, described by appearance and relations; no coordinates.'}
                action['required'] = ['action', 'target_query']
        return types.FunctionDeclaration(name=self.name, description=self.description, parameters_json_schema=schema)


class OneActionRuntime(ComparisonRuntime):
    comparison_arm = 'B'

    def _on_observation(self, boundary):
        super()._on_observation(boundary)
        if self.memory.phase == 'understand':
            self.memory.phase = 'ground'
            self._snapshot()

    def _machine_transition(self, event):
        if event == 'received':
            self._record('artifacts', 'machine_transition', transition='initial_one_action',
                         source='observe', target='ground', condition='Initial one-action plan', category='normal')
        else:
            super()._machine_transition(event)

    def _route_perception(self):
        # Raw recognition is unchanged. The separate review interprets all effects;
        # do not pre-interpret them with the extra answer_question model calls.
        return

    async def _read_memory(self, work):
        # Fixed recent history is supplied directly, without a model navigator.
        self.memory.memory_brief = {}

    def _context(self, work=None):
        work = work or self.work
        if work not in ('ground', 'reconcile'):
            return super()._context(work)
        m = self.memory
        context = dict(work=work, observation_id=self.obs['observation_id'],
            remaining_actions=self.obs['remaining_actions'], seconds_left=round(self.time_left(), 2),
            available_actions=self.obs['available_actions'], question=m.handoff_question,
            correction=self.rejection, measured_objects=context_record(self.perception, inventory=True),
            last_result=self.outcome, last_review=m.reconciliations[-1:],
            recent_trials=self.recent_trials[-3:], reason=self.replan_reason)
        if work == 'reconcile':
            context.update(plan=m.plan, review=m.review, understanding=m.understanding,
                           current_invocation_result=self._current_result())
        return evidence_context(controller_context(context))

    def _agent(self, work):
        if work not in self.planners:
            tool = OneActionTool(self, work)
            self.planners[work] = LlmAgent(name=work, include_contents='none',
                model=LocalVisionLlm(model='qwen3-vl-4b-instruct',
                    api_base=os.getenv('VLM_API_BASE', 'http://vlm:8080/v1'),
                    max_output_tokens=1000, max_requests=1, completion_tools=(tool.name,)),
                instruction=(PLAN_INSTRUCTION if work == 'ground' else REVIEW_INSTRUCTION)+MEASUREMENT_INSTRUCTION,
                tools=[tool], before_tool_callback=self._before_tool,
                after_tool_callback=self._after_tool, on_tool_error_callback=self._tool_error)
        return self.planners[work]

    def validate_stage(self, work, value):
        if isinstance(value, (OneActionPlan, ActionReview)):
            if value.observation_id != self.obs['observation_id']:
                raise ValueError('stale observation_id')
            if isinstance(value, OneActionPlan):
                validate_intent(value.action, self.obs)
                ids = {c['id'] for c in self.perception['candidates']}
                if any(ref not in ids for t in value.targets for ref in t.candidate_refs):
                    raise ValueError('target references unknown current measured candidate')
            elif self.memory.review is None or self.memory.plan is None:
                raise ValueError('review requires a current attempt')
        else:
            super().validate_stage(work, value)

    def _accept_stage(self, work, value):
        self.validate_stage(work, value)
        original = value.model_dump()
        if isinstance(value, OneActionPlan):
            targets = [t.model_dump() for t in value.targets]
            understanding = dict(observation_id=value.observation_id, targets=targets,
                                 goal_hypothesis=value.goal_hypothesis)
            procedure = dict(name='one-action', when_to_use=value.baseline, effect=value.expected_effect,
                steps=[dict(purpose=value.question, continue_when='This probe has not been executed.',
                    done_when='One acknowledged action result is available.',
                    reconsider_when='Target or action applicability is unclear.',
                    options=[dict(action=value.action.model_dump(), expected_effect=value.expected_effect)])])
            translated = Grounding.model_validate(dict(observation_id=value.observation_id,
                next='execute', reason=value.rationale, plan=dict(goal_id=None, intent='probe',
                    question=value.question, target_query='; '.join(t.appearance for t in value.targets)[:300],
                    baseline=value.baseline, skills=[procedure], reuse=[])))
            super().validate_stage(work, translated)
            self.memory.understanding = understanding
            if self.proposal_perception is not None:
                self.proposal_perception.bind_targets(understanding)
            super()._accept_stage(work, translated)
        elif isinstance(value, ActionReview):
            translated = Reconciliation.model_validate(dict(observation_id=value.observation_id,
                goal_id=None, assessment='probe_result', evidence=value.evidence, goal_status='unknown',
                causal_notes=value.causal_notes, next='ground', reason='Plan one action using the reviewed result.',
                next_question=value.next_question))
            super()._accept_stage(work, translated)
            self.memory.reconciliations[-1]['effect_assessment'] = value.assessment
        else:
            raise ValueError('unexpected comparison stage output')
        self._record('artifacts', 'comparison_stage_accepted', work=work, result=original, arm='B')

    async def _fast(self, work, options, *, timeout=None):
        if work == 'choose_skill':
            # The one-action planner already selected its sole procedure.
            if len(options) != 1:
                raise ValueError('one-action experiment expects one procedure')
            self._record('artifacts', 'sole_procedure_selected', author='host')
            return next(iter(options.values()))
        return await super()._fast(work, options, timeout=timeout)
