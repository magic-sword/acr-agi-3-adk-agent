"""Small model contracts. Task IDs bind answers to host-owned input snapshots."""
from typing import Literal
from pydantic import Field, model_validator
from .state import Contract, Action, Expectation, Region, Draft


class Answer(Contract):
    task_id: str = Field(min_length=1)


class GoalSelection(Answer):
    text: str = Field(min_length=1, max_length=200)
    goal_type: Literal['knowledge', 'progress']
    done_when: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=250)


class GoalAssessment(Answer):
    decision: Literal['continue', 'completed', 'replace', 'deferred']
    reason: str = Field(min_length=1, max_length=300)
    remaining_question: str = Field(default='', max_length=200)
    evidence_ids: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode='after')
    def question(self):
        if self.decision == 'continue' and not self.remaining_question.strip():
            raise ValueError('continuing requires one remaining question')
        return self


class ExperimentDesign(Answer):
    action: Action
    target: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=200)
    hypothesis: str = Field(min_length=1, max_length=250)
    conditions: str = Field(min_length=1, max_length=200)
    expected: Expectation
    context_region: Region | None = None
    max_attempts: int = Field(default=1, ge=1, le=3)
    repeat_reason: str = Field(default='', max_length=200)
    retry_of: str = Field(default='', max_length=80)


class TargetInspection(Answer):
    region: Region | None = Field(description='Actual target bounding rectangle. width/height are cell counts, NOT right/bottom coordinates. x+width and y+height must fit the image. Null if uncertain.')
    finding: str = Field(min_length=1, max_length=300)


class EffectJudgment(Answer):
    verdict: Literal['supported', 'unsupported', 'inconclusive']
    finding: str = Field(min_length=1, max_length=400)
    evidence_ids: list[str] = Field(min_length=1, max_length=4)


class MethodChoice(Answer):
    method: Literal['explore', 'invoke', 'trial', 'learn']
    skill_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    reason: str = Field(min_length=1, max_length=250)

    @model_validator(mode='after')
    def selected(self):
        if (self.method in ('invoke', 'trial')) != (self.skill_id is not None):
            raise ValueError('only invoke/trial specify a skill ID')
        if self.method == 'learn' and not self.evidence_ids:
            raise ValueError('learning requires acknowledged experiences')
        return self


class SkillArguments(Answer):
    arguments: dict[str, int]


class SkillDraft(Draft):
    task_id: str = Field(min_length=1)


class TaskFailure(Answer):
    """Host-only terminal result after two unsuccessful correction opportunities."""
    reason: str


class MissingEvidence(Answer):
    reason: str = Field(min_length=1, max_length=300)


TASKS = {
    'select_goal': ('submit_goal', GoalSelection),
    'assess_goal': ('submit_goal_assessment', GoalAssessment),
    'design_experiment': ('submit_experiment', ExperimentDesign),
    'inspect_target': ('submit_target', TargetInspection),
    'judge_effect': ('submit_effect', EffectJudgment),
    'choose_method': ('submit_method', MethodChoice),
    'resolve_arguments': ('submit_arguments', SkillArguments),
    'skill_creation': ('propose_skill', SkillDraft),
}

INSTRUCTIONS = {
    'select_goal': 'Select ONE small goal from current evidence. For exploration ask a knowledge question that can be answered negatively. Name a specific visible object or relationship to investigate; avoid broad goals such as identify the game goal. Do not choose an action or invent the complete solution. Return a completion condition that one local experiment can answer.',
    'assess_goal': 'Decide ONLY whether the current goal continues, is completed, should be replaced, or lacks evidence. Compare its original completion condition with the supplied facts. A failed experiment need not invalidate the goal. If continuing, name ONE unresolved question. Unknown game rules are a reason to explore, not missing evidence. Use deferred ONLY when actual observations or execution receipts needed for judgment are unavailable. If the current question cannot be answered use replace for a smaller investigable question. Do not choose an action or copy an earlier review.',
    'design_experiment': 'Design ONE informative experiment for the fixed goal and remaining question. You cannot change the goal. Use the current image and any target assessment to choose an actual legal action and an observable expectation. Repeating an unchanged failed test is not a redesign. The target must describe a visible object by appearance and location, never restate the goal. A one-pixel check cannot establish an entire object\'s response. Remote effects may need a different observation region. Declare bounded retries in advance only for a specific delay/stochastic hypothesis.',
    'inspect_target': 'Locate ONLY the described visible object in the original image. Return its actual bounding rectangle, independently of coordinates claimed by the designer; the description may be inaccurate. width and height are cell counts, not endpoint coordinates. If the object cannot be identified return region=null. Do not judge whether the click hits it: the host computes containment. Do not execute or choose actions or infer game rules.',
    'judge_effect': 'Judge ONLY the frozen semantic expectation against the actual acknowledged before/after evidence. Report supported, unsupported or inconclusive with the actual experience ID. Do not change the expectation, decide a goal, or choose an action. A change elsewhere is not evidence that the target responded.',
    'choose_method': 'Choose ONLY the method for the current goal: explore, reuse an active skill, trial a candidate, or build from the supplied acknowledged experiences. Do not choose click coordinates or skill arguments. Choose explore if evidence is insufficient for a reusable procedure.',
    'resolve_arguments': 'Resolve ONLY the integer arguments of the selected skill for the current image and goal. Do not choose another skill or change its procedure. Report missing evidence if you cannot ground the arguments.',
    'skill_creation': 'Build or repair ONE skill from the selected acknowledged experiences. Propose a candidate matching the actual trace; never invent outcomes. Report missing evidence if construction is unsupported. You cannot execute actions or promote a skill.',
}

COMMON = 'Use only the supplied evidence. Images use original pixels: x=column, y=row. Interpretations may be wrong. Complete only the assigned task and return its exact task_id. Report missing evidence instead of guessing.\n'
