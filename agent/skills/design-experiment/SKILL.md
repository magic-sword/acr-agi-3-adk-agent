---
name: design-experiment
description: Select a small knowledge subgoal and predeclare one observable experiment before acting.
---

Read the current goal path and latest verdict. Keep or create one small subgoal with a
completion condition; use its ID unchanged to continue it. Parent it under goal or an active
subgoal. Ask what missing information will help that goal. State a conditional hypothesis
and choose one action whose result answers a specific question.
For exploration, use a question that can be answered negatively, not a requirement to make
one chosen object react. Keep game progress distinct from acquiring evidence.

After a review, read notebook.handoff before drafting the experiment. Check whether the
intended object was actually targeted, whether the observed scope could detect the effect,
and whether a prerequisite was verified. Use the original image or evidence tools when unsure;
screen center is not an object's coordinate. Do not infer an object is unresponsive from a missed click.
If revision_required, cite the exact experiment and review version in experiment.revision.
State the reconsidered assumption and why the changed action, observation_scope or context
answers the remaining question. The host checks the declared change against the actual plan
and observed conditions; rewording a hypothesis or changing only its criterion kind is insufficient.
A predeclared retry uses retry_of and preserves the entire original plan, including its revision.

Declare expected.kind=region_changed only when any pixel change in that rectangle is the
actual question. It measures change, not causation or success. Use level_increased for an
observed level increment. Use semantic for meaning (e.g. the object moved toward the door),
with a specific description that the reviewer can support or refute. Specify the relevant
region, which may include remote effects. If prerequisites are elsewhere, identify them
with context_region so a real change there permits a renewed test. Do not use a changing status strip as evidence
that the target reacted. All expectations refer to the next returned observation.

The host saves the subgoal, hypothesis, action and criteria automatically. Do not separately
copy them into free-text notes. An unsupported result applies only to the tested conditions.
Change the target, action or informative scope after no support. Rephrasing the same test
is not a new experiment. For delay/stochastic hypotheses, predeclare max_attempts (at most
three) and repeat_reason before the first action; retry_of preserves that plan and limit.
Inspect original evidence if a coordinate or precondition is uncertain.
