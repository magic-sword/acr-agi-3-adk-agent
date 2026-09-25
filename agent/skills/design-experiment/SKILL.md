---
name: design-experiment
description: Select a small knowledge subgoal and predeclare one observable experiment before acting.
---

Read the current goal path and latest verdict. Keep or create one small subgoal with a
completion condition; use its ID unchanged to continue it. Parent it under goal or an active
subgoal. Ask what missing information will help that goal. State a conditional hypothesis
and choose one action whose result answers a specific question.

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
