---
name: cause-diagnosis
description: "Learn, inspect and revise persistent causal rules from symbolic interpretations of actual action outcomes, including negative examples, hidden conditions and failed predictions."
---
# Learning causal action models

Use when an action reveals a reusable effect or contradicts a stored rule. The executable learner is [scripts/induction.py](scripts/induction.py), called by the typed `causal_memory` tool. Do not run arbitrary Python or write database files yourself.

## Observe, symbolize, learn

1. Compare actual BEFORE and AFTER evidence and the executed inputs. Use stable ground atoms such as `powered(switch1)` or `gate_open`; keep names consistent across trials. `true` and `false` are explicit observations. Omit hidden or ambiguous facts: absence is unknown, not a negative example.
2. Name the tested abstract action and select its source observation IDs. The host obtains concrete inputs from the intervening recorded actions. A multi-action interval describes that entire macro, not proof that one of its component inputs caused the effect. Mark `confounded=true` if autonomous motion, unclear delay or ambiguous object identity prevents training.
3. Call `causal_memory` with `operation=update`, `transitions` and `induce=true`. Recording and learning happen together. The host binds `stage_clear` to actual level increase/WIN. At completion, describe only `stage_clear` in AFTER, not objects on the next board.
4. Inspect `rules`, their positive/negative/unknown evidence IDs and `learning.unresolved`. `supported` means consistent with recorded symbolic examples, not proven causality or a universally valid rule. Positive-only learning retains the observed context. Contradictory examples may require a missing condition, separate mode or better perception.
5. Keep a rule's default `scope=level` unless the same mechanism has evidence across levels. `scope=game` shares it only within the same full game/version ID. Do not transfer coordinates or object identity without revalidation.

Example (replace IDs and atoms with actual evidence):

```json
{"operation":"update","transitions":[{"id":"switch-trial-1","before_id":"OBS_BEFORE","after_id":"OBS_AFTER","action_name":"activate(switch1)","before":{"true":["powered(switch1)"],"false":["gate_open"]},"after":{"true":["gate_open"]}}],"induce":true}
```

## Inspect, correct, retract

- `causal_memory({})` reads a page of current-level/game rules and trial summaries. Filter with `action_name`, paginate with `offset` and `limit`. `observation_ids` retrieves up to two retained raw records, including final outcomes from earlier runs.
- Replace a transition by its `id` to correct symbolic interpretation; keep the same original observation IDs. Its concrete action evidence cannot be edited. Relearning and rule evidence assessment use the corrected examples.
- `rules` upserts an explicit hypothesis with `id`, `action_name`, concrete `actions`, `preconditions`, `effects` and `scope`. Unsupported hypotheses remain candidates. Their status is computed from examples, not supplied by you.
- Use `delete_rule_ids` to retract a bad rule. Deletions are persistent tombstones so induction cannot immediately recreate the identical rule; an explicit upsert can restore it.
- Use `delete_transition_ids` only for mistaken interpretations. A genuine negative trial is valuable evidence; preserve it and refine the rule. Removing the only positive example downgrades the rule to a candidate.

The finite learner searches conjunctions of ground observed predicates, not arbitrary first-order or recursive programs. Failure to separate examples means the supplied vocabulary or search bounds are insufficient; do not fabricate a condition to make a rule fit.

For input error, delay and mode-change diagnosis, read [references/diagnosis.md](references/diagnosis.md) when those alternatives matter. For formats and a full learning-to-planning example, read [references/symbolic-model.md](references/symbolic-model.md).
