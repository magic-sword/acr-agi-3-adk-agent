---
name: prediction-verification
description: "Compare predicted effects and invariants to real observations. Separate delayed, unobservable and contradicted effects."
---
# Prediction Verification (S03)

1. Inspect the current observation and supplied memory. Identify evidence relevant to this skill.
2. Compare predicted effects and invariants to real observations. Separate delayed, unobservable and contradicted effects.
3. Separate visible facts from hypotheses. Refer only to observation IDs supplied in the input.
4. If evidence is insufficient, report a specific unknown or a distinguishing experiment; do not invent a fact.
5. Return the requested state's JSON contract. A skill must not directly execute game actions or modify budgets.

Read `references/evidence-contract.md` with `load_skill_resource` when checking evidence, coordinates or prediction semantics.

## Evidence-driven VERIFY

OBSERVE only captures evidence. In VERIFY, retrieve the pending action's `observation_id`
and the current ID using `compare_observations`; inspect `get_observation` or recorded
animation when needed. More than one intervening action prevents attributing the whole
change to a single action. Pixels changing in a status display do not prove target progress.
Return Interpretation, including current facts, hypotheses/unknowns as warranted, and a
short summary with evidence_refs. The host evaluates predicates using these facts and
merges the validated review during UPDATE. Empty or ambiguous evidence may remain unknown.
No pending action on the first frame means there is no previous outcome to verify.
