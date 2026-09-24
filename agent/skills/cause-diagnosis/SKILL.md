---
name: cause-diagnosis
description: "Diagnose an unexpected action result by separating wrong input or target, object identity, delay, mode changes and incorrect dynamics before revising a plan."
---
# Diagnosing a contradiction

Use when an expected effect fails or an invariant breaks. Start with the specific discrepancy, not a new global theory.

1. Check the executed action record against the intended input. For CLICK, inspect its recorded coordinates, not the cursor's later position. Confirm that the target and control were actually applicable.
2. Compare the source and result observations. Check object identity and whether the relevant region was visible. A rendering overlay or hidden target can explain an apparent failure without changing the game rules.
3. Check the delay window and any intervening actions. Attribute the discrepancy to a single action only when the evidence isolates it.
4. Check whether a level, control mode or relevant condition changed. Prefer the smallest revision that explains the observations, but keep equally plausible alternatives when evidence cannot discriminate.
5. Link the diagnosis to observations and affected hypotheses. Preserve unrelated knowledge. Propose a repaired plan only if its assumptions hold; otherwise test the unresolved cause.

For worked input-error versus dynamics examples, read [references/diagnosis.md](references/diagnosis.md) when the discrepancy fits more than one cause.

If evidence is missing, the result of diagnosis is an explicit uncertainty. Do not invent a transport error or trigger RESET to explain an unexpected game outcome.
