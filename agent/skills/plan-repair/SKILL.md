---
name: plan-repair
description: "Repair a plan after an assumption, invariant or subgoal fails; preserve useful achieved conditions and rebuild only the remaining feasible dependencies."
---
# Repairing the remaining plan

Use when an existing plan cannot continue because a condition or supporting hypothesis changed.

1. Identify the earliest affected node and its dependents. Separate a broken prerequisite from a contradicted effect or changed objective. Inspect the evidence behind the failure; uncertain observation calls for verification before replacement.
2. Recheck achieved conditions against the current frame. A previously completed node is not proof its result still holds after intervening actions. Preserve results that remain visible and useful.
3. Find an alternate way to establish the failed prerequisite without undoing useful results. UNDO is a legal input only when available; its actual reversing effect needs evidence. A reset is not a generic repair operation.
4. Rebuild the remaining graph, reusing unaffected intentions where feasible. Check dependencies, invariants and timing again. The submitted `plan` replaces the stored plan; include all still-needed future nodes, not just a patch. Omit already achieved nodes and satisfy their former prerequisites from current evidence. New nodes must be `todo`, never claim `done` yourself.
5. Explain the changed assumption in `diagnosis` and reference its evidence. If no first action is justified, identify the precise uncertainty that blocks repair rather than resubmitting the failed plan.

Example: moving a support opened the route but removed access to a switch. Preserve the open route if it remains useful; find a staging position that permits switch access. Repeating the same support movement without a new prediction is not a repair.
