---
name: backward-planning
description: "Build a short conditional plan from a supported success condition when prerequisites, staging space or noncommuting actions make greedy progress unreliable."
---
# Planning from prerequisites

Use when a supported goal needs prerequisites, staging space or a particular operation order.

## Build and check candidate orders

1. Work backward from an unsatisfied goal condition. Identify an evidence-supported operation that establishes it and the access/conditions that operation requires. Stop at an unknown effect and formulate a test; do not invent an operation to complete a path.
2. Identify conditions that each operation removes as well as those it establishes. Conditions already achieved can be destroyed. Goal conditions are not automatically prerequisites of one another; their order in the goal description carries no ordering requirement.
3. **For interacting prerequisites, call `check_plan_order` before selecting an order.** Supply `initial_conditions`, ordered `steps` (id, requires, adds, removes), and `required_final_conditions`. Represent access explicitly. “A blocks the only access to B” means that access is in A.removes and B.requires.
4. Inspect the returned `blocked_step` and `missing_conditions`. A rejected order is not executable under those assumptions. Try another order while preserving the stated effects. If both directions fail, identify a staging operation or alternative assumption to investigate. A successful result only checks the assumptions supplied, not their truth in the game.

Example of the check's input shape (symbolic assumptions, not a game action):

```json
{"initial_conditions":["access_a","access_b"],"steps":[{"id":"A","requires":["access_a"],"adds":["a_done"],"removes":["access_b"]},{"id":"B","requires":["access_b"],"adds":["b_done"]}],"required_final_conditions":["a_done","b_done"]}
```

This order fails at B. Reversing it works only if B leaves A's prerequisites intact. If B also removes access_a, neither order works without another operation. Adapt the conditions to the evidence; never change a known effect just to make the checker pass.

## Turn a feasible order into a conditional plan

Keep a short ordered list of remaining subgoals in the working notebook. Note completion evidence and conditions that must survive later actions. Submit only the first legal action and its prediction; check the actual result before choosing the next step. A formal subgoal graph is unnecessary.

Count prerequisite actions and checkpoints against the budget. CLICK actions sample the current cursor, so distinct click targets require new positioning and observation. If no first action is justified, ask a precise question or propose an experiment instead of an impossible plan.

Read [references/dependencies.md](references/dependencies.md) for staging and plan-repair details when needed.
