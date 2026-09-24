---
name: goal-inference
description: "Infer or revise uncertain success conditions from outcome evidence, including relational, sequential and simultaneous goals when a plausible visual solution does not win."
---
# Inferring success conditions

Use when the objective is unknown, several objectives fit the scene, or a seemingly completed objective has not produced success.

1. List plausible conditions suggested by evidence: occupancy of a region, alignment, a relation between objects, a sequence, or several conditions simultaneously. Visual salience or resemblance alone supplies a candidate, not a confirmed goal.
2. Check past actions and level/outcome signals. Ask which candidate predicts an observation that another does not. Separate learning a control from learning the goal; unknown goals do not prevent a useful control experiment.
3. If a plausible condition is visibly satisfied without WIN, consider a missing conjunct, wrong object identity, unobserved region or delayed completion. Do not immediately discard established dynamics.
4. Choose a short plan if its next step is justified across remaining candidates. Otherwise formulate an inquiry or experiment targeting the condition that changes the next decision.
5. Keep the proposed condition and its evidence in the working notebook, marked as a hypothesis. Revise or withdraw it when evidence changes; retain unrelated control knowledge and still-useful subgoals.

Example: placing a token on a pad does not win. Candidate A requires any token; candidate B requires the matching token; candidate C also requires the exit to remain open. Inspect matching attributes and exit state before treating the pad as irrelevant.

Only the environment's WIN confirms full completion. A revised goal and the next action can be selected together.
