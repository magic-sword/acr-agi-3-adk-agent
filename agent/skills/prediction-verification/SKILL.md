---
name: prediction-verification
description: "Interpret an action outcome when delay, occlusion or intervening actions make predicted effects hard to verify; distinguish contradiction from unobservable evidence."
---
# Interpreting predicted outcomes

Use when the previous prediction is hard to judge from direct measurements alone.

1. Start from the previous action's source observation and prediction, including any expected delay or condition to preserve. Compare it with the current observation; retrieve intermediate frames only if order matters.
2. Examine each effect separately. A directly visible matching result supports it; an observable incompatible result contradicts it. A hidden region, missing baseline or unresolved delay leaves it unknown. Failure to observe an effect before its deadline is not automatically refutation.
3. Check invariants independently from effects. Reaching a target while destroying a required support may satisfy one prediction and still invalidate the plan.
4. Check intervening actions and mode/level boundaries. An observed outcome after multiple actions does not isolate which action caused it. A reset layout is not evidence for the previous action's effect.
5. Record observed changes and supported/refuted assumptions in reflection and the working notebook. Distinguish your interpretation from host measurements; a changed screen does not establish progress or WIN.

Example: a gate opens two actions after pressing a switch, but the second action also touches the gate. Report the open gate as a current fact. Keep switch-caused opening and contact-caused opening unresolved until a discriminating observation exists.

Use the review to choose the next action in the same decision. Identify the failed assumption or precise evidence still missing.
