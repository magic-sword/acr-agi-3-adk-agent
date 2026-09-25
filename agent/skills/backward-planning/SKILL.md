---
name: backward-planning
description: "Search persistent causal models backward from stage completion or another symbolic goal, returning a forward-checked action chain or missing prerequisites to investigate."
---
# Backward planning from learned rules

Use when causal parts have accumulated, a goal needs prerequisites or action order matters. [scripts/regression.py](scripts/regression.py) performs bounded regression through the typed `plan_backward` tool; the three-node game workflow stays unchanged.

1. Ground the CURRENT state in the latest observation. Use the same symbols as stored rules, explicitly distinguishing true, false and unobserved. Start planning from where you are now; use the game-start state only when still there.
2. Call `plan_backward` with the current `observation_id`, `current` and `goal`. The default goal is `stage_clear=true`, whose observed training labels come only from the environment. Inspect/revise causal knowledge with `causal_memory` if needed.
3. On `status=plan`, inspect the rule chain, its evidence and forward state trace. The engine accounts for false/delete effects and interactions between conjunctive subgoals. Validity is conditional on the supplied symbolic models and on unaffected facts persisting. Execute only the next primitive input, after checking its current target and legal controls; observe and replan after it. A stored macro is not permission to replay every input blindly.
4. On `knowledge_gap`, inspect each frontier's missing conditions, unknown current facts, `no_known_producer`, conflicts and excluded rules. An unknown current fact suggests observation. A known-unsatisfied prerequisite with no producer suggests an experiment to discover an action effect. A candidate/conflicting rule suggests a discriminating trial. Do not convert the missing condition into an assumed fact.
5. On `search_limit`, the search is unfinished. Narrow the immediate subgoal or use another bounded search; this result does not establish missing causal knowledge or impossibility.

Example (replace the observation ID and current symbols):

```json
{"observation_id":"CURRENT_OBS","current":{"true":["at(player,switch)","powered(switch1)"],"false":["gate_open"]},"goal":{"true":["stage_clear"]}}
```

`allow_candidates=true` can explore hypothetical chains for experiment design, but such a path is `tentative_plan`. Refuted rules are always excluded. Overlapping models for the same concrete input with different effects are excluded rather than choosing the favorable outcome.

Recording examples and learning can share one `causal_memory` update; `plan_backward` can follow in the second request, leaving the third for `submit_decision`. Optional resource loading is not a required step each turn.

For testing a specific hypothetical order without persistent rules, `check_plan_order` remains available. It checks supplied adds/removes and prerequisites, not the truth of effects. Read [references/dependencies.md](references/dependencies.md) for staging and interference examples when needed.
