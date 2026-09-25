---
name: skill-creator
description: Turn acknowledged game experiences into a bounded executable skill candidate, or repair one using its failed trials.
---

Use retained experience IDs and their original observations. Infer only the demonstrated scope; the candidate game_id must equal the current game version. Propose a short parameterized procedure with propose_skill. Each step has a legal button, explicit original-pixel coordinates for CLICK, cell_is start guards, and measured after conditions (cell_changed, cell_is, color_count_delta or level_increased). At least one after condition must measure a change. Coordinate parameters use $x/$y; integer arguments range 0..63. A cell_is value is a color ID, not a guessed object name. color_count_delta.value selects a color whose count must change, not the amount of change.

Include examples mapping parameters to integers for actual complete contiguous traces. Every example must replay against the cited experiences. Evidence is provided by the host, not by a claimed successful test. Read missing grids using get_observation (which also returns the original color grid). Do not infer unexecuted outcomes from screenshots.

A draft returns an immutable skill_id to the experiment designer, which can select trial, evaluate or invoke. The builder only proposes a candidate or defers with missing evidence. Tests use real game actions and the normal budget. At least two distinct starting grids/arguments beyond the seed cases must succeed, with no failed or unknown trials. Then request kind=evaluate. The host owns the fixed evaluation and registration. If testing is infeasible, leave the candidate unpromoted. Revise failed candidates with parent_id and new supported conditions; never invent evidence or assert status=active. Reuse active procedures with kind=invoke and suspend on unexpected effects. The purpose is a useful observed local capability, not a universal causal claim.
