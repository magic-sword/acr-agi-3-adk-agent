# Symbolic action models

A rule has two explicit sets for preconditions and effects:

```json
{"id":"open-gate","action_name":"activate(switch1)","actions":[{"action":"ACTION5"}],"preconditions":{"true":["powered(switch1)"],"false":["gate_open"]},"effects":{"true":["gate_open"],"false":["switch_accessible"]},"scope":"level"}
```

This is a hypothesis supplied to `causal_memory` in `rules`, not a verified game rule. The false effect means access is lost: a later action requiring that access cannot simply be appended. Concrete action names are ACTION1..ACTION7; ACTION6 needs explicit original-pixel x/y in stored rules. RESET is never a learned game action. Executing CLICK still uses the live host cursor, so position it before submission.

A positive trial makes `gate_open` visibly change from false to true while powered. A negative trial leaves it visibly false. If both trials were powered, power does not explain the difference; inspect target, mode, delay or an unobserved prerequisite. If the negative trial was explicitly unpowered, the learner can infer a separating powered precondition. If power was unobserved, it cannot treat that trial as unpowered.

If the effect was already true and stayed true, that is not evidence that this input established it. Observing the effect alone also does not prove a causal law: autonomous motion and partial visibility remain possible confounders. `confounded` trials are retained but excluded from learning/support counts.

The learner groups examples by abstract action name, exact concrete input sequence and scope. `at(player,door)` is a ground atom, not a Prolog expression to evaluate. Object-variable lifting, arithmetic and recursive predicate invention are not implemented. Choose a shared symbolic vocabulary and record multiple contexts before generalizing.

The host archives up to 128 raw observations per full game ID. A saved transition retains its source frames and all intervening inputs beyond raw-archive eviction. Rules and up to 1024 interpreted transitions persist in SQLite under CAUSAL_MEMORY_DIR. Model notes cannot change those original source frames. Database files are runtime data and are not included in the submission notebook.
