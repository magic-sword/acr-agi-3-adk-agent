# ARC-AGI-3 skill-learning agent

A local Google ADK 2.0 agent with a shared puzzle notebook and a `DECIDE ↔ RUN` loop.
It explores visual games through small subgoals and frozen experiments, reviews each result before
selecting another action, and proposes executable skill candidates from acknowledged experience. It
tests candidates through real game actions, and promotes only versions passing host evaluation.
Qwen3-VL-4B-Instruct runs locally. Without a model, the driver issues deterministic control probes.

See the [implementation and limits](docs/skill-learning-runtime-ja.md),
[experiment-loop design and research](docs/goal-experiment-loop-design-ja.md), and
[evaluation guide](docs/local-evaluation-ja.md).
The implemented task split is documented in [focused task design](docs/focused-task-refactor-design-ja.md).
The [object and question memory design](docs/object-world-memory-design-ja.md) records the motivation, evidence boundaries, references, and evaluation plan for the latest refactor.
Earlier design and validation records are clearly separated under [history](docs/history/README.md).

## Setup

Requires Docker Compose, NVIDIA Container Toolkit, and SSH forwarding for JupyterLab.
The development image uses Python 3.12+, pinned ADK/GenAI/ARC wheels in `third_party/wheels/`,
and the official ARC framework under ignored `vendor/`.
Run commands as your normal SSH user; Compose uses that UID/GID.

```bash
cp .env.example .env            # preserve an existing .env
make build
make setup
make check
make lab                       # localhost:8889 via SSH forwarding
make model-download            # first-time model download
make model-up
make model-check
```

Caches live in `.cache/model-cache/`. For files left by earlier root containers,
`make repair-perms` fixes ownership of the known generated paths.
No API key or internet is needed during local model inference.

## Run and evaluate

Open [agent_observatory.ipynb](notebooks/agent_observatory.ipynb) in JupyterLab after a benchmark.
Choose its evaluation ID and game from the dropdowns, click **読み込む**, then play the saved run.
A state diagram highlights experiment design, result review, skill construction, execution and observation waits.
The replay pairs each subgoal and pre-action prediction with its measured result and verdict.
Playback defaults to state transitions; action and full-event modes are also available.
Click **攻略ノートを読む** for readable goal paths, experiment notes, bookmarks, pages actually
read in that invocation, and before/after edits. Displayed notes keep their recorded versions.
Playback reads only the selected game, updates images only when needed, and opens detailed logs on demand.
It does not launch evaluations or poll for updates. See the [replay guide](docs/agent-monitor-ja.md).

```bash
make eval GAME=ls20 STEPS=20    # deterministic driver probe, no model
make eval-model GAME=ls20 STEPS=50
make test
make benchmark-prepare         # cache public games
make benchmark                 # evaluation limits from .env (defaults: 12 actions / 90 seconds)
make visualize                 # outputs/agent-visualization/index.html
```

`make benchmark` records a source snapshot, model/environment hashes, official SDK scores,
actual actions, observations, model requests and skill lifecycle events under `outputs/evaluations/`.
It never uploads or submits to Kaggle. Short public-game runs are not leaderboard estimates.

Set benchmark parameters in `.env` (plain `NAME=value` assignments):

```dotenv
EVAL_GAMES=ls20,vc33,ft09
EVAL_STEPS=12
EVAL_LEVELS=1
EVAL_SECONDS=600
EVAL_HARD_SECONDS=660
```

`EVAL_LEVELS=0` disables the cleared-level limit. `EVAL_HARD_SECONDS` must exceed
`EVAL_SECONDS`; it caps the entire worker process including initialization and cleanup.
Command-line assignments take precedence, for example `make benchmark EVAL_STEPS=80`.
Without `.env` settings, Makefile defaults apply. Each run's `manifest.json` records
the resolved games and limits for later comparison, including command-line overrides.
`make eval` is a deterministic driver probe; `make eval-model` plays with the model.
Use `make benchmark` for score measurement and comparison reports.

Learning starts with an empty procedural library per game. The model receives one focused task
at a time: goal selection/assessment, experiment design, target inspection, semantic effect judgment,
method selection, skill arguments, or skill construction. Each task has its own schema, limited
input snapshot and task ID. The shared versioned notebook remains the host-owned source of truth.
Pixel and level expectations are measured by code; these facts do not decide goal completion.
Clicks proposed by the experiment designer are checked against the described target before dispatch.
Unchanged failed tests return to goal assessment with a host-computed reason and plan delta.
Recovery is bounded per observation and does not reset when the goal is renamed.
Skill evaluation and promotion remain host-owned. The default 4 model calls / 8 HTTP requests per
observation are unchanged; goal replacement and recovery can exhaust this budget.
See the [implemented design and validation notes](docs/focused-task-refactor-design-ja.md).
Learned procedures are scoped to a game version; they are not Python or shell code.
The first implementation supports up to eight guarded steps and measurable pixel/level effects.
Every step yields to the driver for a fresh observation. Unexpected effects stop reuse.

Candidates need actual seed evidence, two successful distinct fresh trials beyond their seed
examples, negative guard checks and regression checks. Unknown results never count as success.
This validates limited observed effects, not a general causal law or game-solving ability.

```bash
# Explicit transfer evaluation: freeze a previously generated library.
docker compose run --rm --no-deps dev python scripts/benchmark_local.py \
  --games ls20 --no-learning --skill-library outputs/cognition/RUN/skills/library.json
```

Use an actual generated `library.json` path in place of the example. Imported libraries are
explicit inputs; normal benchmark trials do not share learning implicitly.

## Configuration and logs

- `COGNITION_MAX_CALLS=4`: judgments per observation, including semantic review and skill construction. Host-only checks do not spend model calls.
- `COGNITION_MAX_HTTP_REQUESTS=8`: shared HTTP budget per observation.
- `COGNITION_SECONDS=600`: total game runtime budget; benchmarks override it.
- `COGNITION_MAX_RESETS=2`: host-owned episode restarts.
- `COGNITION_LEARNING=1`: permit candidate creation/testing/evaluation.
- `COGNITION_SKILL_LIBRARY`: optional frozen library; unset by default.
- `COGNITION_LOG_DIR=outputs/cognition`: logging location; empty disables persistence.
- `VLM_API_BASE=http://vlm:8080/v1`: OpenAI-compatible local endpoint.

Game images contain the board and coordinate rulers. Button names are text metadata.
There is no host cursor or controller panel to mistake for game objects.
`CLICK` takes explicit original-pixel coordinates, with the host validating bounds and availability.
Historical frame tools never advance game time.

```bash
python3 scripts/analyze_agent.py outputs/cognition
```

The HTML viewer links experiments, their verdicts, actual execution, and skill learning.
`<run>.experiments.jsonl` records frozen plans, measurements, reviews and interruptions. `<run>.learning.jsonl`
records experience, drafts, trials, evaluation, promotion and suspension.
`<run>/skills/library.json` stores immutable versions and evaluation evidence.
`<run>.notebook.jsonl` records opened/read pages, revisions, withdrawals and bookmarks.
`<run>/notebook/` preserves each page version. Replay shows the pages known at that time.
Requests, tool execution, observations and driver acknowledgements have separate journals.
These are observable decision artifacts, not a reconstruction of private model reasoning.

## Offline notebook

```bash
make notebook                  # regenerate from agent source and skill resources
```

`notebooks/submission.ipynb` is generated; edit `agent/` instead. The notebook bundles source,
method skills and pinned ADK wheels. Kaggle reruns use the competition's offline framework
and model bundle. Local execution prepares the code; `make eval` plays locally.

`make push` explicitly builds and uploads the notebook when requested. `make status` checks
an existing kernel run. Authentication uses the local ignored `.kaggle/access_token`.
The present refactor does not run either publishing command.
