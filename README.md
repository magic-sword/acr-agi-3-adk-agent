# ARC-AGI-3 skill-learning agent

A local Google ADK 2.0 agent with one decision controller and a `DECIDE ↔ RUN` loop.
It explores visual games, proposes executable skill candidates from acknowledged experience,
tests candidates through real game actions, and promotes only versions passing host evaluation.
Qwen3-VL-4B-Instruct runs locally. Without a model, the driver issues deterministic control probes.

See the [implementation and limits](docs/skill-learning-runtime-ja.md),
[design and research](docs/autonomous-skill-learning-design-ja.md), and
[evaluation guide](docs/local-evaluation-ja.md).
See the [migration validation results](docs/skill-learning-validation-ja.md) for verified behavior and remaining model limitations.

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
Playback reads only the selected game, updates images only when needed, and opens detailed logs on demand.
It does not launch evaluations or poll for updates. See the [replay guide](docs/agent-monitor-ja.md).

```bash
make eval GAME=ls20 STEPS=20    # deterministic driver probe, no model
make eval-model GAME=ls20 STEPS=50
make test
make benchmark-prepare         # cache public games
make benchmark                 # three games, 12 actions / 90 seconds each
make visualize                 # outputs/agent-visualization/index.html
```

`make benchmark` records a source snapshot, model/environment hashes, official SDK scores,
actual actions, observations, model requests and skill lifecycle events under `outputs/evaluations/`.
It never uploads or submits to Kaggle. Short public-game runs are not leaderboard estimates.

Learning starts with an empty procedural library per game. Two fixed method skills,
`design-experiment` and `skill-creator`, are loaded on demand through ADK.
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

- `COGNITION_MAX_CALLS=4`: judgments per observation, including creation/evaluation work.
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

The HTML viewer links decisions, actual execution, and skill learning. `<run>.learning.jsonl`
records experience, drafts, trials, evaluation, promotion and suspension.
`<run>/skills/library.json` stores immutable versions and evaluation evidence.
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
