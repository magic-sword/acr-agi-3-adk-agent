# ARC-AGI-3 visual-attention agent

A local Google ADK 2.0 agent that looks at the game image and selects one action.
One normal model request interprets the previous result, chooses a visual focus,
and proposes the next action. It does not enumerate objects or select from a fixed
question list. Qwen3-VL-4B-Instruct runs locally; without a model the driver uses
deterministic smoke probes.

See the [current design and limits](docs/visual-attention-runtime-ja.md) and
[evaluation guide](docs/local-evaluation-ja.md). The earlier focused-task,
object-memory, and skill-learning documents describe superseded policies.

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
A state diagram highlights visual attention, action validation, execution and observation waits.
The replay shows the actual model input, selected focus, prediction and measured changes.
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
actual actions, observations, model requests and attention decisions under `outputs/evaluations/`.
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

## Configuration and logs

- `COGNITION_REPAIR_ATTEMPTS=1`: allow one extra request for an invalid output; `0` disables repair. Normal actions use one request.
- `COGNITION_SECONDS=600`: total game runtime budget; benchmarks override it.
- `COGNITION_MAX_RESETS=2`: host-owned episode restarts.
- `COGNITION_LOG_DIR=outputs/cognition`: logging location; empty disables persistence.
- `VLM_API_BASE=http://vlm:8080/v1`: local model endpoint.

The old `COGNITION_MAX_CALLS`, `COGNITION_MAX_HTTP_REQUESTS`, learning and
skill-library settings are no longer used by the production policy. Skill
construction and reuse are not part of this attention loop.

Game images contain the board and coordinate rulers. The model chooses any
visible target or relationship directly; the host checks legal controls and
original-pixel click bounds, without requiring an object mask. Before/after
images are provided when the screen changes. Unchanged screens need one image.

The host supplies global pixel differences and eight recent trials. Short model
notes remain hypotheses. An action with no visible effect cannot be repeated in
the same state beyond its predeclared limit (normally one, at most three).
Changing the description does not renew the limit. Level/reset boundaries clear
local trials and notes; changed states can be explored again. Receipt uncertainty
never causes an automatic resend.

```bash
python3 scripts/analyze_agent.py outputs/cognition
```

`*.model.jsonl` records the exact judgment context, response, requests and timing.
`*.artifacts.jsonl` records `attention_selected` (focus, interpretation, action,
expectation and short notes) and `attention_feedback` (acknowledged measurements).
Requests, tool execution, observations and driver acknowledgements have separate
journals. These are observable decision artifacts, not private model reasoning.

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
