# ARC-AGI-3 fast/slow agent

A local Google ADK 2.0 agent with two decision speeds using Qwen3-VL-4B-Instruct.
Deliberation interprets causal evidence, works backwards from a goal, and builds
small grounded procedures. The fast path selects a skill or action with one
output token. It can advance a step with `7` or return to deliberation with `8`,
without sending a game action. Active procedures persist across observations.
Without a model the driver uses deterministic smoke probes.

See the [current design and limits](docs/fast-slow-runtime-ja.md),
[latency investigation](docs/fast-slow-runtime-research-ja.md), and
[evaluation guide](docs/local-evaluation-ja.md). Earlier attention and
skill-learning documents describe superseded policies.

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
A state diagram highlights deliberation, fast selection, execution and observation waits.
The replay shows the plan, active procedure, predicted effect, measured changes,
one-token choice and returns to deliberation. Playback defaults to state transitions;
action and full-event modes are also available. Click **計画・スキルを読む** to inspect
the recorded causal hypotheses and procedures at that point in time.
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
actual actions, observations, model requests and fast/slow decisions under `outputs/evaluations/`.
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

- `COGNITION_REPAIR_ATTEMPTS=1`: one repair for a malformed deliberation output; `0` disables it.
- `COGNITION_SECONDS=600`: total per-game reasoning deadline.
- `COGNITION_DECISION_SECONDS=45`: time allowed to obtain the next action, including replanning.
- `COGNITION_MAX_RESETS=2`: host-owned episode restarts.
- `COGNITION_LOG_DIR=outputs/cognition`: logging location; empty disables persistence.
- `VLM_API_BASE=http://vlm:8080/v1`: local model endpoint.

The host checks legal controls, original-pixel click bounds and execution receipts.
It supplies before/after images and measured changes, without enumerating objects
or rejecting repeated actions. Procedures contain action options, expected effects,
step completion criteria and reasons to reconsider. Their meaning remains a model
hypothesis; pixel differences alone do not establish success.

Level/reset boundaries clear the grounded plan and active step. Causal notes and
procedures remain available to deliberation for re-grounding. Fast calls receive
only the current goal, step, options and last result. Selection probabilities are
logged as model diagnostics, not calibrated success probabilities.

```bash
python3 scripts/analyze_agent.py outputs/cognition
```

`*.model.jsonl` records the exact judgment context, response, requests and timing.
`*.artifacts.jsonl` records plans, procedure state, fast choices, returns to
deliberation and acknowledged measurements. Evaluation reports split latency
and token counts by deliberation, skill selection and step execution.
Requests, tool execution, observations and driver acknowledgements have separate
journals. These are observable decision artifacts, not private model reasoning.

## Offline notebook

```bash
make notebook                  # regenerate from agent source
```

`notebooks/submission.ipynb` is generated; edit `agent/` instead. The notebook bundles
agent code and pinned ADK wheels. Kaggle reruns use the competition's offline framework
and model bundle. Local execution prepares the code; `make eval` plays locally.

`make push` explicitly builds and uploads the notebook when requested. `make status` checks
an existing kernel run. Authentication uses the local ignored `.kaggle/access_token`.
The present refactor does not run either publishing command.
