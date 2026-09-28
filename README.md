# ARC-AGI-3 fast/slow agent

A local Google ADK 2.0 agent with two decision speeds using Qwen3-VL-4B-Instruct.
Deliberation has four stages: understand the relevant objects, backchain from a goal,
ground a small executable procedure, and reconcile its result. Goals and their
prerequisites persist across replanning. The fast path selects a skill or action
with one output token. `7` proposes step completion; `8` returns to reconciliation.
One-action probes return their acknowledged result for learning without confirming a goal.
Without a model the driver uses deterministic smoke probes.

See the [current design and limits](docs/fast-slow-runtime-ja.md),
[latency investigation](docs/fast-slow-runtime-research-ja.md), and
[evaluation guide](docs/local-evaluation-ja.md). Earlier attention and
skill-learning documents describe superseded policies.

Start with [物体認識の採用構成・検証結果・参考資料](docs/visual-recognition-adopted-ja.md)
for the adopted design, successful measurements, references, and validation logs.
The default uses SAM once per episode to supplement proposals, then tracks pixels
with the program; Qwen interprets small measured changes with one-token questions.
See the [runtime specification](docs/hybrid-perception-runtime-20260928-ja.md) for
configuration, the [handoff](docs/visual-recognition-handoff-ja.md) for outstanding work,
and the [research history](docs/history/visual-recognition/README.md) for brief lessons
and archived experiments. `COGNITION_PROPOSALS=program` retains the previous sensor.

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
A state diagram highlights the four deliberation stages, fast selection, execution and observation waits.
The replay shows semantic targets, concept and role hypotheses, the exact cursor preview, goal dependencies and assessments, the active
procedure and invocation ID, predicted effects, measured changes,
one-token choice and returns to deliberation. Playback defaults to state transitions;
action and full-event modes are also available. Click **測定・計画を読む** to inspect
the recorded causal hypotheses, procedures, memory navigation and selected working set at that point in time.
Deliberation appends evidence and interpretations; one-token memory navigation selects the records
for the next question. See the [memory retrieval design](docs/memory-retrieval-runtime-ja.md).
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

Targets describe shared concepts, instance/group appearance, role hypotheses and relations.
Goals and skills use natural-language target queries; they never store click coordinates.
CLICK first selects a quadrant with one token (1–4), then enters a cursor loop:
1–4 move, 5/6 change stride, 7 clicks, 8 returns to
reconciliation. Host previews include a local zoom; cursor moves do not operate the game.
Bindings expire at the next observation. Directional controls bypass the cursor.
See [semantic targeting and cursor execution](docs/semantic-cursor-runtime-ja.md).

Level/reset boundaries clear scene instances, goals, the plan and active step. Concepts, causal notes and
procedures remain available to deliberation for re-grounding. Fast calls receive the current small goal, semantic targets, baseline, step, options
and the result belonging to this invocation and step. Selection probabilities are
logged as model diagnostics, not calibrated success probabilities.

```bash
python3 scripts/analyze_agent.py outputs/cognition
```

`*.model.jsonl` records the exact judgment context, response, requests and timing.
`*.artifacts.jsonl` records stage results, persistent goals, procedure invocations,
completion candidates, reconciliation and acknowledged measurements. Evaluation reports split latency
and token counts by deliberation, skill selection, step execution and cursor aiming.
Cursor confirmation counts and total aiming latency are separate from environment actions.
Requests, tool execution, observations and driver acknowledgements have separate
journals. These are observable decision artifacts, not private model reasoning.

## Offline notebook

```bash
make submission-ready          # notebook + Qwen/SAM datasets; no upload
make submission-smoke          # GPU check; requires running Qwen (make model-up)
```

`notebooks/submission.ipynb` is generated; edit `agent/` instead. The notebook bundles
agent code and pinned ADK wheels. `config/sam-bundle.json` pins the measured SAM ViT-B
checkpoint and official source. `make submission-ready` uses the existing assets in
`outputs/sam-deps/` to stage `.cache/kaggle-sam-bundle/` (about 375 MB), including the
license. Override source locations with `scripts/build_sam_bundle.py --source PATH
--checkpoint PATH` when building manually. No downloads happen during packaging or
Notebook execution.

Qwen is staged separately at `.cache/kaggle-qwen-bundle/`, including both GGUFs,
the portable CUDA `llama-server`, shared libraries, licenses and a hash manifest.
The current Qwen staging folder is about 3.19 GB.
For a fresh checkout, run `make model-download` and `make model-runtime` first;
the latter builds the pinned llama.cpp commit for T4 (CUDA architecture 75) with
CPU-native optimizations disabled. Its Docker build uses host networking to avoid
the local bridge DNS failure; override with `MODEL_BUILD_NETWORK=default` if needed.

The current metadata expects your private Datasets
`magicsword001/arc-agi-3-qwen3-vl-4b` and `magicsword001/arc-agi-3-sam-vit-b`.
Upload the **contents** of the corresponding staging folders without an extra
parent directory. The commands below use each folder's `dataset-metadata.json`.

```bash
make auth                      # credentials in ignored .kaggle/access_token
make qwen-upload               # first time only: create the private Qwen Dataset
make sam-upload                # first time only: create the private SAM Dataset
# Wait until Kaggle finishes processing both Datasets.
make push                      # verify attached assets, then upload the private Notebook
make status                    # inspect Save & Run status
```

After changing SAM assets, use `make sam-upload-version` instead of `make sam-upload`.
Use `make qwen-upload-version` after changing Qwen weights or its runtime.
Agent-only changes need only `make push`. `make submission-remote-check` checks both
remote manifests and required SAM/Qwen files without uploading. A failed check
stops `make push`; it does not upload a Notebook with missing assets. The
Qwen Dataset must include both GGUFs, `llama-server`, and its runtime libraries.

Both Save & Run and hidden reruns verify SAM hashes and exercise actual SAM/Qwen
image inference. A failure stops startup instead of silently submitting program-only
perception. Hidden reruns warm SAM inside the game worker, which retains one shared
provider across Swarm threads; the Notebook parent does not load another copy.
Kaggle writes `submission-preflight.json`; local smoke results go to
`outputs/kaggle-ready/submission-preflight.json`. Normal per-episode SAM detection
and subsequent pixel tracking are unchanged. See the
[deployment verification and limitations](docs/hybrid-perception-runtime-20260928-ja.md#kaggle提出準備).

The generated Notebook has also passed all seven code cells in a network-disabled
GPU container using the two staged datasets. Those Save & Run results are in
`outputs/kaggle-ready/notebook/`; the competition gateway and actual Kaggle hardware
remain to be checked after you upload.
