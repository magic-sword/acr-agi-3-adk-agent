# Local settings; command-line assignments still take precedence.
-include .env

DC := docker compose
RUN := $(DC) run --rm --no-deps dev
LOCAL_UID := $(shell id -u)
LOCAL_GID := $(shell id -g)
export LOCAL_UID LOCAL_GID
GAME ?=
STEPS ?= 80
EVAL_GAMES ?= ls20,vc33,ft09
EVAL_STEPS ?= 12
EVAL_LEVELS ?= 1
EVAL_SECONDS ?= 90
EVAL_HARD_SECONDS ?= 110
PYTHON ?= python3
FRAMEWORK_REPO := https://github.com/arcprize/ARC-AGI-3-Agents.git
FRAMEWORK_DIR := vendor/ARC-AGI-3-Agents

.PHONY: help build cache-dir repair-perms setup lab down logs shell gpu check auth eval verify notebook push status clean model-download model-up model-check eval-model model-runtime test benchmark benchmark-prepare sam-bundle submission-ready submission-smoke sam-upload sam-upload-version submission-remote-check adk-bundle adk-upload adk-upload-version

help:
	@printf '%s\n' \
	  'make build                 Build the Kaggle-compatible image' \
	  'make setup                 Clone and prepare the ARC framework' \
	  'make lab                   Start JupyterLab on localhost:8889' \
	  'make repair-perms          Fix files created by the previous root container' \
	  'make check                 Inspect local Python/Jupyter/ADK/GPU environment' \
	  'make auth                  Verify Kaggle API authentication' \
	  'make eval [GAME=ls20]      Play local ARC games with the ADK agent' \
	  'make model-download         Download Qwen3-VL GGUF + vision projector' \
	  'make model-up               Start the local GPU vision model' \
	  'make eval-model GAME=ls20   Play locally using Qwen3-VL and ADK' \
	  'make benchmark-prepare      Cache public evaluation games (no submission)' \
	  'make benchmark              Bounded local gateway evaluation + diagnostic report' \
	  'make visualize              Render current workflow + skill connections (no Docker/model)' \
	  'make test                   Check ARC action IDs and ADK model replies' \
	  'make model-runtime          Build portable llama-server for Kaggle bundle' \
	  'make verify                Short two-game smoke test' \
	  'make notebook              Build Kaggle submission.ipynb locally' \
	  'make submission-ready      Build and check notebook + offline SAM dataset (no upload)' \
	  'make submission-smoke      Test staged SAM + running Qwen on GPU (no upload)' \
	  'make sam-upload            Create the private SAM dataset (first upload)' \
	  'make qwen-upload           Create the private Qwen dataset (first upload)' \
	  'make sam-upload-version    Upload a new SAM dataset version when assets change' \
	  'make adk-upload            Create the private ADK wheels dataset (first upload)' \
	  'make adk-upload-version    Upload a new ADK wheels dataset version when wheels change' \
	  'make push                  Build and push a Kaggle Notebook version' \
	  'make status                Check latest Kaggle kernel run' \
	  'make down                  Stop JupyterLab'

build:
	$(DC) build

cache-dir:
	mkdir -p .cache/model-cache

repair-perms: cache-dir
	$(DC) run --rm --no-deps --user 0 dev sh -ec 'for p in .virtual_documents notebooks vendor environment_files outputs recordings logs.log submission.parquet; do if [ -e "$$p" ]; then chown -R "$(LOCAL_UID):$(LOCAL_GID)" "$$p"; fi; done'

setup: cache-dir
	$(RUN) bash -ec 'if [ ! -d $(FRAMEWORK_DIR)/.git ]; then mkdir -p vendor; git clone --depth 1 $(FRAMEWORK_REPO) $(FRAMEWORK_DIR); fi; python scripts/setup_framework.py'

lab: cache-dir
	$(DC) up -d --build dev
	@echo 'JupyterLab: http://localhost:8889 (through your SSH LocalForward)'

logs:
	$(DC) logs -f dev

down:
	$(DC) down

shell: cache-dir
	$(RUN) bash

gpu: cache-dir
	$(RUN) nvidia-smi

check: cache-dir
	$(RUN) python scripts/check_env.py

auth: cache-dir
	$(RUN) bash -ec 'test -s .kaggle/access_token || { echo ".kaggle/access_token missing"; exit 2; }; IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; kaggle kernels list --mine --page-size 1 >/dev/null && echo "Kaggle authentication: OK"'

eval: cache-dir
	$(RUN) python scripts/play_local.py $(if $(GAME),--game $(GAME)) --max-steps $(STEPS)

model-download: cache-dir
	python3 scripts/download_model.py

model-up: cache-dir
	$(DC) --profile vlm up -d vlm

model-check: cache-dir
	python3 scripts/wait_model.py --url http://127.0.0.1:8080

eval-model: model-check
	ADK_MODEL=local/qwen3-vl-4b-instruct $(RUN) python scripts/play_local.py $(if $(GAME),--game $(GAME)) --max-steps $(STEPS)

test: cache-dir
	$(RUN) python -m unittest discover -s tests -v

model-runtime: cache-dir
	bash scripts/build_model_runtime.sh

verify: cache-dir
	$(RUN) python scripts/play_local.py --game ls20,vc33 --max-steps 50

notebook: cache-dir
	$(PYTHON) scripts/build_notebook.py

sam-bundle:
	$(PYTHON) scripts/build_sam_bundle.py

submission-ready: sam-bundle qwen-bundle adk-bundle notebook
	$(PYTHON) scripts/check_submission.py

submission-smoke: submission-ready model-check
	$(RUN) python scripts/check_submission.py --gpu

# These targets publish only when explicitly invoked. Kaggle datasets are private by default.
sam-upload: sam-bundle
	$(RUN) bash -ec 'IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; kaggle datasets create -p .cache/kaggle-sam-bundle --keep-tabular'

sam-upload-version: sam-bundle
	$(RUN) bash -ec 'IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; kaggle datasets version -p .cache/kaggle-sam-bundle --keep-tabular -m "Pinned SAM ViT-B offline runtime"'

adk-bundle:
	$(PYTHON) scripts/build_adk_bundle.py

adk-upload: adk-bundle
	$(RUN) bash -ec 'IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; kaggle datasets create -p .cache/kaggle-adk-bundle --keep-tabular'

adk-upload-version: adk-bundle
	$(RUN) bash -ec 'IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; kaggle datasets version -p .cache/kaggle-adk-bundle --keep-tabular -m "Pinned Google ADK offline wheels"'

submission-remote-check: submission-ready
	$(RUN) python scripts/check_submission.py --remote

push: submission-remote-check
	$(RUN) bash -ec 'test -s .kaggle/access_token || { echo ".kaggle/access_token missing"; exit 2; }; IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; kaggle kernels push -p notebooks/'

status: cache-dir
	$(RUN) bash -ec 'IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; KERNEL_ID=$$(python -c '\''import json; print(json.load(open("notebooks/kernel-metadata.json"))["id"])'\''); kaggle kernels status "$$KERNEL_ID"'

clean:
	$(DC) down --remove-orphans
	rm -f notebooks/submission.ipynb

benchmark-prepare: cache-dir
	$(RUN) python scripts/benchmark_local.py --prepare --games $(EVAL_GAMES)

benchmark: cache-dir
	$(RUN) python scripts/benchmark_local.py --games $(EVAL_GAMES) --steps $(EVAL_STEPS) --levels $(EVAL_LEVELS) --seconds $(EVAL_SECONDS) --hard-seconds $(EVAL_HARD_SECONDS)

.PHONY: visualize
visualize:
	$(PYTHON) scripts/visualize_agent.py

.PHONY: qwen-bundle qwen-upload qwen-upload-version
qwen-bundle:
	$(PYTHON) scripts/build_qwen_bundle.py

qwen-upload: qwen-bundle
	$(RUN) bash -ec 'IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; kaggle datasets create -p .cache/kaggle-qwen-bundle --keep-tabular'

qwen-upload-version: qwen-bundle
	$(RUN) bash -ec 'IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; kaggle datasets version -p .cache/kaggle-qwen-bundle --keep-tabular -m "Qwen GGUF and portable CUDA runtime"'
