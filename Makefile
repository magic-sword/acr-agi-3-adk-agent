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
FRAMEWORK_REPO := https://github.com/arcprize/ARC-AGI-3-Agents.git
FRAMEWORK_DIR := vendor/ARC-AGI-3-Agents

.PHONY: help build cache-dir repair-perms setup lab down logs shell gpu check auth eval verify notebook push status clean model-download model-up model-check eval-model model-runtime test benchmark benchmark-prepare

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
	  'make test                   Check ARC action IDs and ADK model replies' \
	  'make model-runtime          Build portable llama-server for Kaggle bundle' \
	  'make verify                Short two-game smoke test' \
	  'make notebook              Build Kaggle submission.ipynb locally' \
	  'make push                  Build and push a Kaggle Notebook version' \
	  'make status                Check latest Kaggle kernel run' \
	  'make down                  Stop JupyterLab'

build:
	$(DC) build

cache-dir:
	mkdir -p .cache/model-cache

repair-perms: cache-dir
	$(DC) run --rm --no-deps --user 0 dev sh -ec 'for p in .virtual_documents notebooks/submission.ipynb vendor environment_files outputs recordings logs.log submission.parquet; do if [ -e "$$p" ]; then chown -R "$(LOCAL_UID):$(LOCAL_GID)" "$$p"; fi; done'

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
	$(RUN) python scripts/build_notebook.py

push: notebook
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
