DC := docker compose
RUN := $(DC) run --rm --no-deps dev
GAME ?=
STEPS ?= 80
FRAMEWORK_REPO := https://github.com/arcprize/ARC-AGI-3-Agents.git
FRAMEWORK_DIR := vendor/ARC-AGI-3-Agents

.PHONY: help build setup lab down logs shell gpu check auth eval verify notebook push status clean

help:
	@printf '%s\n' \
	  'make build                 Build the Kaggle-compatible image' \
	  'make setup                 Clone and prepare the ARC framework' \
	  'make lab                   Start JupyterLab on localhost:8889' \
	  'make check                 Inspect local Python/Jupyter/ADK/GPU environment' \
	  'make auth                  Verify Kaggle API authentication' \
	  'make eval [GAME=ls20]      Play local ARC games with the ADK agent' \
	  'make verify                Short two-game smoke test' \
	  'make notebook              Build Kaggle submission.ipynb locally' \
	  'make push                  Build and push a Kaggle Notebook version' \
	  'make status                Check latest Kaggle kernel run' \
	  'make down                  Stop JupyterLab'

build:
	$(DC) build

setup:
	$(RUN) bash -ec 'if [ ! -d $(FRAMEWORK_DIR)/.git ]; then mkdir -p vendor; git clone --depth 1 $(FRAMEWORK_REPO) $(FRAMEWORK_DIR); fi; python scripts/setup_framework.py'

lab:
	$(DC) up -d --build dev
	@echo 'JupyterLab: http://localhost:8889 (through your SSH LocalForward)'

logs:
	$(DC) logs -f dev

down:
	$(DC) down

shell:
	$(RUN) bash

gpu:
	$(RUN) nvidia-smi

check:
	$(RUN) python scripts/check_env.py

auth:
	$(RUN) bash -ec 'test -s .kaggle/access_token || { echo ".kaggle/access_token missing"; exit 2; }; IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; kaggle kernels list --mine --page-size 1 >/dev/null && echo "Kaggle authentication: OK"'

eval:
	$(RUN) python scripts/play_local.py $(if $(GAME),--game $(GAME)) --max-steps $(STEPS)

verify:
	$(RUN) python scripts/play_local.py --game ls20,vc33 --max-steps 50

notebook:
	$(RUN) python scripts/build_notebook.py

push: notebook
	$(RUN) bash -ec 'test -s .kaggle/access_token || { echo ".kaggle/access_token missing"; exit 2; }; IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; kaggle kernels push -p notebooks/'

status:
	$(RUN) bash -ec 'IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token || true; test -n "$$KAGGLE_API_TOKEN"; export KAGGLE_API_TOKEN; KERNEL_ID=$$(python -c '\''import json; print(json.load(open("notebooks/kernel-metadata.json"))["id"])'\''); kaggle kernels status "$$KERNEL_ID"'

clean:
	$(DC) down --remove-orphans
	rm -f notebooks/submission.ipynb
