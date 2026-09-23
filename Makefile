DC := docker compose
RUN := $(DC) run --rm --no-deps dev
GAME ?=
STEPS ?= 200

.PHONY: help build lab down logs shell gpu check auth eval verify notebook push status clean

help:
	@printf '%s\n' \
	  'make build                 Build the Kaggle-compatible image' \
	  'make lab                   Start JupyterLab on server localhost:8889' \
	  'make shell                 Open a shell in the dev container' \
	  'make gpu                   Verify NVIDIA GPU visibility' \
	  'make check                 Verify local Python/Jupyter/ADK/GPU environment' \
	  'make auth                  Verify Kaggle API authentication' \
	  'make eval [GAME=ls20]      Run local ARC simulation' \
	  'make verify                Short local smoke test' \
	  'make notebook              Build submission.ipynb if script exists' \
	  'make push                  Build + push Kaggle notebook if configured' \
	  'make status                Check latest Kaggle kernel run' \
	  'make down                  Stop JupyterLab' \
	  'make clean                 Stop containers and remove generated notebook'

build:
	$(DC) build

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
	$(RUN) bash -lc 'test -s .kaggle/access_token || { echo ".kaggle/access_token is missing or empty"; exit 2; }; export KAGGLE_API_TOKEN="$(tr -d "\\r\\n" < .kaggle/access_token)"; kaggle kernels list --mine --page-size 1 >/dev/null && echo "Kaggle authentication: OK"'

# Compatible with the official ARC-AGI-3 Kaggle Starter layout.
eval:
	$(RUN) bash -lc 'if [ -f scripts/play_local.py ]; then python scripts/play_local.py $(if $(GAME),--game $(GAME)) $(if $(STEPS),--max-steps $(STEPS)); else echo "scripts/play_local.py not found. Add it from ARC-AGI-3-Kaggle-Starter or your project."; exit 2; fi'

verify:
	$(RUN) bash -lc 'if [ -f scripts/play_local.py ]; then python scripts/play_local.py --game ls20,vc33 --max-steps 50; else echo "scripts/play_local.py not found."; exit 2; fi'

notebook:
	$(RUN) bash -lc 'test -f scripts/build_notebook.py && python scripts/build_notebook.py'

push: notebook
	$(RUN) bash -lc 'test -f notebooks/kernel-metadata.json || { echo "notebooks/kernel-metadata.json not found"; exit 2; }; test -f .kaggle/access_token || { echo ".kaggle/access_token not found"; exit 2; }; KAGGLE_API_TOKEN="$$(cat .kaggle/access_token)" kaggle kernels push -p notebooks/'

status:
	$(RUN) bash -lc 'test -f notebooks/kernel-metadata.json || { echo "notebooks/kernel-metadata.json not found"; exit 2; }; test -f .kaggle/access_token || { echo ".kaggle/access_token not found"; exit 2; }; KERNEL_ID="$$(python -c '\''import json; print(json.load(open("notebooks/kernel-metadata.json"))["id"])'\'')"; KAGGLE_API_TOKEN="$$(cat .kaggle/access_token)" kaggle kernels status "$$KERNEL_ID"'

clean:
	$(DC) down --remove-orphans
	rm -f notebooks/submission.ipynb
