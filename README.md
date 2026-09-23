# acr-agi-3-adk-agent

Kaggle-compatible local development environment for ARC Prize 2026 / ARC-AGI-3 using Google ADK.

The environment intentionally uses Kaggle's official GPU runtime image instead of rebuilding CUDA, PyTorch, JupyterLab, Transformers, Kaggle CLI, or Google ADK from scratch.

## Architecture

- Base image: `gcr.io/kaggle-gpu-images/python`
- One Docker Compose service: `dev`
- JupyterLab is provided by the Kaggle image
- Jupyter is exposed only on the SSH server loopback: `127.0.0.1:8889`
- Project root is mounted at `/kaggle/working`
- Local input data is mounted read-only at `/kaggle/input`
- Hugging Face/model cache persists in a Docker volume
- Kaggle credentials stay outside the image under `.kaggle/`

## Server prerequisites

The SSH server needs:

1. Docker Engine
2. Docker Compose plugin
3. NVIDIA driver
4. NVIDIA Container Toolkit

Your SSH client can forward JupyterLab with:

```sshconfig
LocalForward localhost:8889 localhost:8889
```

## First start

```bash
cp .env.example .env
# Edit .env and change JUPYTER_TOKEN

make build
make gpu
make check
make lab
```

Then open `http://localhost:8889` on your local machine through the SSH tunnel.

## Daily commands

```bash
make lab
make shell
make gpu
make check
make down
```

The Makefile also reserves the intended competition workflow:

```bash
make eval
make eval GAME=ls20
make verify
make notebook
make push
make status
```

`make eval` expects `scripts/play_local.py`, and `make notebook` expects `scripts/build_notebook.py`. Those agent/submission scripts are intentionally not fabricated here; they should come from the ARC-AGI-3 agent implementation or an adapted official starter workflow.

## Kaggle API token

Create a project-local token file and do not commit it:

```bash
mkdir -p .kaggle
printf '%s\n' 'YOUR_KAGGLE_TOKEN' > .kaggle/access_token
chmod 600 .kaggle/access_token
```

Prepare notebook metadata:

```bash
cp notebooks/kernel-metadata.example.json notebooks/kernel-metadata.json
```

Then edit the `id`, dataset sources, and model sources for your Kaggle account.

## Reproducibility

During exploration the moving Kaggle image tag is convenient:

```text
gcr.io/kaggle-gpu-images/python
```

For a validated competition environment, pin `KAGGLE_BASE` in `.env` to the exact image digest you tested.

## Why there is no SSH daemon in the container

SSH terminates on the host server. Docker is only the reproducible execution environment. This avoids a second SSH daemon, duplicated credentials, and unnecessary network exposure.
