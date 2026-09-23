# ARC-AGI-3 Google ADK starter

Develop locally in JupyterLab, play ARC-AGI-3 games with a Google ADK agent, and build a Kaggle Notebook from the same Python source. The default ADK `BaseAgent` policy is a deterministic baseline; `make eval-model` selects Qwen3-VL-4B-Instruct (Q4_K_M GGUF plus Q8_0 vision projector) through ADK's `LlmAgent`. This is a working inference path, not a trained game solver.

## Server setup

Requires Docker Compose, NVIDIA Container Toolkit, and SSH port forwarding for JupyterLab. The Kaggle GPU image provides JupyterLab and Google ADK. The local build installs pinned ARC packages from `third_party/wheels/` without accessing PyPI; `arc-agi` requires Python 3.12 or later.

Run the `make` commands as the same SSH user that edits this repository (for example, `prog`). Compose then runs JupyterLab and one-off containers with that user's UID and GID, so files written through the bind mount are editable in both JupyterLab and the remote IDE. Cache files go under the ignored `.cache/model-cache/` directory.

```bash
cp .env.example .env             # keep your existing .env if already configured
make build
make setup                     # clone official ARC-AGI-3-Agents under ignored vendor/
make check
make lab                       # JupyterLab at http://localhost:8889 via SSH tunnel
```

If you previously ran JupyterLab as root inside the container, repair the existing files once after pulling this change:

```bash
make down
make repair-perms              # fixes generated notebook, .virtual_documents, vendor, and outputs
make notebook                  # regenerates the local/Kaggle compatible notebook
make lab
```

`make repair-perms` runs a one-off root container solely to transfer ownership of known generated paths to your SSH user. It does not change the ownership of the whole repository. Open the notebook again after restarting JupyterLab.

`make setup` downloads the official framework once and narrows its registry imports to the random agent. First local play may download and cache game environments; later runs can reuse the cache. Setup needs GitHub access. The Kaggle competition rerun uses the competition's offline wheel and framework dataset instead. If the base image lacks an underlying scientific dependency, the build's import check names the missing module; the wheel set assumes the current Kaggle GPU image.

## Local agent loop

```bash
make eval GAME=ls20 STEPS=50
make verify                    # ls20 and vc33, 50 actions each
make eval                      # all available games
```

`agent/my_agent.py` adapts the ARC framework's synchronous API. `agent/adk_policy.py` runs a Google ADK `Runner` for each observation. The default `OfflinePolicy` returns a repeatable action without a model or internet. `make eval` prints the local aggregate score and never accesses Kaggle's hidden competition set.

Edit `agent/*.py` as the source of the agent. The generated `notebooks/submission.ipynb` can be opened and executed in local JupyterLab: its wheel installation runs only when the Kaggle competition wheels exist, and its gateway and placeholder submission steps run only in the appropriate Kaggle environment. For local gameplay, use `make eval`; running the notebook locally prepares the submission code but does not play a game. `make notebook` regenerates the notebook from `agent/*.py` and overwrites edits made directly in the generated notebook. Jupyter's `.virtual_documents/` and the generated notebook are ignored by Git.

## Local vision model

Download the [official Qwen3-VL-4B-Instruct GGUF weights](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) (~3 GB), then start llama.cpp with an NVIDIA GPU:

```bash
make model-download
make model-up
make model-check                 # sends a real image request
make eval-model GAME=ls20 STEPS=50
```

The GGUF files are stored in ignored `.cache/model-cache/qwen3-vl-4b/`; they are never committed. Docker Compose exposes the model on the server's loopback port 8080. `agent/local_vlm.py` implements Google ADK's `BaseLlm` protocol and passes PNG frames with action history to `LlmAgent`; the `Runner` returns one validated legal action. No API key, LiteLLM installation, or internet connection is used during inference. `make model-up` needs access to GHCR for its first image pull. `make eval-model` sets `ADK_MODEL` for that run only; plain `make eval` remains the deterministic baseline.

To use an existing compatible server, set `VLM_API_BASE=http://host.docker.internal:8080/v1` in `.env` and run `ADK_MODEL=local/qwen3-vl-4b-instruct make eval GAME=ls20`; the server must present the `qwen3-vl-4b-instruct` alias and support OpenAI vision chat completions. JupyterLab may use the same endpoint through its Compose network.

## Offline Kaggle model bundle

The competition rerun cannot download models or a server binary. Build a **separate Kaggle Dataset** containing the two GGUF files and a CUDA llama.cpp runtime. This build uses Docker and checks out a pinned llama.cpp revision; the CUDA 12.8 build image may need a compatible host driver. The bundle is about 3 GB plus runtime libraries:

```bash
make model-download
make model-runtime
# Inspect the bundle: both GGUF files, llama-server, lib*.so, dataset-metadata.json.
ls -lh .cache/model-cache/qwen3-vl-4b/
make auth
# Upload once (or use "kaggle datasets version -p ... -m ..." when updating).
docker compose run --rm --no-deps dev bash -ec 'IFS= read -r KAGGLE_API_TOKEN < .kaggle/access_token; export KAGGLE_API_TOKEN; kaggle datasets create -p .cache/model-cache/qwen3-vl-4b'
```

The dataset slug in `.cache/model-cache/qwen3-vl-4b/dataset-metadata.json` and `notebooks/kernel-metadata.json` is `magicsword001/arc-agi-3-qwen3-vl-4b`; update **both** if your Kaggle username differs. Attach that dataset to the notebook before `make push`. The notebook checks for both GGUF files and the runtime, starts the server on `127.0.0.1:8080`, and sends an image request during Save & Run; in the competition rerun, ADK uses this same server. Errors are visible in `/kaggle/working/llama-server.log`. This full GPU path requires an actual Kaggle Save & Run check; Docker images, driver versions, and inference speed can change.

## Kaggle Notebook

Keep your existing `notebooks/kernel-metadata.json` with your Kaggle username and the competition source. Put the personal API token in `.kaggle/access_token` (one line, chmod 600). These secrets stay outside Git.

```bash
make notebook                  # inspect notebooks/submission.ipynb before upload
make auth                      # validate token
make push                      # Kaggle Save & Run; uploads the generated notebook
make status                    # inspect the Kaggle run
```

`make push` uploads a Notebook version after the model dataset exists under the configured slug. After its Save & Run completes, manually select `submission.parquet` in Kaggle's **Submit to Competition** UI to trigger the hidden rerun. The Notebook is generated from the agent sources, writes a placeholder parquet only during Save & Run, and uses Qwen3-VL during the competition rerun. Generated `notebooks/submission.ipynb`, credentials, caches, and the vendored framework are ignored by Git.

If the Kaggle runtime changes its installed Google ADK version or competition dataset paths, rerun `make notebook` and inspect the generated cells before pushing. This repository follows the official [ARC-AGI-3 Kaggle Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter) execution contract and uses the [Google ADK](https://adk.dev/) runtime.
