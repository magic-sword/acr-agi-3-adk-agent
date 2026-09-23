# ARC-AGI-3 Google ADK starter

Develop locally in JupyterLab, play ARC-AGI-3 games with a Google ADK agent, and build a Kaggle Notebook from the same Python source. The default ADK `BaseAgent` policy is a deterministic offline baseline. It proves the connection to the game framework; it does not solve the games. Set `ADK_MODEL` to test a local OpenAI-compatible model with frame images.

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

For a local vision model with an OpenAI-compatible `/v1` endpoint, add these values to `.env` and rebuild the container when installing optional LiteLLM dependencies:

```env
ADK_MODEL=openai/your-vision-model
OPENAI_API_BASE=http://host.docker.internal:8000/v1
OPENAI_API_KEY=local-only
```

The optional backend requires `litellm` installed in the development image and a running vision-capable server. The adapter sends a PNG of the latest frame and action/state metadata to ADK `LlmAgent`. This path has not been validated against a specific model server; malformed action responses raise an error. The generated Kaggle Notebook explicitly uses the offline baseline regardless of local `ADK_MODEL`, because the competition rerun has no external model server. To run a VLM on Kaggle later, provide its weights through `model_sources`/`dataset_sources`, start an in-process inference server, and revise that Notebook path.

## Kaggle Notebook

Keep your existing `notebooks/kernel-metadata.json` with your Kaggle username and the competition source. Put the personal API token in `.kaggle/access_token` (one line, chmod 600). These secrets stay outside Git.

```bash
make notebook                  # inspect notebooks/submission.ipynb before upload
make auth                      # validate token
make push                      # Kaggle Save & Run; uploads the generated notebook
make status                    # inspect the Kaggle run
```

`make push` uploads a Notebook version. After its Save & Run completes, manually select `submission.parquet` in Kaggle's **Submit to Competition** UI to trigger the hidden rerun. The Notebook is generated from both files in `agent/` and uses the same policy as `make eval`. It writes a placeholder parquet only during Save & Run; the gateway produces the real output during the competition rerun. Generated `notebooks/submission.ipynb`, credentials, caches, and the vendored framework are ignored by Git.

If the Kaggle runtime changes its installed Google ADK version or competition dataset paths, rerun `make notebook` and inspect the generated cells before pushing. This repository follows the official [ARC-AGI-3 Kaggle Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter) execution contract and uses the [Google ADK](https://adk.dev/) runtime.
