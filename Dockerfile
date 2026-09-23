# syntax=docker/dockerfile:1

# Kaggle's official GPU runtime image.
# Pin this to a digest once you want a frozen competition environment.
ARG KAGGLE_BASE=gcr.io/kaggle-gpu-images/python
FROM ${KAGGLE_BASE}

WORKDIR /kaggle/working

# Kaggle's image provides JupyterLab, Google ADK and the common scientific
# dependencies. Install the small ARC packages from pinned local wheels so
# Docker builds do not depend on DNS or PyPI access.
COPY requirements.local.txt /tmp/requirements.local.txt
COPY third_party/wheels/ /tmp/arc-wheels/
RUN python -m pip install --no-index --no-deps --find-links=/tmp/arc-wheels \
      -r /tmp/requirements.local.txt && \
    python -c 'import arc_agi, arcengine, dotenv, flask; print("ARC packages: OK")'

CMD ["bash"]
