# syntax=docker/dockerfile:1

# Kaggle's official GPU runtime image.
# Pin this to a digest once you want a frozen competition environment.
ARG KAGGLE_BASE=gcr.io/kaggle-gpu-images/python
FROM ${KAGGLE_BASE}

WORKDIR /kaggle/working

# Keep the image intentionally thin. Kaggle's image already contains
# JupyterLab, PyTorch, Transformers, Kaggle CLI and Google ADK.
# Install only project-specific packages when requirements.local.txt is non-empty.
COPY requirements.local.txt /tmp/requirements.local.txt
RUN if [ -s /tmp/requirements.local.txt ]; then \
      python -m pip install --no-cache-dir -r /tmp/requirements.local.txt; \
    fi

CMD ["bash"]
