FROM python:3.13-slim

ARG TORCH_VERSION=2.9.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are installed before application sources so normal code changes
# keep the expensive ML dependency layers cached.
COPY pyproject.toml README.md ./

# This PoC performs embeddings on CPU. Pre-install the official CPU-only wheel
# so sentence-transformers does not resolve CUDA/NVIDIA packages from PyPI.
RUN python -m pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==${TORCH_VERSION}"

# Keep pyproject.toml as the single source of truth for runtime dependencies.
# Extract its dependency list with stdlib tomllib, then install it once.
RUN python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb')); print('\\n'.join(p['project']['dependencies']))" \
      > /tmp/runtime-requirements.txt \
    && python -m pip install -r /tmp/runtime-requirements.txt \
    && rm /tmp/runtime-requirements.txt

COPY src ./src
RUN python -m pip install --no-deps .

# The same backend image is used by both API and dataset-loader.
COPY datasets ./datasets

EXPOSE 8000
CMD ["uvicorn", "pe.axiz.graphrag_payments.main:app", "--host", "0.0.0.0", "--port", "8000"]
