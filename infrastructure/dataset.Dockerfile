FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY datasets ./datasets
RUN pip install --upgrade pip && pip install .

CMD ["python", "datasets/load_dataset.py"]
