FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/frontend

# Install stable UI dependencies before copying source so code-only changes
# reuse this layer.
RUN python -m pip install \
      streamlit==1.64.0 \
      httpx==0.28.1

COPY frontend ./
# Share the exact payment-reference parser with FastAPI; no divergent UI heuristics.
COPY src/pe ./pe

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
