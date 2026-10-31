FROM python:3.15-slim

LABEL org.opencontainers.image.source=https://github.com/mitch-b/vllm-hub

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Runs as root on purpose: it needs the Docker socket to manage the vLLM container.
EXPOSE 7860
# start-interval probes every second while booting so a self-update can confirm health almost immediately.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --start-interval=1s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=3)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-graceful-shutdown", "2"]
