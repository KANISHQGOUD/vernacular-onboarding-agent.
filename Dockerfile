# The image a public demo runs. Strict by default: PUBLIC_DEPLOYMENT=1 means
# the server refuses to start without an access code, a daily session cap, a
# reachable database and a Gemini key (see app/web/gate.py). Secrets come from
# the host's environment at run time and are never baked into the image --
# .dockerignore keeps .env out of the build context entirely.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PUBLIC_DEPLOYMENT=1 \
    SPEECH_CACHE_DIR=/data/speech-cache \
    PORT=8000

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY web ./web
COPY scripts ./scripts

# Not root. /data is where a host mounts its persistent volume for the speech
# cache; it is created here so the app can write to it without one too.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/speech-cache \
    && chown -R app:app /data
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/healthz', timeout=4)"

# python -m app.web, not a bare uvicorn command: the frame limit, the single
# worker and the proxy headers are fixed in code there, so a host's start
# command cannot quietly drop them.
CMD ["python", "-m", "app.web"]
