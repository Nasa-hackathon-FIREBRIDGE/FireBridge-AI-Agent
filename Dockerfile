# syntax=docker/dockerfile:1.7

# ── build: resolve the locked dependencies and install the package into a venv ──
FROM python:3.15-alpine@sha256:a0f2e268f2ce8f3709611a90054ec6a8ae31f6251f5896b8648b9bb99835e30a AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.28@sha256:0f36cb9361a3346885ca3677e3767016687b5a170c1a6b88465ec14aefec90aa /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable

# ── runtime: patched Alpine image, non-root user, no build tools, no secrets ─────────────
FROM python:3.15-alpine@sha256:a0f2e268f2ce8f3709611a90054ec6a8ae31f6251f5896b8648b9bb99835e30a
ARG REVISION=development
LABEL org.opencontainers.image.source="https://github.com/Nasa-hackathon-FIREBRIDGE/FireBridge-AI-Agent" \
      org.opencontainers.image.revision="${REVISION}"
LABEL org.opencontainers.image.title="FireBridge AI Agent" \
      org.opencontainers.image.description="Grounded explanations of FIREBRIDGE satellite fire analyses"
RUN apk upgrade --no-cache && addgroup -g 10001 agent && adduser -D -u 10001 -G agent agent \
    && mkdir -p /app/data && chown agent:agent /app/data
WORKDIR /app
COPY --from=build --chown=agent:agent /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FBA_DATABASE_PATH=/app/data/agent.db \
    FBA_LOG_JSON=true \
    FBA_RELEASE_REVISION=${REVISION}
USER agent
VOLUME ["/app/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).status != 200)"]
CMD ["uvicorn", "firebridge_agent.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers", "--no-access-log"]
