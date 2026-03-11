# =============================================================================
# ANA — Backend
# Targets:
#   dev        → uvicorn --reload + devDeps (para docker compose dev)
#   production → uvicorn sem reload, usuário não-root
#
# Build:
#   docker build --target dev .
#   docker build --target production .
# =============================================================================

# =============================================================================
# Base — sistema + uv + dependências Python de produção
# =============================================================================
FROM python:3.12-slim AS base

LABEL org.opencontainers.image.title="ANA Backend"
LABEL org.opencontainers.image.description="FastAPI + RAG + Sessões + Validação"

# uv — gestor de dependências
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Dependências de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH="/app"
ENV PATH="/app/.venv/bin:$PATH"
# Evita overhead de re-sync do 'uv run' após ativar o venv no PATH
ENV UV_PROJECT_ENVIRONMENT="/app/.venv"


# =============================================================================
# Deps — camada de dependências isolada (cache reutilizável)
# Invalida somente quando pyproject.toml ou uv.lock mudam.
# =============================================================================
FROM base AS deps

COPY pyproject.toml uv.lock* ./
# Instala dependências de produção (sem grupos opcionais pesados)
RUN uv sync --frozen --no-install-project --no-dev


# =============================================================================
# Dev — inclui devDeps, código montado via volume, uvicorn --reload
#
# No docker-compose.dev.yml:
#   volumes: ../ana-backend/config → /app/config
#   develop.watch: sync ana/ → /app/ana
# =============================================================================
FROM deps AS dev

# Instala todas as deps incluindo dev
RUN uv sync --frozen

# Copia código-fonte (será sobrescrito por volume/watch em dev)
COPY ana/     ./ana/
COPY config/  ./config/
COPY scripts/ ./scripts/

# Instala o projeto no venv
RUN uv sync --frozen

EXPOSE 8000

CMD ["uvicorn", "ana.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app/ana", \
     "--log-level", "debug"]


# =============================================================================
# Production — código copiado, usuário não-root, sem reload
# =============================================================================
FROM deps AS production

COPY ana/     ./ana/
COPY config/  ./config/
COPY scripts/ ./scripts/

# Instala o projeto no venv (sem deps de dev)
RUN uv sync --frozen --no-dev

# Usuário não-root
RUN useradd --create-home --shell /bin/bash --uid 1000 ana \
    && chown -R ana:ana /app
USER ana

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8000/health/ || exit 1

CMD ["uvicorn", "ana.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info"]
