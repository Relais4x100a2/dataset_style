# syntax=docker/dockerfile:1
# Image unique : Streamlit (8501, défaut CMD) ou FastAPI webapp (8080, commande compose / CapRover).
FROM python:3.12-slim-bookworm AS deps

WORKDIR /app

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock ./
COPY main.py ./
COPY src ./src

RUN addgroup --system appuser && adduser --system --ingroup appuser appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501 8080

CMD ["/app/.venv/bin/python", "-m", "streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless", "true"]
