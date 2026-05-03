# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY main.py ./
COPY src ./src

RUN uv sync --frozen --no-dev

EXPOSE 8501

CMD ["uv", "run", "python", "-m", "streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless", "true"]
