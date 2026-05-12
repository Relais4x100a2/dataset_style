# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Dataset Style Studio — a multi-tenant Streamlit app for curating literary datasets for LLM fine-tuning. Auth via SuperTokens (invitation-only), persistence via PostgreSQL 16.

### Running the stack

The full dev stack (PostgreSQL + SuperTokens + Streamlit app) is managed via Docker Compose:

```bash
cp .env.example .env   # first time only
make dev               # or: docker compose --env-file .env up
```

See `README.md` § "Dev local" for all `make` targets.

### Important gotcha: SuperTokens API key

The `SUPERTOKENS_CORE_API_KEY` / `SUPERTOKENS_API_KEY` values in `.env` **must** be at least 20 characters and contain only `=`, `-`, and alphanumeric chars (no underscores). The default in `.env.example` (`change_me_core_key_2026`) violates this — replace with e.g. `changeMeCoreKey2026DevLocal`.

### Lint

```bash
ruff check .
ruff format --check .
```

Ruff is configured in `pyproject.toml`. Install with `uv tool install ruff` (not a project dependency).

### Tests

```bash
uv run pytest -q
```

All tests run in-memory (SQLite) with mocks — no external services needed. `pytest` must be installed in the venv (`uv pip install pytest`).

### Build & run (non-Docker, local dev)

```bash
uv sync
uv run streamlit run main.py
```

Requires `DATABASE_URL` and `SUPERTOKENS_CONNECTION_URI` pointing to running Postgres and SuperTokens instances.

### CI

Defined in `.github/workflows/ci.yml`: ruff check + ruff format + pytest.
