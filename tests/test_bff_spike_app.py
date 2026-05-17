"""Tests du spike FastAPI BFF (issue-006) : garde-fous ``require_role`` / ``require_admin``."""

from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy.engine import Engine
from src.bff_spike_app import SPIKE_ACTOR_HEADER, create_bff_spike_app
from starlette.testclient import TestClient


def test_spike_app_exposes_migration_routes() -> None:
    """Sans base : vérifie l'enregistrement des routes (régression structurelle)."""
    app = create_bff_spike_app(engine=MagicMock())
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/migration-spike/v1/projects/{project_id}/entries-summary" in paths
    assert "/migration-spike/v1/projects/{project_id}/settings/active-preset" in paths


def test_missing_actor_header_returns_401_envelope() -> None:
    app = create_bff_spike_app(engine=MagicMock())
    client = TestClient(app)
    r = client.get("/migration-spike/v1/projects/p_x/entries-summary")
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["code"] == "AUTH_SESSION_EXPIRED"


@pytest.fixture()
def pg_engine() -> Engine:
    """PostgreSQL requis : ``ensure_schema`` n'est pas compatible SQLite (index fonctionnel)."""
    url = (os.environ.get("PYTEST_BFF_SPIKE_DATABASE_URL") or "").strip()
    if not url:
        pytest.skip("PYTEST_BFF_SPIKE_DATABASE_URL non défini (tests intégration spike BFF).")
    from src.database import create_db_engine, ensure_schema

    engine = create_db_engine(url)
    ensure_schema(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def _unique_email(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}@example.invalid"


@pytest.fixture()
def owned_project(pg_engine: Engine) -> dict[str, Any]:
    """Propriétaire + projet frais pour isoler les tests."""
    from src.database import create_project, upsert_user_from_su

    owner = upsert_user_from_su(
        pg_engine,
        f"su_{uuid.uuid4().hex[:16]}",
        _unique_email("owner"),
        "Owner",
    )
    pid = create_project(pg_engine, owner.user_id, f"P-{uuid.uuid4().hex[:8]}")
    stranger = upsert_user_from_su(
        pg_engine, f"su_{uuid.uuid4().hex[:16]}", _unique_email("stranger"), "Stranger"
    )
    return {
        "engine": pg_engine,
        "project_id": pid,
        "owner_id": owner.user_id,
        "stranger_id": stranger.user_id,
    }


def test_entries_summary_happy_path_and_opaque_denial(owned_project: dict[str, Any]) -> None:
    from src.database import ENTRY_COLUMNS

    app = create_bff_spike_app(engine=owned_project["engine"])
    client = TestClient(app)
    pid = str(owned_project["project_id"])
    owner_id = str(owned_project["owner_id"])
    stranger_id = str(owned_project["stranger_id"])

    ok = client.get(
        f"/migration-spike/v1/projects/{pid}/entries-summary",
        headers={SPIKE_ACTOR_HEADER: owner_id},
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["project_id"] == pid
    assert payload["row_count"] == 0
    assert payload["entry_column_count"] == len(ENTRY_COLUMNS)

    denied = client.get(
        f"/migration-spike/v1/projects/{pid}/entries-summary",
        headers={SPIKE_ACTOR_HEADER: stranger_id},
    )
    assert denied.status_code == 404
    err = denied.json()["error"]
    assert err["code"] == "NOT_FOUND_GENERIC"


def test_patch_active_preset_returns_canonical_settings(owned_project: dict[str, Any]) -> None:
    app = create_bff_spike_app(engine=owned_project["engine"])
    client = TestClient(app)
    pid = str(owned_project["project_id"])
    owner_id = str(owned_project["owner_id"])
    stranger_id = str(owned_project["stranger_id"])

    r = client.patch(
        f"/migration-spike/v1/projects/{pid}/settings/active-preset",
        headers={SPIKE_ACTOR_HEADER: owner_id},
        json={"active_preset_key": "pro"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["project_id"] == pid
    assert body["settings"]["active_preset_key"] == "pro"

    denied = client.patch(
        f"/migration-spike/v1/projects/{pid}/settings/active-preset",
        headers={SPIKE_ACTOR_HEADER: stranger_id},
        json={"active_preset_key": "roman"},
    )
    assert denied.status_code == 404
    assert denied.json()["error"]["code"] == "NOT_FOUND_GENERIC"
