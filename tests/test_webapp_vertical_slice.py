"""Tests du slice vertical FastAPI (issue-007) — mocks des primitives ``database``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.database import ProjectRecord
from src.webapp import deps as webapp_deps
from src.webapp.app import create_slice_app


def test_projects_requires_bearer() -> None:
    app = create_slice_app(engine=MagicMock())
    with TestClient(app) as client:
        r = client.get("/api/projects")
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["code"] == "AUTH_SESSION_EXPIRED"


def test_projects_returns_owned_projects() -> None:
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    fake_projects = [
        ProjectRecord(project_id="p1", name="Alpha", role="admin"),
    ]
    with patch("src.webapp.app.list_projects_for_user", return_value=fake_projects):
        with TestClient(app) as client:
            r = client.get("/api/projects", headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    assert r.json()["projects"][0]["id"] == "p1"


def test_patch_entry_calls_update_project_entries() -> None:
    """Vérifie que PATCH enchaîne load + update (relecture côté client via GET séparé)."""
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    df = pd.DataFrame(
        [
            {
                "id": "e1",
                "project_id": "p1",
                "date": "",
                "type": "",
                "structure": "",
                "ton": "",
                "format": "",
                "public": "",
                "input": "a",
                "output": "b",
                "statut": "En cours",
                "notes": "",
            }
        ]
    )
    with (
        patch("src.webapp.entry_mutations.load_project_entries", return_value=df) as load_m,
        patch("src.webapp.entry_mutations.update_project_entries") as save_m,
    ):
        with TestClient(app) as client:
            r = client.patch(
                "/api/projects/p1/entries/e1",
                headers={"Authorization": "Bearer t"},
                json={"input": "x", "output": "y"},
            )
    assert r.status_code == 200
    load_m.assert_called_once()
    save_m.assert_called_once()


def test_export_jsonl_lfm2_includes_system_when_stylometry_columns_present() -> None:
    """issue-015: JSONL export matches Streamlit (``include_stylometry=True``)."""
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    df = pd.DataFrame(
        [
            {
                "id": "e1",
                "project_id": "p1",
                "statut": "Fait et validé",
                "input": "draft",
                "output": "final",
                "type": "T1",
                "structure": "S1",
                "ton": "tonal",
                "format": "fmt",
                "public": "pub",
                "date": "",
                "notes": "",
                "_ttr": "0.42",
            }
        ]
    )
    with patch("src.webapp.app.load_project_entries", return_value=df):
        with TestClient(app) as client:
            r = client.get(
                "/api/projects/p1/export.jsonl",
                params={"scope": "validated_only", "format": "lfm2"},
                headers={"Authorization": "Bearer t"},
            )
    assert r.status_code == 200
    first = r.text.strip().split("\n")[0]
    assert '"role": "system"' in first


def test_export_rejects_when_row_count_exceeds_configured_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """issue-015: optional ``WEBAPP_EXPORT_MAX_ROWS`` avoids unbounded responses."""
    monkeypatch.setenv("WEBAPP_EXPORT_MAX_ROWS", "1")
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    df = pd.DataFrame(
        [
            {
                "id": "e1",
                "project_id": "p1",
                "statut": "Fait et validé",
                "input": "a",
                "output": "b",
                "type": "",
                "structure": "",
                "ton": "",
                "format": "",
                "public": "",
                "date": "",
                "notes": "",
            },
            {
                "id": "e2",
                "project_id": "p1",
                "statut": "Fait et validé",
                "input": "c",
                "output": "d",
                "type": "",
                "structure": "",
                "ton": "",
                "format": "",
                "public": "",
                "date": "",
                "notes": "",
            },
        ]
    )
    with patch("src.webapp.app.load_project_entries", return_value=df):
        with TestClient(app) as client:
            r_csv = client.get(
                "/api/projects/p1/export.csv",
                params={"scope": "validated_only"},
                headers={"Authorization": "Bearer t"},
            )
            r_jsonl = client.get(
                "/api/projects/p1/export.jsonl",
                params={"scope": "validated_only"},
                headers={"Authorization": "Bearer t"},
            )
    assert r_csv.status_code == 413
    assert r_jsonl.status_code == 413
    for body in (r_csv.json(), r_jsonl.json()):
        assert body["error"]["code"] == "EXPORT_PAYLOAD_TOO_LARGE"


def test_export_csv_uses_dataframe_for_export() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    df = pd.DataFrame(
        [
            {
                "id": "e1",
                "project_id": "p1",
                "statut": "En cours",
                "input": "i",
                "output": "o",
                "type": "",
                "structure": "",
                "ton": "",
                "format": "",
                "public": "",
                "date": "",
                "notes": "",
            }
        ]
    )
    with patch("src.webapp.app.load_project_entries", return_value=df):
        with TestClient(app) as client:
            r = client.get(
                "/api/projects/p1/export.csv",
                params={"scope": "validated_only"},
                headers={"Authorization": "Bearer t"},
            )
    assert r.status_code == 200
    lines = [ln for ln in r.text.strip().splitlines() if ln.strip()]
    assert len(lines) == 1
