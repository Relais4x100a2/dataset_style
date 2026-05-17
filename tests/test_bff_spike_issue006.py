"""Spike BFF issue-006 : routes JSON, garde-fous ``database`` uniquement, enveloppe d'erreurs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from src.api_errors import NOT_FOUND_GENERIC, TenantResourceOpaqueDenial


def _make_client(engine: Engine, *, actor_user_id: str) -> TestClient:
    from src.bff_spike_app import create_spike_bff_app

    app = create_spike_bff_app(engine, actor_user_id_factory=lambda: actor_user_id)
    return TestClient(app, raise_server_exceptions=True)


def test_get_entries_returns_rows_and_calls_load_project_entries() -> None:
    engine = MagicMock(spec=Engine)
    df = pd.DataFrame(
        [
            {
                "id": "e1",
                "project_id": "p_x",
                "date": "2024-01-01",
                "type": "t",
                "structure": "",
                "ton": "",
                "format": "",
                "public": "",
                "input": "in",
                "output": "out",
                "statut": "",
                "notes": "",
            }
        ]
    )
    with patch("src.bff_spike_app.load_project_entries", return_value=df) as load_mock:
        client = _make_client(engine, actor_user_id="u_owner")
        resp = client.get("/issue-006-spike/projects/p_x/entries")
    assert resp.status_code == 200
    body = resp.json()
    assert "entries" in body
    assert len(body["entries"]) == 1
    assert body["entries"][0]["id"] == "e1"
    load_mock.assert_called_once_with(engine, "p_x", "u_owner")


def test_get_entries_tenant_denial_maps_to_not_found_envelope() -> None:
    engine = MagicMock(spec=Engine)
    with patch(
        "src.bff_spike_app.load_project_entries",
        side_effect=TenantResourceOpaqueDenial(),
    ):
        client = _make_client(engine, actor_user_id="u_intruder")
        resp = client.get("/issue-006-spike/projects/p_foreign/entries")
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == NOT_FOUND_GENERIC
    assert "p_foreign" not in err["message"]


def test_patch_settings_returns_canonical_state_after_merge() -> None:
    engine = MagicMock(spec=Engine)
    from src.database import ProjectSettings

    before = ProjectSettings(active_preset_key="roman", llm_model="old-model")
    after = ProjectSettings(active_preset_key="news", llm_model="old-model")

    with (
        patch("src.bff_spike_app.require_admin") as admin_mock,
        patch("src.bff_spike_app.get_project_settings", side_effect=[before, after]) as get_mock,
        patch("src.bff_spike_app.update_project_settings") as upd_mock,
    ):
        client = _make_client(engine, actor_user_id="u_owner")
        resp = client.patch(
            "/issue-006-spike/projects/p_x/settings",
            json={"active_preset_key": "news"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["settings"]["active_preset_key"] == "news"
    assert body["settings"]["llm_model"] == "old-model"
    admin_mock.assert_called_once_with(engine, "p_x", "u_owner")
    assert get_mock.call_count == 2
    upd_mock.assert_called_once()
    merged: ProjectSettings = upd_mock.call_args[0][2]
    assert merged.active_preset_key == "news"
    assert merged.llm_model == "old-model"


def test_patch_settings_propagates_operational_error_envelope() -> None:
    from sqlalchemy.exc import OperationalError

    engine = MagicMock(spec=Engine)
    with patch(
        "src.bff_spike_app.require_admin",
        side_effect=OperationalError("stmt", {}, orig=Exception("db-down")),
    ):
        client = _make_client(engine, actor_user_id="u_owner")
        resp = client.patch(
            "/issue-006-spike/projects/p_x/settings",
            json={"active_preset_key": "x"},
        )
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "DB_UNAVAILABLE"


def test_bff_spike_router_only_imports_database_for_data_guards() -> None:
    """Contrat léger : pas de RBAC ad hoc dans le module spike (hors identité injectée)."""
    from pathlib import Path

    text = Path(__file__).resolve().parents[1] / "src" / "bff_spike_app.py"
    src = text.read_text(encoding="utf-8")
    assert "require_admin" in src
    assert "load_project_entries" in src
    assert "get_role(" not in src  # pas de branche métier locale sur rôle
