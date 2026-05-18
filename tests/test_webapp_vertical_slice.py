"""Tests du slice vertical FastAPI (issue-007) — mocks des primitives ``database``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api_errors import TenantResourceOpaqueDenial
from src.database import ProjectRecord
from src.webapp import deps as webapp_deps
from src.webapp import entry_mutations
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
    """Vérifie que PATCH applique la persistance NLP partagée et renvoie ``entries`` fraîches."""
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
    df_after = df.copy()
    df_after.loc[df_after["id"] == "e1", "input"] = "x"
    df_after.loc[df_after["id"] == "e1", "output"] = "y"
    df_after["_stylometry_blob"] = "secret"
    with (
        patch("src.webapp.entry_mutations.load_project_entries", return_value=df) as load_m,
        patch(
            "src.webapp.entry_mutations.persist_edited_entry_with_nlp_cache",
        ) as save_m,
        patch("src.webapp.app.load_project_entries", return_value=df_after),
    ):
        with TestClient(app) as client:
            r = client.patch(
                "/api/projects/p1/entries/e1",
                headers={"Authorization": "Bearer t"},
                json={"input": "x", "output": "y"},
            )
    assert r.status_code == 200
    assert load_m.call_count >= 1
    save_m.assert_called_once()
    body = r.json()
    assert body["status"] == "ok"
    assert "entries" in body
    assert len(body["entries"]) == 1
    assert body["entries"][0]["input"] == "x"
    assert "_stylometry_blob" not in body["entries"][0]


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


def test_account_requires_bearer() -> None:
    app = create_slice_app(engine=MagicMock())
    with TestClient(app) as client:
        r = client.get("/api/account")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "AUTH_SESSION_EXPIRED"


def test_account_json_is_whitelisted_curator_fields() -> None:
    """issue-016 / #138 : pas d'exposition ``is_super_admin`` ni ``su_user_id``."""
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u_cur"
    with (
        patch(
            "src.webapp.app.get_user_email_display_name_by_id",
            return_value=("me@example.com", "Me Display"),
        ),
        patch("src.webapp.app.count_owned_projects", return_value=2),
        patch("src.webapp.app.count_active_memberships", return_value=1),
    ):
        with TestClient(app) as client:
            r = client.get("/api/account", headers={"Authorization": "Bearer t"})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"appUserId", "email", "displayName", "counts"}
    assert body["appUserId"] == "u_cur"
    assert body["email"] == "me@example.com"
    assert body["displayName"] == "Me Display"
    assert body["counts"] == {"ownedProjects": 2, "activeMemberships": 1}
    assert "is_super_admin" not in body
    assert "su_user_id" not in body
    assert "suUserId" not in body


def test_account_unknown_profile_returns_opaque_404() -> None:
    """Profil absent en base après auth (course rare) : déni opaque."""
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "ghost"
    with patch("src.webapp.app.get_user_email_display_name_by_id", return_value=None):
        with TestClient(app) as client:
            r = client.get("/api/account", headers={"Authorization": "Bearer t"})
    assert r.status_code == 404
    assert "error" in r.json()


def test_signout_returns_allowlisted_redirect_only() -> None:
    """Cible post-déconnexion : valeur demandée ignorée si hors liste."""
    app = create_slice_app(engine=MagicMock())
    with patch.dict("os.environ", {"WEBAPP_SIGNOUT_REDIRECT_ALLOWLIST": "/,/safe"}, clear=False):
        with TestClient(app) as client:
            r = client.post(
                "/api/auth/signout",
                json={"access_token": "x", "redirect_after": "https://evil.example/phish"},
            )
    assert r.status_code == 200
    assert r.json()["status"] == "signed_out"
    assert r.json()["redirect"] == "/"
    with patch.dict("os.environ", {"WEBAPP_SIGNOUT_REDIRECT_ALLOWLIST": "/,/safe"}, clear=False):
        with TestClient(app) as client:
            r2 = client.post(
                "/api/auth/signout",
                json={"access_token": "x", "redirect_after": "/safe"},
            )
    assert r2.json()["redirect"] == "/safe"


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


def test_post_entry_returns_entries_after_create() -> None:
    """POST doit renvoyer la vérité serveur (``entries``) comme après ``invalidate`` + rerun."""
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    df_one = pd.DataFrame(
        [
            {
                "id": "e_new",
                "project_id": "p1",
                "date": "",
                "type": "",
                "structure": "",
                "ton": "",
                "format": "",
                "public": "",
                "input": "i",
                "output": "o",
                "statut": "En cours",
                "notes": "",
                "_row_cache": "internal",
            }
        ]
    )
    with (
        patch("src.webapp.app.entry_mutations.append_minimal_entry", return_value="e_new"),
        patch("src.webapp.app.load_project_entries", return_value=df_one),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/projects/p1/entries",
                headers={"Authorization": "Bearer t"},
                json={"input": "i", "output": "o"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "e_new"
    assert body["status"] == "ok"
    assert len(body["entries"]) == 1
    assert body["entries"][0]["id"] == "e_new"
    assert "_row_cache" not in body["entries"][0]


def test_get_entries_with_edition_filter_invokes_prepare_and_filter() -> None:
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    df = pd.DataFrame(
        [
            {
                "id": "e1",
                "statut": "En cours",
                "_coherence_score": "",
                "input": "a",
                "output": "b",
                "project_id": "p1",
                "date": "",
                "type": "",
                "structure": "",
                "ton": "",
                "format": "",
                "public": "",
                "notes": "",
            }
        ]
    )
    empty = df.iloc[0:0].copy()
    with patch("src.webapp.app.load_project_entries", return_value=df):
        with patch("src.webapp.app.prepare_for_edition_tab", return_value=df) as prep_m:
            with patch(
                "src.webapp.app.filter_edition_entries_dataframe",
                return_value=empty,
            ) as fil_m:
                with TestClient(app) as client:
                    r = client.get(
                        "/api/projects/p1/entries",
                        params={"edition_score_mode": "na_only"},
                        headers={"Authorization": "Bearer t"},
                    )
    assert r.status_code == 200
    prep_m.assert_called_once()
    fil_m.assert_called_once()
    assert r.json()["entries"] == []


def test_get_entries_invalid_edition_score_mode_returns_400() -> None:
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    df = pd.DataFrame([{"id": "e1", "statut": "En cours", "input": "a", "output": "b"}])
    with patch("src.webapp.app.load_project_entries", return_value=df):
        with TestClient(app) as client:
            r = client.get(
                "/api/projects/p1/entries",
                params={"edition_score_mode": "not_a_mode"},
                headers={"Authorization": "Bearer t"},
            )
    assert r.status_code == 400
    assert "error" in r.json()


def test_append_minimal_entry_checks_role_before_project_settings() -> None:
    """Refus d'accès : ``require_role`` avant ``get_project_settings`` (pas de fuite IDOR)."""
    engine = MagicMock()
    calls: list[str] = []

    def _require_role(
        _engine: MagicMock,
        _project_id: str,
        _user_id: str,
        _allowed: tuple[str, ...],
    ) -> str:
        calls.append("require_role")
        raise TenantResourceOpaqueDenial()

    def _get_project_settings(*_a: object, **_k: object) -> MagicMock:
        calls.append("get_project_settings")
        return MagicMock()

    with (
        patch("src.webapp.entry_mutations.require_role", side_effect=_require_role),
        patch("src.webapp.entry_mutations.get_project_settings", side_effect=_get_project_settings),
    ):
        with pytest.raises(TenantResourceOpaqueDenial):
            entry_mutations.append_minimal_entry(
                engine, "p1", "u1", input_text="a", output_text="b"
            )
    assert calls == ["require_role"]


def test_list_entries_json_omits_leading_underscore_columns() -> None:
    """Les colonnes cache NLP (préfixe ``_``) ne doivent pas apparaître dans le JSON des entrées."""
    app = create_slice_app(engine=MagicMock())
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
                "_coherence_score": 42,
            }
        ]
    )
    with patch("src.webapp.app.load_project_entries", return_value=df):
        with TestClient(app) as client:
            r = client.get("/api/projects/p1/entries", headers={"Authorization": "Bearer t"})
    assert r.status_code == 200
    row = r.json()["entries"][0]
    assert row["id"] == "e1"
    assert "_coherence_score" not in row


def test_dashboard_endpoint_returns_dataset_quality_envelope() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    df = pd.DataFrame(
        [
            {
                "id": "e1",
                "project_id": "p1",
                "date": "",
                "type": "T",
                "structure": "",
                "ton": "",
                "format": "",
                "public": "",
                "input": "i",
                "output": "o",
                "statut": "En cours",
                "notes": "",
                "_coherence_score": "70",
                "_syntax_contrast": "0.5",
            }
        ]
    )
    with patch("src.webapp.app.load_project_entries", return_value=df):
        with TestClient(app) as client:
            r = client.get(
                "/api/projects/p1/dashboard",
                params={"dashboard_scope": "all"},
                headers={"Authorization": "Bearer t"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body.get("technical") is None
    assert "dataset_quality" in body
    assert body["dataset_quality"]["headline"]["total_rows"] == 1
    assert "client_contract" in body
