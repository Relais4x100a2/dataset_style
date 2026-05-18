"""Tests du slice vertical FastAPI (issue-007) — mocks des primitives ``database``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
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
