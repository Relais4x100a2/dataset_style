"""Parité curateur webapp (issue-010) : contexte shell, projets, résolution projet actif."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from src.database import ProjectRecord, UserRecord
from src.tab_layout import main_tab_labels
from src.webapp import deps as webapp_deps
from src.webapp.app import create_slice_app


def test_api_me_returns_user_and_main_tab_labels() -> None:
    app = create_slice_app(engine=MagicMock())
    user = UserRecord(
        user_id="u1",
        email="a@b.c",
        display_name="Alice",
        is_super_admin=True,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    with TestClient(app) as client:
        r = client.get("/api/me", headers={"Authorization": "Bearer t"})
    assert r.status_code == 200
    body = r.json()
    assert body["user"]["appUserId"] == "u1"
    assert body["user"]["isSuperAdmin"] is True
    assert body["mainTabLabels"] == main_tab_labels(include_super_admin=True)


def test_api_me_non_super_admin_omits_super_admin_tab() -> None:
    app = create_slice_app(engine=MagicMock())
    user = UserRecord(user_id="u2", email="x@y.z", display_name="Bob", is_super_admin=False)
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    with TestClient(app) as client:
        r = client.get("/api/me", headers={"Authorization": "Bearer t"})
    assert r.status_code == 200
    assert r.json()["mainTabLabels"] == main_tab_labels(include_super_admin=False)


def test_projects_returns_active_project_id_from_hint() -> None:
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    fake_projects = [
        ProjectRecord(project_id="p-first", name="First", role="admin"),
        ProjectRecord(project_id="p-second", name="Second", role="admin"),
    ]
    with patch("src.webapp.app.list_projects_for_user", return_value=fake_projects):
        with TestClient(app) as client:
            r = client.get(
                "/api/projects",
                params={"active_hint": "p-second"},
                headers={"Authorization": "Bearer x"},
            )
    assert r.status_code == 200
    assert r.json()["activeProjectId"] == "p-second"


def test_projects_active_hint_stale_falls_back_to_first() -> None:
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    fake_projects = [ProjectRecord(project_id="p1", name="Only", role="admin")]
    with patch("src.webapp.app.list_projects_for_user", return_value=fake_projects):
        with TestClient(app) as client:
            r = client.get(
                "/api/projects",
                params={"active_hint": "gone"},
                headers={"Authorization": "Bearer x"},
            )
    assert r.json()["activeProjectId"] == "p1"


def test_post_projects_calls_create_project() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    with patch("src.webapp.app.create_project", return_value="p_new") as create_m:
        with TestClient(app) as client:
            r = client.post(
                "/api/projects",
                headers={"Authorization": "Bearer t"},
                json={"name": "Mon jeu", "description": "d"},
            )
    assert r.status_code == 200
    assert r.json() == {"id": "p_new", "status": "ok"}
    create_m.assert_called_once_with(engine, "u1", "Mon jeu", "d")


def test_post_projects_rejects_empty_name() -> None:
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    with TestClient(app) as client:
        r = client.post(
            "/api/projects",
            headers={"Authorization": "Bearer t"},
            json={"name": "   "},
        )
    assert r.status_code == 422


def test_delete_project_calls_delete_project_as_admin() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    with patch("src.webapp.app.delete_project_as_admin") as del_m:
        with TestClient(app) as client:
            r = client.delete(
                "/api/projects/p9",
                headers={"Authorization": "Bearer t"},
            )
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    del_m.assert_called_once_with(engine, "p9", "u1")
