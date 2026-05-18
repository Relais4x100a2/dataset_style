"""Super-admin annuaire paginé (issue-018) — GET /api/super-admin/accounts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from src.database import AccountAdminRow, UserRecord
from src.webapp import deps as webapp_deps
from src.webapp.app import create_slice_app


def test_super_admin_accounts_returns_403_for_non_super_admin() -> None:
    app = create_slice_app(engine=MagicMock())
    user = UserRecord(
        user_id="u1",
        email="curator@example.com",
        display_name="C",
        is_super_admin=False,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    with TestClient(app) as client:
        r = client.get(
            "/api/super-admin/accounts",
            headers={"Authorization": "Bearer t"},
        )
    assert r.status_code == 403
    body = r.json()
    assert body["error"]["code"] == "FORBIDDEN"
    assert "totalActiveAccounts" not in body
    assert "accounts" not in body


def test_super_admin_accounts_returns_400_when_page_out_of_range() -> None:
    app = create_slice_app(engine=MagicMock())
    user = UserRecord(
        user_id="admin1",
        email="admin@example.com",
        display_name="A",
        is_super_admin=True,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    with patch("src.webapp.app.count_users_for_admin", return_value=0):
        with TestClient(app) as client:
            r = client.get(
                "/api/super-admin/accounts?page=2&page_size=25",
                headers={"Authorization": "Bearer t"},
            )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "BAD_REQUEST"


def test_super_admin_accounts_returns_200_contract() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    user = UserRecord(
        user_id="admin1",
        email="admin@example.com",
        display_name="A",
        is_super_admin=True,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    row = AccountAdminRow(
        user_id="u_target",
        email="bob@example.com",
        display_name="Bob",
        is_super_admin=False,
        project_count=2,
        last_login_at="",
        entries_total=5,
        entries_validated=1,
    )
    with (
        patch("src.webapp.app.count_users_for_admin", return_value=1),
        patch("src.webapp.app.list_accounts_for_super_admin", return_value=[row]),
    ):
        with TestClient(app) as client:
            r = client.get(
                "/api/super-admin/accounts?page=1&page_size=10",
                headers={"Authorization": "Bearer t"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["totalActiveAccounts"] == 1
    assert body["page"] == 1
    assert body["pageSize"] == 10
    assert body["totalPages"] == 1
    assert len(body["accounts"]) == 1
    acc = body["accounts"][0]
    assert acc["accountId"] == "u_target"
    assert acc["email"] == "bob@example.com"
    assert acc["displayName"] == "Bob"
    assert acc["isSuperAdmin"] is False
    assert acc["ownedProjects"] == 2
    assert acc["entriesTotal"] == 5
    assert acc["entriesValidated"] == 1
    assert acc["lastLoginAt"] is None
    raw = r.text
    assert "su_user_id" not in raw.lower()


def test_super_admin_accounts_forbids_query_page_size_below_min() -> None:
    app = create_slice_app(engine=MagicMock())
    user = UserRecord(
        user_id="admin1",
        email="admin@example.com",
        display_name="A",
        is_super_admin=True,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    with TestClient(app) as client:
        r = client.get(
            "/api/super-admin/accounts?page=1&page_size=9",
            headers={"Authorization": "Bearer t"},
        )
    assert r.status_code == 422
