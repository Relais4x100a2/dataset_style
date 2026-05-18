"""Super-admin invite par e-mail (issue-017) — endpoint + contrat invitation-only au démarrage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.database import UserRecord
from src.mailer import MailDeliveryResult
from src.webapp import deps as webapp_deps
from src.webapp.app import create_slice_app


def test_lifespan_raises_when_invitation_only_contract_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_ENFORCE_INVITATION_ONLY", "true")
    monkeypatch.setenv("SUPERTOKENS_SIGNUP_DISABLED", "false")
    app = create_slice_app(engine=MagicMock())
    with pytest.raises(RuntimeError, match="SUPERTOKENS_SIGNUP_DISABLED"):
        with TestClient(app):
            pass


def test_super_admin_invite_returns_403_for_non_super_admin() -> None:
    app = create_slice_app(engine=MagicMock())
    user = UserRecord(
        user_id="u1",
        email="curator@example.com",
        display_name="C",
        is_super_admin=False,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    with TestClient(app) as client:
        r = client.post(
            "/api/super-admin/invite",
            headers={"Authorization": "Bearer t"},
            json={"email": "new@example.com"},
        )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "FORBIDDEN"


def test_super_admin_invite_returns_200_without_raw_token_in_json() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    user = UserRecord(
        user_id="admin1",
        email="admin@example.com",
        display_name="A",
        is_super_admin=True,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    secret_token = "x" * 80
    fake_link = f"https://app.example/?flow=set-password&token={secret_token}"
    delivery = MailDeliveryResult(mode="dev", delivered=True, preview="masked-preview")
    with (
        patch(
            "src.webapp.super_admin_invite.create_invitation_link",
            return_value=fake_link,
        ),
        patch(
            "src.webapp.super_admin_invite.send_account_link_email",
            return_value=delivery,
        ),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/super-admin/invite",
                headers={"Authorization": "Bearer t"},
                json={"email": "  New@Example.com "},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mailMode"] == "dev"
    assert "masked-preview" in body["message"]
    assert secret_token not in body["message"]
    raw = r.text
    assert secret_token not in raw


def test_super_admin_invite_smtp_success_message() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    user = UserRecord(
        user_id="admin1",
        email="admin@example.com",
        display_name="A",
        is_super_admin=True,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    delivery = MailDeliveryResult(mode="smtp", delivered=True, preview="Email envoyé via SMTP.")
    with (
        patch(
            "src.webapp.super_admin_invite.create_invitation_link",
            return_value="https://app.example/?flow=set-password&token=abc",
        ),
        patch(
            "src.webapp.super_admin_invite.send_account_link_email",
            return_value=delivery,
        ),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/super-admin/invite",
                headers={"Authorization": "Bearer t"},
                json={"email": "bob@example.com"},
            )
    assert r.status_code == 200
    assert r.json()["mailMode"] == "smtp"
    assert "Invitation envoyée" in r.json()["message"]
