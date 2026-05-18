"""Tests du client HTTP SuperTokens partagé (issue-007)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src import supertokens_recipe_client as stc
from src.api_errors import AuthSessionExpiredError


def test_verify_access_token_raises_on_non_ok_status() -> None:
    """Une réponse session non OK doit lever AuthSessionExpiredError."""
    with patch.dict("os.environ", {"SUPERTOKENS_CONNECTION_URI": "http://st:3567"}, clear=False):
        with patch.object(stc, "recipe_post", return_value={"status": "TRY_REFRESH_TOKEN"}):
            with pytest.raises(AuthSessionExpiredError):
                stc.verify_access_token("bad-token")


def test_verify_access_token_returns_payload_on_ok() -> None:
    with patch.dict("os.environ", {"SUPERTOKENS_CONNECTION_URI": "http://st:3567"}, clear=False):
        payload = {"status": "OK", "userId": "su-1"}
        with patch.object(stc, "recipe_post", return_value=payload) as m:
            out = stc.verify_access_token("good-token")
            assert out == payload
            m.assert_called_once()
            args, kwargs = m.call_args
            assert args[0] == "/recipe/session/verify"
            assert args[1]["accessToken"] == "good-token"


def test_signin_email_password_falls_back_to_flat_payload() -> None:
    """Compat formFields vs email/password comme pour Streamlit."""
    with patch.dict("os.environ", {"SUPERTOKENS_CONNECTION_URI": "http://st:3567"}, clear=False):
        mock_post = MagicMock(
            side_effect=[
                RuntimeError("Field name 'email' is invalid in JSON input"),
                {"status": "OK"},
            ]
        )
        with patch.object(stc, "recipe_post", mock_post):
            out = stc.signin_email_password("u@example.com", "secret")
            assert out["status"] == "OK"
        assert mock_post.call_count == 2
        second = mock_post.call_args_list[1]
        assert second[0][0] == "/recipe/signin"
        assert second[0][1] == {"email": "u@example.com", "password": "secret"}
