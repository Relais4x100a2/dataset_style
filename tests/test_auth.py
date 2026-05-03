"""Tests unitaires auth (contrats sécurité)."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import auth


class AuthSecurityTests(unittest.TestCase):
    """Couvre les règles invitation-only et vérification email."""

    def test_extract_email_verified_true(self) -> None:
        payload = {"emails": [{"email": "x@example.com", "isVerified": True}]}
        self.assertTrue(auth._extract_email_verified(payload))

    def test_extract_email_verified_false(self) -> None:
        payload = {"emails": [{"email": "x@example.com", "isVerified": False}]}
        self.assertFalse(auth._extract_email_verified(payload))

    def test_invitation_only_policy_raises_if_signup_flag_missing(self) -> None:
        with patch.dict(
            os.environ,
            {"AUTH_ENFORCE_INVITATION_ONLY": "true", "SUPERTOKENS_SIGNUP_DISABLED": "false"},
            clear=False,
        ):
            with patch.object(auth.st, "session_state", {}):
                with self.assertRaises(RuntimeError):
                    auth.ensure_invitation_only_policy()

    def test_invitation_only_policy_noop_if_disabled(self) -> None:
        with patch.dict(os.environ, {"AUTH_ENFORCE_INVITATION_ONLY": "false"}, clear=False):
            with patch.object(auth, "_signup") as signup_mock:
                auth.ensure_invitation_only_policy()
                signup_mock.assert_not_called()

    def test_invitation_only_policy_marks_checked_when_core_blocks_signup(self) -> None:
        with patch.dict(
            os.environ,
            {"AUTH_ENFORCE_INVITATION_ONLY": "true", "SUPERTOKENS_SIGNUP_DISABLED": "true"},
            clear=False,
        ):
            state = {}
            with patch.object(auth.st, "session_state", state):
                with patch.object(auth, "_signup") as signup_mock:
                    signup_mock.return_value = {"status": "GENERAL_ERROR"}
                    auth.ensure_invitation_only_policy()
                    signup_mock.assert_called_once()
                    self.assertTrue(state.get("auth_invitation_policy_checked"))

    def test_saga_provider_done_error_increments_retry(self) -> None:
        op = SimpleNamespace(state="provider_done", retry_count=0)
        engine = object()
        with patch.object(auth, "require_super_admin"):
            with patch.object(auth, "create_deprovision_operation", return_value=op):
                with patch.object(auth, "mark_user_disabled"):
                    with patch.object(auth, "get_deprovision_operation", return_value=op):
                        with patch.object(
                            auth, "delete_user_if_detached", side_effect=RuntimeError("db-fail")
                        ):
                            with patch.object(
                                auth,
                                "record_deprovision_failure",
                                return_value=SimpleNamespace(state="provider_done"),
                            ) as failure_mock:
                                with self.assertRaises(RuntimeError):
                                    auth.revoke_account_with_saga(
                                        engine=engine,  # type: ignore[arg-type]
                                        actor_user_id="u_admin",
                                        target_user_id="u_target",
                                        operation_id="op_1",
                                        max_retries=3,
                                        detach_memberships=False,
                                    )
        failure_mock.assert_called_with(
            engine,
            operation_id="op_1",
            expected_state="provider_done",
            error_message="db-fail",
            max_retries=3,
            backoff_seconds=60,
        )

    def test_saga_raises_if_quarantined(self) -> None:
        op = SimpleNamespace(state="quarantined", retry_count=5)
        with patch.object(auth, "require_super_admin"):
            with patch.object(auth, "create_deprovision_operation", return_value=op):
                with self.assertRaises(RuntimeError):
                    auth.revoke_account_with_saga(
                        engine=object(),  # type: ignore[arg-type]
                        actor_user_id="u_admin",
                        target_user_id="u_target",
                        operation_id="op_2",
                        max_retries=5,
                    )


if __name__ == "__main__":
    unittest.main()
