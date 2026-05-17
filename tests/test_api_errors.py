"""Contrat erreurs API (issue-005) : codes stables, messages FR, politique IDOR."""

from __future__ import annotations

import logging

import pytest
from sqlalchemy.exc import OperationalError


def test_auth_session_expired_resolves_to_stable_code_and_401() -> None:
    from src.api_errors import (
        AUTH_SESSION_EXPIRED,
        AuthSessionExpiredError,
        resolve_exception_for_api,
    )

    resolved = resolve_exception_for_api(AuthSessionExpiredError(), include_technical_detail=False)
    assert resolved.code == AUTH_SESSION_EXPIRED
    assert resolved.http_status == 401
    assert "session" in resolved.message_fr.lower()


def test_tenant_resource_opaque_denial_maps_to_not_found_generic_without_leak() -> None:
    """Interdit vs absent : même code, même statut, pas d'identifiant projet dans le message."""
    from src.api_errors import (
        NOT_FOUND_GENERIC,
        TenantResourceOpaqueDenial,
        resolve_exception_for_api,
    )

    resolved = resolve_exception_for_api(
        TenantResourceOpaqueDenial(), include_technical_detail=False
    )
    assert resolved.code == NOT_FOUND_GENERIC
    assert resolved.http_status == 404
    assert "p_" not in resolved.message_fr
    assert "@" not in resolved.message_fr


def test_db_operational_error_maps_to_db_unavailable_503() -> None:
    from src.api_errors import DB_UNAVAILABLE, resolve_exception_for_api

    exc = OperationalError("SELECT 1", {}, orig=Exception("connection refused"))
    resolved = resolve_exception_for_api(exc, include_technical_detail=False)
    assert resolved.code == DB_UNAVAILABLE
    assert resolved.http_status == 503


def test_json_envelope_matches_issue006_shape() -> None:
    from src.api_errors import AuthSessionExpiredError, error_envelope_for_client

    body = error_envelope_for_client(
        AuthSessionExpiredError(),
        include_technical_detail=False,
    )
    err = body["error"]
    assert set(err.keys()) >= {"code", "message", "title", "suggested_action"}
    assert err["detail"] is None


def test_dev_detail_only_when_flag_true() -> None:
    from src.api_errors import error_envelope_for_client

    body = error_envelope_for_client(
        RuntimeError("secret-internals"), include_technical_detail=False
    )
    assert body["error"]["detail"] is None

    body_dev = error_envelope_for_client(
        RuntimeError("secret-internals"),
        include_technical_detail=True,
    )
    assert body_dev["error"]["detail"] is not None
    assert "RuntimeError" in body_dev["error"]["detail"]


def test_require_role_uses_opaque_denial_in_database_module() -> None:
    """Garde-fou : le refus d'accès projet ne doit pas exposer PermissionError textuel."""
    from pathlib import Path

    text = Path(__file__).resolve().parents[1] / "src" / "database.py"
    assert "raise TenantResourceOpaqueDenial()" in text.read_text(encoding="utf-8")


def test_log_structured_includes_code(caplog: pytest.LogCaptureFixture) -> None:
    from src.api_errors import AuthSessionExpiredError, log_resolved_api_error

    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_api_errors_struct")
    resolved = log_resolved_api_error(
        logger,
        AuthSessionExpiredError(),
        extra_context={"route": "/api/x"},
    )
    assert resolved.code == "AUTH_SESSION_EXPIRED"
    assert "AUTH_SESSION_EXPIRED" in caplog.text
