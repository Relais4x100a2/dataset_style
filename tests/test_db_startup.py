"""Tests pour la couche messages d'erreur base de données au démarrage."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError
from src.db_startup import (
    DbFailureCategory,
    classify_database_startup_error,
    effective_database_url,
    is_development_ui,
    technical_hint_for_dev,
    user_facing_summary,
)


def test_effective_database_url_prefers_environment_over_streamlit() -> None:
    """La valeur issue de la configuration runtime (env) prime sur les secrets Streamlit."""
    env = {"DATABASE_URL": "postgresql://from-env"}
    assert effective_database_url(env, "postgresql://from-streamlit") == "postgresql://from-env"


def test_effective_database_url_falls_back_to_streamlit_secret() -> None:
    assert (
        effective_database_url({}, "postgresql://from-streamlit") == "postgresql://from-streamlit"
    )


def test_effective_database_url_empty_when_unconfigured() -> None:
    assert effective_database_url({}, None) == ""
    assert effective_database_url({"DATABASE_URL": ""}, None) == ""


def test_effective_database_url_strips_whitespace() -> None:
    assert effective_database_url({"DATABASE_URL": "  trimmed  "}, None) == "trimmed"


def test_user_facing_summary_mentions_instance_administrator() -> None:
    """Critère produit : consigne explicite pour l'invité non technique."""
    for category in ("missing_url", "invalid_config", "connection", "other"):
        text = user_facing_summary(category)
        assert "administrateur de l'instance" in text.lower()


def test_user_facing_summary_never_exposes_raw_env_name() -> None:
    """Les messages utilisateur ne doivent pas reprendre la chaîne brute « DATABASE_URL »."""
    categories: tuple[DbFailureCategory, ...] = (
        "missing_url",
        "invalid_config",
        "connection",
        "other",
    )
    for category in categories:
        text = user_facing_summary(category)
        assert "DATABASE_URL" not in text


def test_classify_operational_error_as_connection() -> None:
    exc = OperationalError("statement", {}, None)
    assert classify_database_startup_error(exc) == "connection"


def test_classify_message_connection_refused() -> None:
    exc = RuntimeError("could not connect to server: Connection refused")
    assert classify_database_startup_error(exc) == "connection"


def test_is_development_ui_respects_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("STREAMLIT_ENV", raising=False)
    monkeypatch.delenv("SHOW_DB_TECHNICAL_ERRORS", raising=False)
    assert is_development_ui() is False

    monkeypatch.setenv("APP_ENV", "development")
    assert is_development_ui() is True

    monkeypatch.setenv("APP_ENV", "production")
    assert is_development_ui() is False


def test_is_development_ui_show_db_technical_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("SHOW_DB_TECHNICAL_ERRORS", "1")
    assert is_development_ui() is True


def test_technical_hint_missing_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    hint = technical_hint_for_dev(None, category="missing_url")
    assert "DATABASE_URL" in hint


def test_sanitize_redacts_password_like_patterns() -> None:
    from src.db_startup import _sanitize_technical_message

    raw = "connect user=x password=secret123 host=db"
    out = _sanitize_technical_message(raw)
    assert "secret123" not in out
    assert "redacted" in out.lower()
