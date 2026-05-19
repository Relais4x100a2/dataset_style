"""Tests pour la bannière de communication migration (issue-021 / #143)."""

from __future__ import annotations

import pytest
from src.migration_communication import (
    INDEX_HTML_BANNER_PLACEHOLDER,
    MIGRATION_INFO_BANNER_ENV,
    migration_info_banner_html_fragment,
    migration_info_banner_text,
)


@pytest.fixture
def clear_banner_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retire la variable de bannière pour des tests isolés."""
    monkeypatch.delenv(MIGRATION_INFO_BANNER_ENV, raising=False)


def test_migration_info_banner_text_none_when_unset(
    clear_banner_env: None,
) -> None:
    assert migration_info_banner_text() is None


def test_migration_info_banner_text_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MIGRATION_INFO_BANNER_ENV, "  Bonjour  ")
    assert migration_info_banner_text() == "Bonjour"


def test_migration_info_banner_html_fragment_empty_when_unset(
    clear_banner_env: None,
) -> None:
    assert migration_info_banner_html_fragment() == ""


def test_migration_info_banner_html_escapes_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        MIGRATION_INFO_BANNER_ENV,
        'Suite <script>alert(1)</script> à "lire"',
    )
    frag = migration_info_banner_html_fragment()
    assert "<script>" not in frag
    assert "&lt;script&gt;" in frag
    assert "migration-info" in frag


def test_index_template_contains_placeholder() -> None:
    from src.webapp import index_template

    assert INDEX_HTML_BANNER_PLACEHOLDER in index_template.INDEX_HTML
