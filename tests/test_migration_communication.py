"""Tests pour la bannière de communication migration (issue-021 / #143, #184)."""

from __future__ import annotations

import json

import pytest
from src.migration_communication import (
    INDEX_HTML_BANNER_PLACEHOLDER,
    MIGRATION_INFO_BANNER_ENV,
    migration_banner_html_section,
    migration_info_banner_html_fragment,
    migration_info_banner_text,
    parse_migration_banner_config,
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
    assert "ds-banner--info" in frag
    assert "ds-migration-banner" in frag
    assert 'role="region"' in frag


def test_banner_text_json_returns_message_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        MIGRATION_INFO_BANNER_ENV,
        json.dumps(
            {"message": "Corps court", "help_url": "https://example.org/x"},
            ensure_ascii=False,
        ),
    )
    assert migration_info_banner_text() == "Corps court"


def test_parse_migration_banner_config_none_when_unset(clear_banner_env: None) -> None:
    assert parse_migration_banner_config() is None


def test_parse_valid_json_with_links(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "message": "L'interface évolue : gardez ce lien en favori.",
        "help_url": "https://example.org/aide",
        "help_label": "Guide utilisateur",
        "support_url": "mailto:support@example.org",
        "support_label": "Écrire au support",
        "calendar_note": "Basculer prévu : semaine du 2 juin.",
    }
    monkeypatch.setenv(MIGRATION_INFO_BANNER_ENV, json.dumps(payload, ensure_ascii=False))
    cfg = parse_migration_banner_config()
    assert cfg is not None
    assert cfg.message == payload["message"]
    assert cfg.help_url == payload["help_url"]
    assert cfg.help_label == payload["help_label"]
    assert cfg.support_url == payload["support_url"]
    assert cfg.support_label == payload["support_label"]
    assert cfg.calendar_note == payload["calendar_note"]


def test_parse_rejects_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MIGRATION_INFO_BANNER_ENV, '"just a string"')
    assert parse_migration_banner_config() is None


def test_parse_rejects_missing_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MIGRATION_INFO_BANNER_ENV, json.dumps({"help_url": "https://x.org"}))
    assert parse_migration_banner_config() is None


def test_parse_strips_dangerous_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        MIGRATION_INFO_BANNER_ENV,
        json.dumps(
            {
                "message": "Info",
                "help_url": "javascript:alert(1)",
                "support_url": "https://safe.example/help",
            },
            ensure_ascii=False,
        ),
    )
    cfg = parse_migration_banner_config()
    assert cfg is not None
    assert cfg.help_url is None
    assert cfg.support_url == "https://safe.example/help"


def test_structured_html_escapes_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        MIGRATION_INFO_BANNER_ENV,
        json.dumps({"message": 'Texte <script>x</script> & "quotes"'}, ensure_ascii=False),
    )
    cfg = parse_migration_banner_config()
    assert cfg is not None
    html = migration_banner_html_section(cfg)
    assert "ds-migration-banner" in html
    assert 'role="region"' in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_structured_html_includes_actionable_links(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        MIGRATION_INFO_BANNER_ENV,
        json.dumps(
            {
                "message": "Besoin d'aide ?",
                "help_url": "https://docs.example/page",
                "help_label": "Documentation",
            },
            ensure_ascii=False,
        ),
    )
    cfg = parse_migration_banner_config()
    assert cfg is not None
    html = migration_banner_html_section(cfg)
    assert 'href="https://docs.example/page"' in html
    assert "rel=" in html and "noopener" in html
    assert ">Documentation<" in html


def test_index_template_contains_placeholder() -> None:
    from src.webapp import index_template

    assert INDEX_HTML_BANNER_PLACEHOLDER in index_template.INDEX_HTML
