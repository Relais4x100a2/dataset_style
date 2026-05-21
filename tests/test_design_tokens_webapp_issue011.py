"""Tokens design webapp (issue-011 / GitHub #185) : CSS canonique et lien shell."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from src.webapp.app import create_slice_app
from src.webapp.index_template import INDEX_HTML


def _design_tokens_css_text() -> str:
    path = Path(__file__).resolve().parents[1] / "src" / "webapp" / "static" / "design_tokens.css"
    return path.read_text(encoding="utf-8")


def test_index_html_links_shared_design_tokens_stylesheet() -> None:
    """Le shell ne doit plus embarquer de styles inline : feuille partagée sous /static."""
    assert 'href="/static/design_tokens.css"' in INDEX_HTML
    assert "<style>" not in INDEX_HTML


def test_design_tokens_css_defines_vision_tokens_and_button_layers() -> None:
    """Couvre les jetons listés dans docs/design_tokens_webapp.md (vision UI)."""
    css = _design_tokens_css_text()
    for token in (
        "--ds-color-action-fill:",
        "--ds-color-on-action-fill:",
        "--ds-space-1:",
        "--ds-space-6:",
        "--ds-radius-sm:",
        "--ds-chip-bg:",
        "--ds-color-row-hover:",
        "--ds-skeleton-base:",
    ):
        assert token in css
    assert ".ds-btn--primary" in css
    assert ".ds-btn--secondary" in css
    assert ".ds-btn--danger" in css
    assert "prefers-contrast" in css
    assert "forced-colors" in css
    assert "prefers-reduced-motion" in css


def test_static_mount_serves_design_tokens_css() -> None:
    """La feuille est servie comme asset statique (compose / CapRover)."""
    app = create_slice_app(engine=MagicMock())
    with TestClient(app) as client:
        r = client.get("/static/design_tokens.css")
    assert r.status_code == 200
    assert "ds-btn--primary" in r.text
    assert r.headers.get("content-type", "").startswith("text/css")
