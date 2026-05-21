"""Coquille webapp : panneau assistance curateur (issue-006)."""

from __future__ import annotations

from src.webapp.index_template import INDEX_HTML


def test_index_html_exposes_curator_assistance_section() -> None:
    """Le shell minimal doit brancher l'UI sur les routes curator (sans persistance implicite)."""
    assert "Assistance IA" in INDEX_HTML
    assert "LanguageTool" in INDEX_HTML
    assert "hx-indicator" in INDEX_HTML
    assert "curator/llm-generate" in INDEX_HTML
    assert "curator/languagetool-check" in INDEX_HTML
