"""Coquille webapp : assistance curateur IA/LT (issue-006, intégrée issue-013)."""

from __future__ import annotations

from src.webapp.index_template import INDEX_HTML


def test_index_html_exposes_curator_assistance_section() -> None:
    """Le shell branche l'UI sur les routes curator (sans persistance implicite)."""
    assert "Génération en cours" in INDEX_HTML
    assert "LanguageTool" in INDEX_HTML
    assert "curator/llm-generate" in INDEX_HTML
    assert "curator/languagetool-check" in INDEX_HTML
    assert "data-curator-llm" in INDEX_HTML
