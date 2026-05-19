"""Contrat UI minimal : gabarit HTML relie les endpoints curateur issue-013 (GitHub #135)."""

from __future__ import annotations

from src.webapp.index_template import INDEX_HTML


def test_index_template_wires_curator_llm_and_languagetool_endpoints() -> None:
    """La page d'accueil doit appeler les routes serveur (clés API jamais côté client)."""
    assert "/curator/llm-generate" in INDEX_HTML
    assert "/curator/languagetool-check" in INDEX_HTML
    assert "/curator/dimensions" in INDEX_HTML
    assert "data-curator-llm" in INDEX_HTML
    assert "data-curator-lt" in INDEX_HTML


def test_index_template_french_loading_copy_for_async_actions() -> None:
    """Feedback chargement visible pendant les appels réseau (latence perçue)."""
    assert "Génération en cours" in INDEX_HTML
    assert "Analyse LanguageTool" in INDEX_HTML
