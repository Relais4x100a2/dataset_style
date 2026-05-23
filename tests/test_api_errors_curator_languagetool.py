"""Catalogue d'erreurs curateur LanguageTool (issue-006 / alignement api_errors)."""

from __future__ import annotations

from src.api_errors import (
    CURATOR_LANGUAGETOOL_UNAVAILABLE,
    curator_languagetool_unavailable_envelope,
)


def test_curator_languagetool_unavailable_envelope_matches_catalog() -> None:
    """L'enveloppe 503 LT reprend le code stable et les textes FR du catalogue."""
    body = curator_languagetool_unavailable_envelope()
    err = body["error"]
    assert err["code"] == CURATOR_LANGUAGETOOL_UNAVAILABLE
    assert err["detail"] is None
    assert "suggested_action" in err
    assert len(err["title"]) > 3
    assert "LanguageTool" in err["message"] or "linguistique" in err["title"].lower()
