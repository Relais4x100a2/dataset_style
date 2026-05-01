"""
Tests pour le moteur NLP (sans appel réseau sauf mocks) : URL LanguageTool, paliers,
cohérence, corrections API.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from src.nlp_engine import (
    coherence_level,
    compute_coherence_score,
    corriger_texte_fr,
    languagetool_check_url,
    normalize_signature,
    palier_details,
    translate_trigram,
)


def test_languagetool_check_url_public_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sans LANGUAGETOOL_BASE_URL, l'URL publique /v2/check est utilisée."""
    monkeypatch.delenv("LANGUAGETOOL_BASE_URL", raising=False)
    url = languagetool_check_url()
    assert url.endswith("/v2/check")
    assert "api.languagetool.org" in url


def test_languagetool_check_url_custom_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avec base locale, l'URL est jointe correctement."""
    monkeypatch.setenv("LANGUAGETOOL_BASE_URL", "http://srv-captain--lt:8010")
    assert languagetool_check_url() == "http://srv-captain--lt:8010/v2/check"


def test_palier_details_ratio() -> None:
    """Valeur faible → palier « Minimal »."""
    niveau, _ = palier_details("ratio", 1.0)
    assert niveau == "Minimal"


def test_normalize_signature_clamped() -> None:
    """La normalisation reste dans [0, 1]."""
    assert normalize_signature("Noms & adjectifs", 999.0) == 1.0
    assert normalize_signature("Noms & adjectifs", -5.0) == 0.0


def test_coherence_level_buckets() -> None:
    """Les seuils renvoient le label attendu."""
    assert coherence_level(85)[0] == "Excellent"
    assert coherence_level(50)[0] == "À surveiller"
    assert coherence_level(30)[0] == "Critique"


def test_translate_trigram() -> None:
    """Les tags POS sont traduits via la table interne."""
    s = translate_trigram("DET-NOUN-VERB")
    assert "Dét" in s
    assert "Verbe" in s


def test_compute_coherence_score_penalizes_repetition() -> None:
    """Les mots répétés réduisent le score par rapport à zéro répétition."""
    sig_fiche = {"Noms & adjectifs": 0.3, "Verbes d'action": 0.2}
    sig_ds = {"Noms & adjectifs": 0.3, "Verbes d'action": 0.2}
    s0, _ = compute_coherence_score(sig_fiche, sig_ds, [])
    s_rep, _ = compute_coherence_score(sig_fiche, sig_ds, ["a", "b", "c"])
    assert s_rep < s0


def test_corriger_texte_fr_empty() -> None:
    """Texte vide → chaîne vide sans requête."""
    assert corriger_texte_fr("") == ""
    assert corriger_texte_fr("   ") == ""


def test_corriger_texte_fr_applies_replacement() -> None:
    """Les remplacements LanguageTool sont appliqués du dernier offset au premier."""
    fake_json = {
        "matches": [
            {
                "offset": 0,
                "length": 5,
                "replacements": [{"value": "world"}],
            },
        ]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_json
    mock_resp.raise_for_status = MagicMock()

    with patch("src.nlp_engine.requests.post", return_value=mock_resp):
        out = corriger_texte_fr("hello there")
    assert out == "world there"


def test_corriger_texte_fr_invalid_json_raises() -> None:
    """Réponse non-JSON → ValueError."""
    mock_resp = MagicMock()
    mock_resp.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
    mock_resp.raise_for_status = MagicMock()

    with patch("src.nlp_engine.requests.post", return_value=mock_resp):
        with pytest.raises(ValueError, match="invalide"):
            corriger_texte_fr("test")
