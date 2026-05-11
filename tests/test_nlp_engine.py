"""
Tests pour le moteur NLP (sans appel réseau sauf mocks) : URL LanguageTool, paliers,
cohérence, corrections API.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from src.nlp_engine import (
    CURATOR_MESSAGE_ADVICE_BALANCED,
    CURATOR_MESSAGE_STATS_UNAVAILABLE,
    RowNlpCacheResult,
    avg_signature_from_cache,
    coherence_level,
    compute_coherence_score,
    compute_row_cache,
    corriger_texte_fr,
    curator_advices_after_save,
    languagetool_check_url,
    normalize_signature,
    palier_details,
    post_save_stylometric_session_payload,
    qualitative_coherence_feedback,
    row_nlp_feedback_bundle_after_persist,
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


def test_qualitative_coherence_feedback_uses_buckets_in_range() -> None:
    """Dans 0–100, les libellés suivent ``coherence_level``."""
    assert qualitative_coherence_feedback(85) == coherence_level(85)
    assert qualitative_coherence_feedback(0) == coherence_level(0)


def test_qualitative_coherence_feedback_none_and_out_of_range() -> None:
    """Absent ou hors plage : pas de bucket ``coherence_level`` inventé."""
    label_none, _tone = qualitative_coherence_feedback(None)
    assert "Non calculé" in label_none
    label_bad, tone_bad = qualitative_coherence_feedback(150)
    assert "hors plage" in label_bad.lower()
    assert tone_bad == "warning"


def test_post_save_stylometric_session_payload_from_bundle() -> None:
    """Issue M5 : payload UI (métriques + conseils) sans Streamlit ni export JSONL."""
    sig = {"Noms & adjectifs": 0.35, "Verbes d'action": 0.25}
    df = pd.DataFrame(
        [
            {
                "id": "other",
                "input": "a",
                "output": "b " * 50,
                "_signature_json": json.dumps(sig),
                "_ratio": "1.2",
                "_ttr": "0.60",
                "_long_phrases": "14.0",
                "_coherence_score": "77",
                "_syntax_contrast": "0.40",
            },
            {
                "id": "target",
                "input": "in text",
                "output": "out text " * 30,
                "_signature_json": json.dumps(sig),
                "_ratio": "2.0",
                "_ttr": "0.70",
                "_long_phrases": "15.0",
                "_coherence_score": "77",
                "_syntax_contrast": "0.40",
            },
        ]
    )
    cols = [
        "_ratio",
        "_ttr",
        "_long_phrases",
        "_signature_json",
        "_coherence_score",
        "_syntax_contrast",
    ]
    bundle = row_nlp_feedback_bundle_after_persist(df, "target", None, cols)
    payload = post_save_stylometric_session_payload(bundle)
    assert payload["score"] == 77
    assert payload["ttr"] == "0.70"
    assert payload["contrast"] == "0.40"
    assert payload["level"] == coherence_level(77)[0]
    assert len(payload["advices"]) >= 1


def test_post_save_stylometric_payload_invalid_score_no_false_excellent() -> None:
    """Score persisté hors 0–100 : métrique absente, libellé explicite (pas Excellent)."""
    cache = {
        "_ratio": "2.0",
        "_ttr": "0.70",
        "_long_phrases": "15.0",
        "_coherence_score": "150",
        "_syntax_contrast": "0.1",
        "_signature_json": "",
    }
    stats = {"ratio": 2.0, "ttr": 0.7, "long_moy_phrases": 15.0, "mots_repetes": []}
    pkg = RowNlpCacheResult(cache, 150, {}, stats)
    payload = post_save_stylometric_session_payload(pkg)
    assert payload["score"] is None
    assert coherence_level(150)[0] == "Excellent"
    assert payload["level"] != "Excellent"
    assert "hors plage" in payload["level"].lower()


def test_translate_trigram() -> None:
    """Les tags POS sont traduits via la table interne."""
    s = translate_trigram("DET-NOUN-VERB")
    assert "Dét" in s
    assert "Verbe" in s


def test_avg_signature_from_cache_averages_json_signatures() -> None:
    """La moyenne par axe suit les signatures JSON persistées."""
    sig_a = {"Noms & adjectifs": 0.4, "Verbes d'action": 0.2}
    sig_b = {"Noms & adjectifs": 0.2, "Verbes d'action": 0.4}
    df = pd.DataFrame(
        [
            {"id": "a", "_signature_json": json.dumps(sig_a)},
            {"id": "b", "_signature_json": json.dumps(sig_b)},
        ]
    )
    mean_sig = avg_signature_from_cache(df)
    assert mean_sig is not None
    assert abs(mean_sig["Noms & adjectifs"] - 0.3) < 1e-9
    assert abs(mean_sig["Verbes d'action"] - 0.3) < 1e-9


def test_avg_signature_from_cache_none_when_empty() -> None:
    """Sans JSON exploitable, pas de moyenne (première entrée ou cache vide)."""
    df = pd.DataFrame([{"id": "1", "_signature_json": ""}])
    assert avg_signature_from_cache(df) is None


def test_curator_advices_after_save_without_stats() -> None:
    """Sans stats NLP, message explicite pour l'utilisateur."""
    out = curator_advices_after_save({}, {})
    assert out == [CURATOR_MESSAGE_STATS_UNAVAILABLE]


def test_curator_advices_after_save_fallback_when_no_prioritized() -> None:
    """Zone équilibrée → message dédié si ``prioritized_actions`` ne retourne rien."""
    stats = {
        "ratio": 2.0,
        "ttr": 0.7,
        "long_moy_phrases": 15.0,
        "mots_repetes": [],
    }
    deltas = {"Noms & adjectifs": 0.1, "Verbes d'action": 0.05}
    out = curator_advices_after_save(stats, deltas)
    assert out == [CURATOR_MESSAGE_ADVICE_BALANCED]


def test_compute_row_cache_without_nlp_returns_empty_bundle() -> None:
    """Sans spaCy, pas de score ni colonnes remplies (chemins dégradés)."""
    df = pd.DataFrame([{"id": "1", "_signature_json": ""}])
    cols = ["_coherence_score", "_ttr"]
    r = compute_row_cache("in", "out", None, df, "1", cols, avg_signature_from_cache)
    assert r.coherence_score is None
    assert r.cache["_coherence_score"] == ""
    assert r.cache["_ttr"] == ""


def test_row_nlp_feedback_bundle_uses_persisted_score_and_cache() -> None:
    """Après relecture type SQL, le score et les colonnes cache viennent de la ligne persistée."""
    sig = {"Noms & adjectifs": 0.35, "Verbes d'action": 0.25}
    df = pd.DataFrame(
        [
            {
                "id": "other",
                "input": "a",
                "output": "b " * 50,
                "_signature_json": json.dumps(sig),
                "_ratio": "1.2",
                "_ttr": "0.60",
                "_long_phrases": "14.0",
                "_coherence_score": "81",
                "_syntax_contrast": "0.33",
            },
            {
                "id": "target",
                "input": "in text",
                "output": "out text " * 30,
                "_signature_json": json.dumps(sig),
                "_ratio": "2.0",
                "_ttr": "0.70",
                "_long_phrases": "15.0",
                "_coherence_score": "77",
                "_syntax_contrast": "0.40",
            },
        ]
    )
    cols = [
        "_ratio",
        "_ttr",
        "_long_phrases",
        "_signature_json",
        "_coherence_score",
        "_syntax_contrast",
    ]
    bundle = row_nlp_feedback_bundle_after_persist(df, "target", None, cols)
    assert bundle.coherence_score == 77
    assert bundle.cache["_ttr"] == "0.70"
    assert bundle.cache["_syntax_contrast"] == "0.40"
    adv = curator_advices_after_save(bundle.advice_stats, bundle.coherence_deltas)
    assert isinstance(adv, list)
    assert len(adv) >= 1


def test_row_nlp_feedback_bundle_unknown_row_empty() -> None:
    """Identifiant absent → bundle vide."""
    df = pd.DataFrame([{"id": "x", "_coherence_score": "50"}])
    b = row_nlp_feedback_bundle_after_persist(df, "missing", None, ["_coherence_score"])
    assert b.coherence_score is None
    assert b.cache["_coherence_score"] == ""


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
