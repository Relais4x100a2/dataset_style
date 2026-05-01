"""
Module: tests.test_database
Tests unitaires pour ``src.database`` : normalisation, agrégats cache, détection problèmes.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError
from src.database import (
    ALL_COLUMNS,
    CACHE_COLUMNS,
    STATUT_VALIDE,
    _is_retryable_db_error,
    _normalize_for_write,
    _normalize_loaded_frame,
    audit_rows_from_cache,
    avg_signature_from_cache,
    avg_trigrams_from_cache,
    dataset_cache_stats,
    flag_problematic_rows,
)


def test_normalize_loaded_frame_adds_missing_cache_columns() -> None:
    """Les colonnes cache manquantes sont ajoutées et les NaN sont normalisées."""
    df = pd.DataFrame({"id": ["1"], "type": ["Expansion"]})
    out = _normalize_loaded_frame(df)
    for col in CACHE_COLUMNS:
        assert col in out.columns
    assert out["id"].iloc[0] == "1"


def test_normalize_for_write_restricts_columns() -> None:
    """L'écriture ne conserve que ALL_COLUMNS avec chaînes vides par défaut."""
    df = pd.DataFrame({"id": ["x"], "extra": ["drop"]})
    out = _normalize_for_write(df)
    assert list(out.columns) == ALL_COLUMNS
    assert "extra" not in out.columns


def test_is_retryable_operational_error() -> None:
    """OperationalError est considérée comme retryable."""
    exc = OperationalError("stmt", {}, None)
    assert _is_retryable_db_error(exc) is True


def test_is_retryable_message_timeout() -> None:
    """Un message contenant 'timeout' est traité comme retryable."""
    assert _is_retryable_db_error(RuntimeError("read timeout occurred")) is True


def test_avg_signature_from_cache_happy_path() -> None:
    """Deux signatures JSON valides produisent une moyenne par clé."""
    df = pd.DataFrame(
        {
            "_signature_json": [
                json.dumps({"a": 1.0, "b": 3.0}),
                json.dumps({"a": 3.0, "b": 5.0}),
            ]
        }
    )
    got = avg_signature_from_cache(df)
    assert got is not None
    assert got["a"] == pytest.approx(2.0)
    assert got["b"] == pytest.approx(4.0)


def test_avg_signature_from_cache_invalid_json_skipped() -> None:
    """JSON invalide est ignoré ; le reste suffit pour une moyenne."""
    df = pd.DataFrame(
        {
            "_signature_json": [
                "not json",
                json.dumps({"x": 10.0}),
            ]
        }
    )
    got = avg_signature_from_cache(df)
    assert got is not None
    assert got["x"] == pytest.approx(10.0)


def test_avg_signature_from_cache_empty_returns_none() -> None:
    """Aucune signature exploitable → None."""
    df = pd.DataFrame({"_signature_json": ["", ""]})
    assert avg_signature_from_cache(df) is None


def test_audit_rows_from_cache_happy_path() -> None:
    """Une ligne avec _ratio numérique produit une entrée d'audit."""
    df = pd.DataFrame(
        {
            "id": ["1"],
            "type": ["Normalisation"],
            "_ratio": ["2.5"],
            "_long_phrases": ["12"],
            "_ttr": ["0.6"],
        }
    )
    rows = audit_rows_from_cache(df)
    assert len(rows) == 1
    assert rows[0]["id"] == "1"
    assert rows[0]["ratio"] == 2.5


def test_audit_rows_from_cache_invalid_ratio_skipped() -> None:
    """_ratio non numérique : la ligne est ignorée."""
    df = pd.DataFrame(
        {
            "id": ["1"],
            "type": ["Normalisation"],
            "_ratio": ["nope"],
            "_long_phrases": ["12"],
            "_ttr": ["0.6"],
        }
    )
    assert audit_rows_from_cache(df) == []


def test_dataset_cache_stats_requires_all_metrics() -> None:
    """Une ligne sans une métrique complète n'entre pas dans les stats."""
    df = pd.DataFrame(
        [
            {
                "_ratio": "1.0",
                "_ttr": "0.7",
                "_long_phrases": "10",
                "_coherence_score": "80",
            },
            {
                "_ratio": "2.0",
                "_ttr": "",
                "_long_phrases": "12",
                "_coherence_score": "70",
            },
        ]
    )
    stats = dataset_cache_stats(df)
    assert stats is not None
    assert stats["n"] == 1
    assert 0 <= stats["health_score"] <= 100


def test_dataset_cache_stats_empty_returns_none() -> None:
    """Aucune ligne complète → None."""
    df = pd.DataFrame([{"_ratio": "", "_ttr": "", "_long_phrases": "", "_coherence_score": ""}])
    assert dataset_cache_stats(df) is None


def test_flag_problematic_rows_coherence_critical() -> None:
    """Cohérence < 45 déclenche une alerte."""
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "type": "Normalisation",
                "forme": "Narration",
                "ton": "Neutre",
                "_ratio": "2.0",
                "_ttr": "0.6",
                "_coherence_score": "40",
            }
        ]
    )
    flags = flag_problematic_rows(df)
    assert len(flags) == 1
    assert "Cohérence critique" in flags[0]["alertes"]


def test_flag_problematic_rows_expansion_weak() -> None:
    """Type Expansion et ratio < 1.5 → Expansion faible."""
    df = pd.DataFrame(
        [
            {
                "id": "b",
                "type": "Expansion",
                "forme": "Scène",
                "ton": "Lyrique",
                "_ratio": "1.2",
                "_ttr": "0.7",
                "_coherence_score": "80",
            }
        ]
    )
    flags = flag_problematic_rows(df)
    assert any("Expansion faible" in f["alertes"] for f in flags)


def test_flag_problematic_rows_skips_incomplete_cache() -> None:
    """Valeurs non convertibles en float : ligne ignorée."""
    df = pd.DataFrame(
        [
            {
                "id": "c",
                "type": "Normalisation",
                "forme": "Narration",
                "ton": "Neutre",
                "_ratio": "x",
                "_ttr": "0.6",
                "_coherence_score": "70",
            }
        ]
    )
    assert flag_problematic_rows(df) == []


def test_avg_trigrams_from_cache_merges_counts() -> None:
    """Plusieurs lignes agrègent les comptages de trigrammes."""
    df = pd.DataFrame(
        {
            "_trigrams_json": [
                json.dumps({"DET NOUN VERB": 2}),
                json.dumps({"DET NOUN VERB": 3, "ADJ NOUN PUNCT": 1}),
            ]
        }
    )
    ctr = avg_trigrams_from_cache(df)
    assert ctr is not None
    assert ctr["DET NOUN VERB"] == 5
    assert ctr["ADJ NOUN PUNCT"] == 1


def test_avg_trigrams_from_cache_empty_returns_none() -> None:
    """Pas de trigrammes valides → None."""
    df = pd.DataFrame({"statut": [STATUT_VALIDE], "_trigrams_json": [""]})
    assert avg_trigrams_from_cache(df) is None
