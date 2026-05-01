"""
Tests unitaires et d'intégration légers pour ``src.database`` (normalisation, agrégats,
détection de lignes problématiques, persistance via SQLite en mémoire).
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
    ensure_entries_table,
    flag_problematic_rows,
    load_data,
    update_data,
)
from src.nlp_engine import signature_variance


def test_is_retryable_db_error_operational() -> None:
    """Les erreurs OperationalError sont considérées comme réessayables."""
    assert _is_retryable_db_error(OperationalError("stmt", {}, None)) is True


def test_is_retryable_db_error_message_fallback() -> None:
    """Une exception générique avec message de timeout est réessayable."""
    assert _is_retryable_db_error(RuntimeError("connection timeout")) is True


def test_is_retryable_db_error_not_retryable() -> None:
    """Une erreur sans marqueur connu n'est pas traitée comme transitoire."""
    assert _is_retryable_db_error(ValueError("invalid input")) is False


def test_normalize_loaded_frame_adds_cache_columns() -> None:
    """Les colonnes cache manquantes sont ajoutées ; les colonnes existantes conservées."""
    df = pd.DataFrame({"id": ["1"], "type": ["Normalisation"]})
    out = _normalize_loaded_frame(df)
    for col in CACHE_COLUMNS:
        assert col in out.columns
    assert "id" in out.columns


def test_normalize_for_write_subset_and_order() -> None:
    """L'écriture ne conserve que ALL_COLUMNS dans l'ordre attendu."""
    df = pd.DataFrame(
        {
            "id": ["a"],
            "type": ["Expansion"],
            "extra_col": [99],
        }
    )
    out = _normalize_for_write(df)
    assert list(out.columns) == ALL_COLUMNS
    assert "extra_col" not in out.columns


def test_load_data_empty_table(sqlite_engine) -> None:
    """Après création de table vide, load_data renvoie un frame avec toutes les colonnes."""
    ensure_entries_table(sqlite_engine)
    df = load_data(sqlite_engine, max_retries=1)
    assert df.empty
    assert list(df.columns) == ALL_COLUMNS


def test_update_data_round_trip(sqlite_engine) -> None:
    """Round-trip : mise à jour puis rechargement conserve les lignes."""
    row = {c: "" for c in ALL_COLUMNS}
    row.update(
        {
            "id": "row-1",
            "type": "Normalisation",
            "forme": "Narration",
            "ton": "Neutre",
            "support": "Narratif",
            "input": "Bonjour.",
            "output": "Bonjour le monde.",
            "statut": STATUT_VALIDE,
            "notes": "",
        }
    )
    update_data(sqlite_engine, pd.DataFrame([row]))
    df = load_data(sqlite_engine, max_retries=1)
    assert len(df) == 1
    assert df.iloc[0]["id"] == "row-1"


def test_avg_signature_from_cache_parses_and_averages() -> None:
    """Les JSON de signature valides sont moyennés ; JSON invalide ignoré."""
    sig1 = {"a": 1.0, "b": 2.0}
    sig2 = {"a": 3.0, "b": 4.0}
    df = pd.DataFrame(
        {
            "_signature_json": [json.dumps(sig1), json.dumps(sig2), "not-json"],
        }
    )
    out = avg_signature_from_cache(df)
    assert out is not None
    assert out["a"] == pytest.approx(2.0)
    assert out["b"] == pytest.approx(3.0)


def test_avg_signature_from_cache_empty_returns_none() -> None:
    """Sans signature utilisable, la fonction renvoie None."""
    assert avg_signature_from_cache(pd.DataFrame({"_signature_json": ["", ""]})) is None


def test_flag_problematic_rows_detects_rules() -> None:
    """Cohérence basse, expansion faible, TTR bas déclenchent les alertes attendues."""
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "type": "Expansion",
                "forme": "Scène",
                "ton": "Neutre",
                "_ratio": "1.0",
                "_ttr": "0.40",
                "_coherence_score": "40",
            }
        ]
    )
    flagged = flag_problematic_rows(df)
    assert len(flagged) == 1
    alerts = flagged[0]["alertes"]
    assert "Cohérence critique" in alerts
    assert "Expansion faible" in alerts
    assert "Vocabulaire répétitif" in alerts


def test_flag_problematic_rows_skips_invalid_cache() -> None:
    """Les lignes sans métriques numériques valides sont ignorées."""
    df = pd.DataFrame(
        [
            {
                "id": "x",
                "type": "Normalisation",
                "forme": "",
                "ton": "",
                "_ratio": "",
                "_ttr": "nope",
                "_coherence_score": "",
            }
        ]
    )
    assert flag_problematic_rows(df) == []


def test_dataset_cache_stats_happy_path() -> None:
    """Toutes les métriques présentes pour une fiche → stats et health_score calculés."""
    df = pd.DataFrame(
        [
            {
                "_ratio": "2.0",
                "_ttr": "0.60",
                "_long_phrases": "15",
                "_coherence_score": "80",
            }
        ]
    )
    stats = dataset_cache_stats(df)
    assert stats is not None
    assert stats["n"] == 1
    assert "health_score" in stats
    assert 0 <= stats["health_score"] <= 100


def test_dataset_cache_stats_incomplete_row_skipped() -> None:
    """Une ligne avec une colonne cache vide est exclue de l'agrégat."""
    df = pd.DataFrame(
        [
            {"_ratio": "2.0", "_ttr": "", "_long_phrases": "10", "_coherence_score": "70"},
        ]
    )
    assert dataset_cache_stats(df) is None


def test_avg_trigrams_from_cache() -> None:
    """Les compteurs JSON sont fusionnés ; entrées invalides ignorées."""
    df = pd.DataFrame(
        {
            "_trigrams_json": [
                json.dumps({"A-B-C": 2}),
                json.dumps({"A-B-C": 1, "D-E-F": 1}),
                "",
            ],
        }
    )
    ctr = avg_trigrams_from_cache(df)
    assert ctr is not None
    assert ctr["A-B-C"] == 3
    assert ctr["D-E-F"] == 1


def test_signature_variance_requires_two_signatures() -> None:
    """Moins de deux signatures valides → None."""
    df = pd.DataFrame({"_signature_json": [json.dumps({"x": 1.0})]})
    assert signature_variance(df) is None


def test_audit_rows_from_cache_numeric_ratio() -> None:
    """audit_rows_from_cache inclut les lignes avec _ratio convertible."""
    df = pd.DataFrame(
        [{"id": "1", "type": "N", "_ratio": "1.5", "_long_phrases": "12", "_ttr": "0.5"}]
    )
    rows = audit_rows_from_cache(df)
    assert len(rows) == 1
    assert rows[0]["ratio"] == 1.5
