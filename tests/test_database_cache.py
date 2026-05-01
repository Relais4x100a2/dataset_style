"""
Tests for cache-derived statistics and quality flags (no PostgreSQL).
"""

from __future__ import annotations

import pandas as pd
import pytest
from src.database import (
    STATUT_VALIDE,
    avg_signature_from_cache,
    avg_trigrams_from_cache,
    dataset_cache_stats,
    flag_problematic_rows,
)


def test_flag_problematic_rows_coherence_critical() -> None:
    """Coherence below threshold triggers alert."""
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "type": "Normalisation",
                "forme": "Narration",
                "ton": "Neutre",
                "_ratio": "2.0",
                "_ttr": "0.6",
                "_coherence_score": "30",
                "statut": STATUT_VALIDE,
            }
        ]
    )
    rows = flag_problematic_rows(df)
    assert len(rows) == 1
    assert "Cohérence critique" in rows[0]["alertes"]


def test_flag_problematic_rows_expansion_weak_ratio() -> None:
    """Expansion type with low ratio triggers expansion alert."""
    df = pd.DataFrame(
        [
            {
                "id": "2",
                "type": "Expansion",
                "forme": "Narration",
                "ton": "Neutre",
                "_ratio": "1.0",
                "_ttr": "0.6",
                "_coherence_score": "80",
                "statut": STATUT_VALIDE,
            }
        ]
    )
    rows = flag_problematic_rows(df)
    assert any("Expansion faible" in r["alertes"] for r in rows)


def test_flag_problematic_rows_skips_incomplete_cache() -> None:
    """Rows without parseable ratio/ttr/coherence are skipped."""
    df = pd.DataFrame(
        [
            {
                "id": "3",
                "type": "Normalisation",
                "forme": "Narration",
                "ton": "Neutre",
                "_ratio": "",
                "_ttr": "0.6",
                "_coherence_score": "80",
                "statut": STATUT_VALIDE,
            }
        ]
    )
    assert flag_problematic_rows(df) == []


def test_dataset_cache_stats_happy_path(df_validated: pd.DataFrame) -> None:
    """Aggregated stats and health score when cache is complete."""
    valid_only = df_validated[df_validated["statut"] == STATUT_VALIDE]
    stats = dataset_cache_stats(valid_only)
    assert stats is not None
    assert stats["n"] >= 1
    assert "health_score" in stats
    assert 0 <= stats["health_score"] <= 100


def test_dataset_cache_stats_empty_when_no_cache() -> None:
    """No complete cache metrics returns None."""
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "type": "Normalisation",
                "forme": "Narration",
                "ton": "Neutre",
                "_ratio": "",
                "_ttr": "",
                "_long_phrases": "",
                "_coherence_score": "",
                "statut": STATUT_VALIDE,
            }
        ]
    )
    assert dataset_cache_stats(df) is None


def test_avg_signature_from_cache(df_validated: pd.DataFrame) -> None:
    """Mean of JSON signatures from _signature_json."""
    valid_only = df_validated[df_validated["statut"] == STATUT_VALIDE]
    avg = avg_signature_from_cache(valid_only)
    assert avg is not None
    assert avg["a"] == pytest.approx(0.5)
    assert avg["b"] == pytest.approx(0.3)


def test_avg_trigrams_from_cache(df_validated: pd.DataFrame) -> None:
    """Aggregates trigram counts from cache column."""
    valid_only = df_validated[df_validated["statut"] == STATUT_VALIDE]
    ctr = avg_trigrams_from_cache(valid_only)
    assert ctr is not None
    assert ctr["DET+NOUN+VERB"] == 2
