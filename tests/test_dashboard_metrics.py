"""Tests for stylometric dashboard aggregations (issue-006)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from src.database import STATUT_VALIDE
from src.nlp_engine import (
    coherence_score_bucket_table,
    dataframe_for_dashboard_scope,
    list_parsed_coherence_scores,
    mean_syntax_contrast_parsed,
    outliers_low_coherence_table,
    parse_persisted_syntax_contrast,
    signature_variance,
)


def test_parse_persisted_syntax_contrast_empty() -> None:
    assert parse_persisted_syntax_contrast("") is None
    assert parse_persisted_syntax_contrast(None) is None
    assert parse_persisted_syntax_contrast("   ") is None


def test_parse_persisted_syntax_contrast_decimal_comma() -> None:
    assert parse_persisted_syntax_contrast("0,33") == pytest.approx(0.33)
    assert parse_persisted_syntax_contrast("0.40") == pytest.approx(0.40)


def test_parse_persisted_syntax_contrast_invalid() -> None:
    assert parse_persisted_syntax_contrast("n/a") is None


def test_list_parsed_coherence_scores_skips_invalid() -> None:
    df = pd.DataFrame(
        {
            "_coherence_score": ["80", "", "xx", "75.2"],
        }
    )
    scores = list_parsed_coherence_scores(df)
    assert scores == [80, 75]


def test_mean_syntax_contrast_parsed_excludes_empty() -> None:
    df = pd.DataFrame({"_syntax_contrast": ["0.2", "", "0.4", "bad"]})
    m = mean_syntax_contrast_parsed(df)
    assert m == pytest.approx(0.3)


def test_mean_syntax_contrast_parsed_none_when_no_values() -> None:
    df = pd.DataFrame({"_syntax_contrast": ["", "x"]})
    assert mean_syntax_contrast_parsed(df) is None


def test_dataframe_for_dashboard_scope_validated_only() -> None:
    df = pd.DataFrame(
        [
            {"id": "1", "statut": STATUT_VALIDE},
            {"id": "2", "statut": "Brouillon"},
        ]
    )
    sub = dataframe_for_dashboard_scope(df, validated_only=True, validated_label=STATUT_VALIDE)
    assert len(sub) == 1
    assert sub.iloc[0]["id"] == "1"


def test_coherence_score_bucket_table_counts() -> None:
    tbl = coherence_score_bucket_table([5, 15, 15, 95])
    assert int(tbl.loc[tbl["Tranche (score)"] == "0–9", "Nombre"].iloc[0]) == 1
    assert int(tbl.loc[tbl["Tranche (score)"] == "10–19", "Nombre"].iloc[0]) == 2
    assert int(tbl.loc[tbl["Tranche (score)"] == "90–100", "Nombre"].iloc[0]) == 1


def test_outliers_low_coherence_table_sorted_and_limited() -> None:
    df = pd.DataFrame(
        [
            {"id": "a", "statut": "x", "type": "T1", "_coherence_score": "90"},
            {"id": "b", "statut": "x", "type": "T2", "_coherence_score": "10"},
            {"id": "c", "statut": "x", "type": "T3", "_coherence_score": "20"},
            {"id": "d", "statut": "x", "type": "T4", "_coherence_score": ""},
        ]
    )
    out = outliers_low_coherence_table(df, limit=2)
    assert list(out["id"]) == ["b", "c"]
    assert list(out["score_coherence"]) == [10, 20]


def test_signature_variance_ignores_non_dict_json() -> None:
    """Non-object JSON must not crash variance and must not count as a signature."""
    sig = {"Noms & adjectifs": 0.4, "Verbes d'action": 0.2}
    df = pd.DataFrame(
        [
            {"_signature_json": json.dumps(sig), "statut": STATUT_VALIDE},
            {"_signature_json": "[1,2,3]", "statut": STATUT_VALIDE},
            {"_signature_json": json.dumps(sig), "statut": STATUT_VALIDE},
        ]
    )
    v = signature_variance(df)
    assert v is not None
    assert "Noms & adjectifs" in v


def test_signature_variance_skips_axes_with_single_observation() -> None:
    """Axes present in only one fiche must not appear with a spurious zero spread."""
    s1 = {"A": 0.0, "B": 0.0}
    s2 = {"B": 1.0, "C": 0.0}
    df = pd.DataFrame(
        [
            {"_signature_json": json.dumps(s1), "statut": STATUT_VALIDE},
            {"_signature_json": json.dumps(s2), "statut": STATUT_VALIDE},
        ]
    )
    v = signature_variance(df)
    assert v is not None
    assert "B" in v
    assert v["B"] > 0
    assert "A" not in v
    assert "C" not in v


def test_signature_variance_none_when_no_axis_has_two_observations() -> None:
    """Fully disjoint axis keys across two fiches yield no comparable axis."""
    df = pd.DataFrame(
        [
            {"_signature_json": json.dumps({"axis_x": 0.1}), "statut": STATUT_VALIDE},
            {"_signature_json": json.dumps({"axis_y": 0.2}), "statut": STATUT_VALIDE},
        ]
    )
    assert signature_variance(df) is None
