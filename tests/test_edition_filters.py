"""Tests filtres édition (statut + score de cohérence) — issue 009."""

from __future__ import annotations

import pandas as pd
import pytest
from src.nlp_engine import (
    EditionScoreFilterSpec,
    edition_statut_filter_options,
    filter_edition_entries_dataframe,
)


def test_edition_statut_filter_options_preset_then_legacy() -> None:
    """Preset order first, then legacy statuts present in df only."""
    preset = ["A faire", "En cours", "Fait et validé"]
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "statut": ["En cours", "Legacy X", "A faire"],
        }
    )
    opts = edition_statut_filter_options(preset, df)
    assert opts[:3] == preset
    assert "Legacy X" in opts
    assert opts.index("Legacy X") == len(preset)


def test_edition_statut_filter_options_no_statut_column() -> None:
    """Sans colonne statut, seule la liste preset est renvoyée."""
    df = pd.DataFrame({"id": ["1"]})
    assert edition_statut_filter_options(["A", "B"], df) == ["A", "B"]


def test_filter_edition_by_statut_only() -> None:
    """Filtre statut exact (chaîne)."""
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "statut": ["En cours", "Validé"],
            "_coherence_score": ["50", "80"],
        }
    )
    out = filter_edition_entries_dataframe(
        df,
        statut_label="En cours",
        score_spec=EditionScoreFilterSpec(),
    )
    assert len(out) == 1
    assert out.iloc[0]["id"] == "a"


def test_filter_edition_statut_all_passes_through() -> None:
    """statut_label None ne filtre pas."""
    df = pd.DataFrame({"id": ["1"], "statut": ["X"], "_coherence_score": [""]})
    out = filter_edition_entries_dataframe(
        df,
        statut_label=None,
        score_spec=EditionScoreFilterSpec(),
    )
    assert len(out) == 1


def test_filter_edition_score_below_excludes_na_by_default() -> None:
    """Sous seuil : N/A exclus par défaut (include_na=False)."""
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "statut": ["S", "S", "S"],
            "_coherence_score": ["40", "60", ""],
        }
    )
    spec = EditionScoreFilterSpec(mode="below", threshold_lt=50, include_na=False)
    out = filter_edition_entries_dataframe(df, statut_label=None, score_spec=spec)
    assert set(out["id"].tolist()) == {"1"}


def test_filter_edition_score_below_includes_na_when_requested() -> None:
    """Sous seuil + include_na : lignes sans score incluses."""
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "statut": ["S", "S", "S"],
            "_coherence_score": ["40", "60", ""],
        }
    )
    spec = EditionScoreFilterSpec(mode="below", threshold_lt=50, include_na=True)
    out = filter_edition_entries_dataframe(df, statut_label=None, score_spec=spec)
    assert set(out["id"].tolist()) == {"1", "3"}


def test_filter_edition_score_bucket() -> None:
    """Tranche 40–49 via décile 4."""
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "statut": ["S", "S", "S"],
            "_coherence_score": ["45", "50", "44.9"],
        }
    )
    spec = EditionScoreFilterSpec(mode="bucket", bucket_decile=4, include_na=False)
    out = filter_edition_entries_dataframe(df, statut_label=None, score_spec=spec)
    assert set(out["id"].tolist()) == {"a", "c"}


def test_filter_edition_combined_statut_and_score() -> None:
    """Combinaison statut + score."""
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "statut": ["En cours", "En cours", "Validé"],
            "_coherence_score": ["30", "80", "20"],
        }
    )
    spec = EditionScoreFilterSpec(mode="below", threshold_lt=50, include_na=False)
    out = filter_edition_entries_dataframe(df, statut_label="En cours", score_spec=spec)
    assert len(out) == 1
    assert out.iloc[0]["id"] == "1"


def test_filter_edition_missing_score_column_treated_as_na() -> None:
    """Colonne _coherence_score absente : tout N/A pour filtre score."""
    df = pd.DataFrame({"id": ["1", "2"], "statut": ["S", "S"]})
    spec = EditionScoreFilterSpec(mode="below", threshold_lt=50, include_na=False)
    out = filter_edition_entries_dataframe(df, statut_label=None, score_spec=spec)
    assert out.empty
    spec2 = EditionScoreFilterSpec(mode="below", threshold_lt=50, include_na=True)
    out2 = filter_edition_entries_dataframe(df, statut_label=None, score_spec=spec2)
    assert len(out2) == 2


def test_edition_score_filter_spec_bucket_decile_invalid() -> None:
    """bucket_decile hors 0..9 lève ValueError."""
    with pytest.raises(ValueError):
        EditionScoreFilterSpec(mode="bucket", bucket_decile=10)
