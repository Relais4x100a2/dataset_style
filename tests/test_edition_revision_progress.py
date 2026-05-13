"""Tests indicateur position et compteurs révision (liste filtrée) — issues 016 / 033."""

from __future__ import annotations

import pandas as pd
import pytest
from src.nlp_engine import (
    EditionPickRevisionStats,
    EditionScoreFilterSpec,
    edition_entry_k_of_n,
    edition_pick_revision_stats,
    filter_edition_entries_dataframe,
)


def test_edition_entry_k_of_n_matches_ordered_list() -> None:
    """k est en base 1 et suit l'ordre explicite (identique au selectbox / nav)."""
    ordered = ["10", "2", "3"]
    assert edition_entry_k_of_n(ordered, "10") == (1, 3)
    assert edition_entry_k_of_n(ordered, "2") == (2, 3)
    assert edition_entry_k_of_n(ordered, "3") == (3, 3)


def test_edition_entry_k_of_n_empty_returns_zero() -> None:
    """Liste vide : (0, 0) sans lever d'exception."""
    assert edition_entry_k_of_n([], "x") == (0, 0)


def test_edition_entry_k_of_n_unknown_id_raises() -> None:
    """current_id absent de la liste : erreur explicite (l'UI garantit une id valide)."""
    with pytest.raises(ValueError, match="ordered_entry_ids"):
        edition_entry_k_of_n(["a", "b"], "z")


def test_edition_pick_revision_stats_preset_statuts() -> None:
    """Compteurs « à réviser » = A faire + En cours ; validé = Fait et validé."""
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "statut": ["A faire", "En cours", "Fait et validé", "Legacy"],
        }
    )
    s = edition_pick_revision_stats(df)
    assert s == EditionPickRevisionStats(total=4, needing_review=2, validated=1, draft=1)


def test_edition_pick_revision_stats_no_statut_column() -> None:
    """Sans colonne statut, tout est classé « autre »."""
    df = pd.DataFrame({"id": ["1"]})
    s = edition_pick_revision_stats(df)
    assert s.total == 1 and s.needing_review == 0 and s.validated == 0 and s.draft == 1


def test_edition_pick_revision_stats_counts_validated_preset_label() -> None:
    """Le libellé « Validé » du preset pro est compté comme validée."""
    df = pd.DataFrame({"id": ["1", "2"], "statut": ["Validé", "Brouillon"]})
    s = edition_pick_revision_stats(df)
    assert s == EditionPickRevisionStats(total=2, needing_review=0, validated=1, draft=1)


def test_filter_change_updates_k_and_n() -> None:
    """Après filtre plus strict, n diminue et k se recalcule sur la même liste triée."""
    df = pd.DataFrame(
        {
            "id": ["c", "a", "b"],
            "statut": ["S", "S", "S"],
            "_coherence_score": ["10", "50", "90"],
        }
    )
    wide = filter_edition_entries_dataframe(
        df, statut_label=None, score_spec=EditionScoreFilterSpec()
    )
    ids_wide = wide["id"].astype(str).tolist()
    assert ids_wide == ["a", "b", "c"]
    assert edition_entry_k_of_n(ids_wide, "b") == (2, 3)

    narrow = filter_edition_entries_dataframe(
        df,
        statut_label=None,
        score_spec=EditionScoreFilterSpec(mode="below", threshold_lt=80, include_na=False),
    )
    ids_narrow = narrow["id"].astype(str).tolist()
    assert ids_narrow == ["a", "c"]
    assert edition_entry_k_of_n(ids_narrow, "c") == (2, 2)
