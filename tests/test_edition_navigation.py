"""Tests navigation édition (précédent / suivant dans la liste filtrée) — issue 015."""

from __future__ import annotations

import pytest
from src.nlp_engine import edition_nav_neighbor_entry_id


def test_edition_nav_prev_from_second() -> None:
    """Précédent depuis la deuxième entrée renvoie la première."""
    ids = ["a", "b", "c"]
    assert edition_nav_neighbor_entry_id(ids, "b", direction="prev") == "a"


def test_edition_nav_next_from_second() -> None:
    """Suivant depuis la deuxième entrée renvoie la troisième."""
    ids = ["a", "b", "c"]
    assert edition_nav_neighbor_entry_id(ids, "b", direction="next") == "c"


def test_edition_nav_prev_at_first_returns_none() -> None:
    """À la première entrée, précédent est indisponible."""
    assert edition_nav_neighbor_entry_id(["x", "y"], "x", direction="prev") is None


def test_edition_nav_next_at_last_returns_none() -> None:
    """À la dernière entrée, suivant est indisponible."""
    assert edition_nav_neighbor_entry_id(["x", "y"], "y", direction="next") is None


def test_edition_nav_unknown_current_returns_none() -> None:
    """ID absent de la liste filtrée : pas de voisin (réinitialisation côté UI)."""
    assert edition_nav_neighbor_entry_id(["a", "b"], "z", direction="next") is None


def test_edition_nav_empty_list_returns_none() -> None:
    """Liste vide : toujours None."""
    assert edition_nav_neighbor_entry_id([], "a", direction="next") is None


def test_edition_nav_invalid_direction_raises() -> None:
    """Direction inconnue : erreur explicite."""
    with pytest.raises(ValueError, match="direction"):
        edition_nav_neighbor_entry_id(["a"], "a", direction="invalid")  # type: ignore[arg-type]
