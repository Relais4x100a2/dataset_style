"""Tests des textes / règles de navigation séquentielle édition (issue 032)."""

from __future__ import annotations

from src.services.edition_sequential_navigation import (
    edition_nav_boundary_caption_fr,
    edition_nav_singleton_filtered_caption_fr,
    edition_nav_unsaved_changes_notice_fr,
)


def test_boundary_prev_when_blocked() -> None:
    assert edition_nav_boundary_caption_fr("prev", can_navigate=False) is not None


def test_boundary_prev_when_allowed() -> None:
    assert edition_nav_boundary_caption_fr("prev", can_navigate=True) is None


def test_boundary_next_when_blocked() -> None:
    text = edition_nav_boundary_caption_fr("next", can_navigate=False)
    assert text is not None
    assert "suivante" in text.lower() or "Dernière" in text


def test_singleton_caption_only_for_one() -> None:
    assert edition_nav_singleton_filtered_caption_fr(n_filtered=1) is not None
    assert edition_nav_singleton_filtered_caption_fr(n_filtered=2) is None


def test_unsaved_notice_is_non_empty() -> None:
    assert "Sauvegarder" in edition_nav_unsaved_changes_notice_fr()
