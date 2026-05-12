"""Tests for post-save curator feedback display helpers (issue-021)."""

from __future__ import annotations

from src.post_save_feedback_display import (
    post_save_freshness_caption_fr,
    post_save_stylistic_metric_labels_fr,
)


def test_post_save_stylistic_metric_labels_reference_cache_columns() -> None:
    """UI labels must tie French copy to persisted cache column names."""
    labels = post_save_stylistic_metric_labels_fr()
    assert "_coherence_score" in labels["coherence_score"]
    assert "_ttr" in labels["ttr"]
    assert "_syntax_contrast" in labels["syntax_contrast"]


def test_post_save_freshness_caption_sync_mentions_commit_and_sync() -> None:
    """Synchronous pipeline copy must state post-commit read and no deferred job."""
    text = post_save_freshness_caption_fr(synchronous_before_commit=True)
    lowered = text.lower()
    assert "synchrone" in lowered or "synchrone" in text.lower()
    assert "commit" in lowered or "base" in lowered


def test_post_save_freshness_caption_async_warns_provisional() -> None:
    """Future async path should steer curators toward provisional interpretation."""
    text = post_save_freshness_caption_fr(synchronous_before_commit=False)
    assert "provis" in text.lower() or "différ" in text.lower()
