"""Tests for the Streamlit cache wrapper around ``load_project_entries``."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from src.project_entries_cache import (
    PROJECT_ENTRIES_CACHE_TTL_SECONDS,
    cached_load_project_entries,
    engine_url_cache_token,
    invalidate_project_entries_cache,
)


def test_ttl_matches_issue_spec() -> None:
    """Acceptance: TTL documented at 30 s (issue-010 / S4)."""
    assert PROJECT_ENTRIES_CACHE_TTL_SECONDS == 30


def test_engine_url_cache_token_matches_sqlalchemy_url_string() -> None:
    """Cache partition key must be stable for the same DB URL across reruns."""
    engine: Engine = create_engine("sqlite:///:memory:")
    assert engine_url_cache_token(engine) == str(engine.url)


def test_cached_loader_exposes_streamlit_invalidation_api() -> None:
    """Decorated function must support ``clear()`` for post-write invalidation."""
    assert callable(getattr(cached_load_project_entries, "clear", None))
    assert callable(invalidate_project_entries_cache)
