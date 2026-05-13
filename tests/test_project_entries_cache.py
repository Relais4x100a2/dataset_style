"""Tests for the Streamlit cache wrapper around ``load_project_entries``."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from src.project_entries_cache import (
    PROJECT_ENTRIES_CACHE_TTL_SECONDS,
    cached_load_project_entries,
    engine_url_cache_token,
    invalidate_project_entries_cache,
    project_entries_cache_tenant_partition,
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


def test_tenant_partition_separates_projects_and_users() -> None:
    """Acceptance issue-027: cache identity must not collide across users or projects."""
    a = project_entries_cache_tenant_partition("proj-1", "user-a")
    b = project_entries_cache_tenant_partition("proj-1", "user-b")
    c = project_entries_cache_tenant_partition("proj-2", "user-a")
    assert a == ("proj-1", "user-a")
    assert a != b
    assert a != c
    assert b != c


def test_tenant_partition_normalizes_to_strings() -> None:
    """Partition values are stringified for stable hashing with Streamlit cache."""
    assert project_entries_cache_tenant_partition("p", "u") == ("p", "u")
