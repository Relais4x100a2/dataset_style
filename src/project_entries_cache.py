"""Streamlit ``@st.cache_data`` wrapper for ``load_project_entries``.

The underlying loader in :mod:`src.database` stays free of Streamlit so it can be
imported from scripts and tests without a UI runtime. This module adds a short
TTL cache for tab navigation latency. Callers must invoke
:func:`invalidate_project_entries_cache` after **every** successful database write
that affects which rows ``load_project_entries`` would return for a project
(including project lifecycle operations such as create/delete project), **before**
any ``st.rerun()`` that must show fresh data. A TTL alone does not replace this
explicit invalidation (issue-027).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy.engine import Engine

from src.database import load_project_entries

PROJECT_ENTRIES_CACHE_TTL_SECONDS: int = 30


def project_entries_cache_tenant_partition(project_id: str, user_id: str) -> tuple[str, str]:
    """Return the stable (project, account) pair used in the Streamlit cache key.

    ``@st.cache_data`` hashes all positional arguments. Alongside the engine URL
    token (:func:`engine_url_cache_token`), these strings must match the
    ``project_id`` and ``CurrentUser.user_id`` passed from the UI so frames never
    leak across users or projects (same rule as session_state keys in issues 1–2).

    Args:
        project_id: Active project identifier.
        user_id: Authenticated account id (authorization in ``load_project_entries``).

    Returns:
        ``(project_id, user_id)`` as plain strings for hashing.
    """
    return (str(project_id), str(user_id))


def engine_url_cache_token(engine: Engine) -> str:
    """Return a stable hash input for ``Engine`` instances across Streamlit reruns.

    The connection object is recreated each script run; hashing by identity would
    never hit the cache. The SQLAlchemy URL string matches for the same database.

    Args:
        engine: Active SQLAlchemy engine.

    Returns:
        Canonical URL string used as cache partition key.
    """
    return str(engine.url)


@st.cache_data(
    ttl=PROJECT_ENTRIES_CACHE_TTL_SECONDS,
    show_spinner=False,
    hash_funcs={Engine: engine_url_cache_token},
)
def cached_load_project_entries(engine: Engine, project_id: str, user_id: str) -> pd.DataFrame:
    """Load project rows with TTL caching for responsive tab switches.

    Call :func:`invalidate_project_entries_cache` after each successful write that
    changes persisted entries or the active project for this session, before any
    rerun that should show up-to-date rows.

    Args:
        engine: Database engine.
        project_id: Active project identifier.
        user_id: Current account id (authorization in ``load_project_entries``).

    Returns:
        Normalized entries dataframe.
    """
    project_entries_cache_tenant_partition(project_id, user_id)
    return load_project_entries(engine, project_id, user_id)


def invalidate_project_entries_cache() -> None:
    """Drop all cached entry frames (call after successful entries/project writes)."""
    cached_load_project_entries.clear()
