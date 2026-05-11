"""Streamlit ``@st.cache_data`` wrapper for ``load_project_entries``.

The underlying loader in :mod:`src.database` stays free of Streamlit so it can be
imported from scripts and tests without a UI runtime. This module adds a short
TTL cache for tab navigation latency; callers must invoke
:func:`invalidate_project_entries_cache` after any mutation of ``entries`` rows.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy.engine import Engine

from src.database import load_project_entries

PROJECT_ENTRIES_CACHE_TTL_SECONDS: int = 30


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

    Call :func:`invalidate_project_entries_cache` after ``update_project_entries``
    or any other write affecting ``entries`` so the next read reflects the DB.

    Args:
        engine: Database engine.
        project_id: Active project identifier.
        user_id: Current account id (authorization in ``load_project_entries``).

    Returns:
        Normalized entries dataframe.
    """
    return load_project_entries(engine, project_id, user_id)


def invalidate_project_entries_cache() -> None:
    """Drop all cached entry frames (call after successful entries mutations)."""
    cached_load_project_entries.clear()
