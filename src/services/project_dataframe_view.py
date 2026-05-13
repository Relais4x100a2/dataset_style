"""Normalize project entry dataframes for edition, dashboard, and NLP persistence."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from src.database import CACHE_COLUMNS

_NlpCachePolicy = Literal["edition", "dashboard_stylometry"]

_DASHBOARD_STYLOMETRY_CACHE_COLS: tuple[str, ...] = (
    "_coherence_score",
    "_syntax_contrast",
    "_signature_json",
)


def normalize_legacy_dimension_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy ``df`` and align legacy column names (``forme``/``support``) with current schema.

    Args:
        df: Raw project entries (possibly older column names).

    Returns:
        Dataframe copy with ``structure``, ``format``, and ``public`` columns guaranteed
        when legacy sources exist; ``public`` defaults to empty string when absent.
    """
    out = df.copy()
    if "structure" not in out.columns and "forme" in out.columns:
        out["structure"] = out["forme"]
    if "format" not in out.columns and "support" in out.columns:
        out["format"] = out["support"]
    if "public" not in out.columns:
        out["public"] = ""
    return out


def ensure_nlp_cache_columns(df: pd.DataFrame, *, policy: _NlpCachePolicy) -> pd.DataFrame:
    """Ensure NLP-related cache columns exist (empty string when missing).

    Args:
        df: Project entries, already normalized for dimension aliases when applicable.
        policy: ``edition`` fills every ``CACHE_COLUMNS`` entry; ``dashboard_stylometry``
            only the three columns used by stylometry widgets on the dashboard.

    Returns:
        A copy of ``df`` with missing cache columns added.
    """
    out = df.copy()
    if policy == "edition":
        cols: tuple[str, ...] = CACHE_COLUMNS
    else:
        cols = _DASHBOARD_STYLOMETRY_CACHE_COLS
    for col in cols:
        if col not in out.columns:
            out[col] = ""
    return out


def prepare_for_edition_tab(df: pd.DataFrame) -> pd.DataFrame:
    """Return a view suitable for edition filters and NLP row operations."""
    return ensure_nlp_cache_columns(normalize_legacy_dimension_columns(df), policy="edition")


def prepare_for_dashboard_tab(df: pd.DataFrame) -> pd.DataFrame:
    """Return a view suitable for dashboard stylometry metrics."""
    return ensure_nlp_cache_columns(
        normalize_legacy_dimension_columns(df),
        policy="dashboard_stylometry",
    )
