"""Dashboard headline counts and coherence coverage helpers."""

from __future__ import annotations

import pandas as pd

from src.nlp_engine import parse_persisted_coherence_score


def project_dataset_headline_metrics(
    df: pd.DataFrame,
    *,
    validated_status_label: str,
) -> tuple[int, int, int]:
    """Return total row count, validated count, and distinct ``type`` values.

    Args:
        df: Prepared project view (non-empty expected by callers for display).
        validated_status_label: Canonical validated status (``STATUT_VALIDE`` in DB).

    Returns:
        ``(total_rows, validated_rows, distinct_type_count)``
    """
    total = len(df)
    validated = int((df["statut"] == validated_status_label).sum())
    n_types = int(df["type"].nunique())
    return total, validated, n_types


def count_rows_missing_parseable_coherence_score(work_df: pd.DataFrame) -> int:
    """Count rows whose ``_coherence_score`` cell is not parseable as a 0–100 score.

    Uses the same parser as the dashboard distribution and edition filters.

    Args:
        work_df: Sample or full dataframe (possibly from
            :func:`src.nlp_engine.dataframe_for_coherence_distribution_scan`).

    Returns:
        Number of rows treated as missing a numeric coherence score.
    """
    if "_coherence_score" not in work_df.columns:
        return len(work_df)
    return sum(
        1
        for v in work_df["_coherence_score"].tolist()
        if parse_persisted_coherence_score(v) is None
    )
