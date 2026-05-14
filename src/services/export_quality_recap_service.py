"""Quality recap for the active export perimeter (issue-036)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.database import STATUT_VALIDE
from src.nlp_engine import (
    EXPORT_PERIMETER_COHERENCE_MEAN_ALERT_LT,
    EXPORT_PERIMETER_LOW_COHERENCE_OUTLIER_COUNT_THRESHOLD_LT,
    count_persisted_coherence_scores_strictly_below,
    list_parsed_coherence_scores,
    summarize_parsed_coherence_scores,
)


@dataclass(frozen=True, slots=True)
class ExportQualityRecap:
    """Aggregates on ``df_export`` (perimeter slice), not the raw project frame."""

    export_row_count: int
    validated_row_count: int
    coherence_mean: float | None
    low_coherence_outlier_count: int
    coherence_mean_alert: bool


def build_export_quality_recap(
    df_export: pd.DataFrame,
    *,
    statut_column: str = "statut",
) -> ExportQualityRecap:
    """Build curator-facing export recap metrics (validated count, mean coherence, outliers).

    All coherence statistics use the same parsers as the dashboard and
    :func:`~src.nlp_engine.list_parsed_coherence_scores` on ``df_export`` only.

    Args:
        df_export: Rows included in CSV/JSONL for the current scope
            (same object as :attr:`ExportPerimeterSummary.dataframe`).
        statut_column: Status column name (defaults to project schema).

    Returns:
        Recap suitable for metrics above download buttons.
    """
    n_export = len(df_export)
    if n_export == 0 or statut_column not in df_export.columns:
        validated = 0
    else:
        validated = int((df_export[statut_column].astype(str) == str(STATUT_VALIDE)).sum())

    scores = list_parsed_coherence_scores(df_export)
    summary = summarize_parsed_coherence_scores(scores)
    mean_val = summary.mean if summary is not None else None

    outlier_count = count_persisted_coherence_scores_strictly_below(
        df_export,
        threshold_lt=EXPORT_PERIMETER_LOW_COHERENCE_OUTLIER_COUNT_THRESHOLD_LT,
    )

    mean_alert = mean_val is not None and mean_val < float(EXPORT_PERIMETER_COHERENCE_MEAN_ALERT_LT)

    return ExportQualityRecap(
        export_row_count=n_export,
        validated_row_count=validated,
        coherence_mean=mean_val,
        low_coherence_outlier_count=outlier_count,
        coherence_mean_alert=mean_alert,
    )
