"""Compose export perimeter slicing with French UI recap strings."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.export_utils import ExportScope, dataframe_for_export, export_perimeter_ui_recap_fr


@dataclass(frozen=True, slots=True)
class ExportPerimeterSummary:
    """Row count, filtered slice, and recap strings for the export tab."""

    row_count: int
    dataframe: pd.DataFrame
    recap_caption: str
    recap_warning: str | None


def summarize_export_perimeter(df: pd.DataFrame, scope: ExportScope) -> ExportPerimeterSummary:
    """Slice ``df`` for the active export scope and build UI recap strings.

    Args:
        df: Full project dataset loaded in the UI.
        scope: ``validated_only`` or ``full_dataset``.

    Returns:
        Summary suitable for metrics, captions, and download buffers.
    """
    slice_df = dataframe_for_export(df, scope)
    row_count = len(slice_df)
    recap_caption, recap_warning = export_perimeter_ui_recap_fr(row_count, scope)
    return ExportPerimeterSummary(
        row_count=row_count,
        dataframe=slice_df,
        recap_caption=recap_caption,
        recap_warning=recap_warning,
    )
