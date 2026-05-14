"""Tests for export perimeter quality recap (issue-036)."""

from __future__ import annotations

import pandas as pd
from src.database import STATUT_VALIDE
from src.nlp_engine import (
    EXPORT_PERIMETER_COHERENCE_MEAN_ALERT_LT,
    EXPORT_PERIMETER_LOW_COHERENCE_OUTLIER_COUNT_THRESHOLD_LT,
)
from src.services.export_quality_recap_service import build_export_quality_recap


def _row(
    rid: str,
    statut: str,
    score: str,
) -> dict[str, str]:
    return {"id": rid, "statut": statut, "type": "T", "_coherence_score": score}


def test_build_export_quality_recap_empty_export_frame() -> None:
    df = pd.DataFrame(columns=["id", "statut", "type", "_coherence_score"])
    recap = build_export_quality_recap(df)
    assert recap.export_row_count == 0
    assert recap.validated_row_count == 0
    assert recap.coherence_mean is None
    assert recap.low_coherence_outlier_count == 0
    assert recap.coherence_mean_alert is False


def test_build_export_quality_recap_validated_scope_counts_and_mean() -> None:
    df = pd.DataFrame(
        [
            _row("1", STATUT_VALIDE, "80"),
            _row("2", STATUT_VALIDE, "60"),
        ]
    )
    recap = build_export_quality_recap(df)
    assert recap.export_row_count == 2
    assert recap.validated_row_count == 2
    assert recap.coherence_mean == 70.0
    assert recap.low_coherence_outlier_count == 0
    assert recap.coherence_mean_alert is (70.0 < EXPORT_PERIMETER_COHERENCE_MEAN_ALERT_LT)


def test_build_export_quality_recap_outliers_use_documented_threshold() -> None:
    thr = EXPORT_PERIMETER_LOW_COHERENCE_OUTLIER_COUNT_THRESHOLD_LT
    df = pd.DataFrame(
        [
            _row("1", STATUT_VALIDE, str(thr - 1)),
            _row("2", STATUT_VALIDE, str(thr)),
            _row("3", STATUT_VALIDE, str(thr + 5)),
        ]
    )
    recap = build_export_quality_recap(df)
    assert recap.low_coherence_outlier_count == 1


def test_build_export_quality_recap_full_dataset_validated_subset() -> None:
    df = pd.DataFrame(
        [
            _row("1", STATUT_VALIDE, "50"),
            _row("2", "A faire", "50"),
            _row("3", "A faire", ""),
        ]
    )
    recap = build_export_quality_recap(df)
    assert recap.export_row_count == 3
    assert recap.validated_row_count == 1


def test_build_export_quality_recap_mean_triggers_product_alert() -> None:
    mean_lt = EXPORT_PERIMETER_COHERENCE_MEAN_ALERT_LT
    df = pd.DataFrame(
        [
            _row("1", STATUT_VALIDE, str(mean_lt - 5)),
            _row("2", STATUT_VALIDE, str(mean_lt - 5)),
        ]
    )
    recap = build_export_quality_recap(df)
    assert recap.coherence_mean is not None
    assert recap.coherence_mean < mean_lt
    assert recap.coherence_mean_alert is True


def test_build_export_quality_recap_custom_statut_column() -> None:
    df = pd.DataFrame([{"id": "1", "status": STATUT_VALIDE, "_coherence_score": "10"}])
    recap = build_export_quality_recap(df, statut_column="status")
    assert recap.validated_row_count == 1
