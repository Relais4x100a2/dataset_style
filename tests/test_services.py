"""Unit tests for issue-031 service layer (perimeter, dashboard aggregates, edition filters)."""

from __future__ import annotations

import pandas as pd
import pytest
from src.database import CACHE_COLUMNS, STATUT_VALIDE
from src.export_utils import ExportScope
from src.nlp_engine import EditionScoreFilterSpec
from src.services.dashboard_stylometry_service import (
    count_rows_missing_parseable_coherence_score,
    project_dataset_headline_metrics,
)
from src.services.edition_filters_service import (
    build_edition_score_filter_spec,
    coherence_bucket_label_fr,
)
from src.services.export_scope_service import summarize_export_perimeter
from src.services.project_dataframe_view import (
    normalize_legacy_dimension_columns,
    prepare_for_dashboard_tab,
    prepare_for_edition_tab,
)


def test_normalize_legacy_dimension_columns_forme_to_structure() -> None:
    df = pd.DataFrame({"id": ["1"], "forme": ["Dialogue"], "type": ["T"]})
    out = normalize_legacy_dimension_columns(df)
    assert "structure" in out.columns
    assert out.iloc[0]["structure"] == "Dialogue"
    assert "forme" in out.columns


def test_normalize_legacy_support_to_format_and_public_default() -> None:
    df = pd.DataFrame({"id": ["1"], "support": ["Narratif"], "type": ["T"]})
    out = normalize_legacy_dimension_columns(df)
    assert out.iloc[0]["format"] == "Narratif"
    assert out.iloc[0]["public"] == ""


def test_prepare_for_edition_tab_adds_all_cache_columns() -> None:
    df = pd.DataFrame({"id": ["1"], "type": ["X"], "statut": ["A faire"]})
    out = prepare_for_edition_tab(df)
    for col in CACHE_COLUMNS:
        assert col in out.columns


def test_prepare_for_edition_tab_does_not_mutate_input() -> None:
    df = pd.DataFrame({"id": ["1"], "forme": ["F"], "type": ["T"], "statut": ["S"]})
    before = set(df.columns)
    _ = prepare_for_edition_tab(df)
    assert set(df.columns) == before


def test_prepare_for_dashboard_tab_minimal_cache_columns() -> None:
    df = pd.DataFrame({"id": ["1"], "type": ["T"], "statut": [STATUT_VALIDE]})
    out = prepare_for_dashboard_tab(df)
    for col in ("_coherence_score", "_syntax_contrast", "_signature_json"):
        assert col in out.columns


def test_summarize_export_perimeter_validated_only() -> None:
    df = pd.DataFrame(
        [
            {"id": "a", "statut": STATUT_VALIDE, "type": "T"},
            {"id": "b", "statut": "A faire", "type": "T"},
        ]
    )
    summary = summarize_export_perimeter(df, "validated_only")
    assert summary.row_count == 1
    assert len(summary.dataframe) == 1
    assert summary.recap_warning is None
    assert "1" in summary.recap_caption


def test_summarize_export_perimeter_full_dataset() -> None:
    df = pd.DataFrame([{"id": "a", "statut": STATUT_VALIDE, "type": "T"}])
    summary = summarize_export_perimeter(df, "full_dataset")
    assert summary.row_count == 1


def test_build_edition_score_filter_spec_all() -> None:
    assert build_edition_score_filter_spec("all") == EditionScoreFilterSpec()


def test_build_edition_score_filter_spec_na_only() -> None:
    spec = build_edition_score_filter_spec("na_only")
    assert spec.mode == "na_only"


def test_build_edition_score_filter_spec_below() -> None:
    spec = build_edition_score_filter_spec("below", threshold_lt=40, include_na=True)
    assert spec.mode == "below"
    assert spec.threshold_lt == 40
    assert spec.include_na is True


def test_build_edition_score_filter_spec_bucket() -> None:
    spec = build_edition_score_filter_spec("bucket", bucket_decile=3, include_na=False)
    assert spec.mode == "bucket"
    assert spec.bucket_decile == 3
    assert spec.include_na is False


def test_coherence_bucket_label_fr_decile_zero() -> None:
    label = coherence_bucket_label_fr(0)
    assert label == f"0{'\u2013'}9"


def test_count_rows_missing_parseable_coherence_score() -> None:
    df = pd.DataFrame({"_coherence_score": ["10", "", "n/a", "20"]})
    assert count_rows_missing_parseable_coherence_score(df) == 2


def test_count_rows_missing_parseable_coherence_score_no_column() -> None:
    df = pd.DataFrame({"id": ["1", "2"]})
    assert count_rows_missing_parseable_coherence_score(df) == 2


def test_project_dataset_headline_metrics() -> None:
    df = pd.DataFrame(
        [
            {"id": "1", "statut": STATUT_VALIDE, "type": "A"},
            {"id": "2", "statut": "X", "type": "B"},
        ]
    )
    total, validated, n_types = project_dataset_headline_metrics(
        df, validated_status_label=STATUT_VALIDE
    )
    assert total == 2
    assert validated == 1
    assert n_types == 2


def test_build_edition_score_filter_spec_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        build_edition_score_filter_spec("invalid_mode")  # type: ignore[arg-type]


def test_summarize_export_perimeter_typing_scope() -> None:
    """ExportScope literal is accepted by summarize_export_perimeter."""
    df = pd.DataFrame([{"id": "a", "statut": STATUT_VALIDE, "type": "T"}])
    s: ExportScope = "validated_only"
    assert summarize_export_perimeter(df, s).row_count == 1
