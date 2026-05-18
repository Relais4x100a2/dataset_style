"""Tests agrégats tableau de bord curateur (issue-014) — parité Streamlit / API."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from src.database import STATUT_VALIDE
from src.services.curator_dashboard_snapshot import build_curator_dashboard_envelope


def _minimal_row(
    rid: str,
    *,
    statut: str = STATUT_VALIDE,
    coherence: str = "80",
    syntax_contrast: str = "0.5",
) -> dict[str, str | int]:
    return {
        "id": rid,
        "project_id": "p1",
        "date": "",
        "type": "T",
        "structure": "",
        "ton": "",
        "format": "",
        "public": "",
        "input": "in",
        "output": "out",
        "statut": statut,
        "notes": "",
        "_coherence_score": coherence,
        "_syntax_contrast": syntax_contrast,
        "_signature_json": '{"a": 0.5}',
    }


def test_empty_project_envelope() -> None:
    out = build_curator_dashboard_envelope(
        pd.DataFrame(),
        scope="validated",
        validated_label=STATUT_VALIDE,
    )
    assert out["technical"] is None
    dq = out["dataset_quality"]
    assert dq["empty"] is True
    assert "Aucune donnée" in dq["message_fr"]


def test_envelope_has_dataset_quality_and_null_technical() -> None:
    df = pd.DataFrame([_minimal_row("e1")])
    out = build_curator_dashboard_envelope(df, scope="validated", validated_label=STATUT_VALIDE)
    assert out["technical"] is None
    assert "headline" in out["dataset_quality"]
    assert out["dataset_quality"]["headline"]["total_rows"] == 1


def test_coherence_sampling_flag_when_over_cap() -> None:
    rows = [_minimal_row(str(i), coherence=str(50 + (i % 10))) for i in range(8)]
    df = pd.DataFrame(rows)
    with patch(
        "src.services.curator_dashboard_snapshot.DASHBOARD_COHERENCE_SCORE_MAX_ROWS_FULL_SCAN",
        5,
    ):
        out = build_curator_dashboard_envelope(df, scope="all", validated_label=STATUT_VALIDE)
    dist = out["dataset_quality"]["coherence_distribution"]
    assert dist["used_sample"] is True
    assert dist["n_scope"] == 8
    assert dist["work_row_count"] == 5
    assert dist["sampling_caption_fr"] is not None
    assert "8" in dist["sampling_caption_fr"]
    assert "5" in dist["sampling_caption_fr"]


def test_table_rows_exclude_signature_json() -> None:
    df = pd.DataFrame(
        [
            _minimal_row("low", coherence="10", syntax_contrast="0.05"),
            _minimal_row("high", coherence="90", syntax_contrast="0.9"),
        ]
    )
    out = build_curator_dashboard_envelope(df, scope="all", validated_label=STATUT_VALIDE)
    trivial_rows = out["dataset_quality"]["trivial_syntax_pairs"]["rows"]
    assert trivial_rows
    for row in trivial_rows:
        assert "_signature_json" not in row
    outliers = out["dataset_quality"]["low_coherence_outliers"]["rows"]
    assert outliers
    for row in outliers:
        assert "_signature_json" not in row


def test_trivial_syntax_triggers_dataset_quality_warning_alert() -> None:
    df = pd.DataFrame([_minimal_row("x", coherence="50", syntax_contrast="0.05")])
    out = build_curator_dashboard_envelope(df, scope="all", validated_label=STATUT_VALIDE)
    codes = {a["code"] for a in out["dataset_quality"]["alerts"]}
    assert "DATASET_TRIVIAL_SYNTAX_PAIRS_PRESENT" in codes


def test_client_contract_documents_refetch() -> None:
    df = pd.DataFrame([_minimal_row("e1")])
    out = build_curator_dashboard_envelope(df, scope="validated", validated_label=STATUT_VALIDE)
    cc = out["client_contract"]
    assert "invalidate_project_entries_cache" in cc["refetch_after_entry_mutation_fr"]
    assert "GET" in cc["refetch_after_entry_mutation_fr"]


def test_entry_preview_excludes_cache_columns() -> None:
    df = pd.DataFrame([_minimal_row("e1")])
    out = build_curator_dashboard_envelope(df, scope="all", validated_label=STATUT_VALIDE)
    prev = out["dataset_quality"]["entry_preview"]["rows"]
    assert prev
    assert "_signature_json" not in prev[0]
    assert "_coherence_score" not in prev[0]
