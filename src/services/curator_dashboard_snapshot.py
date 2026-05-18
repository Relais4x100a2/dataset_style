"""Envelope JSON « tableau de bord curateur » (issue-014) — parité agrégats Streamlit."""

from __future__ import annotations

import math
from typing import Any, Literal

import pandas as pd

from src.corpus_stylometry_alerts_fr import (
    TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR,
    coherence_distribution_sampling_caption_fr,
    coherence_missing_scores_caption_fr,
    curator_dashboard_refetch_after_entry_mutation_fr,
    dashboard_stylometry_glossary_markdown_fr,
    dashboard_stylometry_scope_caption_all_fr,
    dashboard_stylometry_scope_caption_validated_fr,
    low_coherence_outliers_help_fr,
    no_numeric_coherence_scores_message_fr,
    signature_variance_unavailable_message_fr,
    trivial_syntax_pair_curator_warning_fr,
    trivial_syntax_pair_threshold_rule_sentence_fr,
)
from src.nlp_engine import (
    DASHBOARD_COHERENCE_SCORE_MAX_ROWS_FULL_SCAN,
    DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT,
    SYNTAX_CONTRAST_TRIVIAL_PAIR_THRESHOLD_LT,
    coherence_score_bucket_table,
    count_trivial_syntax_contrast_entries,
    dataframe_for_coherence_distribution_scan,
    dataframe_for_dashboard_scope,
    list_parsed_coherence_scores,
    mean_syntax_contrast_parsed,
    outliers_low_coherence_table,
    signature_variance,
    summarize_parsed_coherence_scores,
    trivial_syntax_contrast_entries_table,
)
from src.services.dashboard_stylometry_service import (
    count_rows_missing_parseable_coherence_score,
    project_dataset_headline_metrics,
)
from src.services.project_dataframe_view import prepare_for_dashboard_tab

DashboardStylometryScope = Literal["validated", "all"]


def _dataframe_records_json_safe(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize rows for JSON (NaN/NA → ``None``)."""
    if df.empty:
        return []
    clean = df.replace({pd.NA: None})
    out: list[dict[str, Any]] = []
    for row in clean.to_dict(orient="records"):
        fixed: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                fixed[k] = None
            else:
                fixed[k] = v
        out.append(fixed)
    return out


def _bucket_rows_for_json(bucket_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, r in bucket_df.iterrows():
        rows.append(
            {
                "tranche": str(r.get("Tranche (score)", "")),
                "nombre": int(r.get("Nombre", 0)),
            }
        )
    return rows


def build_curator_dashboard_envelope(
    df_entries: pd.DataFrame,
    *,
    scope: DashboardStylometryScope,
    validated_label: str,
) -> dict[str, Any]:
    """Construit la charge utile « qualité dataset » alignée sur l’onglet Tableau de bord.

    Les seuils et agrégats transitent exclusivement par ``prepare_for_dashboard_tab``,
    les helpers ``nlp_engine`` et les libellés ``corpus_stylometry_alerts_fr`` (pas de
    duplication de constantes métier côté front).

    Args:
        df_entries: DataFrame projet tel que ``load_project_entries``.
        scope: ``validated`` (fiches validées) ou ``all`` (tous statuts).
        validated_label: Libellé statut validé (ex. ``STATUT_VALIDE``).

    Returns:
        Dictionnaire avec ``dataset_quality``, ``technical`` (``None`` si succès) et
        ``client_contract`` (rafraîchissement après mutation).
    """
    client_contract = {
        "refetch_after_entry_mutation_fr": curator_dashboard_refetch_after_entry_mutation_fr(),
        "dashboard_scope_query_param": "dashboard_scope",
        "dashboard_scope_values": ("validated", "all"),
    }
    if df_entries.empty:
        return {
            "dataset_quality": {
                "empty": True,
                "message_fr": "Aucune donnée.",
            },
            "technical": None,
            "client_contract": client_contract,
        }

    df_view = prepare_for_dashboard_tab(df_entries)
    total_rows, validated_rows, n_types = project_dataset_headline_metrics(
        df_view,
        validated_status_label=validated_label,
    )
    validated_only = scope == "validated"
    scope_df = dataframe_for_dashboard_scope(
        df_view,
        validated_only=validated_only,
        validated_label=validated_label,
    )
    scope_caption = (
        dashboard_stylometry_scope_caption_validated_fr()
        if validated_only
        else dashboard_stylometry_scope_caption_all_fr()
    )

    work_df, used_sample, n_scope = dataframe_for_coherence_distribution_scan(
        scope_df,
        max_rows_without_sampling=DASHBOARD_COHERENCE_SCORE_MAX_ROWS_FULL_SCAN,
    )
    scores = list_parsed_coherence_scores(work_df)
    missing_scores = count_rows_missing_parseable_coherence_score(work_df)
    summary = summarize_parsed_coherence_scores(scores)
    bucket_df = coherence_score_bucket_table(scores) if scores else pd.DataFrame()

    sampling_caption_fr: str | None = None
    if used_sample and n_scope > 0:
        sampling_caption_fr = coherence_distribution_sampling_caption_fr(
            n_scope=n_scope,
            sample_size=len(work_df),
        )

    missing_caption_fr: str | None = None
    if missing_scores:
        missing_caption_fr = coherence_missing_scores_caption_fr(
            missing_scores,
            used_sample=used_sample,
            work_row_count=len(work_df),
            n_scope=n_scope,
        )

    df_valid = dataframe_for_dashboard_scope(
        df_view,
        validated_only=True,
        validated_label=validated_label,
    )
    var_axes = signature_variance(df_valid)
    axis_rows: list[dict[str, Any]] | None = None
    variance_unavailable_fr: str | None = None
    if var_axes is None:
        variance_unavailable_fr = signature_variance_unavailable_message_fr()
    else:
        axis_rows = [
            {"axe": axe, "ecart_type": val}
            for axe, val in sorted(var_axes.items(), key=lambda kv: kv[1], reverse=True)
        ]

    out_tbl = outliers_low_coherence_table(scope_df, limit=DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT)
    n_trivial = count_trivial_syntax_contrast_entries(scope_df)
    trivial_tbl = trivial_syntax_contrast_entries_table(
        scope_df, limit=DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT
    )
    mean_contrast = mean_syntax_contrast_parsed(scope_df)

    preview_cols = [
        c
        for c in ("id", "date", "type", "structure", "ton", "format", "public", "statut")
        if c in df_view.columns
    ]
    preview_df = df_view[preview_cols] if preview_cols else df_view.iloc[0:0].copy()

    alerts: list[dict[str, Any]] = []
    if not scores and not scope_df.empty:
        alerts.append(
            {
                "severity": "info",
                "code": "DATASET_NO_NUMERIC_COHERENCE_SCORES",
                "title_fr": "Information",
                "message_fr": no_numeric_coherence_scores_message_fr(),
            }
        )
    if missing_scores:
        alerts.append(
            {
                "severity": "warning",
                "code": "DATASET_COHERENCE_SCORE_MISSING",
                "title_fr": "Qualité du dataset",
                "message_fr": missing_caption_fr or "",
            }
        )
    if n_trivial > 0:
        alerts.append(
            {
                "severity": "warning",
                "code": "DATASET_TRIVIAL_SYNTAX_PAIRS_PRESENT",
                "title_fr": "Qualité du dataset",
                "message_fr": trivial_syntax_pair_curator_warning_fr(),
            }
        )
    if variance_unavailable_fr is not None and not df_valid.empty:
        alerts.append(
            {
                "severity": "info",
                "code": "DATASET_SIGNATURE_VARIANCE_UNAVAILABLE",
                "title_fr": "Information",
                "message_fr": variance_unavailable_fr,
            }
        )

    dataset_quality: dict[str, Any] = {
        "empty": False,
        "headline": {
            "total_rows": total_rows,
            "validated_rows": validated_rows,
            "distinct_type_count": n_types,
        },
        "stylometry_scope": {
            "value": scope,
            "validated_only": validated_only,
            "caption_fr": scope_caption,
        },
        "constants": {
            "coherence_full_scan_max_rows": DASHBOARD_COHERENCE_SCORE_MAX_ROWS_FULL_SCAN,
            "stylometry_alert_table_limit": DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT,
            "trivial_syntax_contrast_threshold_lt": SYNTAX_CONTRAST_TRIVIAL_PAIR_THRESHOLD_LT,
        },
        "coherence_distribution": {
            "n_scope": n_scope,
            "used_sample": used_sample,
            "work_row_count": len(work_df),
            "sampling_caption_fr": sampling_caption_fr,
            "summary": (
                None
                if summary is None
                else {
                    "mean": summary.mean,
                    "median": summary.median,
                    "minimum": summary.minimum,
                    "count": summary.count,
                }
            ),
            "bucket_counts": _bucket_rows_for_json(bucket_df),
            "missing_parseable_score_count": missing_scores,
            "missing_scores_caption_fr": missing_caption_fr,
        },
        "axis_stddev_validated": {
            "caption_fr": (
                "Indicateur calculé exclusivement sur les fiches validées, conformément au "
                "contrat analytique de signature_variance()."
            ),
            "axes": axis_rows,
            "unavailable_message_fr": variance_unavailable_fr,
        },
        "low_coherence_outliers": {
            "help_markdown_fr": low_coherence_outliers_help_fr(),
            "limit": DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT,
            "rows": _dataframe_records_json_safe(out_tbl),
        },
        "trivial_syntax_pairs": {
            "label_fr": TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR,
            "threshold_rule_fr": trivial_syntax_pair_threshold_rule_sentence_fr(),
            "count": int(n_trivial),
            "rows": _dataframe_records_json_safe(trivial_tbl),
        },
        "mean_syntax_contrast": mean_contrast,
        "glossary_markdown_fr": dashboard_stylometry_glossary_markdown_fr(),
        "entry_preview": {
            "columns": preview_cols,
            "rows": _dataframe_records_json_safe(preview_df),
        },
        "alerts": alerts,
    }

    return {
        "dataset_quality": dataset_quality,
        "technical": None,
        "client_contract": client_contract,
    }
