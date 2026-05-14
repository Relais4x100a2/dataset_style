"""Tests for shared French copy on corpus stylometry alerts (issue-023)."""

from __future__ import annotations

from src.corpus_stylometry_alerts_fr import (
    TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR,
    dashboard_stylometry_glossary_markdown_fr,
    trivial_syntax_contrast_missing_cache_caption_fr,
    trivial_syntax_pair_curator_warning_fr,
    trivial_syntax_pair_threshold_rule_sentence_fr,
)
from src.nlp_engine import (
    DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT,
    EXPORT_PERIMETER_LOW_COHERENCE_OUTLIER_COUNT_THRESHOLD_LT,
    SYNTAX_CONTRAST_TRIVIAL_PAIR_THRESHOLD_LT,
)


def test_trivial_pair_business_label_is_stable() -> None:
    assert TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR == "Paire quasi identique"


def test_threshold_rule_sentence_reflects_engine_constant() -> None:
    text = trivial_syntax_pair_threshold_rule_sentence_fr()
    assert str(SYNTAX_CONTRAST_TRIVIAL_PAIR_THRESHOLD_LT) in text
    assert TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR in text
    assert "_syntax_contrast" in text


def test_curator_warning_mentions_label_and_optional_contrast() -> None:
    base = trivial_syntax_pair_curator_warning_fr()
    assert TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR in base
    assert "_syntax_contrast" not in base
    with_val = trivial_syntax_pair_curator_warning_fr(contrast_raw_display="0,12")
    assert "0,12" in with_val
    assert "_syntax_contrast" in with_val


def test_missing_cache_caption_uses_same_business_label() -> None:
    cap = trivial_syntax_contrast_missing_cache_caption_fr()
    assert TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR in cap


def test_dashboard_glossary_covers_variance_outliers_and_trivial_pairs() -> None:
    md = dashboard_stylometry_glossary_markdown_fr()
    assert "écart-type" in md.lower() or "Écart-type" in md
    assert str(DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT) in md
    assert str(SYNTAX_CONTRAST_TRIVIAL_PAIR_THRESHOLD_LT) in md
    assert str(EXPORT_PERIMETER_LOW_COHERENCE_OUTLIER_COUNT_THRESHOLD_LT) in md
    assert TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR in md
