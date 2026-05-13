"""Map edition filter widget selections to :class:`~src.nlp_engine.EditionScoreFilterSpec`."""

from __future__ import annotations

from typing import Literal

from src.nlp_engine import EditionScoreFilterSpec, _edition_coherence_bucket_bounds

EditionScoreFilterMode = Literal["all", "below", "bucket", "na_only"]

# Same dash as the former inline Streamlit label (U+2013 EN DASH).
_BUCKET_RANGE_SEP = "–"


def coherence_bucket_label_fr(decile: int) -> str:
    """Human-readable coherence bucket label for a decile index (0–9)."""
    lo, hi = _edition_coherence_bucket_bounds(decile)
    return f"{lo}{_BUCKET_RANGE_SEP}{hi}"


def build_edition_score_filter_spec(
    mode: EditionScoreFilterMode | str,
    *,
    threshold_lt: int = 50,
    bucket_decile: int = 0,
    include_na: bool = False,
) -> EditionScoreFilterSpec:
    """Build a score filter spec from widget-level mode and parameters.

    Args:
        mode: Widget mode (``all``, ``below``, ``bucket``, ``na_only``).
        threshold_lt: Exclusive upper bound when ``mode == "below"``.
        bucket_decile: Decile index 0–9 when ``mode == "bucket"``.
        include_na: Whether N/A scores are kept for ``below`` / ``bucket``.

    Returns:
        Frozen :class:`EditionScoreFilterSpec` passed to
        :func:`src.nlp_engine.filter_edition_entries_dataframe`.

    Raises:
        ValueError: If ``mode`` is not a supported widget value.
    """
    if mode == "all":
        return EditionScoreFilterSpec()
    if mode == "na_only":
        return EditionScoreFilterSpec(mode="na_only")
    if mode == "below":
        return EditionScoreFilterSpec(
            mode="below",
            threshold_lt=threshold_lt,
            include_na=include_na,
        )
    if mode == "bucket":
        return EditionScoreFilterSpec(
            mode="bucket",
            bucket_decile=bucket_decile,
            include_na=include_na,
        )
    msg = f"Unknown edition score filter mode: {mode!r}"
    raise ValueError(msg)
