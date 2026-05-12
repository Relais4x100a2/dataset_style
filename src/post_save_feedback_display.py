"""Pure UI copy for post-save stylometric feedback (Streamlit-free, testable)."""


def post_save_stylistic_metric_labels_fr() -> dict[str, str]:
    """Return French business labels keyed for coherence, TTR, and syntax contrast metrics.

    Values intentionally include the internal cache column names so curators can
    map UI to exports and database columns.

    Returns:
        Mapping with keys ``coherence_score``, ``ttr``, ``syntax_contrast``.
    """
    return {
        "coherence_score": "Score de cohérence (_coherence_score)",
        "ttr": "TTR — richesse lexicale (_ttr)",
        "syntax_contrast": "Contraste syntaxique brouillon ↔ généré (_syntax_contrast)",
    }


def post_save_freshness_caption_fr(*, synchronous_before_commit: bool) -> str:
    """Short caption explaining how fresh persisted NLP cache fields are for the UI.

    Args:
        synchronous_before_commit: When True, NLP cache for the saved row is filled
            before the SQL commit (current architecture). When False, reserved for a
            future deferred pipeline where metrics may lag.

    Returns:
        One or two sentences for ``st.caption`` under the post-save panel.
    """
    if synchronous_before_commit:
        return (
            "Source : ligne relue en base après commit ; le cache stylométrique "
            "(score, TTR, contraste) est calculé de façon synchrone avant l’écriture SQL — "
            "pas d’analyse différée côté serveur pour l’instant."
        )
    return (
        "Analyse NLP différée : les métriques affichées peuvent être provisoires jusqu’à "
        "mise à jour des champs de cache en base."
    )
