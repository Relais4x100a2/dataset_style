"""French copy for corpus-wide stylometry alerts (dashboard, edition, post-save).

The numeric trivial-pair bound for persisted ``_syntax_contrast`` lives only in
:mod:`src.nlp_engine` as ``SYNTAX_CONTRAST_TRIVIAL_PAIR_THRESHOLD_LT`` (strict
``<``). This module imports it for explanatory text so thresholds are never
duplicated as literals in UI strings.
"""

from __future__ import annotations

from src.nlp_engine import (
    DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT,
    EXPORT_PERIMETER_LOW_COHERENCE_OUTLIER_COUNT_THRESHOLD_LT,
    SYNTAX_CONTRAST_TRIVIAL_PAIR_THRESHOLD_LT,
)

TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR: str = "Paire quasi identique"


def trivial_syntax_pair_threshold_rule_sentence_fr() -> str:
    """One sentence: strict threshold and unparseable cells (French, curator-facing)."""
    thr = SYNTAX_CONTRAST_TRIVIAL_PAIR_THRESHOLD_LT
    return (
        f"Alerte « {TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR} » : `_syntax_contrast` mesuré et "
        f"strictement inférieur à {thr} (la valeur {thr} est exclue). Cellule vide ou "
        "invalide → pas d'alerte."
    )


def trivial_syntax_pair_curator_warning_fr(*, contrast_raw_display: str | None = None) -> str:
    """Body for ``st.warning`` (post-save, edition). Optional raw persisted value as secondary."""
    core = (
        f"{TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR} : le texte généré ressemble fortement au "
        "brouillon au niveau des motifs grammaticaux (hors sens sémantique ni similarité "
        "textuelle). Pour le fine-tuning, la paire risque d'apporter peu d'information "
        "nouvelle — envisagez de réviser ou d'exclure l'exemple."
    )
    trimmed = (contrast_raw_display or "").strip()
    if trimmed:
        return f"{core}\n\nValeur `_syntax_contrast` (persistée) : {trimmed}."
    return core


def trivial_syntax_contrast_missing_cache_caption_fr() -> str:
    """Caption when no parseable contrast is available (edition tab)."""
    return (
        "Contraste syntaxique non disponible pour cette fiche (cache vide ou invalide). "
        f"L'alerte « {TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR} » ne s'affiche pas tant qu'une "
        "mesure exploitable n'est pas présente dans `_syntax_contrast`."
    )


def signature_axis_variance_help_fr() -> str:
    """Markdown paragraph: aggregated axis variance (validated scope, no raw JSON)."""
    return (
        "**Écart-type par axe** : résumé agrégé des signatures stylométriques persistées "
        "(`_signature_json` côté données, jamais affiché brut ici). Calcul **uniquement sur "
        "les fiches au statut validé** ; un axe n'apparaît que s'il comporte au moins deux "
        "observations numériques. Un écart-type élevé sur un axe indique une dispersion "
        "stylométrique du corpus sur cette dimension."
    )


def low_coherence_outliers_help_fr() -> str:
    """Markdown paragraph: top-N lowest persisted coherence scores."""
    n = DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT
    thr = EXPORT_PERIMETER_LOW_COHERENCE_OUTLIER_COUNT_THRESHOLD_LT
    return (
        f"**Scores de cohérence bas** : jusqu'à {n} entrées du périmètre sélectionné avec "
        "les valeurs `_coherence_score` numériques les plus faibles (tri croissant). "
        "Identifiant et statut pour retrouver la fiche dans « Gestion & édition »."
        "\n\n**Export (Réglages & export)** : le compteur « scores bas » avant téléchargement "
        f"compte les fiches du périmètre CSV/JSONL avec score parseable strictement sous **{thr}** "
        "(constante produit), indépendamment de ce classement top-N."
    )


def trivial_syntax_pair_block_help_fr() -> str:
    """Markdown paragraph: trivial syntax pair alert + semantics."""
    return (
        "**Contraste syntaxique** : "
        + trivial_syntax_pair_threshold_rule_sentence_fr()
        + " Indicateur basé sur des motifs grammaticaux (brouillon ↔ généré), pas une "
        "mesure de similarité sémantique."
    )


def dashboard_stylometry_glossary_markdown_fr() -> str:
    """Full glossary for the dashboard expander (French)."""
    parts = [
        signature_axis_variance_help_fr(),
        "",
        low_coherence_outliers_help_fr(),
        "",
        trivial_syntax_pair_block_help_fr(),
    ]
    return "\n".join(parts)
