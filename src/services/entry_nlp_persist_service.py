"""Chemin de persistance entrées + cache NLP partagé Streamlit / webapp (issue-012)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from src.database import CACHE_COLUMNS, update_project_entries
from src.nlp_engine import avg_signature_from_cache, compute_row_cache
from src.services.project_dataframe_view import prepare_for_edition_tab

logger = logging.getLogger(__name__)


def load_fr_core_nlp_optional() -> Any:
    """Charge ``fr_core_news_sm`` ou retourne ``None`` si indisponible.

    Streamlit enveloppe cet appel avec ``@st.cache_resource`` ; la webapp utilise
    :func:`load_fr_core_nlp_for_webapp` pour un cache processus sans Streamlit.

    Returns:
        Pipeline spaCy ou ``None``.
    """
    try:
        import spacy

        return spacy.load("fr_core_news_sm")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Impossible de charger fr_core_news_sm: %s", exc)
        return None


@lru_cache(maxsize=1)
def load_fr_core_nlp_for_webapp() -> Any:
    """Instance spaCy mise en cache pour le service FastAPI (un chargement par worker)."""
    return load_fr_core_nlp_optional()


def persist_new_entry_with_nlp_cache(
    engine: Engine,
    project_id: str,
    user_id: str,
    *,
    df_existing: pd.DataFrame,
    new_row_df: pd.DataFrame,
    input_text: str,
    output_text: str,
    nlp: Any,
) -> str:
    """Ajoute une ligne, calcule le cache NLP comme l'onglet Streamlit, persiste.

    ``update_project_entries`` applique ``require_role(..., ("admin", "collaborator"))``.

    Args:
        engine: Moteur SQLAlchemy.
        project_id: Identifiant projet.
        user_id: Acteur courant.
        df_existing: DataFrame ``load_project_entries`` avant ajout.
        new_row_df: Un seul enregistrement (avec ``id`` déjà fixé).
        input_text: Texte input persisté.
        output_text: Texte output persisté.
        nlp: Pipeline spaCy ou ``None``.

    Returns:
        Identifiant de la nouvelle fiche.
    """
    df_base = prepare_for_edition_tab(df_existing)
    new_row_prepared = prepare_for_edition_tab(new_row_df)
    combined = pd.concat([df_base, new_row_prepared], ignore_index=True)
    row_id = str(new_row_prepared.iloc[0]["id"])
    pkg = compute_row_cache(
        input_text,
        output_text,
        nlp,
        combined,
        row_id,
        CACHE_COLUMNS,
        avg_signature_from_cache,
    )
    for col, val in pkg.cache.items():
        new_row_prepared.at[0, col] = val
    to_persist = pd.concat([df_base, new_row_prepared], ignore_index=True)
    update_project_entries(engine, project_id, to_persist, user_id)
    return row_id


def persist_edited_entry_with_nlp_cache(
    engine: Engine,
    project_id: str,
    user_id: str,
    *,
    df_full: pd.DataFrame,
    entry_id: str,
    input_text: str,
    output_text: str,
    nlp: Any,
) -> None:
    """Recalcule le cache NLP pour une ligne modifiée et persiste tout le dataset.

    Args:
        engine: Moteur SQLAlchemy.
        project_id: Identifiant projet.
        user_id: Acteur courant.
        df_full: DataFrame projet après application des champs édités (copie mutable).
        entry_id: Identifiant de la fiche à rafraîchir côté NLP.
        input_text: Input persisté pour le calcul cache.
        output_text: Output persisté pour le calcul cache.
        nlp: Pipeline spaCy ou ``None``.
    """
    out = prepare_for_edition_tab(df_full)
    eid = str(entry_id)
    pkg = compute_row_cache(
        input_text,
        output_text,
        nlp,
        out,
        eid,
        CACHE_COLUMNS,
        avg_signature_from_cache,
    )
    for col, val in pkg.cache.items():
        out.loc[out["id"].astype(str) == eid, col] = val
    update_project_entries(engine, project_id, out, user_id)
