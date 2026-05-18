"""Mutations d'entrées pour le slice vertical (réutilise ``database.update_project_entries``)."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from src.database import get_project_settings, load_project_entries, require_role
from src.presets import load_active_dimensions
from src.services.entry_nlp_persist_service import (
    load_fr_core_nlp_for_webapp,
    persist_edited_entry_with_nlp_cache,
    persist_new_entry_with_nlp_cache,
)

_PATCHABLE: frozenset[str] = frozenset(
    {
        "input",
        "output",
        "statut",
        "notes",
        "type",
        "structure",
        "ton",
        "format",
        "public",
        "date",
    }
)


def apply_entry_field_updates(
    engine: Engine,
    project_id: str,
    user_id: str,
    entry_id: str,
    updates: dict[str, Any],
) -> None:
    """Met à jour les champs autorisés d'une ligne puis persiste via ``update_project_entries``."""
    df = load_project_entries(engine, project_id, user_id)
    mask = df["id"] == entry_id
    if not mask.any():
        raise KeyError("entry_not_found")
    for key, raw in updates.items():
        if key not in _PATCHABLE:
            continue
        if key not in df.columns:
            continue
        df.loc[mask, key] = str(raw)
    input_text = str(df.loc[mask, "input"].iloc[0])
    output_text = str(df.loc[mask, "output"].iloc[0])
    persist_edited_entry_with_nlp_cache(
        engine,
        project_id,
        user_id,
        df_full=df,
        entry_id=entry_id,
        input_text=input_text,
        output_text=output_text,
        nlp=load_fr_core_nlp_for_webapp(),
    )


def append_minimal_entry(
    engine: Engine,
    project_id: str,
    user_id: str,
    *,
    input_text: str,
    output_text: str,
) -> str:
    """Ajoute une fiche avec dimensions par défaut du preset actif (admin ou collaborateur)."""
    require_role(engine, project_id, user_id, ("admin", "collaborator"))
    settings = get_project_settings(engine, project_id)
    _pk, _custom, dims = load_active_dimensions(settings)
    types = dims.get("types") or [""]
    structures = dims.get("structures") or [""]
    tons = dims.get("tons") or [""]
    formats = dims.get("formats") or [""]
    publics = dims.get("publics") or [""]
    statuts = dims.get("statuts") or ["En cours"]
    new_id = f"e_{uuid.uuid4().hex[:12]}"
    row = {
        "id": new_id,
        "date": date.today().isoformat(),
        "type": str(types[0]),
        "structure": str(structures[0]),
        "ton": str(tons[0]),
        "format": str(formats[0]),
        "public": str(publics[0]),
        "input": input_text,
        "output": output_text,
        "statut": str(statuts[0]),
        "notes": "",
    }
    df = load_project_entries(engine, project_id, user_id)
    new_row_df = pd.DataFrame([row])
    persist_new_entry_with_nlp_cache(
        engine,
        project_id,
        user_id,
        df_existing=df,
        new_row_df=new_row_df,
        input_text=input_text,
        output_text=output_text,
        nlp=load_fr_core_nlp_for_webapp(),
    )
    return new_id
