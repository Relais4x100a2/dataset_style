"""Mutations d'entrées pour le slice vertical (réutilise ``database.update_project_entries``)."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from src.database import (
    CACHE_COLUMNS,
    get_project_settings,
    load_project_entries,
    require_role,
)
from src.presets import load_active_dimensions
from src.services.entry_nlp_persist_service import (
    load_fr_core_nlp_for_webapp,
    persist_edited_entry_with_nlp_cache,
    persist_new_entry_with_nlp_cache,
)


def _resolve_closed_dimension_value(
    dims: dict[str, list[str]],
    dim_key: str,
    override: str | None,
) -> str:
    """Retourne une valeur de dimension ; ``override`` doit être dans la liste projet si fourni."""
    raw = dims.get(dim_key) or [""]
    allowed = [str(x) for x in raw]
    default = str(allowed[0]) if allowed else ""
    if override is None:
        return default
    choice = str(override).strip()
    if choice not in allowed:
        msg = f"Dimension « {dim_key} » : valeur non autorisée pour ce projet."
        raise ValueError(msg)
    return choice


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

# Champs d'entrée → clés ``load_active_dimensions`` (listes fermées), alignés sur ``append_minimal_entry``.
_PATCH_CLOSED_FIELD_TO_DIMS_KEY: dict[str, str] = {
    "type": "types",
    "structure": "structures",
    "ton": "tons",
    "format": "formats",
    "public": "publics",
    "statut": "statuts",
}


def apply_entry_field_updates(
    engine: Engine,
    project_id: str,
    user_id: str,
    entry_id: str,
    updates: dict[str, Any],
) -> None:
    """Met à jour les champs autorisés d'une ligne puis persiste via le cache NLP.

    Les dimensions fermées (``type``, ``structure``, etc.) sont validées contre
    ``load_active_dimensions`` comme à la création (``_resolve_closed_dimension_value``).
    """
    df = load_project_entries(engine, project_id, user_id)
    mask = df["id"] == entry_id
    if not mask.any():
        raise KeyError("entry_not_found")

    dims: dict[str, list[str]] | None = None
    if updates.keys() & _PATCH_CLOSED_FIELD_TO_DIMS_KEY.keys():
        settings = get_project_settings(engine, project_id)
        dims = load_active_dimensions(settings)[2]

    for key, raw in updates.items():
        if key not in _PATCHABLE:
            continue
        if key not in df.columns:
            continue
        if dims is not None and key in _PATCH_CLOSED_FIELD_TO_DIMS_KEY:
            dim_key = _PATCH_CLOSED_FIELD_TO_DIMS_KEY[key]
            df.loc[mask, key] = _resolve_closed_dimension_value(dims, dim_key, str(raw))
        else:
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
    type_: str | None = None,
    structure: str | None = None,
    ton: str | None = None,
    format_: str | None = None,
    public: str | None = None,
    statut: str | None = None,
    notes: str | None = None,
) -> str:
    """Ajoute une fiche avec dimensions du preset actif (admin ou collaborateur).

    Les champs de dimension fermée non fournis prennent la première valeur du preset.
    Une valeur fournie doit appartenir à la liste active du projet.
    """
    require_role(engine, project_id, user_id, ("admin", "collaborator"))
    settings = get_project_settings(engine, project_id)
    _pk, _custom, dims = load_active_dimensions(settings)
    new_id = f"e_{uuid.uuid4().hex[:12]}"
    notes_val = "" if notes is None else str(notes)
    row = {
        "id": new_id,
        "date": date.today().isoformat(),
        "type": _resolve_closed_dimension_value(dims, "types", type_),
        "structure": _resolve_closed_dimension_value(dims, "structures", structure),
        "ton": _resolve_closed_dimension_value(dims, "tons", ton),
        "format": _resolve_closed_dimension_value(dims, "formats", format_),
        "public": _resolve_closed_dimension_value(dims, "publics", public),
        "input": input_text,
        "output": output_text,
        "statut": _resolve_closed_dimension_value(dims, "statuts", statut),
        "notes": notes_val,
    }
    for col in CACHE_COLUMNS:
        row[col] = ""
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
