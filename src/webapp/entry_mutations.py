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
    require_admin,
    update_project_entries,
)
from src.presets import load_active_dimensions

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
    update_project_entries(engine, project_id, df, user_id)


def append_minimal_entry(
    engine: Engine,
    project_id: str,
    user_id: str,
    *,
    input_text: str,
    output_text: str,
) -> str:
    """Ajoute une fiche avec dimensions par défaut du preset actif (propriétaire / admin)."""
    require_admin(engine, project_id, user_id)
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
    for col in CACHE_COLUMNS:
        row[col] = ""
    df = load_project_entries(engine, project_id, user_id)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    update_project_entries(engine, project_id, df, user_id)
    return new_id
