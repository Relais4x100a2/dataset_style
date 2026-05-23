"""Lecture et mutation des presets / dimensions projet (issue-011, aligné ``src/presets``).

La persistance reste ``project_settings`` (``active_preset_key``, ``custom_presets_json``,
``dimensions_override_json``) via :func:`src.database.update_project_settings`.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy.engine import Engine

from src.database import (
    get_project_settings,
    get_role,
    load_project_entries,
    require_admin,
    update_project_settings,
)
from src.presets import (
    apply_load_preset_to_settings,
    apply_replace_dimensions_to_settings,
    apply_save_custom_preset_to_settings,
    available_presets,
    load_active_dimensions,
)


class ProjectDimensionsPatchBody(BaseModel):
    """Corps ``PATCH`` pour les trois actions alignées sur l’UI Streamlit (dimensions)."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["load_preset", "replace_dimensions", "save_custom_preset"]
    preset_key: str | None = None
    dimensions: dict[str, Any] | None = None
    custom_preset_name: str | None = None
    custom_preset_label: str | None = None

    @model_validator(mode="after")
    def _fields_for_action(self) -> Self:
        if self.action == "load_preset":
            if not (self.preset_key or "").strip():
                raise ValueError("Le paramètre preset_key est requis pour charger un profil.")
        elif self.action == "replace_dimensions":
            if self.dimensions is None:
                raise ValueError("Le champ dimensions est requis pour remplacer les listes.")
        elif self.action == "save_custom_preset":
            if self.dimensions is None:
                raise ValueError(
                    "Le champ dimensions est requis pour enregistrer un profil personnalisé."
                )
        return self


def build_project_dimensions_settings_payload(
    engine: Engine,
    project_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Charge le contexte dimensions après contrôle d’accès lecture (viewer inclus)."""
    load_project_entries(engine, project_id, user_id)
    role = get_role(engine, project_id, user_id)
    settings = get_project_settings(engine, project_id)
    active_key, custom, dims = load_active_dimensions(settings)
    presets_map = available_presets(custom)
    presets = [{"key": k, "label": str(v.get("label") or k)} for k, v in presets_map.items()]
    can_edit = role == "admin"
    return {
        "activePresetKey": active_key,
        "dimensions": dims,
        "presets": presets,
        "projectRole": role or "",
        "canEditDimensions": can_edit,
    }


def apply_project_dimensions_settings_patch(
    engine: Engine,
    project_id: str,
    user_id: str,
    body: ProjectDimensionsPatchBody,
) -> tuple[dict[str, Any] | None, str | None]:
    """Applique une mutation dimensions ; ``require_admin`` puis ``update_project_settings``."""
    require_admin(engine, project_id, user_id)
    current = get_project_settings(engine, project_id)
    if body.action == "load_preset":
        merged, err = apply_load_preset_to_settings(current, body.preset_key or "")
    elif body.action == "replace_dimensions":
        merged, err = apply_replace_dimensions_to_settings(current, body.dimensions)
    else:
        merged, err = apply_save_custom_preset_to_settings(
            current,
            body.custom_preset_name or "",
            body.custom_preset_label,
            body.dimensions or {},
        )
    if err:
        return None, err
    update_project_settings(engine, project_id, merged)
    return build_project_dimensions_settings_payload(engine, project_id, user_id), None
