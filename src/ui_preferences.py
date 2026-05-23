"""Préférences d'affichage curateur (densité, confort lecture) — issue-023 / #186.

La persistance SQL est dans ``users.ui_preferences_json`` (voir ``database.ensure_schema``).
Les clés exposées au client REST sont en camelCase (``density``, ``readingComfort``).

Couplage issue-022 (design tokens) : le rendu s'appuie sur ``data-ds-density`` et
``data-ds-reading`` sur ``document.documentElement`` ; les bandeaux d'erreur et zones
sensibles sont exclus du scope CSS (voir ``docs/ui_display_preferences.md``).
"""

from __future__ import annotations

import json
from typing import Any, Final

UI_PREFERENCES_JSON_MAX_BYTES: Final[int] = 4096

_DENSITY: Final[frozenset[str]] = frozenset({"default", "compact", "comfortable"})
_READING: Final[frozenset[str]] = frozenset({"default", "high_contrast", "reduced_motion"})


def default_ui_preferences() -> dict[str, str]:
    """Valeurs par défaut = expérience recommandée sans réglages personnalisés."""
    return {"density": "default", "readingComfort": "default"}


def raise_if_preferences_json_too_large(json_text: str) -> None:
    """Lève ``ValueError`` si la charge utile JSON dépasse le plafond UTF-8."""
    if len(json_text.encode("utf-8")) > UI_PREFERENCES_JSON_MAX_BYTES:
        raise ValueError("Préférences trop volumineuses.")


def load_from_stored_raw(raw: str | None) -> dict[str, str]:
    """Interprète le texte SQL ; JSON illisible ou non objet → défauts sûrs."""
    if raw is None or not str(raw).strip():
        return default_ui_preferences()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return default_ui_preferences()
    if not isinstance(obj, dict):
        return default_ui_preferences()
    return _sanitize_object(obj)


def _sanitize_object(obj: dict[str, Any]) -> dict[str, str]:
    out = default_ui_preferences()
    d = obj.get("density")
    if isinstance(d, str) and d in _DENSITY:
        out["density"] = d
    rc = obj.get("readingComfort")
    if isinstance(rc, str) and rc in _READING:
        out["readingComfort"] = rc
    return out


def merge_patch_into_canonical(
    current: dict[str, str],
    patch: dict[str, Any],
) -> dict[str, str]:
    """Fusionne un patch partiel ; clés inconnues ou valeurs invalides → ``ValueError``."""
    if not patch:
        return dict(current)
    out = dict(current)
    for key, val in patch.items():
        if val is None:
            continue
        if key not in ("density", "readingComfort"):
            raise ValueError(f"Champ de préférences inconnu : {key!r}.")
        if not isinstance(val, str):
            raise ValueError(f"Valeur invalide pour {key!r}.")
        if key == "density":
            if val not in _DENSITY:
                raise ValueError(f"Valeur « density » non autorisée : {val!r}.")
            out["density"] = val
        elif key == "readingComfort":
            if val not in _READING:
                raise ValueError(f"Valeur « readingComfort » non autorisée : {val!r}.")
            out["readingComfort"] = val
    return out


def serialize_canonical_preferences(prefs: dict[str, str]) -> str:
    """Sérialise le dict canonique (exactement deux clés) avec contrôle de taille."""
    if set(prefs.keys()) != {"density", "readingComfort"}:
        raise ValueError("Préférences canoniques attendues : density, readingComfort.")
    if prefs["density"] not in _DENSITY or prefs["readingComfort"] not in _READING:
        raise ValueError("Préférences canoniques invalides.")
    json_text = json.dumps(
        {"density": prefs["density"], "readingComfort": prefs["readingComfort"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    raise_if_preferences_json_too_large(json_text)
    return json_text
