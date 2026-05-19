"""
Presets de dimensions textuelles et utilitaires de persistance.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

from src.database import ProjectSettings

DIMENSION_KEYS: tuple[str, ...] = ("types", "structures", "tons", "formats", "publics", "statuts")
DEFAULT_PRESET_KEY = "roman"

PRESETS: dict[str, dict[str, Any]] = {
    "roman": {
        "label": "Roman / Fiction",
        "types": ["Normalisation", "Expansion", "Réécriture", "Continuation"],
        "structures": [
            "Narration",
            "Description",
            "Portrait",
            "Dialogue",
            "Monologue intérieur",
            "Réflexion",
            "Scène",
        ],
        "tons": [
            "Neutre",
            "Lyrique",
            "Mélancolique",
            "Tendu",
            "Sardonique",
            "Chaleureux",
            "Clinique",
        ],
        "formats": [
            "Chapitre",
            "Épistolaire",
            "Journal intime",
            "Nouvelle",
            "Fragment",
            "Prologue/Épilogue",
        ],
        "publics": ["Lecteur général", "Jeune adulte", "Adulte averti", "Enfant"],
        "statuts": ["A faire", "En cours", "A relire", "Fait et validé"],
    },
    "pro": {
        "label": "Communication professionnelle",
        "types": ["Rédaction", "Réécriture", "Résumé", "Expansion", "Adaptation"],
        "structures": [
            "Argumentaire",
            "Liste",
            "Mémo",
            "Compte-rendu",
            "Q&A",
            "Brief",
            "Procédure",
        ],
        "tons": ["Neutre", "Formel", "Chaleureux", "Assertif", "Diplomatique", "Factuel"],
        "formats": [
            "Email",
            "Rapport",
            "Note interne",
            "Présentation",
            "Lettre officielle",
            "Slack/Chat",
        ],
        "publics": ["Collègue", "Direction", "Client", "Partenaire", "Équipe"],
        "statuts": ["Brouillon", "En revue", "Validé", "Envoyé"],
    },
    "contenu": {
        "label": "Contenu & Marketing",
        "types": ["Génération", "Réécriture", "Adaptation", "Résumé", "Déclinaison"],
        "structures": [
            "Pitch",
            "Storytelling",
            "Tutoriel",
            "Liste",
            "Témoignage",
            "Comparatif",
            "Accroche",
        ],
        "tons": [
            "Enthousiaste",
            "Inspirant",
            "Conversationnel",
            "Provocateur",
            "Expert",
            "Accessible",
        ],
        "formats": [
            "Post LinkedIn",
            "Post Instagram",
            "Article blog",
            "Newsletter",
            "Fiche produit",
            "Script vidéo",
            "Landing page",
        ],
        "publics": ["Audience large", "Prospect", "Client existant", "Communauté", "Décideur B2B"],
        "statuts": ["Idée", "Brouillon", "En validation", "Publié"],
    },
}


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def normalize_dimensions(raw: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in DIMENSION_KEYS:
        out[key] = _clean_list(raw.get(key, []))
    # Valeur de sécurité minimale
    if not out["statuts"]:
        out["statuts"] = ["A faire", "En cours", "A relire", "Fait et validé"]
    return out


def parse_custom_presets(raw_json: str) -> dict[str, dict[str, Any]]:
    payload = (raw_json or "").strip()
    if not payload:
        return {}
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        preset_key = str(key).strip()
        if not preset_key:
            continue
        out[preset_key] = {
            "label": str(value.get("label") or preset_key).strip() or preset_key,
            **normalize_dimensions(value),
        }
    return out


def dumps_custom_presets(custom_presets: dict[str, dict[str, Any]]) -> str:
    if not custom_presets:
        return ""
    return json.dumps(custom_presets, ensure_ascii=False)


def available_presets(custom_presets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged = deepcopy(PRESETS)
    merged.update(custom_presets)
    return merged


def preset_dimensions(preset: dict[str, Any]) -> dict[str, list[str]]:
    return normalize_dimensions(preset)


def parse_dimensions_override(raw_json: str) -> dict[str, list[str]] | None:
    payload = (raw_json or "").strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return normalize_dimensions(data)


def dumps_dimensions_override(dimensions: dict[str, list[str]]) -> str:
    return json.dumps(normalize_dimensions(dimensions), ensure_ascii=False)


STATUTS_LIST_CANNOT_BE_EMPTY_FR = "La liste des statuts ne peut pas être vide."
DIMENSIONS_JSON_OBJECT_EXPECTED_FR = "Dimensions invalides : objet JSON attendu."
RESERVED_BUILTIN_PRESET_KEY_FR = "Ce nom est réservé par un profil fourni avec l'application."
UNKNOWN_PRESET_KEY_FR = "Profil de dimensions inconnu ou indisponible."
CUSTOM_PRESET_ID_REQUIRED_FR = "Identifiant du profil requis."


def validate_replace_dimensions_payload(raw: Any) -> tuple[dict[str, list[str]] | None, str | None]:
    """Valide un remplacement complet des listes (parité UI Streamlit).

    Args:
        raw: Objet JSON des six listes (``types``, ``structures``, etc.).

    Returns:
        Paire ``(dimensions normalisées, message d'erreur FR)``. Si le second
        élément est non nul, la persistance doit être refusée.
    """
    if not isinstance(raw, dict):
        return None, DIMENSIONS_JSON_OBJECT_EXPECTED_FR
    if "statuts" in raw and isinstance(raw["statuts"], list) and not _clean_list(raw["statuts"]):
        return None, STATUTS_LIST_CANNOT_BE_EMPTY_FR
    return normalize_dimensions(raw), None


def normalize_custom_preset_storage_key(raw_name: str) -> str:
    """Identifiant technique du profil (équivalent UI Streamlit, sans espaces)."""
    return str(raw_name or "").strip().lower().replace(" ", "_")


def apply_load_preset_to_settings(
    current: ProjectSettings,
    preset_key: str,
) -> tuple[ProjectSettings | None, str | None]:
    """Applique un profil prédéfini ou personnalisé (clé active + override dimensions)."""
    custom = parse_custom_presets(current.custom_presets_json)
    presets_map = available_presets(custom)
    key = (preset_key or "").strip()
    if key not in presets_map:
        return None, UNKNOWN_PRESET_KEY_FR
    target = preset_dimensions(presets_map[key])
    return (
        replace(
            current,
            active_preset_key=key,
            dimensions_override_json=dumps_dimensions_override(target),
        ),
        None,
    )


def apply_replace_dimensions_to_settings(
    current: ProjectSettings,
    raw_dimensions: Any,
) -> tuple[ProjectSettings | None, str | None]:
    """Met à jour uniquement ``dimensions_override_json`` (clé de preset inchangée)."""
    normalized, err = validate_replace_dimensions_payload(raw_dimensions)
    if err:
        return None, err
    return replace(current, dimensions_override_json=dumps_dimensions_override(normalized)), None


def apply_save_custom_preset_to_settings(
    current: ProjectSettings,
    raw_name: str,
    raw_label: str | None,
    raw_dimensions: Any,
) -> tuple[ProjectSettings | None, str | None]:
    """Fusionne un profil personnalisé dans ``custom_presets_json`` (parité Streamlit)."""
    normalized, err = validate_replace_dimensions_payload(raw_dimensions)
    if err:
        return None, err
    preset_id = normalize_custom_preset_storage_key(raw_name)
    if not preset_id:
        return None, CUSTOM_PRESET_ID_REQUIRED_FR
    if preset_id in PRESETS:
        return None, RESERVED_BUILTIN_PRESET_KEY_FR
    custom = parse_custom_presets(current.custom_presets_json)
    updated = deepcopy(custom)
    saved_label = str(raw_label or "").strip() or preset_id
    updated[preset_id] = {"label": saved_label, **normalized}
    return (
        replace(
            current,
            active_preset_key=preset_id,
            custom_presets_json=dumps_custom_presets(updated),
            dimensions_override_json=dumps_dimensions_override(normalized),
        ),
        None,
    )


def load_active_dimensions(
    settings: ProjectSettings,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, list[str]]]:
    custom = parse_custom_presets(settings.custom_presets_json)
    presets = available_presets(custom)
    active_key = (settings.active_preset_key or "").strip() or DEFAULT_PRESET_KEY
    if active_key not in presets:
        active_key = DEFAULT_PRESET_KEY
    override = parse_dimensions_override(settings.dimensions_override_json)
    if override is not None:
        return active_key, custom, override
    return active_key, custom, preset_dimensions(presets[active_key])
