"""
Presets de dimensions textuelles et utilitaires de persistance.
"""

from __future__ import annotations

import json
from copy import deepcopy
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
