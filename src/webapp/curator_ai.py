"""Parité curateur : génération IA et LanguageTool (issue-013).

Logique serveur partagée par le slice FastAPI ; réutilise ``llm_generate``,
``nlp_engine`` et les primitives ``database`` / ``presets`` (pas de clé API
exposée au client).
"""

from __future__ import annotations

from typing import Any, Literal

from sqlalchemy.engine import Engine

from src.database import get_project_settings, load_project_entries, require_role
from src.llm_generate import (
    LlmRuntimeSettings,
    generate_input_from_output,
    generate_output_from_input,
)
from src.nlp_engine import languagetool_fr_corrected_with_matches
from src.presets import load_active_dimensions

CURATOR_GENERATION_FAILED_FR = (
    "La génération a échoué. Vérifiez l'URL du service, le modèle d'IA et la clé "
    "d'API dans Réglages projet, puis réessayez."
)

CURATOR_LLM_EMPTY_DRAFT_FR = (
    "Saisissez un brouillon avant de lancer la génération du texte assistée par l'IA."
)

CURATOR_LLM_EMPTY_OUTPUT_FR = (
    "Saisissez un texte généré (output) avant de lancer la génération du brouillon."
)

LT_UNAVAILABLE_TITLE_FR = "Correction linguistique indisponible"
LT_UNAVAILABLE_MESSAGE_FR = (
    "Impossible de joindre le service LanguageTool (réseau ou délai dépassé). "
    "Vérifiez la connectivité ou réessayez."
)
LT_UNAVAILABLE_ACTION_FR = "Réessayez dans quelques instants ou contactez un administrateur."


def curator_languagetool_unavailable_envelope() -> dict[str, Any]:
    """Enveloppe JSON (issue-005) lorsque LanguageTool est injoignable."""
    return {
        "error": {
            "code": "CURATOR_LANGUAGETOOL_UNAVAILABLE",
            "title": LT_UNAVAILABLE_TITLE_FR,
            "message": LT_UNAVAILABLE_MESSAGE_FR,
            "suggested_action": LT_UNAVAILABLE_ACTION_FR,
            "detail": None,
        }
    }


def build_curator_dimensions_payload(
    engine: Engine, project_id: str, user_id: str
) -> dict[str, Any]:
    """Dimensions actives du projet après contrôle d'accès lecture (viewer inclus)."""
    load_project_entries(engine, project_id, user_id)
    settings = get_project_settings(engine, project_id)
    active_key, _custom, dims = load_active_dimensions(settings)
    return {"activePresetKey": active_key, "dimensions": dims}


def run_curator_llm_generate(
    engine: Engine,
    project_id: str,
    user_id: str,
    *,
    mode: Literal["draft_to_output", "output_to_draft"],
    input_text: str,
    output_text: str,
    type_: str,
    structure: str,
    ton: str,
    format_: str,
    public: str,
) -> dict[str, Any]:
    """Appelle ``generate_*`` avec les réglages projet (timeouts et URL côté serveur)."""
    require_role(engine, project_id, user_id, ("admin", "collaborator"))
    settings = get_project_settings(engine, project_id)
    runtime = LlmRuntimeSettings(
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        llm_api_key=settings.llm_api_key,
        llm_timeout_seconds=settings.llm_timeout_seconds,
    )
    if mode == "draft_to_output":
        if not input_text.strip():
            return {"status": "validation_error", "message": CURATOR_LLM_EMPTY_DRAFT_FR}
        generated = generate_output_from_input(
            api_key=settings.llm_api_key,
            input_text=input_text,
            type_=type_,
            structure=structure,
            ton=ton,
            format_=format_,
            public=public,
            model=settings.llm_model or None,
            settings=runtime,
        )
        if generated:
            return {"status": "ok", "text": str(generated)}
        return {"status": "failed", "message": CURATOR_GENERATION_FAILED_FR}

    if not output_text.strip():
        return {"status": "validation_error", "message": CURATOR_LLM_EMPTY_OUTPUT_FR}
    generated = generate_input_from_output(
        api_key=settings.llm_api_key,
        output=output_text,
        type_=type_,
        structure=structure,
        ton=ton,
        format_=format_,
        public=public,
        model=settings.llm_model or None,
        settings=runtime,
    )
    if generated:
        return {"status": "ok", "text": str(generated)}
    return {"status": "failed", "message": CURATOR_GENERATION_FAILED_FR}


def run_curator_languagetool_check(
    engine: Engine,
    project_id: str,
    user_id: str,
    *,
    text: str,
) -> dict[str, Any]:
    """Contrôle LanguageTool : texte corrigé + suggestions (une requête HTTP LT)."""
    require_role(engine, project_id, user_id, ("admin", "collaborator"))
    settings = get_project_settings(engine, project_id)
    base = settings.languagetool_base_url or None
    corrected, matches = languagetool_fr_corrected_with_matches(text, languagetool_base_url=base)
    return {"corrected": corrected, "matches": matches}
