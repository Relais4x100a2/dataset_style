"""
Génération Input / Output par LLM pour le dataset de style.

Backend :
- **OpenRouter** (défaut) si aucune URL locale n'est définie.
- **Ollama** (ou tout serveur compatible OpenAI Chat Completions) si ``LLM_BASE_URL``
  ou ``OLLAMA_BASE_URL`` est défini dans l'environnement.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_OPENROUTER = "mistralai/mistral-small-creative"


@dataclass
class LlmRuntimeSettings:
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_timeout_seconds: str = ""


def llm_base_url(settings: LlmRuntimeSettings | None = None) -> str:
    """URL de base du serveur compatible OpenAI (sans ``/v1/...``), ou chaîne vide."""
    if settings and settings.llm_base_url.strip():
        return settings.llm_base_url.strip()
    return (os.environ.get("LLM_BASE_URL") or os.environ.get("OLLAMA_BASE_URL") or "").strip()


def is_local_llm_enabled(settings: LlmRuntimeSettings | None = None) -> bool:
    """True si un serveur local (Ollama, etc.) est configuré via l'environnement."""
    return bool(llm_base_url(settings))


def _chat_completions_url(base: str) -> str:
    b = base.rstrip("/")
    if b.endswith("/v1/chat/completions"):
        return b
    return f"{b}/v1/chat/completions"


def _llm_timeout_seconds(settings: LlmRuntimeSettings | None = None) -> int:
    raw = ""
    if settings:
        raw = settings.llm_timeout_seconds.strip()
    if not raw:
        raw = (os.environ.get("LLM_TIMEOUT_SECONDS") or "300").strip()
    try:
        return max(30, int(raw))
    except ValueError:
        return 300


def _default_model_id(
    model_override: str | None, settings: LlmRuntimeSettings | None = None
) -> str:
    return (
        (model_override or "").strip()
        or ((settings.llm_model.strip()) if settings else "")
        or (os.environ.get("LLM_MODEL") or "").strip()
        or MODEL_OPENROUTER
    )


def _call_chat_completions(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    settings: LlmRuntimeSettings | None = None,
) -> str | None:
    """
    Appelle ``/v1/chat/completions`` (OpenRouter ou serveur type Ollama).

    Returns:
        Le contenu texte de la réponse, ou None en cas d'erreur ou configuration invalide.
    """
    model_id = _default_model_id(model, settings)
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    timeout = _llm_timeout_seconds(settings)
    base = llm_base_url(settings)

    if base:
        url = _chat_completions_url(base)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = (
            (api_key or "").strip()
            or ((settings.llm_api_key.strip()) if settings else "")
            or (os.environ.get("LLM_API_KEY") or "").strip()
        )
        if key:
            headers["Authorization"] = f"Bearer {key}"
    else:
        if not (api_key and api_key.strip()):
            return None
        url = OPENROUTER_URL
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/dataset-style/app",
            "X-OpenRouter-Title": "Dataset Style Studio",
        }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            logger.warning("LLM: réponse sans choices")
            return None
        content = choices[0].get("message", {}).get("content")
        return (content or "").strip() or None
    except requests.Timeout:
        logger.warning("LLM: timeout après %ss", timeout)
        return None
    except requests.RequestException as e:
        logger.exception("LLM: %s", e)
        return None
    except (KeyError, TypeError, ValueError) as e:
        logger.exception("LLM: parsing réponse %s", e)
        return None


SYSTEM_INPUT_FROM_OUTPUT = (
    "Tu es un assistant pour la création de jeux de données littéraires.\n"
    "Ta tâche : à partir d'une prose littéraire, produire un brouillon synthétique "
    "(notes brutes, idées, plan) qui aurait pu servir de base à cette prose.\n"
    "Le brouillon doit être concis, en style note, sans développer les phrases. "
    'Il sera utilisé comme "input" pour un modèle de fine-tuning.'
)

SYSTEM_OUTPUT_FROM_INPUT = (
    "Tu es un assistant pour la création de jeux de données littéraires.\n"
    "Ta tâche : à partir d'un brouillon synthétique (notes, idées), rédiger une prose "
    "littéraire développée qui respecte strictement le type de transformation, la "
    "structure textuelle, la tonalité textuelle, le format de sortie et le public cible "
    "demandés.\n"
    'La prose doit être aboutie, dans un style littéraire, prête à servir d\'"output" '
    "pour un modèle de fine-tuning."
)


def generate_input_from_output(
    api_key: str,
    output: str,
    type_: str,
    structure: str,
    ton: str,
    format_: str,
    public: str,
    model: str | None = None,
    settings: LlmRuntimeSettings | None = None,
) -> str | None:
    """
    Génère un brouillon (input) à partir de la prose (output) et des paramètres de style.

    Returns:
        Le texte du brouillon généré, ou None en cas d'erreur ou clé API absente
        (mode OpenRouter uniquement).
    """
    user_prompt = f"""Paramètres de style à prendre en compte :
- Type de transformation : {type_}
- Structure textuelle : {structure}
- Tonalité textuelle : {ton}
- Format de sortie : {format_}
- Public cible : {public}

Prose à résumer en brouillon :

---
{output}
---

Rédige uniquement le brouillon synthétique (notes, idées), sans introduction ni conclusion."""
    return _call_chat_completions(
        api_key,
        SYSTEM_INPUT_FROM_OUTPUT,
        user_prompt,
        model=model,
        settings=settings,
    )


def generate_output_from_input(
    api_key: str,
    input_text: str,
    type_: str,
    structure: str,
    ton: str,
    format_: str,
    public: str,
    model: str | None = None,
    settings: LlmRuntimeSettings | None = None,
) -> str | None:
    """
    Génère la prose (output) à partir du brouillon (input) et des paramètres de style.

    Returns:
        Le texte de la prose générée, ou None en cas d'erreur ou clé API absente
        (mode OpenRouter uniquement).
    """
    user_prompt = f"""Paramètres de style à respecter strictement :
- Type de transformation : {type_}
- Structure textuelle : {structure}
- Tonalité textuelle : {ton}
- Format de sortie : {format_}
- Public cible : {public}

Brouillon (notes, idées) à développer en prose :

---
{input_text}
---

Rédige uniquement la prose littéraire développée, sans introduction ni métadonnées."""
    return _call_chat_completions(
        api_key,
        SYSTEM_OUTPUT_FROM_INPUT,
        user_prompt,
        model=model,
        settings=settings,
    )
