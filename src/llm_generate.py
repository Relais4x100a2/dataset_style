"""
Génération Input / Output par LLM pour le dataset de style.

Backend :
- **OpenRouter** (défaut) si aucune URL locale n'est définie.
- **Ollama** (ou tout serveur compatible OpenAI Chat Completions) si ``LLM_BASE_URL``
  ou ``OLLAMA_BASE_URL`` est défini dans l'environnement.
"""

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_OPENROUTER = "mistralai/mistral-small-creative"


def llm_base_url() -> str:
    """URL de base du serveur compatible OpenAI (sans ``/v1/...``), ou chaîne vide."""
    return (os.environ.get("LLM_BASE_URL") or os.environ.get("OLLAMA_BASE_URL") or "").strip()


def is_local_llm_enabled() -> bool:
    """True si un serveur local (Ollama, etc.) est configuré via l'environnement."""
    return bool(llm_base_url())


def _chat_completions_url(base: str) -> str:
    b = base.rstrip("/")
    if b.endswith("/v1/chat/completions"):
        return b
    return f"{b}/v1/chat/completions"


def _llm_timeout_seconds() -> int:
    raw = (os.environ.get("LLM_TIMEOUT_SECONDS") or "300").strip()
    try:
        return max(30, int(raw))
    except ValueError:
        return 300


def _default_model_id(model_override: str | None) -> str:
    return (
        (model_override or "").strip()
        or (os.environ.get("LLM_MODEL") or "").strip()
        or MODEL_OPENROUTER
    )


def _call_chat_completions(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
) -> str | None:
    """
    Appelle ``/v1/chat/completions`` (OpenRouter ou serveur type Ollama).

    Returns:
        Le contenu texte de la réponse, ou None en cas d'erreur ou configuration invalide.
    """
    model_id = _default_model_id(model)
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    timeout = _llm_timeout_seconds()
    base = llm_base_url()

    if base:
        url = _chat_completions_url(base)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = (api_key or "").strip() or (os.environ.get("LLM_API_KEY") or "").strip()
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
    "littéraire développée qui respecte strictement le type, la forme, le ton et le "
    "support demandés.\n"
    'La prose doit être aboutie, dans un style littéraire, prête à servir d\'"output" '
    "pour un modèle de fine-tuning."
)


def generate_input_from_output(
    api_key: str,
    output: str,
    type_: str,
    forme: str,
    ton: str,
    support: str,
    model: str | None = None,
) -> str | None:
    """
    Génère un brouillon (input) à partir de la prose (output) et des paramètres de style.

    Returns:
        Le texte du brouillon généré, ou None en cas d'erreur ou clé API absente
        (mode OpenRouter uniquement).
    """
    user_prompt = f"""Paramètres de style à prendre en compte :
- Type : {type_}
- Forme : {forme}
- Ton : {ton}
- Support : {support}

Prose à résumer en brouillon :

---
{output}
---

Rédige uniquement le brouillon synthétique (notes, idées), sans introduction ni conclusion."""
    return _call_chat_completions(api_key, SYSTEM_INPUT_FROM_OUTPUT, user_prompt, model=model)


def generate_output_from_input(
    api_key: str,
    input_text: str,
    type_: str,
    forme: str,
    ton: str,
    support: str,
    model: str | None = None,
) -> str | None:
    """
    Génère la prose (output) à partir du brouillon (input) et des paramètres de style.

    Returns:
        Le texte de la prose générée, ou None en cas d'erreur ou clé API absente
        (mode OpenRouter uniquement).
    """
    user_prompt = f"""Paramètres de style à respecter strictement :
- Type : {type_}
- Forme : {forme}
- Ton : {ton}
- Support : {support}

Brouillon (notes, idées) à développer en prose :

---
{input_text}
---

Rédige uniquement la prose littéraire développée, sans introduction ni métadonnées."""
    return _call_chat_completions(api_key, SYSTEM_OUTPUT_FROM_INPUT, user_prompt, model=model)
