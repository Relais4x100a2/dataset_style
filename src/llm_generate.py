"""
Génération Input / Output par LLM (OpenRouter) pour le dataset de style.

Utilise les paramètres type, forme, ton et support pour générer soit un brouillon
à partir de la prose, soit la prose à partir du brouillon.
"""
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_OPENROUTER = "mistralai/mistral-small-creative"


def _call_openrouter(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
) -> str | None:
    """
    Appelle l'API OpenRouter (chat completions).

    Args:
        model: ID du modèle OpenRouter (ex. mistralai/mistral-small-creative).
               Si vide ou None, utilise MODEL_OPENROUTER.

    Returns:
        Le contenu texte de la réponse, ou None en cas d'erreur.
    """
    if not (api_key and api_key.strip()):
        return None
    model_id = (model or "").strip() or MODEL_OPENROUTER
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/dataset-style/app",
        "X-OpenRouter-Title": "Dataset Style Studio",
    }
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    try:
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            logger.warning("OpenRouter: réponse sans choices")
            return None
        content = choices[0].get("message", {}).get("content")
        return (content or "").strip() or None
    except requests.Timeout:
        logger.warning("OpenRouter: timeout")
        return None
    except requests.RequestException as e:
        logger.exception("OpenRouter: %s", e)
        return None
    except (KeyError, TypeError, ValueError) as e:
        logger.exception("OpenRouter: parsing réponse %s", e)
        return None


SYSTEM_INPUT_FROM_OUTPUT = """Tu es un assistant pour la création de jeux de données littéraires.
Ta tâche : à partir d'une prose littéraire, produire un brouillon synthétique (notes brutes, idées, plan) qui aurait pu servir de base à cette prose.
Le brouillon doit être concis, en style note, sans développer les phrases. Il sera utilisé comme "input" pour un modèle de fine-tuning."""

SYSTEM_OUTPUT_FROM_INPUT = """Tu es un assistant pour la création de jeux de données littéraires.
Ta tâche : à partir d'un brouillon synthétique (notes, idées), rédiger une prose littéraire développée qui respecte strictement le type, la forme, le ton et le support demandés.
La prose doit être aboutie, dans un style littéraire, prête à servir d'"output" pour un modèle de fine-tuning."""


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
        Le texte du brouillon généré, ou None en cas d'erreur ou clé API absente.
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
    return _call_openrouter(api_key, SYSTEM_INPUT_FROM_OUTPUT, user_prompt, model=model)


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
        Le texte de la prose générée, ou None en cas d'erreur ou clé API absente.
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
    return _call_openrouter(api_key, SYSTEM_OUTPUT_FROM_INPUT, user_prompt, model=model)
