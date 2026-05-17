"""Client HTTP minimal vers SuperTokens Core (CDI), partagé par Streamlit et le slice web."""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from src.api_errors import AuthSessionExpiredError

logger = logging.getLogger(__name__)


def _base_url() -> str:
    return (os.environ.get("SUPERTOKENS_CONNECTION_URI") or "").strip().rstrip("/")


def _headers() -> dict[str, str]:
    api_key = (os.environ.get("SUPERTOKENS_API_KEY") or "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    return headers


def recipe_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON vers le core SuperTokens ; lève ``RuntimeError`` si HTTP >= 400."""
    base = _base_url()
    if not base:
        raise RuntimeError("SUPERTOKENS_CONNECTION_URI manquant.")
    resp = requests.post(f"{base}{path}", json=payload, headers=_headers(), timeout=20)
    if resp.status_code >= 400:
        body = (resp.text or "").strip()
        raise RuntimeError(f"SuperTokens {path} HTTP {resp.status_code}: {body}")
    return resp.json()


def signin_email_password(email: str, password: str) -> dict[str, Any]:
    """Connexion email / mot de passe (variantes formFields vs champs plats)."""
    normalized = email.strip().lower()
    try:
        return recipe_post(
            "/recipe/signin",
            {
                "formFields": [
                    {"id": "email", "value": normalized},
                    {"id": "password", "value": password},
                ]
            },
        )
    except RuntimeError as exc:
        if "Field name 'email' is invalid in JSON input" not in str(exc):
            raise
        return recipe_post(
            "/recipe/signin",
            {"email": normalized, "password": password},
        )


def signup_email_password(email: str, password: str) -> dict[str, Any]:
    """Inscription provider (réservé aux flux invitation / bootstrap)."""
    normalized = email.strip().lower()
    try:
        return recipe_post(
            "/recipe/signup",
            {
                "formFields": [
                    {"id": "email", "value": normalized},
                    {"id": "password", "value": password},
                ]
            },
        )
    except RuntimeError as exc:
        if "Field name 'email' is invalid in JSON input" not in str(exc):
            raise
        return recipe_post(
            "/recipe/signup",
            {"email": normalized, "password": password},
        )


def verify_access_token(access_token: str) -> dict[str, Any]:
    """Vérifie un jeton d'accès ; lève ``AuthSessionExpiredError`` si session invalide."""
    out = recipe_post("/recipe/session/verify", {"accessToken": access_token.strip()})
    status = str(out.get("status") or "").strip()
    if status != "OK":
        raise AuthSessionExpiredError()
    return out


def try_revoke_access_token(access_token: str) -> None:
    """Révoque la session côté provider si l'API est disponible (best-effort)."""
    try:
        recipe_post("/recipe/session/revoke", {"accessToken": access_token.strip()})
    except Exception:  # noqa: BLE001
        logger.debug("try_revoke_access_token: échec ignoré", exc_info=True)


def extract_su_user_id_from_verify_payload(payload: dict[str, Any]) -> str:
    """Extrait l'identifiant utilisateur SuperTokens depuis la réponse ``/session/verify``."""
    direct = payload.get("userId") or payload.get("user_id")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    session = payload.get("session")
    if isinstance(session, dict):
        nested = session.get("userId") or session.get("user_id")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return ""
