"""Dépendances FastAPI : moteur SQLAlchemy et utilisateur authentifié."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Header, Request
from sqlalchemy.engine import Engine

from src.api_errors import AuthSessionExpiredError, error_envelope_for_client
from src.database import get_user_record_by_su_user_id
from src.supertokens_recipe_client import (
    extract_su_user_id_from_verify_payload,
    verify_access_token,
)
from src.webapp.errors import EnvelopeHttpError

logger = logging.getLogger(__name__)


def get_engine(request: Request) -> Engine:
    """Retourne le moteur SQLAlchemy attaché à l'application."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise RuntimeError("Moteur SQLAlchemy non initialisé.")
    return engine  # type: ignore[no-any-return]


def _parse_bearer(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthSessionExpiredError()
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise AuthSessionExpiredError()
    return token


def require_app_user_id(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> str:
    """Vérifie le jeton SuperTokens et retourne l'``user_id`` applicatif."""
    try:
        token = _parse_bearer(authorization)
    except AuthSessionExpiredError as exc:
        logger.info("Session refusée ou expirée (jeton absent).")
        raise EnvelopeHttpError(
            401,
            error_envelope_for_client(exc, include_technical_detail=False),
        ) from exc
    engine = get_engine(request)
    try:
        payload = verify_access_token(token)
        su_uid = extract_su_user_id_from_verify_payload(payload)
        if not su_uid:
            raise AuthSessionExpiredError()
        record = get_user_record_by_su_user_id(engine, su_uid)
        if record is None:
            raise AuthSessionExpiredError()
        if (record.disabled_at or "").strip():
            raise AuthSessionExpiredError()
        return record.user_id
    except AuthSessionExpiredError as exc:
        logger.info("Session refusée ou expirée.")
        raise EnvelopeHttpError(
            401,
            error_envelope_for_client(exc, include_technical_detail=False),
        ) from exc
