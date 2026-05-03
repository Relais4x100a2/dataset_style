"""
Configuration runtime unifiée (dev + prod).

Ordre de priorité:
1) variables d'environnement déjà présentes
2) APP_CONFIG_JSON (objet JSON injecté en une seule variable)
3) .env local
4) variables dérivées (DATABASE_URL, SUPERTOKENS_CONNECTION_URI)
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_json_config() -> dict[str, str]:
    raw = _normalize_value(os.environ.get("APP_CONFIG_JSON"))
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("APP_CONFIG_JSON doit être un objet JSON.")
    return {str(k): _normalize_value(v) for k, v in data.items()}


def _apply_defaults(payload: dict[str, str]) -> None:
    for key, value in payload.items():
        if key not in os.environ and value:
            os.environ[key] = value


def _derive_database_url() -> None:
    if _normalize_value(os.environ.get("DATABASE_URL")):
        return

    host = _normalize_value(os.environ.get("POSTGRES_HOST")) or "localhost"
    port = _normalize_value(os.environ.get("POSTGRES_PORT")) or "5432"
    database = _normalize_value(os.environ.get("POSTGRES_DB"))
    user = _normalize_value(os.environ.get("POSTGRES_USER"))
    password = _normalize_value(os.environ.get("POSTGRES_PASSWORD"))

    if not all([database, user, password]):
        return

    os.environ["DATABASE_URL"] = (
        f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
    )


def _derive_su_connection_uri() -> None:
    if _normalize_value(os.environ.get("SUPERTOKENS_CONNECTION_URI")):
        return

    host = _normalize_value(os.environ.get("SUPERTOKENS_HOST")) or "localhost"
    port = _normalize_value(os.environ.get("SUPERTOKENS_PORT")) or "3567"
    os.environ["SUPERTOKENS_CONNECTION_URI"] = f"http://{host}:{port}"


def initialize_runtime_config() -> None:
    """Charge et harmonise la configuration runtime."""
    load_dotenv()
    _apply_defaults(_load_json_config())
    _derive_database_url()
    _derive_su_connection_uri()
