"""
Diagnostics démarrage base de données : classification d'erreurs et messages UI.

Sans dépendance Streamlit — testable unitairement.
"""

from __future__ import annotations

import os
import re
from typing import Literal

from sqlalchemy.exc import ArgumentError, OperationalError

DbFailureCategory = Literal["missing_url", "invalid_config", "connection", "other"]

_DEV_ENV_VALUES = frozenset({"development", "dev", "local", "debug"})


def is_development_ui() -> bool:
    """Retourne True si l'UI peut afficher des détails techniques (dev / flag explicite)."""
    raw = (
        os.environ.get("APP_ENV")
        or os.environ.get("ENVIRONMENT")
        or os.environ.get("STREAMLIT_ENV")
        or ""
    ).strip().lower()
    if raw in _DEV_ENV_VALUES:
        return True
    flag = os.environ.get("SHOW_DB_TECHNICAL_ERRORS", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def classify_database_startup_error(exc: BaseException) -> DbFailureCategory:
    """Classifie une exception levée lors de la création du moteur ou du schéma."""
    if isinstance(exc, OperationalError):
        return "connection"
    if isinstance(exc, ArgumentError):
        return "invalid_config"
    msg = str(exc).lower()
    if any(
        token in msg
        for token in (
            "could not connect",
            "connection refused",
            "connection timed out",
            "timeout expired",
            "server closed the connection",
            "network is unreachable",
            "name or service not known",
            "temporary failure in name resolution",
        )
    ):
        return "connection"
    if "invalid" in msg and ("url" in msg or "dsn" in msg or "database" in msg):
        return "invalid_config"
    return "other"


def user_facing_summary(category: DbFailureCategory) -> str:
    """Message principal en français, ton neutre (affiché en production comme en dev)."""
    if category == "missing_url":
        return (
            "L'application n'est pas correctement configurée côté serveur. "
            "Ce n'est pas un problème lié à votre compte. "
            "Veuillez contacter l'administrateur ou le support."
        )
    if category == "invalid_config":
        return (
            "La connexion à la base de données est invalide côté serveur. "
            "Ce n'est pas un problème lié à votre compte. "
            "Veuillez contacter l'administrateur ou le support."
        )
    if category == "connection":
        return (
            "Le service de données est momentanément indisponible ou inaccessible. "
            "Ce n'est généralement pas lié à votre compte. "
            "Réessayez plus tard ; si le problème persiste, contactez l'administrateur ou le support."
        )
    return (
        "Une erreur technique empêche l'accès aux données. "
        "Les équipes techniques peuvent consulter les journaux du serveur pour le diagnostic. "
        "Si vous avez besoin d'aide, contactez l'administrateur ou le support."
    )


def technical_hint_for_dev(exc: BaseException | None, *, category: DbFailureCategory) -> str:
    """Texte court pour la zone « détail technique » en environnement de développement."""
    if category == "missing_url":
        return (
            "Variable DATABASE_URL absente ou vide après chargement de la configuration "
            "(variables d'environnement, APP_CONFIG_JSON, dérivation POSTGRES_* ou secrets Streamlit)."
        )
    if exc is None:
        return "Aucune exception source disponible."
    return f"{type(exc).__name__}: {_sanitize_technical_message(str(exc))}"


def _sanitize_technical_message(text: str, *, max_len: int = 800) -> str:
    """Évite d'exposer des secrets accidentels dans l'UI même en mode dev."""
    cleaned = re.sub(
        r"(password|pwd|secret|token)\s*[=:]\s*[^\s&]+",
        r"\1=[redacted]",
        text,
        flags=re.IGNORECASE,
    )
    if len(cleaned) > max_len:
        return f"{cleaned[: max_len - 3]}..."
    return cleaned
