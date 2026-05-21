"""
Contrat d'erreurs API / futur BFF : codes stables, textes FR, statuts HTTP.

Les clients (front, intégrations) doivent s'appuyer sur ``error.code``, jamais sur
des messages SuperTokens/SQL bruts. Voir ``docs/api_error_contract.md``.

``include_technical_detail`` / détail JSON : aligné sur ``is_development_ui()`` dans
``src/db_startup.py`` lorsque l'appelant passe ``None``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Final

from sqlalchemy.exc import OperationalError

from src.db_startup import DbFailureCategory, is_development_ui, user_facing_summary

AUTH_SESSION_EXPIRED: Final = "AUTH_SESSION_EXPIRED"
DB_UNAVAILABLE: Final = "DB_UNAVAILABLE"
FORBIDDEN: Final = "FORBIDDEN"
NOT_FOUND_GENERIC: Final = "NOT_FOUND_GENERIC"
INTERNAL_ERROR: Final = "INTERNAL_ERROR"
EXPORT_PAYLOAD_TOO_LARGE: Final = "EXPORT_PAYLOAD_TOO_LARGE"
CURATOR_LANGUAGETOOL_UNAVAILABLE: Final = "CURATOR_LANGUAGETOOL_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ResolvedApiError:
    """Représentation normalisée d'une erreur exposée au client."""

    code: str
    http_status: int
    title_fr: str
    message_fr: str
    suggested_action_fr: str


class AuthSessionExpiredError(Exception):
    """Jeton de session expiré, révoqué ou refusé (côté provider / BFF)."""


class TenantResourceOpaqueDenial(Exception):
    """Accès à une ressource tenantée refusé : réponse identique à « introuvable » (anti-IDOR)."""


class ExportPayloadTooLargeError(Exception):
    """Export refusé : le périmètre dépasse la limite ``WEBAPP_EXPORT_MAX_ROWS`` (issue-015)."""

    def __init__(self, row_count: int, max_rows: int) -> None:
        self.row_count = row_count
        self.max_rows = max_rows
        super().__init__(f"export rows {row_count} > cap {max_rows}")


_CATALOG: dict[str, ResolvedApiError] = {
    AUTH_SESSION_EXPIRED: ResolvedApiError(
        code=AUTH_SESSION_EXPIRED,
        http_status=401,
        title_fr="Session expirée",
        message_fr="Votre session a expiré ou n'est plus valide.",
        suggested_action_fr="Déconnectez-vous puis reconnectez-vous.",
    ),
    DB_UNAVAILABLE: ResolvedApiError(
        code=DB_UNAVAILABLE,
        http_status=503,
        title_fr="Service de données indisponible",
        message_fr=(
            "Le service de données est momentanément indisponible ou inaccessible. "
            "Ce n'est généralement pas lié à votre compte."
        ),
        suggested_action_fr="Réessayez dans quelques minutes ; si le problème persiste, contactez l'administrateur.",
    ),
    FORBIDDEN: ResolvedApiError(
        code=FORBIDDEN,
        http_status=403,
        title_fr="Accès refusé",
        message_fr="Vous n'avez pas les droits suffisants pour cette opération.",
        suggested_action_fr="Si vous pensez qu'il s'agit d'une erreur, contactez un administrateur.",
    ),
    NOT_FOUND_GENERIC: ResolvedApiError(
        code=NOT_FOUND_GENERIC,
        http_status=404,
        title_fr="Ressource introuvable",
        message_fr=(
            "Cette ressource n'existe pas, n'est plus disponible, ou vous n'y avez pas accès."
        ),
        suggested_action_fr="Vérifiez votre sélection ou l'URL ; reconnectez-vous si besoin.",
    ),
    EXPORT_PAYLOAD_TOO_LARGE: ResolvedApiError(
        code=EXPORT_PAYLOAD_TOO_LARGE,
        http_status=413,
        title_fr="Export trop volumineux",
        message_fr=(
            "Le nombre de fiches dans ce périmètre dépasse la limite configurée pour ce service. "
            "Réduisez le périmètre (validées seulement) ou fractionnez l’export."
        ),
        suggested_action_fr=(
            "Contactez l’administrateur pour relever la limite ``WEBAPP_EXPORT_MAX_ROWS`` "
            "ou exportez depuis le studio Streamlit."
        ),
    ),
    INTERNAL_ERROR: ResolvedApiError(
        code=INTERNAL_ERROR,
        http_status=500,
        title_fr="Erreur interne",
        message_fr="Une erreur technique s'est produite.",
        suggested_action_fr="Réessayez plus tard. Si le problème persiste, contactez l'administrateur.",
    ),
    CURATOR_LANGUAGETOOL_UNAVAILABLE: ResolvedApiError(
        code=CURATOR_LANGUAGETOOL_UNAVAILABLE,
        http_status=503,
        title_fr="Correction linguistique indisponible",
        message_fr=(
            "Impossible de joindre le service LanguageTool (réseau ou délai dépassé). "
            "Vérifiez la connectivité ou réessayez."
        ),
        suggested_action_fr=("Réessayez dans quelques instants ou contactez un administrateur."),
    ),
}


def _catalog_entry(code: str) -> ResolvedApiError:
    return _CATALOG.get(code, _CATALOG[INTERNAL_ERROR])


def curator_languagetool_unavailable_envelope() -> dict[str, Any]:
    """Enveloppe JSON ``CURATOR_LANGUAGETOOL_UNAVAILABLE`` (issue-006 / parité issue-005)."""
    resolved = _catalog_entry(CURATOR_LANGUAGETOOL_UNAVAILABLE)
    return {
        "error": {
            "code": resolved.code,
            "title": resolved.title_fr,
            "message": resolved.message_fr,
            "suggested_action": resolved.suggested_action_fr,
            "detail": None,
        }
    }


def resolve_db_startup_category(category: DbFailureCategory) -> ResolvedApiError:
    """Erreur de démarrage base : code stable + texte utilisateur (``user_facing_summary``)."""
    base = _catalog_entry(DB_UNAVAILABLE)
    return ResolvedApiError(
        code=base.code,
        http_status=base.http_status,
        title_fr=base.title_fr,
        message_fr=user_facing_summary(category),
        suggested_action_fr=base.suggested_action_fr,
    )


def resolve_exception_for_api(
    exc: BaseException,
    *,
    include_technical_detail: bool,
) -> ResolvedApiError:
    """Mappe une exception applicative vers code / HTTP / messages FR."""
    if isinstance(exc, AuthSessionExpiredError):
        return _catalog_entry(AUTH_SESSION_EXPIRED)
    if isinstance(exc, TenantResourceOpaqueDenial):
        return _catalog_entry(NOT_FOUND_GENERIC)
    if isinstance(exc, ExportPayloadTooLargeError):
        base = _catalog_entry(EXPORT_PAYLOAD_TOO_LARGE)
        return ResolvedApiError(
            code=base.code,
            http_status=base.http_status,
            title_fr=base.title_fr,
            message_fr=(
                f"{base.message_fr} (ici : {exc.row_count} fiches, limite {exc.max_rows})."
            ),
            suggested_action_fr=base.suggested_action_fr,
        )
    if isinstance(exc, OperationalError):
        return _catalog_entry(DB_UNAVAILABLE)
    if isinstance(exc, PermissionError):
        return _catalog_entry(FORBIDDEN)
    return _catalog_entry(INTERNAL_ERROR)


def _effective_dev_detail(include_technical_detail: bool | None) -> bool:
    if include_technical_detail is None:
        return is_development_ui()
    return include_technical_detail


def _sanitize_detail_fragment(text: str, *, max_len: int = 800) -> str:
    cleaned = re.sub(
        r"(password|pwd|secret|token)\s*[=:]\s*[^\s&]+",
        r"\1=[redacted]",
        text,
        flags=re.IGNORECASE,
    )
    if len(cleaned) > max_len:
        return f"{cleaned[: max_len - 3]}..."
    return cleaned


def technical_detail_text(exc: BaseException) -> str:
    """Chaîne courte pour champs ``detail`` (mode développement uniquement)."""
    return _sanitize_detail_fragment(f"{type(exc).__name__}: {exc}")


def error_envelope_for_client(
    exc: BaseException,
    *,
    include_technical_detail: bool | None = None,
) -> dict[str, Any]:
    """Charge utile JSON unique ``{"error": {...}}`` (issue-006 / BFF).

    Args:
        exc: Exception source.
        include_technical_detail: Si ``None``, suit ``is_development_ui()``.
    """
    resolved = resolve_exception_for_api(exc, include_technical_detail=False)
    show_detail = _effective_dev_detail(include_technical_detail)
    detail: str | None = technical_detail_text(exc) if show_detail else None
    return {
        "error": {
            "code": resolved.code,
            "title": resolved.title_fr,
            "message": resolved.message_fr,
            "suggested_action": resolved.suggested_action_fr,
            "detail": detail,
        }
    }


def log_resolved_api_error(
    logger: logging.Logger,
    exc: BaseException,
    *,
    extra_context: dict[str, Any] | None = None,
) -> ResolvedApiError:
    """Journalise l'exception complète côté serveur et retourne l'enveloppe résolue."""
    resolved = resolve_exception_for_api(exc, include_technical_detail=False)
    payload: dict[str, Any] = {
        "api_error_code": resolved.code,
        "api_http_status": resolved.http_status,
    }
    if extra_context:
        for k, v in extra_context.items():
            payload[f"ctx_{k}"] = v
    logger.log(
        logging.ERROR,
        "api_error code=%s http=%s",
        resolved.code,
        resolved.http_status,
        exc_info=exc,
        extra=payload,
    )
    return resolved
