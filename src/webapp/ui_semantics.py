"""Variants de bandeau pour le slice webapp (issue-022 / #144).

Le frontal mappe ``error.code`` (contrat issue-005) vers un variant visuel ;
on n'utilise pas le statut HTTP comme source de vérité pour la sémantique UI.
"""

from __future__ import annotations

import json
from typing import Final, Literal

from src.api_errors import (
    AUTH_SESSION_EXPIRED,
    DB_UNAVAILABLE,
    EXPORT_PAYLOAD_TOO_LARGE,
    FORBIDDEN,
    INTERNAL_ERROR,
    NOT_FOUND_GENERIC,
)

BannerVariant = Literal["success", "warning", "danger", "info"]

# Codes additionnels renvoyés par le BFF hors ``_CATALOG`` principal.
_BAD_REQUEST: Final = "BAD_REQUEST"
_CURATOR_LT_UNAVAILABLE: Final = "CURATOR_LANGUAGETOOL_UNAVAILABLE"
_CLIENT: Final = "CLIENT"

_API_ERROR_CODE_TO_BANNER_VARIANT: dict[str, BannerVariant] = {
    # Session : action utilisateur (reconnexion), pas panne infrastructure.
    AUTH_SESSION_EXPIRED: "warning",
    # Indisponibilité données / technique.
    DB_UNAVAILABLE: "danger",
    # Droits insuffisants : guidage, pas erreur serveur générique.
    FORBIDDEN: "warning",
    # Anti-IDOR / sélection : message neutre.
    NOT_FOUND_GENERIC: "info",
    # Limite métier export : réduction de périmètre.
    EXPORT_PAYLOAD_TOO_LARGE: "warning",
    INTERNAL_ERROR: "danger",
    _BAD_REQUEST: "warning",
    _CURATOR_LT_UNAVAILABLE: "danger",
    _CLIENT: "warning",
}


def banner_variant_for_api_error_code(code: str | None) -> BannerVariant:
    """Retourne le variant bandeau pour un ``error.code`` API (issue-005).

    Args:
        code: Code stable de l'enveloppe ``{"error": {"code": ...}}``.

    Returns:
        Un des quatre variants sémantiques ; défaut ``danger`` si inconnu.
    """
    if not code:
        return "danger"
    return _API_ERROR_CODE_TO_BANNER_VARIANT.get(code, "danger")


def banner_variant_for_dataset_quality_severity(severity: str | None) -> BannerVariant:
    """Mappe ``severity`` des alertes ``dataset_quality.alerts`` (issue-014).

    Args:
        severity: ``warning`` ou ``info`` dans l'enveloppe dashboard.

    Returns:
        Variant aligné sur les classes ``ds-banner--*``.
    """
    if severity == "info":
        return "info"
    if severity == "warning":
        return "warning"
    return "warning"


def api_error_banner_variant_json_for_index_script() -> str:
    """Fragment JSON (objet JS) injecté dans ``index_template`` pour le client."""
    return json.dumps(_API_ERROR_CODE_TO_BANNER_VARIANT, ensure_ascii=False, separators=(",", ":"))
