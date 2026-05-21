"""Mapping bandeaux sémantiques (issue-022 / #144) — codes API et alertes qualité."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from src.api_errors import (
    AUTH_SESSION_EXPIRED,
    DB_UNAVAILABLE,
    EXPORT_PAYLOAD_TOO_LARGE,
    FORBIDDEN,
    INTERNAL_ERROR,
    NOT_FOUND_GENERIC,
)
from src.webapp.app import create_slice_app
from src.webapp.ui_semantics import (
    banner_variant_for_api_error_code,
    banner_variant_for_dataset_quality_severity,
)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (AUTH_SESSION_EXPIRED, "warning"),
        (DB_UNAVAILABLE, "danger"),
        (FORBIDDEN, "warning"),
        (NOT_FOUND_GENERIC, "info"),
        (EXPORT_PAYLOAD_TOO_LARGE, "warning"),
        (INTERNAL_ERROR, "danger"),
        ("MAIL_DELIVERY_FAILED", "danger"),
        ("BAD_REQUEST", "warning"),
        ("CURATOR_LANGUAGETOOL_UNAVAILABLE", "danger"),
        ("CLIENT", "warning"),
        ("UNKNOWN_CODE_XYZ", "danger"),
    ],
)
def test_banner_variant_for_api_error_code_explicit_mapping(code: str, expected: str) -> None:
    """Chaque code connu du contrat API mappe vers un variant bandeau (hors statut HTTP)."""
    assert banner_variant_for_api_error_code(code) == expected


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        ("warning", "warning"),
        ("info", "info"),
    ],
)
def test_dataset_quality_severity_maps_to_banner_variant(severity: str, expected: str) -> None:
    """Les alertes ``dataset_quality`` du dashboard réutilisent les mêmes noms de variant."""
    assert banner_variant_for_dataset_quality_severity(severity) == expected


def test_dataset_quality_unknown_severity_falls_back_to_warning() -> None:
    """Valeur inattendue : défaut prudent côté qualité dataset (visible, non bloquant)."""
    assert banner_variant_for_dataset_quality_severity("weird") == "warning"


def test_index_html_links_tokens_and_injects_api_banner_map() -> None:
    """Page d'accueil : lien CSS tokens + injection du mapping ``error.code``."""
    from src.webapp import index_template

    html = index_template.INDEX_HTML
    assert "/static/design_tokens.css" in html
    assert '"AUTH_SESSION_EXPIRED":"warning"' in html
    assert '"MAIL_DELIVERY_FAILED":"danger"' in html
    assert "const API_ERROR_BANNER_VARIANT = {" in html


def test_static_design_tokens_css_served() -> None:
    """Montage FastAPI ``/static`` : feuille de style des tokens (issue-022)."""
    app = create_slice_app(engine=MagicMock())
    with TestClient(app) as client:
        r = client.get("/static/design_tokens.css")
    assert r.status_code == 200
    assert b"ds-banner--warning" in r.content
    assert b"system-ui" in r.content
