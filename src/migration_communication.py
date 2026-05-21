"""
Bannière optionnelle pilotée par configuration pendant la migration de frontal.

GitHub #143 (issue-021) : message opérateur affiché dans Streamlit et dans la
coquille HTML du service ``webapp`` lorsque ``APP_MIGRATION_INFO_BANNER`` est
non vide. **GitHub #184** : le même nom de variable accepte désormais un objet
JSON (message + liens actionnables) ; le texte **seul** reste pris en charge
(rétrocompatibilité).

Après cutover, retirer ou vider la variable pour désactiver l'affichage.

Les fragments HTML sont construits avec échappement des textes et filtrage des
URL (``http`` / ``https`` / ``mailto`` uniquement). Pas de HTML riche depuis la
configuration.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from typing import Any

logger = logging.getLogger(__name__)

MIGRATION_INFO_BANNER_ENV = "APP_MIGRATION_INFO_BANNER"

# Marqueur dans ``src/webapp/index_template.py`` (servi par ``GET /``).
INDEX_HTML_BANNER_PLACEHOLDER = "<!--DS_MIGRATION_BANNER_PLACEHOLDER-->"

_MAX_MESSAGE_LEN = 800
_MAX_LABEL_LEN = 120
_MAX_CALENDAR_LEN = 280
_MAX_URL_LEN = 600

_RE_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True, slots=True)
class MigrationBannerConfig:
    """Contenu typé pour la bannière migration (mode JSON)."""

    message: str
    help_url: str | None = None
    help_label: str | None = None
    support_url: str | None = None
    support_label: str | None = None
    calendar_note: str | None = None


def _clip_text(value: Any, *, max_len: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    return s[:max_len]


def _safe_external_href(url: str) -> str | None:
    u = url.strip()
    if not u or len(u) > _MAX_URL_LEN:
        return None
    if _RE_CONTROL.search(u):
        return None
    lowered = u.lower()
    if lowered.startswith(("https://", "http://")):
        return u
    if lowered.startswith("mailto:"):
        return u
    return None


def parse_migration_banner_config(
    environ: Mapping[str, str] | None = None,
) -> MigrationBannerConfig | None:
    """Lit ``APP_MIGRATION_INFO_BANNER`` comme JSON objet (clé ``message`` obligatoire)."""
    env = environ if environ is not None else os.environ
    raw = (env.get(MIGRATION_INFO_BANNER_ENV) or "").strip()
    if not raw or not raw.startswith("{"):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("APP_MIGRATION_INFO_BANNER JSON invalide: %s", exc)
        return None
    if not isinstance(data, dict):
        logger.warning("APP_MIGRATION_INFO_BANNER: le JSON doit être un objet (clé « message »).")
        return None
    message = _clip_text(data.get("message"), max_len=_MAX_MESSAGE_LEN)
    if not message:
        logger.warning(
            "APP_MIGRATION_INFO_BANNER: clé « message » absente, vide ou uniquement des espaces."
        )
        return None

    help_raw = _clip_text(data.get("help_url"), max_len=_MAX_URL_LEN)
    help_url = _safe_external_href(help_raw) if help_raw else None
    support_raw = _clip_text(data.get("support_url"), max_len=_MAX_URL_LEN)
    support_url = _safe_external_href(support_raw) if support_raw else None

    return MigrationBannerConfig(
        message=message,
        help_url=help_url,
        help_label=_clip_text(data.get("help_label"), max_len=_MAX_LABEL_LEN),
        support_url=support_url,
        support_label=_clip_text(data.get("support_label"), max_len=_MAX_LABEL_LEN),
        calendar_note=_clip_text(data.get("calendar_note"), max_len=_MAX_CALENDAR_LEN),
    )


def migration_info_banner_text() -> str | None:
    """Retourne un libellé court pour diagnostics, ou ``None`` si désactivé.

    En mode JSON structuré, seul le champ ``message`` est retourné. En mode
    texte brut historique, la valeur entière (strippée) est retournée.
    """
    raw = (os.environ.get(MIGRATION_INFO_BANNER_ENV) or "").strip()
    if not raw:
        return None
    cfg = parse_migration_banner_config()
    if cfg is not None:
        return cfg.message
    if raw.startswith("{"):
        return None
    return raw


def _region_aria_label_attr() -> str:
    return "aria-label=\"Informations sur le changement d'interface ou d'URL\""


def migration_banner_html_section(cfg: MigrationBannerConfig) -> str:
    """Retourne le HTML du bandeau structuré (tests et extensions éventuelles)."""
    return _structured_banner_html(cfg)


def _structured_banner_html(cfg: MigrationBannerConfig) -> str:
    """Bandeau info avec liens ; classes alignées sur ``design_tokens.css`` (issue-022)."""
    parts: list[str] = [
        '<section class="ds-banner ds-banner--info ds-migration-banner" role="region" '
        f"{_region_aria_label_attr()}>",
        f'<p class="ds-banner__message">{escape(cfg.message)}</p>',
    ]
    if cfg.calendar_note:
        parts.append(
            f'<p class="ds-banner__message ds-migration-banner__calendar">'
            f"{escape(cfg.calendar_note)}</p>"
        )
    links: list[str] = []
    if cfg.help_url:
        label = escape(cfg.help_label or "Où trouver l'aide")
        href = escape(cfg.help_url, quote=True)
        links.append(
            f'<a class="ds-migration-banner__link" href="{href}" '
            f'rel="noopener noreferrer" target="_blank">{label}</a>'
        )
    if cfg.support_url:
        label = escape(cfg.support_label or "Contacter le support")
        href = escape(cfg.support_url, quote=True)
        target = ' target="_blank"' if cfg.support_url.lower().startswith("http") else ""
        links.append(
            f'<a class="ds-migration-banner__link" href="{href}" rel="noopener noreferrer"{target}>'
            f"{label}</a>"
        )
    if links:
        joined = ' <span aria-hidden="true">·</span> '.join(links)
        parts.append(f'<p class="ds-banner__message ds-migration-banner__links">{joined}</p>')
    parts.append("</section>")
    return "".join(parts)


def _legacy_plain_banner_html(text: str) -> str:
    return (
        '<section class="ds-banner ds-banner--info ds-migration-banner" role="region" '
        f"{_region_aria_label_attr()}>"
        f'<p class="ds-banner__message">{escape(text)}</p></section>'
    )


def migration_info_banner_html_fragment() -> str:
    """
    Fragment HTML sûr à injecter dans la page d'accueil ``webapp``.

    Returns:
        Chaîne vide si la bannière est désactivée ; sinon bandeau ``ds-migration-banner``.
    """
    raw = (os.environ.get(MIGRATION_INFO_BANNER_ENV) or "").strip()
    if not raw:
        return ""
    cfg = parse_migration_banner_config()
    if cfg is not None:
        return _structured_banner_html(cfg)
    if raw.startswith("{"):
        return ""
    return _legacy_plain_banner_html(raw)


def render_streamlit_migration_banner_if_configured() -> None:
    """Affiche la bannière (persistante, non bloquante) sous les flashes post-rerun."""
    import streamlit as st

    raw = (os.environ.get(MIGRATION_INFO_BANNER_ENV) or "").strip()
    if not raw:
        return
    cfg = parse_migration_banner_config()
    if cfg is not None:
        st.markdown(_structured_banner_html(cfg), unsafe_allow_html=True)
        return
    if raw.startswith("{"):
        return
    st.markdown(_legacy_plain_banner_html(raw), unsafe_allow_html=True)
