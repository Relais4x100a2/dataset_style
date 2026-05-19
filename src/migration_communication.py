"""
Bannière optionnelle pilotée par configuration pendant la migration de frontal.

Issue interne 021 / GitHub #143 : message opérateur (texte brut) affiché dans
Streamlit et dans la coquille HTML du service ``webapp`` lorsque la variable
d'environnement ``APP_MIGRATION_INFO_BANNER`` est non vide. Après cutover,
retirer ou vider la variable pour désactiver l'affichage.

Le fragment HTML est construit à partir du texte **échappé** (pas de HTML
riche) pour éviter l'injection de balises depuis la configuration.
"""

from __future__ import annotations

import os
from html import escape

MIGRATION_INFO_BANNER_ENV = "APP_MIGRATION_INFO_BANNER"

# Marqueur dans ``src/webapp/index_template.py`` (servi par ``GET /``).
INDEX_HTML_BANNER_PLACEHOLDER = "<!--DS_MIGRATION_BANNER_PLACEHOLDER-->"


def migration_info_banner_text() -> str | None:
    """Retourne le texte brut de la bannière, ou ``None`` si désactivée."""
    raw = (os.environ.get(MIGRATION_INFO_BANNER_ENV) or "").strip()
    return raw or None


def migration_info_banner_html_fragment() -> str:
    """
    Fragment HTML sûr à injecter dans la page d'accueil ``webapp``.

    Returns:
        Chaîne vide si la bannière est désactivée ; sinon un bandeau
        ``ds-banner--info`` (issue-022) au contenu entièrement échappé.
    """
    text = migration_info_banner_text()
    if not text:
        return ""
    return (
        f'<div class="ds-banner ds-banner--info ds-migration-banner" role="status">'
        f'<p class="ds-banner__message">{escape(text)}</p></div>'
    )
