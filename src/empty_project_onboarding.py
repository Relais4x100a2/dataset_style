"""Copy and flags for the empty-project onboarding flow (issue-008, issue-025, issue-028).

Streamlit widgets stay in ``ui_components``; this module holds testable
strings and environment-driven behaviour.

Issue-035 (invitation email): the super-admin invitation body is composed here
and sent via ``src/mailer.send_account_link_email`` from ``ui_components``.
SuperTokens Core email templates are not versioned in this repository; any
parallel lifecycle mail from Core must be kept consistent at deployment time
to avoid duplicate or contradictory wordings.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass

from src.tab_layout import EXPECTED_WORKFLOW_TAB_ORDER

DISABLE_SELF_SERVICE_PROJECT_CREATION_ENV = "DISABLE_SELF_SERVICE_PROJECT_CREATION"

# Widget key roots — must stay distinct between onboarding vs in-tab project actions.
ONBOARDING_MAIN_CREATE_FORM_KEY_PREFIX = "onboarding_main"
# Additional / follow-up project creation + delete guard in ``render_tab_projects`` (issue-028).
TAB_PROJECTS_ACTIONS_CREATE_FORM_KEY_PREFIX = "proj_tab"

NO_PROJECT_CREATION_DISABLED_MESSAGE = (
    "Tu n’as accès à aucun projet et la création de projet est désactivée sur cette instance. "
    "Contacte un administrateur pour recevoir une invitation à un projet existant."
)

# Submit label for the main onboarding form (issue-025 CTA wording).
ONBOARDING_PRIMARY_SUBMIT_LABEL_FR = "Créer un projet"

# One non-technical French line on stylometric value (issue-025).
STYLOMETRIC_VALUE_SENTENCE_FR = (
    "Tu peux harmoniser progressivement la voix de tes textes pour que ton dataset "
    "reste lisible et cohérent pour ceux qui l’exploitent."
)


def invitation_account_link_email_intro_fr() -> str:
    """Plain-text lead for super-admin invitation emails (issue-035).

    Reuses :data:`STYLOMETRIC_VALUE_SENTENCE_FR` so the recipient lands with the
    same stylometric value framing as the in-app onboarding.

    Returns:
        French paragraphs passed as ``intro`` to :func:`src.mailer.send_account_link_email`.
    """
    return (
        "Tu as reçu une invitation pour rejoindre Dataset Style Studio.\n\n"
        f"{STYLOMETRIC_VALUE_SENTENCE_FR}\n\n"
        "Clique sur le lien ci-dessous pour définir ton mot de passe."
    )


# Canonical wording: issue 8 / 11 (règle unique), refined issue-028 (contexte vs actions).
PRODUCT_RULE_ISSUE_11_CREATION_PATHS_FR = (
    "Règle produit (issues 8, 11 et 28) — **Contexte** : le menu **☰** (barre latérale) "
    "affiche ton compte et, dès qu’au moins un projet existe, permet de choisir le "
    "**projet courant**. **Actions projet** : l’onglet **Projets** regroupe la création "
    "(premier projet comme les suivants), la lecture du projet actif et la suppression ; "
    "il n’y a pas de formulaire de création dans la barre latérale pour éviter deux "
    "« endroits officiels » concurrents."
)

SIDEBAR_CONTEXT_HINT_FR = (
    "Astuce : le menu **☰** sert surtout au **contexte** (compte, projet courant). "
    "Pour créer ou retirer un projet, utilise l’onglet **Projets** dans la zone principale."
)


@dataclass(frozen=True)
class OnboardingStep:
    """One numbered step shown when self-service project creation is allowed."""

    index: int
    body_markdown: str


def is_self_service_project_creation_allowed() -> bool:
    """Return False when operators disable first-project creation via env."""
    raw = (os.environ.get(DISABLE_SELF_SERVICE_PROJECT_CREATION_ENV) or "").strip().lower()
    return raw not in ("1", "true", "yes", "on")


_EMPTY_DATASET_CURATOR_GUIDANCE_TEMPLATE_FR = (
    "**Projet sans fiches** — Ce projet ne contient encore aucune ligne à afficher ici. "
    "Dans le studio complet (onglets du bandeau), ouvre **{tab_new}** pour créer ta "
    "première fiche, puis **{tab_settings}** pour les dimensions et l’export CSV / JSONL."
)


def empty_dataset_curator_guidance_markdown_fr() -> str:
    """Markdown court quand le projet existe mais le jeu de fiches est vide (issue-015).

    Les intitulés d’onglets viennent de :data:`EXPECTED_WORKFLOW_TAB_ORDER` (alignement
    Streamlit / ``main_tab_labels``).

    Returns:
        Texte FR avec emphases ``**…**`` pour affichage ``st.markdown``.
    """
    return _EMPTY_DATASET_CURATOR_GUIDANCE_TEMPLATE_FR.format(
        tab_new=EXPECTED_WORKFLOW_TAB_ORDER[2],
        tab_settings=EXPECTED_WORKFLOW_TAB_ORDER[1],
    )


def empty_dataset_curator_guidance_html_fr() -> str:
    """Fragment HTML sûr pour la page slice ``webapp`` (issue-015).

    Returns:
        Paragraphe unique encodé pour injection dans :func:`src.webapp.app.create_slice_app`.
    """
    tab_new = html.escape(EXPECTED_WORKFLOW_TAB_ORDER[2])
    tab_settings = html.escape(EXPECTED_WORKFLOW_TAB_ORDER[1])
    return (
        '<p class="onboarding-empty" role="status"><strong>Projet sans fiches</strong> — '
        "Ce projet ne contient encore aucune ligne à afficher ici. "
        "Dans le studio complet (onglets du bandeau), ouvre "
        f"<strong>{tab_new}</strong> pour créer ta première fiche, puis "
        f"<strong>{tab_settings}</strong> pour les dimensions et l’export CSV / JSONL.</p>"
    )


def no_projects_slice_guidance_html_fr() -> str:
    """Message HTML lorsque l’utilisateur n’a aucun projet (slice ``webapp``, issue-015)."""
    if not is_self_service_project_creation_allowed():
        return f'<p class="err">{html.escape(NO_PROJECT_CREATION_DISABLED_MESSAGE)}</p>'
    return (
        "<p>Crée un premier projet depuis le studio Streamlit (onglet <strong>Projets</strong>), "
        "puis sélectionne-le ici après reconnexion si besoin.</p>"
    )


def onboarding_steps_when_creation_allowed() -> tuple[OnboardingStep, ...]:
    """Three guided steps: create project, dimensions tab, first entry tab (issue-025).

    Tab names are taken from ``EXPECTED_WORKFLOW_TAB_ORDER`` so onboarding stays aligned
    with ``main_tab_labels`` / ``st.tabs`` order after the first studio load.
    """
    tab_settings = EXPECTED_WORKFLOW_TAB_ORDER[1]
    tab_new_entry = EXPECTED_WORKFLOW_TAB_ORDER[2]
    cta = ONBOARDING_PRIMARY_SUBMIT_LABEL_FR
    return (
        OnboardingStep(
            1,
            f"**Étape 1 — Créer ton premier projet** — Tu es dans l’onglet **Projets** : "
            f"saisis un nom ci-dessous puis clique sur **{cta}**. "
            "Après chargement du studio, tu reviendras ici pour les autres actions "
            "(nouveau projet, suppression).",
        ),
        OnboardingStep(
            2,
            f"**Étape 2 — Ajuster les dimensions** — Après création, le studio s’ouvre : "
            f"va dans l’onglet **{tab_settings}** pour choisir un preset et tes listes "
            "de valeurs.",
        ),
        OnboardingStep(
            3,
            f"**Étape 3 — Première entrée** — Puis ouvre l’onglet **{tab_new_entry}** "
            "pour enregistrer ta première fiche du dataset.",
        ),
    )
