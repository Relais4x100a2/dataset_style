"""Copy and flags for the empty-project onboarding flow (issue-008, issue-025).

Streamlit widgets stay in ``ui_components``; this module holds testable
strings and environment-driven behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.tab_layout import EXPECTED_WORKFLOW_TAB_ORDER

DISABLE_SELF_SERVICE_PROJECT_CREATION_ENV = "DISABLE_SELF_SERVICE_PROJECT_CREATION"

# Widget key roots — must stay distinct between sidebar and main column forms.
ONBOARDING_MAIN_CREATE_FORM_KEY_PREFIX = "onboarding_main"
SIDEBAR_FIRST_PROJECT_CREATE_FORM_KEY_PREFIX = "sb_first"

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

# Same canonical wording as issue 11 / PR (sidebar vs onglet Projets).
PRODUCT_RULE_ISSUE_11_CREATION_PATHS_FR = (
    "Règle produit (issue 11) — Parcours principal pour le tout premier projet : "
    "ce formulaire plein écran. La barre latérale propose la même création et sert "
    "surtout à fixer le projet courant. L’onglet Projets du studio gère la liste des "
    "projets existants (créations supplémentaires, suppression) une fois le studio ouvert."
)

SIDEBAR_CONTEXT_HINT_FR = (
    "Le menu **☰** regroupe compte et projet courant ; tu n’as pas besoin de l’ouvrir "
    "pour créer depuis cette page."
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
            f"**Étape 1 — Créer ton premier projet** — Saisis un nom ci-dessous puis "
            f"clique sur **{cta}**. Tu peux aussi utiliser le même formulaire dans la "
            "barre latérale (menu **☰**, zone « Projet courant ») : ce n’est pas "
            "obligatoire pour démarrer ici.",
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
