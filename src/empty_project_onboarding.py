"""Copy and flags for the empty-project onboarding flow (issue-008).

Streamlit widgets stay in ``ui_components``; this module holds testable
strings and environment-driven behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DISABLE_SELF_SERVICE_PROJECT_CREATION_ENV = "DISABLE_SELF_SERVICE_PROJECT_CREATION"

# Widget key roots — must stay distinct between sidebar and main column forms.
ONBOARDING_MAIN_CREATE_FORM_KEY_PREFIX = "onboarding_main"
SIDEBAR_FIRST_PROJECT_CREATE_FORM_KEY_PREFIX = "sb_first"

NO_PROJECT_CREATION_DISABLED_MESSAGE = (
    "Tu n’as accès à aucun projet et la création de projet est désactivée sur cette instance. "
    "Contacte un administrateur pour recevoir une invitation à un projet existant."
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
    """Three guided steps: sidebar, form, submit (auto rerun loads the studio)."""
    return (
        OnboardingStep(
            1,
            "**Étape 1** — Ouvre le menu **☰** en haut à gauche pour afficher la barre "
            "latérale (compte et zone « Projet courant »).",
        ),
        OnboardingStep(
            2,
            "**Étape 2** — Saisis le nom de ton projet dans le formulaire ci-dessous "
            "(tu peux aussi utiliser le même type de formulaire dans la barre latérale "
            "une fois ouverte).",
        ),
        OnboardingStep(
            3,
            "**Étape 3** — Clique sur **Créer**. Les onglets du studio (projets, "
            "dataset, export, etc.) s’affichent tout de suite après rechargement.",
        ),
    )
