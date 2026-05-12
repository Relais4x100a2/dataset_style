"""French UI strings for the Super Admin tab (issue-012).

Separated from Streamlit widgets for straightforward unit testing.
"""

from __future__ import annotations

SAGA_STATE_METRIC_LABELS: dict[str, str] = {
    "pending": "En attente",
    "provider_done": "Étape fournisseur terminée",
    "failed": "Échecs",
    "quarantined": "En quarantaine",
}

SUPER_ADMIN_INVITE_SECTION_TITLE = "Inviter un collaborateur"
SUPER_ADMIN_ACCOUNTS_SECTION_TITLE = "Comptes de la plateforme"
SUPER_ADMIN_ACTIONS_SECTION_TITLE = "Actions sur le compte sélectionné"

SUPER_ADMIN_TECH_EXPANDER_TITLE = "Suivi technique — suppressions de compte et files d'attente"
SUPER_ADMIN_TECH_EXPANDER_CAPTION = (
    "Réservé aux interventions avancées : les identifiants d'opération et les "
    "messages d'erreur détaillés figurent dans les tableaux ci-dessous."
)

SUPER_ADMIN_SAGA_SECTION_TITLE = "État du traitement automatique"
SUPER_ADMIN_DLQ_SECTION_TITLE = "Comptes bloqués — traitement manuel"


def super_admin_tab_labels() -> tuple[str, str]:
    """Labels for the Super Admin inner tab strip (accounts first, technical second).

    Returns:
        A pair ``(primary tab, technical tab)``. Both bodies still run each script
        execution (Streamlit); this only affects default visibility in the UI.
    """
    return (
        "Invitations et gestion des comptes",
        "Suivi technique — saga, blocages et relances",
    )


def selectbox_target_account() -> str:
    """Label for choosing which account to act on (business wording)."""
    return "Compte à gérer"


def selectbox_dlq_operation() -> str:
    """Label for DLQ replay selectbox; technical id stays in options, not the title."""
    return "Opération à relancer (référence dans le tableau)"


def button_detach_memberships() -> str:
    """Primary action: remove user from all shared projects."""
    return "Retirer l'utilisateur de tous les projets partagés"


def button_replay_quarantined() -> str:
    """Replay a quarantined deprovision operation."""
    return "Relancer le traitement bloqué"


def saga_metric_label(state: str) -> str:
    """French label for a saga state metric, with safe fallback for unknown states."""
    return SAGA_STATE_METRIC_LABELS.get(state, state)
