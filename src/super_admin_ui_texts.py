"""French UI strings for the Super Admin tab (issue-012, issue-029).

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

# Issue-029: umbrella section for invitations + day-to-day account operations.
SUPER_ADMIN_ACCOUNT_MANAGEMENT_HUB_TITLE = "Gestion des comptes"

SUPER_ADMIN_WORKFLOW_HINT = (
    "Commencez par les invitations et les actions courantes sur les comptes. "
    "Le suivi technique (traitements automatiques, blocages, relances) se trouve "
    "dans l’onglet suivant, séparé pour limiter le vocabulaire infrastructure."
)

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
        "Gestion des comptes — invitations et actions courantes",
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


def super_admin_accounts_table_column_labels() -> dict[str, str]:
    """French column titles for the super-admin accounts table (display only).

    Returns:
        Mapping from internal dataframe keys to user-visible headers.
    """
    return {
        "user_id": "Identifiant (technique)",
        "nom_affichage": "Nom affiché",
        "email": "Courriel",
        "super_admin": "Super administrateur",
        "nb_projets": "Projets actifs",
        "derniere_connexion": "Dernière connexion",
        "entrees_total": "Entrées (total)",
        "entrees_validees": "Entrées validées",
    }


def super_admin_warning_detach_memberships(*, membership_count: int, email: str) -> str:
    """Warning copy before detaching all shared-project access for a user."""
    return (
        f"Action destructive : retirer {membership_count} accès collaborateur(s) "
        f"associés au compte {email}."
    )


def flash_memberships_detached(count: int) -> str:
    """Success flash after shared memberships were removed."""
    return (
        f"Accès aux projets partagés retirés pour ce compte "
        f"({count} collaboration(s) supprimée(s))."
    )


def error_detach_memberships_failed() -> str:
    """User-facing prefix when detaching memberships fails."""
    return "Retrait des accès projets partagés impossible"


def error_delete_target_account_failed() -> str:
    """User-facing prefix when deleting another user's account fails."""
    return "Suppression du compte impossible"
