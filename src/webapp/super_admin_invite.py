"""Invitation collaborateur par super-admin (issue-017) — logique partagée webapp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy.engine import Engine

from src.auth import create_invitation_link
from src.empty_project_onboarding import invitation_account_link_email_intro_fr
from src.mailer import send_account_link_email

_INVITE_SUBJECT = "Invitation Dataset Style Studio"
# Comportement voulu (SuperTokens) : ``create_invitation_link`` accepte ``EMAIL_ALREADY_EXISTS_ERROR``
# puis renvoie un lien de reset ; l’UI ne distingue pas nouvelle invitation / compte existant.
_INVITE_EXISTING_ACCOUNT_UX_NOTE = (
    " Remarque support : si l'adresse avait déjà un compte, le fournisseur renvoie un lien de "
    "réinitialisation ; l'interface affiche le même message que pour une nouvelle invitation."
)


@dataclass(frozen=True, slots=True)
class SuperAdminInviteOutcome:
    """Résultat métier d'une invitation (hors enveloppe HTTP)."""

    message_fr: str
    mail_mode: Literal["dev", "smtp"]


def invite_collaborator_by_email(
    engine: Engine,
    actor_user_id: str,
    email: str,
) -> SuperAdminInviteOutcome:
    """Crée le lien provider et envoie l'e-mail (mode dev ou SMTP).

    Args:
        engine: Moteur SQLAlchemy.
        actor_user_id: Utilisateur connecté (super-admin requis via ``create_invitation_link``).
        email: Adresse déjà normalisée (trim + casse).

    Returns:
        Message utilisateur unique (FR) et mode d'envoi.

    Raises:
        PermissionError: Si l'acteur n'est pas super-admin.
        RuntimeError: Échec provider ou configuration (ex. ``APP_PUBLIC_BASE_URL``).

    Note:
        Compte e-mail déjà connu côté fournisseur : même chaîne utilisateur qu'une nouvelle
        invitation (pas de libellé distinct), cf. ``_INVITE_EXISTING_ACCOUNT_UX_NOTE``.
    """
    invite_link = create_invitation_link(engine, actor_user_id, email)
    delivery = send_account_link_email(
        to_email=email,
        subject=_INVITE_SUBJECT,
        intro=invitation_account_link_email_intro_fr(),
        link=invite_link,
    )
    if delivery.mode == "smtp":
        return SuperAdminInviteOutcome(
            message_fr="Invitation envoyée par e-mail." + _INVITE_EXISTING_ACCOUNT_UX_NOTE,
            mail_mode="smtp",
        )
    return SuperAdminInviteOutcome(
        message_fr=(
            "Mode développement : aucun e-mail réel n'est envoyé par ce serveur. "
            "Transmets au destinataire le lien d'activation en t'appuyant sur l'aperçu "
            f"masqué suivant (le jeton complet n'est pas affiché) : {delivery.preview}"
            + _INVITE_EXISTING_ACCOUNT_UX_NOTE
        ),
        mail_mode="dev",
    )
