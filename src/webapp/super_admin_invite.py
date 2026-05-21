"""Invitation collaborateur par super-admin (issue-007 / issue-017) — logique partagée webapp."""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.engine import Engine

from src.api_errors import MailDeliveryFailedError
from src.auth import create_invitation_link
from src.empty_project_onboarding import invitation_account_link_email_intro_fr
from src.mailer import send_account_link_email

logger = logging.getLogger(__name__)

_INVITE_SUBJECT = "Invitation Dataset Style Studio"

InviteResultCode = Literal["new_invitation", "existing_account_reset"]


@dataclass(frozen=True, slots=True)
class SuperAdminInviteOutcome:
    """Résultat métier d'une invitation (hors enveloppe HTTP)."""

    message_fr: str
    mail_mode: Literal["dev", "smtp"]
    invite_result: InviteResultCode


def invite_collaborator_by_email(
    engine: Engine,
    actor_user_id: str,
    email: str,
) -> SuperAdminInviteOutcome:
    """Crée le lien provider et envoie l'e-mail (mode dev ou SMTP).

    Réutilise :func:`src.auth.create_invitation_link` puis
    :func:`src.mailer.send_account_link_email` (même chaîne que Streamlit).

    Args:
        engine: Moteur SQLAlchemy.
        actor_user_id: Utilisateur connecté (super-admin requis via ``create_invitation_link``).
        email: Adresse déjà normalisée (trim + casse).

    Returns:
        Message utilisateur (FR), mode d'envoi et code de résultat (parité « e-mail déjà connu »).

    Raises:
        PermissionError: Si l'acteur n'est pas super-admin.
        RuntimeError: Échec provider ou configuration (ex. ``APP_PUBLIC_BASE_URL``).
        MailDeliveryFailedError: Échec SMTP ou transport réseau lors de l'envoi.
    """
    invite = create_invitation_link(engine, actor_user_id, email)
    invite_code: InviteResultCode = (
        "existing_account_reset" if invite.email_already_registered else "new_invitation"
    )
    try:
        delivery = send_account_link_email(
            to_email=email,
            subject=_INVITE_SUBJECT,
            intro=invitation_account_link_email_intro_fr(),
            link=invite.link,
        )
    except (smtplib.SMTPException, TimeoutError, ConnectionError, BrokenPipeError) as exc:
        logger.warning(
            "super_admin_invite: mail transport failed actor=%s to=%s",
            actor_user_id,
            email,
            exc_info=True,
        )
        raise MailDeliveryFailedError(exc) from exc
    if delivery.mode == "dev":
        logger.info(
            "super_admin_invite_dev: actor=%s to=%s invite_result=%s masked_preview=%s",
            actor_user_id,
            email,
            invite_code,
            delivery.preview,
        )
    if delivery.mode == "smtp":
        return SuperAdminInviteOutcome(
            message_fr="Invitation envoyée par e-mail.",
            mail_mode="smtp",
            invite_result=invite_code,
        )
    return SuperAdminInviteOutcome(
        message_fr=(
            "Mode développement : aucun e-mail réel n'est envoyé par ce serveur. "
            "Transmets au destinataire le lien d'activation en t'appuyant sur l'aperçu "
            f"masqué suivant (le jeton complet n'est pas affiché) : {delivery.preview}"
        ),
        mail_mode="dev",
        invite_result=invite_code,
    )
