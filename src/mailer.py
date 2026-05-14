"""Services d'envoi d'emails (mode dev/smtp)."""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage


@dataclass
class MailDeliveryResult:
    """Résultat d'un envoi email."""

    mode: str
    delivered: bool
    preview: str


def _mail_mode() -> str:
    mode = (os.environ.get("MAIL_MODE") or "dev").strip().lower()
    return mode if mode in {"dev", "smtp"} else "dev"


def _mask_link(link: str) -> str:
    if "token=" not in link:
        return link
    prefix, token = link.split("token=", 1)
    clean = token.strip()
    if len(clean) < 12:
        return f"{prefix}token=***"
    return f"{prefix}token={clean[:6]}...{clean[-6:]}"


def _send_smtp_email(*, to_email: str, subject: str, body: str) -> None:
    host = (os.environ.get("SMTP_HOST") or "").strip()
    port = int((os.environ.get("SMTP_PORT") or "587").strip())
    username = (os.environ.get("SMTP_USER") or "").strip()
    password = (os.environ.get("SMTP_PASSWORD") or "").strip()
    from_email = (os.environ.get("SMTP_FROM_EMAIL") or username).strip()
    if not all([host, username, password, from_email]):
        raise RuntimeError("Configuration SMTP incomplète.")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = to_email
    message.set_content(body)

    with smtplib.SMTP(host, port, timeout=20) as client:
        client.starttls()
        client.login(username, password)
        client.send_message(message)


def send_account_link_email(
    *,
    to_email: str,
    subject: str,
    intro: str,
    link: str,
) -> MailDeliveryResult:
    """Envoie un lien d'activation/reset en mode dev ou smtp.

    Le corps est du texte brut (``EmailMessage.set_content``). Les invitations
    super-admin construisent ``intro`` côté application (issue-035) pour rester
    alignées sur l’onboarding ; elles ne passent pas par les templates SuperTokens
    versionnés dans ce dépôt.
    """
    mode = _mail_mode()
    body = f"{intro}\n\n{link}\n"
    if mode == "smtp":
        _send_smtp_email(to_email=to_email, subject=subject, body=body)
        return MailDeliveryResult(mode=mode, delivered=True, preview="Email envoyé via SMTP.")
    return MailDeliveryResult(mode=mode, delivered=True, preview=_mask_link(link))
