"""Authentification applicative via SuperTokens (API HTTP)."""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass

import requests
import streamlit as st
from sqlalchemy.engine import Engine

from src import mailer
from src.database import (
    UserRecord,
    create_deprovision_operation,
    delete_user_if_detached,
    detach_memberships_as_super_admin,
    get_deprovision_operation,
    get_su_user_id_by_user_id,
    grant_super_admin_by_email,
    is_user_super_admin,
    mark_user_disabled,
    mark_user_login,
    record_deprovision_failure,
    require_super_admin,
    transition_deprovision_operation,
    upsert_user_from_su,
)

logger = logging.getLogger(__name__)


@dataclass
class CurrentUser:
    """Utilisateur courant authentifié."""

    user_id: str
    email: str
    display_name: str
    access_token: str
    is_super_admin: bool = False


def _su_base_url() -> str:
    return (os.environ.get("SUPERTOKENS_CONNECTION_URI") or "").strip().rstrip("/")


def _su_header() -> dict[str, str]:
    api_key = (os.environ.get("SUPERTOKENS_API_KEY") or "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    return headers


def _post(path: str, payload: dict) -> dict:
    base = _su_base_url()
    if not base:
        raise RuntimeError("SUPERTOKENS_CONNECTION_URI manquant.")
    resp = requests.post(f"{base}{path}", json=payload, headers=_su_header(), timeout=20)
    if resp.status_code >= 400:
        body = (resp.text or "").strip()
        raise RuntimeError(f"SuperTokens {path} HTTP {resp.status_code}: {body}")
    return resp.json()


def _signin(email: str, password: str) -> dict:
    try:
        return _post(
            "/recipe/signin",
            {
                "formFields": [
                    {"id": "email", "value": email},
                    {"id": "password", "value": password},
                ]
            },
        )
    except RuntimeError as exc:
        # Compatibilité entre versions SuperTokens Core (formFields vs email/password).
        if "Field name 'email' is invalid in JSON input" not in str(exc):
            raise
        return _post("/recipe/signin", {"email": email, "password": password})


def _signup(email: str, password: str) -> dict:
    try:
        return _post(
            "/recipe/signup",
            {
                "formFields": [
                    {"id": "email", "value": email},
                    {"id": "password", "value": password},
                ]
            },
        )
    except RuntimeError as exc:
        # Compatibilité entre versions SuperTokens Core (formFields vs email/password).
        if "Field name 'email' is invalid in JSON input" not in str(exc):
            raise
        return _post("/recipe/signup", {"email": email, "password": password})


def _extract_email_verified(user: object) -> bool:
    """Retourne l'état de vérification email depuis le payload provider."""
    if not isinstance(user, dict):
        return False
    direct = user.get("isEmailVerified")
    if isinstance(direct, bool):
        return direct
    emails = user.get("emails")
    if isinstance(emails, list) and emails:
        first = emails[0]
        if isinstance(first, dict):
            first_verified = first.get("isVerified")
            if isinstance(first_verified, bool):
                return first_verified
    return False


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _super_admin_email_set() -> set[str]:
    raw = (os.environ.get("SUPER_ADMIN_EMAILS") or "").strip()
    if not raw:
        return set()
    return {_normalize_email(item) for item in raw.split(",") if _normalize_email(item)}


def _create_password_reset_link(email: str) -> str:
    payload = {"email": _normalize_email(email)}
    out = _post("/recipe/user/password/reset/token", payload)
    token = str(out.get("token") or "").strip()
    if not token:
        raise RuntimeError("Token de réinitialisation absent.")
    app_base = (os.environ.get("APP_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if not app_base:
        raise RuntimeError("APP_PUBLIC_BASE_URL manquant pour générer le lien.")
    return f"{app_base}/?flow=set-password&token={token}"


def _provider_revoke_all_sessions(su_user_id: str) -> None:
    """Révoque les sessions provider selon les variantes d'API connues."""
    candidates = [
        ("/recipe/session/removeUserSessions", {"userId": su_user_id}),
        ("/recipe/session/remove-user-sessions", {"userId": su_user_id}),
    ]
    last_error: Exception | None = None
    for path, payload in candidates:
        try:
            _post(path, payload)
            return
        except Exception as exc:  # noqa: BLE001
            if "unknown user" in str(exc).lower() or "user not found" in str(exc).lower():
                return
            last_error = exc
    if last_error:
        raise RuntimeError(f"Révocation sessions provider impossible: {last_error}") from last_error


def _provider_delete_user(su_user_id: str) -> None:
    """Supprime un compte provider selon les variantes d'API connues."""
    candidates = [
        ("/recipe/user/remove", {"userId": su_user_id}),
        ("/recipe/user/remove", {"user_id": su_user_id}),
    ]
    last_error: Exception | None = None
    for path, payload in candidates:
        try:
            _post(path, payload)
            return
        except Exception as exc:  # noqa: BLE001
            if "unknown user" in str(exc).lower() or "user not found" in str(exc).lower():
                return
            last_error = exc
    if last_error:
        raise RuntimeError(
            f"Suppression utilisateur provider impossible: {last_error}"
        ) from last_error


def _mask_link(link: str) -> str:
    """Masque un lien pour l'affichage sans fuite complète du token."""
    if "token=" not in link:
        return link
    prefix, token = link.split("token=", 1)
    token_clean = token.strip()
    if len(token_clean) <= 10:
        return f"{prefix}token=***"
    return f"{prefix}token={token_clean[:5]}...{token_clean[-5:]}"


def create_invitation_link(engine: Engine, actor_user_id: str, email: str) -> str:
    """Crée ou recycle un compte provider puis génère un lien de définition de mot de passe."""
    require_super_admin(engine, actor_user_id)
    normalized_email = _normalize_email(email)
    temporary_password = f"Aa1!{secrets.token_urlsafe(24)}"
    out = _signup(normalized_email, temporary_password)
    status = str(out.get("status") or "").strip()
    if status not in {"OK", "EMAIL_ALREADY_EXISTS_ERROR"}:
        raise RuntimeError(f"Invitation impossible: {status}")
    return _create_password_reset_link(normalized_email)


def request_password_reset_link(email: str) -> str:
    """Génère un lien de reset mot de passe."""
    return _create_password_reset_link(_normalize_email(email))


def _probe_signup_disabled() -> None:
    """Vérifie empiriquement que le Core SuperTokens refuse le signup public (fail-secure)."""
    probe_email = f"probe.noreply.{secrets.token_hex(8)}@probe.invalid"
    probe_password = f"Probe!{secrets.token_urlsafe(16)}"
    try:
        out = _signup(probe_email, probe_password)
    except RuntimeError:
        # signup bloqué au niveau HTTP — c'est ce qu'on veut
        return
    status = str(out.get("status") or "").strip()
    if status in {"OK", "EMAIL_ALREADY_EXISTS_ERROR"}:
        raise RuntimeError(
            "SuperTokens Core accepte encore le signup public. "
            "Désactiver côté Core avant déploiement."
        )
    # Tout autre status (GENERAL_ERROR, etc.) → signup bloqué
    return


def ensure_invitation_only_policy() -> None:
    """
    Vérifie contractuellement que le signup provider est bloqué.

    Vérification non destructive activée seulement si AUTH_ENFORCE_INVITATION_ONLY=true.
    """
    enforce = (os.environ.get("AUTH_ENFORCE_INVITATION_ONLY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enforce:
        return
    if st.session_state.get("auth_invitation_policy_checked"):
        return
    signup_disabled = (os.environ.get("SUPERTOKENS_SIGNUP_DISABLED") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not signup_disabled:
        raise RuntimeError(
            "AUTH_ENFORCE_INVITATION_ONLY=true exige SUPERTOKENS_SIGNUP_DISABLED=true."
        )
    _probe_signup_disabled()
    st.session_state["auth_invitation_policy_checked"] = True


def _state_key() -> str:
    return "current_user"


def get_current_user() -> CurrentUser | None:
    raw = st.session_state.get(_state_key())
    if not raw:
        return None
    return CurrentUser(**raw)


def logout() -> None:
    st.session_state.pop(_state_key(), None)


def _set_user(record: UserRecord, access_token: str) -> None:
    st.session_state[_state_key()] = {
        "user_id": record.user_id,
        "email": record.email,
        "display_name": record.display_name,
        "access_token": access_token,
        "is_super_admin": bool(record.is_super_admin),
    }


def _extract_user_id(user: object) -> str:
    if not isinstance(user, dict):
        return ""
    return str(user.get("id") or user.get("userId") or "").strip()


def _extract_email(user: object, fallback_email: str) -> str:
    fallback = fallback_email.strip()
    if not isinstance(user, dict):
        return fallback

    direct_email = user.get("email")
    if isinstance(direct_email, str) and direct_email.strip():
        return direct_email.strip()

    emails = user.get("emails")
    if isinstance(emails, list) and emails:
        first = emails[0]
        if isinstance(first, dict):
            value = first.get("email")
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(first, str) and first.strip():
            return first.strip()
    return fallback


def _maybe_promote_super_admin(engine: Engine, email: str, email_verified: bool) -> None:
    """Promeut un utilisateur au premier login selon policy stricte."""
    if not email_verified:
        return
    normalized_email = _normalize_email(email)
    if normalized_email not in _super_admin_email_set():
        return
    grant_super_admin_by_email(engine, normalized_email)


def revoke_account_with_saga(
    engine: Engine,
    *,
    actor_user_id: str,
    target_user_id: str,
    operation_id: str,
    max_retries: int,
    detach_memberships: bool = False,
) -> str:
    """
    Exécute une saga idempotente de révocation/suppression.

    Returns:
        État final de l'opération (`completed` ou `failed`).
    """
    if max_retries < 1:
        raise ValueError("max_retries doit être >= 1.")
    if actor_user_id != target_user_id:
        require_super_admin(engine, actor_user_id)
    op = create_deprovision_operation(
        engine,
        operation_id=operation_id,
        actor_user_id=actor_user_id,
        target_user_id=target_user_id,
    )
    if op.state == "completed":
        return "completed"
    if op.state == "quarantined":
        raise RuntimeError("Opération en quarantaine (max_retries atteint).")
    if op.retry_count >= max_retries:
        raise RuntimeError("Opération bloquée: max_retries atteint. Replay admin requis.")
    mark_user_disabled(engine, target_user_id)
    current = get_deprovision_operation(engine, operation_id)
    if current is None:
        raise RuntimeError("Saga introuvable.")
    if current.state == "failed":
        transition_deprovision_operation(
            engine,
            operation_id=operation_id,
            expected_state="failed",
            next_state="pending",
            error_message="",
        )
        current = get_deprovision_operation(engine, operation_id)
        if current is None:
            raise RuntimeError("Saga introuvable après reprise.")
    try:
        if current.state == "pending":
            su_user_id = get_su_user_id_by_user_id(engine, target_user_id)
            _provider_revoke_all_sessions(su_user_id)
            _provider_delete_user(su_user_id)
            transition_deprovision_operation(
                engine,
                operation_id=operation_id,
                expected_state="pending",
                next_state="provider_done",
            )
        current = get_deprovision_operation(engine, operation_id)
        if current is None:
            raise RuntimeError("Saga introuvable après étape provider.")
        if current.state == "provider_done":
            if detach_memberships:
                detach_memberships_as_super_admin(engine, actor_user_id, target_user_id)
            delete_user_if_detached(engine, target_user_id)
            transition_deprovision_operation(
                engine,
                operation_id=operation_id,
                expected_state="provider_done",
                next_state="db_done",
            )
            transition_deprovision_operation(
                engine,
                operation_id=operation_id,
                expected_state="db_done",
                next_state="completed",
            )
        final_state = get_deprovision_operation(engine, operation_id)
        return "completed" if final_state and final_state.state == "completed" else "failed"
    except Exception as exc:  # noqa: BLE001
        current_failed = get_deprovision_operation(engine, operation_id)
        if current_failed is not None and current_failed.state in {
            "pending",
            "provider_done",
            "db_done",
        }:
            try:
                next_op = record_deprovision_failure(
                    engine,
                    operation_id=operation_id,
                    expected_state=current_failed.state,
                    error_message=str(exc),
                    max_retries=max_retries,
                    backoff_seconds=min(3600, 60 * (2 ** min(current_failed.retry_count, 6))),
                )
                if next_op.state == "quarantined":
                    logger.error("Saga placée en quarantaine operation_id=%s", operation_id)
            except Exception:  # noqa: BLE001
                logger.exception("Transition vers failed impossible")
        raise


def render_auth_gate(engine: Engine) -> CurrentUser | None:
    """
    Affiche login et retourne l'utilisateur courant si authentifié.
    """
    current = get_current_user()
    if current:
        return current

    try:
        ensure_invitation_only_policy()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Configuration auth invalide: {exc}")
        return None

    st.title("Connexion")
    st.caption("Authentification par email + mot de passe.")

    with st.form("auth_form"):
        email = st.text_input("Email")
        password = st.text_input("Mot de passe", type="password")
        signin_btn = st.form_submit_button("Se connecter")
    with st.expander("Mot de passe oublié"):
        reset_email = st.text_input("Email de réinitialisation", key="auth_reset_email_input")
        reset_btn = st.button("Générer un lien de réinitialisation", key="auth_reset_btn")

    if signin_btn:
        try:
            out = _signin(_normalize_email(email), password)
            if out.get("status") != "OK":
                st.error(f"Échec connexion: {out.get('status')}")
                return None
            user = out.get("user", {})
            su_user_id = _extract_user_id(user)
            em = _normalize_email(_extract_email(user, email))
            email_verified = _extract_email_verified(user)
            record = upsert_user_from_su(
                engine=engine,
                su_user_id=su_user_id,
                email=em,
                display_name=em.split("@")[0],
            )
            _maybe_promote_super_admin(engine, em, email_verified)
            record.is_super_admin = is_user_super_admin(engine, record.user_id)
            mark_user_login(engine, record.user_id)
            _set_user(record, out.get("accessToken", ""))
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Connexion impossible: {exc}")
            return None

    if reset_btn:
        normalized_email = _normalize_email(reset_email)
        try:
            link = request_password_reset_link(normalized_email)
            result = mailer.send_account_link_email(
                to_email=normalized_email,
                subject="Réinitialisation de votre mot de passe",
                intro="Cliquez sur le lien ci-dessous pour définir votre mot de passe.",
                link=link,
            )
            if result.mode == "dev":
                logger.debug("reset link for %s: %s", normalized_email, link)
        except Exception as exc:  # noqa: BLE001
            logger.debug("password reset failed: %s", exc)
        st.success("Si cet email existe, un lien a été envoyé.")
    return None
