"""Persisted UI feedback for actions followed by ``st.rerun()``.

Streamlit clears ephemeral widgets after a rerun; storing a short-lived message
in ``session_state`` and rendering it at the start of the next run matches the
« consume once » pattern used elsewhere in the app (e.g. post-save stylometry).

Issue-030 / TTL: a scheduled message is shown **once**, on the **next** full
script run (typically right after ``render_auth_gate``), then removed from
``session_state``. It does not persist across further reruns or tab changes unless
you schedule again — équivalent produit d’un TTL d’**une exécution**.

Super-admin actions use ``channel="super_admin"`` and a dedicated session key
(``POST_RERUN_FLASH_ADMIN_KEY``) so their flashes do not collide with curator keys;
scheduling on one channel clears the other channel’s pending payload.

La bannière de migration (``APP_MIGRATION_INFO_BANNER``) est rendue **après** le
flash consommé et **avant** le formulaire de connexion dans ``render_auth_gate``,
puis sous le titre principal côté curateur connecté : elle reste visible tant
que la variable est définie, indépendamment du flash « une exécution ».
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Literal, NotRequired, TypedDict, cast

import streamlit as st

POST_RERUN_FLASH_KEY = "post_rerun_flash_payload"
POST_RERUN_FLASH_ADMIN_KEY = "ds_super_admin_post_rerun_flash_v1"

FlashLevel = Literal["success", "warning", "error", "info"]
FlashChannel = Literal["default", "super_admin"]

_FLASH_LEVELS: frozenset[str] = frozenset({"success", "warning", "error", "info"})


class PostRerunFlashPayload(TypedDict):
    """Serialized flash shown once after the next script run."""

    message: str
    level: FlashLevel
    code: NotRequired[str]


def _payload_from_stored_raw(raw: object) -> PostRerunFlashPayload | None:
    """Build a validated payload from session storage, or ``None`` if invalid."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    message = raw.get("message")
    level = raw.get("level", "success")
    code_raw = raw.get("code")
    if not isinstance(message, str) or not message.strip():
        return None
    if not isinstance(level, str) or level not in _FLASH_LEVELS:
        level = "success"
    out: PostRerunFlashPayload = PostRerunFlashPayload(
        message=message, level=cast(FlashLevel, level)
    )
    if isinstance(code_raw, str) and code_raw.strip():
        out["code"] = code_raw.strip()
    return out


def schedule_post_rerun_flash(
    session: MutableMapping[str, Any],
    message: str,
    *,
    level: FlashLevel = "success",
    channel: FlashChannel = "default",
    code: str | None = None,
) -> None:
    """Store a message to display on the next run.

    Call ``st.rerun()`` **after** this function from the UI code path; this helper
    does not rerun by itself.

    Replaces any previously scheduled flash on the **same** channel; scheduling
    also clears the **other** channel so at most one cross-app flash is pending.

    Args:
        session: Typically ``st.session_state``.
        message: User-facing text (French UI copy allowed).
        level: Which Streamlit banner to use when rendering.
        channel: ``default`` (curateur / compte) or ``super_admin`` (namespace
            dédié, clé ``POST_RERUN_FLASH_ADMIN_KEY``).
        code: Code d'erreur stable (issue-005 / issue-022) pour le futur front ; optionnel.
    """
    payload: PostRerunFlashPayload = PostRerunFlashPayload(message=message, level=level)
    if code is not None and str(code).strip():
        payload["code"] = str(code).strip()
    if channel == "super_admin":
        session.pop(POST_RERUN_FLASH_KEY, None)
        session[POST_RERUN_FLASH_ADMIN_KEY] = payload
    else:
        session.pop(POST_RERUN_FLASH_ADMIN_KEY, None)
        session[POST_RERUN_FLASH_KEY] = payload


def consume_post_rerun_flash(session: MutableMapping[str, Any]) -> PostRerunFlashPayload | None:
    """Remove and return a scheduled flash payload, if any.

    Tries the default key first, then the super-admin key (issue-030).

    Args:
        session: Typically ``st.session_state``.

    Returns:
        The payload that was stored, or ``None`` if none was scheduled.
    """
    for state_key in (POST_RERUN_FLASH_KEY, POST_RERUN_FLASH_ADMIN_KEY):
        raw = session.pop(state_key, None)
        payload = _payload_from_stored_raw(raw)
        if payload is not None:
            return payload
    return None


def render_post_rerun_flash_once(session: MutableMapping[str, Any]) -> None:
    """Pop a scheduled flash from ``session`` and render it with Streamlit.

    Safe to call on every run: does nothing when no flash was scheduled.

    Args:
        session: Typically ``st.session_state``.
    """
    payload = consume_post_rerun_flash(session)
    if payload is None:
        return
    message = payload["message"]
    level = payload["level"]
    err_code = payload.get("code")
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)
    if err_code:
        st.caption(f"code: {err_code}")
