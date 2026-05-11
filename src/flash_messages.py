"""Persisted UI feedback for actions followed by ``st.rerun()``.

Streamlit clears ephemeral widgets after a rerun; storing a short-lived message
in ``session_state`` and rendering it at the start of the next run matches the
« consume once » pattern used elsewhere in the app (e.g. post-save stylometry).
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Literal, TypedDict, cast

import streamlit as st

POST_RERUN_FLASH_KEY = "post_rerun_flash_payload"

FlashLevel = Literal["success", "warning", "error", "info"]

_FLASH_LEVELS: frozenset[str] = frozenset({"success", "warning", "error", "info"})


class PostRerunFlashPayload(TypedDict):
    """Serialized flash shown once after the next script run."""

    message: str
    level: FlashLevel


def schedule_post_rerun_flash(
    session: MutableMapping[str, Any],
    message: str,
    *,
    level: FlashLevel = "success",
) -> None:
    """Store a message to display on the next run, then ``st.rerun()``.

    Replaces any previously scheduled flash for this session (no queue buildup).

    Args:
        session: Typically ``st.session_state``.
        message: User-facing text (French UI copy allowed).
        level: Which Streamlit banner to use when rendering.
    """
    session[POST_RERUN_FLASH_KEY] = PostRerunFlashPayload(message=message, level=level)


def consume_post_rerun_flash(session: MutableMapping[str, Any]) -> PostRerunFlashPayload | None:
    """Remove and return a scheduled flash payload, if any.

    Args:
        session: Typically ``st.session_state``.

    Returns:
        The payload that was stored, or ``None`` if none was scheduled.
    """
    raw = session.pop(POST_RERUN_FLASH_KEY, None)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    message = raw.get("message")
    level = raw.get("level", "success")
    if not isinstance(message, str) or not message.strip():
        return None
    if not isinstance(level, str) or level not in _FLASH_LEVELS:
        level = "success"
    return PostRerunFlashPayload(message=message, level=cast(FlashLevel, level))


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
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)
