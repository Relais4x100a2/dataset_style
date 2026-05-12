"""Tests for post-rerun flash messages (session persistence)."""

from __future__ import annotations

from typing import Any, cast

import pytest
from src.flash_messages import (
    POST_RERUN_FLASH_KEY,
    FlashLevel,
    consume_post_rerun_flash,
    schedule_post_rerun_flash,
)


def test_schedule_then_consume_returns_payload_and_clears_session() -> None:
    """A scheduled flash is returned once and removed from session."""
    session: dict[str, Any] = {}
    schedule_post_rerun_flash(session, "Projet supprimé.", level="success")
    assert POST_RERUN_FLASH_KEY in session
    out = consume_post_rerun_flash(session)
    assert out == {"message": "Projet supprimé.", "level": "success"}
    assert consume_post_rerun_flash(session) is None
    assert POST_RERUN_FLASH_KEY not in session


def test_consume_empty_session_returns_none() -> None:
    """No flash scheduled yields None without mutating unrelated keys."""
    session: dict[str, Any] = {"other": 1}
    assert consume_post_rerun_flash(session) is None
    assert session == {"other": 1}


def test_reschedule_overwrites_previous_flash() -> None:
    """Only one flash is kept; later schedule replaces the first (no accumulation)."""
    session: dict[str, Any] = {}
    schedule_post_rerun_flash(session, "Premier", level="warning")
    schedule_post_rerun_flash(session, "Deuxième", level="success")
    assert consume_post_rerun_flash(session) == {"message": "Deuxième", "level": "success"}


def test_render_post_rerun_flash_once_invokes_streamlit() -> None:
    """After render, session key is cleared and the correct widget is called."""
    from unittest.mock import MagicMock, patch

    from src.flash_messages import render_post_rerun_flash_once

    session: dict[str, Any] = {}
    schedule_post_rerun_flash(session, "OK", level="success")
    fake_st = MagicMock()
    with patch("src.flash_messages.st", fake_st):
        render_post_rerun_flash_once(session)
    fake_st.success.assert_called_once_with("OK")
    assert POST_RERUN_FLASH_KEY not in session


@pytest.mark.parametrize(
    ("level", "attr"),
    [
        ("success", "success"),
        ("warning", "warning"),
        ("error", "error"),
        ("info", "info"),
    ],
)
def test_render_post_rerun_flash_routes_level(level: str, attr: str) -> None:
    """Each supported level maps to the matching Streamlit call."""
    from unittest.mock import MagicMock, patch

    from src.flash_messages import render_post_rerun_flash_once

    session: dict[str, Any] = {}
    schedule_post_rerun_flash(session, "Msg", level=cast(FlashLevel, level))
    fake_st = MagicMock()
    with patch("src.flash_messages.st", fake_st):
        render_post_rerun_flash_once(session)
    getattr(fake_st, attr).assert_called_once_with("Msg")


def test_logout_preserves_scheduled_flash_payload() -> None:
    """Self-delete schedules flash before ``logout()``; auth and new-entry drafts cleared."""
    from unittest.mock import patch

    from src import auth

    session: dict[str, Any] = {
        "current_user": {
            "user_id": "u1",
            "email": "a@example.com",
            "display_name": "a",
            "access_token": "tok",
            "is_super_admin": False,
        },
        "new_entry_p1_u_u1_input": "draft",
        "_pending_clear_new_entry_p1_u_u1": True,
    }
    schedule_post_rerun_flash(session, "Compte supprimé.")
    with patch.object(auth.st, "session_state", session):
        auth.logout()
    assert "current_user" not in session
    assert "new_entry_p1_u_u1_input" not in session
    assert "_pending_clear_new_entry_p1_u_u1" not in session
    assert POST_RERUN_FLASH_KEY in session
    assert session[POST_RERUN_FLASH_KEY]["message"] == "Compte supprimé."


def test_render_auth_gate_invokes_post_rerun_flash_once() -> None:
    """Every authenticated request path must consume scheduled flash before widgets."""
    from unittest.mock import MagicMock, patch

    from src import auth
    from src.auth import CurrentUser

    engine = MagicMock()
    user = CurrentUser(
        user_id="u1",
        email="a@example.com",
        display_name="alice",
        access_token="tok",
        is_super_admin=False,
    )
    state: dict[str, Any] = {}
    with patch.object(auth, "render_post_rerun_flash_once") as flash_mock:
        with patch.object(auth.st, "session_state", state):
            with patch.object(auth, "get_current_user", return_value=user):
                out = auth.render_auth_gate(engine)
    flash_mock.assert_called_once_with(state)
    assert out == user


def test_persist_settings_schedules_flash_not_ephemeral_success() -> None:
    """Settings save + ``st.rerun()`` must not rely on ``st.success`` (cleared by rerun)."""
    from unittest.mock import MagicMock, patch

    from src.auth import CurrentUser
    from src.database import ProjectSettings
    from src.ui_components import _persist_settings

    user = CurrentUser(
        user_id="u1",
        email="a@example.com",
        display_name="alice",
        access_token="tok",
        is_super_admin=False,
    )
    engine = MagicMock()
    settings = ProjectSettings()
    session: dict[str, Any] = {}

    with patch("src.ui_components.update_project_settings_as_admin"):
        with patch("src.ui_components.schedule_post_rerun_flash") as sched_mock:
            fake_st = MagicMock()
            fake_st.session_state = session
            with patch("src.ui_components.st", fake_st):
                _persist_settings(user, engine, "p1", settings, "Réglages projet enregistrés.")
    sched_mock.assert_called_once_with(session, "Réglages projet enregistrés.", level="success")
    fake_st.rerun.assert_called_once()
    fake_st.success.assert_not_called()


def test_persist_settings_on_error_does_not_schedule_flash() -> None:
    """Persistence errors must not enqueue a success flash."""
    from unittest.mock import MagicMock, patch

    from src.auth import CurrentUser
    from src.database import ProjectSettings
    from src.ui_components import _persist_settings

    user = CurrentUser(
        user_id="u1",
        email="a@example.com",
        display_name="alice",
        access_token="tok",
        is_super_admin=False,
    )
    engine = MagicMock()
    settings = ProjectSettings()
    session: dict[str, Any] = {}

    with patch(
        "src.ui_components.update_project_settings_as_admin",
        side_effect=RuntimeError("db"),
    ):
        with patch("src.ui_components.schedule_post_rerun_flash") as sched_mock:
            fake_st = MagicMock()
            fake_st.session_state = session
            with patch("src.ui_components.st", fake_st):
                _persist_settings(user, engine, "p1", settings, "Ne doit pas s'afficher.")
    sched_mock.assert_not_called()
    fake_st.rerun.assert_not_called()
