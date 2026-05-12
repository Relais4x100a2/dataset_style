"""Tests for new-entry session purge helpers."""

from __future__ import annotations

from src.new_entry_session_state import purge_all_new_entry_session_state


def test_purge_all_new_entry_session_state_removes_drafts_and_pending_flags() -> None:
    """All new-entry keys must disappear while unrelated session keys stay."""
    session: dict[str, object] = {
        "current_user": {"user_id": "u1"},
        "new_entry_p1_u_u1_input": "draft",
        "_pending_clear_new_entry_p1_u_u1": True,
        "sidebar_expanded": True,
    }
    purge_all_new_entry_session_state(session)
    assert "new_entry_p1_u_u1_input" not in session
    assert "_pending_clear_new_entry_p1_u_u1" not in session
    assert session["current_user"] == {"user_id": "u1"}
    assert session["sidebar_expanded"] is True


def test_purge_all_new_entry_session_state_legacy_pending_also_cleared() -> None:
    """Project-only pending-clear flags use the same prefix rule."""
    session: dict[str, object] = {"_pending_clear_new_entry_proj-x": True}
    purge_all_new_entry_session_state(session)
    assert "_pending_clear_new_entry_proj-x" not in session
