"""Tests for edition tab session-state helpers."""

from __future__ import annotations

from src.ui_components import sync_edition_output_widget_state


def test_sync_edition_output_initializes_from_row() -> None:
    """First visit sets widget session value from persisted row output."""
    session: dict[str, object] = {}
    key = sync_edition_output_widget_state(session, "entry1", "stored text")
    assert key == "edit_output_entry1"
    assert session[key] == "stored text"
    assert session["edition_last_entry_id"] == "entry1"


def test_sync_edition_output_resets_on_entry_change() -> None:
    """Switching the selected entry loads that row's output into session."""
    session: dict[str, object] = {}
    sync_edition_output_widget_state(session, "a", "text_a")
    sync_edition_output_widget_state(session, "b", "text_b")
    assert session["edit_output_b"] == "text_b"
    assert session["edition_last_entry_id"] == "b"


def test_sync_edition_output_preserves_draft_same_entry() -> None:
    """Unsaved edits (e.g. after spellcheck) are not overwritten from the DB row."""
    session: dict[str, object] = {}
    key = sync_edition_output_widget_state(session, "x", "from_db")
    session[key] = "user_or_api_corrected"
    sync_edition_output_widget_state(session, "x", "from_db")
    assert session[key] == "user_or_api_corrected"
