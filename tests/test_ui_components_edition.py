"""Tests for edition tab session-state helpers."""

from __future__ import annotations

from src.ui_components import (
    read_edition_output_text_for_persist,
    sync_edition_output_widget_state,
)


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


def test_read_edition_output_prefers_session_widget_buffer() -> None:
    """Persistence must read the keyed text_area buffer (e.g. after LT correction)."""
    key = "edit_output_e1"
    session: dict[str, object] = {key: "corrected_full_text" * 50}
    out = read_edition_output_text_for_persist(session, key, "from_database_row")
    assert out == session[key]
    assert out != "from_database_row"


def test_read_edition_output_falls_back_when_widget_key_absent() -> None:
    """If the widget key is missing, use the row snapshot (defensive)."""
    session: dict[str, object] = {}
    key = "edit_output_missing"
    assert read_edition_output_text_for_persist(session, key, "row_only") == "row_only"


def test_read_edition_output_empty_string_in_session() -> None:
    """Explicit empty buffer in session must not fall back to stale row text."""
    key = "edit_output_e2"
    session: dict[str, object] = {key: ""}
    assert read_edition_output_text_for_persist(session, key, "non_empty_row") == ""
