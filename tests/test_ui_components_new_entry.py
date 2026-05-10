"""Tests for « Nouvelle entrée » session-state helpers."""

from __future__ import annotations

from src.ui_components import (
    ensure_new_entry_widget_keys_initialized,
    new_entry_missing_required_body_message,
    new_entry_session_keys,
)


def test_new_entry_session_keys_are_prefixed_by_project() -> None:
    """Keys must be distinct per project to avoid draft bleed."""
    k_a = new_entry_session_keys("proj-a")
    k_b = new_entry_session_keys("proj-b")
    assert k_a["input"] != k_b["input"]
    assert k_a["input"].startswith("new_entry_proj-a_")


def test_ensure_new_entry_initializes_defaults() -> None:
    """First visit seeds dimension defaults and empty text buffers."""
    session: dict[str, object] = {}
    dims = {
        "types": ["t1", "t2"],
        "structures": ["s1"],
        "tons": ["n1"],
        "formats": ["f1"],
        "publics": ["p1"],
        "statuts": ["draft", "done"],
    }
    keys = ensure_new_entry_widget_keys_initialized(session, "p1", dims)
    assert session[keys["type"]] == "t1"
    assert session[keys["statut"]] == "draft"
    assert session[keys["input"]] == ""
    assert session[keys["output"]] == ""
    assert session[keys["notes"]] == ""


def test_ensure_new_entry_repairs_stale_select_value() -> None:
    """If preset options change, invalid stored select resets to first option."""
    session: dict[str, object] = {}
    keys = new_entry_session_keys("p1")
    session[keys["type"]] = "gone"
    dims = {
        "types": ["a", "b"],
        "structures": ["s"],
        "tons": ["n"],
        "formats": ["f"],
        "publics": ["p"],
        "statuts": ["x"],
    }
    ensure_new_entry_widget_keys_initialized(session, "p1", dims)
    assert session[keys["type"]] == "a"


def test_ensure_new_entry_preserves_long_buffers_when_repairing_dimension() -> None:
    """LLM buffers must survive preset repair (no accidental wipe on init)."""
    session: dict[str, object] = {}
    keys = new_entry_session_keys("p1")
    long_in = "draft " * 200
    long_out = "output " * 200
    session[keys["input"]] = long_in
    session[keys["output"]] = long_out
    session[keys["type"]] = "gone"
    dims = {
        "types": ["a", "b"],
        "structures": ["s"],
        "tons": ["n"],
        "formats": ["f"],
        "publics": ["p"],
        "statuts": ["x"],
    }
    ensure_new_entry_widget_keys_initialized(session, "p1", dims)
    assert session[keys["input"]] == long_in
    assert session[keys["output"]] == long_out
    assert session[keys["type"]] == "a"


def test_new_entry_missing_required_body_message() -> None:
    """Validation message only when one or both bodies are blank."""
    assert new_entry_missing_required_body_message("", "x") is not None
    assert new_entry_missing_required_body_message("x", "") is not None
    assert new_entry_missing_required_body_message("  ", "y") is not None
    assert new_entry_missing_required_body_message("a", "b") is None
