"""Tests for « Nouvelle entrée » session-state helpers."""

from __future__ import annotations

from src.ui_components import (
    commit_new_entry_llm_result,
    ensure_new_entry_widget_keys_initialized,
    new_entry_missing_required_body_message,
    new_entry_pending_clear_session_key,
    new_entry_session_keys,
)


def test_new_entry_session_keys_are_scoped_by_project_and_user() -> None:
    """Keys must be distinct per project and per user to avoid draft bleed."""
    k_a = new_entry_session_keys("proj-a", "user-1")
    k_b = new_entry_session_keys("proj-b", "user-1")
    k_other_user = new_entry_session_keys("proj-a", "user-2")
    assert k_a["input"] != k_b["input"]
    assert k_a["input"] != k_other_user["input"]
    assert k_a["input"].startswith("new_entry_proj-a_u_user-1_")


def test_new_entry_session_keys_sanitize_user_id_for_key_string() -> None:
    """Unsafe characters in user ids must not break session key strings."""
    keys = new_entry_session_keys("p1", "weird@id!")
    assert "@" not in keys["input"]
    assert "new_entry_p1_u_weird_id_" in keys["input"]


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
    keys = ensure_new_entry_widget_keys_initialized(session, "p1", "u1", dims)
    assert session[keys["type"]] == "t1"
    assert session[keys["statut"]] == "draft"
    assert session[keys["input"]] == ""
    assert session[keys["output"]] == ""
    assert session[keys["notes"]] == ""


def test_ensure_new_entry_repairs_stale_select_value() -> None:
    """If preset options change, invalid stored select resets to first option."""
    session: dict[str, object] = {}
    keys = new_entry_session_keys("p1", "u1")
    session[keys["type"]] = "gone"
    dims = {
        "types": ["a", "b"],
        "structures": ["s"],
        "tons": ["n"],
        "formats": ["f"],
        "publics": ["p"],
        "statuts": ["x"],
    }
    ensure_new_entry_widget_keys_initialized(session, "p1", "u1", dims)
    assert session[keys["type"]] == "a"


def test_ensure_new_entry_preserves_long_buffers_when_repairing_dimension() -> None:
    """LLM buffers must survive preset repair (no accidental wipe on init)."""
    session: dict[str, object] = {}
    keys = new_entry_session_keys("p1", "u1")
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
    ensure_new_entry_widget_keys_initialized(session, "p1", "u1", dims)
    assert session[keys["input"]] == long_in
    assert session[keys["output"]] == long_out
    assert session[keys["type"]] == "a"


def test_ensure_new_entry_discards_legacy_project_only_buffers() -> None:
    """Legacy project-only keys must not be copied into the current user's buffers."""
    session: dict[str, object] = {}
    session["new_entry_p9_input"] = "legacy draft"
    session["new_entry_p9_output"] = "legacy output"
    dims = {
        "types": ["t"],
        "structures": ["s"],
        "tons": ["n"],
        "formats": ["f"],
        "publics": ["p"],
        "statuts": ["st"],
    }
    keys = ensure_new_entry_widget_keys_initialized(session, "p9", "curator-1", dims)
    assert session[keys["input"]] == ""
    assert session[keys["output"]] == ""
    assert "new_entry_p9_input" not in session
    assert "new_entry_p9_output" not in session


def test_ensure_new_entry_second_user_does_not_inherit_legacy_text() -> None:
    """Regression (QA): another account must not receive legacy buffers on same session."""
    session: dict[str, object] = {}
    session["new_entry_p9_input"] = "written under unknown account"
    dims = {
        "types": ["t"],
        "structures": ["s"],
        "tons": ["n"],
        "formats": ["f"],
        "publics": ["p"],
        "statuts": ["st"],
    }
    keys_b = ensure_new_entry_widget_keys_initialized(session, "p9", "user-b", dims)
    assert session[keys_b["input"]] == ""
    assert session[keys_b["output"]] == ""
    assert "new_entry_p9_input" not in session


def test_new_entry_pending_clear_session_key_is_scoped_by_project_and_user() -> None:
    """Pending-clear flag must not collide across projects or users."""
    assert new_entry_pending_clear_session_key("a", "u1") != new_entry_pending_clear_session_key(
        "b", "u1"
    )
    assert new_entry_pending_clear_session_key("a", "u1") != new_entry_pending_clear_session_key(
        "a", "u2"
    )


def test_commit_new_entry_llm_result_writes_canonical_buffers() -> None:
    """LLM results must land on the same keys used by the Streamlit widgets."""
    keys = new_entry_session_keys("proj-z", "user-x")
    session: dict[str, object] = {
        keys["input"]: "keep",
        keys["output"]: "",
    }
    commit_new_entry_llm_result(session, keys, target="output", text="LLM out")
    assert session[keys["output"]] == "LLM out"
    assert session[keys["input"]] == "keep"
    commit_new_entry_llm_result(session, keys, target="input", text="LLM in")
    assert session[keys["input"]] == "LLM in"


def test_new_entry_missing_required_body_message() -> None:
    """Validation message only when one or both bodies are blank."""
    assert new_entry_missing_required_body_message("", "x") is not None
    assert new_entry_missing_required_body_message("x", "") is not None
    assert new_entry_missing_required_body_message("  ", "y") is not None
    assert new_entry_missing_required_body_message("a", "b") is None
