"""Tests unitaires — préférences d'affichage curateur (issue-023 / #186)."""

from __future__ import annotations

import json

import pytest
from src.ui_preferences import (
    UI_PREFERENCES_JSON_MAX_BYTES,
    default_ui_preferences,
    load_from_stored_raw,
    merge_patch_into_canonical,
    raise_if_preferences_json_too_large,
    serialize_canonical_preferences,
)


def test_default_preferences() -> None:
    assert default_ui_preferences() == {"density": "default", "readingComfort": "default"}


def test_load_from_empty_raw() -> None:
    assert load_from_stored_raw(None) == default_ui_preferences()
    assert load_from_stored_raw("") == default_ui_preferences()
    assert load_from_stored_raw("   ") == default_ui_preferences()


def test_load_from_malformed_json_falls_back() -> None:
    assert load_from_stored_raw("{") == default_ui_preferences()


def test_load_from_non_object_json_falls_back() -> None:
    assert load_from_stored_raw('"x"') == default_ui_preferences()


def test_load_sanitizes_unknown_keys_and_invalid_values() -> None:
    raw = json.dumps(
        {"density": "compact", "readingComfort": "nope", "extra": 1},
        ensure_ascii=False,
    )
    assert load_from_stored_raw(raw) == {
        "density": "compact",
        "readingComfort": "default",
    }


def test_merge_partial_updates_one_axis() -> None:
    base = default_ui_preferences()
    out = merge_patch_into_canonical(base, {"readingComfort": "high_contrast"})
    assert out == {"density": "default", "readingComfort": "high_contrast"}


def test_merge_preserves_other_axis() -> None:
    cur = {"density": "compact", "readingComfort": "high_contrast"}
    out = merge_patch_into_canonical(cur, {"density": "comfortable"})
    assert out == {"density": "comfortable", "readingComfort": "high_contrast"}


def test_merge_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="inconnu"):
        merge_patch_into_canonical(default_ui_preferences(), {"theme": "dark"})  # type: ignore[arg-type]


def test_merge_rejects_invalid_density() -> None:
    with pytest.raises(ValueError, match="density"):
        merge_patch_into_canonical(default_ui_preferences(), {"density": "ultra"})  # type: ignore[arg-type]


def test_json_size_guard() -> None:
    blob = "x" * (UI_PREFERENCES_JSON_MAX_BYTES + 1)
    with pytest.raises(ValueError, match="volumineuses"):
        raise_if_preferences_json_too_large(blob)


def test_serialize_accepts_canonical_under_cap() -> None:
    s = serialize_canonical_preferences(default_ui_preferences())
    assert len(s.encode("utf-8")) <= UI_PREFERENCES_JSON_MAX_BYTES
