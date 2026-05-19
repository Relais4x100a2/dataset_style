"""Validation des dimensions projet (parité Streamlit / issue-011)."""

from __future__ import annotations

import json

import pytest
from src.database import ProjectSettings
from src.presets import (
    DIMENSIONS_JSON_OBJECT_EXPECTED_FR,
    RESERVED_BUILTIN_PRESET_KEY_FR,
    STATUTS_LIST_CANNOT_BE_EMPTY_FR,
    apply_load_preset_to_settings,
    apply_replace_dimensions_to_settings,
    apply_save_custom_preset_to_settings,
    normalize_custom_preset_storage_key,
    validate_replace_dimensions_payload,
)


def test_validate_replace_dimensions_rejects_explicit_empty_statuts() -> None:
    raw = {"types": ["A"], "statuts": []}
    dims, err = validate_replace_dimensions_payload(raw)
    assert dims is None
    assert err == STATUTS_LIST_CANNOT_BE_EMPTY_FR


def test_validate_replace_dimensions_accepts_omitted_statuts_defaults() -> None:
    raw = {"types": ["Un type"]}
    dims, err = validate_replace_dimensions_payload(raw)
    assert err is None
    assert dims is not None
    assert dims["types"] == ["Un type"]
    assert dims["statuts"]


def test_validate_replace_dimensions_rejects_non_object() -> None:
    dims, err = validate_replace_dimensions_payload([])
    assert dims is None
    assert err == DIMENSIONS_JSON_OBJECT_EXPECTED_FR


def test_normalize_custom_preset_storage_key() -> None:
    assert normalize_custom_preset_storage_key("  Mon Profil  ") == "mon_profil"


def test_apply_load_preset_unknown_key() -> None:
    s = ProjectSettings(active_preset_key="roman", custom_presets_json="")
    out, err = apply_load_preset_to_settings(s, "not_a_real_preset")
    assert out is None
    assert err is not None


def test_apply_load_preset_builtin_roman() -> None:
    s = ProjectSettings(
        active_preset_key="contenu",
        custom_presets_json="",
        dimensions_override_json="",
    )
    out, err = apply_load_preset_to_settings(s, "roman")
    assert err is None
    assert out is not None
    assert out.active_preset_key == "roman"
    assert out.dimensions_override_json


def test_apply_save_custom_rejects_builtin_key() -> None:
    s = ProjectSettings()
    dims = {
        "types": ["T"],
        "structures": [],
        "tons": [],
        "formats": [],
        "publics": [],
        "statuts": ["A faire"],
    }
    out, err = apply_save_custom_preset_to_settings(s, "roman", "Lib", dims)
    assert out is None
    assert err == RESERVED_BUILTIN_PRESET_KEY_FR


@pytest.mark.parametrize(
    ("name", "label", "dims", "expect_err"),
    [
        (
            "perso_x",
            "Mon perso",
            {
                "types": ["T"],
                "structures": ["S"],
                "tons": ["N"],
                "formats": ["F"],
                "publics": ["P"],
                "statuts": ["Brouillon"],
            },
            None,
        ),
        ("", "L", {"statuts": ["X"]}, "Identifiant du profil requis."),
    ],
)
def test_apply_save_custom_preset(
    name: str, label: str, dims: dict, expect_err: str | None
) -> None:
    s = ProjectSettings(
        active_preset_key="roman",
        custom_presets_json="",
        dimensions_override_json="",
    )
    out, err = apply_save_custom_preset_to_settings(s, name, label, dims)
    if expect_err:
        assert err is not None
        assert expect_err in err
        assert out is None
    else:
        assert err is None
        assert out is not None
        assert out.active_preset_key == "perso_x"
        assert "perso_x" in (out.custom_presets_json or "")


def test_apply_replace_dimensions_persists_normalized_json() -> None:
    s = ProjectSettings()
    raw = {
        "types": ["  A ", "A", "B"],
        "structures": [],
        "tons": [],
        "formats": [],
        "publics": [],
        "statuts": ["En cours"],
    }
    out, err = apply_replace_dimensions_to_settings(s, raw)
    assert err is None
    assert out is not None
    parsed = json.loads(out.dimensions_override_json)
    assert parsed["types"] == ["A", "B"]
