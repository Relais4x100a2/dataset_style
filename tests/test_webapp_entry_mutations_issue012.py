"""Mutations entrées webapp (issue-012 / GitHub #134) — dimensions fermées à la création."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from src.webapp import entry_mutations


def test_append_minimal_entry_uses_type_override_when_allowed() -> None:
    """Une valeur ``type_`` dans la liste active remplace la première entrée du preset."""
    engine = MagicMock()
    dims = {
        "types": ["A", "B"],
        "structures": ["S0"],
        "tons": ["T0"],
        "formats": ["F0"],
        "publics": ["P0"],
        "statuts": ["St0"],
    }
    with (
        patch.object(entry_mutations, "require_role"),
        patch.object(entry_mutations, "get_project_settings", return_value=MagicMock()),
        patch.object(entry_mutations, "load_active_dimensions", return_value=("k", {}, dims)),
        patch.object(entry_mutations, "load_project_entries", return_value=pd.DataFrame()),
        patch.object(entry_mutations, "persist_new_entry_with_nlp_cache") as p_m,
        patch.object(entry_mutations, "load_fr_core_nlp_for_webapp", return_value=None),
    ):
        entry_mutations.append_minimal_entry(
            engine,
            "p1",
            "u1",
            input_text="in",
            output_text="out",
            type_="B",
        )
    new_df = p_m.call_args.kwargs["new_row_df"]
    assert new_df.iloc[0]["type"] == "B"


def test_append_minimal_entry_rejects_unknown_type_value() -> None:
    """Valeur hors liste : ``ValueError`` (converti en 400 côté route POST)."""
    engine = MagicMock()
    dims = {
        "types": ["A"],
        "structures": ["S0"],
        "tons": ["T0"],
        "formats": ["F0"],
        "publics": ["P0"],
        "statuts": ["St0"],
    }
    with (
        patch.object(entry_mutations, "require_role"),
        patch.object(entry_mutations, "get_project_settings", return_value=MagicMock()),
        patch.object(entry_mutations, "load_active_dimensions", return_value=("k", {}, dims)),
        patch.object(entry_mutations, "load_project_entries", return_value=pd.DataFrame()),
        patch.object(entry_mutations, "persist_new_entry_with_nlp_cache"),
        patch.object(entry_mutations, "load_fr_core_nlp_for_webapp", return_value=None),
    ):
        with pytest.raises(ValueError, match="types"):
            entry_mutations.append_minimal_entry(
                engine,
                "p1",
                "u1",
                input_text="in",
                output_text="out",
                type_="ZZZ",
            )


def _minimal_entry_row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "e1",
        "project_id": "p1",
        "date": "",
        "type": "A",
        "structure": "S0",
        "ton": "T0",
        "format": "F0",
        "public": "P0",
        "input": "in",
        "output": "out",
        "statut": "St0",
        "notes": "",
    }
    base.update(overrides)
    return base


def test_apply_entry_field_updates_rejects_patch_type_not_in_active_list() -> None:
    """PATCH métier : même contrôle de liste fermée que ``append_minimal_entry`` (QA PR #171)."""
    engine = MagicMock()
    df = pd.DataFrame([_minimal_entry_row()])
    dims = {
        "types": ["A"],
        "structures": ["S0"],
        "tons": ["T0"],
        "formats": ["F0"],
        "publics": ["P0"],
        "statuts": ["St0"],
    }
    with (
        patch.object(entry_mutations, "load_project_entries", return_value=df),
        patch.object(entry_mutations, "get_project_settings", return_value=MagicMock()),
        patch.object(entry_mutations, "load_active_dimensions", return_value=("k", {}, dims)),
        patch.object(entry_mutations, "persist_edited_entry_with_nlp_cache") as save_m,
    ):
        with pytest.raises(ValueError, match="types"):
            entry_mutations.apply_entry_field_updates(engine, "p1", "u1", "e1", {"type": "intrus"})
    save_m.assert_not_called()


def test_apply_entry_field_updates_accepts_patch_type_when_in_active_list() -> None:
    engine = MagicMock()
    df = pd.DataFrame([_minimal_entry_row()])
    dims = {
        "types": ["A", "B"],
        "structures": ["S0"],
        "tons": ["T0"],
        "formats": ["F0"],
        "publics": ["P0"],
        "statuts": ["St0"],
    }
    with (
        patch.object(entry_mutations, "load_project_entries", return_value=df),
        patch.object(entry_mutations, "get_project_settings", return_value=MagicMock()),
        patch.object(entry_mutations, "load_active_dimensions", return_value=("k", {}, dims)),
        patch.object(entry_mutations, "persist_edited_entry_with_nlp_cache") as save_m,
        patch.object(entry_mutations, "load_fr_core_nlp_for_webapp", return_value=None),
    ):
        entry_mutations.apply_entry_field_updates(engine, "p1", "u1", "e1", {"type": "B"})
    save_m.assert_called_once()
    passed_df = save_m.call_args.kwargs["df_full"]
    assert str(passed_df.loc[passed_df["id"] == "e1", "type"].iloc[0]) == "B"
