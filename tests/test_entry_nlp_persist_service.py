"""Tests du service de persistance entrées + cache NLP (issue-012)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
from sqlalchemy.engine import Engine
from src.services import entry_nlp_persist_service as enp


def test_persist_new_entry_combines_prepare_compute_and_update() -> None:
    engine = MagicMock(spec=Engine)
    df_existing = pd.DataFrame(
        [{"id": "e0", "input": "a", "output": "b", "statut": "En cours", "project_id": "p1"}]
    )
    new_row = pd.DataFrame(
        [
            {
                "id": "e_new",
                "input": "in",
                "output": "out",
                "statut": "En cours",
                "project_id": "p1",
            }
        ]
    )
    fake_pkg = MagicMock()
    fake_pkg.cache = {"_coherence_score": "42"}
    nlp = object()
    with (
        patch(
            "src.services.entry_nlp_persist_service.prepare_for_edition_tab",
            side_effect=lambda d: d,
        ) as prep_m,
        patch(
            "src.services.entry_nlp_persist_service.compute_row_cache",
            return_value=fake_pkg,
        ) as nlp_m,
        patch("src.services.entry_nlp_persist_service.update_project_entries") as upd_m,
    ):
        rid = enp.persist_new_entry_with_nlp_cache(
            engine,
            "p1",
            "u1",
            df_existing=df_existing,
            new_row_df=new_row,
            input_text="in",
            output_text="out",
            nlp=nlp,
        )
    assert rid == "e_new"
    assert prep_m.call_count == 2
    nlp_m.assert_called_once()
    upd_m.assert_called_once()
    args, kwargs = upd_m.call_args
    assert args[0] is engine
    assert args[1] == "p1"
    persisted: pd.DataFrame = args[2]
    assert len(persisted) == 2
    row_new = persisted[persisted["id"] == "e_new"].iloc[0]
    assert row_new["_coherence_score"] == "42"


def test_persist_edited_entry_applies_cache_to_matching_row() -> None:
    engine = MagicMock(spec=Engine)
    df = pd.DataFrame(
        [
            {"id": "e1", "input": "x", "output": "y", "statut": "En cours"},
            {"id": "e2", "input": "a", "output": "b", "statut": "Validé"},
        ]
    )
    fake_pkg = MagicMock()
    fake_pkg.cache = {"_ratio": "1.5"}
    with (
        patch(
            "src.services.entry_nlp_persist_service.prepare_for_edition_tab",
            side_effect=lambda d: d,
        ),
        patch(
            "src.services.entry_nlp_persist_service.compute_row_cache",
            return_value=fake_pkg,
        ) as nlp_m,
        patch("src.services.entry_nlp_persist_service.update_project_entries") as upd_m,
    ):
        enp.persist_edited_entry_with_nlp_cache(
            engine,
            "p1",
            "u1",
            df_full=df,
            entry_id="e1",
            input_text="x",
            output_text="y",
            nlp=None,
        )
    nlp_m.assert_called_once()
    upd_m.assert_called_once()
    out_df: pd.DataFrame = upd_m.call_args[0][2]
    assert out_df.loc[out_df["id"] == "e1", "_ratio"].iloc[0] == "1.5"
