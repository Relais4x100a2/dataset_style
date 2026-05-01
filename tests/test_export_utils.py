"""
Module: tests.test_export_utils
Tests pour ``src.export_utils`` : filtrage statut, formats JSONL, erreurs.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from src.database import STATUT_VALIDE
from src.export_utils import _stylometry_summary, convert_to_jsonl


def _minimal_valid_row() -> dict[str, str]:
    """Une fiche minimaliste « Fait et validé » pour les exports."""
    return {
        "id": "1",
        "type": "Normalisation",
        "forme": "Narration",
        "ton": "Neutre",
        "support": "Narratif",
        "input": "brouillon test",
        "output": "prose finale",
        "statut": STATUT_VALIDE,
        "notes": "",
    }


def test_convert_to_jsonl_filters_only_validated() -> None:
    """Seules les lignes au statut validé sont exportées."""
    df = pd.DataFrame(
        [
            {**_minimal_valid_row(), "id": "1"},
            {
                **_minimal_valid_row(),
                "id": "2",
                "statut": "A faire",
                "output": "ignoré",
            },
        ]
    )
    out = convert_to_jsonl(df, "lfm2", include_stylometry=False)
    lines = [ln for ln in out.strip().split("\n") if ln]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["messages"][-1]["content"] == "prose finale"


def test_convert_to_jsonl_lfm2_structure() -> None:
    """Format LFM2 : user puis assistant sans system si stylométrie absente."""
    df = pd.DataFrame([_minimal_valid_row()])
    out = convert_to_jsonl(df, "lfm2", include_stylometry=False)
    payload = json.loads(out.strip().split("\n")[0])
    msgs = payload["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert "Brouillon" in msgs[0]["content"]


def test_convert_to_jsonl_lfm2_with_stylometry_adds_system() -> None:
    """Avec stylométrie et colonnes cache, un message system est ajouté."""
    row = _minimal_valid_row()
    row["_ttr"] = "0.55"
    row["_long_phrases"] = "14"
    df = pd.DataFrame([row])
    out = convert_to_jsonl(df, "lfm2", include_stylometry=True)
    payload = json.loads(out.strip().split("\n")[0])
    msgs = payload["messages"]
    assert msgs[0]["role"] == "system"
    assert "TTR" in msgs[0]["content"]


def test_convert_to_jsonl_baguettotron_h_token_by_type() -> None:
    """Normalisation utilise <H≈0.3>, Expansion utilise <H≈1.5>."""
    df = pd.DataFrame(
        [
            {**_minimal_valid_row(), "type": "Normalisation"},
            {**_minimal_valid_row(), "id": "2", "type": "Expansion"},
        ]
    )
    out = convert_to_jsonl(df, "baguettotron", include_stylometry=False)
    lines = out.strip().split("\n")
    t0 = json.loads(lines[0])["text"]
    t1 = json.loads(lines[1])["text"]
    assert "<H≈0.3>" in t0
    assert "<H≈1.5>" in t1


def test_convert_to_jsonl_unknown_format_raises() -> None:
    """Format non reconnu lève ValueError."""
    df = pd.DataFrame([_minimal_valid_row()])
    with pytest.raises(ValueError, match="Format inconnu"):
        convert_to_jsonl(df, "unknown_format")  # type: ignore[arg-type]


def test_stylometry_summary_skips_invalid_numbers() -> None:
    """Valeurs non numériques dans le cache sont ignorées sans planter."""
    s = pd.Series({"_ttr": "bad", "_long_phrases": "not-a-float"})
    assert _stylometry_summary(s) == ""


def test_stylometry_summary_joins_known_metrics() -> None:
    """Plusieurs métriques valides sont reliées par ' | '."""
    s = pd.Series({"_ttr": "0.5", "_nb_sentences": "4"})
    txt = _stylometry_summary(s)
    assert "TTR≈0.50" in txt
    assert "4 phrases" in txt
    assert " | " in txt
