"""
Tests for JSONL export (filtered rows, formats, stylometry option).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from src.export_utils import convert_to_jsonl


def test_convert_to_jsonl_only_validated_status(df_mixed_status: pd.DataFrame) -> None:
    """Export must include only rows with statut « Fait et validé »."""
    out = convert_to_jsonl(df_mixed_status, "lfm2", include_stylometry=False)
    lines = [ln for ln in out.strip().split("\n") if ln]
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert "messages" in data
    assert any("Prose polie" in m.get("content", "") for m in data["messages"])


def test_convert_to_jsonl_empty_when_no_validated() -> None:
    """No validated rows yields empty string (no JSONL lines)."""
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "type": "Normalisation",
                "forme": "Narration",
                "ton": "Neutre",
                "support": "Narratif",
                "input": "x",
                "output": "y",
                "statut": "A faire",
                "notes": "",
            }
        ]
    )
    assert convert_to_jsonl(df, "mistral") == ""


def test_convert_to_jsonl_unknown_format_raises() -> None:
    """Invalid format key raises ValueError."""
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "type": "Normalisation",
                "forme": "Narration",
                "ton": "Neutre",
                "support": "Narratif",
                "input": "in",
                "output": "out",
                "statut": "Fait et validé",
                "notes": "",
            }
        ]
    )
    with pytest.raises(ValueError, match="Format inconnu"):
        convert_to_jsonl(df, "not-a-format")  # type: ignore[arg-type]


def test_convert_to_jsonl_lfm2_includes_stylometry_in_system(df_validated: pd.DataFrame) -> None:
    """With include_stylometry, LFM2 adds a system message when cache has metrics."""
    out = convert_to_jsonl(df_validated, "lfm2", include_stylometry=True)
    data = json.loads(out.strip().split("\n")[0])
    roles = [m["role"] for m in data["messages"]]
    assert "system" in roles
    system_msg = next(m for m in data["messages"] if m["role"] == "system")
    assert "TTR" in system_msg["content"] or "Indicateurs" in system_msg["content"]
