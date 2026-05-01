"""
Pytest configuration and shared fixtures.
"""

from __future__ import annotations

import pandas as pd
import pytest
from src.database import ALL_COLUMNS, STATUT_VALIDE


@pytest.fixture
def sample_valid_row() -> dict[str, str]:
    """One validated entry with cache columns filled (stylometry pipeline output)."""
    row = {c: "" for c in ALL_COLUMNS}
    row.update(
        {
            "id": "test-1",
            "type": "Normalisation",
            "forme": "Narration",
            "ton": "Neutre",
            "support": "Narratif",
            "input": "Brouillon court.",
            "output": "Prose polie.",
            "statut": STATUT_VALIDE,
            "notes": "",
            "_ratio": "1.2",
            "_ttr": "0.65",
            "_long_phrases": "12",
            "_signature_json": '{"a": 0.5, "b": 0.3}',
            "_coherence_score": "70",
            "_trigrams_json": '{"DET+NOUN+VERB": 2}',
            "_lexical_density": "0.4",
            "_weak_verb_ratio": "0.1",
            "_syntax_contrast": "0.2",
            "_nb_sentences": "3",
            "_punct_exp": "0.05",
            "_stop_ratio_out": "0.12",
        }
    )
    return row


@pytest.fixture
def df_validated(sample_valid_row: dict[str, str]) -> pd.DataFrame:
    """DataFrame with one validated row."""
    return pd.DataFrame([sample_valid_row])


@pytest.fixture
def df_mixed_status(sample_valid_row: dict[str, str]) -> pd.DataFrame:
    """Validated + draft rows (export must only keep validated)."""
    draft = dict(sample_valid_row)
    draft["id"] = "draft-1"
    draft["statut"] = "A faire"
    return pd.DataFrame([sample_valid_row, draft])
