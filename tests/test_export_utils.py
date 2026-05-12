"""
Tests pour les exports JSONL multi-formats.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from src.database import STATUT_VALIDE
from src.export_utils import (
    convert_to_jsonl,
    dataframe_for_export,
    export_perimeter_ui_recap_fr,
)


def _minimal_valid_row() -> dict[str, str]:
    """Une ligne minimaliste « Fait et validé » pour les exports."""
    return {
        "id": "e1",
        "type": "Normalisation",
        "forme": "Dialogue",
        "ton": "Neutre",
        "support": "Narratif",
        "input": "Il fait beau.",
        "output": "Le ciel est dégagé et la lumière douce.",
        "statut": STATUT_VALIDE,
        "notes": "",
        "_ttr": "0.65",
        "_long_phrases": "12",
        "_lexical_density": "0.55",
        "_nb_sentences": "2",
        "_weak_verb_ratio": "0.2",
    }


def test_convert_to_jsonl_lfm2_happy_path() -> None:
    """Une fiche validée produit une ligne JSON avec messages user/assistant."""
    df = pd.DataFrame([_minimal_valid_row()])
    raw = convert_to_jsonl(df, "lfm2", include_stylometry=False)
    lines = [ln for ln in raw.strip().split("\n") if ln]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert "messages" in obj
    roles = [m["role"] for m in obj["messages"]]
    assert roles == ["user", "assistant"]


def test_convert_to_jsonl_lfm2_includes_system_when_stylometry() -> None:
    """Avec stylométrie et colonnes cache, un message system est ajouté."""
    df = pd.DataFrame([_minimal_valid_row()])
    raw = convert_to_jsonl(df, "lfm2", include_stylometry=True)
    obj = json.loads(raw.strip().split("\n")[0])
    roles = [m["role"] for m in obj["messages"]]
    assert "system" in roles


def test_convert_to_jsonl_only_validated_rows_by_default() -> None:
    """Par défaut, seules les fiches STATUT_VALIDE sont exportées."""
    df = pd.DataFrame(
        [
            {**_minimal_valid_row(), "id": "a", "statut": STATUT_VALIDE},
            {**_minimal_valid_row(), "id": "b", "statut": "A faire"},
        ]
    )
    raw = convert_to_jsonl(df, "mistral", include_stylometry=False)
    lines = [ln for ln in raw.strip().split("\n") if ln]
    assert len(lines) == 1


def test_convert_to_jsonl_full_dataset_includes_all_statuses() -> None:
    """Avec scope full_dataset, les lignes non validées sont incluses."""
    df = pd.DataFrame(
        [
            {**_minimal_valid_row(), "id": "a", "statut": STATUT_VALIDE},
            {**_minimal_valid_row(), "id": "b", "statut": "A faire"},
        ]
    )
    raw = convert_to_jsonl(
        df,
        "mistral",
        include_stylometry=False,
        scope="full_dataset",
    )
    lines = [ln for ln in raw.strip().split("\n") if ln]
    assert len(lines) == 2


def test_export_record_count_matches_between_csv_and_jsonl_per_scope() -> None:
    """Même périmètre : autant d'enregistrements en CSV (lignes de données) qu'en JSONL."""
    df = pd.DataFrame(
        [
            {**_minimal_valid_row(), "id": "a", "statut": STATUT_VALIDE},
            {**_minimal_valid_row(), "id": "b", "statut": "A faire"},
        ]
    )
    for scope in ("validated_only", "full_dataset"):
        df_slice = dataframe_for_export(df, scope)
        jsonl_lines = [
            ln for ln in convert_to_jsonl(df, "lfm2", scope=scope).split("\n") if ln.strip()
        ]
        assert len(jsonl_lines) == len(df_slice)


def test_convert_to_jsonl_baguettotron_h_token() -> None:
    """Normalisation vs Expansion influence le jeton <H≈…> dans le texte."""
    base = _minimal_valid_row()
    df_norm = pd.DataFrame([{**base, "type": "Normalisation"}])
    df_exp = pd.DataFrame([{**base, "id": "e2", "type": "Expansion"}])
    t_norm = json.loads(convert_to_jsonl(df_norm, "baguettotron").strip().split("\n")[0])["text"]
    t_exp = json.loads(convert_to_jsonl(df_exp, "baguettotron").strip().split("\n")[0])["text"]
    assert "<H≈0.3>" in t_norm
    assert "<H≈1.5>" in t_exp


def test_export_perimeter_ui_recap_fr_positive_count() -> None:
    """Issue 019 : récap minimal quand le périmètre contient des fiches."""
    caption, warning = export_perimeter_ui_recap_fr(3, "validated_only")
    assert warning is None
    assert "3" in caption
    assert "csv" in caption.lower()
    assert "jsonl" in caption.lower()


def test_export_perimeter_ui_recap_fr_singular() -> None:
    """Une seule fiche : libellé au singulier."""
    caption, warning = export_perimeter_ui_recap_fr(1, "full_dataset")
    assert warning is None
    assert "1" in caption
    assert "fiche sera" in caption.lower()


def test_export_perimeter_ui_recap_fr_zero_validated_scope() -> None:
    """Périmètre validées vide : message explicite + conseil d'action."""
    caption, warning = export_perimeter_ui_recap_fr(0, "validated_only")
    assert "0" in caption
    assert warning is not None
    assert "fait et validé" in warning.lower()


def test_export_perimeter_ui_recap_fr_zero_full_dataset() -> None:
    """Cas limite : dataset complet mais 0 ligne (ex. df vide côté service)."""
    caption, warning = export_perimeter_ui_recap_fr(0, "full_dataset")
    assert warning is not None
    assert "0" in caption


def test_export_perimeter_ui_recap_fr_rejects_negative_count() -> None:
    """Le récap refuse un compte incohérent."""
    with pytest.raises(ValueError, match="non-negative"):
        export_perimeter_ui_recap_fr(-1, "validated_only")


def test_convert_to_jsonl_unknown_format_raises() -> None:
    """Un format non pris en charge lève ValueError."""
    df = pd.DataFrame([_minimal_valid_row()])
    with pytest.raises(ValueError, match="inconnu"):
        convert_to_jsonl(df, "unknown_format")  # type: ignore[arg-type]
