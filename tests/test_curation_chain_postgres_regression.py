"""Jalon non-régression chaîne curation avec persistance Postgres (issue-009, GitHub #183).

Les scénarios s'alignent sur les ID flux de ``docs/migration_parity_matrix.md`` (issue-004) :
PRJ-CREATE, ENT-NEW-WRITE, EDI-SAVE, EXP-SCOPE, EXP-DL. Aucun appel LLM ni calcul NLP
non maîtrisé — uniquement ``database`` + ``export_utils`` + routes export du slice webapp.

Requiert ``DATASET_STYLE_REGRESSION_DATABASE_URL`` (PostgreSQL). En CI la variable est
fournie par le workflow ; en local : démarrer Postgres puis exporter l'URL.
"""

from __future__ import annotations

import json
import os
from io import StringIO
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from src.database import (
    STATUT_VALIDE,
    create_db_engine,
    create_project,
    ensure_schema,
    load_project_entries,
    upsert_user_from_su,
)
from src.export_utils import (
    ExportScope,
    convert_to_jsonl,
    csv_text_from_export_dataframe,
    dataframe_for_export,
)
from src.webapp import deps as webapp_deps
from src.webapp import entry_mutations
from src.webapp.app import create_slice_app

_REGRESSION_DB_URL = (os.environ.get("DATASET_STYLE_REGRESSION_DATABASE_URL") or "").strip()

pytestmark = [
    pytest.mark.postgres_regression,
    pytest.mark.skipif(
        not _REGRESSION_DB_URL,
        reason=(
            "Définir DATASET_STYLE_REGRESSION_DATABASE_URL (PostgreSQL) pour les tests "
            "de persistance issue-009 — voir docs/migration_parity_matrix.md."
        ),
    ),
]


@pytest.fixture(scope="module")
def pg_engine() -> Engine:
    """Moteur Postgres partagé pour le module (schéma aligné prod via ``ensure_schema``)."""
    eng = create_db_engine(_REGRESSION_DB_URL)
    ensure_schema(eng)
    return eng


@pytest.fixture()
def fresh_owner_project(pg_engine: Engine) -> tuple[Engine, str, str]:
    """Propriétaire + projet vides pour isoler chaque test."""
    suffix = uuid4().hex[:12]
    user = upsert_user_from_su(
        pg_engine,
        f"su_curation_reg_{suffix}",
        f"curation_reg_{suffix}@example.invalid",
        "Curation regression",
    )
    project_id = create_project(pg_engine, user.user_id, f"regression-{suffix}")
    return pg_engine, user.user_id, project_id


def _webapp_csv_body(df: pd.DataFrame, scope: ExportScope) -> str:
    """Reproduit ``GET .../export.csv`` (colonnes publiques sans préfixe ``_``)."""
    export_df = dataframe_for_export(df, scope)
    return csv_text_from_export_dataframe(export_df)


def _webapp_jsonl_body(df: pd.DataFrame, scope: ExportScope) -> str:
    """Reproduit ``GET .../export.jsonl`` (LFM2, ``include_stylometry=True`` comme Streamlit)."""
    return convert_to_jsonl(df, "lfm2", include_stylometry=True, scope=scope)


def test_prj_create_ent_new_write_edi_save_reload_coherence(
    fresh_owner_project: tuple[Engine, str, str],
) -> None:
    """PRJ-CREATE + ENT-NEW-WRITE + EDI-SAVE : persistance et relecture cohérente."""
    engine, user_id, project_id = fresh_owner_project

    entry_id = entry_mutations.append_minimal_entry(
        engine,
        project_id,
        user_id,
        input_text="brouillon café",
        output_text="sortie initiale",
    )

    df_after_create = load_project_entries(engine, project_id, user_id)
    row = df_after_create.loc[df_after_create["id"] == entry_id].iloc[0]
    assert row["input"] == "brouillon café"
    assert row["output"] == "sortie initiale"

    entry_mutations.apply_entry_field_updates(
        engine,
        project_id,
        user_id,
        entry_id,
        {"output": "sortie corrigée", "statut": STATUT_VALIDE},
    )

    df_after_edit = load_project_entries(engine, project_id, user_id)
    row2 = df_after_edit.loc[df_after_edit["id"] == entry_id].iloc[0]
    assert row2["output"] == "sortie corrigée"
    assert row2["statut"] == STATUT_VALIDE


def test_exp_scope_validated_only_excludes_drafts(
    fresh_owner_project: tuple[Engine, str, str],
) -> None:
    """EXP-SCOPE : ``validated_only`` suit ``export_utils.dataframe_for_export``."""
    engine, user_id, project_id = fresh_owner_project

    draft_id = entry_mutations.append_minimal_entry(
        engine,
        project_id,
        user_id,
        input_text="x",
        output_text="y",
    )
    entry_mutations.apply_entry_field_updates(
        engine,
        project_id,
        user_id,
        draft_id,
        {"statut": "En cours"},
    )

    valid_id = entry_mutations.append_minimal_entry(
        engine,
        project_id,
        user_id,
        input_text="v1",
        output_text="v2",
    )
    entry_mutations.apply_entry_field_updates(
        engine,
        project_id,
        user_id,
        valid_id,
        {"statut": STATUT_VALIDE},
    )

    df = load_project_entries(engine, project_id, user_id)
    validated = dataframe_for_export(df, "validated_only")
    full = dataframe_for_export(df, "full_dataset")
    assert len(validated) == 1
    assert len(full) == 2
    assert validated.iloc[0]["id"] == valid_id


def test_exp_dl_csv_jsonl_matches_export_utils_and_http_slice(
    fresh_owner_project: tuple[Engine, str, str],
) -> None:
    """EXP-DL : CSV/JSONL identiques entre ``export_utils``, helper webapp et HTTP."""
    engine, user_id, project_id = fresh_owner_project

    entry_id = entry_mutations.append_minimal_entry(
        engine,
        project_id,
        user_id,
        input_text="entrée avec accents éè",
        output_text="réponse modèle",
    )
    entry_mutations.apply_entry_field_updates(
        engine,
        project_id,
        user_id,
        entry_id,
        {"statut": STATUT_VALIDE},
    )

    df = load_project_entries(engine, project_id, user_id)
    expected_csv = _webapp_csv_body(df, "validated_only")
    expected_jsonl = _webapp_jsonl_body(df, "validated_only")

    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: user_id
    with TestClient(app) as client:
        r_csv = client.get(
            f"/api/projects/{project_id}/export.csv",
            params={"scope": "validated_only"},
            headers={"Authorization": "Bearer integration-placeholder"},
        )
        r_jsonl = client.get(
            f"/api/projects/{project_id}/export.jsonl",
            params={"scope": "validated_only", "format": "lfm2"},
            headers={"Authorization": "Bearer integration-placeholder"},
        )

    assert r_csv.status_code == 200
    assert r_jsonl.status_code == 200
    assert r_csv.text == expected_csv
    assert r_jsonl.text == expected_jsonl

    export_df = dataframe_for_export(df, "validated_only")
    public_cols = [c for c in export_df.columns if not str(c).startswith("_")]
    assert all(not str(c).startswith("_") for c in public_cols)

    buf = StringIO(expected_csv)
    roundtrip = pd.read_csv(buf, dtype=str)
    assert roundtrip.iloc[0]["input"] == "entrée avec accents éè"
    assert roundtrip.iloc[0]["output"] == "réponse modèle"

    first_line = expected_jsonl.strip().splitlines()[0]
    payload = json.loads(first_line)
    assert "messages" in payload
