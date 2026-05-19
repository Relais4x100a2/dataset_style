"""Télémétrie UX webapp (issue-020) — mêmes ``milestone_code`` que Streamlit."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src import ux_scenario_telemetry as uxt
from src.database import STATUT_VALIDE, ProjectRecord
from src.webapp import deps as webapp_deps
from src.webapp import ux_telemetry
from src.webapp.app import create_slice_app


def test_webapp_sb_ctx_emitted_when_shell_init_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    fake_projects = [
        ProjectRecord(project_id="p1", name="Alpha", role="admin"),
    ]
    rid = "ux_" + "b" * 32
    with patch("src.webapp.app.list_projects_for_user", return_value=fake_projects):
        with TestClient(app) as client:
            r = client.get(
                "/api/projects?active_hint=p1",
                headers={
                    "Authorization": "Bearer t",
                    ux_telemetry.UX_RUN_ID_HEADER: rid,
                    ux_telemetry.UX_SHELL_INIT_HEADER: "1",
                },
            )
    assert r.status_code == 200
    files = list(tmp_path.glob("ux_scenario_*.jsonl"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert payload["milestone_code"] == "SB-CTX"
    assert payload["surface"] == "webapp"
    assert payload["run_id"] == rid


def test_webapp_export_emits_scope_then_dl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    rid = "ux_" + "c" * 32
    df = pd.DataFrame(
        [
            {
                "id": "e1",
                "project_id": "p1",
                "date": "",
                "type": "",
                "structure": "",
                "ton": "",
                "format": "",
                "public": "",
                "input": "a",
                "output": "b",
                "statut": STATUT_VALIDE,
                "notes": "",
            }
        ]
    )
    with patch("src.webapp.app.load_project_entries", return_value=df):
        with TestClient(app) as client:
            r = client.get(
                "/api/projects/p1/export.csv",
                headers={"Authorization": "Bearer t", ux_telemetry.UX_RUN_ID_HEADER: rid},
            )
    assert r.status_code == 200
    lines = (
        (tmp_path / next(tmp_path.glob("ux_scenario_*.jsonl")).name)
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(lines) == 2
    m0, m1 = (json.loads(x) for x in lines)
    assert m0["milestone_code"] == "EXP-SCOPE"
    assert m1["milestone_code"] == "EXP-DL"
    assert m1["extra"]["delivery"] == "csv"


def test_webapp_patch_edi_save_milestone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    rid = "ux_" + "d" * 32
    df = pd.DataFrame(
        [
            {
                "id": "e1",
                "project_id": "p1",
                "date": "",
                "type": "",
                "structure": "",
                "ton": "",
                "format": "",
                "public": "",
                "input": "a",
                "output": "b",
                "statut": "En cours",
                "notes": "",
            }
        ]
    )
    df_after = df.copy()
    with (
        patch("src.webapp.entry_mutations.load_project_entries", return_value=df),
        patch("src.webapp.entry_mutations.persist_edited_entry_with_nlp_cache"),
        patch("src.webapp.app.load_project_entries", return_value=df_after),
    ):
        with TestClient(app) as client:
            r = client.patch(
                "/api/projects/p1/entries/e1",
                headers={"Authorization": "Bearer t", ux_telemetry.UX_RUN_ID_HEADER: rid},
                json={"input": "x"},
            )
    assert r.status_code == 200
    path = next(tmp_path.glob("ux_scenario_*.jsonl"))
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["milestone_code"] == "EDI-SAVE"
