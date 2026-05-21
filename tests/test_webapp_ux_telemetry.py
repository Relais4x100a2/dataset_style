"""Télémétrie UX webapp (issue-020 / GitHub #182) — jalons alignés matrice."""

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


def test_webapp_sb_ctx_emitted_without_shell_init_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    ux_telemetry.reset_webapp_ux_dedupe_for_tests()
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
                },
            )
    assert r.status_code == 200
    assert r.headers.get(ux_telemetry.UX_TELEMETRY_ACTIVE_HEADER.lower()) == "1"
    assert r.headers.get(ux_telemetry.UX_RUN_ID_HEADER.lower()) == rid
    files = list(tmp_path.glob("ux_scenario_*.jsonl"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert payload["milestone_code"] == "SB-CTX"
    assert payload["surface"] == "webapp"
    assert payload["run_id"] == rid


def test_webapp_custom_scenario_id_in_sb_ctx_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    ux_telemetry.reset_webapp_ux_dedupe_for_tests()
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    fake_projects = [
        ProjectRecord(project_id="p1", name="Alpha", role="admin"),
    ]
    rid = "ux_" + "9" * 32
    scenario = "qa_baseline_panel_a"
    with patch("src.webapp.app.list_projects_for_user", return_value=fake_projects):
        with TestClient(app) as client:
            r = client.get(
                "/api/projects?active_hint=p1",
                headers={
                    "Authorization": "Bearer t",
                    ux_telemetry.UX_RUN_ID_HEADER: rid,
                    ux_telemetry.UX_SCENARIO_ID_HEADER: scenario,
                },
            )
    assert r.status_code == 200
    scenario_file = next(tmp_path.glob("ux_scenario_*.jsonl"))
    payload = json.loads(scenario_file.read_text(encoding="utf-8").strip())
    assert payload["scenario_id"] == scenario


def test_webapp_sb_ctx_deduped_on_repeated_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    ux_telemetry.reset_webapp_ux_dedupe_for_tests()
    app = create_slice_app(engine=MagicMock())
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    fake_projects = [
        ProjectRecord(project_id="p1", name="Alpha", role="admin"),
    ]
    rid = "ux_" + "a" * 32
    with patch("src.webapp.app.list_projects_for_user", return_value=fake_projects):
        with TestClient(app) as client:
            for _ in range(3):
                r = client.get(
                    "/api/projects",
                    params={"active_hint": "p1"},
                    headers={"Authorization": "Bearer t", ux_telemetry.UX_RUN_ID_HEADER: rid},
                )
                assert r.status_code == 200
    assert len(list(tmp_path.glob("ux_scenario_*.jsonl"))) == 1


def test_webapp_export_emits_only_exp_dl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    ux_telemetry.reset_webapp_ux_dedupe_for_tests()
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
    assert len(lines) == 1
    m0 = json.loads(lines[0])
    assert m0["milestone_code"] == "EXP-DL"
    assert m0["extra"]["delivery"] == "csv"


def test_webapp_repeated_identical_csv_export_emits_two_exp_dl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chaque GET réussi doit produire un jalon (relance export identique)."""
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    ux_telemetry.reset_webapp_ux_dedupe_for_tests()
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    rid = "ux_" + "7" * 32
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
    auth = {"Authorization": "Bearer t", ux_telemetry.UX_RUN_ID_HEADER: rid}
    with patch("src.webapp.app.load_project_entries", return_value=df):
        with TestClient(app) as client:
            for _ in range(2):
                r = client.get("/api/projects/p1/export.csv", headers=auth)
                assert r.status_code == 200
    milestones: list[str] = []
    for path in tmp_path.glob("ux_scenario_*.jsonl"):
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            milestones.append(json.loads(line)["milestone_code"])
    assert milestones.count("EXP-DL") == 2


def test_webapp_export_csv_then_jsonl_two_exp_dl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    ux_telemetry.reset_webapp_ux_dedupe_for_tests()
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    rid = "ux_" + "1" * 32
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
    auth = {"Authorization": "Bearer t", ux_telemetry.UX_RUN_ID_HEADER: rid}
    with patch("src.webapp.app.load_project_entries", return_value=df):
        with TestClient(app) as client:
            r1 = client.get(
                "/api/projects/p1/export.csv",
                params={"scope": "validated_only"},
                headers=auth,
            )
            r2 = client.get(
                "/api/projects/p1/export.jsonl",
                params={"scope": "validated_only", "format": "lfm2"},
                headers=auth,
            )
    assert r1.status_code == 200
    assert r2.status_code == 200
    milestones: list[str] = []
    for path in tmp_path.glob("ux_scenario_*.jsonl"):
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            milestones.append(json.loads(line)["milestone_code"])
    assert milestones.count("EXP-DL") == 2


def test_webapp_patch_edi_save_milestone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    ux_telemetry.reset_webapp_ux_dedupe_for_tests()
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


def test_webapp_post_ent_new_write_milestone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    ux_telemetry.reset_webapp_ux_dedupe_for_tests()
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    rid = "ux_" + "f" * 32
    df_after = pd.DataFrame(
        [
            {
                "id": "e_new",
                "project_id": "p1",
                "date": "",
                "type": "",
                "structure": "",
                "ton": "",
                "format": "",
                "public": "",
                "input": "nin",
                "output": "nout",
                "statut": STATUT_VALIDE,
                "notes": "",
            }
        ]
    )
    with (
        patch("src.webapp.entry_mutations.append_minimal_entry", return_value="e_new"),
        patch("src.webapp.app.load_project_entries", return_value=df_after),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/projects/p1/entries",
                headers={"Authorization": "Bearer t", ux_telemetry.UX_RUN_ID_HEADER: rid},
                json={"input": "nin", "output": "nout"},
            )
    assert r.status_code == 200
    path = next(tmp_path.glob("ux_scenario_*.jsonl"))
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["milestone_code"] == "ENT-NEW-WRITE"


def test_webapp_patch_unknown_entry_records_ux_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(uxt.UX_TELEMETRY_DIR_ENV, str(tmp_path))
    ux_telemetry.reset_webapp_ux_dedupe_for_tests()
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    rid = "ux_" + "e" * 32
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
    with patch("src.webapp.entry_mutations.load_project_entries", return_value=df):
        with TestClient(app) as client:
            r = client.patch(
                "/api/projects/p1/entries/unknown",
                headers={"Authorization": "Bearer t", ux_telemetry.UX_RUN_ID_HEADER: rid},
                json={"input": "x"},
            )
    assert r.status_code == 404
    err_files = list(tmp_path.glob("ux_error_*.jsonl"))
    assert len(err_files) == 1
    payload = json.loads(err_files[0].read_text(encoding="utf-8").strip())
    assert payload["kind"] == "ux_error"
    assert payload["milestone_context"] == "EDI-SAVE"
    assert payload["api_error_code"] == "NOT_FOUND_GENERIC"
