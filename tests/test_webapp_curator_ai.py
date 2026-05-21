"""Tests aides curateur webapp (issue-013) : dimensions, LLM, LanguageTool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import requests
from fastapi.testclient import TestClient
from src.database import ProjectSettings
from src.webapp import deps as webapp_deps
from src.webapp.app import create_slice_app


def test_curator_dimensions_returns_active_preset_and_lists() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    dims = {
        "types": ["T1"],
        "structures": ["S1"],
        "tons": ["Ton1"],
        "formats": ["F1"],
        "publics": ["P1"],
        "statuts": ["En cours"],
    }
    with (
        patch("src.webapp.curator_ai.load_project_entries", return_value=pd.DataFrame()),
        patch("src.webapp.curator_ai.get_project_settings", return_value=ProjectSettings()),
        patch("src.webapp.curator_ai.load_active_dimensions", return_value=("roman", {}, dims)),
    ):
        with TestClient(app) as client:
            r = client.get(
                "/api/projects/p1/curator/dimensions",
                headers={"Authorization": "Bearer t"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["activePresetKey"] == "roman"
    assert body["dimensions"]["types"] == ["T1"]


def test_curator_llm_returns_ok_when_generate_succeeds() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    with (
        patch("src.webapp.curator_ai.require_role", return_value="admin"),
        patch(
            "src.webapp.curator_ai.get_project_settings",
            return_value=ProjectSettings(llm_api_key="k"),
        ),
        patch(
            "src.webapp.curator_ai.generate_output_from_input",
            return_value="prose générée",
        ) as gen_m,
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/projects/p1/curator/llm-generate",
                headers={"Authorization": "Bearer t", "Content-Type": "application/json"},
                json={
                    "mode": "draft_to_output",
                    "input": "notes",
                    "output": "",
                    "type": "A",
                    "structure": "B",
                    "ton": "C",
                    "format": "D",
                    "public": "E",
                },
            )
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "text": "prose générée"}
    gen_m.assert_called_once()


def test_curator_llm_returns_failed_message_when_generate_returns_none() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    with (
        patch("src.webapp.curator_ai.require_role", return_value="admin"),
        patch("src.webapp.curator_ai.get_project_settings", return_value=ProjectSettings()),
        patch("src.webapp.curator_ai.generate_output_from_input", return_value=None),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/projects/p1/curator/llm-generate",
                headers={"Authorization": "Bearer t", "Content-Type": "application/json"},
                json={
                    "mode": "draft_to_output",
                    "input": "x",
                    "type": "",
                    "structure": "",
                    "ton": "",
                    "format": "",
                    "public": "",
                },
            )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "failed"
    assert "Réglages projet" in data["message"]


def test_curator_llm_validation_error_when_draft_empty() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    with (
        patch("src.webapp.curator_ai.require_role", return_value="admin"),
        patch("src.webapp.curator_ai.get_project_settings", return_value=ProjectSettings()),
        patch("src.webapp.curator_ai.generate_output_from_input") as gen_m,
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/projects/p1/curator/llm-generate",
                headers={"Authorization": "Bearer t", "Content-Type": "application/json"},
                json={
                    "mode": "draft_to_output",
                    "input": "   ",
                    "type": "",
                    "structure": "",
                    "ton": "",
                    "format": "",
                    "public": "",
                },
            )
    assert r.status_code == 200
    assert r.json()["status"] == "validation_error"
    gen_m.assert_not_called()


def test_curator_languagetool_returns_corrected_and_matches() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    fake_matches = [{"offset": 0, "length": 4, "message": "m", "replacements": [{"value": "très"}]}]
    with (
        patch("src.webapp.curator_ai.require_role", return_value="admin"),
        patch("src.webapp.curator_ai.get_project_settings", return_value=ProjectSettings()),
        patch(
            "src.webapp.curator_ai.languagetool_fr_corrected_with_matches",
            return_value=("très bien", fake_matches),
        ),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/projects/p1/curator/languagetool-check",
                headers={"Authorization": "Bearer t", "Content-Type": "application/json"},
                json={"text": "tres bien"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["corrected"] == "très bien"
    assert body["matches"] == fake_matches


def test_curator_languagetool_validation_error_when_text_blank() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    with (
        patch("src.webapp.curator_ai.require_role", return_value="admin"),
        patch("src.webapp.curator_ai.get_project_settings", return_value=ProjectSettings()),
        patch("src.webapp.curator_ai.languagetool_fr_corrected_with_matches") as lt_m,
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/projects/p1/curator/languagetool-check",
                headers={"Authorization": "Bearer t", "Content-Type": "application/json"},
                json={"text": "   "},
            )
    assert r.status_code == 200
    lt_m.assert_not_called()
    data = r.json()
    assert data["status"] == "validation_error"
    assert "message" in data


def test_curator_languagetool_maps_timeout_to_envelope_503() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    with (
        patch("src.webapp.curator_ai.require_role", return_value="admin"),
        patch("src.webapp.curator_ai.get_project_settings", return_value=ProjectSettings()),
        patch(
            "src.webapp.curator_ai.languagetool_fr_corrected_with_matches",
            side_effect=requests.Timeout(),
        ),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/projects/p1/curator/languagetool-check",
                headers={"Authorization": "Bearer t", "Content-Type": "application/json"},
                json={"text": "x"},
            )
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "CURATOR_LANGUAGETOOL_UNAVAILABLE"
    assert "indisponible" in err["title"].lower()
