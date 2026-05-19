"""API webapp : presets et dimensions projet (issue-011 / GitHub #133)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from src.database import ProjectSettings
from src.webapp import deps as webapp_deps
from src.webapp.app import create_slice_app


def test_get_dimensions_settings_requires_bearer() -> None:
    app = create_slice_app(engine=MagicMock())
    with TestClient(app) as client:
        r = client.get("/api/projects/p1/settings/dimensions")
    assert r.status_code == 401


def test_get_dimensions_settings_returns_presets_and_dimensions() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    settings = ProjectSettings(
        active_preset_key="roman",
        custom_presets_json="",
        dimensions_override_json="",
    )
    with (
        patch("src.webapp.project_dimensions_settings.load_project_entries") as load_m,
        patch("src.webapp.project_dimensions_settings.get_role", return_value="admin"),
        patch("src.webapp.project_dimensions_settings.get_project_settings", return_value=settings),
    ):
        with TestClient(app) as client:
            r = client.get(
                "/api/projects/p1/settings/dimensions",
                headers={"Authorization": "Bearer t"},
            )
    assert r.status_code == 200
    load_m.assert_called_once()
    body = r.json()
    assert body["activePresetKey"] == "roman"
    assert body["canEditDimensions"] is True
    assert isinstance(body["dimensions"], dict)
    assert any(p["key"] == "roman" for p in body["presets"])


def test_patch_load_preset_updates_storage() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    current = ProjectSettings(
        active_preset_key="contenu",
        custom_presets_json="",
        dimensions_override_json="",
    )
    captured: dict[str, ProjectSettings | None] = {"merged": None}

    def _get(_e: object, _pid: str) -> ProjectSettings:
        if captured["merged"] is not None:
            return captured["merged"]
        return current

    def _upd(_e: object, _pid: str, s: ProjectSettings) -> None:
        captured["merged"] = s

    with (
        patch("src.webapp.project_dimensions_settings.load_project_entries"),
        patch("src.webapp.project_dimensions_settings.require_admin"),
        patch("src.webapp.project_dimensions_settings.get_project_settings", side_effect=_get),
        patch("src.webapp.project_dimensions_settings.update_project_settings", side_effect=_upd),
        patch("src.webapp.project_dimensions_settings.get_role", return_value="admin"),
    ):
        with TestClient(app) as client:
            r = client.patch(
                "/api/projects/p1/settings/dimensions",
                headers={"Authorization": "Bearer t"},
                json={"action": "load_preset", "preset_key": "roman"},
            )
    assert r.status_code == 200
    assert captured["merged"] is not None
    assert captured["merged"].active_preset_key == "roman"
    assert captured["merged"].dimensions_override_json
    out = r.json()
    assert out["activePresetKey"] == "roman"


def test_patch_replace_dimensions_rejects_empty_statuts_fr() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    with (
        patch("src.webapp.project_dimensions_settings.load_project_entries"),
        patch("src.webapp.project_dimensions_settings.require_admin"),
        patch(
            "src.webapp.project_dimensions_settings.get_project_settings",
            return_value=ProjectSettings(),
        ),
        patch("src.webapp.project_dimensions_settings.update_project_settings") as upd,
    ):
        with TestClient(app) as client:
            r = client.patch(
                "/api/projects/p1/settings/dimensions",
                headers={"Authorization": "Bearer t"},
                json={
                    "action": "replace_dimensions",
                    "dimensions": {"types": ["A"], "statuts": []},
                },
            )
    assert r.status_code == 400
    assert "statuts" in r.json()["error"]["message"].lower()
    upd.assert_not_called()


def test_smoke_patch_preset_reflected_in_curator_dimensions() -> None:
    """Smoke issue-009 : PATCH profil puis cohérence ``GET .../curator/dimensions``."""
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    app.dependency_overrides[webapp_deps.require_app_user_id] = lambda: "u1"
    state = {
        "settings": ProjectSettings(
            active_preset_key="contenu",
            custom_presets_json="",
            dimensions_override_json="",
        )
    }

    def _get(_e: object, _pid: str) -> ProjectSettings:
        return state["settings"]

    def _upd(_e: object, _pid: str, s: ProjectSettings) -> None:
        state["settings"] = s

    with (
        patch("src.webapp.project_dimensions_settings.load_project_entries"),
        patch("src.webapp.project_dimensions_settings.require_admin"),
        patch("src.webapp.project_dimensions_settings.get_project_settings", side_effect=_get),
        patch("src.webapp.project_dimensions_settings.update_project_settings", side_effect=_upd),
        patch("src.webapp.project_dimensions_settings.get_role", return_value="admin"),
        patch("src.webapp.curator_ai.load_project_entries"),
        patch("src.webapp.curator_ai.get_project_settings", side_effect=_get),
    ):
        with TestClient(app) as client:
            r1 = client.get(
                "/api/projects/p1/curator/dimensions",
                headers={"Authorization": "Bearer t"},
            )
            assert r1.json()["activePresetKey"] == "contenu"
            r2 = client.patch(
                "/api/projects/p1/settings/dimensions",
                headers={"Authorization": "Bearer t"},
                json={"action": "load_preset", "preset_key": "roman"},
            )
            assert r2.status_code == 200
            r3 = client.get(
                "/api/projects/p1/curator/dimensions",
                headers={"Authorization": "Bearer t"},
            )
    assert r3.json()["activePresetKey"] == "roman"
    roman_statuts = json.loads(state["settings"].dimensions_override_json)["statuts"]
    assert "A faire" in roman_statuts
