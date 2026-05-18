"""Tests unitaires pour ``projects_list_response`` (issue-010)."""

from __future__ import annotations

from src.database import ProjectRecord
from src.webapp.workspace_payload import projects_list_response


def test_projects_list_response_resolves_active_hint() -> None:
    projects = [
        ProjectRecord(project_id="a", name="A", role="admin"),
        ProjectRecord(project_id="b", name="B", role="admin"),
    ]
    out = projects_list_response(projects, "b")
    assert out["activeProjectId"] == "b"
    assert len(out["projects"]) == 2


def test_projects_list_response_empty_projects() -> None:
    out = projects_list_response([], "x")
    assert out["activeProjectId"] == ""
    assert out["projects"] == []
