"""Contract léger sur la doc issue-003 (modèle d'accès projet)."""

from __future__ import annotations

from pathlib import Path


def _doc_path() -> Path:
    """Chemin absolu vers ``docs/architecture/project_access_model.md``."""
    return (
        Path(__file__).resolve().parent.parent / "docs" / "architecture" / "project_access_model.md"
    )


def test_project_access_architecture_doc_exists_and_anchors() -> None:
    """Évite la régression silencieuse du document de décision issue-003."""
    path = _doc_path()
    assert path.is_file(), f"missing architecture doc: {path}"
    text = path.read_text(encoding="utf-8")
    assert "projects.created_by" in text
    assert "project_memberships" in text
    assert "get_role" in text
    assert "issue-010" in text
