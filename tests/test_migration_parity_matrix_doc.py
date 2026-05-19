"""Guardrails for the Streamlit → API migration parity matrix (issue-004)."""

from __future__ import annotations

from pathlib import Path

from src.tab_layout import main_tab_labels

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOC_PATH = _REPO_ROOT / "docs" / "migration_parity_matrix.md"


def test_migration_parity_matrix_file_exists() -> None:
    """The versioned parity matrix must be present for engineering handoff."""
    assert _DOC_PATH.is_file(), f"Expected {_DOC_PATH}"


def test_migration_parity_matrix_lists_all_product_tab_titles() -> None:
    """Titles must stay aligned with ``main_tab_labels`` / ``EXPECTED_WORKFLOW_TAB_ORDER``."""
    text = _DOC_PATH.read_text()
    for title in main_tab_labels(include_super_admin=True):
        assert title in text, f"Missing tab title {title!r} in migration parity doc"


def test_migration_parity_matrix_defines_sprint_status_columns() -> None:
    """Backlog issues 010–016 own per-row parity sign-off; headers must remain addressable."""
    text = _DOC_PATH.read_text()
    for n in range(10, 17):
        token = f"issue-{n:03d}"
        assert token in text, f"Missing sprint column marker {token}"


def test_migration_parity_matrix_documents_cache_invalidation_contract() -> None:
    """Post-write UI must clear the entries cache before reruns (issue-027)."""
    text = _DOC_PATH.read_text()
    assert "invalidate_project_entries_cache" in text


def test_migration_parity_matrix_has_manual_and_automated_checklists() -> None:
    text = _DOC_PATH.read_text()
    assert "## Checklist recette minimale" in text
    assert "## Jeux de non-régression automatisés" in text


def test_migration_parity_matrix_documents_ux_baseline_issue020() -> None:
    """issue-020 : baseline UX liée aux IDs flux de la matrice."""
    text = _DOC_PATH.read_text()
    assert "## Baseline UX (issue-020)" in text
    assert "docs/ux_baseline_issue_020.md" in text
    assert "DATASET_STYLE_UX_TELEMETRY_DIR" in text


def test_migration_parity_matrix_documents_issue007_vertical_slice() -> None:
    """issue-004 : marquage slice vertical + service webapp versionné."""
    text = _DOC_PATH.read_text()
    assert "## Slice vertical (issue-007" in text
    assert "webapp" in text.lower()
    assert "EXP-DL" in text
    assert "/api/projects/{id}/dashboard" in text or "/dashboard" in text
