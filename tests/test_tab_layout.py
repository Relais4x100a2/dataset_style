"""Tests for central tab label ordering (workflow-aligned)."""

from src.tab_layout import EXPECTED_WORKFLOW_TAB_ORDER, main_tab_labels


def test_main_tab_labels_follows_product_workflow_order() -> None:
    """Order: projet → réglages → saisie → édition → tableau de bord, puis compte."""
    labels = main_tab_labels(include_super_admin=False)
    workflow = labels[: len(EXPECTED_WORKFLOW_TAB_ORDER)]
    assert workflow == EXPECTED_WORKFLOW_TAB_ORDER
    assert labels == [*EXPECTED_WORKFLOW_TAB_ORDER, "Mon compte"]


def test_issue_024_slot_indices_dashboard_then_account() -> None:
    """Regression guard: compte stays after pilotage; édition before dashboard (issue-024)."""
    labels = main_tab_labels(include_super_admin=False)
    assert labels.index("Gestion & édition") < labels.index("Tableau de bord")
    assert labels.index("Tableau de bord") < labels.index("Mon compte")
    assert labels == [
        "Projets",
        "Réglages & Export",
        "Nouvelle entrée",
        "Gestion & édition",
        "Tableau de bord",
        "Mon compte",
    ]


def test_main_tab_labels_appends_super_admin_when_requested() -> None:
    labels = main_tab_labels(include_super_admin=True)
    assert labels == [*EXPECTED_WORKFLOW_TAB_ORDER, "Mon compte", "Super Admin"]
