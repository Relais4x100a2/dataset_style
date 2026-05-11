"""Tests for central tab label ordering (workflow-aligned)."""

from src.tab_layout import EXPECTED_WORKFLOW_TAB_ORDER, main_tab_labels


def test_main_tab_labels_follows_product_workflow_order() -> None:
    """Order: projet → réglages → saisie → révision → tableau de bord, puis compte."""
    labels = main_tab_labels(include_super_admin=False)
    workflow = labels[: len(EXPECTED_WORKFLOW_TAB_ORDER)]
    assert workflow == EXPECTED_WORKFLOW_TAB_ORDER
    assert labels == [*EXPECTED_WORKFLOW_TAB_ORDER, "Mon compte"]


def test_main_tab_labels_appends_super_admin_when_requested() -> None:
    labels = main_tab_labels(include_super_admin=True)
    assert labels == [*EXPECTED_WORKFLOW_TAB_ORDER, "Mon compte", "Super Admin"]
