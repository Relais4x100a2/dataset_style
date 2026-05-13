"""Tests for empty-project onboarding copy and feature flags (issue-008, issue-025)."""

import pytest
from src import empty_project_onboarding as ep
from src.tab_layout import EXPECTED_WORKFLOW_TAB_ORDER


def test_onboarding_steps_when_creation_allowed_has_three_items() -> None:
    steps = ep.onboarding_steps_when_creation_allowed()
    assert len(steps) == 3
    assert [s.index for s in steps] == [1, 2, 3]


def test_onboarding_steps_reference_workflow_tab_labels() -> None:
    """Narrative steps must match the real tab strip (issue-025 / architecture)."""
    steps = ep.onboarding_steps_when_creation_allowed()
    labels = "\n".join(s.body_markdown for s in steps)
    assert EXPECTED_WORKFLOW_TAB_ORDER[1] in labels
    assert EXPECTED_WORKFLOW_TAB_ORDER[2] in labels


def test_onboarding_step_one_points_to_projects_tab_for_first_project() -> None:
    """Issue-028: premier projet se crée dans l'onglet Projets, pas depuis la sidebar."""
    joined = ep.onboarding_steps_when_creation_allowed()[0].body_markdown
    assert "Projets" in joined
    assert "barre latérale" not in joined.lower()


def test_primary_submit_label_for_onboarding() -> None:
    assert ep.ONBOARDING_PRIMARY_SUBMIT_LABEL_FR == "Créer un projet"


def test_stylometric_value_sentence_is_french_and_non_empty() -> None:
    text = ep.STYLOMETRIC_VALUE_SENTENCE_FR.strip()
    assert len(text) >= 30
    assert "stylom" not in text.lower()


def test_product_rule_issue_11_separates_context_sidebar_and_actions_tab() -> None:
    """Issue-028 / issue-8: contexte (sidebar) vs actions (onglet Projets)."""
    rule = ep.PRODUCT_RULE_ISSUE_11_CREATION_PATHS_FR
    assert "contexte" in rule.lower()
    assert "action" in rule.lower() or "actions" in rule.lower()
    assert "latérale" in rule.lower() or "sidebar" in rule.lower()
    assert "projet" in rule.lower()


def test_creation_allowed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISABLE_SELF_SERVICE_PROJECT_CREATION", raising=False)
    assert ep.is_self_service_project_creation_allowed() is True


@pytest.mark.parametrize(
    "value",
    ["1", "true", "TRUE", "yes", "on"],
)
def test_creation_disabled_when_env_truthy(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DISABLE_SELF_SERVICE_PROJECT_CREATION", value)
    assert ep.is_self_service_project_creation_allowed() is False


def test_creation_disabled_message_is_non_trivial() -> None:
    assert len(ep.NO_PROJECT_CREATION_DISABLED_MESSAGE.strip()) >= 40


def test_main_onboarding_and_tab_projects_create_key_prefixes_differ() -> None:
    """Distinct Streamlit key roots for first-project vs in-studio project actions (issue-028)."""
    main_p = ep.ONBOARDING_MAIN_CREATE_FORM_KEY_PREFIX
    tab_p = ep.TAB_PROJECTS_ACTIONS_CREATE_FORM_KEY_PREFIX
    assert main_p != tab_p
