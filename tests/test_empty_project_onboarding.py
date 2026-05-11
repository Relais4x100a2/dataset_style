"""Tests for empty-project onboarding copy and feature flags (issue-008)."""

import pytest
from src import empty_project_onboarding as ep


def test_onboarding_steps_when_creation_allowed_has_three_items() -> None:
    steps = ep.onboarding_steps_when_creation_allowed()
    assert len(steps) == 3
    assert [s.index for s in steps] == [1, 2, 3]


def test_onboarding_steps_include_sidebar_and_create_cta() -> None:
    joined = "\n".join(s.body_markdown for s in ep.onboarding_steps_when_creation_allowed())
    assert "☰" in joined or "menu" in joined.lower()
    assert "Créer" in joined


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


def test_main_and_sidebar_create_key_prefixes_differ() -> None:
    main_p = ep.ONBOARDING_MAIN_CREATE_FORM_KEY_PREFIX
    sb_p = ep.SIDEBAR_FIRST_PROJECT_CREATE_FORM_KEY_PREFIX
    assert main_p != sb_p
