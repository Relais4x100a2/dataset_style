"""Unit tests for Super Admin French copy (issue-012)."""

from src.super_admin_ui_texts import (
    SAGA_STATE_METRIC_LABELS,
    SUPER_ADMIN_ACCOUNTS_SECTION_TITLE,
    SUPER_ADMIN_ACTIONS_SECTION_TITLE,
    SUPER_ADMIN_DLQ_SECTION_TITLE,
    SUPER_ADMIN_INVITE_SECTION_TITLE,
    SUPER_ADMIN_SAGA_SECTION_TITLE,
    SUPER_ADMIN_TECH_EXPANDER_CAPTION,
    SUPER_ADMIN_TECH_EXPANDER_TITLE,
    button_detach_memberships,
    button_replay_quarantined,
    saga_metric_label,
    selectbox_dlq_operation,
    selectbox_target_account,
    super_admin_tab_labels,
)


def test_technical_expander_title_signals_secondary_technical_area() -> None:
    assert "Suivi technique" in SUPER_ADMIN_TECH_EXPANDER_TITLE
    assert len(SUPER_ADMIN_TECH_EXPANDER_TITLE) >= 20


def test_technical_expander_caption_mentions_technical_ids() -> None:
    lowered = SUPER_ADMIN_TECH_EXPANDER_CAPTION.lower()
    assert "opération" in lowered or "identifiant" in lowered


def test_primary_section_titles_are_french_and_actionable() -> None:
    assert "Inviter" in SUPER_ADMIN_INVITE_SECTION_TITLE
    assert "compte" in SUPER_ADMIN_ACCOUNTS_SECTION_TITLE.lower()
    assert "compte" in SUPER_ADMIN_ACTIONS_SECTION_TITLE.lower()


def test_saga_state_metric_labels_cover_known_states() -> None:
    for key in ("pending", "provider_done", "failed", "quarantined"):
        assert key in SAGA_STATE_METRIC_LABELS
        assert len(SAGA_STATE_METRIC_LABELS[key]) >= 3


def test_selectbox_dlq_operation_label_avoids_raw_operation_id_in_title() -> None:
    label = selectbox_dlq_operation()
    assert "operation_id" not in label.lower()
    assert len(label) >= 8


def test_button_labels_are_french() -> None:
    assert button_detach_memberships().startswith("Retirer")
    assert "Relancer" in button_replay_quarantined()


def test_secondary_subsection_titles_are_french() -> None:
    assert "traitement" in SUPER_ADMIN_SAGA_SECTION_TITLE.lower()
    assert "bloqu" in SUPER_ADMIN_DLQ_SECTION_TITLE.lower()


def test_target_account_selectbox_uses_business_language() -> None:
    assert "Compte" in selectbox_target_account()


def test_super_admin_tab_labels_accounts_tab_first_and_distinct() -> None:
    accounts_tab, tech_tab = super_admin_tab_labels()
    lowered_accounts = accounts_tab.lower()
    assert "invitation" in lowered_accounts or "compte" in lowered_accounts
    assert "technique" in tech_tab.lower() or "suivi" in tech_tab.lower()
    assert accounts_tab != tech_tab


def test_saga_metric_label_fallback_returns_raw_state() -> None:
    assert saga_metric_label("unknown_custom_state") == "unknown_custom_state"
