"""Unit tests for Super Admin French copy (issue-012, issue-029)."""

from src.super_admin_ui_texts import (
    SAGA_STATE_METRIC_LABELS,
    SUPER_ADMIN_ACCOUNT_MANAGEMENT_HUB_TITLE,
    SUPER_ADMIN_ACCOUNTS_SECTION_TITLE,
    SUPER_ADMIN_ACTIONS_SECTION_TITLE,
    SUPER_ADMIN_DLQ_SECTION_TITLE,
    SUPER_ADMIN_INVITE_SECTION_TITLE,
    SUPER_ADMIN_SAGA_SECTION_TITLE,
    SUPER_ADMIN_TECH_EXPANDER_CAPTION,
    SUPER_ADMIN_TECH_EXPANDER_TITLE,
    SUPER_ADMIN_WORKFLOW_HINT,
    button_detach_memberships,
    button_replay_quarantined,
    error_delete_target_account_failed,
    error_detach_memberships_failed,
    flash_memberships_detached,
    saga_metric_label,
    selectbox_dlq_operation,
    selectbox_target_account,
    super_admin_accounts_table_column_labels,
    super_admin_tab_labels,
    super_admin_warning_detach_memberships,
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
    assert "gestion" in lowered_accounts and "compte" in lowered_accounts
    assert "invitation" in lowered_accounts or "action" in lowered_accounts
    assert "technique" in tech_tab.lower() or "suivi" in tech_tab.lower()
    assert accounts_tab != tech_tab


def test_account_management_hub_title_matches_product_section() -> None:
    assert SUPER_ADMIN_ACCOUNT_MANAGEMENT_HUB_TITLE.lower() == "gestion des comptes"


def test_workflow_hint_mentions_technical_tab_without_claiming_lazy_load() -> None:
    lowered = SUPER_ADMIN_WORKFLOW_HINT.lower()
    assert "invitation" in lowered or "action" in lowered
    assert "technique" in lowered
    assert "onglet" in lowered


def test_accounts_table_column_labels_cover_dataframe_keys() -> None:
    labels = super_admin_accounts_table_column_labels()
    expected_keys = (
        "user_id",
        "nom_affichage",
        "email",
        "super_admin",
        "nb_projets",
        "derniere_connexion",
        "entrees_total",
        "entrees_validees",
    )
    for key in expected_keys:
        assert key in labels
        assert len(labels[key]) >= 3


def test_detach_warning_avoids_english_membership_jargon() -> None:
    text = super_admin_warning_detach_memberships(membership_count=2, email="a@b.c")
    lowered = text.lower()
    assert "membership" not in lowered
    assert "accès collaborateur" in lowered


def test_flash_and_error_messages_are_french() -> None:
    assert "collaboration" in flash_memberships_detached(1).lower()
    assert error_detach_memberships_failed().lower().startswith("retrait")
    assert "suppression" in error_delete_target_account_failed().lower()


def test_saga_metric_label_fallback_returns_raw_state() -> None:
    assert saga_metric_label("unknown_custom_state") == "unknown_custom_state"
