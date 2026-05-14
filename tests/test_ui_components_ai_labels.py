"""Régression issue-034 : messages de génération assistée sans jargon « LLM » côté curateur."""

from src import ui_components


def test_generation_failure_message_avoids_llm_label() -> None:
    """Le message d'échec de génération reste en français métier (pas « LLM » visible)."""
    msg = ui_components._GENERATION_FAILED_REVIEW_IA_SETTINGS_FR
    assert "LLM" not in msg
    assert "modèle d'ia" in msg.lower()


def test_project_settings_ui_mapping_matches_project_settings_fields() -> None:
    """Cartographie explicite : une entrée UI par attribut ProjectSettings du formulaire."""
    expected = frozenset(
        {
            "llm_base_url",
            "llm_model",
            "llm_api_key",
            "llm_timeout_seconds",
            "languagetool_base_url",
        }
    )
    assert frozenset(ui_components.PROJECT_SETTINGS_FIELD_UI_FR.keys()) == expected


def test_project_settings_ui_labels_are_french_business_surface() -> None:
    """Libellés visibles : français métier ; pas de surface anglaise héritée (issue-034)."""
    ui = ui_components.PROJECT_SETTINGS_FIELD_UI_FR
    assert ui["llm_model"].label == "Modèle d'IA"
    assert "LanguageTool base URL" not in ui["languagetool_base_url"].label
    assert "base url" not in ui["languagetool_base_url"].label.lower()
    for field in ui.values():
        assert "LLM" not in field.label


def test_project_settings_ui_help_mentions_env_fallbacks() -> None:
    """Infobulles : rappel technique des variables d'environnement (hors libellé)."""
    ui = ui_components.PROJECT_SETTINGS_FIELD_UI_FR
    assert "LLM_BASE_URL" in ui["llm_base_url"].help
    assert "LLM_MODEL" in ui["llm_model"].help
    assert "LLM_API_KEY" in ui["llm_api_key"].help
    assert "LLM_TIMEOUT_SECONDS" in ui["llm_timeout_seconds"].help
    assert "LANGUAGETOOL_BASE_URL" in ui["languagetool_base_url"].help
