"""Régression issue-034 : messages de génération assistée sans jargon « LLM » côté curateur."""

from src import ui_components


def test_generation_failure_message_avoids_llm_label() -> None:
    """Le message d'échec de génération reste en français métier (pas « LLM » visible)."""
    msg = ui_components._GENERATION_FAILED_REVIEW_IA_SETTINGS_FR
    assert "LLM" not in msg
    assert "modèle d'ia" in msg.lower()
