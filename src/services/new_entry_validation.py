"""Validation métier partagée Streamlit / webapp pour la création d'entrée."""

from __future__ import annotations


def new_entry_missing_required_body_message(input_text: str, output_text: str) -> str | None:
    """Vérifie que les deux champs corps sont non vides avant persistance.

    Args:
        input_text: Contenu brouillon.
        output_text: Contenu texte généré / sortie.

    Returns:
        Message d'erreur court en français si la validation échoue, sinon ``None``.
    """
    if not str(input_text).strip() or not str(output_text).strip():
        return "Brouillon/Texte généré obligatoires."
    return None
