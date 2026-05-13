"""Textes et règles UX pour la navigation séquentielle de l'onglet édition (issue 032).

La confirmation « modifications non sauvegardées » est **systématique** avant tout
changement de fiche (boutons Précédent / Suivant ou liste « Entrée »), car les
champs d'édition vivent dans un ``st.form`` : Streamlit n'expose pas l'état
intermédiaire de tous les widgets avant « Sauvegarder », ce qui interdit une
détection fiable champ par champ sans refonte du formulaire.
"""

from __future__ import annotations

from typing import Literal


def edition_nav_boundary_caption_fr(
    direction: Literal["prev", "next"],
    *,
    can_navigate: bool,
) -> str | None:
    """Message court lorsque la navigation atteint une extrémité de la liste filtrée.

    Args:
        direction: Sens demandé.
        can_navigate: ``False`` si le bouton correspondant est désactivé.

    Returns:
        Texte d'aide en français, ou ``None`` si la navigation est possible.
    """
    if can_navigate:
        return None
    if direction == "prev":
        return "Première entrée de la liste filtrée — aucune fiche précédente."
    return "Dernière entrée de la liste filtrée — aucune fiche suivante."


def edition_nav_singleton_filtered_caption_fr(*, n_filtered: int) -> str | None:
    """Message lorsqu'un seul résultat correspond aux filtres (les deux boutons sont inactifs).

    Args:
        n_filtered: Nombre d'entrées dans ``df_pick``.

    Returns:
        Légende ou ``None`` si ``n_filtered != 1``.
    """
    if n_filtered != 1:
        return None
    return (
        "Une seule entrée correspond aux filtres — navigation précédente / suivante indisponible."
    )


def edition_nav_unsaved_changes_notice_fr() -> str:
    """Avertissement affiché dans la boîte de confirmation avant changement de fiche."""
    return (
        "Vous quittez la fiche affichée. Toute modification du formulaire **non "
        "enregistrée** via « Sauvegarder » sera perdue. "
        "(Confirmation systématique : les champs du formulaire ne sont pas tous "
        "inspectables avant enregistrement, limitation Streamlit `st.form`.)"
    )
