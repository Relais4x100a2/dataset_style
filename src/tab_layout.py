"""Central Streamlit tab labels, ordered to match the curator workflow.

Issue-024 (validated order): projet → réglages → saisie → édition → tableau de bord,
then ``Mon compte``; optional ``Super Admin`` is appended for super-admins only.
"""

# Product workflow: issue-007 baseline, issue-024 wording (édition = Gestion & édition).
EXPECTED_WORKFLOW_TAB_ORDER: list[str] = [
    "Projets",
    "Réglages & Export",
    "Nouvelle entrée",
    "Gestion & édition",
    "Tableau de bord",
]


def main_tab_labels(*, include_super_admin: bool) -> list[str]:
    """Return tab titles for the main multipage strip, in display order.

    Args:
        include_super_admin: When True, append the super-admin-only tab.

    Returns:
        Ordered tab labels shown in ``st.tabs`` after a project is selected.
    """
    labels = [*EXPECTED_WORKFLOW_TAB_ORDER, "Mon compte"]
    if include_super_admin:
        labels.append("Super Admin")
    return labels
