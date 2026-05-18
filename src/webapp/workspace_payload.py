"""Sérialisation des listes projet pour le webapp (alignement ``project_session``)."""

from __future__ import annotations

from typing import Any

from src.database import ProjectRecord
from src.project_session import MembershipProject, resolve_active_project


def projects_list_response(
    projects: list[ProjectRecord], active_hint: str | None
) -> dict[str, Any]:
    """Construit la charge utile ``GET /api/projects`` avec résolution du projet actif.

    ``active_hint`` correspond à la préférence persistée côté client (équivalent
    ``st.session_state[\"project_id\"]``) ; la résolution suit ``resolve_active_project``.

    Args:
        projects: Projets visibles pour l'utilisateur (ordre DB).
        active_hint: Identifiant projet préféré ou chaîne vide / inconnue.

    Returns:
        Dictionnaire ``projects`` + ``activeProjectId`` (chaîne vide si aucun projet).
    """
    summaries = [MembershipProject(p.project_id, p.role) for p in projects]
    pid, _ = resolve_active_project((active_hint or "").strip(), summaries)
    return {
        "projects": [{"id": p.project_id, "name": p.name, "role": p.role} for p in projects],
        "activeProjectId": pid,
    }
