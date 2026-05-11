"""
Resolve which project is active from session preference and DB-backed membership list.

Pure logic (no Streamlit) so onboarding vs normal flow can be tested without UI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MembershipProject:
    """One project the user can access, in list order (e.g. DB sort)."""

    project_id: str
    role: str


def resolve_active_project(
    session_project_id: str,
    projects: Sequence[MembershipProject],
) -> tuple[str, str]:
    """Pick the active ``project_id`` and role for the sidebar / main app.

    If the user has no projects, returns ``("", "")`` (onboarding).

    If ``session_project_id`` is empty, whitespace-only, or not in the list
    (stale tab, revoked access, etc.), the first project in ``projects`` is used
    so ``session_state`` can be normalized before widgets render.

    Args:
        session_project_id: Value from ``st.session_state["project_id"]`` or
            equivalent.
        projects: Non-mutable sequence of accessible projects, typically
            ordered like ``list_projects_for_user``.

    Returns:
        ``(project_id, role)`` or ``("", "")`` when there are no projects.
    """
    ordered = list(projects)
    if not ordered:
        return "", ""
    valid_ids = {p.project_id for p in ordered}
    pid = (session_project_id or "").strip()
    if pid in valid_ids:
        for p in ordered:
            if p.project_id == pid:
                return pid, p.role
    first = ordered[0]
    return first.project_id, first.role
