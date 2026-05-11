"""Tests for project session resolution (onboarding vs active project)."""

from src.project_session import MembershipProject, resolve_active_project


def test_resolve_no_projects_empty_session_returns_empty() -> None:
    assert resolve_active_project("", []) == ("", "")


def test_resolve_no_projects_stale_session_returns_empty() -> None:
    assert resolve_active_project("p_old", []) == ("", "")


def test_resolve_empty_session_picks_first_ordered() -> None:
    projects = [
        MembershipProject("p2", "admin"),
        MembershipProject("p1", "admin"),
    ]
    assert resolve_active_project("", projects) == ("p2", "admin")


def test_resolve_whitespace_session_picks_first() -> None:
    projects = [MembershipProject("p1", "admin")]
    assert resolve_active_project("   ", projects) == ("p1", "admin")


def test_resolve_matching_session_returns_that_project() -> None:
    projects = [
        MembershipProject("p1", "admin"),
        MembershipProject("p2", "admin"),
    ]
    assert resolve_active_project("p2", projects) == ("p2", "admin")


def test_resolve_stale_id_falls_back_to_first() -> None:
    projects = [
        MembershipProject("p1", "admin"),
        MembershipProject("p2", "admin"),
    ]
    assert resolve_active_project("unknown", projects) == ("p1", "admin")
