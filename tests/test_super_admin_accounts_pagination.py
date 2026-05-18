"""Pagination annuaire super-admin (issue-018) — validation bornes page / taille."""

from __future__ import annotations

import math

import pytest
from src.database import (
    SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MAX,
    SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MIN,
    validate_super_admin_accounts_list_params,
)


def test_page_size_below_min_is_rejected() -> None:
    with pytest.raises(ValueError, match="10"):
        validate_super_admin_accounts_list_params(
            page=1, page_size=SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MIN - 1, total_active_accounts=0
        )


def test_page_size_above_max_is_rejected() -> None:
    with pytest.raises(ValueError, match="100"):
        validate_super_admin_accounts_list_params(
            page=1, page_size=SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MAX + 1, total_active_accounts=5
        )


def test_page_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="page"):
        validate_super_admin_accounts_list_params(page=0, page_size=25, total_active_accounts=10)


def test_page_above_total_pages_is_rejected_when_accounts_exist() -> None:
    total = 25
    page_size = 10
    max_page = math.ceil(total / page_size)
    with pytest.raises(ValueError, match="page"):
        validate_super_admin_accounts_list_params(
            page=max_page + 1, page_size=page_size, total_active_accounts=total
        )


def test_page_above_one_rejected_when_no_accounts() -> None:
    with pytest.raises(ValueError, match="page"):
        validate_super_admin_accounts_list_params(page=2, page_size=10, total_active_accounts=0)


def test_valid_params_return_normalized_ints() -> None:
    p, s = validate_super_admin_accounts_list_params(
        page=2, page_size=50, total_active_accounts=120
    )
    assert p == 2 and s == 50


def test_boundary_page_size_accepted() -> None:
    p1, s1 = validate_super_admin_accounts_list_params(
        page=1, page_size=SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MIN, total_active_accounts=1
    )
    assert s1 == SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MIN
    p2, s2 = validate_super_admin_accounts_list_params(
        page=1, page_size=SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MAX, total_active_accounts=1
    )
    assert s2 == SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MAX
    assert p1 == 1 and p2 == 1
