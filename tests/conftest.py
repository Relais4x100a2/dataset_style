"""
Pytest fixtures partagés (moteur SQLite en mémoire pour les tests d'intégration DB).
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


@pytest.fixture()
def sqlite_engine() -> Engine:
    """Moteur SQLAlchemy SQLite en mémoire (API proche de PostgreSQL pour read_sql/to_sql)."""
    return create_engine("sqlite:///:memory:", future=True)
