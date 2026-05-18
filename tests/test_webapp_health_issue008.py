"""Healthcheck HTTP dédié FastAPI (issue-008) — symétrie compose / CapRover."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from src.webapp.app import create_slice_app


def test_health_returns_ok_without_auth() -> None:
    """Le healthcheck ne doit pas exiger d'en-tête Authorization."""
    app = create_slice_app(engine=MagicMock())
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
