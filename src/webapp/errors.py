"""Erreurs HTTP avec enveloppe ``{"error": ...}`` (contrat issue-005)."""

from __future__ import annotations

from typing import Any


class EnvelopeHttpError(Exception):
    """Exception levée pour renvoyer une charge utile JSON homogène."""

    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = int(status_code)
        self.body = body
        super().__init__(str(status_code))
