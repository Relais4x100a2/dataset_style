"""
Spike FastAPI issue-006 : lecture / mutation via ``src/database.py`` uniquement.

L'identité de l'acteur est injectée (tests, intégration) ; en production le BFF
résoudra l'utilisateur depuis la session SuperTokens (voir ADR ``docs/adr/``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from src.api_errors import (
    AuthSessionExpiredError,
    TenantResourceOpaqueDenial,
    error_envelope_for_client,
    log_resolved_api_error,
    resolve_exception_for_api,
)
from src.database import (
    ProjectSettings,
    get_project_settings,
    load_project_entries,
    require_admin,
    update_project_settings,
)

logger = logging.getLogger(__name__)


class SpikeProjectSettingsPatch(BaseModel):
    """Champs partiels acceptés pour la mutation (aucun champ requis)."""

    model_config = ConfigDict(extra="forbid")

    llm_base_url: str | None = Field(default=None)
    llm_model: str | None = Field(default=None)
    llm_api_key: str | None = Field(default=None)
    llm_timeout_seconds: str | None = Field(default=None)
    languagetool_base_url: str | None = Field(default=None)
    active_preset_key: str | None = Field(default=None)
    custom_presets_json: str | None = Field(default=None)
    dimensions_override_json: str | None = Field(default=None)


def _merge_settings(current: ProjectSettings, patch: SpikeProjectSettingsPatch) -> ProjectSettings:
    """Fusionne les champs non nuls du patch sur l'état courant."""
    updates = patch.model_dump(exclude_none=True)
    return replace(current, **updates)


def _settings_payload(settings: ProjectSettings) -> dict[str, Any]:
    """Représentation JSON stable (pas de logique métier hors ``database``)."""
    return jsonable_encoder(asdict(settings))


def create_spike_bff_app(
    engine: Engine,
    *,
    actor_user_id_factory: Callable[[], str],
) -> FastAPI:
    """Construit l'application FastAPI du spike (issue-006).

    Args:
        engine: Moteur SQLAlchemy partagé avec le reste de l'app.
        actor_user_id_factory: Retourne l'identifiant applicatif ``users.id`` ;
            en tests on injecte une lambda fixe ; en prod, résolution session.
    """

    def get_actor_user_id() -> str:
        return actor_user_id_factory()

    app = FastAPI(
        title="Dataset Style — BFF spike issue-006",
        version="0.0.1",
        docs_url="/issue-006-spike/docs",
        redoc_url=None,
    )

    @app.exception_handler(TenantResourceOpaqueDenial)
    async def _opaque_denial(_request: object, exc: TenantResourceOpaqueDenial) -> JSONResponse:
        log_resolved_api_error(logger, exc, extra_context={"route": "bff_spike"})
        resolved = resolve_exception_for_api(exc, include_technical_detail=False)
        return JSONResponse(
            status_code=resolved.http_status,
            content=error_envelope_for_client(exc, include_technical_detail=None),
        )

    @app.exception_handler(AuthSessionExpiredError)
    async def _auth_expired(_request: object, exc: AuthSessionExpiredError) -> JSONResponse:
        log_resolved_api_error(logger, exc, extra_context={"route": "bff_spike"})
        resolved = resolve_exception_for_api(exc, include_technical_detail=False)
        return JSONResponse(
            status_code=resolved.http_status,
            content=error_envelope_for_client(exc, include_technical_detail=None),
        )

    @app.exception_handler(OperationalError)
    async def _db_op_error(_request: object, exc: OperationalError) -> JSONResponse:
        log_resolved_api_error(logger, exc, extra_context={"route": "bff_spike"})
        resolved = resolve_exception_for_api(exc, include_technical_detail=False)
        return JSONResponse(
            status_code=resolved.http_status,
            content=error_envelope_for_client(exc, include_technical_detail=None),
        )

    @app.exception_handler(PermissionError)
    async def _permission(_request: object, exc: PermissionError) -> JSONResponse:
        log_resolved_api_error(logger, exc, extra_context={"route": "bff_spike"})
        resolved = resolve_exception_for_api(exc, include_technical_detail=False)
        return JSONResponse(
            status_code=resolved.http_status,
            content=error_envelope_for_client(exc, include_technical_detail=None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: object, exc: RequestValidationError) -> JSONResponse:
        wrapped = ValueError(str(exc))
        log_resolved_api_error(logger, wrapped, extra_context={"route": "bff_spike"})
        resolved = resolve_exception_for_api(wrapped, include_technical_detail=False)
        return JSONResponse(
            status_code=resolved.http_status,
            content=error_envelope_for_client(wrapped, include_technical_detail=None),
        )

    @app.get("/issue-006-spike/projects/{project_id}/entries")
    def list_entries(
        project_id: str,
        actor_user_id: str = Depends(get_actor_user_id),
    ) -> dict[str, Any]:
        """Liste les entrées du projet (``load_project_entries`` / ``require_role``)."""
        df = load_project_entries(engine, project_id, actor_user_id)
        return {"entries": jsonable_encoder(df.to_dict(orient="records"))}

    @app.patch("/issue-006-spike/projects/{project_id}/settings")
    def patch_project_settings(
        project_id: str,
        body: SpikeProjectSettingsPatch,
        actor_user_id: str = Depends(get_actor_user_id),
    ) -> dict[str, Any]:
        """Met à jour les réglages projet ; réponse = état canonique serveur post-écriture."""
        require_admin(engine, project_id, actor_user_id)
        current = get_project_settings(engine, project_id)
        merged = _merge_settings(current, body)
        update_project_settings(engine, project_id, merged)
        out = get_project_settings(engine, project_id)
        return {"settings": _settings_payload(out)}

    return app
