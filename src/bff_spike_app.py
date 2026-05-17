"""
Spike FastAPI minimal (issue-006) : BFF JSON sans dupliquer la logique métier hors de
``src/database.py``.

L'authentification est un **placeholder** de recette : en-tête ``X-Spike-Actor-User-Id``.
En production, ce point d'extension doit être remplacé par la vérification de session
SuperTokens (cookies HTTP-only, alignement ``APP_PUBLIC_BASE_URL`` — voir
``docs/streamlit_to_new_frontend_cutover.md``).
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
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
from src.presets import PRESETS

logger = logging.getLogger(__name__)

SPIKE_ACTOR_HEADER = "X-Spike-Actor-User-Id"


def _project_settings_public_dict(settings: ProjectSettings) -> dict[str, Any]:
    """Sérialise ``ProjectSettings`` pour la charge utile JSON (spike)."""
    return {
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "llm_api_key": settings.llm_api_key,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "languagetool_base_url": settings.languagetool_base_url,
        "active_preset_key": settings.active_preset_key,
        "custom_presets_json": settings.custom_presets_json,
        "dimensions_override_json": settings.dimensions_override_json,
    }


def _actor_user_id_from_header(raw: str | None) -> str:
    """Valide l'en-tête acteur ; lève ``AuthSessionExpiredError`` si absent."""
    uid = (raw or "").strip()
    if not uid:
        raise AuthSessionExpiredError()
    return uid


def create_bff_spike_app(*, engine: Engine) -> FastAPI:
    """Construit l'application FastAPI du spike, branchée sur ``engine``."""

    router = APIRouter(prefix="/migration-spike/v1", tags=["migration-spike"])

    @router.get("/projects/{project_id}/entries-summary")
    def get_entries_summary(
        project_id: str,
        x_spike_actor_user_id: Annotated[str | None, Header(alias=SPIKE_ACTOR_HEADER)] = None,
    ) -> dict[str, Any]:
        """Lecture : ``load_project_entries`` applique ``require_role`` (viewer+)."""
        actor_user_id = _actor_user_id_from_header(x_spike_actor_user_id)
        df = load_project_entries(engine, project_id, actor_user_id)
        return {
            "project_id": project_id,
            "row_count": int(len(df)),
            "entry_column_count": int(len(df.columns)),
        }

    class ActivePresetPatch(BaseModel):
        """Corps de mutation pour le preset actif."""

        active_preset_key: str = Field(..., min_length=1)

        @field_validator("active_preset_key")
        @classmethod
        def _preset_must_exist(cls, value: str) -> str:
            if value not in PRESETS:
                raise ValueError("Clé de preset inconnue.")
            return value

    @router.patch("/projects/{project_id}/settings/active-preset")
    def patch_active_preset(
        project_id: str,
        body: ActivePresetPatch,
        x_spike_actor_user_id: Annotated[str | None, Header(alias=SPIKE_ACTOR_HEADER)] = None,
    ) -> dict[str, Any]:
        """Mutation admin : ``require_admin`` puis persistance ; réponse = état relu en base."""
        actor_user_id = _actor_user_id_from_header(x_spike_actor_user_id)
        require_admin(engine, project_id, actor_user_id)
        current = get_project_settings(engine, project_id)
        merged = ProjectSettings(
            llm_base_url=current.llm_base_url,
            llm_model=current.llm_model,
            llm_api_key=current.llm_api_key,
            llm_timeout_seconds=current.llm_timeout_seconds,
            languagetool_base_url=current.languagetool_base_url,
            active_preset_key=body.active_preset_key,
            custom_presets_json=current.custom_presets_json,
            dimensions_override_json=current.dimensions_override_json,
        )
        update_project_settings(engine, project_id, merged)
        canonical = get_project_settings(engine, project_id)
        return {
            "project_id": project_id,
            "settings": _project_settings_public_dict(canonical),
        }

    app = FastAPI(title="Dataset Style — spike BFF migration", version="0.0.0")

    @app.exception_handler(AuthSessionExpiredError)
    async def _on_auth_expired(_request: Request, exc: AuthSessionExpiredError) -> JSONResponse:
        resolved = resolve_exception_for_api(exc, include_technical_detail=False)
        return JSONResponse(
            status_code=resolved.http_status,
            content=error_envelope_for_client(exc, include_technical_detail=None),
        )

    @app.exception_handler(TenantResourceOpaqueDenial)
    async def _on_opaque_denial(request: Request, exc: TenantResourceOpaqueDenial) -> JSONResponse:
        log_resolved_api_error(
            logger,
            exc,
            extra_context={"route": str(request.url.path)},
        )
        resolved = resolve_exception_for_api(exc, include_technical_detail=False)
        return JSONResponse(
            status_code=resolved.http_status,
            content=error_envelope_for_client(exc, include_technical_detail=None),
        )

    @app.exception_handler(OperationalError)
    async def _on_db_op(request: Request, exc: OperationalError) -> JSONResponse:
        log_resolved_api_error(logger, exc, extra_context={"route": str(request.url.path)})
        resolved = resolve_exception_for_api(exc, include_technical_detail=False)
        return JSONResponse(
            status_code=resolved.http_status,
            content=error_envelope_for_client(exc, include_technical_detail=None),
        )

    app.include_router(router)
    return app
