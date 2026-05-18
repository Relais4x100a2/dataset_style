"""Application FastAPI — slice vertical (issue-007) + shell curateur (issue-010)."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.engine import Engine

from src.api_errors import (
    TenantResourceOpaqueDenial,
    error_envelope_for_client,
    log_resolved_api_error,
)
from src.auth import persist_user_from_signin_ok
from src.database import (
    UserRecord,
    create_db_engine,
    create_project,
    delete_project_as_admin,
    list_projects_for_user,
    load_project_entries,
)
from src.export_utils import ExportFormat, ExportScope, convert_to_jsonl, dataframe_for_export
from src.nlp_engine import filter_edition_entries_dataframe
from src.services.edition_filters_service import build_edition_score_filter_spec
from src.services.project_dataframe_view import prepare_for_edition_tab
from src.supertokens_recipe_client import signin_email_password, try_revoke_access_token
from src.tab_layout import main_tab_labels
from src.webapp import deps as webapp_deps
from src.webapp import entry_mutations
from src.webapp.errors import EnvelopeHttpError
from src.webapp.index_template import INDEX_HTML as _INDEX_HTML
from src.webapp.workspace_payload import projects_list_response

logger = logging.getLogger(__name__)


def _serialize_entries_df(df: pd.DataFrame) -> list[Any]:
    """Sérialise les colonnes « publiques » ; exclut le cache NLP et champs internes (préfixe ``_``)."""
    cols = [c for c in df.columns if not str(c).startswith("_")]
    if not cols:
        return []
    return jsonable_encoder(df[cols].to_dict(orient="records"))


def _edition_filter_params_present(
    edition_statut: str | None,
    edition_score_mode: str | None,
    edition_score_threshold_lt: int | None,
    edition_score_bucket_decile: int | None,
    edition_score_include_na: bool | None,
) -> bool:
    """Vrai si au moins un paramètre de filtre édition a été fourni dans la query string."""
    return any(
        v is not None
        for v in (
            edition_statut,
            edition_score_mode,
            edition_score_threshold_lt,
            edition_score_bucket_decile,
            edition_score_include_na,
        )
    )


class SigninBody(BaseModel):
    """Corps de connexion slice vertical."""

    model_config = ConfigDict(extra="forbid")

    email: str
    password: str


class SignoutBody(BaseModel):
    """Corps optionnel pour révoquer explicitement le jeton courant."""

    model_config = ConfigDict(extra="forbid")

    access_token: str | None = None


class NewEntryBody(BaseModel):
    """Création minimale d'une fiche."""

    model_config = ConfigDict(extra="forbid")

    input: str = ""
    output: str = ""


class EntryPatchBody(BaseModel):
    """Champs partiels autorisés pour une fiche existante."""

    model_config = ConfigDict(extra="forbid")

    input: str | None = None
    output: str | None = None
    statut: str | None = None
    notes: str | None = None
    type: str | None = None
    structure: str | None = None
    ton: str | None = None
    format: str | None = None
    public: str | None = None
    date: str | None = None


class CreateProjectBody(BaseModel):
    """Création d'un projet propriétaire (``database.create_project``)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=500)
    description: str = Field(default="", max_length=10_000)

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("description", mode="before")
    @classmethod
    def _strip_description(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


def _cors_origins() -> list[str]:
    raw = (os.environ.get("WEBAPP_CORS_ORIGINS") or "http://localhost:8080").strip()
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_slice_app(*, engine: Engine | None = None) -> FastAPI:
    """Fabrique l'application ; ``engine`` injectable pour les tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if engine is not None:
            app.state.engine = engine
        else:
            url = (os.environ.get("DATABASE_URL") or "").strip()
            if not url:
                raise RuntimeError("DATABASE_URL requis pour le slice web.")
            app.state.engine = create_db_engine(url)
        yield

    app = FastAPI(title="Dataset Style — slice vertical", lifespan=lifespan)

    @app.get("/health", include_in_schema=False)
    async def http_health() -> dict[str, str]:
        """Liveness HTTP pour compose et CapRover (sans auth ni requête base).

        Returns:
            Statut minimal attendu par les sondes HTTP.
        """
        return {"status": "ok"}

    @app.exception_handler(EnvelopeHttpError)
    async def _envelope_handler(_request: Request, exc: EnvelopeHttpError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.body)

    @app.exception_handler(TenantResourceOpaqueDenial)
    async def _tenant_denied(_request: Request, exc: TenantResourceOpaqueDenial) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=error_envelope_for_client(exc, include_technical_detail=False),
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return _INDEX_HTML

    @app.post("/api/auth/signin", response_model=None)
    async def api_signin(request: Request, body: SigninBody) -> Any:
        eng = webapp_deps.get_engine(request)
        try:
            out = signin_email_password(body.email, body.password)
            if out.get("status") != "OK":
                raise ValueError(str(out.get("status")))
            record = persist_user_from_signin_ok(
                eng, out, submitted_email=body.email.strip().lower()
            )
            return {
                "accessToken": out.get("accessToken", ""),
                "user": {
                    "appUserId": record.user_id,
                    "email": record.email,
                    "displayName": record.display_name,
                },
            }
        except Exception as exc:  # noqa: BLE001
            log_resolved_api_error(logger, exc, extra_context={"route": "signin"})
            return JSONResponse(
                status_code=401,
                content=error_envelope_for_client(exc, include_technical_detail=None),
            )

    @app.post("/api/auth/signout")
    async def api_signout(
        body: SignoutBody | None = None,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    ) -> dict[str, str]:
        token = (body.access_token if body and body.access_token else None) or ""
        if not token and authorization and authorization.startswith("Bearer "):
            token = authorization.removeprefix("Bearer ").strip()
        if token:
            try_revoke_access_token(token)
        return {"status": "signed_out"}

    @app.get("/api/me")
    async def api_me(
        user: Annotated[UserRecord, Depends(webapp_deps.require_app_user)],
    ) -> dict[str, Any]:
        """Contexte utilisateur + ordre des onglets (``main_tab_labels`` / issue-010)."""
        labels = main_tab_labels(include_super_admin=bool(user.is_super_admin))
        return {
            "user": {
                "appUserId": user.user_id,
                "email": user.email,
                "displayName": user.display_name,
                "isSuperAdmin": bool(user.is_super_admin),
            },
            "mainTabLabels": labels,
        }

    @app.get("/api/projects")
    async def api_projects(
        request: Request,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
        active_hint: Annotated[
            str | None, Query(description="Préférence client projet actif")
        ] = None,
    ) -> dict[str, Any]:
        eng = webapp_deps.get_engine(request)
        projects = list_projects_for_user(eng, user_id)
        return projects_list_response(projects, active_hint)

    @app.post("/api/projects")
    async def api_create_project(
        request: Request,
        body: CreateProjectBody,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, str]:
        eng = webapp_deps.get_engine(request)
        pid = create_project(eng, user_id, body.name, body.description)
        return {"id": pid, "status": "ok"}

    @app.delete("/api/projects/{project_id}")
    async def api_delete_project(
        request: Request,
        project_id: str,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, str]:
        eng = webapp_deps.get_engine(request)
        delete_project_as_admin(eng, project_id, user_id)
        return {"status": "ok"}

    @app.get("/api/projects/{project_id}/entries")
    async def api_list_entries(
        request: Request,
        project_id: str,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
        edition_statut: Annotated[str | None, Query(alias="edition_statut")] = None,
        edition_score_mode: Annotated[str | None, Query(alias="edition_score_mode")] = None,
        edition_score_threshold_lt: Annotated[
            int | None, Query(alias="edition_score_threshold_lt", ge=0, le=100)
        ] = None,
        edition_score_bucket_decile: Annotated[
            int | None, Query(alias="edition_score_bucket_decile", ge=0, le=9)
        ] = None,
        edition_score_include_na: Annotated[
            bool | None, Query(alias="edition_score_include_na")
        ] = None,
    ) -> dict[str, Any]:
        """Liste les entrées ; paramètres ``edition_*`` optionnels = mêmes filtres que Streamlit."""
        eng = webapp_deps.get_engine(request)
        df = load_project_entries(eng, project_id, user_id)
        if _edition_filter_params_present(
            edition_statut,
            edition_score_mode,
            edition_score_threshold_lt,
            edition_score_bucket_decile,
            edition_score_include_na,
        ):
            statut_filter = (edition_statut or "").strip() or None
            mode = edition_score_mode or "all"
            try:
                score_spec = build_edition_score_filter_spec(
                    mode,
                    threshold_lt=edition_score_threshold_lt
                    if edition_score_threshold_lt is not None
                    else 50,
                    bucket_decile=edition_score_bucket_decile
                    if edition_score_bucket_decile is not None
                    else 0,
                    include_na=edition_score_include_na
                    if edition_score_include_na is not None
                    else False,
                )
            except ValueError as exc:
                raise EnvelopeHttpError(
                    400,
                    error_envelope_for_client(exc, include_technical_detail=False),
                ) from exc
            basis = prepare_for_edition_tab(df)
            df = filter_edition_entries_dataframe(
                basis,
                statut_label=statut_filter,
                score_spec=score_spec,
            )
        rows = _serialize_entries_df(df)
        return {"entries": rows}

    @app.patch("/api/projects/{project_id}/entries/{entry_id}")
    async def api_patch_entry(
        request: Request,
        project_id: str,
        entry_id: str,
        body: EntryPatchBody,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, Any]:
        eng = webapp_deps.get_engine(request)
        updates = {k: v for k, v in body.model_dump(exclude_none=True).items()}
        try:
            entry_mutations.apply_entry_field_updates(eng, project_id, user_id, entry_id, updates)
        except KeyError as exc:
            raise TenantResourceOpaqueDenial() from exc
        df = load_project_entries(eng, project_id, user_id)
        return {"status": "ok", "entries": _serialize_entries_df(df)}

    @app.post("/api/projects/{project_id}/entries")
    async def api_create_entry(
        request: Request,
        project_id: str,
        body: NewEntryBody,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, Any]:
        eng = webapp_deps.get_engine(request)
        new_id = entry_mutations.append_minimal_entry(
            eng,
            project_id,
            user_id,
            input_text=body.input,
            output_text=body.output,
        )
        df = load_project_entries(eng, project_id, user_id)
        return {"id": new_id, "status": "ok", "entries": _serialize_entries_df(df)}

    @app.get("/api/projects/{project_id}/export.csv")
    async def api_export_csv(
        request: Request,
        project_id: str,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
        scope: ExportScope = "validated_only",
    ) -> Response:
        eng = webapp_deps.get_engine(request)
        df = load_project_entries(eng, project_id, user_id)
        export_df = dataframe_for_export(df, scope)
        cols = [c for c in export_df.columns if not str(c).startswith("_")]
        text = export_df[cols].to_csv(index=False)
        return PlainTextResponse(
            content=text,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="export-{project_id}.csv"'},
        )

    @app.get("/api/projects/{project_id}/export.jsonl")
    async def api_export_jsonl(
        request: Request,
        project_id: str,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
        scope: ExportScope = "validated_only",
        export_format: Annotated[ExportFormat, Query(alias="format")] = "lfm2",
    ) -> Response:
        eng = webapp_deps.get_engine(request)
        df = load_project_entries(eng, project_id, user_id)
        payload = convert_to_jsonl(df, format=export_format, include_stylometry=False, scope=scope)
        return PlainTextResponse(
            content=payload,
            media_type="application/x-ndjson; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="export-{project_id}.jsonl"'},
        )

    return app


app = create_slice_app()
