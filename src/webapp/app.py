"""Application FastAPI — slice vertical (issue-007) + shell curateur (issue-010)."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
import requests
from fastapi import Body, Depends, FastAPI, Header, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from src.api_errors import (
    ExportPayloadTooLargeError,
    TenantResourceOpaqueDenial,
    error_envelope_for_client,
    log_resolved_api_error,
)
from src.auth import persist_user_from_signin_ok, verify_invitation_only_contract
from src.database import (
    STATUT_VALIDE,
    SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MAX,
    SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MIN,
    UserRecord,
    count_active_memberships,
    count_owned_projects,
    count_users_for_admin,
    create_db_engine,
    create_project,
    delete_project_as_admin,
    get_user_email_display_name_by_id,
    get_user_ui_preferences_raw,
    list_accounts_for_super_admin,
    list_projects_for_user,
    load_project_entries,
    replay_quarantined_operation,
    update_user_ui_preferences_raw,
    validate_super_admin_accounts_list_params,
)
from src.export_utils import ExportFormat, ExportScope, convert_to_jsonl, dataframe_for_export
from src.migration_communication import (
    INDEX_HTML_BANNER_PLACEHOLDER,
    migration_info_banner_html_fragment,
)
from src.nlp_engine import filter_edition_entries_dataframe
from src.services.curator_dashboard_snapshot import (
    DashboardStylometryScope,
    build_curator_dashboard_envelope,
)
from src.services.edition_filters_service import build_edition_score_filter_spec
from src.services.project_dataframe_view import prepare_for_edition_tab
from src.supertokens_recipe_client import signin_email_password, try_revoke_access_token
from src.tab_layout import main_tab_labels
from src.ui_preferences import (
    load_from_stored_raw,
    merge_patch_into_canonical,
    serialize_canonical_preferences,
)
from src.webapp import curator_ai, entry_mutations
from src.webapp import deps as webapp_deps
from src.webapp.errors import EnvelopeHttpError
from src.webapp.index_template import INDEX_HTML as _INDEX_HTML
from src.webapp.super_admin_invite import invite_collaborator_by_email
from src.webapp.super_admin_saga import build_deprovision_telemetry_payload
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
    redirect_after: str | None = None


class AccountUiPreferencesPatchBody(BaseModel):
    """Mise à jour partielle des préférences d'affichage (issue-023 / #145)."""

    model_config = ConfigDict(extra="forbid")

    density: Literal["default", "compact", "comfortable"] | None = None
    readingComfort: Literal["default", "high_contrast", "reduced_motion"] | None = None


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


class CuratorLlmBody(BaseModel):
    """Paramètres génération IA (alignés sur l'onglet Nouvelle entrée Streamlit)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["draft_to_output", "output_to_draft"]
    input: str = ""
    output: str = ""
    type: str = ""
    structure: str = ""
    ton: str = ""
    format: str = ""
    public: str = ""


class CuratorLanguageToolBody(BaseModel):
    """Texte output (ou extrait) à contrôler via LanguageTool."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""


class SuperAdminInviteBody(BaseModel):
    """Corps d'invitation super-admin (e-mail collaborateur)."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(..., min_length=1, max_length=320)

    @field_validator("email", mode="before")
    @classmethod
    def _strip_email(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        em = value.lower()
        if "@" not in em:
            raise ValueError("Adresse e-mail invalide.")
        return em


class SuperAdminSagaReplayBody(BaseModel):
    """Relance d'une opération saga en quarantaine : ``confirm: true`` obligatoire (issue-019)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    confirm: bool
    operation_id: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("operationId", "operation_id"),
    )

    @field_validator("confirm")
    @classmethod
    def _must_confirm(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("La relance nécessite confirm: true.")
        return value


def _cors_origins() -> list[str]:
    raw = (os.environ.get("WEBAPP_CORS_ORIGINS") or "http://localhost:8080").strip()
    return [o.strip() for o in raw.split(",") if o.strip()]


def _webapp_export_max_rows() -> int | None:
    """Limite optionnelle (nombre de fiches exportables) via ``WEBAPP_EXPORT_MAX_ROWS``."""
    raw = (os.environ.get("WEBAPP_EXPORT_MAX_ROWS") or "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        logger.warning("WEBAPP_EXPORT_MAX_ROWS ignoré (entier attendu): %s", raw)
        return None
    if n <= 0:
        return None
    return n


def _enforce_export_row_cap(export_df: pd.DataFrame) -> None:
    """Lève :exc:`ExportPayloadTooLargeError` si un plafond métier est configuré et dépassé."""
    cap = _webapp_export_max_rows()
    if cap is None:
        return
    n = len(export_df.index)
    if n > cap:
        raise ExportPayloadTooLargeError(row_count=n, max_rows=cap)


def _signout_redirect_allowlist() -> list[str]:
    """Chemins ou URLs autorisés après déconnexion (``WEBAPP_SIGNOUT_REDIRECT_ALLOWLIST``)."""
    raw = (os.environ.get("WEBAPP_SIGNOUT_REDIRECT_ALLOWLIST") or "/").strip()
    entries = [p.strip() for p in raw.split(",") if p.strip()]
    return entries if entries else ["/"]


def approved_post_signout_redirect(requested: str | None) -> str:
    """Retourne une cible allow-listée ; refuse les redirections non listées ou avec schéma arbitraire."""
    allow = _signout_redirect_allowlist()
    default = allow[0]
    if not requested or not requested.strip():
        return default
    candidate = requested.strip()
    if candidate.startswith("//") or "://" in candidate:
        return default
    return candidate if candidate in allow else default


def _bad_request_client_envelope(message: str) -> dict[str, Any]:
    """Charge utile JSON 400 alignée sur les autres routes ``webapp`` (code ``BAD_REQUEST``)."""
    return {
        "error": {
            "code": "BAD_REQUEST",
            "title": "Requête invalide",
            "message": message,
            "suggested_action": "Corrigez le corps JSON puis réessayez.",
            "detail": None,
        }
    }


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
        verify_invitation_only_contract()
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

    _static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        frag = migration_info_banner_html_fragment()
        if frag and INDEX_HTML_BANNER_PLACEHOLDER in _INDEX_HTML:
            return _INDEX_HTML.replace(INDEX_HTML_BANNER_PLACEHOLDER, frag, 1)
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
        body: Annotated[SignoutBody | None, Body()] = None,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    ) -> dict[str, str]:
        token = (body.access_token if body and body.access_token else None) or ""
        if not token and authorization and authorization.startswith("Bearer "):
            token = authorization.removeprefix("Bearer ").strip()
        if token:
            try_revoke_access_token(token)
        redirect = approved_post_signout_redirect(body.redirect_after if body else None)
        return {"status": "signed_out", "redirect": redirect}

    @app.get("/api/account")
    async def api_account(
        request: Request,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, Any]:
        """Profil curateur whiteliste (issue-016) — pas de champs super-admin."""
        eng = webapp_deps.get_engine(request)
        row = get_user_email_display_name_by_id(eng, user_id)
        if row is None:
            raise TenantResourceOpaqueDenial()
        email, display_name = row
        prefs_raw = get_user_ui_preferences_raw(eng, user_id)
        if prefs_raw is None:
            raise TenantResourceOpaqueDenial()
        ui_preferences = load_from_stored_raw(prefs_raw)
        return {
            "appUserId": user_id,
            "email": email,
            "displayName": display_name,
            "counts": {
                "ownedProjects": count_owned_projects(eng, user_id),
                "activeMemberships": count_active_memberships(eng, user_id),
            },
            "uiPreferences": ui_preferences,
        }

    @app.patch("/api/account/ui-preferences")
    async def api_patch_account_ui_preferences(
        request: Request,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
        body: AccountUiPreferencesPatchBody,
    ) -> dict[str, Any]:
        """Fusion partielle des préférences d'affichage (densité, confort lecture) — issue-023."""
        eng = webapp_deps.get_engine(request)
        row = get_user_email_display_name_by_id(eng, user_id)
        if row is None:
            raise TenantResourceOpaqueDenial()
        prefs_raw = get_user_ui_preferences_raw(eng, user_id)
        if prefs_raw is None:
            raise TenantResourceOpaqueDenial()
        patch = body.model_dump(exclude_unset=True, exclude_none=True)
        current = load_from_stored_raw(prefs_raw)
        try:
            merged = merge_patch_into_canonical(current, patch)
            if patch:
                blob = serialize_canonical_preferences(merged)
                update_user_ui_preferences_raw(eng, user_id, blob)
        except ValueError as exc:
            raise EnvelopeHttpError(400, _bad_request_client_envelope(str(exc))) from exc
        return {"uiPreferences": merged}

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

    @app.get("/api/projects/{project_id}/dashboard")
    async def api_project_dashboard(
        request: Request,
        project_id: str,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
        dashboard_scope: Annotated[DashboardStylometryScope, Query()] = "validated",
    ) -> Any:
        """Agrégats stylométrie / cohérence (issue-014) — aligné ``prepare_for_dashboard_tab``."""
        eng = webapp_deps.get_engine(request)
        try:
            df = load_project_entries(eng, project_id, user_id)
        except OperationalError as exc:
            log_resolved_api_error(logger, exc, extra_context={"route": "project_dashboard"})
            return JSONResponse(
                status_code=503,
                content=error_envelope_for_client(exc, include_technical_detail=None),
            )
        return build_curator_dashboard_envelope(
            df,
            scope=dashboard_scope,
            validated_label=STATUT_VALIDE,
        )

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
        try:
            _enforce_export_row_cap(export_df)
        except ExportPayloadTooLargeError as exc:
            return JSONResponse(
                status_code=413,
                content=error_envelope_for_client(exc, include_technical_detail=None),
            )
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
        export_df = dataframe_for_export(df, scope)
        try:
            _enforce_export_row_cap(export_df)
        except ExportPayloadTooLargeError as exc:
            return JSONResponse(
                status_code=413,
                content=error_envelope_for_client(exc, include_technical_detail=None),
            )
        payload = convert_to_jsonl(
            df,
            format=export_format,
            include_stylometry=True,
            scope=scope,
        )
        return PlainTextResponse(
            content=payload,
            media_type="application/x-ndjson; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="export-{project_id}.jsonl"'},
        )

    @app.get("/api/projects/{project_id}/curator/dimensions")
    async def api_curator_dimensions(
        request: Request,
        project_id: str,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, Any]:
        """Dimensions actives (profil projet) pour les aides curateur."""
        eng = webapp_deps.get_engine(request)
        return curator_ai.build_curator_dimensions_payload(eng, project_id, user_id)

    @app.post("/api/projects/{project_id}/curator/llm-generate")
    async def api_curator_llm_generate(
        request: Request,
        project_id: str,
        body: CuratorLlmBody,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, Any]:
        """Génération assistée (serveur : clé API et timeouts projet)."""
        eng = webapp_deps.get_engine(request)
        return curator_ai.run_curator_llm_generate(
            eng,
            project_id,
            user_id,
            mode=body.mode,
            input_text=body.input,
            output_text=body.output,
            type_=body.type,
            structure=body.structure,
            ton=body.ton,
            format_=body.format,
            public=body.public,
        )

    @app.post("/api/projects/{project_id}/curator/languagetool-check")
    async def api_curator_languagetool_check(
        request: Request,
        project_id: str,
        body: CuratorLanguageToolBody,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, Any]:
        """Contrôle LanguageTool : texte corrigé + liste de suggestions."""
        eng = webapp_deps.get_engine(request)
        try:
            return curator_ai.run_curator_languagetool_check(
                eng, project_id, user_id, text=body.text
            )
        except (requests.RequestException, ValueError) as exc:
            logger.warning("curator_languagetool_check failed: %s", exc, exc_info=True)
            raise EnvelopeHttpError(
                503,
                curator_ai.curator_languagetool_unavailable_envelope(),
            ) from exc

    @app.post("/api/super-admin/invite")
    async def api_super_admin_invite(
        request: Request,
        body: SuperAdminInviteBody,
        user: Annotated[UserRecord, Depends(webapp_deps.require_super_admin_app_user)],
    ) -> Any:
        """Invitation collaborateur (même chaîne que Streamlit : lien + mailer)."""
        eng = webapp_deps.get_engine(request)
        try:
            outcome = invite_collaborator_by_email(eng, user.user_id, body.email)
        except PermissionError as exc:
            raise EnvelopeHttpError(
                403,
                error_envelope_for_client(exc, include_technical_detail=False),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            log_resolved_api_error(logger, exc, extra_context={"route": "super_admin_invite"})
            return JSONResponse(
                status_code=500,
                content=error_envelope_for_client(exc, include_technical_detail=None),
            )
        return {
            "status": "ok",
            "mailMode": outcome.mail_mode,
            "message": outcome.message_fr,
        }

    @app.get("/api/super-admin/accounts")
    async def api_super_admin_accounts(
        request: Request,
        user: Annotated[UserRecord, Depends(webapp_deps.require_super_admin_app_user)],
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[
            int,
            Query(
                ge=SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MIN,
                le=SUPER_ADMIN_ACCOUNTS_PAGE_SIZE_MAX,
            ),
        ] = 25,
    ) -> Any:
        """Annuaire paginé des comptes actifs (issue-018)."""
        eng = webapp_deps.get_engine(request)
        total = count_users_for_admin(eng)
        try:
            p, s = validate_super_admin_accounts_list_params(
                page=page, page_size=page_size, total_active_accounts=total
            )
        except ValueError as exc:
            raise EnvelopeHttpError(
                400,
                {
                    "error": {
                        "code": "BAD_REQUEST",
                        "title": "Paramètres invalides",
                        "message": str(exc),
                        "suggested_action": "Corrigez les paramètres « page » ou « page_size » puis réessayez.",
                        "detail": None,
                    }
                },
            ) from exc
        rows = list_accounts_for_super_admin(eng, user.user_id, page=p, page_size=s)
        total_pages = max(1, math.ceil(total / s)) if total else 1
        accounts: list[dict[str, Any]] = []
        for row in rows:
            last_login = row.last_login_at.strip() if row.last_login_at else ""
            accounts.append(
                {
                    "accountId": row.user_id,
                    "email": row.email,
                    "displayName": row.display_name,
                    "isSuperAdmin": row.is_super_admin,
                    "ownedProjects": row.project_count,
                    "entriesTotal": row.entries_total,
                    "entriesValidated": row.entries_validated,
                    "lastLoginAt": last_login or None,
                }
            )
        return {
            "totalActiveAccounts": total,
            "page": p,
            "pageSize": s,
            "totalPages": total_pages,
            "accounts": accounts,
        }

    @app.get("/api/super-admin/saga/telemetry")
    async def api_super_admin_saga_telemetry(
        request: Request,
        user: Annotated[UserRecord, Depends(webapp_deps.require_super_admin_app_user)],
    ) -> Any:
        """Télémétrie saga, file de retry et DLQ (issue-019)."""
        eng = webapp_deps.get_engine(request)
        try:
            return build_deprovision_telemetry_payload(eng, user.user_id)
        except Exception as exc:  # noqa: BLE001
            log_resolved_api_error(
                logger, exc, extra_context={"route": "super_admin_saga_telemetry"}
            )
            return JSONResponse(
                status_code=500,
                content=error_envelope_for_client(exc, include_technical_detail=None),
            )

    @app.post("/api/super-admin/saga/replay-quarantined")
    async def api_super_admin_saga_replay_quarantined(
        request: Request,
        user: Annotated[UserRecord, Depends(webapp_deps.require_super_admin_app_user)],
        body: SuperAdminSagaReplayBody,
    ) -> Any:
        """Relance une opération DLQ ; la réponse inclut la télémétrie rafraîchie (issue-019)."""
        eng = webapp_deps.get_engine(request)
        try:
            replay_quarantined_operation(eng, user.user_id, body.operation_id)
        except RuntimeError as exc:
            raise EnvelopeHttpError(
                400,
                {
                    "error": {
                        "code": "BAD_REQUEST",
                        "title": "Relance impossible",
                        "message": str(exc),
                        "suggested_action": (
                            "Vérifiez qu'une opération en quarantaine est sélectionnée et qu'aucune autre saga "
                            "n'est en cours pour le même utilisateur cible."
                        ),
                        "detail": None,
                    }
                },
            ) from exc
        except PermissionError as exc:
            raise EnvelopeHttpError(
                403,
                error_envelope_for_client(exc, include_technical_detail=False),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            log_resolved_api_error(
                logger, exc, extra_context={"route": "super_admin_saga_replay_quarantined"}
            )
            return JSONResponse(
                status_code=500,
                content=error_envelope_for_client(exc, include_technical_detail=None),
            )
        try:
            telemetry = build_deprovision_telemetry_payload(eng, user.user_id)
        except Exception as exc:  # noqa: BLE001
            log_resolved_api_error(
                logger, exc, extra_context={"route": "super_admin_saga_replay_telemetry"}
            )
            return JSONResponse(
                status_code=500,
                content=error_envelope_for_client(exc, include_technical_detail=None),
            )
        return {
            "status": "ok",
            "message": "Opération remise en file d'attente.",
            "telemetry": telemetry,
        }

    return app


app = create_slice_app()
