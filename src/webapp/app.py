"""Application FastAPI — slice vertical issue-007 (auth, projets, entrées, export)."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict
from sqlalchemy.engine import Engine

from src.api_errors import (
    TenantResourceOpaqueDenial,
    error_envelope_for_client,
    log_resolved_api_error,
)
from src.auth import persist_user_from_signin_ok
from src.database import (
    count_active_memberships,
    count_owned_projects,
    create_db_engine,
    get_user_email_display_name_by_id,
    list_projects_for_user,
    load_project_entries,
)
from src.export_utils import ExportFormat, ExportScope, convert_to_jsonl, dataframe_for_export
from src.supertokens_recipe_client import signin_email_password, try_revoke_access_token
from src.webapp import deps as webapp_deps
from src.webapp import entry_mutations
from src.webapp.errors import EnvelopeHttpError

logger = logging.getLogger(__name__)


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


def _cors_origins() -> list[str]:
    raw = (os.environ.get("WEBAPP_CORS_ORIGINS") or "http://localhost:8080").strip()
    return [o.strip() for o in raw.split(",") if o.strip()]


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
        return {
            "appUserId": user_id,
            "email": email,
            "displayName": display_name,
            "counts": {
                "ownedProjects": count_owned_projects(eng, user_id),
                "activeMemberships": count_active_memberships(eng, user_id),
            },
        }

    @app.get("/api/projects")
    async def api_projects(
        request: Request,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, Any]:
        eng = webapp_deps.get_engine(request)
        projects = list_projects_for_user(eng, user_id)
        return {
            "projects": [{"id": p.project_id, "name": p.name, "role": p.role} for p in projects]
        }

    @app.get("/api/projects/{project_id}/entries")
    async def api_list_entries(
        request: Request,
        project_id: str,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, Any]:
        eng = webapp_deps.get_engine(request)
        df = load_project_entries(eng, project_id, user_id)
        cols = [c for c in df.columns if not str(c).startswith("_")]
        rows = df[cols].to_dict(orient="records")
        return {"entries": rows}

    @app.patch("/api/projects/{project_id}/entries/{entry_id}")
    async def api_patch_entry(
        request: Request,
        project_id: str,
        entry_id: str,
        body: EntryPatchBody,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, str]:
        eng = webapp_deps.get_engine(request)
        updates = {k: v for k, v in body.model_dump(exclude_none=True).items()}
        try:
            entry_mutations.apply_entry_field_updates(eng, project_id, user_id, entry_id, updates)
        except KeyError as exc:
            raise TenantResourceOpaqueDenial() from exc
        return {"status": "ok"}

    @app.post("/api/projects/{project_id}/entries")
    async def api_create_entry(
        request: Request,
        project_id: str,
        body: NewEntryBody,
        user_id: Annotated[str, Depends(webapp_deps.require_app_user_id)],
    ) -> dict[str, str]:
        eng = webapp_deps.get_engine(request)
        new_id = entry_mutations.append_minimal_entry(
            eng,
            project_id,
            user_id,
            input_text=body.input,
            output_text=body.output,
        )
        return {"id": new_id, "status": "ok"}

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

_INDEX_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dataset Style — slice vertical</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; }
    label { display: block; margin-top: 0.75rem; }
    input, textarea, select, button { width: 100%; max-width: 32rem; box-sizing: border-box; }
    textarea { min-height: 6rem; }
    .row { margin: 1rem 0; }
    .err { color: #a40000; white-space: pre-wrap; }
    .ok { color: #0b5; }
    code { font-size: 0.85rem; }
    #shellNav { display: flex; gap: 0.5rem; margin: 1rem 0; flex-wrap: wrap; }
    #shellNav button { width: auto; max-width: none; }
    #shellNav button.tab-active { font-weight: 700; text-decoration: underline; }
    .account-dl dt { font-weight: 600; margin-top: 0.5rem; }
    .account-dl dd { margin: 0.15rem 0 0 0; }
  </style>
</head>
<body>
  <h1>Coquille curateur (issue-010 / slice vertical)</h1>
  <p>Parcours minimal : connexion, navigation shell, projet, édition, export ; <strong>Mon compte</strong> (issue-016).</p>
  <p>Streamlit reste sur le port <code>8501</code> ; cette coquille est servie sur le port du service <code>webapp</code>.</p>

  <section id="auth">
    <h2>Connexion</h2>
    <label>Email <input type="email" id="email" autocomplete="username" /></label>
    <label>Mot de passe <input type="password" id="password" autocomplete="current-password" /></label>
    <div class="row"><button type="button" id="btnSignin">Se connecter</button></div>
    <p id="authMsg" class="err" aria-live="polite"></p>
  </section>

  <div id="shellLoggedIn" hidden>
    <nav id="shellNav" aria-label="Navigation principale">
      <button type="button" id="tabWorkspace" class="tab-active">Espace de travail</button>
      <button type="button" id="tabAccount">Mon compte</button>
    </nav>

    <section id="panelWorkspace">
    <h2>Projet</h2>
    <label>Projet sélectionné
      <select id="projectSel"></select>
    </label>
    <h2>Entrées</h2>
    <p><button type="button" id="btnReloadEntries">Recharger les entrées</button></p>
    <div id="entriesTable"></div>
    <h3>Édition (id de fiche)</h3>
    <label>id <input type="text" id="entryId" /></label>
    <label>input <textarea id="fldInput"></textarea></label>
    <label>output <textarea id="fldOutput"></textarea></label>
    <div class="row"><button type="button" id="btnSave">Enregistrer</button></div>
    <h3>Création rapide</h3>
    <label>nouveau input <textarea id="newInput"></textarea></label>
    <label>nouveau output <textarea id="newOutput"></textarea></label>
    <div class="row"><button type="button" id="btnCreate">Créer une fiche</button></div>
    <h3>Export (périmètre <code>export_utils</code>)</h3>
    <label>Périmètre
      <select id="exportScope">
        <option value="validated_only">Validées seulement</option>
        <option value="full_dataset">Tout le dataset</option>
      </select>
    </label>
    <div class="row">
      <button type="button" id="btnCsv">Télécharger CSV</button>
      <button type="button" id="btnJsonl">Télécharger JSONL (LFM2)</button>
    </div>
    </section>

    <section id="panelAccount" hidden>
      <h2>Mon compte</h2>
      <p class="row">Informations visibles côté curateur (pas d’indicateurs d’administration).</p>
      <div id="accountDetail" class="account-dl"></div>
      <p id="accountLoadErr" class="err" aria-live="polite"></p>
      <div class="row"><button type="button" id="btnSignout">Se déconnecter</button></div>
    </section>
  </div>

  <script>
    const LS = "slice_vertical_access_token";
    const authMsg = document.getElementById("authMsg");
    const shellLoggedIn = document.getElementById("shellLoggedIn");
    const panelWorkspace = document.getElementById("panelWorkspace");
    const panelAccount = document.getElementById("panelAccount");
    const tabWorkspace = document.getElementById("tabWorkspace");
    const tabAccount = document.getElementById("tabAccount");

    function token() { return localStorage.getItem(LS) || ""; }
    function setToken(t) {
      if (t) localStorage.setItem(LS, t); else localStorage.removeItem(LS);
      shellLoggedIn.hidden = !t;
    }
    setToken(token());

    function showErr(obj) {
      if (obj && obj.error) {
        const e = obj.error;
        authMsg.textContent = (e.title || "") + "\\n" + (e.message || "") + (e.code ? "\\ncode: " + e.code : "");
        authMsg.className = "err";
      } else {
        authMsg.textContent = JSON.stringify(obj);
        authMsg.className = "err";
      }
    }
    function showOk(msg) {
      authMsg.textContent = msg || "OK";
      authMsg.className = "ok";
    }

    async function api(path, opts = {}) {
      const headers = Object.assign({}, opts.headers || {});
      if (token()) headers["Authorization"] = "Bearer " + token();
      const r = await fetch(path, Object.assign({}, opts, { headers }));
      const ct = r.headers.get("content-type") || "";
      const body = ct.includes("application/json") ? await r.json() : await r.text();
      if (!r.ok) throw body;
      return body;
    }

    function showShellTab(which) {
      const isWs = which === "workspace";
      panelWorkspace.hidden = !isWs;
      panelAccount.hidden = isWs;
      tabWorkspace.classList.toggle("tab-active", isWs);
      tabAccount.classList.toggle("tab-active", !isWs);
      if (!isWs) loadAccountPanel();
    }
    tabWorkspace.onclick = () => showShellTab("workspace");
    tabAccount.onclick = () => showShellTab("account");

    async function loadAccountPanel() {
      const errEl = document.getElementById("accountLoadErr");
      const detail = document.getElementById("accountDetail");
      errEl.textContent = "";
      try {
        const a = await api("/api/account");
        detail.innerHTML =
          "<dl>"
          + "<dt>Identifiant applicatif</dt><dd><code>" + escapeHtml(a.appUserId) + "</code></dd>"
          + "<dt>Email</dt><dd>" + escapeHtml(a.email) + "</dd>"
          + "<dt>Nom affiché</dt><dd>" + escapeHtml(a.displayName) + "</dd>"
          + "<dt>Projets possédés</dt><dd>" + a.counts.ownedProjects + "</dd>"
          + "<dt>Memberships actives</dt><dd>" + a.counts.activeMemberships + "</dd>"
          + "</dl>";
      } catch (e) {
        detail.innerHTML = "";
        if (e && e.error) errEl.textContent = (e.error.message || "") + " (" + (e.error.code || "") + ")";
        else errEl.textContent = "Impossible de charger le profil.";
      }
    }
    function escapeHtml(s) {
      return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/\"/g,"&quot;");
    }

    document.getElementById("btnSignin").onclick = async () => {
      authMsg.textContent = "";
      const email = document.getElementById("email").value.trim();
      const password = document.getElementById("password").value;
      try {
        const out = await api("/api/auth/signin", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
        setToken(out.accessToken || "");
        showOk("Connecté.");
        showShellTab("workspace");
        await loadProjects();
      } catch (e) { showErr(e); }
    };

    async function performSignout() {
      const t = token();
      let target = "/";
      try {
        const out = await api("/api/auth/signout", {
          method: "POST",
          headers: { "Content-Type": "application/json", "Authorization": "Bearer " + t },
          body: JSON.stringify({ access_token: t, redirect_after: "/" }),
        });
        if (out && typeof out.redirect === "string" && out.redirect) target = out.redirect;
      } catch (_) { /* jeton déjà invalide : on purge quand même */ }
      setToken("");
      window.location.assign(target);
    }
    document.getElementById("btnSignout").onclick = () => performSignout();

    async function loadProjects() {
      const data = await api("/api/projects");
      const sel = document.getElementById("projectSel");
      sel.innerHTML = "";
      for (const p of data.projects) {
        const o = document.createElement("option");
        o.value = p.id; o.textContent = p.name + " (" + p.role + ")";
        sel.appendChild(o);
      }
      await loadEntries();
    }

    document.getElementById("projectSel").onchange = () => loadEntries();
    document.getElementById("btnReloadEntries").onclick = () => loadEntries();

    async function loadEntries() {
      const pid = document.getElementById("projectSel").value;
      if (!pid) return;
      const data = await api("/api/projects/" + encodeURIComponent(pid) + "/entries");
      const div = document.getElementById("entriesTable");
      if (!data.entries.length) { div.textContent = "(aucune fiche)"; return; }
      const t = document.createElement("table");
      t.border = "1";
      const keys = Object.keys(data.entries[0]).filter(k => !k.startsWith("_"));
      const trh = document.createElement("tr");
      for (const k of keys) { const th = document.createElement("th"); th.textContent = k; trh.appendChild(th); }
      t.appendChild(trh);
      for (const row of data.entries) {
        const tr = document.createElement("tr");
        for (const k of keys) { const td = document.createElement("td"); td.textContent = row[k]; tr.appendChild(td); }
        t.appendChild(tr);
      }
      div.innerHTML = "";
      div.appendChild(t);
    }

    document.getElementById("btnSave").onclick = async () => {
      const pid = document.getElementById("projectSel").value;
      const eid = document.getElementById("entryId").value.trim();
      const input = document.getElementById("fldInput").value;
      const output = document.getElementById("fldOutput").value;
      try {
        await api("/api/projects/" + encodeURIComponent(pid) + "/entries/" + encodeURIComponent(eid), {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ input, output }),
        });
        showOk("Fiche enregistrée.");
        await loadEntries();
      } catch (e) { showErr(e); }
    };

    document.getElementById("btnCreate").onclick = async () => {
      const pid = document.getElementById("projectSel").value;
      const input = document.getElementById("newInput").value;
      const output = document.getElementById("newOutput").value;
      try {
        const out = await api("/api/projects/" + encodeURIComponent(pid) + "/entries", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ input, output }),
        });
        document.getElementById("entryId").value = out.id;
        showOk("Fiche créée : " + out.id);
        await loadEntries();
      } catch (e) { showErr(e); }
    };

    function scopeParam() {
      return "?scope=" + encodeURIComponent(document.getElementById("exportScope").value);
    }

    document.getElementById("btnCsv").onclick = async () => {
      const pid = document.getElementById("projectSel").value;
      const r = await fetch("/api/projects/" + encodeURIComponent(pid) + "/export.csv" + scopeParam(), {
        headers: { "Authorization": "Bearer " + token() },
      });
      if (!r.ok) { showErr(await r.json()); return; }
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "export.csv";
      a.click();
    };

    document.getElementById("btnJsonl").onclick = async () => {
      const pid = document.getElementById("projectSel").value;
      const r = await fetch("/api/projects/" + encodeURIComponent(pid) + "/export.jsonl" + scopeParam(), {
        headers: { "Authorization": "Bearer " + token() },
      });
      if (!r.ok) { showErr(await r.json()); return; }
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "export.jsonl";
      a.click();
    };
  </script>
</body>
</html>
"""
