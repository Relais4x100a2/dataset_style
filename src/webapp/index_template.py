"""Page d'accueil HTML du service ``webapp`` (shell curateur issue-010 + slice issue-007)."""

# Contenu servi tel quel par ``GET /`` ; pas de logique Python ici (uniquement chaîne).
INDEX_HTML = """<!DOCTYPE html>
<html lang="fr">
<script>(function(){try{var r=sessionStorage.getItem('ds_ui_prefs_v1');if(!r)return;var o=JSON.parse(r);if(!o||typeof o!=='object')return;var e=document.documentElement;if(o.density&&o.density!=='default')e.setAttribute('data-ds-density',o.density);else e.removeAttribute('data-ds-density');if(o.readingComfort&&o.readingComfort!=='default')e.setAttribute('data-ds-reading',o.readingComfort);else e.removeAttribute('data-ds-reading');}catch(x){}})();</script>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dataset Style — curateur</title>
  <link rel="stylesheet" href="/static/design_tokens.css" />
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #f6f7f9; color: #1a1a1a; }
    .wrap { max-width: 56rem; margin: 0 auto; padding: 1rem 1rem 2rem; }
    label { display: block; margin-top: 0.75rem; }
    input, textarea, select, button { width: 100%; max-width: 36rem; box-sizing: border-box; }
    textarea { min-height: 5rem; }
    .row { margin: 0.75rem 0; }
    .err { color: #a40000; white-space: pre-wrap; }
    .ok { color: #0b5; }
    .warn { color: #964; white-space: pre-wrap; }
    code { font-size: 0.85rem; }
    nav#mainNav {
      display: flex; flex-wrap: wrap; gap: 0.25rem; border-bottom: 1px solid #ccd; margin: 0.5rem 0 1rem;
      background: #fff; padding: 0.35rem 0.25rem 0; border-radius: 6px 6px 0 0;
    }
    nav#mainNav button {
      width: auto; max-width: none; padding: 0.45rem 0.65rem; border: 1px solid transparent;
      background: transparent; border-radius: 4px 4px 0 0; cursor: pointer; font-size: 0.9rem;
    }
    nav#mainNav button:hover { background: #eef1f6; }
    nav#mainNav button.active { background: #f6f7f9; border-color: #ccd #ccd #f6f7f9; font-weight: 600; }
    .panel { display: none; background: #fff; padding: 1rem; border: 1px solid #ccd; border-top: none; border-radius: 0 0 6px 6px; }
    .panel.active { display: block; }
    .muted { color: #555; font-size: 0.9rem; }
    h1 { font-size: 1.35rem; }
    h2 { font-size: 1.1rem; margin-top: 0; }
    .banner { font-size: 0.85rem; color: #444; margin-bottom: 1rem; }
    .account-dl dt { font-weight: 600; margin-top: 0.5rem; }
    .account-dl dd { margin: 0.15rem 0 0 0; }
    table.sa-accounts { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 0.5rem; }
    table.sa-accounts th, table.sa-accounts td { border: 1px solid #ccd; padding: 0.35rem 0.5rem; text-align: left; }
    table.sa-saga { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.35rem; }
    table.sa-saga th, table.sa-saga td { border: 1px solid #ccd; padding: 0.3rem 0.45rem; text-align: left; word-break: break-all; }
    .danger-zone { border: 2px solid #c0392b; background: #fdecea; padding: 0.85rem 1rem; border-radius: 6px; margin-top: 1rem; }
    .danger-zone h4 { margin-top: 0; color: #7b241c; font-size: 1rem; }
    .entries-toolbar label { margin-top: 0.35rem; }
    .entries-toolbar input[type="number"] { max-width: 8rem; }
    table.entries-list { border-collapse: collapse; font-size: 0.9rem; margin-top: 0.35rem; }
    table.entries-list th, table.entries-list td { border: 1px solid #ccd; padding: 0.35rem 0.45rem; text-align: left; }
    tr.entries-row-openable:hover { background: #eef4ff; }
    details.curator-assist { margin-top: 1rem; padding: 0.5rem 0; }
    details.curator-assist summary { cursor: pointer; font-weight: 600; }
    .assist-busy { color: #555; font-size: 0.9rem; margin: 0.35rem 0; }
    details.curator-assist.htmx-request { outline: 2px dashed #7a9fd1; }
    p.hx-indicator { margin: 0.35rem 0; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Dataset Style — coquille curateur</h1>
    <p class="banner">Shell de navigation aligné sur Streamlit (ordre des onglets via <code>/api/me</code>).
      Streamlit reste sur le port <code>8501</code> ; ce service <code>webapp</code> porte le slice API + UI minimale.</p>

    <section id="auth">
      <h2>Connexion</h2>
      <label>Email <input type="email" id="email" autocomplete="username" /></label>
      <label>Mot de passe <input type="password" id="password" autocomplete="current-password" /></label>
      <div class="row"><button type="button" id="btnSignin">Se connecter</button></div>
      <div class="row"><button type="button" id="btnSignout">Se déconnecter</button></div>
      <p id="authMsg" class="err" aria-live="polite"></p>
    </section>

    <section id="workspace" hidden>
      <p id="userLine" class="muted"></p>
      <nav id="mainNav" aria-label="Workflow"></nav>
      <div id="panels"></div>
    </section>
  </div>

  <template id="tplPanels">
    <div class="panel" data-tab-idx="0">
      <h2>Projets</h2>
      <p class="muted">Projet courant persisté dans la session du navigateur (sessionStorage).</p>
      <label>Projet actif <select id="projectSel"></select></label>
      <h3>Créer un projet</h3>
      <label>Nom <input type="text" id="newProjectName" maxlength="500" /></label>
      <label>Description (optionnel) <input type="text" id="newProjectDesc" maxlength="10000" /></label>
      <div class="row"><button type="button" id="btnCreateProject">Créer</button></div>
      <h3>Supprimer le projet actif</h3>
      <label><input type="checkbox" id="delProjConfirm" /> Je confirme vouloir supprimer ce projet</label>
      <label>Retaper le nom exact du projet pour confirmer <input type="text" id="delProjName" /></label>
      <div class="row"><button type="button" id="btnDeleteProject">Supprimer ce projet</button></div>
    </div>
    <div class="panel" data-tab-idx="1">
      <h2>Réglages &amp; Export</h2>
      <p class="muted">Réglages détaillés : parité sprint ultérieur. Export (même périmètre que Streamlit) :</p>
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
    </div>
    <div class="panel" data-tab-idx="2">
      <h2>Nouvelle entrée</h2>
      <p class="muted">Création minimale (slice issue-007).</p>
      <label>input <textarea id="newInput"></textarea></label>
      <label>output <textarea id="newOutput"></textarea></label>
      <div class="row"><button type="button" id="btnCreate">Créer une fiche</button></div>
      <details class="curator-assist">
        <summary>Assistance IA &amp; LanguageTool</summary>
        <p class="muted">Aucune écriture en base : les résultats restent dans le navigateur jusqu'à la création de la fiche.
          Appels stateless : <code>/curator/llm-generate</code>, <code>/curator/languagetool-check</code>.</p>
        <p id="assistBusyNew" class="assist-busy hx-indicator" style="display:none" aria-live="polite">Chargement…</p>
        <label>Type <select id="assistNewType"><option value="">—</option></select></label>
        <label>Structure <select id="assistNewStructure"><option value="">—</option></select></label>
        <label>Ton <select id="assistNewTon"><option value="">—</option></select></label>
        <label>Format <select id="assistNewFormat"><option value="">—</option></select></label>
        <label>Public <select id="assistNewPublic"><option value="">—</option></select></label>
        <div class="row"><button type="button" id="btnAssistLlmDoNew">Générer output depuis le brouillon (IA)</button></div>
        <div class="row"><button type="button" id="btnAssistLlmOdNew">Générer brouillon depuis l'output (IA)</button></div>
        <div class="row"><button type="button" id="btnAssistLtNew">Contrôler LanguageTool sur l'output</button></div>
        <p id="assistMsgNew" role="status" class="muted" aria-live="polite"></p>
        <div id="assistLlmPreviewNew" class="muted" hidden></div>
        <div class="row" id="assistLlmActionsNew" hidden>
          <button type="button" id="btnAssistLlmInsertNew">Insérer le résultat IA</button>
          <button type="button" id="btnAssistLlmDismissNew">Ignorer</button>
        </div>
        <div id="assistLtPreviewNew" class="muted" hidden></div>
        <div class="row" id="assistLtActionsNew" hidden>
          <button type="button" id="btnAssistLtReplaceNew">Remplacer tout l'output par la proposition LanguageTool</button>
          <button type="button" id="btnAssistLtDismissNew">Ignorer la proposition</button>
        </div>
      </details>
    </div>
    <div class="panel" data-tab-idx="3">
      <h2>Gestion &amp; édition</h2>
      <p class="muted">Filtres <code>edition_*</code> : même logique que <code>GET /api/projects/{id}/entries</code>.
        Valeurs persistées par projet dans <code>sessionStorage</code> (clé <code>webapp_edition_filters:&lt;project_id&gt;</code>).
        Après <code>POST</code>/<code>PATCH</code>, la liste et les champs persistés de la fiche se basent sur le tableau <code>entries</code> de la réponse (mêmes filtres en query).</p>
      <div class="entries-toolbar">
        <label>Statut
          <select id="editionStatutFilter">
            <option value="">(tous)</option>
            <option value="A faire">A faire</option>
            <option value="En cours">En cours</option>
            <option value="Fait et validé">Fait et validé</option>
          </select>
        </label>
        <label>Score cohérence
          <select id="editionScoreMode">
            <option value="">(aucun filtre score)</option>
            <option value="all">Tous les scores</option>
            <option value="below">Strictement sous le seuil</option>
            <option value="bucket">Tranche (décile 0–9)</option>
            <option value="na_only">Sans score (N/A)</option>
          </select>
        </label>
        <label>Seuil &lt; (0–100) <input type="number" id="editionScoreThreshold" min="0" max="100" value="50" /></label>
        <label>Décile (0–9) <input type="number" id="editionScoreDecile" min="0" max="9" value="0" /></label>
        <label><input type="checkbox" id="editionScoreIncludeNa" /> Inclure N/A (sous seuil / tranche)</label>
      </div>
      <p><button type="button" id="btnReloadEntries">Recharger les entrées</button></p>
      <p id="entriesStatus" role="status" class="muted" aria-live="polite"></p>
      <div id="entriesTable"></div>
      <h3>Édition (id de fiche)</h3>
      <label>id <input type="text" id="entryId" /></label>
      <label>input <textarea id="fldInput"></textarea></label>
      <label>output <textarea id="fldOutput"></textarea></label>
      <div class="row"><button type="button" id="btnSave">Enregistrer</button></div>
      <details class="curator-assist">
        <summary>Assistance IA &amp; LanguageTool</summary>
        <p class="muted">Aucune écriture en base : enregistrez la fiche (<code>PATCH …/entries/…</code>) pour persister les changements.
          Appels stateless : <code>/curator/llm-generate</code>, <code>/curator/languagetool-check</code>.</p>
        <p id="assistBusyEdit" class="assist-busy hx-indicator" style="display:none" aria-live="polite">Chargement…</p>
        <label>Type <select id="assistEditType"><option value="">—</option></select></label>
        <label>Structure <select id="assistEditStructure"><option value="">—</option></select></label>
        <label>Ton <select id="assistEditTon"><option value="">—</option></select></label>
        <label>Format <select id="assistEditFormat"><option value="">—</option></select></label>
        <label>Public <select id="assistEditPublic"><option value="">—</option></select></label>
        <div class="row"><button type="button" id="btnAssistLlmDoEdit">Générer output depuis le brouillon (IA)</button></div>
        <div class="row"><button type="button" id="btnAssistLlmOdEdit">Générer brouillon depuis l'output (IA)</button></div>
        <div class="row"><button type="button" id="btnAssistLtEdit">Contrôler LanguageTool sur l'output</button></div>
        <p id="assistMsgEdit" role="status" class="muted" aria-live="polite"></p>
        <div id="assistLlmPreviewEdit" class="muted" hidden></div>
        <div class="row" id="assistLlmActionsEdit" hidden>
          <button type="button" id="btnAssistLlmInsertEdit">Insérer le résultat IA</button>
          <button type="button" id="btnAssistLlmDismissEdit">Ignorer</button>
        </div>
        <div id="assistLtPreviewEdit" class="muted" hidden></div>
        <div class="row" id="assistLtActionsEdit" hidden>
          <button type="button" id="btnAssistLtReplaceEdit">Remplacer tout l'output par la proposition LanguageTool</button>
          <button type="button" id="btnAssistLtDismissEdit">Ignorer la proposition</button>
        </div>
      </details>
    </div>
    <div class="panel" data-tab-idx="4">
      <h2>Tableau de bord</h2>
      <p class="muted">Placeholder — parité métriques / stylométrie : sprints suivants.</p>
    </div>
    <div class="panel" data-tab-idx="5">
      <h2>Mon compte</h2>
      <p class="muted">Profil curateur (issue-016) — pas d’indicateurs super-admin.</p>
      <div id="accountDetail" class="account-dl"></div>
      <h3>Réglages d&apos;affichage</h3>
      <p class="muted">Optionnel — par défaut l&apos;interface reste celle recommandée.</p>
      <label>Densité
        <select id="prefDensity">
          <option value="default">Recommandée</option>
          <option value="compact">Compacte</option>
          <option value="comfortable">Confortable</option>
        </select>
      </label>
      <label>Confort lecture
        <select id="prefReading">
          <option value="default">Recommandé</option>
          <option value="high_contrast">Contraste renforcé</option>
          <option value="reduced_motion">Moins d&apos;animation</option>
        </select>
      </label>
      <div class="row"><button type="button" id="btnSaveUiPrefs">Enregistrer l&apos;affichage</button></div>
      <p id="uiPrefsMsg" class="muted" aria-live="polite"></p>
      <p id="accountLoadErr" class="err" aria-live="polite"></p>
    </div>
    <div class="panel" data-tab-idx="6">
      <h2>Super Admin</h2>
      <p class="muted">Invitation d’un collaborateur (invitation-only). Aucune inscription publique.</p>
      <label>E-mail du collaborateur <input type="email" id="saInviteEmail" autocomplete="off" /></label>
      <div class="row"><button type="button" id="btnSaInvite">Envoyer l’invitation</button></div>
      <p id="saInviteMsg" class="muted" aria-live="polite"></p>
      <h3>Comptes actifs</h3>
      <p class="muted">Liste paginée via <code>GET /api/super-admin/accounts</code> : <code>page</code> ≥ 1 ;
        <code>page_size</code> entre 10 et 100 (défaut 25).</p>
      <label>Taille de page
        <select id="saAccountsPageSize">
          <option value="10">10</option>
          <option value="25" selected>25</option>
          <option value="50">50</option>
          <option value="100">100</option>
        </select>
      </label>
      <label>Page <input type="number" id="saAccountsPage" min="1" value="1" /></label>
      <div class="row"><button type="button" id="btnSaAccountsReload">Actualiser la liste</button></div>
      <p id="saAccountsErr" class="err" aria-live="polite"></p>
      <p id="saAccountsSummary" class="muted"></p>
      <div id="saAccountsTableWrap"></div>
      <h3>Panneau technique (saga)</h3>
      <p class="muted">Métriques alignées sur le studio : cartes = répartition sur les N dernières mises à jour ;
        totaux = ensemble de la table. File = opérations éligibles au worker (<code>retry_deprovision_ops</code>).</p>
      <p><button type="button" id="btnSaSagaReload">Actualiser la télémétrie</button></p>
      <p id="saSagaErr" class="err" aria-live="polite"></p>
      <div id="saSagaSummary" class="muted"></div>
      <div id="saSagaTables"></div>
      <div class="danger-zone" id="saSagaDanger">
        <h4>Zone sensible — relance manuelle (DLQ)</h4>
        <p class="muted">Même effet qu'une relance confirmée côté studio. Ne pas utiliser sans diagnostic.</p>
        <label>Opération en quarantaine
          <select id="saSagaDlqSelect"></select>
        </label>
        <label><input type="checkbox" id="saSagaReplayConfirm" /> Je confirme la remise en file de l'opération sélectionnée</label>
        <div class="row">
          <button type="button" id="btnSaSagaReplay" disabled>Relancer l'opération</button>
        </div>
        <p id="saSagaReplayMsg" class="muted" aria-live="polite"></p>
      </div>
    </div>
  </template>

  <script>
    const LS = "slice_vertical_access_token";
    const SS = "webapp_active_project_id";
    const UIPREFS_SS = "ds_ui_prefs_v1";
    const SS_EDITION_FILTER_PREFIX = "webapp_edition_filters:";
    let curatorDimsCache = null;
    const authMsg = document.getElementById("authMsg");
    const workspace = document.getElementById("workspace");
    const mainNav = document.getElementById("mainNav");
    const panelsHost = document.getElementById("panels");
    let mainTabLabels = [];

    function token() { return localStorage.getItem(LS) || ""; }
    function activeProjectHint() { return sessionStorage.getItem(SS) || ""; }
    function setActiveProjectHint(pid) {
      if (pid) sessionStorage.setItem(SS, pid); else sessionStorage.removeItem(SS);
    }
    function setToken(t) {
      if (t) localStorage.setItem(LS, t); else localStorage.removeItem(LS);
      workspace.hidden = !t;
    }
    setToken(token());

    function clearEntryState() {
      const ids = ["entryId", "fldInput", "fldOutput", "newInput", "newOutput"];
      ids.forEach((id) => { const el = document.getElementById(id); if (el) el.value = ""; });
      const div = document.getElementById("entriesTable");
      if (div) div.innerHTML = "";
      const st = document.getElementById("entriesStatus");
      if (st) st.textContent = "";
    }

    function editionFilterStorageKey(pid) {
      return SS_EDITION_FILTER_PREFIX + pid;
    }

    function persistEditionFiltersToSession() {
      const sel = document.getElementById("projectSel");
      const pid = sel ? sel.value : "";
      if (!pid) return;
      const payload = {
        statut: (document.getElementById("editionStatutFilter") || {}).value || "",
        scoreMode: (document.getElementById("editionScoreMode") || {}).value || "",
        threshold: (document.getElementById("editionScoreThreshold") || {}).value || "50",
        decile: (document.getElementById("editionScoreDecile") || {}).value || "0",
        includeNa: !!(document.getElementById("editionScoreIncludeNa") || {}).checked,
      };
      sessionStorage.setItem(editionFilterStorageKey(pid), JSON.stringify(payload));
    }

    function restoreEditionFiltersFromSession() {
      const sel = document.getElementById("projectSel");
      const pid = sel ? sel.value : "";
      if (!pid) return;
      const raw = sessionStorage.getItem(editionFilterStorageKey(pid));
      if (!raw) return;
      try {
        const o = JSON.parse(raw);
        const a = document.getElementById("editionStatutFilter");
        if (a && o.statut !== undefined) a.value = o.statut;
        const b = document.getElementById("editionScoreMode");
        if (b && o.scoreMode !== undefined) b.value = o.scoreMode;
        const c = document.getElementById("editionScoreThreshold");
        if (c && o.threshold !== undefined) c.value = o.threshold;
        const d = document.getElementById("editionScoreDecile");
        if (d && o.decile !== undefined) d.value = o.decile;
        const e = document.getElementById("editionScoreIncludeNa");
        if (e) e.checked = !!o.includeNa;
      } catch (_) { /* ignore */ }
    }

    function buildEntriesQueryString() {
      const params = new URLSearchParams();
      const est = document.getElementById("editionStatutFilter");
      if (est && est.value) params.set("edition_statut", est.value);
      const modeEl = document.getElementById("editionScoreMode");
      const mode = modeEl && modeEl.value;
      if (mode) {
        params.set("edition_score_mode", mode);
        if (mode === "below") {
          const th = document.getElementById("editionScoreThreshold");
          const t = parseInt(th && th.value, 10);
          if (!isNaN(t)) params.set("edition_score_threshold_lt", String(t));
        }
        if (mode === "bucket") {
          const dc = document.getElementById("editionScoreDecile");
          const d = parseInt(dc && dc.value, 10);
          if (!isNaN(d)) params.set("edition_score_bucket_decile", String(d));
        }
        const inc = document.getElementById("editionScoreIncludeNa");
        if (inc && inc.checked) params.set("edition_score_include_na", "true");
      }
      const s = params.toString();
      return s ? ("?" + s) : "";
    }

    function renderEntriesTableFromPayload(entries) {
      const div = document.getElementById("entriesTable");
      if (!div) return;
      if (!entries || !entries.length) {
        div.textContent = "(aucune fiche dans cette vue)";
        return;
      }
      const t = document.createElement("table");
      t.className = "entries-list";
      const keys = Object.keys(entries[0]).filter((k) => !k.startsWith("_"));
      const trh = document.createElement("tr");
      for (const k of keys) {
        const th = document.createElement("th");
        th.textContent = k;
        trh.appendChild(th);
      }
      t.appendChild(trh);
      for (const row of entries) {
        const tr = document.createElement("tr");
        tr.className = "entries-row-openable";
        tr.title = "Ouvrir la fiche (données serveur)";
        tr.onclick = () => openEntryFromServerRow(row);
        for (const k of keys) {
          const td = document.createElement("td");
          td.textContent = row[k] == null ? "" : String(row[k]);
          tr.appendChild(td);
        }
        t.appendChild(tr);
      }
      div.innerHTML = "";
      div.appendChild(t);
    }

    function openEntryFromServerRow(row) {
      const idEl = document.getElementById("entryId");
      const inEl = document.getElementById("fldInput");
      const outEl = document.getElementById("fldOutput");
      const st = document.getElementById("entriesStatus");
      if (idEl) idEl.value = row.id != null ? String(row.id) : "";
      if (inEl) inEl.value = row.input != null ? String(row.input) : "";
      if (outEl) outEl.value = row.output != null ? String(row.output) : "";
      if (st) st.textContent = "Fiche ouverte : " + (row.id != null ? String(row.id) : "");
    }

    function syncFormFieldsFromEntryId(entries, eid) {
      if (!eid || !entries || !entries.length) return;
      const row = entries.find((r) => String(r.id) === String(eid));
      if (!row) return;
      const inEl = document.getElementById("fldInput");
      const outEl = document.getElementById("fldOutput");
      if (inEl) inEl.value = row.input != null ? String(row.input) : "";
      if (outEl) outEl.value = row.output != null ? String(row.output) : "";
    }

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

    function mountPanelsFromTemplate() {
      const tpl = document.getElementById("tplPanels");
      panelsHost.innerHTML = "";
      for (const node of tpl.content.children) {
        panelsHost.appendChild(node.cloneNode(true));
      }
    }

    function escapeHtml(s) {
      return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/\"/g,"&quot;");
    }

    function applyUiPrefsToDom(p) {
      const el = document.documentElement;
      if (!p) return;
      if (p.density && p.density !== "default") el.setAttribute("data-ds-density", p.density);
      else el.removeAttribute("data-ds-density");
      if (p.readingComfort && p.readingComfort !== "default") el.setAttribute("data-ds-reading", p.readingComfort);
      else el.removeAttribute("data-ds-reading");
    }

    function persistUiPrefsCache(p) {
      try { sessionStorage.setItem(UIPREFS_SS, JSON.stringify(p)); } catch (_) {}
      applyUiPrefsToDom(p);
    }

    async function syncUiPrefsFromAccount() {
      if (!token()) return;
      try {
        const a = await api("/api/account");
        if (a.uiPreferences) persistUiPrefsCache(a.uiPreferences);
      } catch (_) { /* session expirée : pas bloquant */ }
    }

    function wireAccountUiPrefsOnce() {
      const btn = document.getElementById("btnSaveUiPrefs");
      if (!btn || btn.dataset.wired === "1") return;
      btn.dataset.wired = "1";
      btn.onclick = async () => {
        const msg = document.getElementById("uiPrefsMsg");
        if (msg) { msg.textContent = ""; msg.className = "muted"; }
        const d = document.getElementById("prefDensity");
        const r = document.getElementById("prefReading");
        const body = {};
        if (d) body.density = d.value;
        if (r) body.readingComfort = r.value;
        btn.disabled = true;
        try {
          const out = await api("/api/account/ui-preferences", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          if (out.uiPreferences) persistUiPrefsCache(out.uiPreferences);
          if (msg) { msg.textContent = "Préférences enregistrées."; msg.className = "ok"; }
        } catch (e) {
          if (msg) {
            msg.className = "err";
            if (e && e.error) msg.textContent = (e.error.message || "") + " (" + (e.error.code || "") + ")";
            else msg.textContent = "Enregistrement impossible.";
          }
        } finally {
          btn.disabled = false;
        }
      };
    }

    async function loadAccountPanel() {
      const errEl = document.getElementById("accountLoadErr");
      const detail = document.getElementById("accountDetail");
      if (!errEl || !detail) return;
      errEl.textContent = "";
      try {
        const a = await api("/api/account");
        if (a.uiPreferences) persistUiPrefsCache(a.uiPreferences);
        const pd = document.getElementById("prefDensity");
        const pr = document.getElementById("prefReading");
        if (pd && a.uiPreferences) pd.value = a.uiPreferences.density || "default";
        if (pr && a.uiPreferences) pr.value = a.uiPreferences.readingComfort || "default";
        wireAccountUiPrefsOnce();
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

    function activateMainTab(idx) {
      mainNav.querySelectorAll("button").forEach((b, j) => b.classList.toggle("active", j === idx));
      panelsHost.querySelectorAll(".panel").forEach((p) => {
        const i = parseInt(p.getAttribute("data-tab-idx"), 10);
        p.classList.toggle("active", i === idx);
      });
      if (mainTabLabels[idx] === "Mon compte") loadAccountPanel().catch(showErr);
      if (mainTabLabels[idx] === "Super Admin") {
        loadSuperAdminAccounts().catch(showErr);
        loadSuperAdminSagaTelemetry().catch(function() {});
      }
    }

    function renderMainTabs(labels) {
      mainTabLabels = labels;
      mainNav.innerHTML = "";
      mountPanelsFromTemplate();
      panelsHost.querySelectorAll(".panel").forEach((el) => {
        const i = parseInt(el.getAttribute("data-tab-idx"), 10);
        if (i >= labels.length) el.remove();
      });
      labels.forEach((label, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.className = "main-tab";
        const idx = i;
        b.onclick = () => activateMainTab(idx);
        if (i === 0) b.classList.add("active");
        mainNav.appendChild(b);
      });
      panelsHost.querySelectorAll(".panel").forEach((p) => {
        const i = parseInt(p.getAttribute("data-tab-idx"), 10);
        p.classList.toggle("active", i === 0);
      });
      wireProjectAndEntries();
    }

    async function loadSuperAdminAccounts() {
      const errEl = document.getElementById("saAccountsErr");
      const sumEl = document.getElementById("saAccountsSummary");
      const wrap = document.getElementById("saAccountsTableWrap");
      if (!errEl || !sumEl || !wrap) return;
      errEl.textContent = "";
      const pgEl = document.getElementById("saAccountsPage");
      const psEl = document.getElementById("saAccountsPageSize");
      const page = Math.max(1, parseInt(pgEl && pgEl.value, 10) || 1);
      const pageSize = parseInt(psEl && psEl.value, 10) || 25;
      try {
        const data = await api(
          "/api/super-admin/accounts?page=" + encodeURIComponent(page) + "&page_size=" + encodeURIComponent(pageSize)
        );
        if (pgEl) pgEl.value = String(data.page);
        sumEl.textContent =
          "Comptes actifs : " + data.totalActiveAccounts + " — page " + data.page + " / " + data.totalPages
          + " (" + data.pageSize + " par page).";
        let html = "<table class='sa-accounts'><thead><tr>"
          + "<th>Courriel</th><th>Nom affiché</th><th>Super admin</th><th>Projets</th>"
          + "<th>Dernière connexion</th><th>Entrées (tot. / valid.)</th>"
          + "</tr></thead><tbody>";
        for (const a of data.accounts) {
          const ll = a.lastLoginAt ? escapeHtml(a.lastLoginAt) : "—";
          html += "<tr><td>" + escapeHtml(a.email) + "</td><td>" + escapeHtml(a.displayName) + "</td><td>"
            + (a.isSuperAdmin ? "oui" : "non") + "</td><td>" + a.ownedProjects + "</td><td>" + ll + "</td><td>"
            + a.entriesTotal + " / " + a.entriesValidated + "</td></tr>";
        }
        html += "</tbody></table>";
        if (!data.accounts.length) html = "<p class='muted'>Aucun compte sur cette page.</p>";
        wrap.innerHTML = html;
      } catch (e) {
        wrap.innerHTML = "";
        if (e && e.error) errEl.textContent = (e.error.message || "") + " (" + (e.error.code || "") + ")";
        else errEl.textContent = "Impossible de charger l’annuaire.";
      }
    }

    function wireSuperAdminInvite() {
      const btn = document.getElementById("btnSaInvite");
      const msg = document.getElementById("saInviteMsg");
      const inp = document.getElementById("saInviteEmail");
      if (!btn || !msg || !inp) return;
      btn.onclick = async () => {
        msg.textContent = "";
        msg.className = "muted";
        const email = inp.value.trim();
        if (!email) {
          msg.textContent = "Saisis une adresse e-mail.";
          msg.className = "err";
          return;
        }
        btn.disabled = true;
        try {
          const out = await api("/api/super-admin/invite", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email }),
          });
          msg.textContent = out.message;
          msg.className = out.mailMode === "smtp" ? "ok" : "warn";
          loadSuperAdminAccounts().catch(function() {});
        } catch (e) {
          if (e && e.error) {
            msg.textContent = (e.error.title || "") + "\\n" + (e.error.message || "");
            msg.className = "err";
          } else {
            msg.textContent = "Erreur inattendue.";
            msg.className = "err";
          }
        } finally {
          btn.disabled = false;
        }
      };
    }

    function wireSuperAdminAccountsPanel() {
      wireSuperAdminInvite();
      const b = document.getElementById("btnSaAccountsReload");
      if (b) b.onclick = function() { loadSuperAdminAccounts().catch(function() {}); };
      const ps = document.getElementById("saAccountsPageSize");
      if (ps) ps.onchange = function() {
        const pg = document.getElementById("saAccountsPage");
        if (pg) pg.value = "1";
      };
      wireSuperAdminSagaPanel();
    }

    function saSagaDlqTableRows(ops) {
      if (!ops || !ops.length) return "<p class='muted'>(aucune ligne en quarantaine dans l'aperçu)</p>";
      let h = "<table class='sa-saga'><thead><tr><th>Opération</th><th>Cible</th><th>Tentatives</th><th>Erreur</th></tr></thead><tbody>";
      for (const o of ops) {
        h += "<tr><td>" + escapeHtml(o.operationId) + "</td><td>" + escapeHtml(o.targetUserId) + "</td><td>"
          + o.retryCount + "</td><td>" + escapeHtml((o.lastError || "").slice(0, 120)) + "</td></tr>";
      }
      h += "</tbody></table>";
      return h;
    }

    function saSagaOpsTable(title, ops) {
      if (!ops || !ops.length) return "<h4>" + title + "</h4><p class='muted'>(vide)</p>";
      let h = "<h4>" + title + "</h4><table class='sa-saga'><thead><tr><th>Opération</th><th>État</th><th>Cible</th></tr></thead><tbody>";
      for (const o of ops) {
        h += "<tr><td>" + escapeHtml(o.operationId) + "</td><td>" + escapeHtml(o.state) + "</td><td>"
          + escapeHtml(o.targetUserId) + "</td></tr>";
      }
      h += "</tbody></table>";
      return h;
    }

    function applySuperAdminSagaTelemetry(data) {
      const sumEl = document.getElementById("saSagaSummary");
      const tblEl = document.getElementById("saSagaTables");
      const sel = document.getElementById("saSagaDlqSelect");
      const chk = document.getElementById("saSagaReplayConfirm");
      const btn = document.getElementById("btnSaSagaReplay");
      if (!sumEl || !tblEl || !sel || !chk || !btn) return;
      const w = data.stateCountsInRecentWindow || {};
      const g = data.totalsByState || {};
      sumEl.innerHTML =
        "<p><strong>Fenêtre (" + data.recentOpsLimit + " dernières mises à jour)</strong> — en attente : "
        + (w.pending || 0) + ", fournisseur : " + (w.provider_done || 0) + ", échec : " + (w.failed || 0)
        + ", quarantaine : " + (w.quarantined || 0) + "</p>"
        + "<p><strong>Totaux (toute la table)</strong> — pending : " + (g.pending || 0) + ", fournisseur : "
        + (g.provider_done || 0) + ", base : " + (g.db_done || 0) + ", terminé : " + (g.completed || 0)
        + ", échec : " + (g.failed || 0) + ", quarantaine : " + (g.quarantined || 0)
        + " — file (aperçu " + (data.retryQueuePreviewLimit || 0) + ") : "
        + ((data.retryQueueOps && data.retryQueueOps.length) || 0) + " op.</p>";
      tblEl.innerHTML =
        saSagaOpsTable("Opérations récentes (aperçu)", data.recentOps || [])
        + saSagaOpsTable("File de retry (aperçu)", data.retryQueueOps || [])
        + "<h4>Quarantaine (aperçu)</h4>" + saSagaDlqTableRows(data.dlqOps || []);
      sel.innerHTML = "";
      const dlq = data.dlqOps || [];
      for (const o of dlq) {
        const opt = document.createElement("option");
        opt.value = o.operationId;
        opt.textContent = o.operationId + " → " + o.targetUserId;
        sel.appendChild(opt);
      }
      chk.checked = false;
      btn.disabled = true;
      if (!dlq.length) {
        sel.disabled = true;
        chk.disabled = true;
      } else {
        sel.disabled = false;
        chk.disabled = false;
      }
    }

    function wireSuperAdminSagaPanel() {
      const btnReload = document.getElementById("btnSaSagaReload");
      if (btnReload) btnReload.onclick = function() { loadSuperAdminSagaTelemetry().catch(function() {}); };
      const chk = document.getElementById("saSagaReplayConfirm");
      const sel = document.getElementById("saSagaDlqSelect");
      const btn = document.getElementById("btnSaSagaReplay");
      function syncReplayBtn() {
        if (!btn || !chk || !sel) return;
        btn.disabled = !(chk.checked && sel.value && !sel.disabled);
      }
      if (chk) chk.onchange = syncReplayBtn;
      if (sel) sel.onchange = syncReplayBtn;
      if (btn) btn.onclick = async function() {
        const msg = document.getElementById("saSagaReplayMsg");
        if (msg) { msg.textContent = ""; msg.className = "muted"; }
        if (!sel || !sel.value) return;
        btn.disabled = true;
        try {
          const out = await api("/api/super-admin/saga/replay-quarantined", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ confirm: true, operationId: sel.value }),
          });
          if (msg) {
            msg.textContent = out.message || "OK";
            msg.className = "ok";
          }
          if (out.telemetry) applySuperAdminSagaTelemetry(out.telemetry);
        } catch (e) {
          if (msg) {
            msg.className = "err";
            if (e && e.error) msg.textContent = (e.error.title || "") + "\\n" + (e.error.message || "");
            else msg.textContent = "Relance impossible.";
          }
        } finally {
          syncReplayBtn();
        }
      };
    }

    async function loadSuperAdminSagaTelemetry() {
      const errEl = document.getElementById("saSagaErr");
      if (errEl) errEl.textContent = "";
      try {
        const data = await api("/api/super-admin/saga/telemetry");
        applySuperAdminSagaTelemetry(data);
      } catch (e) {
        if (errEl) {
          if (e && e.error) errEl.textContent = (e.error.message || "") + " (" + (e.error.code || "") + ")";
          else errEl.textContent = "Impossible de charger la télémétrie saga.";
        }
      }
    }

    function wireEditionFilterControls() {
      const onFilterChange = () => {
        persistEditionFiltersToSession();
        loadEntries().catch(showErr);
      };
      ["editionStatutFilter", "editionScoreMode", "editionScoreThreshold", "editionScoreDecile"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.onchange = onFilterChange;
      });
      const inc = document.getElementById("editionScoreIncludeNa");
      if (inc) inc.onchange = onFilterChange;
    }

    function assistSetBusy(prefix, busy) {
      const busyEl = document.getElementById("assistBusy" + prefix);
      const det = busyEl ? busyEl.closest("details.curator-assist") : null;
      if (busyEl) busyEl.style.display = busy ? "block" : "none";
      if (det) det.classList.toggle("htmx-request", !!busy);
      [
        "btnAssistLlmDo" + prefix,
        "btnAssistLlmOd" + prefix,
        "btnAssistLlmInsert" + prefix,
        "btnAssistLlmDismiss" + prefix,
        "btnAssistLt" + prefix,
        "btnAssistLtReplace" + prefix,
        "btnAssistLtDismiss" + prefix,
      ].forEach((bid) => {
        const b = document.getElementById(bid);
        if (b) b.disabled = !!busy;
      });
    }

    function assistFillSelect(selectId, values) {
      const el = document.getElementById(selectId);
      if (!el) return;
      el.innerHTML = "<option value=\"\">—</option>";
      for (const v of values || []) {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = v;
        el.appendChild(o);
      }
    }

    async function assistLoadDimensions(prefix) {
      const pid = document.getElementById("projectSel").value;
      if (!pid) {
        throw { error: { title: "Projet", message: "Sélectionnez un projet actif.", code: "CLIENT" } };
      }
      if (!curatorDimsCache) {
        curatorDimsCache = await api(
          "/api/projects/" + encodeURIComponent(pid) + "/curator/dimensions"
        );
      }
      const dims = curatorDimsCache.dimensions || {};
      const p = prefix;
      assistFillSelect("assist" + p + "Type", dims.types);
      assistFillSelect("assist" + p + "Structure", dims.structures);
      assistFillSelect("assist" + p + "Ton", dims.tons);
      assistFillSelect("assist" + p + "Format", dims.formats);
      assistFillSelect("assist" + p + "Public", dims.publics);
    }

    function assistDimPayload(prefix) {
      const g = (id) => {
        const el = document.getElementById(id);
        return el && el.value ? el.value : "";
      };
      const p = prefix;
      return {
        type: g("assist" + p + "Type"),
        structure: g("assist" + p + "Structure"),
        ton: g("assist" + p + "Ton"),
        format: g("assist" + p + "Format"),
        public: g("assist" + p + "Public"),
      };
    }

    function assistClearLlmUi(prefix) {
      const preview = document.getElementById("assistLlmPreview" + prefix);
      const actions = document.getElementById("assistLlmActions" + prefix);
      if (preview) {
        preview.hidden = true;
        preview.innerHTML = "";
      }
      if (actions) actions.hidden = true;
      window["__llmText" + prefix] = null;
      window["__llmMode" + prefix] = null;
    }

    function assistClearLtUi(prefix) {
      const preview = document.getElementById("assistLtPreview" + prefix);
      const actions = document.getElementById("assistLtActions" + prefix);
      if (preview) {
        preview.hidden = true;
        preview.innerHTML = "";
      }
      if (actions) actions.hidden = true;
      window["__ltCorrected" + prefix] = null;
    }

    function assistClearAssistMsg(prefix) {
      const msg = document.getElementById("assistMsg" + prefix);
      if (msg) {
        msg.textContent = "";
        msg.className = "muted";
      }
    }

    function wireCuratorAssistance() {
      function bindLlm(prefix, inputId, outputId) {
        const doBtn = document.getElementById("btnAssistLlmDo" + prefix);
        const odBtn = document.getElementById("btnAssistLlmOd" + prefix);
        const run = async (mode) => {
          const msg = document.getElementById("assistMsg" + prefix);
          assistClearLlmUi(prefix);
          assistClearLtUi(prefix);
          assistClearAssistMsg(prefix);
          assistSetBusy(prefix, true);
          try {
            await assistLoadDimensions(prefix);
            const pid = document.getElementById("projectSel").value;
            const dim = assistDimPayload(prefix);
            const inp = document.getElementById(inputId);
            const out = document.getElementById(outputId);
            const body = {
              mode: mode,
              input: inp ? inp.value : "",
              output: out ? out.value : "",
              type: dim.type,
              structure: dim.structure,
              ton: dim.ton,
              format: dim.format,
              public: dim.public,
            };
            const res = await api(
              "/api/projects/" + encodeURIComponent(pid) + "/curator/llm-generate",
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
              }
            );
            if (!msg) return;
            if (res.status === "validation_error") {
              msg.textContent = res.message || "Saisie invalide.";
              msg.className = "err";
              return;
            }
            if (res.status === "failed") {
              msg.textContent = res.message || "La génération a échoué.";
              msg.className = "err";
              return;
            }
            if (res.status === "ok" && res.text != null) {
              window["__llmText" + prefix] = res.text;
              window["__llmMode" + prefix] = mode;
              if (msg) {
                msg.textContent =
                  mode === "draft_to_output"
                    ? "Génération prête : examinez la proposition puis insérez-la dans output si elle convient."
                    : "Génération prête : examinez la proposition puis insérez-la dans le brouillon (input) si elle convient.";
                msg.className = "ok";
              }
              const lp = document.getElementById("assistLlmPreview" + prefix);
              const la = document.getElementById("assistLlmActions" + prefix);
              const insBtn = document.getElementById("btnAssistLlmInsert" + prefix);
              if (lp) {
                lp.hidden = false;
                lp.innerHTML =
                  "<strong>Proposition IA</strong><pre style='white-space:pre-wrap'>"
                  + escapeHtml(res.text) + "</pre>";
              }
              if (insBtn) {
                insBtn.textContent =
                  mode === "draft_to_output"
                    ? "Insérer dans output"
                    : "Insérer dans le brouillon (input)";
              }
              if (la) la.hidden = false;
              return;
            }
            msg.textContent = "Réponse inattendue du service.";
            msg.className = "err";
          } catch (e) {
            if (msg) {
              msg.className = "err";
              if (e && e.error) {
                msg.textContent =
                  (e.error.title || "") + "\\n" + (e.error.message || "")
                  + (e.error.suggested_action ? "\\n" + e.error.suggested_action : "");
              } else {
                msg.textContent = "Erreur réseau ou serveur.";
              }
            }
          } finally {
            assistSetBusy(prefix, false);
          }
        };
        if (doBtn) doBtn.onclick = () => run("draft_to_output");
        if (odBtn) odBtn.onclick = () => run("output_to_draft");
        const insLlm = document.getElementById("btnAssistLlmInsert" + prefix);
        const disLlm = document.getElementById("btnAssistLlmDismiss" + prefix);
        if (insLlm) {
          insLlm.onclick = () => {
            const text = window["__llmText" + prefix];
            const mode = window["__llmMode" + prefix];
            const targetId = mode === "draft_to_output" ? outputId : inputId;
            const tel = document.getElementById(targetId);
            if (tel && text != null) {
              tel.value = text;
              tel.focus();
            }
            assistClearLlmUi(prefix);
            const m = document.getElementById("assistMsg" + prefix);
            if (m) {
              m.textContent = "Texte IA inséré dans le champ (non enregistré en base).";
              m.className = "ok";
            }
          };
        }
        if (disLlm) {
          disLlm.onclick = () => {
            assistClearLlmUi(prefix);
            const m = document.getElementById("assistMsg" + prefix);
            if (m) {
              m.textContent = "Proposition IA ignorée.";
              m.className = "muted";
            }
          };
        }
      }
      function bindLt(prefix, outputId) {
        const ltBtn = document.getElementById("btnAssistLt" + prefix);
        const repBtn = document.getElementById("btnAssistLtReplace" + prefix);
        const disBtn = document.getElementById("btnAssistLtDismiss" + prefix);
        if (ltBtn) {
          ltBtn.onclick = async () => {
            const msg = document.getElementById("assistMsg" + prefix);
            const preview = document.getElementById("assistLtPreview" + prefix);
            const actions = document.getElementById("assistLtActions" + prefix);
            assistSetBusy(prefix, true);
            assistClearLlmUi(prefix);
            assistClearLtUi(prefix);
            assistClearAssistMsg(prefix);
            try {
              const pid = document.getElementById("projectSel").value;
              const outEl = document.getElementById(outputId);
              const text = outEl ? outEl.value : "";
              const res = await api(
                "/api/projects/" + encodeURIComponent(pid) + "/curator/languagetool-check",
                {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ text: text }),
                }
              );
              if (res.status === "validation_error") {
                if (msg) {
                  msg.textContent = res.message || "Texte vide.";
                  msg.className = "err";
                }
                return;
              }
              if (res.status === "ok") {
                if (msg) {
                  msg.textContent =
                    "Contrôle terminé : proposition affichée (aucune écriture en base tant que vous n'enregistrez pas).";
                  msg.className = "ok";
                }
                window["__ltCorrected" + prefix] = res.corrected;
                if (preview) {
                  preview.hidden = false;
                  preview.innerHTML =
                    "<strong>Texte corrigé proposé</strong><pre style='white-space:pre-wrap'>"
                    + escapeHtml(res.corrected) + "</pre>";
                }
                if (actions) actions.hidden = false;
                return;
              }
              if (msg) {
                msg.textContent = "Réponse LanguageTool inattendue.";
                msg.className = "err";
              }
            } catch (e) {
              if (msg) {
                msg.className = "err";
                if (e && e.error) {
                  msg.textContent =
                    (e.error.title || "") + "\\n" + (e.error.message || "")
                    + (e.error.suggested_action ? "\\n" + e.error.suggested_action : "");
                } else {
                  msg.textContent = "Échec du contrôle LanguageTool.";
                }
              }
            } finally {
              assistSetBusy(prefix, false);
            }
          };
        }
        if (repBtn) {
          repBtn.onclick = () => {
            const corrected = window["__ltCorrected" + prefix];
            const outEl = document.getElementById(outputId);
            if (outEl && corrected != null) {
              outEl.value = corrected;
              outEl.focus();
            }
            const msg = document.getElementById("assistMsg" + prefix);
            if (msg) {
              msg.textContent = "Proposition appliquée au champ output (non enregistrée).";
              msg.className = "ok";
            }
            assistClearLtUi(prefix);
          };
        }
        if (disBtn) {
          disBtn.onclick = () => {
            assistClearLtUi(prefix);
            const msg = document.getElementById("assistMsg" + prefix);
            if (msg) {
              msg.textContent = "Proposition ignorée.";
              msg.className = "muted";
            }
          };
        }
      }
      bindLlm("Edit", "fldInput", "fldOutput");
      bindLlm("New", "newInput", "newOutput");
      bindLt("Edit", "fldOutput");
      bindLt("New", "newOutput");
    }

    function wireProjectAndEntries() {
      const ps = document.getElementById("projectSel");
      if (ps) ps.onchange = onProjectChanged;
      const b1 = document.getElementById("btnReloadEntries");
      if (b1) b1.onclick = () => loadEntries().catch(showErr);
      const b2 = document.getElementById("btnSave");
      if (b2) b2.onclick = () => saveEntry();
      const b3 = document.getElementById("btnCreate");
      if (b3) b3.onclick = () => createEntry();
      const b4 = document.getElementById("btnCreateProject");
      if (b4) b4.onclick = () => createProject();
      const b5 = document.getElementById("btnDeleteProject");
      if (b5) b5.onclick = () => deleteProject();
      const b6 = document.getElementById("btnCsv");
      if (b6) b6.onclick = () => downloadCsv();
      const b7 = document.getElementById("btnJsonl");
      if (b7) b7.onclick = () => downloadJsonl();
      wireEditionFilterControls();
      wireSuperAdminAccountsPanel();
      wireCuratorAssistance();
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
        const me = await api("/api/me");
        document.getElementById("userLine").textContent =
          me.user.displayName + " · " + me.user.email + (me.user.isSuperAdmin ? " · super admin" : "");
        renderMainTabs(me.mainTabLabels);
        await syncUiPrefsFromAccount();
        await loadProjects();
      } catch (e) { showErr(e); }
    };

    document.getElementById("btnSignout").onclick = async () => {
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
      sessionStorage.removeItem(SS);
      sessionStorage.removeItem(UIPREFS_SS);
      document.documentElement.removeAttribute("data-ds-density");
      document.documentElement.removeAttribute("data-ds-reading");
      window.location.assign(target);
    };

    if (token()) syncUiPrefsFromAccount().catch(function() {});

    async function loadProjects() {
      const hint = activeProjectHint();
      const q = hint ? ("?active_hint=" + encodeURIComponent(hint)) : "";
      const data = await api("/api/projects" + q);
      const sel = document.getElementById("projectSel");
      if (!sel) return;
      sel.innerHTML = "";
      for (const p of data.projects) {
        const o = document.createElement("option");
        o.value = p.id;
        o.textContent = p.name + " (" + p.role + ")";
        sel.appendChild(o);
      }
      const resolved = data.activeProjectId || "";
      if (resolved) {
        sel.value = resolved;
        setActiveProjectHint(resolved);
      } else {
        setActiveProjectHint("");
      }
      await loadEntries();
    }

    function onProjectChanged() {
      const sel = document.getElementById("projectSel");
      const pid = sel ? sel.value : "";
      curatorDimsCache = null;
      setActiveProjectHint(pid);
      clearEntryState();
      restoreEditionFiltersFromSession();
      loadEntries().catch(showErr);
    }

    async function loadEntries() {
      const sel = document.getElementById("projectSel");
      const pid = sel ? sel.value : "";
      if (!pid) return;
      restoreEditionFiltersFromSession();
      const qs = buildEntriesQueryString();
      const statusEl = document.getElementById("entriesStatus");
      const reloadBtn = document.getElementById("btnReloadEntries");
      if (reloadBtn) {
        reloadBtn.setAttribute("aria-busy", "true");
        reloadBtn.disabled = true;
      }
      try {
        const data = await api("/api/projects/" + encodeURIComponent(pid) + "/entries" + qs);
        renderEntriesTableFromPayload(data.entries);
        if (statusEl) {
          const n = (data.entries || []).length;
          statusEl.textContent = n + " fiche(s) dans la vue (alignée sur le serveur).";
        }
      } catch (e) {
        showErr(e);
      } finally {
        if (reloadBtn) {
          reloadBtn.removeAttribute("aria-busy");
          reloadBtn.disabled = false;
        }
      }
    }

    async function saveEntry() {
      const pid = document.getElementById("projectSel").value;
      const eid = document.getElementById("entryId").value.trim();
      const input = document.getElementById("fldInput").value;
      const output = document.getElementById("fldOutput").value;
      const qs = buildEntriesQueryString();
      const btn = document.getElementById("btnSave");
      if (btn) {
        btn.setAttribute("aria-busy", "true");
        btn.disabled = true;
      }
      try {
        const out = await api(
          "/api/projects/" + encodeURIComponent(pid) + "/entries/" + encodeURIComponent(eid) + qs,
          {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ input, output }),
          }
        );
        showOk("Fiche enregistrée.");
        renderEntriesTableFromPayload(out.entries || []);
        syncFormFieldsFromEntryId(out.entries || [], eid);
      } catch (e) {
        showErr(e);
      } finally {
        if (btn) {
          btn.removeAttribute("aria-busy");
          btn.disabled = false;
        }
      }
    }

    async function createEntry() {
      const pid = document.getElementById("projectSel").value;
      const input = document.getElementById("newInput").value;
      const output = document.getElementById("newOutput").value;
      const qs = buildEntriesQueryString();
      const btn = document.getElementById("btnCreate");
      if (btn) {
        btn.setAttribute("aria-busy", "true");
        btn.disabled = true;
      }
      try {
        const out = await api("/api/projects/" + encodeURIComponent(pid) + "/entries" + qs, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ input, output }),
        });
        document.getElementById("entryId").value = out.id;
        showOk("Fiche créée : " + out.id);
        renderEntriesTableFromPayload(out.entries || []);
        syncFormFieldsFromEntryId(out.entries || [], out.id);
      } catch (e) {
        showErr(e);
      } finally {
        if (btn) {
          btn.removeAttribute("aria-busy");
          btn.disabled = false;
        }
      }
    }

    async function createProject() {
      const name = document.getElementById("newProjectName").value.trim();
      const description = document.getElementById("newProjectDesc").value.trim();
      try {
        const out = await api("/api/projects", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, description }),
        });
        setActiveProjectHint(out.id);
        document.getElementById("newProjectName").value = "";
        document.getElementById("newProjectDesc").value = "";
        showOk("Projet créé.");
        clearEntryState();
        await loadProjects();
      } catch (e) { showErr(e); }
    }

    async function deleteProject() {
      const sel = document.getElementById("projectSel");
      const pid = sel ? sel.value : "";
      if (!pid) return;
      const opt = sel.selectedOptions[0];
      const raw = opt ? opt.textContent : "";
      const pname = raw.replace(/\\s+\\([^)]+\\)\\s*$/, "").trim();
      if (!document.getElementById("delProjConfirm").checked) {
        showErr({ error: { title: "Confirmation", message: "Coche la case de confirmation.", code: "CLIENT" } });
        return;
      }
      if (document.getElementById("delProjName").value.trim() !== pname) {
        showErr({ error: { title: "Nom incorrect", message: "Le nom tapé ne correspond pas au projet actif.", code: "CLIENT" } });
        return;
      }
      try {
        await api("/api/projects/" + encodeURIComponent(pid), { method: "DELETE" });
        setActiveProjectHint("");
        document.getElementById("delProjConfirm").checked = false;
        document.getElementById("delProjName").value = "";
        showOk("Projet supprimé.");
        clearEntryState();
        await loadProjects();
      } catch (e) { showErr(e); }
    }

    function scopeParam() {
      return "?scope=" + encodeURIComponent(document.getElementById("exportScope").value);
    }

    async function downloadCsv() {
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
    }

    async function downloadJsonl() {
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
    }

    (async function restoreSession() {
      if (!token()) return;
      try {
        const me = await api("/api/me");
        setToken(token());
        document.getElementById("userLine").textContent =
          me.user.displayName + " · " + me.user.email + (me.user.isSuperAdmin ? " · super admin" : "");
        renderMainTabs(me.mainTabLabels);
        await loadProjects();
      } catch (_) {
        setToken("");
      }
    })();
  </script>
</body>
</html>
"""
