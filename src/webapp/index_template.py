"""Page d'accueil HTML du service ``webapp`` (shell curateur issue-010 + slice issue-007).

Les couleurs et le bandeau sémantique sont dans ``static/design_tokens.css`` (issue-022).
Le mapping ``error.code`` → variant est injecté depuis ``ui_semantics`` (pas d'inférence HTTP).
"""

from src.webapp.ui_semantics import api_error_banner_variant_json_for_index_script

# Gabarit ``GET /`` : espaces réservés remplis au chargement du module.
_RAW_INDEX_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dataset Style — curateur</title>
  <link rel="stylesheet" href="/static/design_tokens.css" />
</head>
<body>
  <div class="wrap">
    <!--DS_MIGRATION_BANNER_PLACEHOLDER-->
    <h1>Dataset Style — coquille curateur</h1>
    <div class="shell-lede">Shell de navigation aligné sur Streamlit (ordre des onglets via <code>/api/me</code>).
      Streamlit reste sur le port <code>8501</code> ; ce service <code>webapp</code> porte le slice API + UI minimale.</div>

    <section id="auth">
      <h2>Connexion</h2>
      <label>Email <input type="email" id="email" autocomplete="username" /></label>
      <label>Mot de passe <input type="password" id="password" autocomplete="current-password" /></label>
      <div class="row"><button type="button" id="btnSignin">Se connecter</button></div>
      <div class="row"><button type="button" id="btnSignout">Se déconnecter</button></div>
      <div id="authMsg" class="ds-banner-stack" aria-live="polite"></div>
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
    </div>
    <div class="panel" data-tab-idx="3">
      <h2>Gestion &amp; édition</h2>
      <p><button type="button" id="btnReloadEntries">Recharger les entrées</button></p>
      <div id="entriesTable"></div>
      <h3>Édition (id de fiche)</h3>
      <label>id <input type="text" id="entryId" /></label>
      <label>input <textarea id="fldInput"></textarea></label>
      <label>output <textarea id="fldOutput"></textarea></label>
      <div class="row"><button type="button" id="btnSave">Enregistrer</button></div>
    </div>
    <div class="panel" data-tab-idx="4">
      <h2>Tableau de bord</h2>
      <p class="muted">Alertes qualité dataset (issue-014) : chargement via <code>GET /api/projects/…/dashboard</code>.</p>
      <div id="dashBannerStack" class="ds-banner-stack" aria-live="polite"></div>
      <p id="dashMetricsHint" class="muted"></p>
    </div>
    <div class="panel" data-tab-idx="5">
      <h2>Mon compte</h2>
      <p class="muted">Profil curateur (issue-016) — pas d’indicateurs super-admin.</p>
      <div id="accountDetail" class="account-dl"></div>
      <div id="accountLoadErr" class="ds-banner-stack" aria-live="polite"></div>
    </div>
    <div class="panel" data-tab-idx="6">
      <h2>Super Admin</h2>
      <p class="muted">Invitation d’un collaborateur (invitation-only). Aucune inscription publique.</p>
      <label>E-mail du collaborateur <input type="email" id="saInviteEmail" autocomplete="off" /></label>
      <div class="row"><button type="button" id="btnSaInvite">Envoyer l’invitation</button></div>
      <div id="saInviteMsg" class="ds-banner-stack" aria-live="polite"></div>
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
      <div id="saAccountsErr" class="ds-banner-stack" aria-live="polite"></div>
      <p id="saAccountsSummary" class="muted"></p>
      <div id="saAccountsTableWrap"></div>
      <h3>Panneau technique (saga)</h3>
      <p class="muted">Métriques alignées sur le studio : cartes = répartition sur les N dernières mises à jour ;
        totaux = ensemble de la table. File = opérations éligibles au worker (<code>retry_deprovision_ops</code>).</p>
      <p><button type="button" id="btnSaSagaReload">Actualiser la télémétrie</button></p>
      <div id="saSagaErr" class="ds-banner-stack" aria-live="polite"></div>
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
        <div id="saSagaReplayMsg" class="ds-banner-stack" aria-live="polite"></div>
      </div>
    </div>
  </template>

  <script>
    const API_ERROR_BANNER_VARIANT = __API_ERROR_BANNER_VARIANT_JSON__;
    const LS = "slice_vertical_access_token";
    const SS = "webapp_active_project_id";
    const authMsg = document.getElementById("authMsg");
    const workspace = document.getElementById("workspace");
    const mainNav = document.getElementById("mainNav");
    const panelsHost = document.getElementById("panels");
    let mainTabLabels = [];
    let activeMainTabIdx = 0;

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
    }

    function apiBannerVariantForCode(code) {
      const v = API_ERROR_BANNER_VARIANT[code];
      return v || "danger";
    }

    function datasetAlertVariantFromSeverity(sev) {
      return sev === "info" ? "info" : "warning";
    }

    function clearDsBannerStack(host) {
      if (!host) return;
      host.innerHTML = "";
      host.className = "ds-banner-stack";
    }

    function fillDsBanner(el, variant, titleText, messageText) {
      el.className = "ds-banner ds-banner--" + variant;
      el.setAttribute("role", "status");
      el.innerHTML = "";
      const t = document.createElement("strong");
      t.className = "ds-banner__title";
      t.textContent = titleText || "";
      const m = document.createElement("p");
      m.className = "ds-banner__message";
      m.textContent = messageText || "";
      el.appendChild(t);
      el.appendChild(m);
    }

    function appendBannerToStack(stack, variant, titleText, messageText) {
      const inner = document.createElement("div");
      stack.appendChild(inner);
      fillDsBanner(inner, variant, titleText, messageText);
    }

    function renderApiErrorIntoStack(stack, err) {
      clearDsBannerStack(stack);
      if (!err) return;
      const variant = apiBannerVariantForCode(err.code || "");
      const title = err.title || "Erreur";
      const parts = [err.message || "", err.suggested_action || "", err.code ? "code : " + err.code : ""].filter(Boolean);
      appendBannerToStack(stack, variant, title, parts.join(String.fromCharCode(10)));
    }

    function showErr(obj) {
      clearDsBannerStack(authMsg);
      if (obj && obj.error) {
        renderApiErrorIntoStack(authMsg, obj.error);
      } else {
        appendBannerToStack(authMsg, "danger", "Erreur", typeof obj === "string" ? obj : JSON.stringify(obj));
      }
    }
    function showOk(msg) {
      clearDsBannerStack(authMsg);
      appendBannerToStack(authMsg, "success", "Succès", msg || "OK");
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

    async function loadAccountPanel() {
      const errEl = document.getElementById("accountLoadErr");
      const detail = document.getElementById("accountDetail");
      if (!errEl || !detail) return;
      clearDsBannerStack(errEl);
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
        if (e && e.error) renderApiErrorIntoStack(errEl, e.error);
        else appendBannerToStack(errEl, "danger", "Profil", "Impossible de charger le profil.");
      }
    }

    function activateMainTab(idx) {
      activeMainTabIdx = idx;
      mainNav.querySelectorAll("button").forEach((b, j) => b.classList.toggle("active", j === idx));
      panelsHost.querySelectorAll(".panel").forEach((p) => {
        const i = parseInt(p.getAttribute("data-tab-idx"), 10);
        p.classList.toggle("active", i === idx);
      });
      if (mainTabLabels[idx] === "Mon compte") loadAccountPanel().catch(showErr);
      if (mainTabLabels[idx] === "Tableau de bord") loadDashboardBanners().catch(showErr);
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
      clearDsBannerStack(errEl);
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
        if (e && e.error) renderApiErrorIntoStack(errEl, e.error);
        else appendBannerToStack(errEl, "danger", "Annuaire", "Impossible de charger l’annuaire.");
      }
    }

    function wireSuperAdminInvite() {
      const btn = document.getElementById("btnSaInvite");
      const msg = document.getElementById("saInviteMsg");
      const inp = document.getElementById("saInviteEmail");
      if (!btn || !msg || !inp) return;
      btn.onclick = async () => {
        clearDsBannerStack(msg);
        const email = inp.value.trim();
        if (!email) {
          appendBannerToStack(msg, "warning", "Invitation", "Saisis une adresse e-mail.");
          return;
        }
        btn.disabled = true;
        try {
          const out = await api("/api/super-admin/invite", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email }),
          });
          const okSmtp = out.mailMode === "smtp";
          appendBannerToStack(
            msg,
            okSmtp ? "success" : "warning",
            okSmtp ? "Invitation" : "Invitation (mail simulé)",
            out.message || "OK"
          );
          loadSuperAdminAccounts().catch(function() {});
        } catch (e) {
          if (e && e.error) renderApiErrorIntoStack(msg, e.error);
          else appendBannerToStack(msg, "danger", "Invitation", "Erreur inattendue.");
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
        if (msg) clearDsBannerStack(msg);
        if (!sel || !sel.value) return;
        btn.disabled = true;
        try {
          const out = await api("/api/super-admin/saga/replay-quarantined", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ confirm: true, operationId: sel.value }),
          });
          if (msg) appendBannerToStack(msg, "success", "Relance", out.message || "OK");
          if (out.telemetry) applySuperAdminSagaTelemetry(out.telemetry);
        } catch (e) {
          if (msg) {
            if (e && e.error) renderApiErrorIntoStack(msg, e.error);
            else appendBannerToStack(msg, "danger", "Relance", "Relance impossible.");
          }
        } finally {
          syncReplayBtn();
        }
      };
    }

    async function loadSuperAdminSagaTelemetry() {
      const errEl = document.getElementById("saSagaErr");
      if (errEl) clearDsBannerStack(errEl);
      try {
        const data = await api("/api/super-admin/saga/telemetry");
        applySuperAdminSagaTelemetry(data);
      } catch (e) {
        if (errEl) {
          if (e && e.error) renderApiErrorIntoStack(errEl, e.error);
          else appendBannerToStack(errEl, "danger", "Saga", "Impossible de charger la télémétrie saga.");
        }
      }
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
      wireSuperAdminAccountsPanel();
    }

    document.getElementById("btnSignin").onclick = async () => {
      clearDsBannerStack(authMsg);
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
      window.location.assign(target);
    };

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
      setActiveProjectHint(pid);
      clearEntryState();
      loadEntries().catch(showErr);
      if (mainTabLabels[activeMainTabIdx] === "Tableau de bord") loadDashboardBanners().catch(showErr);
    }

    async function loadEntries() {
      const sel = document.getElementById("projectSel");
      const pid = sel ? sel.value : "";
      if (!pid) return;
      const data = await api("/api/projects/" + encodeURIComponent(pid) + "/entries");
      const div = document.getElementById("entriesTable");
      if (!div) return;
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

    async function saveEntry() {
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
    }

    async function createEntry() {
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

    async function loadDashboardBanners() {
      const stack = document.getElementById("dashBannerStack");
      const hint = document.getElementById("dashMetricsHint");
      if (!stack || !hint) return;
      clearDsBannerStack(stack);
      hint.textContent = "";
      const sel = document.getElementById("projectSel");
      const pid = sel ? sel.value : "";
      if (!pid) {
        appendBannerToStack(stack, "info", "Tableau de bord", "Sélectionnez un projet pour afficher les alertes qualité.");
        return;
      }
      try {
        const body = await api(
          "/api/projects/" + encodeURIComponent(pid) + "/dashboard?dashboard_scope=validated"
        );
        const alerts = (body.dataset_quality && body.dataset_quality.alerts) || [];
        for (const a of alerts) {
          const v = datasetAlertVariantFromSeverity(a.severity);
          appendBannerToStack(stack, v, a.title_fr || "Qualité du dataset", a.message_fr || "");
        }
        if (!alerts.length) {
          appendBannerToStack(
            stack,
            "success",
            "Tableau de bord",
            "Aucune alerte qualité pour ce périmètre (fiches validées)."
          );
        }
        hint.textContent = "Périmètre chargé : fiches validées (dashboard_scope=validated), aligné issue-014.";
      } catch (e) {
        clearDsBannerStack(stack);
        if (e && e.error) renderApiErrorIntoStack(stack, e.error);
        else appendBannerToStack(stack, "danger", "Tableau de bord", "Impossible de charger les agrégats.");
      }
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

INDEX_HTML = _RAW_INDEX_HTML.replace(
    "__API_ERROR_BANNER_VARIANT_JSON__",
    api_error_banner_variant_json_for_index_script(),
)
