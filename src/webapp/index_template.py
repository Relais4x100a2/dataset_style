"""Page d'accueil HTML du service ``webapp`` (shell curateur issue-010 + slice issue-007)."""

# Contenu servi tel quel par ``GET /`` ; pas de logique Python ici (uniquement chaîne).
INDEX_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dataset Style — curateur</title>
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
      <p class="muted">Placeholder — parité métriques / stylométrie : sprints suivants.</p>
    </div>
    <div class="panel" data-tab-idx="5">
      <h2>Mon compte</h2>
      <p class="muted">Profil curateur (issue-016) — pas d’indicateurs super-admin.</p>
      <div id="accountDetail" class="account-dl"></div>
      <p id="accountLoadErr" class="err" aria-live="polite"></p>
    </div>
    <div class="panel" data-tab-idx="6">
      <h2>Super Admin</h2>
      <p class="muted">Invitation d’un collaborateur (invitation-only). Aucune inscription publique.</p>
      <label>E-mail du collaborateur <input type="email" id="saInviteEmail" autocomplete="off" /></label>
      <div class="row"><button type="button" id="btnSaInvite">Envoyer l’invitation</button></div>
      <p id="saInviteMsg" class="muted" aria-live="polite"></p>
    </div>
  </template>

  <script>
    const LS = "slice_vertical_access_token";
    const SS = "webapp_active_project_id";
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

    async function loadAccountPanel() {
      const errEl = document.getElementById("accountLoadErr");
      const detail = document.getElementById("accountDetail");
      if (!errEl || !detail) return;
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

    function activateMainTab(idx) {
      mainNav.querySelectorAll("button").forEach((b, j) => b.classList.toggle("active", j === idx));
      panelsHost.querySelectorAll(".panel").forEach((p) => {
        const i = parseInt(p.getAttribute("data-tab-idx"), 10);
        p.classList.toggle("active", i === idx);
      });
      if (mainTabLabels[idx] === "Mon compte") loadAccountPanel().catch(showErr);
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
      wireSuperAdminInvite();
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
