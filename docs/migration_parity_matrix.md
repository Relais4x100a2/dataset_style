# Matrice de parité Streamlit → API (migration)

**Backlog :** issue-004 — publier la matrice versionnée et les jeux de non-régression.  
**Dépendance notée :** issue-003 (règles d’accès projet).  
**Source de vérité de l’ordre des onglets produit :** `EXPECTED_WORKFLOW_TAB_ORDER` et `main_tab_labels()` dans `src/tab_layout.py`, montés dans `main.py` (corps des onglets alignés sur le même ordre).

Cette matrice sert de carte pour exposer une API HTTP (ou équivalent) sans perdre les flux critiques aujourd’hui portés par Streamlit. Les colonnes **issue-010** à **issue-016** matérialisent le **statut de parité** par sprint backlog : elles **doivent être mises à jour** à la clôture de chaque issue correspondante (remplacer `⏳` par `OK`, `N/A` ou `Écart documenté`).

**Primitive persistance partagée :** `src/database.py` (SQLAlchemy / PostgreSQL, schéma multi-tenant). Aucune dépendance Streamlit dans ce module.

---

## Décision stack frontal (ADR 0006 / GitHub #177)

**Statut** : **accepté** (2026-05-19) — jalon UX « écran dense » : liste d’entrées, fiche édition, **Génération IA** et **LanguageTool** sur la webapp (`GET /`, `src/webapp/index_template.py`, routes `src/webapp/app.py` et logique `src/webapp/curator_ai.py`). La trajectoire reste **FastAPI + gabarits HTML + API JSON** (SPA non imposée au jour 1). Critères de notation, redevables, plan B chiffré et conditions de re-décision : **`docs/adr/0006-front-stack-bff-spa-vs-htmx.md`** (section *Jalon UX*).

**Post-mutation (webapp)** : inchangé par rapport au contrat décrit ci-dessous — corps `entries` après succès sur création / patch d’entrées, aligné sur l’équivalent Streamlit (`invalidate_project_entries_cache` + relecture).

---

## Ordre des onglets (alignement code)

Ordre courant après sélection d’un projet (bandeau `st.tabs`) :

1. `Projets`
2. `Réglages & Export`
3. `Nouvelle entrée`
4. `Gestion & édition`
5. `Tableau de bord`
6. `Mon compte`
7. `Super Admin` — **uniquement** si `user.is_super_admin` (`main_tab_labels(include_super_admin=True)`).

Sans projet actif, les onglets workflow 2–5 affichent un message guidé ; la création du premier projet vit sous **Projets** (`render_no_project_onboarding` / issue-028).

---

## Matrice principale (flux critiques)

Légende **Lecture / Écriture (cible API)** : noms indicatifs REST à figer dans l’OpenAPI ; l’implémentation peut différer (GraphQL, RPC) tant que le contrat fonctionnel reste équivalent.

Légende **Post-mutation (équivalent Streamlit)** : aujourd’hui, après chaque écriture qui change les lignes `entries` ou le contexte projet, l’UI appelle `invalidate_project_entries_cache()` (`src/project_entries_cache.py`) puis `st.rerun()` pour que `cached_load_project_entries` relise via `load_project_entries` (`database.py`). Toute API doit exposer un comportement cohérent (invalidation côté client, ETag, ou relecture explicite). **Webapp (issue-012 / #134)** : les réponses `POST` / `PATCH` sur les entrées incluent un tableau `entries` issu d’un `load_project_entries` immédiat après succès ; la coquille `GET /` met à jour le tableau **depuis ce tableau** après création ou enregistrement (pas de `GET` supplémentaire sur ce chemin heureux). Les dimensions fermées sur `PATCH` sont validées côté serveur avec les mêmes listes actives que `POST` (`_resolve_closed_dimension_value`, `src/webapp/entry_mutations.py`). Pour un chargement initial ou après changement de filtres `edition_*`, le client appelle `GET …/entries` comme Streamlit recharge le dataframe après invalidation de cache.

| Onglet Streamlit | ID flux | Lecture (UI / service actuel) | Écriture (UI actuelle) | Post-mutation | Primitives `database.py` (indicatif) | Autres modules |
| --- | --- | --- | --- | --- | --- | --- |
| *(sidebar)* | SB-CTX | `list_projects_for_user` | — | `st.session_state["project_id"]`, `project_role` | `list_projects_for_user` | `src/ui_components.render_sidebar`, `src/project_session` |
| Projets | PRJ-VIEW | `list_projects_for_user` | — | — | `list_projects_for_user` | `render_tab_projects` |
| Projets | PRJ-CREATE | — | formulaire création | `invalidate_project_entries_cache`, `project_id` ← nouveau, `st.rerun` | `create_project` | `render_no_project_onboarding` / `_render_project_create_form` |
| Projets | PRJ-DELETE | — | suppression gardée | `invalidate_project_entries_cache`, retrait `project_id`, flash, `st.rerun` | `delete_project_as_admin` | `_render_project_delete_guarded_form` |
| Réglages & Export | SET-READ | `get_project_settings` | — | — | `get_project_settings` | `_render_project_settings_form`, `src/presets` |
| Réglages & Export | SET-WRITE | — | `update_project_settings_as_admin` | `st.rerun` (pas d’invalidation cache entrées si seuls réglages) | `update_project_settings_as_admin` | `_persist_settings` |
| Réglages & Export | DIM-WRITE | — | profils / dimensions (même persistance réglages) | `st.rerun` | `update_project_settings_as_admin` | `_render_dimensions_section`, `src/presets` |
| Réglages & Export | EXP-SCOPE | `load_project_entries` → `df` (via cache) | choix périmètre UI | — (lecture seule) | *(données déjà chargées)* | `export_scope_service.summarize_export_perimeter`, `export_utils.dataframe_for_export` |
| Réglages & Export | EXP-DL | `dataframe_for_export` + sérialisation | téléchargement CSV / JSONL | — | `STATUT_VALIDE` via `export_utils` | `export_utils`, `export_quality_recap_service.build_export_quality_recap` |
| Nouvelle entrée | ENT-NEW-READ | dimensions actives, `df` | — | — | `get_project_settings`, `load_project_entries` | `src/presets.load_active_dimensions`, `nlp_engine`, `llm_generate` |
| Nouvelle entrée | ENT-NEW-WRITE | — | `update_project_entries` (ligne + cache NLP) | `invalidate_project_entries_cache`, `cached_load_project_entries`, feedback stylométrie session, `st.rerun` | `update_project_entries`, `require_role` | `nlp_engine.compute_row_cache`, `project_entries_cache` |
| Nouvelle entrée | ENT-LLM | — | appel HTTP LLM (OpenRouter / local) | mise à jour champs session | *(pas de persistance directe LLM)* | `llm_generate.generate_*`, réglages `ProjectSettings` |
| Gestion & édition | EDI-SAVE | `df` + fiche courante | `update_project_entries` | idem ENT-NEW-WRITE | `update_project_entries` | `services/edition_*`, `nlp_engine`, `services/project_dataframe_view` ; **suppression d'une fiche seule** : non exposée dans l'UI Streamlit actuelle — **N/A** côté webapp (issue-012). |
| Gestion & édition | EDI-LT | texte output | correction LT | `st.rerun` widget | — | `nlp_engine.corriger_texte_fr` (LanguageTool) |
| Tableau de bord | DASH-METRICS | agrégats sur `df` | — | — | *(lecture dataframe en mémoire)* | `services/project_dataframe_view.prepare_for_dashboard_tab`, métriques ; **API** `GET /api/projects/{id}/dashboard` (issue-014) |
| Tableau de bord | DASH-STYLO | stylométrie filtrée | — | — | colonnes `CACHE_COLUMNS` | `services/dashboard_stylometry_service` ; **API** `GET /api/projects/{id}/dashboard?dashboard_scope=…` (issue-014) |
| Mon compte | ACC-INFO | compte + compteurs | — | — | `count_owned_projects`, `count_active_memberships` | `render_tab_account` |
| Mon compte | ACC-DEL | — | `revoke_account_with_saga` | `logout`, `st.rerun` | saga + `database` (opérations compte) | `src/auth.revoke_account_with_saga` |
| Super Admin | SA-LIST | listes admin | — | — | `list_accounts_for_super_admin`, compteurs | `render_tab_super_admin` |
| Super Admin | SA-INVITE | — | `create_invitation_link` + `send_account_link_email` | rechargement annuaire (webapp) | `create_invitation_link` (`require_super_admin`) | `POST /api/super-admin/invite` + ``inviteResult`` / ``bannerTone`` ; e-mail déjà connu : message succès identique (reset) ; SMTP : ``MAIL_DELIVERY_FAILED`` (502) — **#181** |
| Super Admin | SA-DETACH | — | `detach_memberships_as_super_admin` | `st.rerun` | `detach_memberships_as_super_admin` | UI confirmation |
| Super Admin | SA-DELETE | — | `revoke_account_with_saga` | `invalidate_project_entries_cache`, flash, `st.rerun` | saga + invalidation | `auth`, `database` |
| Super Admin | SA-REPLAY | DLQ | `replay_quarantined_operation` | `st.rerun` | `replay_quarantined_operation` | panneau technique |

---

## Slice vertical (issue-007 / GitHub #129)

Parcours minimal livré côté **service `webapp`** (FastAPI, port **8080** par défaut avec `make dev` / compose) : connexion invitation-only via SuperTokens, liste des projets **propriétaire** (`list_projects_for_user`), lecture/édition/création d’entrées via `load_project_entries` / `update_project_entries`, export **CSV** et **JSONL** via `dataframe_for_export` + `convert_to_jsonl` (mêmes périmètres `validated_only` / `full_dataset` que `export_utils`, **même** `include_stylometry=True` que Streamlit, paramètre query `format` pour JSONL). Plafond optionnel d’export : variable **`WEBAPP_EXPORT_MAX_ROWS`** (réponse `413` + code `EXPORT_PAYLOAD_TOO_LARGE`). Onboarding projet vide / sans projet : textes injectés depuis `empty_project_onboarding` sur la page HTML du slice. Les erreurs JSON suivent `src/api_errors.py`. **Streamlit** reste sur **8501** en coexistence.

**Issue-021 (communication migration / GitHub #143)** : message interne type, recette et bannière optionnelle pilotée par `APP_MIGRATION_INFO_BANNER` — `docs/migration_communication_plan.md`.

| ID flux | Slice vertical (issue-007 / issue-015) |
| --- | --- |
| SB-CTX | OK — jeton vérifié (`/recipe/session/verify`) + résolution `users.su_user_id` |
| PRJ-VIEW | OK — `GET /api/projects` |
| EDI-SAVE | OK — `PATCH /api/projects/{id}/entries/{entry_id}` (+ `POST` création) ; corps `entries` après succès ; filtres édition optionnels sur `GET .../entries` (`edition_*`) |
| EXP-DL | OK — `GET .../export.csv` et `.../export.jsonl` (plafond `WEBAPP_EXPORT_MAX_ROWS`, JSONL stylométrie issue-015) |

**Issue-010 (coquille curateur / webapp / GitHub #132)** : navigation par onglets alignée sur `main_tab_labels` (`GET /api/me`), ordre workflow = `EXPECTED_WORKFLOW_TAB_ORDER` + `Mon compte` (+ `Super Admin` si super-admin), persistance du projet actif côté client (`sessionStorage` clé `webapp_active_project_id` + paramètre query `active_hint` sur `GET /api/projects`), création et suppression projet via `POST` / `DELETE /api/projects` (primitives `database.create_project`, `delete_project_as_admin` / RBAC `require_admin`). Voir `src/webapp/index_template.py`, `src/webapp/workspace_payload.py`. Tests automatisés : `tests/test_webapp_issue010_shell.py`.

Lien PR : https://github.com/Relais4x100a2/dataset_style/pull/151 (ferme #129).

---

## Réglages presets & dimensions (issue-011 / GitHub #133)

Slice **webapp** : `GET /api/projects/{id}/settings/dimensions` et `PATCH` sur la même ressource (`src/webapp/project_dimensions_settings.py`, `src/webapp/app.py`) — validation et fusion alignées sur `src/presets.py` ; persistance via `update_project_settings` (`active_preset_key`, `custom_presets_json`, `dimensions_override_json`). Coquille HTML : onglet **Réglages & Export** (`src/webapp/index_template.py`). **Écart documenté** : le formulaire Streamlit complet (`get_project_settings` / champs LLM, timeouts, etc.) n’est pas entièrement porté par cette ressource — dimensions + presets seulement ; champs LLM / LanguageTool restent éditables côté Streamlit (voir tableau § issue-011).

| ID flux | Slice (issue-011) |
| --- | --- |
| SET-READ | OK — `GET .../settings/dimensions` |
| SET-WRITE | OK — `PATCH .../settings/dimensions` (dimensions + presets ; hors champs LLM / LT coquille) |
| DIM-WRITE | OK — mêmes actions que `_render_dimensions_section` / `src/presets` |
| ENT-NEW-READ | OK — après `PATCH`, `GET .../curator/dimensions` reflète `load_active_dimensions` (`tests/test_webapp_project_dimensions_settings.py`) |

---

## Nouvelle entrée & gestion / édition (issue-012 / GitHub #134)

**API** : `GET /api/projects/{id}/entries` (filtres `edition_*`), `POST …/entries`, `PATCH …/entries/{entry_id}` ; corps `entries` après mutation ; colonnes internes préfixées `_` exclues du JSON (`tests/test_webapp_vertical_slice.py`, `src/webapp/entry_mutations.py`). **UI** : tableau et fiche d’édition dans la coquille (`src/webapp/index_template.py`) ; dimensions fermées sur `POST` / `PATCH` validées côté serveur (`_resolve_closed_dimension_value`). **Écart documenté** : **suppression de fiche** non exposée (aligné sur l’UI Streamlit actuelle, voir matrice principale ligne EDI-SAVE).

---

## Génération IA & LanguageTool (issue-013 / GitHub #135)

Slice **webapp** : `GET /api/projects/{id}/curator/dimensions`, `POST …/curator/llm-generate`, `POST …/curator/languagetool-check` (`src/webapp/curator_ai.py`, `src/webapp/app.py`) — chaîne métier alignée Streamlit (`llm_generate`, `nlp_engine.languagetool_fr_corrected_with_matches`, réglages `ProjectSettings` côté serveur). Coquille HTML : onglets **Nouvelle entrée** et **Gestion & édition** ; retours via `ds-banner-stack` + `renderApiErrorIntoStack` (issue-005). Tests : `tests/test_webapp_curator_ai.py`, `tests/test_webapp_index_template_issue013.py`.

| ID flux | Slice (issue-013) |
| --- | --- |
| ENT-LLM | OK — `POST …/curator/llm-generate` + UI coquille |
| EDI-LT | OK — `POST …/curator/languagetool-check` + UI coquille |

---

## Stylométrie, cohérence, tableau de bord (issue-014 / GitHub #136)

**API** : `GET /api/projects/{id}/dashboard` (`dashboard_scope`) — `build_curator_dashboard_envelope` ; tests `tests/test_webapp_vertical_slice.py`, `tests/test_curator_dashboard_snapshot.py`. **UI** : onglet **Tableau de bord** — chargement `fetch` vers `/dashboard` et bandeaux (script embarqué dans `src/webapp/index_template.py`). **Écart documenté (rendu analytique)** : pas de reproduction des graphiques / onglets denses du studio Streamlit ; alertes et résumés `dataset_quality` côté coquille.

---

## Export CSV/JSONL & onboarding projet vide (issue-015 / GitHub #137)

**API** : `GET …/export.csv` et `…/export.jsonl` (`scope`, `dataframe_for_export`, CSV sans colonnes `_`) — `tests/test_webapp_vertical_slice.py`, `src/webapp/app.py`. **UI** : `#exportScope` dans `index_template.py`. Onboarding projet vide : `empty_project_onboarding` (slice issue-007 / 015). **Écart documenté** : récap qualité pré-export Streamlit (`export_quality_recap_service.build_export_quality_recap`) — non porté HTTP. **Écart documenté (CSV)** : colonnes `_*` — voir § *Comparaison coexistence Streamlit vs webapp*.

---

## Mon compte curateur (issue-016 / GitHub #138)

Slice **webapp** : `GET /api/account` (JSON whiteliste : `appUserId`, `email`, `displayName`, `counts`, `uiPreferences` avec `density` / `readingComfort` par défaut `default`) ; `PATCH /api/account/ui-preferences` (fusion partielle, issue-023) ; `POST /api/auth/signout` renvoie `redirect` allow-listé (`WEBAPP_SIGNOUT_REDIRECT_ALLOWLIST`, défaut `/`) ; coquille HTML : navigation shell + onglet **Mon compte** (issue-010).

| ID flux | Slice (issue-016) |
| --- | --- |
| ACC-INFO | OK — `GET /api/account` + `PATCH /api/account/ui-preferences` + affichage shell |
| ACC-DEL | Écart documenté — suppression compte (saga) hors scope du slice ; reste Streamlit |

---

## Grille statut sprint backlog (issues issue-010 à issue-016)

Chaque cellule = **`OK`**, **`N/A`** ou **`Écart documenté`** + **référence** (section `§` ci-dessus, route HTTP, ou fichier de test abrégé).

**Abréviations tests** : `T010` → `tests/test_webapp_issue010_shell.py` ; `TDIM` → `tests/test_webapp_project_dimensions_settings.py` ; `TPRESET` → `tests/test_presets_dimensions_patch_validation.py` ; `TWV` → `tests/test_webapp_vertical_slice.py` ; `T013` → `tests/test_webapp_curator_ai.py` ; `T013T` → `tests/test_webapp_index_template_issue013.py` ; `T014` → `tests/test_curator_dashboard_snapshot.py` ; `REG` → `tests/test_curation_chain_postgres_regression.py` (opt-in CI, issue-009).

**`N/A` (colonne issue-0XX)** : le flux n’est pas un livrable de cette issue ; la décision de périmètre est celle du backlog sprint (pas une assertion de complétude globale du produit).

| ID flux | issue-010 | issue-011 | issue-012 | issue-013 | issue-014 | issue-015 | issue-016 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SB-CTX | OK — T010, `GET /api/me`, `GET /api/projects` + `active_hint` | N/A | N/A | N/A | N/A | N/A | N/A |
| PRJ-VIEW | OK — T010, TWV | N/A | N/A | N/A | N/A | N/A | N/A |
| PRJ-CREATE | OK — T010 | N/A | N/A | N/A | N/A | N/A | N/A |
| PRJ-DELETE | OK — T010 | N/A | N/A | N/A | N/A | N/A | N/A |
| SET-READ | N/A | OK — TDIM, `GET …/settings/dimensions`, §11 | N/A | N/A | N/A | N/A | N/A |
| SET-WRITE | N/A | OK — TDIM, TPRESET, `PATCH …/settings/dimensions`, §11 | N/A | N/A | N/A | N/A | N/A |
| DIM-WRITE | N/A | OK — §11, TDIM, TPRESET | N/A | N/A | N/A | N/A | N/A |
| EXP-SCOPE | N/A | N/A | N/A | N/A | N/A | OK — TWV, `#exportScope`, §15 | N/A |
| EXP-DL | N/A | N/A | N/A | N/A | N/A | **Écart documenté** — génération fichiers OK (`TWV`, `REG`, `dataframe_for_export`) ; **non** récap qualité pré-export Streamlit ; **non** parité CSV colonnes `_*` vs Streamlit — §15 + § coexistence | N/A |
| ENT-NEW-READ | N/A | OK — §11 (tableau slice) | OK — TWV, gabarit `index_template`, §12 | OK — T013, `GET …/curator/dimensions`, §13 | N/A | N/A | N/A |
| ENT-NEW-WRITE | N/A | N/A | OK — TWV, REG, §12 | N/A | N/A | N/A | N/A |
| ENT-LLM | N/A | N/A | N/A | OK — T013, T013T, §13 | N/A | N/A | N/A |
| EDI-SAVE | N/A | N/A | OK — TWV, REG, §12 | N/A | N/A | N/A | N/A |
| EDI-LT | N/A | N/A | N/A | OK — T013, T013T, §13 | N/A | N/A | N/A |
| DASH-METRICS | N/A | N/A | N/A | N/A | OK — TWV, T014, `GET …/dashboard`, §14 | N/A | N/A |
| DASH-STYLO | N/A | N/A | N/A | N/A | OK — idem DASH-METRICS (`dashboard_scope`), §14 | N/A | N/A |
| ACC-INFO | N/A | N/A | N/A | N/A | N/A | N/A | OK — TWV, `GET /api/account`, §16 |
| ACC-DEL | N/A | N/A | N/A | N/A | N/A | N/A | Écart documenté — §16 (saga hors webapp) |
| SA-LIST | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| SA-INVITE | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| SA-DETACH | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| SA-DELETE | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| SA-REPLAY | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

**Notes lecture produit (colonnes 010–016)** : les lignes **SA-*** sont hors périmètre **parité curateur** (backlog 017–019) ; la cellule **`N/A`** signifie que l’issue colonne ne couvre pas le flux super-admin. **`Écart documenté`** sur **EXP-DL × issue-015** regroupe (1) le récap qualité pré-export Streamlit non porté en HTTP et (2) l’écart CSV colonnes `_*` documenté en § *Comparaison coexistence* — ces points ne doivent pas être relus comme **`OK`** plein studio.

---

## Comparaison coexistence Streamlit vs webapp

Pendant la double exposition (**Streamlit** port **8501**, **webapp** slice vertical port **8080** par défaut), rejouer la même séquence sur les deux interfaces avec le même compte et le même projet : **PRJ-CREATE** (ou sélection), **ENT-NEW-WRITE** / **EDI-SAVE**, puis **EXP-SCOPE** + **EXP-DL** (CSV et JSONL). Les téléchargements côté webapp appliquent **`export_utils`** (périmètres `validated_only` / `full_dataset`) et la règle **CSV sans colonnes préfixées `_`** via `public_export_column_names` / `csv_text_from_export_dataframe` (alignée sur la sérialisation API des entrées). **Écart documenté** : le CSV Streamlit appelle `to_csv` sur le slice complet et inclut donc encore les colonnes cache `_*`. Toute divergence volontaire ou constatée avec Streamlit doit être notée en **Écart documenté** dans la matrice ou la recette associée.

---

## Baseline UX (issue-020)

Protocole versionné pour **temps de parcours**, **erreurs** et **questionnaire interne** :
`docs/ux_baseline_issue_020.md` (questionnaire : `docs/ux_baseline_questionnaire.md`).
Jalons stables alignés sur les IDs flux de cette matrice : `SB-CTX`, `ENT-NEW-WRITE`,
`EDI-SAVE`, `EXP-SCOPE`, `EXP-DL`. Collecte optionnelle côté serveur Streamlit via
`DATASET_STYLE_UX_TELEMETRY_DIR` (fichiers JSONL append-only, hors schéma tenant).
**Webapp** : mêmes jalons **sauf** `EXP-SCOPE` (non émis — écart récap pré-export issue-015) ; en-têtes
`X-Dataset-Style-Ux-Run-Id` / `X-Dataset-Style-Ux-Scenario-Id` et réponses `X-Dataset-Style-Ux-Telemetry` :
voir `docs/ux_baseline_issue_020.md` § Webapp.
**Webapp** : mêmes codes sur les routes FastAPI lorsque le client envoie
`X-Dataset-Style-Ux-Run-Id` (et `X-Dataset-Style-Ux-Shell-Init: 1` une fois pour `SB-CTX`) ;
voir `docs/ux_baseline_issue_020.md` et `scripts/aggregate_ux_baseline_jsonl.py`.

---

## Checklist recette minimale (manuelle)

Chaîne **projet → entrée → export** à rejouer après chaque bascule majeure UI / API :

1. **Projet** : créer un projet (ou sélectionner un projet test), vérifier sidebar + onglet Projets.
2. **Réglages** : ouvrir **Réglages & Export**, vérifier chargement des réglages (Streamlit : `get_project_settings` ; slice webapp : `GET /api/projects/{id}/settings/dimensions`).
3. **Entrée** : sous **Nouvelle entrée**, saisir input/output requis, enregistrer ; vérifier message de succès et qu’une relance affiche la ligne (cache invalidé — pas de « fantôme » d’ancien dataframe).
4. **Export** : même onglet, basculer périmètre « Validées seulement » / « Tout le dataset », télécharger **CSV** et **JSONL** ; ouvrir les fichiers et contrôler cohérence des filtres (notamment statuts et colonnes stylométriques exportées).
5. **Non-régression filtres** : avec des fiches non validées, confirmer que « Validées seulement » exclut bien les lignes hors `STATUT_VALIDE` (`export_utils.dataframe_for_export`).
6. **Aides IA & LanguageTool** (slice webapp issue-013) : sous **Nouvelle entrée** ou **Gestion & édition**, vérifier le chargement des listes (`GET …/curator/dimensions`), une génération brouillon↔output (`POST …/curator/llm-generate`) et une analyse LT sur un texte accentué FR (`POST …/curator/languagetool-check`) ; en erreur réseau, vérifier un bandeau actionnable (`suggested_action`).

---

## Jeux de non-régression automatisés

Exemples de commandes ciblées (à intégrer dans CI ou lancer localement) :

```bash
python3 -m pytest tests/test_curator_dashboard_snapshot.py tests/test_webapp_vertical_slice.py -q
python3 -m pytest tests/test_webapp_issue010_shell.py -q
python3 -m pytest tests/test_webapp_project_dimensions_settings.py tests/test_presets_dimensions_patch_validation.py -q
python3 -m pytest tests/test_webapp_curator_ai.py tests/test_webapp_index_template_issue013.py -q
python3 -m pytest tests/test_migration_parity_matrix_doc.py -q
# issue-009 / GitHub #131 : persistance Postgres + export (PRJ-CREATE, ENT-NEW-WRITE, EDI-SAVE, EXP-SCOPE, EXP-DL)
# Exporter DATASET_STYLE_REGRESSION_DATABASE_URL (postgresql://…) — en CI la variable est définie par .github/workflows/ci.yml
python3 -m pytest tests/test_curation_chain_postgres_regression.py -q
python3 -m pytest tests/test_tab_layout.py tests/test_export_utils.py tests/test_project_entries_cache.py -q
python3 -m pytest tests/test_services.py tests/test_export_quality_recap_service.py -q
python3 -m pytest tests/test_nlp_engine.py tests/test_ui_components_new_entry.py tests/test_ui_components_edition.py -q
python3 -m pytest tests/test_dashboard_metrics.py tests/test_corpus_stylometry_alerts_fr.py -q
python3 -m pytest tests/test_auth.py tests/test_super_admin_ui_texts.py -q
```

**Risque explicité (brief PM)** : oublier les filtres d’export ou les seuils stylométrie conduit à des régressions silencieuses — les tests `test_export_utils`, `test_export_quality_recap_service` et `test_corpus_stylometry_alerts_fr` sont prioritaires lors des changements touchant `export_utils` / `nlp_engine` / `CACHE_COLUMNS`.

---

## invalidate_project_entries_cache (rappel contrat)

Après **création projet**, **suppression projet**, **sauvegarde nouvelle entrée**, **sauvegarde édition**, et **suppression de compte côté Super Admin** (flux SA-DELETE), le code applicatif invalide le cache Streamlit des entrées avant `st.rerun()`. Toute couche API devra documenter l’équivalent (version de ressource, `Cache-Control`, ou réponse obligeant le client à refetch les `entries`). **Webapp** : voir corps `entries` sur `POST`/`PATCH` entrées (`src/webapp/app.py`) + `GET .../entries` avec paramètres `edition_*` pour les filtres liste (équivalent onglet Gestion & édition). Les réponses JSON d’entrées **n’incluent pas** les colonnes internes dont le nom commence par `_` (cache NLP en base) — aligné export CSV slice (`export.csv`).
