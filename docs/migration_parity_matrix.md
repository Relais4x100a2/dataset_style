# Matrice de parité Streamlit → API (migration)

**Backlog :** issue-004 — publier la matrice versionnée et les jeux de non-régression.  
**Dépendance notée :** issue-003 (règles d’accès projet).  
**Source de vérité de l’ordre des onglets produit :** `EXPECTED_WORKFLOW_TAB_ORDER` et `main_tab_labels()` dans `src/tab_layout.py`, montés dans `main.py` (corps des onglets alignés sur le même ordre).

Cette matrice sert de carte pour exposer une API HTTP (ou équivalent) sans perdre les flux critiques aujourd’hui portés par Streamlit. Les colonnes **issue-010** à **issue-016** matérialisent le **statut de parité** par sprint backlog : elles **doivent être mises à jour** à la clôture de chaque issue correspondante (remplacer `⏳` par `OK`, `N/A` ou `Écart documenté`).

**Primitive persistance partagée :** `src/database.py` (SQLAlchemy / PostgreSQL, schéma multi-tenant). Aucune dépendance Streamlit dans ce module.

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

Légende **Post-mutation (équivalent Streamlit)** : aujourd’hui, après chaque écriture qui change les lignes `entries` ou le contexte projet, l’UI appelle `invalidate_project_entries_cache()` (`src/project_entries_cache.py`) puis `st.rerun()` pour que `cached_load_project_entries` relise via `load_project_entries` (`database.py`). Toute API doit exposer un comportement cohérent (invalidation côté client, ETag, ou relecture explicite).

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
| Gestion & édition | EDI-SAVE | `df` + fiche courante | `update_project_entries` | idem ENT-NEW-WRITE | `update_project_entries` | `services/edition_*`, `nlp_engine`, `services/project_dataframe_view` |
| Gestion & édition | EDI-LT | texte output | correction LT | `st.rerun` widget | — | `nlp_engine.corriger_texte_fr` (LanguageTool) |
| Tableau de bord | DASH-METRICS | agrégats sur `df` | — | — | *(lecture dataframe en mémoire)* | `services/project_dataframe_view.prepare_for_dashboard_tab`, métriques |
| Tableau de bord | DASH-STYLO | stylométrie filtrée | — | — | colonnes `CACHE_COLUMNS` | `services/dashboard_stylometry_service` |
| Mon compte | ACC-INFO | compte + compteurs | — | — | `count_owned_projects`, `count_active_memberships` | `render_tab_account` |
| Mon compte | ACC-DEL | — | `revoke_account_with_saga` | `logout`, `st.rerun` | saga + `database` (opérations compte) | `src/auth.revoke_account_with_saga` |
| Super Admin | SA-LIST | listes admin | — | — | `list_accounts_for_super_admin`, compteurs | `render_tab_super_admin` |
| Super Admin | SA-DETACH | — | `detach_memberships_as_super_admin` | `st.rerun` | `detach_memberships_as_super_admin` | UI confirmation |
| Super Admin | SA-DELETE | — | `revoke_account_with_saga` | `invalidate_project_entries_cache`, flash, `st.rerun` | saga + invalidation | `auth`, `database` |
| Super Admin | SA-REPLAY | DLQ | `replay_quarantined_operation` | `st.rerun` | `replay_quarantined_operation` | panneau technique |

---

## Slice vertical (issue-007 / GitHub #129)

Parcours minimal livré côté **service `webapp`** (FastAPI, port **8080** par défaut avec `make dev` / compose) : connexion invitation-only via SuperTokens, liste des projets **propriétaire** (`list_projects_for_user`), lecture/édition/création d’entrées via `load_project_entries` / `update_project_entries`, export **CSV** et **JSONL** via `dataframe_for_export` + `convert_to_jsonl` (mêmes périmètres `validated_only` / `full_dataset` que `export_utils`). Les erreurs JSON suivent `src/api_errors.py`. **Streamlit** reste sur **8501** en coexistence.

| ID flux | Slice vertical (issue-007) |
| --- | --- |
| SB-CTX | OK — jeton vérifié (`/recipe/session/verify`) + résolution `users.su_user_id` |
| PRJ-VIEW | OK — `GET /api/projects` |
| EDI-SAVE | OK — `PATCH /api/projects/{id}/entries/{entry_id}` (+ `POST` création minimale) |
| EXP-DL | OK — `GET .../export.csv` et `.../export.jsonl` |

**Issue-010 (coquille curateur / webapp)** : navigation par onglets alignée sur `main_tab_labels` (`GET /api/me`), persistance du projet actif côté client (`sessionStorage` + `active_hint` sur `GET /api/projects`), création et suppression projet via `POST` / `DELETE /api/projects` (primitives `database.create_project`, `delete_project_as_admin`). Voir `src/webapp/index_template.py`, `src/webapp/workspace_payload.py`.

Lien PR : https://github.com/Relais4x100a2/dataset_style/pull/151 (ferme #129).

---

## Grille statut sprint backlog (issues issue-010 à issue-016)

Remplacez chaque `⏳` lors de la clôture de l’issue backlog correspondante (pas le numéro GitHub — voir règle de synchronisation backlog / GitHub dans la doc projet).

| ID flux | issue-010 | issue-011 | issue-012 | issue-013 | issue-014 | issue-015 | issue-016 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SB-CTX | OK | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| PRJ-VIEW | OK | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| PRJ-CREATE | OK | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| PRJ-DELETE | OK | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| SET-READ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| SET-WRITE | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| DIM-WRITE | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| EXP-SCOPE | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| EXP-DL | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ENT-NEW-READ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ENT-NEW-WRITE | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ENT-LLM | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| EDI-SAVE | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| EDI-LT | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| DASH-METRICS | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| DASH-STYLO | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ACC-INFO | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ACC-DEL | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| SA-LIST | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| SA-DETACH | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| SA-DELETE | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| SA-REPLAY | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

---

## Comparaison coexistence Streamlit vs webapp

Pendant la double exposition (**Streamlit** port **8501**, **webapp** slice vertical port **8080** par défaut), rejouer la même séquence sur les deux interfaces avec le même compte et le même projet : **PRJ-CREATE** (ou sélection), **ENT-NEW-WRITE** / **EDI-SAVE**, puis **EXP-SCOPE** + **EXP-DL** (CSV et JSONL). Les téléchargements côté webapp appliquent **`export_utils`** (périmètres `validated_only` / `full_dataset`) et la règle **CSV sans colonnes préfixées `_`** (alignée sur la sérialisation API des entrées). Toute divergence volontaire ou constatée avec Streamlit doit être notée en **Écart documenté** dans la matrice ou la recette associée.

---

## Checklist recette minimale (manuelle)

Chaîne **projet → entrée → export** à rejouer après chaque bascule majeure UI / API :

1. **Projet** : créer un projet (ou sélectionner un projet test), vérifier sidebar + onglet Projets.
2. **Réglages** : ouvrir **Réglages & Export**, vérifier chargement des réglages (`get_project_settings`).
3. **Entrée** : sous **Nouvelle entrée**, saisir input/output requis, enregistrer ; vérifier message de succès et qu’une relance affiche la ligne (cache invalidé — pas de « fantôme » d’ancien dataframe).
4. **Export** : même onglet, basculer périmètre « Validées seulement » / « Tout le dataset », télécharger **CSV** et **JSONL** ; ouvrir les fichiers et contrôler cohérence des filtres (notamment statuts et colonnes stylométriques exportées).
5. **Non-régression filtres** : avec des fiches non validées, confirmer que « Validées seulement » exclut bien les lignes hors `STATUT_VALIDE` (`export_utils.dataframe_for_export`).

---

## Jeux de non-régression automatisés

Exemples de commandes ciblées (à intégrer dans CI ou lancer localement) :

```bash
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

Après **création projet**, **suppression projet**, **sauvegarde nouvelle entrée**, **sauvegarde édition**, et **suppression de compte côté Super Admin** (flux SA-DELETE), le code applicatif invalide le cache Streamlit des entrées avant `st.rerun()`. Toute couche API devra documenter l’équivalent (version de ressource, `Cache-Control`, ou réponse obligeant le client à refetch les `entries`).
