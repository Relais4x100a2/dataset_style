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

Légende **Post-mutation (équivalent Streamlit)** : aujourd’hui, après chaque écriture qui change les lignes `entries` ou le contexte projet, l’UI appelle `invalidate_project_entries_cache()` (`src/project_entries_cache.py`) puis `st.rerun()` pour que `cached_load_project_entries` relise via `load_project_entries` (`database.py`). Toute API doit exposer un comportement cohérent (invalidation côté client, ETag, ou relecture explicite). **Webapp (issue-012 / issue-179)** : les réponses `POST` / `PATCH` sur les entrées incluent un tableau `entries` issu d’un `load_project_entries` immédiat après succès ; si la requête porte les mêmes paramètres query `edition_*` que la liste, ce tableau est **filtré comme** `GET …/entries` (pas de double logique côté client). Le client peut rafraîchir sans état fantôme sur les champs persistés.

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
| Tableau de bord | DASH-METRICS | agrégats sur `df` | — | — | *(lecture dataframe en mémoire)* | `services/project_dataframe_view.prepare_for_dashboard_tab`, métriques ; **API** `GET /api/projects/{id}/dashboard` (issue-014) |
| Tableau de bord | DASH-STYLO | stylométrie filtrée | — | — | colonnes `CACHE_COLUMNS` | `services/dashboard_stylometry_service` ; **API** `GET /api/projects/{id}/dashboard?dashboard_scope=…` (issue-014) |
| Mon compte | ACC-INFO | compte + compteurs | — | — | `count_owned_projects`, `count_active_memberships` | `render_tab_account` |
| Mon compte | ACC-DEL | — | `revoke_account_with_saga` | `logout`, `st.rerun` | saga + `database` (opérations compte) | `src/auth.revoke_account_with_saga` |
| Super Admin | SA-LIST | listes admin | — | — | `list_accounts_for_super_admin`, compteurs | `render_tab_super_admin` |
| Super Admin | SA-DETACH | — | `detach_memberships_as_super_admin` | `st.rerun` | `detach_memberships_as_super_admin` | UI confirmation |
| Super Admin | SA-DELETE | — | `revoke_account_with_saga` | `invalidate_project_entries_cache`, flash, `st.rerun` | saga + invalidation | `auth`, `database` |
| Super Admin | SA-REPLAY | DLQ | `replay_quarantined_operation` | `st.rerun` | `replay_quarantined_operation` | panneau technique |

---

## Slice vertical (issue-007 / GitHub #129)

Parcours minimal livré côté **service `webapp`** (FastAPI, port **8080** par défaut avec `make dev` / compose) : connexion invitation-only via SuperTokens, liste des projets **propriétaire** (`list_projects_for_user`), lecture/édition/création d’entrées via `load_project_entries` / `update_project_entries`, export **CSV** et **JSONL** via `dataframe_for_export` + `convert_to_jsonl` (mêmes périmètres `validated_only` / `full_dataset` que `export_utils`, **même** `include_stylometry=True` que Streamlit, paramètre query `format` pour JSONL). Plafond optionnel d’export : variable **`WEBAPP_EXPORT_MAX_ROWS`** (réponse `413` + code `EXPORT_PAYLOAD_TOO_LARGE`). Onboarding projet vide / sans projet : textes injectés depuis `empty_project_onboarding` sur la page HTML du slice. Les erreurs JSON suivent `src/api_errors.py`. **Streamlit** reste sur **8501** en coexistence.

| ID flux | Slice vertical (issue-007 / issue-015) |
| --- | --- |
| SB-CTX | OK — jeton vérifié (`/recipe/session/verify`) + résolution `users.su_user_id` |
| PRJ-VIEW | OK — `GET /api/projects` |
| EDI-SAVE | OK — `PATCH /api/projects/{id}/entries/{entry_id}` (+ `POST` création) ; corps `entries` après succès (filtré comme `GET` si mêmes query `edition_*`) ; filtres édition optionnels sur `GET .../entries` (`edition_*`) |
| EXP-DL | OK — `GET .../export.csv` et `.../export.jsonl` (plafond `WEBAPP_EXPORT_MAX_ROWS`, JSONL stylométrie issue-015) |

**Issue-010 (coquille curateur / webapp)** : navigation par onglets alignée sur `main_tab_labels` (`GET /api/me`), persistance du projet actif côté client (`sessionStorage` + `active_hint` sur `GET /api/projects`), création et suppression projet via `POST` / `DELETE /api/projects` (primitives `database.create_project`, `delete_project_as_admin`). Voir `src/webapp/index_template.py`, `src/webapp/workspace_payload.py`.

Lien PR : https://github.com/Relais4x100a2/dataset_style/pull/151 (ferme #129).

---

## Assistance curateur IA & LanguageTool (issue-006 / GitHub #180)

Slice **webapp** : routes stateless `POST /api/projects/{id}/curator/llm-generate`, `POST /api/projects/{id}/curator/languagetool-check`, `GET …/curator/dimensions` (issue-013). Aucune persistance entrée depuis ces routes ; la coquille HTML (`src/webapp/index_template.py`) expose un panneau **Assistance IA & LanguageTool** (chargement type `hx-indicator`, anti double-clic, insertion / remplacement **explicites** pour LT et IA).

| Décision | Détail |
| --- | --- |
| Échecs LLM « métier » | **HTTP 200** + `status: "failed"` + `message` en français (aligné matrice issue-013 / `tests/test_webapp_curator_ai.py`) — pas d’erreur HTTP générique pour un refus ou une réponse vide du fournisseur. |
| Indisponibilité LanguageTool | **HTTP 503** + enveloppe `CURATOR_LANGUAGETOOL_UNAVAILABLE` (`src/api_errors.py`). |
| Succès LanguageTool | **HTTP 200** + `status: "ok"` + `corrected` + `matches` ; texte vide → `status: "validation_error"` sans appel réseau. |

---

## Mon compte curateur (issue-016 / GitHub #138)

Slice **webapp** : `GET /api/account` (JSON whiteliste : `appUserId`, `email`, `displayName`, `counts.ownedProjects`, `counts.activeMemberships`) ; `POST /api/auth/signout` renvoie `redirect` allow-listé (`WEBAPP_SIGNOUT_REDIRECT_ALLOWLIST`, défaut `/`) ; coquille HTML : navigation shell + onglet **Mon compte** (issue-010).

| ID flux | Slice (issue-016) |
| --- | --- |
| ACC-INFO | OK — `GET /api/account` + affichage shell |
| ACC-DEL | Écart documenté — suppression compte (saga) hors scope du slice ; reste Streamlit |

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
| EXP-SCOPE | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | OK | ⏳ |
| EXP-DL | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | OK | ⏳ |
| ENT-NEW-READ | ⏳ | ⏳ | OK | ⏳ | ⏳ | ⏳ | ⏳ |
| ENT-NEW-WRITE | ⏳ | ⏳ | OK | ⏳ | ⏳ | ⏳ | ⏳ |
| ENT-LLM | ⏳ | ⏳ | ⏳ | OK | ⏳ | ⏳ | ⏳ |
| EDI-SAVE | ⏳ | ⏳ | OK | ⏳ | ⏳ | ⏳ | ⏳ |
| EDI-LT | ⏳ | ⏳ | ⏳ | OK | ⏳ | ⏳ | ⏳ |
| DASH-METRICS | ⏳ | ⏳ | ⏳ | ⏳ | OK | ⏳ | ⏳ |
| DASH-STYLO | ⏳ | ⏳ | ⏳ | ⏳ | OK | ⏳ | ⏳ |
| ACC-INFO | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | OK |
| ACC-DEL | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Écart documenté |
| SA-LIST | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| SA-DETACH | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| SA-DELETE | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| SA-REPLAY | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

**issue-015 (export slice + onboarding)** : le récap qualité pré-export Streamlit (`export_quality_recap_service.build_export_quality_recap`) n’est pas porté par le slice HTTP — **écart documenté** ; la parité « données exportées » repose sur `dataframe_for_export` + `convert_to_jsonl` uniquement.

---

## Protocole recette : une surface à la fois (coexistence dev/staging)

**Référence processus / PR** : checklist pré-merge **`docs/merge_ready_checklist.md`** §6 ; clôture suivi GitHub **#178**.

Pendant la double exposition (**Streamlit** port **8501**, **webapp** port **8080** par défaut), la règle de recette est : **un scénario ne change pas d’origine UI** (Streamlit vs webapp) avant la **fin de scénario** — **export réussi** (`EXP-DL`) ou **terminaison explicite** (abandon noté, fin de session de test). Les **IDs de flux** de cette matrice (`SB-CTX`, `PRJ-VIEW`, `ENT-NEW-WRITE`, `EDI-SAVE`, `EXP-SCOPE`, `EXP-DL`, etc.) servent de fil conducteur : une fois le scénario démarré sur une surface, **liste → fiche → export** s’exécute **entièrement** sur cette surface.

**Exemple attendu** : même compte, même projet — parcourir la chaîne curation **sans** alternance mid-scénario ; pour comparer les deux UIs, enchaîner **deux runs distincts** (scénario complet Streamlit, puis scénario complet webapp, ou l’inverse), pas des allers-retours entre onglets / ports entre deux jalons du même parcours.

**Base de données partagée** : PostgreSQL est **unique** entre les deux interfaces. La **non-régression « données »** (contrats API, tests, état relu en base) et la **non-régression « perception UI »** (cache navigateur, reruns Streamlit, fragments HTMX, cookies SuperTokens) ne sont **pas** équivalentes. Ce protocole réduit surtout les **fausses régressions** d’**attribution causale** côté perception. Les tests de **divergence volontaire** entre surfaces restent **hors chemin critique** de recette standard, sauf **story** ou ticket dédié qui en fixe le périmètre.

### Alternances volontaires (documentées avant exécution)

Toute exception au mono-surface **mid-scénario** doit être **nommée**, **justifiée** (objectif) et accompagnée de **données figées** (compte, projet, révision des entrées ou jeu de test) dans la recette, le ticket ou la PR — afin d’éviter les conclusions hâtives sur le backend.

| Cas | Objectif | Données figées (minimum) |
| --- | --- | --- |
| **Test de divergence volontaire** | Mesurer ou prouver un écart d’affichage / de parcours entre Streamlit et webapp sur un **même** état de données | Compte, `project_id` (ou identifiant de jeu), état des entrées figé (pas d’écritures concurrentes pendant la comparaison) |
| **Reprise après incident** | Reprendre un scénario interrompu (crash, session expirée) | Nouveau scénario explicite : noter la surface de reprise et la raison ; ne pas mélanger les jalons d’un `run_id` baseline entre deux origines |

---

## Comparaison coexistence Streamlit vs webapp

Pour valider la **parité** fonctionnelle sans biais de recette : exécuter le chemin critique **en entier** sur **une** interface (**PRJ-CREATE** ou sélection `SB-CTX` / `PRJ-VIEW`, **ENT-NEW-WRITE** / **EDI-SAVE**, **EXP-SCOPE** + **EXP-DL** CSV et JSONL), puis **rejouer la séquence complète** sur l’autre (deux parcours distincts), plutôt que d’alterner les ports au milieu d’un même scénario. Les téléchargements côté webapp appliquent **`export_utils`** (périmètres `validated_only` / `full_dataset`) et la règle **CSV sans colonnes préfixées `_`** (alignée sur la sérialisation API des entrées). Toute divergence volontaire ou constatée avec Streamlit doit être notée en **Écart documenté** dans la matrice ou la recette associée.

---

## Baseline UX (issue-020)

Protocole versionné pour **temps de parcours**, **erreurs** et **questionnaire interne** :
`docs/ux_baseline_issue_020.md` (questionnaire : `docs/ux_baseline_questionnaire.md`).
Jalons stables alignés sur les IDs flux de cette matrice : `SB-CTX`, `ENT-NEW-WRITE`,
`EDI-SAVE`, `EXP-SCOPE`, `EXP-DL`. Collecte optionnelle côté serveur Streamlit via
`DATASET_STYLE_UX_TELEMETRY_DIR` (fichiers JSONL append-only, hors schéma tenant).

---

## Checklist recette minimale (manuelle)

Chaîne **projet → entrée → export** à rejouer après chaque bascule majeure UI / API. **Coexistence Streamlit / webapp** : respecter le protocole **une surface à la fois** (section ci-dessus) — ne pas alterner les ports au milieu du scénario.

1. **Projet** : créer un projet (ou sélectionner un projet test), vérifier sidebar + onglet Projets.
2. **Réglages** : ouvrir **Réglages & Export**, vérifier chargement des réglages (`get_project_settings`).
3. **Entrée** : sous **Nouvelle entrée**, saisir input/output requis, enregistrer ; vérifier message de succès et qu’une relance affiche la ligne (cache invalidé — pas de « fantôme » d’ancien dataframe).
4. **Export** : même onglet, basculer périmètre « Validées seulement » / « Tout le dataset », télécharger **CSV** et **JSONL** ; ouvrir les fichiers et contrôler cohérence des filtres (notamment statuts et colonnes stylométriques exportées).
5. **Non-régression filtres** : avec des fiches non validées, confirmer que « Validées seulement » exclut bien les lignes hors `STATUT_VALIDE` (`export_utils.dataframe_for_export`).

---

## Jeux de non-régression automatisés

Exemples de commandes ciblées (à intégrer dans CI ou lancer localement) :

```bash
python3 -m pytest tests/test_curator_dashboard_snapshot.py tests/test_webapp_vertical_slice.py -q
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
