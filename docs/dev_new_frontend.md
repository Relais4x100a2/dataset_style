# Développement — stack nouveau frontal (FastAPI)

Ce guide décrit comment lancer **uniquement** le service `webapp` (FastAPI) et ses dépendances, sans démarrer Streamlit (`app`). Le périmètre correspond au squelette livré pour l’intégration incrémentale (issue-008, slice vertical `src/webapp/`).

## Prérequis

- Docker et Docker Compose v2
- Fichier `.env` à la racine (copier depuis `.env.example` puis adapter)

## Commande rapide

```bash
cp .env.example .env   # une fois
make dev-web
```

Équivalent explicite :

```bash
docker compose --env-file .env up postgres supertokens webapp
```

## Ports et URL

| Service      | Port hôte (défaut) | Rôle                          |
|-------------|--------------------|-------------------------------|
| `webapp`    | **8080** (`WEBAPP_PORT`) | BFF FastAPI + coquille HTML |
| `postgres`  | non exposé par défaut | Base de données               |
| `supertokens` | non exposé par défaut | Core auth                     |

Interface locale : `http://localhost:8080` — healthcheck ops : `http://localhost:8080/health`.

Les **assets statiques** du slice (tokens CSS, etc.) sont servis sous `http://localhost:8080/static/…` (ex. `design_tokens.css`). Voir `docs/design_tokens_webapp.md`.

## Cohérence auth (ADR 0006)

- Aligner **`APP_PUBLIC_BASE_URL`** sur l’URL réellement servie aux testeurs (ex. `http://localhost:8080` pour un test local du slice web seul), afin que les liens invitation / reset et les cookies SuperTokens restent cohérents.
- **`WEBAPP_CORS_ORIGINS`** : liste fermée d’origines autorisées (séparées par des virgules) si le navigateur appelle le BFF depuis une origine distincte ; en monorigine (même schéma, hôte et port), la valeur par défaut `http://localhost:8080` suffit.

Référence : `docs/adr/0006-front-stack-bff-spa-vs-htmx.md` et `docs/streamlit_to_new_frontend_cutover.md` pour le mode production (cutover unique).

## Bannière d’information (optionnel)

Variable **`APP_MIGRATION_INFO_BANNER`** : texte brut affiché en haut de la page `GET /` du service `webapp` (ainsi que dans Streamlit lorsque les deux coexistent en recette). Voir `docs/migration_communication_plan.md`. Laisser vide pour masquer.

## Tests et CI

Les routes FastAPI sont couvertes par `pytest` (`tests/test_webapp_vertical_slice.py`, `tests/test_webapp_health_issue008.py`, `tests/test_webapp_project_dimensions_settings.py`, `tests/test_webapp_curator_ai.py`, spike ADR `tests/test_bff_spike_issue006.py`). Le workflow `.github/workflows/ci.yml` inclut une étape de smoke dédiée au healthcheck avant la suite complète.

## Shell curateur — projets et navigation (issue-010 / GitHub #132)

- **`GET /api/me`** : `user` (identité + `isSuperAdmin`) et `mainTabLabels` — même ordre que Streamlit (`src/tab_layout.main_tab_labels`).
- **`GET /api/projects`** : liste des projets du tenant + `activeProjectId` résolu (paramètre optionnel `active_hint` = dernier projet choisi côté navigateur).
- **`POST /api/projects`** / **`DELETE /api/projects/{id}`** : création et suppression (admin propriétaire uniquement ; refus = enveloppe opaque `404` / `NOT_FOUND_GENERIC`, comme les autres routes protégées).
- **Persistance client** : la coquille HTML (`GET /`) mémorise l’identifiant projet actif dans `sessionStorage` sous la clé `webapp_active_project_id` (équivalent fonctionnel de `st.session_state["project_id"]` côté Streamlit). Voir aussi `docs/session_state_keys.md` (section webapp).

Tests : `tests/test_webapp_issue010_shell.py`.

## API — presets et dimensions projet (issue-011)

- `GET /api/projects/{project_id}/settings/dimensions` — lecture après contrôle d’accès (`load_project_entries`) : `activePresetKey`, `dimensions` (effectives), liste `presets` (`key` + `label`), `projectRole`, `canEditDimensions` (admin projet uniquement).
- `PATCH /api/projects/{project_id}/settings/dimensions` — mutations réservées à l’admin (`require_admin` + `update_project_settings`), corps JSON `action` : `load_preset` (`preset_key`), `replace_dimensions` (`dimensions` objet), `save_custom_preset` (`custom_preset_name`, `custom_preset_label`, `dimensions`). Validation et persistance alignées sur `src/presets.py` (mêmes champs `project_settings` que Streamlit).

## Génération IA & LanguageTool (issue-013 / GitHub #135)

- `GET /api/projects/{project_id}/curator/dimensions` — dimensions actives pour les sélecteurs d'aide (profil `load_active_dimensions`).
- `POST /api/projects/{project_id}/curator/llm-generate` — génération brouillon↔output (`src/llm_generate`, timeouts et URL LLM issus des réglages projet ; pas de clé exposée au navigateur).
- `POST /api/projects/{project_id}/curator/languagetool-check` — texte corrigé + suggestions (`src/nlp_engine` ; URL LT projet ou défaut).

La coquille `GET /` appelle ces routes depuis les onglets **Nouvelle entrée** et **Gestion & édition** (bandeaux `ds-banner-stack`). Tests : `tests/test_webapp_curator_ai.py`, `tests/test_webapp_index_template_issue013.py`.

## Préférences d'affichage (issue-023)

- Lecture : champ `uiPreferences` sur `GET /api/account` (`density`, `readingComfort`, valeurs par défaut `default`).
- Mise à jour : `PATCH /api/account/ui-preferences` avec corps JSON partiel (`density` et/ou `readingComfort`). Réponse : `{ "uiPreferences": { ... } }`.
- Persistance : colonne `users.ui_preferences_json` (créée par `ensure_schema()` au boot Streamlit / webapp). Détails UX et mapping CSS : `docs/ui_display_preferences.md`.
