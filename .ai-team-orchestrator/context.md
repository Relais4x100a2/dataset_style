# Contexte orchestrateur — Dataset Style

## Branches Git

| Concept | Branche |
|---------|---------|
| **Branche pipeline (work_branch)** | `ai-team/pipeline` — branche fixe réutilisée par tous les runs `pipeline next` |
| **Cible de merge / prod CapRover Relais4** | `deploy-caprover-relais4` — **pas `main`** |
| Branche d'intégration migration (préprod) | `deploy-newfrontend` *(à créer ; distincte de la prod)* |
| Secondaire / historique | `main` — pas la rampe de déploiement actuelle |

Règle : les PR issues du pipeline mergent `ai-team/pipeline` → `deploy-caprover-relais4`.
Ne pas confondre `deploy-caprover-relais4` (tiret) avec l'ancien nom `deploy/caprover-relais4` (slash).

Référence canonique : `docs/release_train_caprover.md`.

## Stack et architecture

- **Python 3.12** — `uv` / `pyproject.toml`
- **Streamlit** `main.py` (port 8501) — interface actuelle en production
- **FastAPI BFF** `src/webapp/app.py` (port 8080) — nouveau frontal en cours de migration
- **PostgreSQL 16** — base unique partagée entre Streamlit et FastAPI
- **SuperTokens** — auth (invitation-only, cookies httpOnly, EmailPassword)
- Dépendances runtime principales : `fastapi`, `uvicorn`, `streamlit`, `sqlalchemy`, `psycopg[binary]`, `pandas`, `httpx`

Même `Dockerfile` pour `app` (Streamlit) et `webapp` (FastAPI) ; commande surchar­gée dans `compose.yaml`.
Stack locale : `docker compose up postgres supertokens webapp` (ou `make dev-web` pour FastAPI seul).

## Migration Streamlit → FastAPI + HTMX

**ADR 0006** (statut : proposé, spike livré) : décision recommandée **FastAPI + HTMX + templates** (vélocité Python, même origine CapRover, majoritairement `pytest`).
Re-décision conditionnelle si prototype UX sur écran dense invalide HTMX (voir `docs/adr/0006-front-stack-bff-spa-vs-htmx.md`).

**Bascule** : cutover unique (`docs/streamlit_to_new_frontend_cutover.md`) — pas de double interface officielle pérenne. Développement sur `deploy-newfrontend` (préprod), puis promotion vers `deploy-caprover-relais4`.

Prérequis cité dans les stories 007–016 :
> Décision bascule — `docs/streamlit_to_new_frontend_cutover.md` — mode prod **cutover unique** ; coexistence deux interfaces en prod **non** (0 jour) ; support sur `APP_PUBLIC_BASE_URL`.

**Spike BFF** disponible : `src/bff_spike_app.py` (lecture `GET /issue-006-spike/projects/{id}/entries`, mutation `PATCH …/settings`), tests dans `tests/test_bff_spike_issue006.py`.

## Modèle de données et contrôle d'accès

- `users` / `projects` / `entries` / `project_settings` / `project_memberships`
- **Propriétaire unique** : `projects.created_by → users.id` — source de vérité pour la propriété.
- `project_memberships` (`admin`, `collaborator`, `viewer`) : persistance de collaboration, **sans second propriétaire**. `get_role()` ne lit **pas encore** cette table — seul le propriétaire passe les gardes-fous sur les données dataset.
- Super-admin (`users.is_super_admin`) : gouvernance des comptes uniquement, **pas** co-propriétaire implicite des projets d'autrui.
- Toute nouvelle route (BFF, API) doit réutiliser `require_role` / `require_admin` / `get_role` depuis `src/database.py` sans dupliquer la logique RBAC.

Référence : `docs/architecture/project_access_model.md` et `docs/multi_tenant_architecture.md`.

## Contrat d'erreurs API

`src/api_errors.py` : `error_envelope_for_client` / `resolve_exception_for_api` — codes stables (`AUTH_SESSION_EXPIRED`, `DB_UNAVAILABLE`, `FORBIDDEN`, `NOT_FOUND_GENERIC`), messages FR, pas de stack trace en prod.
Référence : `docs/api_error_contract.md`.

## Variables d'environnement clés

| Variable | Rôle |
|----------|------|
| `APP_CONFIG_JSON` | Config prod CapRover (secrets, URLs) — voir `docs/caprover_env_example.md` |
| `APP_PUBLIC_BASE_URL` | URL canonique publique (liens invitation/reset, cookies SuperTokens) |
| `DATABASE_URL` | `postgresql+psycopg://…` |
| `SUPERTOKENS_CONNECTION_URI` / `SUPERTOKENS_API_KEY` | Core SuperTokens |
| `AUTH_ENFORCE_INVITATION_ONLY` / `SUPERTOKENS_SIGNUP_DISABLED` | Contrôle inscription |
| `MAIL_MODE` | `dev` = log console sans envoi SMTP |
| `WEBAPP_CORS_ORIGINS` | Origines CORS autorisées si BFF et front sur origines distinctes |

Pas de secrets en dur dans le code.

## CI / qualité

- **`ci.yml`** : `ruff check .` + `ruff format --check .` + `pytest -q` — push et PR vers `main` et `deploy-caprover-relais4`.
- **`auth-contract.yml`** : cron + PR sur chemins sensibles (`src/auth.py`, `src/database.py`, `src/supertokens_recipe_client.py`, UI auth, tests auth, spike BFF).
- Marker pytest `postgres_regression` : requiert `DATASET_STYLE_REGRESSION_DATABASE_URL` (opt-in, non lancé par défaut).

En local avant merge vers `deploy-caprover-relais4` :
1. `pytest -q` — suite verte
2. `ruff check .` + `ruff format --check .`
3. `uv run python scripts/bootstrap_check.py`
4. Si PR modifie chemins auth : vérifier `auth-contract.yml` vert

Référence complète : `docs/release_train_caprover.md` et `docs/merge_ready_checklist.md`.

## Conventions

- Identifiants / modules en **anglais** ; UI en **français**.
- Chaînes FR longues et blocs Streamlit : `E501` ignoré pour fichiers listés dans `pyproject.toml`.
- `uv` pour gestion dépendances et exécution scripts.

## Backlog sprint actif — état (2026-05-19)

Sprint `01KRVME29F74DQ3KN8AXSF5MVX` — source : `.ai-team-orchestrator/sprints/01KRVME29F74DQ3KN8AXSF5MVX/backlog.json`.

| Issue | Titre (résumé) | Statut |
|-------|----------------|--------|
| 001 | Stratégie de bascule Streamlit → nouveau frontal | ✅ done |
| 002 | Figer la branche d'intégration et train de release | ✅ done |
| 003 | Trancher le modèle d'accès projet (propriétaire vs memberships) | ✅ done |
| 004 | Matrice de parité Streamlit → API | ✅ done |
| 005 | Contrat d'erreurs API | ✅ done |
| 006 | ADR stack frontal (BFF+SPA vs HTMX) avec spike | ✅ done |
| 007 | Slice vertical : auth + projet + 1 édition + export | ✅ done |
| 008 | Squelette FastAPI, Docker/compose, CI nouveau frontal | ✅ done |
| 009 | Jalon de non-régression chaîne curation (référence Streamlit) | ✅ done |
| 010 | [Parité curateur] Projets, session courante, shell navigation | 🔄 in_progress |
| 011 | [Parité curateur] Réglages projet, presets, dimensions | ⬜ todo |
| 012 | [Parité curateur] Nouvelle entrée et gestion & édition | 🔄 in_progress |
| 013 | [Parité curateur] Génération IA et contrôle LanguageTool | 🔄 in_progress |
| 014 | [Parité curateur] Stylométrie, cohérence, tableau de bord | ✅ done |
| 015 | [Parité curateur] Export CSV/JSONL et onboarding projet vide | 🔄 in_progress |
| 016 | [Parité curateur] Mon compte (profil, déconnexion) | ✅ done |
| 017 | [Super-admin] Inviter un collaborateur par e-mail | 🔄 in_progress |
| 018 | [Super-admin] Lister les comptes avec pagination | ✅ done |
| 019 | [Super-admin] Panneau technique saga avec relances protégées | ✅ done |
| 020 | Définir une baseline UX (temps scénario critique) | 🔄 in_progress |
| 021 | Communiquer la continuité de service | ✅ done |
| 022 | Design tokens minimaux et bandeaux sémantiques (phase 1) | ⬜ todo |
| 023 | Réglages d'affichage optionnels (densité, confort) | ⬜ todo |

Clôture epic **parité curateur** : issues 010–016 toutes `done`.

## Sync GitHub ↔ backlog

Toutes les issues du sprint actif ont un `githubIssueNumber` (GitHub #124–145).
Ne **pas** créer de doublons : vérifier `githubIssueNumber` avant tout `--sync-issues`.
Cible de merge pipeline : **`deploy-caprover-relais4`**, pas `main`.

## Docs de référence

| Document | Sujet |
|----------|-------|
| `docs/release_train_caprover.md` | Branches, CapRover, checklist pré-merge |
| `docs/merge_ready_checklist.md` | Checklist complète avant merge |
| `docs/adr/0006-front-stack-bff-spa-vs-htmx.md` | Décision stack frontal |
| `docs/streamlit_to_new_frontend_cutover.md` | Stratégie de bascule |
| `docs/architecture/project_access_model.md` | Modèle propriétaire/memberships |
| `docs/multi_tenant_architecture.md` | Architecture multi-tenant |
| `docs/api_error_contract.md` | Contrat erreurs API |
| `docs/dev_new_frontend.md` | Lancer le service FastAPI localement |
| `docs/caprover_deployment.md` / `docs/caprover_env_example.md` | Déploiement prod |
| `docs/migration_parity_matrix.md` | Matrice parité Streamlit → API |
| `docs/session_state_keys.md` | Clés session Streamlit (référence migration) |
