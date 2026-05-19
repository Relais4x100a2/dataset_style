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

## Cohérence auth (ADR 0006)

- Aligner **`APP_PUBLIC_BASE_URL`** sur l’URL réellement servie aux testeurs (ex. `http://localhost:8080` pour un test local du slice web seul), afin que les liens invitation / reset et les cookies SuperTokens restent cohérents.
- **`WEBAPP_CORS_ORIGINS`** : liste fermée d’origines autorisées (séparées par des virgules) si le navigateur appelle le BFF depuis une origine distincte ; en monorigine (même schéma, hôte et port), la valeur par défaut `http://localhost:8080` suffit.

Référence : `docs/adr/0006-front-stack-bff-spa-vs-htmx.md` et `docs/streamlit_to_new_frontend_cutover.md` pour le mode production (cutover unique).

## Tests et CI

Les routes FastAPI sont couvertes par `pytest` (`tests/test_webapp_vertical_slice.py`, `tests/test_webapp_health_issue008.py`, spike ADR `tests/test_bff_spike_issue006.py`). Le workflow `.github/workflows/ci.yml` inclut une étape de smoke dédiée au healthcheck avant la suite complète.

## Préférences d'affichage (issue-023)

- Lecture : champ `uiPreferences` sur `GET /api/account` (`density`, `readingComfort`, valeurs par défaut `default`).
- Mise à jour : `PATCH /api/account/ui-preferences` avec corps JSON partiel (`density` et/ou `readingComfort`). Réponse : `{ "uiPreferences": { ... } }`.
- Persistance : colonne `users.ui_preferences_json` (créée par `ensure_schema()` au boot Streamlit / webapp). Détails UX et mapping CSS : `docs/ui_display_preferences.md`.
