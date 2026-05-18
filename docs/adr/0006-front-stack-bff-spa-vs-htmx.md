# ADR 0006 — Stack frontal : FastAPI BFF + SPA (JSON) vs FastAPI + HTMX / templates

- **Statut** : proposé (issue-006, spike livré)
- **Date** : 2026-05-17
- **Contexte** : migration hors Streamlit (issue-001 / `docs/streamlit_to_new_frontend_cutover.md`) sans dupliquer la logique métier hors de `src/database.py` ; alignement modèle d’accès issue-003 (`docs/architecture/project_access_model.md`).

## Contexte

L’application historique couple UI et session dans Streamlit. Le futur frontal doit :

- réutiliser les garde-fous existants (`require_role`, `require_admin`, `get_role`) ;
- exposer des erreurs JSON stables (`src/api_errors.py`, `docs/api_error_contract.md`) ;
- rester déployable sur CapRover avec la même base PostgreSQL et SuperTokens.

## Critères de comparaison

| Critère | FastAPI + API JSON + SPA (React / Vue / Svelte) | FastAPI + HTMX + templates (Jinja2 ou équivalent) |
|--------|--------------------------------------------------|---------------------------------------------------|
| Vélocité équipe Python | Moins : surface API + contrats + équipe front ou courbe SPA | Plus : pages et handlers côté serveur, itération mono-langage |
| UX riche (édition texte, grilles denses) | Plus : écosystème composants (DataGrid, éditeurs) | Mitigé : possible (HTMX + libs JS ciblées) mais moins standard pour grilles très interactives |
| Déploiement CapRover | Simple en **monorigine** (même host) ; plus sensible si API et assets sur **origines distinctes** (CORS, cookies, double build) | Simple : un service (ou BFF + assets statiques) derrière le même domaine public |
| Tests | `pytest` sur API + tests front dédiés (Vitest, etc.) | `pytest` sur routes et fragments HTML ; moins de duplication de contrat si le HTML reste fin |
| Cohérence avec anti-IDOR | Identique si le BFF appelle uniquement `database.py` et mappe `TenantResourceOpaqueDenial` → enveloppe 404 | Identique |

## Décision — session SuperTokens (alignement issue-001)

Décision documentée, cohérente avec la bascule décrite dans `docs/streamlit_to_new_frontend_cutover.md` :

1. **Transport de session** : cookies httpOnly émis par SuperTokens (flux existant), pas stockage du jeton en `localStorage` pour le chemin nominal.
2. **Domaine et SameSite** : le domaine des cookies et les URL de callback / reset / invitation restent alignés sur **`APP_PUBLIC_BASE_URL`** et l’hôte réellement servi derrière CapRover (éviter dérives `www` vs apex).
3. **SameSite** : `Lax` par défaut pour le parcours utilisateur classique ; n’ajuster vers `None` + `Secure` **que** si un scénario légitime impose des navigations cross-site contrôlées (et après revue explicite sécurité).
4. **CORS** : si le frontal et le BFF ne partagent **pas** la même origine, dimensionner explicitement `Access-Control-Allow-Origin` (liste fermée, pas `*`) avec `credentials: true` côté client, et documenter les en-têtes autorisés ; idéalement **réduire l’écart d’origine** (reverse proxy unique, préfixe `/api`).

## Spike technique (preuve de faisabilité)

Code : `src/bff_spike_app.py`, usine `create_spike_bff_app(engine, actor_user_id_factory=...)`.

- **Lecture** : `GET /issue-006-spike/projects/{project_id}/entries` → `load_project_entries` (donc `require_role` dans `database.py`).
- **Mutation** : `PATCH /issue-006-spike/projects/{project_id}/settings` → `require_admin`, fusion partielle des champs sur l’état courant via `get_project_settings` + `update_project_settings`, puis réponse avec **état canonique serveur** (`get_project_settings` après écriture). Aucun cache Streamlit.
- **Erreurs** : `TenantResourceOpaqueDenial`, `AuthSessionExpiredError`, `OperationalError`, `PermissionError`, `RequestValidationError` mappées via `error_envelope_for_client` / `resolve_exception_for_api` (`docs/api_error_contract.md`).
- **Identité dans le spike** : injectée par `actor_user_id_factory` (tests) ; en production le BFF résoudra `users.id` depuis la session SuperTokens **sans** réimplémenter RBAC dans les handlers.

**Comportement métier actuel** : tant que `get_role` ne lit pas `project_memberships` pour le dataset, seul le **propriétaire** du projet passe les garde-fous sur les données (voir ADR et `project_access_model.md`). Le spike reflète ce comportement (pas de présupposé « membre = droits d’édition dataset »).

## Filets qualité et sécurité

- **`pytest`** : tests du spike dans `tests/test_bff_spike_issue006.py` (contrats HTTP, enveloppes, absence de branche RBAC locale dans le module BFF).
- **`.github/workflows/ci.yml`** : `ruff` + `pytest` sur les branches d’intégration définies dans le train de release.
- **`.github/workflows/auth-contract.yml`** : déclenché lorsque des chemins sensibles auth / DB / UI auth évoluent ; inclut désormais le spike BFF pour forcer une revue lorsque l’identité côté API change.

## Go / no-go

- **Go (recommandation par défaut)** : enchaîner avec **FastAPI + HTMX + templates** pour la première slice migrée **si** l’objectif prioritaire est la vélocité Python, un déploiement CapRover simple (même origine), et des tests majoritairement `pytest` sans dette SPA immédiate.
- **No-go sur « SPA obligatoire dès le jour 1 »** sans prototype : si une UX dense (grille + fiche édition riche) est **must-have** et qu’HTMX ne suffit pas après prototype court, imposer une SPA **sans** prototype UX est un **no-go** (sous-estimation du coût SPA vs charge équipe).
- **Go SPA conditionnel** : acceptable si (a) une **preuve UX** sur un écran dense valide le besoin, **ou** (b) une capacité front dédiée assume explicitement contrats API, accessibilité, perf et CI front.

### Alternatives si la décision initiale se révèle trop étroite

1. **Hybride** : coquille HTMX pour navigation + îlots SPA (web components ou montage React ciblé) pour la grille seule.
2. **SPA monorigine** : build statique servi par le même FastAPI derrière CapRover (pas de CORS nominal).
3. **BFF dédié + SPA hébergée ailleurs** : garder le BFF Python comme seule couche parlant à PostgreSQL ; accepter la complexité CORS/cookies et la formaliser dans la config CapRover / reverse proxy.

### Re-decision (sans calendrier wall-clock)

Re-trancher **après** : (i) prototype ou slice réelle sur **un** écran dense (table + édition), **ou** (ii) changement majeur sur `get_role` / membres projet (issue-003) impactant les permissions dataset, **ou** (iii) contrainte produit imposant une origine front distincte non neutralisable par reverse proxy.

## Liens

- Spike : `src/bff_spike_app.py`
- Contrat erreurs : `docs/api_error_contract.md`
- Bascule / URL canonique : `docs/streamlit_to_new_frontend_cutover.md`
