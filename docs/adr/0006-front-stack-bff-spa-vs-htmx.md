# ADR 0006 — Stack frontal : FastAPI BFF + SPA (JSON) vs FastAPI + HTMX / templates

- **Statut** : **accepté** (jalon UX écran dense + revue architecture — GitHub **#177** ; spike historique issue-006)
- **Date de la dernière mise à jour** : 2026-05-19
- **Date du spike initial** : 2026-05-17
- **Redevables (jalon / décision)** : sponsor produit / orchestrateur backlog ; architecte technique référent BFF + `src/database.py` ; curateur référent (recette parité) ; UX interne pour les timebox critiques suivantes (voir § jalons futurs).
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

## Jalon UX « écran dense » (clôture critère #177)

**Objectif** : trancher le statut « proposé » après évaluation d’un **écran représentatif** couvrant **liste + fiche + panneaux IA / LanguageTool**, avec la grille implicite de l’ADR (densité, friction de navigation verticale, focus / charge cognitive) et les critères du tableau *Critères de comparaison* ci-dessus.

### Surface évaluée (webapp FastAPI)

| Zone | Réalisation dans le dépôt |
|------|---------------------------|
| Liste d’entrées dense (colonnes métier + filtres édition) | `GET /api/projects/{project_id}/entries` + paramètres `edition_*` ; affichage côté client dans la coquille `GET /` (`src/webapp/index_template.py`). |
| Fiche / édition | `PATCH /api/projects/{project_id}/entries/{entry_id}` ; corps `entries` après succès (alignement `docs/migration_parity_matrix.md`, post-mutation). |
| Génération IA | `POST /api/projects/{project_id}/curator/llm-generate` — logique `src/webapp/curator_ai.py` (`require_role`, `get_project_settings`, pas de clé exposée). |
| LanguageTool | `POST /api/projects/{project_id}/curator/languagetool-check` — même module ; erreurs via enveloppes stables (`CURATOR_LANGUAGETOOL_UNAVAILABLE`, etc.). |

Aucune logique RBAC dupliquée dans les handlers : les garde-fous passent par **`src/database.py`** ; les erreurs HTTP JSON par **`src/api_errors.py`** / résolveurs existants.

### Grille de notation (jalon)

Échelle : **A** = confortable pour viser le cutover ; **B** = acceptable avec dette UX / technique connue et suivie ; **C** = risque produit avant cutover ; **D** = bloquant.

| Critère jalon | Observation (synthèse) | Note |
|---------------|-------------------------|------|
| **Densité** (liste + champs fiche visibles sans perte de contexte) | Table scrollable + panneau fiche ; pas de grille type DataGrid Excel, mais suffisant pour le périmètre curateur actuel. | **B** |
| **Friction scroll** (passage liste ↔ fiche, retour contexte projet) | Navigation **single page** côté client (`fetch` JSON) ; pas de rechargement HTML complet à chaque action. | **B** |
| **Focus / charge cognitive** | Onglets alignés `main_tab_labels` isolent IA, LT, export et compte ; charge maîtrisable. | **B** |
| **Vélocité équipe Python** (ligne ADR) | Itération sur handlers + gabarit + `pytest` sans chaîne front séparée. | **A** |
| **UX riche type grille ultra-interactive** (ligne ADR) | Pas au niveau d’une SPA composants ; acceptable tant que le besoin métier reste édition ligne à ligne + filtres. | **B** |
| **Déploiement CapRover monorigine** | Inchangé — un service `webapp`, cookies SuperTokens cohérents avec `APP_PUBLIC_BASE_URL`. | **A** |
| **Tests** | Majoritairement `pytest` sur routes et contrats JSON ; pas de suite Vitest obligatoire. | **A** |
| **Anti-IDOR** | Aligné : `TenantResourceOpaqueDenial` → enveloppe client documentée. | **A** |

**Conclusion du jalon** : aucun signal **C/D** sur le périmètre évalué ; la trajectoire **FastAPI + gabarits HTML + API JSON** (éventuels swaps partiels type HTMX **plus tard**, non bloquants) reste **validée** pour la suite des slices **010–016** et le cutover décrit dans `docs/streamlit_to_new_frontend_cutover.md`.

### Risque technique noté (suivi dette)

Le fichier `src/webapp/index_template.py` est aujourd’hui **monolithique**. Si une **re-décision** imposait une SPA ou un îlot riche, le coût sera maîtrisé en **extrayant tôt** des fragments HTML / endpoints partiels — ce point reste une recommandation d’architecture, pas un bloquant à la décision #177.

## Décision tranchée (post-jalon)

1. **Stack cible** : **FastAPI + gabarits** (HTML statique servi par FastAPI — aujourd’hui `INDEX_HTML` dans `src/webapp/index_template.py` ; évolution possible vers Jinja2 ou swaps HTMX ciblés) avec **consommation JSON** côté navigateur (`fetch`) sur les routes `/api/...` existantes ; **pas** d’exigence « SPA jour 1 » ni d’empilement HTMX obligatoire tant que la navigation reste prévisible et testable.
2. **Pas de rejet** du scénario SPA : il reste **plan B** documenté ci-dessous si un besoin **C/D** apparaît (grille extrême, exigence produit cross-origin non résolvable, ou autre).
3. **Re-décision** : déclenchée seulement si les conditions du § *Re-decision* ci-dessous se réalisent **après** ce jalon.

## Go / no-go (historique — conservé pour traçabilité)

- **Go (recommandation par défaut)** : enchaîner avec **FastAPI + HTMX + templates** pour la première slice migrée **si** l’objectif prioritaire est la vélocité Python, un déploiement CapRover simple (même origine), et des tests majoritairement `pytest` sans dette SPA immédiate.
- **No-go sur « SPA obligatoire dès le jour 1 »** sans prototype : si une UX dense (grille + fiche édition riche) est **must-have** et qu’HTMX ne suffit pas après prototype court, imposer une SPA **sans** prototype UX est un **no-go** (sous-estimation du coût SPA vs charge équipe).
- **Go SPA conditionnel** : acceptable si (a) une **preuve UX** sur un écran dense valide le besoin, **ou** (b) une capacité front dédiée assume explicitement contrats API, accessibilité, perf et CI front.

### Alternatives si la décision initiale se révèle trop étroite

1. **Hybride** : coquille HTMX pour navigation + îlots SPA (web components ou montage React ciblé) pour la grille seule.
2. **SPA monorigine** : build statique servi par le même FastAPI derrière CapRover (pas de CORS nominal).
3. **BFF dédié + SPA hébergée ailleurs** : garder le BFF Python comme seule couche parlant à PostgreSQL ; accepter la complexité CORS/cookies et la formaliser dans la config CapRover / reverse proxy.

**Estimation d’ordre de grandeur si plan B « SPA large » avant cutover** (non contractuelle) : **8 à 14** user stories équivalentes supplémentaires (contrats API stabilisés, CI front, accessibilité dense, perf réseau) **en plus** du travail de parité déjà planifié — à resynchroniser avec le backlog si un sponsor impose ce virage.

### Re-decision (sans calendrier wall-clock)

Re-trancher **après** : (i) ~~prototype ou slice réelle sur un écran dense~~ — **réalisé (jalon #177)** ; **réouverture** si une **nouvelle** exigence UX passe en grade **C/D** (ex. grille > N lignes éditables simultanément, latence interaction non acceptable), **ou** (ii) changement majeur sur `get_role` / membres projet (issue-003) impactant les permissions dataset, **ou** (iii) contrainte produit imposant une origine front distincte non neutralisable par reverse proxy.

## Liens

- Spike : `src/bff_spike_app.py`
- Contrat erreurs : `docs/api_error_contract.md`
- Bascule / URL canonique : `docs/streamlit_to_new_frontend_cutover.md`
