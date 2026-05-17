# ADR 0006 — Stack frontal après Streamlit : FastAPI BFF + SPA vs HTMX / templates

## Statut

**Accepté** (issue-006) — **Go** sur la trajectoire principale décrite ci-dessous, avec **re-arbitrage** après prototype d’édition riche (voir section *Go / no-go*).

Référence spike code : `src/bff_spike_app.py` (routes préfixées `/migration-spike/v1/…`, hors surface prod tant que non routées).

---

## Contexte

Le produit quitte Streamlit pour un frontal HTTP ; la logique métier et les garde-fous multi-tenant restent centralisés dans `src/database.py` (`require_role`, `require_admin`, etc.). Il faut trancher une stack **compatible CapRover**, **testable**, et **alignée** sur la bascule auth / domaines (issue-001 / `docs/streamlit_to_new_frontend_cutover.md`).

---

## Décision (résumé)

| Option | Décision |
|--------|----------|
| **FastAPI + API JSON + client SPA** (React, Vue, Svelte ou équivalent) | **Voie principale retenue** pour le périmètre « dataset » (édition riche, tableaux, filtres). |
| **FastAPI + HTMX + templates Jinja** (SSR poussée) | **Complément possible** pour des écrans simples ou du contenu majoritairement form-based ; **non retenu** comme **unique** stack si l’objectif est la parité d’UX avec l’édition Streamlit actuelle sur grilles / vues denses. |

---

## Critères de comparaison

### 1. Vélocité équipe Python

- **HTMX + templates** : avantage sur formulaires et pages CRUD simples (moins de toolchain JS, cycles courts).
- **SPA** : courbe initiale plus haute (tooling, état client, design system) ; la vélocité **moyenne** dépend fortement de l’expertise front de l’équipe. Mitigation : monorepo minimal, composants data-grid éprouvés, BFF Python qui reste la source de vérité métier.

### 2. UX riche (édition texte, tableaux, navigation dense)

- **SPA** : meilleur alignement sur composants avancés (grilles virtualisées, édition inline, undo local, raccourcis) sans contorsion du modèle « document HTML + fragments HTMX ».
- **HTMX** : viable pour des flux **par action serveur** ; l’édition type « feuille » reste souvent **plus coûteuse** en interactions fines (latence réseau par cellule, état partiel côté client) sauf à réintroduire beaucoup de JavaScript — auquel cas la frontière avec une SPA s’efface.

### 3. Déploiement CapRover

- **Les deux** : une app **FastAPI** (image Docker) derrière CapRover, comme aujourd’hui pour Streamlit (un conteneur, healthcheck HTTP). Le client SPA est servi soit par le même conteneur (fichiers statiques + catch-all), soit par un build statique séparé selon le pipeline CI — sans impact sur le modèle **PostgreSQL + SuperTokens** existant.

### 4. Tests

- **Les deux** : tests Python sur le BFF (`pytest`, contrat d’erreurs `src/api_errors.py`). Couche SPA : tests composants / e2e (Playwright ou équivalent) **en plus** — coût réel mais maîtrisable par périmètre incrémental.
- **Filets existants** : la suite **`pytest`** et le workflow **`.github/workflows/auth-contract.yml`** restent les garde-fous ; tout fichier touchant l’auth applicative ou les garde-fous DB (ex. `src/bff_spike_app.py`) doit continuer à déclencher **auth-contract** sur PR.

---

## Session SuperTokens (cohérence issue-001)

Décisions alignées sur `docs/streamlit_to_new_frontend_cutover.md` :

- **Cookies HTTP-only** gérés par le flux SuperTokens ; le **BFF** valide la session (via SDK / introspection recipe) et résout l’`user_id` applicatif — **pas** de stockage de secrets métier dans le navigateur au-delà de la session.
- **Same-site / domaine** : une **seule** URL canonique utilisateur en production après cutover ; `APP_PUBLIC_BASE_URL` et la configuration SuperTokens (callbacks, liens e-mail, cookies) doivent rester **strictement alignés**.
- **CORS** : en **same-origin** (front et BFF servis sur le même site), CORS reste **minimal**. Si un build statique devait être servi sur un autre sous-domaine, une **décision écrite** fixerait les origines autorisées et la politique cookies (sinon risque de double session, voir section coexistence du document de cutover).

---

## Spike technique (implémenté)

Objectif : prouver qu’un routeur FastAPI peut **réutiliser** les fonctions existantes sans recopier la logique métier.

| Verbe | Route | Garde-fou `database.py` | Réponse |
|-------|-------|---------------------------|---------|
| `GET` | `/migration-spike/v1/projects/{project_id}/entries-summary` | `load_project_entries` → `require_role(..., ("admin","collaborator","viewer"))` | JSON dérivé du DataFrame serveur (comptages). |
| `PATCH` | `/migration-spike/v1/projects/{project_id}/settings/active-preset` | `require_admin` puis `update_project_settings` ; relecture `get_project_settings` | JSON = **état canonique** post-écriture (pas d’état client « cache Streamlit »). |

Auth spike : en-tête **`X-Spike-Actor-User-Id`** (placeholder de recette uniquement). En intégration réelle : remplacer par la résolution d’utilisateur depuis SuperTokens.

Erreurs : enveloppe `error_envelope_for_client` / statuts via `resolve_exception_for_api` (`docs/api_error_contract.md`).

---

## Go / no-go

| Verdict | Détail |
|---------|--------|
| **Go** | Poursuite de la migration sur **FastAPI BFF + API JSON + SPA** comme référence pour le frontal principal, avec rappel que la logique d’accès projet reste dans `src/database.py` et le modèle d’accès documenté (`docs/architecture/project_access_model.md`). |
| **No-go (seul mode)** | **Non** au remplacement **exclusif** par **HTMX + templates** si le périmètre « parité édition riche » est maintenu sans réduction du scope : risque trop élevé de complexité cachée côté client ou de dégradation UX. |
| **Alternatives si repli** | (1) SPA pour le cœur dataset + pages annexes en SSR minimal ; (2) réduction du scope métier sur les vues denses ; (3) **re-décision** après un **prototype utilisateur** ciblant la grille d’édition (critères : latence perçue, erreurs concurrentes, accessibilité). **Délai de re-décision** : à caler avec le sponsor à la fin du sprint contenant le prototype grille (pas de date calendaire imposée ici). |

---

## Conséquences

- Maintenir **un seul** socle métier (`src/database.py`) pour les opérations projet ; le BFF ne fait que traduire HTTP ↔ appels typés.
- Prévoir la suite : intégration SuperTokens sur le BFF, suppression de l’en-tête spike, exposition derrière la même origine que le front.
- Documenter dans les stories 007+ la dépendance à ce ADR et à `docs/streamlit_to_new_frontend_cutover.md`.

---

## Références

- `docs/streamlit_to_new_frontend_cutover.md` (issue-001)
- `docs/architecture/project_access_model.md` (issue-003)
- `docs/api_error_contract.md`
- `docs/caprover_deployment.md`
- `.github/workflows/ci.yml`, `.github/workflows/auth-contract.yml`
