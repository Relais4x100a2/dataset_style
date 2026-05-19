# Train de release CapRover et branches Git

Ce document fige la **source de vérité** côté Git et le **train de déploiement** CapRover, pour que la CI, les contrats auth et les recettes s’appliquent au bon code.

## Rôles des branches

| Branche | Rôle | Source de vérité |
|---------|------|-------------------|
| `main` | Historique / miroir éventuel ; **ne pas y merger** les livraisons Relais4 tant que la prod suit `deploy-caprover-relais4`. | Non pour le déploiement CapRover actuel. |
| `deploy-caprover-relais4` | **Branche d’intégration et de déploiement production** Relais4 : toute PR fonctionnelle et doc d’exploitation destinée à la prod doit **merger ici** (`make prod` / `caprover deploy`). | Oui pour ce qui est **réellement déployé** sur l’instance CapRover Relais4. |
| `deploy-newfrontend` | **Branche d’intégration de migration** (nouveau front, refonte Streamlit) : rampe **préprod** distincte de la prod actuelle. Décision de bascule (cutover prod, URL canonique, rollback) : **`docs/streamlit_to_new_frontend_cutover.md`** — prérequis pour les stories **007–016**. | Oui pour l’**intégration continue** de la migration **avant** promotion vers `deploy-caprover-relais4`. |

**À ne pas confondre** : l’ancien nom documenté `deploy/caprover-relais4` (avec slash) ne correspond **pas** à une branche du dépôt distant ; la branche réelle est **`deploy-caprover-relais4`** (tiret).

## Stratégie de merge (sans ambiguïté)

1. **Développement** : branches de travail (ex. `cursor/…`, `ai-team/pipeline`, features) ouvrent des **PR vers `deploy-newfrontend`** pour intégrer le code de migration et la doc associée.
2. **Préprod** : la branche **`deploy-newfrontend`** est déployée sur l’**environnement CapRover staging** (PG + SuperTokens + apps dédiés préprod, voir `docs/caprover_deployment.md` §4.6). Les garde-fous locaux sont les mêmes qu’en checklist ci‑dessous (`pytest`, ruff, `bootstrap_check`).
3. **Promotion production** : lorsque la préprod est validée, une **PR de promotion** merge **`deploy-newfrontend` → `deploy-caprover-relais4`** (tiret). La production Relais4 ne suit **pas** `main` pour ce train ; ne pas utiliser le nom erroné **`deploy/caprover-relais4`** (slash) comme branche Git.

Les PR issues du pipeline d’équipe peuvent cibler `deploy-newfrontend` ou, pour les correctifs directement prod‑ready, **`deploy-caprover-relais4`** selon la gouvernance release ; la règle fixe reste : **intégration migration sur `deploy-newfrontend`**, **cutover et déploiement utilisateur sur `deploy-caprover-relais4`**.

## Publier la branche `deploy-newfrontend` sur GitHub

La branche doit **exister sur le remote** pour que les triggers `on.push` / `on.pull_request` de la CI s’exécutent. À faire **une fois** (compte avec droits push), à partir du point d’intégration souhaité (souvent l’extrémité actuelle de `deploy-caprover-relais4`) :

```bash
git fetch origin
git branch deploy-newfrontend origin/deploy-caprover-relais4
git push -u origin deploy-newfrontend
```

Renommer ultérieurement la branche impliquerait de mettre à jour **les mêmes listes** dans `.github/workflows/ci.yml` et `.github/workflows/auth-contract.yml` pour éviter l’écart doc / CI.

## Tableau branche → CapRover → workflows GitHub Actions

| Branche Git | Environnement CapRover (typique) | `ci.yml` (lint ruff + format + `pytest -q`) | `auth-contract.yml` |
|-------------|----------------------------------|-------------------------------------------|----------------------|
| `main` | Pas d’environnement implicite ; recette locale / preview selon l’équipe. | **Oui** : push et PR ciblant `main`. | **Planifié** (cron) ; **PR** si les chemins sensibles auth changent (voir workflow). |
| `deploy-caprover-relais4` | App **`dataset-style`** (production Relais4), avec `APP_CONFIG_JSON` et stack PG + SuperTokens décrite dans `docs/caprover_deployment.md`. | **Oui** : push et PR ciblant cette branche. | **Planifié** (cron) ; **PR** si chemins sensibles. |
| `deploy-newfrontend` | **Staging / préprod** dédié : triplet PG + SuperTokens + app(s) avec secrets et **`APP_PUBLIC_BASE_URL` préprod** (voir `docs/caprover_deployment.md` §4.6 et placeholders `docs/caprover_env_example.md`). | **Oui** : push et PR ciblant cette branche (listées dans `ci.yml`). | **Push** sur cette branche : workflow **complet** à chaque commit (pas de filtre `paths`, évite un faux négatif sur la rampe migration) ; **PR** si chemins sensibles ; **cron** inchangé. |

Détails des filtres `auth-contract` : voir `.github/workflows/auth-contract.yml` (`on.push.branches`, `on.pull_request.paths`).

## Procédure : valider avant merge sur la branche de déploiement

Avant tout merge vers **`deploy-caprover-relais4`**, vers **`deploy-newfrontend`**, ou avant promotion **`deploy-newfrontend` → `deploy-caprover-relais4`**, exécuter et cocher :

1. **Tests** : `pytest -q` — suite verte, pas de régression.
2. **Lint / format** : `ruff check .` et `ruff format --check .` (aligné sur la CI).
3. **Bootstrap / schéma** : `uv run python scripts/bootstrap_check.py` (ajouter `--apply-schema` si le ticket l’exige après validation DBA).
4. **Contrats auth** : si la PR modifie un fichier listé dans `auth-contract.yml` (`src/auth.py`, `src/database.py`, etc.), vérifier que le workflow **Auth Contract** est vert sur la PR ; pour un **push direct** sur `deploy-newfrontend`, le workflow se déclenche **systématiquement** (pas de filtre par chemins). En local, respecter la checklist `docs/merge_ready_checklist.md` (section sécurité auth).
5. **Merge-ready métier** : parcourir `docs/merge_ready_checklist.md` pour les garde-fous invitation-only, saga comptes et super admin.

Référence CapRover (variables, ordre des apps) : `docs/caprover_deployment.md` et `docs/caprover_env_example.md`.
