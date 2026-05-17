# Train de release CapRover et branches Git

Ce document fige la **source de vérité** côté Git et le **train de déploiement** CapRover, pour que la CI, les contrats auth et les recettes s’appliquent au bon code.

## Rôles des branches

| Branche | Rôle | Source de vérité |
|---------|------|-------------------|
| `main` | Historique / miroir éventuel ; **ne pas y merger** les livraisons Relais4 tant que la prod suit `deploy-caprover-relais4`. | Non pour le déploiement CapRover actuel. |
| `deploy-caprover-relais4` | **Branche d’intégration et de déploiement production** Relais4 : toute PR fonctionnelle et doc d’exploitation destinée à la prod doit **merger ici** (`make prod` / `caprover deploy`). | Oui pour ce qui est **réellement déployé** sur l’instance CapRover Relais4. |
| `deploy-newfrontend` *(exemple / future)* | **Branche d’intégration de migration** (nouveau front, refonte Streamlit, etc.) : à créer ou renommer selon la stratégie de bascule ; distincte de la prod actuelle. Décision de bascule (cutover prod, URL canonique, rollback) : **`docs/streamlit_to_new_frontend_cutover.md`** — prérequis pour les stories **007–016**. | Non tant qu’elle n’existe pas ; une fois créée, elle sert de rampe d’intégration **avant** promotion vers `deploy-caprover-relais4`. |

**À ne pas confondre** : l’ancien nom documenté `deploy/caprover-relais4` (avec slash) ne correspond **pas** à une branche du dépôt distant ; la branche réelle est **`deploy-caprover-relais4`** (tiret).

## Tableau branche → CapRover → workflows GitHub Actions

| Branche Git | Environnement CapRover (typique) | `ci.yml` (lint ruff + format + `pytest -q`) | `auth-contract.yml` |
|-------------|----------------------------------|-------------------------------------------|----------------------|
| `main` | Pas d’environnement implicite ; recette locale / preview selon l’équipe. | **Oui** : push et PR ciblant `main`. | **Planifié** (cron) ; **PR** si les chemins sensibles auth changent (voir workflow). |
| `deploy-caprover-relais4` | App **`dataset-style`** (production Relais4), avec `APP_CONFIG_JSON` et stack PG + SuperTokens décrite dans `docs/caprover_deployment.md`. | **Oui** : push et PR ciblant cette branche. | Idem `main`. |
| Branche de migration *(ex. `deploy-newfrontend` une fois créée)* | Environnement CapRover **staging / préprod** dédié (à provisionner ; pas la prod Relais4). | **Oui** si la branche est ajoutée explicitement dans `on:` de `ci.yml` (sinon uniquement via PR vers `main` / `deploy-caprover-relais4`). | Idem selon chemins des fichiers sur la PR. |

Détails des filtres `auth-contract` : voir `.github/workflows/auth-contract.yml` (section `on.pull_request.paths`).

## Procédure : valider avant merge sur la branche de déploiement

Avant tout merge vers **`deploy-caprover-relais4`** (ou vers une branche d’intégration de migration qui alimente la prod), exécuter et cocher :

1. **Tests** : `pytest -q` — suite verte, pas de régression.
2. **Lint / format** : `ruff check .` et `ruff format --check .` (aligné sur la CI).
3. **Bootstrap / schéma** : `uv run python scripts/bootstrap_check.py` (ajouter `--apply-schema` si le ticket l’exige après validation DBA).
4. **Contrats auth** : si la PR modifie un fichier listé dans `auth-contract.yml` (`src/auth.py`, `src/database.py`, etc.), vérifier que le workflow **Auth Contract** est vert sur la PR ; en local, respecter la checklist `docs/merge_ready_checklist.md` (section sécurité auth).
5. **Merge-ready métier** : parcourir `docs/merge_ready_checklist.md` pour les garde-fous invitation-only, saga comptes et super admin.

Référence CapRover (variables, ordre des apps) : `docs/caprover_deployment.md` et `docs/caprover_env_example.md`.
