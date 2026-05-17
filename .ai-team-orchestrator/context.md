# Contexte orchestrateur — Dataset Style

## Source de vérité Git

| Concept | Branche |
|--------|---------|
| **Cible de merge PR / pipeline** (prod Relais4) | `deploy-caprover-relais4` — **pas `main`** |
| Déploiement CapRover production (Relais4) | `deploy-caprover-relais4` |
| Intégration de migration (ex. nouveau front) | Nom à figer côté produit (ex. `deploy-newfrontend`) — **distinct** de `deploy-caprover-relais4` |

La branche distante réelle pour la prod est **`deploy-caprover-relais4`** (tiret, pas `deploy/caprover-relais4`).

## Workflows GitHub Actions

- **`ci.yml`** : `ruff check`, `ruff format --check`, `pytest -q` — déclenché sur push et PR vers **`main`** et **`deploy-caprover-relais4`**.
- **`auth-contract.yml`** : horaire + PR filtrées sur chemins auth / DB / UI auth / tests auth.

## Checklist avant merge déploiement

Voir **`docs/release_train_caprover.md`** (pytest, ruff, bootstrap_check, auth-contract si applicable) et **`docs/merge_ready_checklist.md`**.

## Docs déploiement

- `docs/caprover_deployment.md`
- `docs/caprover_env_example.md`

## Sync GitHub ↔ backlog (attention)

- Les numéros GitHub **ne correspondent pas** aux ids `issue-NNN` du backlog (ex. GitHub #9 = ancien sprint Streamlit, pas `issue-001` migration).
- Ne pas marquer `done` dans `backlog.json` uniquement parce qu’une issue GitHub fermée porte un titre `[issue-00X]`.
- Créer de **nouvelles** issues GitHub par sprint migration ; lier explicitement `githubIssueNumber` après création.
- Cible de merge pipeline : **`deploy-caprover-relais4`**, pas `main`.
