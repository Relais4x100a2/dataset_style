# Dataset Style — repères pour assistants / équipe

## Dépôt

Projet **Dataset Style** (Streamlit, PostgreSQL, SuperTokens). Code applicatif sous `src/`, point d’entrée `main.py`.

## Branches et train de release

- **`deploy-caprover-relais4`** : **cible de merge** pour les PR et le pipeline prod CapRover (Relais4) — **ne pas merger les livraisons sur `main`** pour ce train.
- **`main`** : branche secondaire / historique ; pas la rampe de déploiement actuelle.
- Ne pas confondre avec un nom contenant un slash (`deploy/caprover-relais4` n’est pas la branche distante).
- **Branche d’intégration de migration** (ex. `deploy-newfrontend`) : à distinguer de la prod actuelle ; création et nommage suivent la stratégie de bascule produit.

Documentation canonique : **`docs/release_train_caprover.md`** (tableau branche → CapRover → workflows, checklist pré-merge).

## CI / qualité

- **Lint + tests** : `.github/workflows/ci.yml` sur `main` et `deploy-caprover-relais4`.
- **Contrats auth** : `.github/workflows/auth-contract.yml` (cron + PR sur chemins sensibles).

En local : `ruff check .`, `ruff format --check .`, `pytest -q`, `uv run python scripts/bootstrap_check.py`.

## Conventions

- Identifiants / modules en anglais ; UI souvent en français.
- Pas de secrets en dur ; prod via `APP_CONFIG_JSON` (voir `docs/caprover_env_example.md`).
