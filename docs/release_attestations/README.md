# Attestations de release — jalon 009 (chaîne curation PostgreSQL)

Ce répertoire regroupe les **modèles** d’attestation produit pour le jalon **issue-009** (non-régression persistance + export sur PostgreSQL), alignés sur `docs/migration_parity_matrix.md` et `docs/curation_chain_preprod_regression_issue_009.md`.

## Artefacts CI

Sur chaque exécution réussie du workflow **CI** (`.github/workflows/ci.yml`), l’artefact **`ci-jalon-009-handoff-bundle`** contient :

- `junit-report.xml` — rapport JUnit de la suite `pytest -q` (inclut les tests marqueur `postgres_regression` lorsque `DATASET_STYLE_REGRESSION_DATABASE_URL` est défini dans le job).
- `AUTOMATED_GATE_ISSUE_009.md` — synthèse machine (SHA, lien vers le run GitHub Actions, résultat pytest). **Ne couvre pas** l’auth navigateur SuperTokens.
- `ATTESTATION_PRODUIT.template.md` — copie du modèle ci-dessous à compléter par le release manager après recette préprod / parcours critique.

## Fichiers versionnés

| Fichier | Rôle |
|---------|------|
| `TEMPLATE_jalon_009_curation_chain_preprod.md` | Modèle d’attestation signée (nom, date, réserves, lien matrice). |

Pour une promotion documentée, une copie **remplie** peut être archivée dans ce dossier (ex. `ATTESTATION_jalon_009_preprod_YYYY-MM-DD.md`) après validation DBA/ops sur l’usage de `ensure_schema` sur la base cible.
