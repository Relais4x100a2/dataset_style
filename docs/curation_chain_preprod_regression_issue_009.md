# Jalon 009 — non-régression chaîne curation sur PostgreSQL (préprod)

Objectif : prouver la **persistance** et l’**export** (slice aligné matrice issue-004) sur une base PostgreSQL 16 **dédiée** à la régression, **distincte** de la production, avant promotion vers `deploy-caprover-relais4`.

Références : `docs/migration_parity_matrix.md`, `docs/release_train_caprover.md`, `tests/test_curation_chain_postgres_regression.py`.

## Portée du gate automatisé

Les tests du fichier `tests/test_curation_chain_postgres_regression.py` couvrent :

- création projet + entrée + édition + relecture cohérente (PRJ-CREATE, ENT-NEW-WRITE, EDI-SAVE) ;
- périmètres d’export `validated_only` / `full_dataset` (EXP-SCOPE) ;
- téléchargements CSV / JSONL cohérents avec `export_utils` et les routes slice (EXP-DL).

Ils **ne** remplacent **pas** une preuve **auth navigateur** (cookies SuperTokens, invitation-only). Compléter avec la checklist manuelle matrice et l’attestation produit (`docs/release_attestations/TEMPLATE_jalon_009_curation_chain_preprod.md`).

## Prérequis

1. **URL** : variable `DATASET_STYLE_REGRESSION_DATABASE_URL` au format SQLAlchemy / Psycopg attendu par `src/database.py:create_db_engine` (ex. `postgresql+psycopg://user:pass@host:5432/dbname`).  
2. **Isolement** : base ou schéma **préprod / régression** uniquement — ne jamais pointer vers la base prod.  
3. **Schéma** : les tests appellent `ensure_schema` au démarrage du module. **Valider avec ops/DBA** que l’application de ce schéma sur l’environnement cible est acceptée.

## Commande (préprod)

```bash
export DATASET_STYLE_REGRESSION_DATABASE_URL='postgresql+psycopg://…'
python3 -m pytest tests/test_curation_chain_postgres_regression.py -q -m postgres_regression
```

## CI

- **Branches** : `main`, `deploy-caprover-relais4`, `deploy-newfrontend` — voir `.github/workflows/ci.yml`.  
- Le job CI définit `DATASET_STYLE_REGRESSION_DATABASE_URL` vers un service **postgres:16** éphémère et exécute la suite complète `pytest -q`.  
- **Artefact** : `ci-jalon-009-handoff-bundle` (JUnit + `AUTOMATED_GATE_ISSUE_009.md` + modèle d’attestation produit).

## Suite CI sur la branche d’intégration migration

La branche `deploy-newfrontend` est incluse dans les événements `push` / `pull_request` de `ci.yml` pour que le commit candidat préprod bénéficie de la même barre qualité que le train CapRover Relais4.

## Attestation et lien « rapport »

Après go : remplir et archiver une copie du modèle sous `docs/release_attestations/` (voir `docs/release_attestations/README.md`). Le **lien de rapport** pour le commit déployé est l’URL du run GitHub Actions (indiquée dans `AUTOMATED_GATE_ISSUE_009.md` de l’artefact CI).
