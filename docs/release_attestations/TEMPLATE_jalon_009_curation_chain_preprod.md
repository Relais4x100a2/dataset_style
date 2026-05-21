# Attestation produit — jalon 009 (chaîne curation PostgreSQL)

**Environnement** : préprod / staging (base ou schéma **dédié**, jamais la base production).  
**PostgreSQL** : 16 (aligné CI / CapRover).  
**Référence matrice** : `docs/migration_parity_matrix.md` (flux PRJ-CREATE, ENT-NEW-WRITE, EDI-SAVE, EXP-SCOPE, EXP-DL).

## Synthèse

- **Verdict** : acceptable sans réserve / acceptable avec réserve(s) / non acceptable  
- **Date** : YYYY-MM-DD  
- **Signataire (nom, rôle)** :  

## Réserves (si applicable)

Lister chaque réserve avec lien vers la ligne ou la section de la matrice concernée, ou vers un ticket de suivi.

1.  

## Preuves automatisées (persistance + export)

- [ ] `pytest tests/test_curation_chain_postgres_regression.py` exécuté avec `DATASET_STYLE_REGRESSION_DATABASE_URL` pointant vers la base préprod **approuvée** — résultat **vert** (voir aussi le marqueur `postgres_regression` dans `pyproject.toml`).
- [ ] Rapport d’exécution CI : artefact **`ci-jalon-009-handoff-bundle`** du workflow **CI** sur le commit déployé (JUnit + `AUTOMATED_GATE_ISSUE_009.md`).

## Parcours critique (hors seul gate Postgres)

Le gate automatisé ci-dessus **ne** prouve **pas** seul la chaîne **auth navigateur → session → UI**. Cocher après recette manuelle ou autre preuve :

- [ ] Auth / session (SuperTokens, invitation-only) conforme à la matrice pour le périmètre attendu.  
- [ ] Checklist recette minimale : section *Checklist recette minimale (manuelle)* dans `docs/migration_parity_matrix.md`.  
- [ ] Protocole *une surface à la fois* respecté pour les scénarios UI : `docs/merge_ready_checklist.md` § 6.

## Validation DBA / ops

- [ ] Exécution des tests avec `ensure_schema` sur la base préprod **explicitement acceptée** (pas de migration implicite non validée).
