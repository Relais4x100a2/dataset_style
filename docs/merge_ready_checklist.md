# Checklist Merge-Ready

## 0) Avant merge sur la branche de déploiement CapRover

Pour tout merge vers **`deploy-caprover-relais4`** (prod Relais4), suivre la procédure détaillée dans **`docs/release_train_caprover.md`** : `pytest -q`, `ruff check` / `ruff format --check`, `bootstrap_check`, contrats auth si la PR touche les chemins listés dans `auth-contract.yml`.

## 1) Sécurité auth validée

- [ ] `AUTH_ENFORCE_INVITATION_ONLY=true` en cible.
- [ ] `SUPERTOKENS_SIGNUP_DISABLED=true` aligné.
- [ ] `SUPERTOKENS_CORE_API_KEY` défini et aligné avec `SUPERTOKENS_API_KEY`.
- [ ] Workflow `auth-contract` vert:
  - provider joignable
  - `POST /recipe/signup` bloqué
  - gate api-key actif côté provider.

## 2) Saga deprovision opérationnelle

- [ ] États DB présents: `pending/provider_done/db_done/completed/failed/quarantined`.
- [ ] `ACCOUNT_SAGA_MAX_RETRIES` configuré en cible.
- [ ] Backoff/retry effectifs visibles dans `retry_count` et `next_retry_at`.
- [ ] Passage `quarantined` observé sur erreurs répétées (test contrôlé).

## 3) Exploitation DLQ prête

- [ ] Onglet Super Admin: DLQ visible.
- [ ] Replay manuel testé sur un cas réel.
- [ ] Conflit replay (op active existante) refusé proprement.
- [ ] Worker planifié actif: `python scripts/retry_deprovision_ops.py`.

## 4) Garde-fous UI/permissions vérifiés

- [ ] `require_super_admin` appliqué backend sur actions globales.
- [ ] `detach memberships` protégé (warning + checkbox + re-saisie email).
- [ ] Suppression compte bloquée tant que projets owner/memberships existent.

## 5) Délivrabilité et runbook minimum

- [ ] `MAIL_MODE=smtp` testé bout-en-bout (invitation + reset).
- [ ] SPF/DKIM/DMARC validés sur le domaine d’envoi.
- [ ] Runbook incident disponible et relu (`docs/incident_accounts_runbook.md`).

## 6) Recette coexistence dev/staging : une surface à la fois (obligatoire)

Pendant la coexistence **Streamlit** (port **8501**) et **webapp** FastAPI (port **8080** par défaut), la recette manuelle ou la collecte **baseline UX** (issue-020) doit respecter le protocole **« un scénario = une UI »** jusqu’à **export réussi** (`EXP-DL`) ou **fin explicite** du scénario (abandon noté, fin de session). Détail, exemples et cas d’exception : **`docs/migration_parity_matrix.md`** (section *Protocole recette : une surface à la fois*) ; jalons et télémétrie : **`docs/ux_baseline_issue_020.md`**.

- [ ] **Un scénario = une origine UI** : ne pas alterner Streamlit ↔ webapp au milieu d’un même parcours (liste → fiche → export) pour le même compte et le même projet, sauf **cas d’essai volontaire documenté** (nom, objectif, données figées — voir la matrice).
- [ ] **Mono-surface par `run_id`** : toute collecte baseline / télémétrie issue-020 reste attachée à **une seule** surface pour la durée du `run_id` ; pas de changement d’origine avant fin de scénario.
- [ ] **BDD partagée** : séparer **non-régression données** (PostgreSQL, API, tests automatisés) et **non-régression perception UI** (cache client, reruns Streamlit, fragments HTMX, session auth). Le protocole vise surtout cette seconde catégorie. Toute PR concluant à une régression **sans** respecter ce protocole doit être **requalifiée** avant ouverture de défaut backend.
