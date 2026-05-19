# Checklist Merge-Ready

## 0) Avant merge sur la branche de déploiement CapRover

Pour tout merge vers **`deploy-caprover-relais4`** (prod Relais4), suivre la procédure détaillée dans **`docs/release_train_caprover.md`** : `pytest -q`, `ruff check` / `ruff format --check`, `bootstrap_check`, contrats auth si la PR touche les chemins listés dans `auth-contract.yml`.

Pour toute PR qui modifie un flux listé dans **`docs/migration_parity_matrix.md`** (routes webapp, tests de parité, ou comportement Streamlit équivalent), mettre à jour la **grille issue-010 → issue-016** et les sections `§` associées dans ce même fichier, puis exécuter `pytest -q tests/test_migration_parity_matrix_doc.py`.

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

## 6) Communication migration (cutover frontal)

- [ ] Message interne / support aligné sur `docs/streamlit_to_new_frontend_cutover.md` (une URL prod, pas de double interface officielle pérenne).
- [ ] Modèle e-mail et recette exports : `docs/migration_communication_plan.md`.
- [ ] Bannière optionnelle : si utilisée, `APP_MIGRATION_INFO_BANNER` retirée ou vidée après cutover (`docs/caprover_env_example.md`).
