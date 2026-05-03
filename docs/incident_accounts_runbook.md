# Runbook Incident Comptes

## Déclencheurs

- Erreurs répétées de suppression/révocation compte.
- Hausse d'opérations `quarantined`.
- Invitation/reset email non distribués.

## Procédure courte (DLQ / replay / rollback opérationnel)

1. Ouvrir l'onglet `Super Admin` > sections `Monitoring saga comptes` et `DLQ`.
2. Identifier l'opération en échec (`operation_id`, `target_user_id`, `last_error`, `retry_count`).
3. Vérifier s'il existe une opération active concurrente pour la même cible.
4. Corriger le blocant:
   - memberships résiduelles: utiliser `Detach memberships` avec confirmation.
   - provider indisponible: attendre rétablissement + relancer worker.
5. Lancer `Replay opération` depuis la DLQ.
6. Si besoin, exécuter le worker: `python scripts/retry_deprovision_ops.py`.
7. Vérifier résultat:
   - état final `completed`
   - plus d'accès côté app pour le compte ciblé.

## Rollback opérationnel (sans rollback DB technique)

- Si suppression/revocation appliquée trop tôt:
  - arrêter les retries automatiques,
  - corriger les droits métiers (memberships/projets),
  - rejouer uniquement les opérations validées.
- Ne pas réactiver manuellement des comptes supprimés sans validation sécurité.

## Délivrabilité email (SMTP)

1. Vérifier `MAIL_MODE=smtp` et les variables SMTP.
2. Tester invitation + reset sur une boîte de test réelle.
3. Vérifier SPF/DKIM/DMARC du domaine:
   - SPF: enregistrement TXT valide pour l'émetteur.
   - DKIM: signature active et validée.
   - DMARC: policy alignée (`p=quarantine` ou `p=reject` selon stratégie).

## Escalade

- Incident persistant > 30 min ou DLQ en croissance continue:
  - escalader DevOps/SRE,
  - geler les suppressions comptes non urgentes,
  - conserver les `operation_id` impactés pour analyse.
