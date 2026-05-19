# Décision produit / ops — Bascule Streamlit → nouveau frontal

Document de référence pour débloquer auth, domaines CapRover et le plan support.  
**Issue fondatrice** : stratégie de bascule (user story issue-001 / GitHub #124).

---

## 1. Décision retenue

| Sujet | Décision |
|-------|----------|
| **Mode en production** | **Cutover unique** : après la bascule, **une seule** interface applicative est servie sur l’URL de production ; pas de double interface officielle pérenne côté utilisateurs. |
| **Intégration avant prod** | Développement et recettes sur la branche **`deploy-newfrontend`** + **environnement CapRover préprod / staging** distinct de la production Relais4 (`deploy-caprover-relais4`). Voir `docs/release_train_caprover.md` (publication de la branche distante, CI, merge vers prod). |
| **Coexistence deux interfaces en prod** | **Non retenue.** Durée maximale d’une double interface **en production** : **0 jour** (bascule unique ; pas de période documentée où deux URL « officielles » coexisteraient pour le même usage métier). |
| **Surface canonique pour le support** | L’**URL publique unique** effectivement servie en production, **alignée sur `APP_PUBLIC_BASE_URL`** (liens d’invitation et de reset e‑mail, cohérence avec la configuration SuperTokens et les origines autorisées). Avant cutover : l’URL Streamlit actuelle en prod ; après cutover : l’URL du nouveau frontal telle qu’arrêtée en release. |

En l’absence de sponsor pour trancher autrement, la **préconisation architecture** appliquée est : préprod sur la rampe de migration, **une URL en prod** après bascule.

---

## 2. Cohérence `APP_PUBLIC_BASE_URL`, e‑mails et SuperTokens

Lors du **switch production** :

- `APP_PUBLIC_BASE_URL`, le domaine exposé par CapRover et les réglages SuperTokens (origines, callbacks, cookies) doivent être **strictement cohérents** pour éviter des liens d’invitation / reset invalides ou des sessions incohérentes.
- Tant que cette cohérence n’est pas validée (recette bout-en-bout), ne pas basculer le trafic utilisateur vers le nouveau frontal.

---

## 3. Impacts utilisateurs internes

| Phase | Impact |
|-------|--------|
| **Production — cutover unique** | **Interruption courte** possible (indisponibilité ou page de maintenance) pendant redeploy / bascule de l’app et contrôles smoke. Pas de **double URL officielle** en prod pour le même périmètre métier. |
| **Préprod** | Les testeurs utilisent l’URL de **préprod** ; les parcours mail de test doivent utiliser une base URL de préprod si on valide les liens, afin de ne pas polluer la prod. |

**Fenêtre calendaire indicative** : à planifier avec le sponsor (créneau typiquement **hors charge métier**, par exemple **soir ou week-end**). Durée indicative **de l’ordre de 1 à 4 heures** pour redeploy + vérifications minimales — **non contractuelle** ; l’équipe ops ajuste selon la complexité réelle et les migrations.

---

## 4. Critères de rollback (niveau usage)

1. **Rollback nominal** : **redeploy** sur CapRover de l’**image Streamlit** (ou de l’artefact précédent) pour l’application production concernée, conformément au runbook de déploiement (`docs/caprover_deployment.md`).
2. **Exports de secours** : s’appuyer sur les exports / procédures déjà prévus dans les runbooks applicables avant toute opération risquée.
3. **PostgreSQL** : si des **migrations destructives** ont été appliquées en prod, le retour arrière « données » **n’est pas** garanti par un simple redeploy seul ; il requiert une **sauvegarde** préalable et un **plan de restauration PG** explicite, validé avec l’exploitation.
4. Après tout rollback, **`APP_PUBLIC_BASE_URL`** et l’URL réellement servie doivent rester **alignées**.

---

## 5. Coexistence (risque et positionnement)

La **coexistence en production** (deux origines / deux interfaces pour les mêmes utilisateurs) **n’est pas le mode retenu**. Si un sponsor demandait ultérieurement une fenêtre de coexistence en prod, le risque principal est la **double origine** (sessions, cookies, CORS). Il faudrait alors une **décision écrite** complémentaire fixant une **durée maximale** et une **recette auth dédiée** avant exposition — ce document reste la base jusqu’à cette nouvelle décision.

---

## 6. Prérequis pour les stories d’implémentation nouveau frontal (007–016)

Les user stories **007 à 016** (implémentation du nouveau frontal, côté dépôt / GitHub) doivent citer **explicitement** ce document en prérequis, avec au minimum :

- **Référence** : `docs/streamlit_to_new_frontend_cutover.md`
- **Mode prod** : cutover unique
- **Durée max coexistence en prod** : 0 jour (non retenue)
- **Surface canonique support** : URL prod alignée sur `APP_PUBLIC_BASE_URL`

**Formulation type (à copier dans chaque story 007–016)** :

> **Prérequis** : décision de bascule — `docs/streamlit_to_new_frontend_cutover.md` — mode prod **cutover unique** ; coexistence deux interfaces en prod **non** (durée max **0 jour**) ; support sur l’URL canonique = **`APP_PUBLIC_BASE_URL`** en phase correspondante.

---

## 7. Références

- Train de release et branche prod : `docs/release_train_caprover.md`
- Déploiement CapRover : `docs/caprover_deployment.md`
- Variables d’environnement : `docs/caprover_env_example.md`
- Communication migration (e-mail type, exports, bannière optionnelle) : `docs/migration_communication_plan.md`

## 8. Artefacts mesure UX (issue-020 / #142)

Pour la revue de bascule (issue-001), archiver avec le paquet décisionnel :

- le répertoire ou une copie des fichiers sous **`DATASET_STYLE_UX_TELEMETRY_DIR`** (`ux_scenario_*.jsonl`, `ux_error_*.jsonl`) ;
- la sortie TSV de **`scripts/aggregate_ux_baseline_jsonl.py`** (voir `docs/ux_baseline_issue_020.md`) ;
- les réponses au questionnaire interne (`docs/ux_baseline_questionnaire.md`) si la campagne l’a prévu.

Les métriques ne sont **pas** stockées dans les tables métier PostgreSQL en v1.
