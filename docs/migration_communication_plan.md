# Plan de communication — continuité de service pendant la migration

**Stories** : issue interne 021 · GitHub [#143](https://github.com/Relais4x100a2/dataset_style/issues/143)  
**Prérequis produit** : décision de bascule — `docs/streamlit_to_new_frontend_cutover.md` (issue-001 / #124) — **cutover unique** en production ; **pas** de double interface officielle pérenne sur la même URL métier ; support sur l’URL alignée avec **`APP_PUBLIC_BASE_URL`**.

Ce document fournit le **message interne** prêt à l’emploi, un **extrait de runbook** pour la documentation d’exploitation, la **référence exports stables**, et la procédure pour activer une **bannière optionnelle** pilotée par configuration (`APP_MIGRATION_INFO_BANNER`).

---

## 1. Distinction production vs préprod (obligatoire dans toute communication)

| Environnement | Surface « officielle » pour les curateurs | Technique |
|----------------|-------------------------------------------|-----------|
| **Production Relais4** | **Une seule** URL applicative à la fois pour le périmètre métier ; après cutover, le nouveau frontal remplace Streamlit sur cette URL. | Branche / train : `deploy-caprover-relais4` — voir `docs/release_train_caprover.md`. |
| **Préprod / recette** | URL **distincte** de la prod ; les testeurs valident les parcours sans imposer deux origines « officielles » en prod. | Rampe de type `deploy-newfrontend` + CapRover préprod ; double surface **technique** (ex. Streamlit + webapp sur des ports ou apps différents) acceptable **uniquement** hors prod utilisateur. |

**À ne pas écrire dans un mail interne** : une « coexistence prolongée en production » entre deux URL ou deux interfaces officielles pour les mêmes curateurs — cela contredirait les prérequis des stories **007–016** et compliquerait cookies, CORS et support.

**Formulation sûre** : en prod, **fenêtre de bascule courte** (éventuelle indisponibilité contrôlée) puis **une** interface ; en préprod, recette sur l’URL de staging.

---

## 2. Exports stables pendant la transition

Les formats d’export **ne sont pas** des conventions ad hoc : ils passent par **`src/export_utils.py`** (et les mêmes périmètres côté slice HTTP), avec une suite de tests dédiée.

- **CSV** : colonnes alignées sur la sémantique métier (pas de colonnes internes préfixées `_` côté API / export slice).
- **JSONL** : via `convert_to_jsonl` avec les formats fermés **`lfm2`**, **`baguettotron`**, **`mistral`** ; périmètres **`validated_only`** / **`full_dataset`** via `dataframe_for_export`.
- **Matrice de parité** et recette chaîne curation : `docs/migration_parity_matrix.md` (chemins **EXP-SCOPE**, **EXP-DL** ; tests `tests/test_export_utils.py`, `tests/test_export_quality_recap_service.py`).

En cas de doute support : demander un export **CSV + JSONL** depuis l’interface active **avant** une opération risquée, et conserver les fichiers comme preuve de périmètre.

---

## 3. Recette des parcours critiques (dev / prod)

### Développement / CI

Aligné sur `docs/migration_parity_matrix.md` et `docs/merge_ready_checklist.md` :

```bash
ruff check .
ruff format --check .
pytest -q
uv run python scripts/bootstrap_check.py
```

Pour la chaîne curation + export (jalon non-régression) :

```bash
python3 -m pytest tests/test_tab_layout.py tests/test_export_utils.py tests/test_project_entries_cache.py -q
python3 -m pytest tests/test_services.py tests/test_export_quality_recap_service.py -q
```

### Production (smoke post-déploiement ou post-cutover)

1. Ouvrir l’URL **unique** de prod (celle publiée aux curateurs, cohérente avec `APP_PUBLIC_BASE_URL`).
2. Connexion invitation-only (compte de test ou pilote).
3. Sélection / création projet, **une** édition ou création de fiche, sauvegarde.
4. Onglet réglages / export : changer le périmètre, télécharger **CSV** et **JSONL**, ouvrir les fichiers (cohérence statuts / colonnes attendues).
5. Vérifier qu’aucune erreur d’auth récurrente n’apparaît (cookies / domaine).

Rollback applicatif : `docs/streamlit_to_new_frontend_cutover.md` § critères de rollback + `docs/caprover_deployment.md`.

---

## 4. Bannière d'information optionnelle (Streamlit + webapp)

Variable unique **`APP_MIGRATION_INFO_BANNER`** (injectable via **`APP_CONFIG_JSON`**). Le service **`webapp`** appelle **`initialize_runtime_config()`** au début du lifespan FastAPI pour fusionner la même config que Streamlit avant de lire cette variable.

### Texte brut (historique #143)

- Chaîne seule : pas de HTML ; contenu **échappé** dans le HTML du `webapp` ; côté Streamlit, même gabarit sémantique (`ds-banner--info`, classe stable **`ds-migration-banner`**, `role="region"`).

### JSON structuré (#184)

Objet JSON UTF-8 ; champs :

| Champ | Obligatoire | Description |
|--------|-------------|-------------|
| `message` | oui | Texte court (FR recommandé), sans HTML. |
| `calendar_note` | non | Fenêtre ou calendrier communiqué (texte brut). |
| `help_url` | non | `http`, `https` ou `mailto` uniquement ; autres schémas ignorés. |
| `help_label` | non | Libellé du lien (défaut : « Où trouver l'aide »). |
| `support_url` | non | Idem schémas autorisés. |
| `support_label` | non | Libellé (défaut : « Contacter le support »). |

Exemple :

```json
{"message":"L'URL du studio change — vos exports restent identiques.","help_url":"https://intranet.example/docs/dataset-style","help_label":"Documentation interne","support_url":"mailto:support@example.org","support_label":"Support","calendar_note":"Bascule prévue : mardi 10 juin, 18h–19h (Paris)."}
```

**Désactivation** : variable absente ou vide après le cutover.

Voir aussi `docs/caprover_env_example.md` et `.env.example`.

---

## 5. E-mail interne (modèle)

**Objet** : `[Dataset Style] Migration interface — une URL prod, exports inchangés`

**Corps** (à adapter : dates, URL préprod, canal support) :

---

Bonjour,

Nous finalisons la migration du studio **Dataset Style** vers le nouveau frontal (FastAPI + coquille web), conformément à la décision documentée (*cutover unique* — voir `docs/streamlit_to_new_frontend_cutover.md`).

**Production**  
- Il n’y aura **qu’une seule URL « officielle »** pour le travail de curation en production, alignée sur **`APP_PUBLIC_BASE_URL`**.  
- La bascule pourra impliquer une **courte interruption** contrôlée ; nous communiquerons le créneau via [indiquer le canal interne].  
- **Pas** de période documentée où deux interfaces seraient toutes deux « officielles » sur la même prod pour le même usage.

**Préproduction**  
- Les tests se font sur **l’URL de préprod** ; merci de ne pas mélanger les liens d’invitation ou de reset e-mail entre préprod et prod.

**Exports**  
- Les téléchargements **CSV** et **JSONL** restent fondés sur la même logique métier (`export_utils`, périmètres validées / tout le dataset). Vous pouvez continuer à exporter vos jeux avant la fenêtre de bascule comme d’habitude.

**Support**  
- En cas de blocage après la bascule : [canal support]. Joindre si possible un export CSV/JSONL récent et l’heure approximative du problème.

Merci,  
[L’équipe]

---

## 6. Références croisées

- `docs/streamlit_to_new_frontend_cutover.md` — décision cutover, rollback, `APP_PUBLIC_BASE_URL`  
- `docs/migration_parity_matrix.md` — parité fonctionnelle et tests  
- `docs/release_train_caprover.md` — branches et merge  
- `docs/dev_new_frontend.md` — lancement local du service `webapp`
