# Dataset Style Studio (Multi-tenant)

Application Streamlit de curation de datasets littéraires, en mode multi-utilisateur et multi-projet.

## Ce qui est implémenté

- Authentification via SuperTokens (`src/auth.py`) en mode invitation-only.
- Modèle `1 projet = 1 dataset`.
- Modèle d'accès simplifié:
  - `1 projet = 1 utilisateur propriétaire`
  - `1 utilisateur = N projets`
- Dimensions textuelles pilotées par presets:
  - `types` (Type de transformation)
  - `structures` (Structure textuelle)
  - `tons` (Tonalité textuelle)
  - `formats` (Format de sortie)
  - `publics` (Public cible)
  - `statuts`
- Paramétrage LLM + LanguageTool par projet (`project_settings`).
- Tableau de bord stylométrique : distribution des scores de cohérence, variance par axe sur fiches validées, outliers, moyenne du contraste syntaxique (`src/ui_components.py`, `src/nlp_engine.py`).
- Persistance PostgreSQL multi-tenant (`src/database.py`).

## Schéma PostgreSQL

Tables principales:

- `users`
- `projects`
- `project_settings`
- `entries` (avec `project_id`)

Les presets par défaut et la logique de sérialisation sont dans `src/presets.py`.

Le schéma est créé automatiquement au démarrage via `ensure_schema()`.

## Variables d'environnement

### Source unique

Le projet centralise la configuration runtime:

- en **dev**: fichier `.env`
- en **prod CapRover**: variable unique `APP_CONFIG_JSON`

Le chargeur (`src/config.py`) applique cet ordre:
1. variables déjà présentes dans l'environnement
2. `APP_CONFIG_JSON` (JSON)
3. `.env`
4. dérivation automatique de `DATABASE_URL` et `SUPERTOKENS_CONNECTION_URI`

### Variables auth/comptes

- `SUPER_ADMIN_EMAILS`: liste d'emails (séparés par virgules) promus au premier login si email vérifié provider.
- `AUTH_ENFORCE_INVITATION_ONLY`: si `true`, l'app vérifie contractuellement que `signup` provider est bloqué.
- `SUPERTOKENS_SIGNUP_DISABLED`: source de vérité non-destructive attendue quand `AUTH_ENFORCE_INVITATION_ONLY=true`.
- `SUPERTOKENS_CORE_API_KEY`: clé API côté SuperTokens Core (à aligner avec `SUPERTOKENS_API_KEY` côté app).
- `APP_PUBLIC_BASE_URL`: URL publique servant à construire les liens invitation/reset.
- `ACCOUNT_SAGA_MAX_RETRIES`: nombre maximum de retries pour les opérations de révocation/suppression compte.
- `ACCOUNT_RETRY_BATCH_SIZE`: taille de lot du worker de reprise des opérations en échec.

### Variables email

- `MAIL_MODE`: `dev` (affiche un lien masqué en UI) ou `smtp`.
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`: configuration SMTP en mode `smtp`.

## Dev local (une commande)

```bash
cp .env.example .env
make dev
```

`make dev` lance le stack complet via `compose.yaml`:
- `postgres`
- `supertokens`
- `app` (Streamlit)

Commandes utiles:

```bash
make dev-build
make rebuild
make logs
make down
```

- `make dev`: démarrage rapide (sans rebuild forcé)
- `make dev-build`: rebuild + démarrage
- `make rebuild`: rebuild image app seul
- `postgres` et `supertokens` n'exposent plus de ports hôte par défaut (évite les conflits avec services locaux)
- En cas de conflit de ports hôtes, adapter `APP_PORT` dans `.env`
- Par défaut, `SUPERTOKENS_DB=dataset_style` (base partagée) pour éviter un bootstrap SQL supplémentaire

### Tests

```bash
pytest -q
```

Les tests couvrent : auth (contrats sécurité, saga), database (SQLite en mémoire), export_utils, nlp_engine.

## Déploiement CapRover (une commande)

Guide pas à pas (variables, ordre des apps, health checks, rollback) : **`docs/caprover_deployment.md`**.

1. Définir une seule variable `APP_CONFIG_JSON` dans CapRover (voir `docs/caprover_env_example.md`).
2. Déployer:

```bash
make prod
```

Cette commande exécute `caprover deploy`.

- Les URL internes entre apps doivent utiliser `srv-captain--<app-name>`.
- Le workflow CI est défini dans `.github/workflows/ci.yml` (ruff).

### Vérification pré-déploiement

```bash
uv run python scripts/bootstrap_check.py
```

Option `--apply-schema` pour exécuter `ensure_schema()` après un ping DB (voir le script).

### Stack complet local (alternative à `compose.yaml`)

```bash
docker compose -f docker-compose.full.yml --env-file .env up --build
```

Ce fichier ajoute un healthcheck HTTP sur SuperTokens et attend que Postgres **et** SuperTokens soient sains avant de démarrer l’app.

## Règles de permissions

- Le propriétaire du projet a tous les droits sur son projet.
- Aucun partage multi-membres n'est actif.
- Le super admin global peut inviter/supprimer des comptes depuis l'onglet dédié.
- Les actions super admin sont validées côté backend (pas uniquement via visibilité UI).

Les garde-fous sont appliqués dans `src/database.py` (`require_role`, `require_admin`), pas uniquement dans l'UI.

## Organisation UI

- Sidebar minimale:
  - compte
  - projet courant
  - rôle
  - déconnexion
- Onglets centraux (ordre métier : projet → réglages → saisie → révision → tableau de bord) :
  - `Projets`
  - `Réglages & Export`
  - `Nouvelle entrée`
  - `Gestion & édition`
  - `Tableau de bord`
  - `Mon compte` (et `Super Admin` si rôle adapté)

Dans **Export** (même onglet), un contrôle unique définit le périmètre pour **CSV et JSONL** :
« Validées seulement » (statut « Fait et validé ») ou « Tout le dataset » (tous les statuts, y compris brouillons).

Dans `Réglages & Export`, section **Dimensions du texte**:
- choix de preset (`roman`, `pro`, `contenu`)
- bouton explicite `Charger le preset` pour appliquer le preset sélectionné au projet
- édition des listes par dimension (ajout/retrait)
- enregistrement comme preset personnalisé
- réinitialisation depuis le preset sélectionné
- message de portée: ces dimensions s'appliquent au projet courant uniquement
- en édition, une valeur hors preset est conservée via l'option `[obsolète] <valeur>` (jamais écrasée silencieusement)

Dans les formulaires:
- `Input` est renommé `Brouillon`
- `Output` est renommé `Texte généré`
- `Corriger output` (LanguageTool) met à jour directement le champ « Texte généré » pour permettre sauvegarde ou validation sans copier depuis un encart
- les actions de génération affichent un spinner `Génération en cours...`

Toutes les actions sensibles (suppression projet, réglages) sont validées côté backend via les fonctions `*_as_admin`, même si l'UI est contournée.

## Parcours comptes

- Connexion: email + mot de passe (pas de signup public UI).
- Mot de passe oublié: génération d'un lien de reset depuis l'écran de connexion.
- Invitation: un super admin crée un compte via email; l'utilisateur définit ensuite son mot de passe.
- Suppression de compte:
  - utilisateur: refusée si projets owner ou memberships actives
  - super admin: procédure de detach memberships puis suppression via saga idempotente.

## Résilience saga comptes

- Les opérations de suppression/révocation peuvent passer en quarantaine (DLQ) après `ACCOUNT_SAGA_MAX_RETRIES`.
- Replay admin disponible dans l'onglet **Super Admin**.
- Worker planifié disponible: `python scripts/retry_deprovision_ops.py`.

## Documentation architecture

Voir `docs/multi_tenant_architecture.md`.

## Merge-ready & incidents

- Checklist de validation: `docs/merge_ready_checklist.md`
- Runbook incident comptes: `docs/incident_accounts_runbook.md`

