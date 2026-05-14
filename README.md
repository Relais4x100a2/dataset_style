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
- Paramétrage de la génération assistée par IA et de LanguageTool par projet (`project_settings`, libellés métier dans l’onglet Réglages).
- Tableau de bord stylométrique : distribution des scores de cohérence avec synthèse numérique (moyenne, médiane, minimum) sur le même périmètre et le même parseur que l’export ; au-delà de 25 000 entrées dans le périmètre, échantillon aléatoire documenté dans l’UI pour l’histogramme et la synthèse. Variance par axe sur fiches validées (union des axes du cache ; écart-type seulement si l’axe a au moins deux valeurs sur des fiches distinctes), outliers, moyenne du contraste syntaxique (`src/ui_components.py`, `src/nlp_engine.py`).
- Persistance PostgreSQL multi-tenant (`src/database.py`).
- Accueil guidé (étapes + formulaire de création dans la zone principale) lorsque l’utilisateur n’a aucun projet, pour les petits écrans et la barre latérale repliée (`src/ui_components.py`, `src/empty_project_onboarding.py`, `src/project_session.py`).
- Mise en cache des lignes d’entrées du projet : `cached_load_project_entries` (`@st.cache_data`, TTL 30 s, clé incluant `project_id` + `user_id` stable) pour accélérer le passage d’un onglet à l’autre ; invalidation explicite via `invalidate_project_entries_cache` après chaque écriture couverte (entrées et cycle de vie projet dans l’UI — issue-027), **avant** tout `st.rerun()` qui doit afficher des données à jour (le TTL seul ne suffit pas).

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

### Variables projets / onboarding

- `DISABLE_SELF_SERVICE_PROJECT_CREATION` : si `1`, `true`, `yes` ou `on`, masque les formulaires de création de premier projet (parcours invitation / admin uniquement ; message utilisateur dans la zone principale et la barre latérale).

### Variables email

- `MAIL_MODE`: `dev` (affiche un lien masqué en UI) ou `smtp`.
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`: configuration SMTP en mode `smtp`.
- Texte d’introduction des invitations super-admin : `invitation_account_link_email_intro_fr()` dans `src/empty_project_onboarding.py` (aligné sur la phrase produit stylométrique de l’onboarding ; envoi via `src/mailer.py`, pas via les templates email versionnés de SuperTokens Core).

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

Les tests couvrent : auth (contrats sécurité, saga), database (SQLite en mémoire), export_utils, récap export (`tests/test_export_quality_recap_service.py`), nlp_engine, agrégats du tableau de bord (`tests/test_dashboard_metrics.py`), libellés d'alertes corpus (`tests/test_corpus_stylometry_alerts_fr.py`), onboarding sans projet (`tests/test_empty_project_onboarding.py`).

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
  - `Tableau de bord` : variance par axe (validées), distribution des scores de cohérence, outliers (top N), alerte **Paire quasi identique** (`_syntax_contrast` sous seuil strict défini dans le code), glossaire FR dans un expander partagé avec l'édition et le retour post-sauvegarde.
  - `Mon compte` (et `Super Admin` si rôle adapté)

Dans **Gestion & édition**, la liste déroulante des fiches peut être réduite par **filtre statut**
(preset + valeurs réellement présentes dans le projet, y compris legacy) et par **filtre score de
cohérence** (`_coherence_score`) : aucun filtre, seuil strict « sous X », tranche de 10 points
(alignée sur le tableau de bord via les mêmes bornes que l’histogramme), ou **score non calculé
uniquement** (N/A). Les entrées sans score exploitable peuvent aussi être incluses ou exclues
explicitement lorsque le filtre score est un seuil ou une tranche. Un libellé indique en permanence
combien d’entrées sont affichées par rapport au total du projet. Des boutons **Précédent** /
**Suivant** et le sélecteur « Entrée » partagent le même ordre que la liste filtrée (tri stable sur
l’identifiant) ; un dialogue de confirmation apparaît avant tout changement de fiche pour rappeler
le risque de perte des modifications non sauvegardées du formulaire (comportement aligné sur les
contraintes Streamlit `st.form`).

Dans **Export** (même onglet), un contrôle unique définit le périmètre pour **CSV et JSONL** :
« Validées seulement » (statut « Fait et validé ») ou « Tout le dataset » (tous les statuts, y compris brouillons).
Un **récap qualité** (fiches exportées, nombre validées dans ce périmètre, moyenne de cohérence, comptage des scores bas selon un seuil produit documenté) s’affiche au-dessus des boutons de téléchargement ; une alerte apparaît si la moyenne de cohérence du périmètre est strictement sous le seuil produit (`EXPORT_PERIMETER_COHERENCE_MEAN_ALERT_LT` dans `src/nlp_engine.py`).

Dans `Réglages & Export`, section **Dimensions du texte**:
- choix d'un **profil de dimensions** (jeu de listes prédéfini, clés techniques `roman`, `pro`, `contenu`, etc.)
- bouton **Charger ce profil** pour appliquer le profil sélectionné au projet
- édition des listes par dimension (ajout/retrait)
- enregistrement d'un **profil personnalisé** (identifiant technique + libellé affiché)
- réinitialisation depuis le profil sélectionné
- message de portée: ces dimensions s'appliquent au projet courant uniquement
- en édition, une valeur hors profil actif est conservée via l'option `[obsolète] <valeur>` (jamais écrasée silencieusement)

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

