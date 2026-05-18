# Déploiement CapRover — Guide pas-à-pas

> Stack : App Streamlit + SuperTokens + PostgreSQL sur CapRover (Docker Swarm, Hetzner)

---

## Prérequis

- Serveur CapRover opérationnel (CapRover ≥ 1.12)
- CLI `caprover` installée localement (`npm install -g caprover`)
- Accès au dashboard CapRover (`https://captain.<domaine>`)
- Domaine DNS configuré pour les sous-domaines applicatifs

---

## 1. Ordre de création des applications

Les dépendances imposent l'ordre suivant :

```
1. postgresql        ← base de données partagée
2. supertokens       ← dépend de PostgreSQL
3. dataset-style     ← dépend de PostgreSQL + SuperTokens
```

> **Important** : créer `postgresql` en premier et vérifier son healthcheck
> avant de démarrer `supertokens`.

---

## 2. Création de l'app PostgreSQL

### 2.1 Nouvelle app CapRover

- Nom : `postgresql`
- Type : **One-click app** → choisir `PostgreSQL`
- Ou déployer manuellement via l'image `postgres:16`

### 2.2 Variables d'environnement

| Variable            | Valeur exemple       | Obligatoire |
|---------------------|----------------------|-------------|
| `POSTGRES_DB`       | `dataset_style`      | ✓           |
| `POSTGRES_USER`     | `dataset_user`       | ✓           |
| `POSTGRES_PASSWORD` | *(secret fort)*      | ✓           |

### 2.3 Volume persistant

Dans l'onglet **App Configs** → **Persistent Directories** :

```
/var/lib/postgresql/data  →  /captain/data/postgresql
```

### 2.4 Vérification

```bash
# Depuis un container de test dans CapRover
psql "postgresql://dataset_user:<password>@srv-captain--postgresql:5432/dataset_style" -c "\l"
```

---

## 3. Création de l'app SuperTokens

### 3.1 Nouvelle app CapRover

- Nom : `supertokens`
- Image Docker : `registry.supertokens.io/supertokens/supertokens-postgresql`
- Port interne : `3567`

### 3.2 Variables d'environnement

| Variable                      | Valeur                                                                                              | Obligatoire |
|-------------------------------|-----------------------------------------------------------------------------------------------------|-------------|
| `POSTGRESQL_CONNECTION_URI`   | `postgresql://dataset_user:<password>@srv-captain--postgresql:5432/dataset_style`                 | ✓           |
| `API_KEYS`                    | *(clé forte, ≥ 20 caractères)*                                                                     | ✓           |

> **Erreur fréquente** : ne pas utiliser le préfixe `postgresql+psycopg://` ici —
> SuperTokens utilise son propre driver JDBC, pas SQLAlchemy.
> Le format doit être `postgresql://` standard.

### 3.3 Vérification du healthcheck

```bash
curl http://srv-captain--supertokens:3567/hello
# Attendu : {"status":"OK"}
```

---

## 4. Création de l'app principale (dataset-style)

### 4.1 Nouvelle app CapRover

- Nom : `dataset-style`
- Déploiement via `captain-definition` + Dockerfile (ou `caprover deploy`)
- Port interne exposé : `8501`
- Activer **HTTPS** (Let's Encrypt automatique dans CapRover)

### 4.2 Variables d'environnement obligatoires

Deux méthodes possibles :

**Méthode A — Variable par variable (recommandée pour la lisibilité)**

| Variable                      | Valeur                                                                                              | Obligatoire |
|-------------------------------|-----------------------------------------------------------------------------------------------------|-------------|
| `DATABASE_URL`                | `postgresql+psycopg://dataset_user:<password>@srv-captain--postgresql:5432/dataset_style`         | ✓           |
| `SUPERTOKENS_CONNECTION_URI`  | `http://srv-captain--supertokens:3567`                                                             | ✓           |
| `SUPERTOKENS_API_KEY`         | *(même valeur que `API_KEYS` dans SuperTokens)*                                                    | ✓           |
| `AUTH_ENFORCE_INVITATION_ONLY`| `true`                                                                                              | ✓           |
| `SUPERTOKENS_SIGNUP_DISABLED` | `true`                                                                                              | recommandé  |

> **Erreur fréquente — URI PostgreSQL** : l'app utilise SQLAlchemy + psycopg v3.
> Le préfixe **doit être** `postgresql+psycopg://` (pas `postgresql://` seul).
> CapRover peut encoder les caractères spéciaux dans les mots de passe —
> vérifier que `%` est bien encodé en `%25` si nécessaire.

**Méthode B — Variable unique `APP_CONFIG_JSON`**

Injecter un seul objet JSON dans `APP_CONFIG_JSON` (voir `docs/caprover_env_example.md`).

### 4.3 Variables optionnelles

| Variable                  | Valeur exemple                       | Usage                     |
|---------------------------|--------------------------------------|---------------------------|
| `LLM_BASE_URL`            | `http://srv-captain--ollama:11434`   | LLM local (Ollama)        |
| `LLM_MODEL`               | `mistral`                            | Modèle LLM par défaut     |
| `LLM_API_KEY`             | *(vide si local)*                    | Clé API LLM externe       |
| `LLM_TIMEOUT_SECONDS`     | `300`                                | Timeout requêtes LLM      |
| `LANGUAGETOOL_BASE_URL`   | `http://srv-captain--languagetool:8010` | Service correction grammaticale |

### 4.4 Configuration Nginx (port 80 vers 8501)

Dans **App Configs** → **Nginx Configurations** de l'app `dataset-style` :

```nginx
proxy_pass http://localhost:8501;
```

CapRover gère le reverse proxy automatiquement si le port interne est déclaré à `8501`.

### 4.5 Deuxième application FastAPI (préprod / recette, même image)

En **développement**, `compose.yaml` lance deux services à partir du **même** `Dockerfile` :

- `app` : Streamlit, port conteneur **8501** (commande par défaut de l’image).
- `webapp` : FastAPI (`uvicorn src.webapp.app:app`), port conteneur **8080**.

Pour une **recette CapRover** (typiquement préprod) où Streamlit et le slice FastAPI doivent coexister **sans** fusionner les processus dans un seul conteneur :

1. Créer une **seconde app** CapRover (ex. `dataset-style-web`) avec la **même image** que `dataset-style`.
2. Définir le **port interne** exposé par CapRover sur **8080** (aligné sur le service `webapp` du compose).
3. Renseigner la **commande de démarrage** (équivalent compose) :
   ` /app/.venv/bin/uvicorn src.webapp.app:app --host 0.0.0.0 --port 8080 `
4. Répliquer les variables d’environnement obligatoires (`DATABASE_URL`, `SUPERTOKENS_*`, `AUTH_ENFORCE_INVITATION_ONLY`, etc.) comme pour Streamlit.
5. **Healthcheck HTTP** côté CapRover : sonder `GET /health` sur le port interne **8080** (réponse JSON `{"status":"ok"}` ; pas d’authentification). Symétrie avec la sonde Streamlit sur `/_stcore/health` (port **8501**).

> **Production Relais4** : la cible produit reste un **cutover unique** (une interface officielle), voir `docs/streamlit_to_new_frontend_cutover.md`. La double app ci-dessus sert surtout la **préprod** ou des fenêtres de recette explicitement cadrées.

**CORS et URL canonique** : si le frontal et le BFF ne partagent pas la même origine, appliquer l’ADR `docs/adr/0006-front-stack-bff-spa-vs-htmx.md` (`WEBAPP_CORS_ORIGINS` liste fermée, `APP_PUBLIC_BASE_URL` aligné sur l’hôte réellement servi).

---

## 5. Déploiement

### 5.1 Via CLI `caprover deploy`

```bash
# Depuis la racine du projet
caprover login
caprover deploy --appName dataset-style --branch deploy-caprover-relais4
```

### 5.2 Via GitHub Actions (CI/CD)

Le workflow `.github/workflows/ci.yml` exécute lint et tests sur **push** et **pull_request**
vers `main` et `deploy-caprover-relais4`. Un déploiement automatique CapRover via GitHub Actions
n’est documenté ici que si l’équipe ajoute un job dédié et les secrets requis ; le chemin nominal
reste `make prod` / `caprover deploy` depuis la branche de déploiement (voir `docs/release_train_caprover.md`).

**Secrets GitHub requis :**

| Secret                  | Usage                                      |
|-------------------------|--------------------------------------------|
| `CAPROVER_URL`          | URL du dashboard CapRover (avec https)     |
| `CAPROVER_PASSWORD`     | Mot de passe admin CapRover                |
| `CAPROVER_APP_NAME`     | Nom de l'app à déployer                   |
| `CAPROVER_APP_TOKEN`    | Token d'app CapRover (optionnel, plus sécurisé) |

### 5.3 Via Makefile

```bash
make prod   # déploiement production
```

---

## 6. Bootstrap du premier administrateur

Après le premier déploiement, aucun utilisateur n'est super admin.

### 6.1 Créer le compte via l'interface

1. Naviguer sur l'URL de l'app
2. S'inscrire (si `SUPERTOKENS_SIGNUP_DISABLED=false`) ou utiliser le script

### 6.2 Promouvoir en super admin via SQL

```sql
-- Connexion directe à PostgreSQL
UPDATE users
SET is_super_admin = TRUE
WHERE lower(email) = lower('admin@example.com');
```

> **Erreur fréquente** : si la table `users` n'existe pas encore, `ensure_schema()`
> s'exécute au premier accès à l'app. Attendre que l'app soit démarrée et
> qu'un utilisateur se connecte une première fois avant d'exécuter ce SQL.

### 6.3 Vérification avec bootstrap_check.py

```bash
python scripts/bootstrap_check.py
```

---

## 7. Healthchecks et monitoring

### 7.1 Healthcheck PostgreSQL

CapRover (si déployé via One-click) ou via Docker :

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U dataset_user -d dataset_style"]
  interval: 10s
  timeout: 5s
  retries: 5
```

### 7.2 Healthcheck SuperTokens

```bash
GET http://srv-captain--supertokens:3567/hello
# → {"status":"OK"}
```

### 7.3 Healthcheck App Streamlit

Sonde HTTP interne (exemple dans `compose.yaml`) : `GET http://127.0.0.1:8501/_stcore/health`
(le corps contient la sous-chaîne `ok` en cas de succès).

CapRover surveille le port `8501`. Si l'app crashe au démarrage (erreur de config),
consulter les logs :

```bash
caprover api --method GET --path /api/v2/user/apps/appData?appName=dataset-style
# Ou depuis le dashboard : Apps → dataset-style → App Logs
```

### 7.4 Healthcheck App FastAPI (`webapp`)

Sonde dédiée : **`GET /health`** sur le port interne **8080**, réponse attendue `{"status":"ok"}` (sans en-tête `Authorization`).

Configurer la même URL dans CapRover pour l’app déployant `uvicorn` (voir §4.5), afin d’aligner le comportement avec `compose.yaml`.

---

## 8. Rollback

### 8.1 Rollback via CapRover

Dans le dashboard : **Apps → dataset-style → Deployment History** → cliquer sur
une version antérieure → **Redeploy**.

### 8.2 Rollback via CLI

```bash
# Lister les versions disponibles
caprover api --method GET --path "/api/v2/user/apps/appData?appName=dataset-style"

# Redéployer une image précédente (tag Docker)
caprover api --method POST --path /api/v2/user/apps/update \
  --data '{"appName":"dataset-style","deployedVersion":N}'
```

---

## 9. Problèmes fréquents et solutions

| Symptôme                                      | Cause probable                                  | Solution                                                            |
|-----------------------------------------------|-------------------------------------------------|---------------------------------------------------------------------|
| `could not connect to server` au démarrage    | `DATABASE_URL` mal formée ou app PG non prête  | Vérifier le préfixe `postgresql+psycopg://` et l'ordre de démarrage |
| `SuperTokens connection refused`              | `SUPERTOKENS_CONNECTION_URI` incorrect          | Vérifier `http://srv-captain--supertokens:3567` (pas HTTPS interne) |
| Colonnes manquantes en base                   | Migration incomplète                            | `ensure_schema()` s'exécute au démarrage, vérifier les logs         |
| Premier admin impossible à créer              | Schéma incomplet lors du premier run            | Redémarrer l'app, puis faire le SQL de promotion                    |
| Variables d'env dupliquées                    | `APP_CONFIG_JSON` + variables individuelles     | Choisir une seule méthode, `APP_CONFIG_JSON` a priorité basse       |
| Mot de passe PostgreSQL avec caractères spéciaux | URL mal encodée                              | Encoder `@` → `%40`, `#` → `%23`, etc. dans l'URL                  |
| Auth JDBC refusé sur SuperTokens              | Utilisation de `postgresql+psycopg://`          | SuperTokens requiert `postgresql://` sans préfixe driver            |
