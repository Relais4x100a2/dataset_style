# CapRover: configuration en une variable

Dans CapRover, vous pouvez définir **une seule variable** `APP_CONFIG_JSON`
et y placer tout le paramétrage runtime.

Exemple:

```json
{
  "DATABASE_URL": "postgresql+psycopg://dataset_user:change_me@srv-captain--postgresql:5432/dataset_style",
  "SUPERTOKENS_CONNECTION_URI": "http://srv-captain--supertokens:3567",
  "SUPERTOKENS_API_KEY": "",
  "LLM_BASE_URL": "http://srv-captain--ollama:11434",
  "LLM_MODEL": "mistral",
  "LLM_API_KEY": "",
  "LLM_TIMEOUT_SECONDS": "300",
  "LANGUAGETOOL_BASE_URL": "http://srv-captain--languagetool:8010",
  "DISABLE_SELF_SERVICE_PROJECT_CREATION": "",
  "DATASET_STYLE_UX_TELEMETRY_DIR": "/tmp/dataset_style_ux_telemetry"
}
```

`DISABLE_SELF_SERVICE_PROJECT_CREATION` : laisser vide pour autoriser la création du premier projet depuis l’UI ; mettre `"true"` (ou `"1"`) pour un parcours **invitation uniquement** (message sans formulaire pour les utilisateurs sans projet).

`APP_MIGRATION_INFO_BANNER` *(optionnel, issue-021 / #143)* : texte court affiché aux curateurs dans Streamlit et sur la page d’accueil du service `webapp` pendant une phase de communication (migration, maintenance annoncée). Laisser **absent** ou **vide** pour désactiver après cutover. Texte brut uniquement (pas de HTML).

Ensuite, déploiement en une commande:

```bash
make prod
```

## Staging / préprod (branche `deploy-newfrontend`)

Même structure que l’exemple ci‑dessus, avec des **valeurs distinctes** de la production : autre base PostgreSQL (ou autre base sur un serveur dédié préprod), autre couple `SUPERTOKENS_*` pointant vers un core SuperTokens de staging, et **`APP_PUBLIC_BASE_URL`** égal à l’URL HTTPS réellement exposée aux testeurs (sous‑domaine CapRover de préprod).

Exemple **sans secrets réels** (placeholders à remplacer par l’équipe ops) :

```json
{
  "APP_PUBLIC_BASE_URL": "https://dataset-style-staging.example.invalid",
  "DATABASE_URL": "postgresql+psycopg://dataset_user_staging:CHANGE_ME@srv-captain--postgresql-staging:5432/dataset_style_staging",
  "SUPERTOKENS_CONNECTION_URI": "http://srv-captain--supertokens-staging:3567",
  "SUPERTOKENS_API_KEY": "CHANGE_ME_STAGING_API_KEY",
  "WEBAPP_CORS_ORIGINS": "https://dataset-style-staging.example.invalid",
  "LLM_BASE_URL": "",
  "LLM_MODEL": "",
  "LLM_API_KEY": "",
  "LLM_TIMEOUT_SECONDS": "300",
  "LANGUAGETOOL_BASE_URL": "",
  "DISABLE_SELF_SERVICE_PROJECT_CREATION": "",
  "DATASET_STYLE_UX_TELEMETRY_DIR": "/tmp/dataset_style_ux_telemetry"
}
```

Si le BFF FastAPI est servi sur **une autre origine** que le HTML (recette double app, voir `docs/caprover_deployment.md` §4.5), compléter `WEBAPP_CORS_ORIGINS` avec une liste JSON des origines autorisées, fermée et documentée (ADR 0006).

