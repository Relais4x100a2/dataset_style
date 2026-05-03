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
  "LANGUAGETOOL_BASE_URL": "http://srv-captain--languagetool:8010"
}
```

Ensuite, déploiement en une commande:

```bash
make prod
```
