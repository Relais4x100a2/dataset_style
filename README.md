# ✒️ Dataset Style Studio

Interface de curation de données pour le **fine-tuning** stylistique (brouillon → prose). Catégorisation forme / ton / support, indicateurs stylométriques et exports multi-modèles (LFM2-24B-A2B, PleIAs/Baguettotron, Mistral Small Creative).

**Branche `deploy/caprover-relais4` :** persistance **PostgreSQL**, déploiement **Docker / CapRover** (ex. serveur **relais4x100a2**), génération LLM via **Ollama** (API compatible OpenAI) ou **OpenRouter**, correction **LanguageTool** local ou public.

---

## Sommaire

- [Architecture et stack](#-architecture-et-stack)
- [Structure du projet](#-structure-du-projet)
- [Installation et lancement](#-installation-et-lancement)
- [Déploiement CapRover (relais4x100a2)](#-déploiement-caprover-relais4x100a2)
- [Variables d'environnement et secrets](#-variables-denvironnement-et-secrets)
- [Import CSV (migration depuis un Sheet)](#-import-csv-migration-depuis-un-sheet)
- [Structure du dataset](#-structure-du-dataset)
- [Fonctionnalités](#-fonctionnalités)
- [Export (CSV et JSONL)](#-export-csv-et-jsonl)
- [Contrôle d'accès](#-contrôle-daccès)
- [Dépannage](#-dépannage)
- [Architecture interne](#-architecture-interne)

---

## 🏗️ Architecture et stack

| Composant        | Technologie |
|------------------|------------|
| Interface        | [Streamlit](https://streamlit.io/) |
| Données          | [PostgreSQL](https://www.postgresql.org/) (table `entries`, SQLAlchemy + pandas) |
| Déploiement      | [Docker](https://www.docker.com/) multi-stage, [CapRover](https://caprover.com/) (`captain-definition`) |
| NLP / analyse    | [spaCy](https://spacy.io/) `fr_core_news_sm` (métriques, cohérence, cache) |
| Correction FR    | [LanguageTool](https://languagetool.org/) — API publique par défaut, ou serveur auto-hébergé (`LANGUAGETOOL_BASE_URL`) |
| Dictionnaire     | [Wiktionnaire](https://fr.wiktionary.org/) via [API Wikimedia](https://www.mediawiki.org/wiki/API:Main_page) |
| Génération LLM   | **Ollama** (ou compatible OpenAI `/v1/chat/completions`) si `LLM_BASE_URL` / `OLLAMA_BASE_URL`, sinon **OpenRouter** |
| Visualisation    | [Plotly](https://plotly.com/python/) (radar, histogrammes, tendances) |

Les lectures PostgreSQL sont retentées en cas d'erreur transitoire (connexion, serveur) avec backoff exponentiel.

---

## 📁 Structure du projet

```
dataset_style/
├── main.py               # Streamlit, DATABASE_URL, hydratation secrets → env
├── requirements.txt
├── runtime.txt           # Python 3.12 (référence)
├── pyproject.toml        # Métadonnées + configuration Ruff
├── Dockerfile            # Image multi-stage (build deps → runtime)
├── captain-definition    # CapRover : build depuis Dockerfile
├── .dockerignore
├── README.md
├── scripts/
│   └── import_csv_to_pg.py   # Import CSV → table entries
└── src/
    ├── __init__.py
    ├── database.py       # PostgreSQL : ensure table, load_data, update_data, helpers cache
    ├── export_utils.py   # Export JSONL multi-modèles + option stylométrie
    ├── nlp_engine.py     # Stylométrie, LanguageTool (URL configurable)
    ├── wiktionary.py
    ├── llm_generate.py # Chat completions : Ollama / OpenRouter
    └── ui_components.py
```

---

## 🚀 Installation et lancement

**Prérequis :** Python 3.12, instance PostgreSQL accessible (`DATABASE_URL`).

```bash
git clone <url-du-depot>
cd dataset_style
git checkout deploy/caprover-relais4
pip install -r requirements.txt
export DATABASE_URL="postgresql://USER:PASS@HOST:5432/DBNAME"
streamlit run main.py
```

En local, tu peux aussi définir `DATABASE_URL` dans `.streamlit/secrets.toml` (clé racine `DATABASE_URL = "..."`).

### Tests automatisés (pytest)

Les tests couvrent la logique métier **sans PostgreSQL** (export JSONL, agrégats / alertes depuis le cache) :

```bash
pip install -r requirements.txt
pytest
```

La configuration (`pythonpath`, répertoire `tests/`) est dans `pyproject.toml`.

---

## 🚢 Déploiement CapRover (relais4x100a2)

Cible typique : VPS **Hetzner CX53** (ex. hostname **relais4x100a2**) avec **CapRover** déjà installé.

### Ordre recommandé des applications CapRover

1. **PostgreSQL** — modèle one-click CapRover ; récupère l’URL JDBC/Postgres et construis `DATABASE_URL` (format `postgresql://user:password@srv-captain--postgres:5432/dbname` selon ton instance).
2. **LanguageTool** (optionnel mais recommandé) — image Docker communautaire (ex. conteneur exposant `/v2/check` sur un port interne). Note le nom d’app CapRover pour l’URL interne.
3. **Ollama** — application dédiée avec volume persistant pour les modèles ; `ollama pull <modèle>` sur le serveur. Les modèles type **Luth** / **Baguettotron** sont gérés côté Ollama, pas dans ce repo.
4. **Dataset Style Studio** — déploiement depuis ce dépôt (branche `deploy/caprover-relais4`) : CapRover détecte `captain-definition` et build le `Dockerfile`.

### Réseau interne CapRover

Depuis le conteneur Streamlit, les autres apps sont joignables sous la forme **`http://srv-captain--<nom-app>:<port>`** (sans `https` entre conteneurs). Exemples :

- Ollama API : `LLM_BASE_URL=http://srv-captain--ollama:11434`
- LanguageTool : `LANGUAGETOOL_BASE_URL=http://srv-captain--languagetool:8010` (adapter le port à l’image utilisée)

### Variables à renseigner dans l’app Streamlit (CapRover → App Configs → Environmental Variables)

| Variable | Rôle |
|----------|------|
| `DATABASE_URL` | Connexion PostgreSQL (obligatoire) |
| `LLM_BASE_URL` ou `OLLAMA_BASE_URL` | Base URL du serveur compatible OpenAI (ex. Ollama) |
| `LLM_MODEL` | Nom du modèle par défaut si la sidebar est vide |
| `LLM_API_KEY` | Optionnel (Bearer) si le proxy l’exige |
| `LLM_TIMEOUT_SECONDS` | Timeout HTTP génération (défaut `300`) |
| `LANGUAGETOOL_BASE_URL` | Origine du serveur LT sans chemin (ex. `http://srv-captain--languagetool:8010`) |

**Confidentialité :** avec PostgreSQL + Ollama + LanguageTool sur le même VPS, les textes peuvent rester entièrement sur ta machine. Aucun token Hugging Face / Google n’est requis pour ce flux.

---

## 🔒 Variables d'environnement et secrets

### CapRover

Définir les variables dans l’interface CapRover (pas besoin de `secrets.toml` dans l’image).

### Développement local (`.streamlit/secrets.toml`)

Exemple minimal :

```toml
DATABASE_URL = "postgresql://user:pass@localhost:5432/dataset_style"
```

Optionnel — mêmes clés que sur CapRover (copiées dans l’environnement au démarrage par `main.py`) :

```toml
LLM_BASE_URL = "http://127.0.0.1:11434"
LLM_MODEL = "mistral"
LANGUAGETOOL_BASE_URL = "http://127.0.0.1:8010"
```

**OpenRouter** (si tu n’utilises pas d’URL locale) :

```toml
[connections.openrouter]
api_key = "votre-cle-openrouter"
```

Sans `LLM_BASE_URL` / `OLLAMA_BASE_URL`, l’app utilise OpenRouter et exige cette clé pour activer les boutons de génération.

---

## 📥 Import CSV (migration depuis un Sheet)

Après export CSV depuis un ancien Google Sheet (mêmes noms de colonnes) :

```bash
export DATABASE_URL="postgresql://..."
python scripts/import_csv_to_pg.py chemin/vers/export.csv
```

L’import **remplace** tout le contenu actuel de la table `entries` (équivalent d’une réécriture complète du jeu de données).

---

## 📊 Structure du dataset

Chaque ligne de la table `entries` correspond à une fiche de curation :

| Champ     | Rôle |
|-----------|------|
| `id`      | Identifiant unique |
| `type`    | **Normalisation** ou **Expansion** |
| `forme`   | Narration, Description, Portrait, Dialogue, Monologue intérieur, Réflexion, Scène |
| `ton`     | Neutre, Lyrique, Mélancolique, Tendu, Sardonique, Chaleureux, Clinique |
| `support` | Narratif, Épistolaire, Instantané, Formel, Journal intime |
| `input`   | Brouillon / note brute |
| `output`  | Prose finale stylisée |
| `statut`  | A faire, En cours, A relire, **Fait et validé** |
| `notes`   | Notes libres |

Colonnes de **cache** (calculées à la sauvegarde, voir [docs/stylometrie_finetuning.md](docs/stylometrie_finetuning.md)) : `_ratio`, `_ttr`, `_long_phrases`, `_signature_json`, `_coherence_score`, `_trigrams_json`, `_lexical_density`, `_weak_verb_ratio`, `_syntax_contrast`, `_nb_sentences`, `_punct_exp`, `_stop_ratio_out`.

---

## ✨ Fonctionnalités

Comportement identique à la version historique (onglets Nouvelle entrée, Gestion & Édition, Tableau de bord, exports JSONL, Wiktionnaire, stylométrie). Les différences portent sur la **persistance PostgreSQL**, le **LLM** (Ollama ou OpenRouter) et l’URL **LanguageTool** configurable.

---

## 📤 Export (CSV et JSONL)

Les exports concernent les lignes dont le **statut** est **« Fait et validé »**. Les formats JSONL (LFM2, Baguettotron, Mistral) et l’option stylométrie sont inchangés.

---

## 🛡️ Contrôle d'accès

- **CapRover** : HTTPS, mots de passe d’app, restriction par IP ou VPN selon ta configuration.
- **PostgreSQL** : utilisateur/mot de passe dédiés, réseau interne Docker uniquement si possible.

---

## 🔧 Dépannage

- **Erreur au chargement PostgreSQL** : vérifie `DATABASE_URL`, que la base existe et que le conteneur Streamlit atteint l’hôte Postgres (`srv-captain--...` ou IP interne).
- **Génération LLM timeout** : augmente `LLM_TIMEOUT_SECONDS` ; sur CPU, les gros modèles sont lents.
- **LanguageTool** : sans `LANGUAGETOOL_BASE_URL`, l’app utilise l’API publique (limites de débit / taille). En local, préfère un conteneur LT dédié.
- **spaCy / modèle FR** : l’image Docker installe `fr_core_news_sm` via `requirements.txt`. En cas d’échec de build, vérifie que l’URL du wheel est accessible depuis le réseau de build.

---

## 🧱 Architecture interne

| Module | Responsabilité |
|--------|----------------|
| `database.py` | Moteur SQLAlchemy, table `entries`, `load_data` / `update_data`, helpers cache |
| `nlp_engine.py` | Analyses, `languagetool_check_url()` |
| `llm_generate.py` | `POST` `/v1/chat/completions` (Ollama ou OpenRouter) |
| `export_utils.py` | Exports JSONL |
| `ui_components.py` | UI Streamlit, `_llm_ready()`, fragments |

`nlp_engine.py` et `llm_generate.py` n’importent pas Streamlit.

### CI

Le workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) exécute **Ruff** (`check` + `format --check`) sur les pushes et pull requests.

---

## Licence et usage

Projet privé / littéraire : adapte la licence et les sauvegardes PostgreSQL (snapshots CapRover ou dumps `pg_dump`) selon tes besoins.
