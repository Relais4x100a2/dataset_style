# ✒️ Dataset Style Studio

Interface de curation de données pour le **fine-tuning** stylistique (brouillon → prose). Catégorisation forme / ton / support, indicateurs stylométriques et exports multi-modèles (LFM2-24B-A2B, PleIAs/Baguettotron, Mistral Small Creative).

---

## Sommaire

- [Architecture et stack](#-architecture-et-stack)
- [Structure du projet](#-structure-du-projet)
- [Installation et lancement](#-installation-et-lancement)
- [Configuration Google Sheets](#-configuration-google-sheets)
- [Secrets (local et Cloud)](#-gestion-des-secrets)
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
| Interface        | [Streamlit](https://streamlit.io/), déploiement possible sur Streamlit Community Cloud |
| Données          | [Google Sheets](https://www.google.com/sheets/about/) (API Sheets + Drive) |
| Connexion        | `st-gsheets-connection` (authentification par compte de service) |
| NLP / analyse    | [spaCy](https://spacy.io/) `fr_core_news_sm` (métriques, cohérence, cache) |
| Correction FR    | [LanguageTool](https://languagetool.org/) (API publique HTTP, pas de Java) |
| Dictionnaire     | [Wiktionnaire](https://fr.wiktionary.org/) via [API Wikimedia](https://www.mediawiki.org/wiki/API:Main_page) (définitions, synonymes, antonymes, vocabulaire apparenté, anagrammes) |
| Génération LLM   | [OpenRouter](https://openrouter.ai/) (modèle `liquid/lfm-2-24b-a2b`) pour brouillon ↔ prose selon type, forme, ton, support |
| Visualisation    | [Plotly](https://plotly.com/python/) (radar, histogrammes, tendances) |

Les appels à l'API Google Sheets sont retentés en cas d'erreur temporaire (503, 429, etc.) avec backoff exponentiel.

---

## 📁 Structure du projet

```
dataset_style/
├── main.py              # Point d'entrée Streamlit, chargement des données, 3 onglets
├── requirements.txt     # Dépendances Python (Streamlit, spaCy, requests, etc.)
├── runtime.txt          # Version Python pour le déploiement Cloud
├── README.md
└── src/
    ├── __init__.py
    ├── database.py      # Connexion Sheets, load_data (retry), update_data, helpers cache
    ├── export_utils.py  # Export JSONL multi-modèles (LFM2, Baguettotron, Mistral) + option stylométrie
    ├── nlp_engine.py    # Insights linguistiques, stylométrie, cohérence, LanguageTool — sans Streamlit
    ├── wiktionary.py    # Wiktionnaire (API Wikimedia) : définitions, synonymes, antonymes, vocabulaire apparenté, anagrammes
    ├── llm_generate.py  # Génération brouillon ↔ prose par LLM (OpenRouter, liquid/lfm-2-24b-a2b)
    └── ui_components.py # Sidebar, onglets Nouvelle Entrée / Gestion & Édition / Tableau de bord
```

---

## 🚀 Installation et lancement

**Prérequis :** Python 3.12 recommandé (compatibilité Streamlit Cloud et blis/spaCy).

```bash
git clone <url-du-depot>
cd dataset_style
pip install -r requirements.txt
streamlit run main.py
```

L'app s'ouvre dans le navigateur. Une configuration Google Sheets (projet Cloud, compte de service, secrets) est nécessaire pour charger et enregistrer les données.

---

## 🔑 Configuration Google Sheets

1. **Créer un projet** dans la [Google Cloud Console](https://console.cloud.google.com/).
2. **Activer les API** : **Google Sheets** et **Google Drive**.
3. **Compte de service** : *Identifiants* → *Créer des identifiants* → *Compte de service*. Dans l'onglet *Clés* du compte, *Ajouter une clé* → *Créer une nouvelle clé* → **JSON**.
4. **Télécharger** le fichier JSON (clés secrètes).
5. **Partager le Google Sheet** avec l'adresse e‑mail du compte de service (ex. `xxx@project-id.iam.gserviceaccount.com`) en **Éditeur**.

---

## 🔒 Gestion des secrets

### En local

Créer `.streamlit/secrets.toml` à la racine du projet :

```toml
[connections.gsheets]
type = "service_account"
project_id = "votre-project-id"
private_key_id = "votre-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "votre-email@project-id.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
spreadsheet = "URL_COMPLETE_DE_VOTRE_GOOGLE_SHEET"
```

Pour activer la **génération par LLM** (brouillon ↔ prose via OpenRouter), ajouter dans le même fichier (ou dans les Secrets du Cloud) :

```toml
[connections.openrouter]
api_key = "votre-cle-openrouter"
```

Le modèle utilisé est `liquid/lfm-2-24b-a2b`. Sans clé, les boutons de génération restent désactivés et un message indique comment configurer.

### Sur Streamlit Community Cloud

*App Settings* → *Secrets* : coller le contenu du `secrets.toml` ci‑dessus.

---

## 📊 Structure du dataset

Chaque ligne du Sheet correspond à une fiche de curation :

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

Colonnes de **cache** (calculées automatiquement à la sauvegarde, utilisables comme **insights pour le fine-tuning**) :

| Colonne             | Contenu |
|---------------------|---------|
| `_ratio`            | Ratio d'amplification (nb mots output / input) |
| `_ttr`              | Type-Token Ratio (diversité du vocabulaire) |
| `_long_phrases`     | Longueur moyenne des phrases (mots) |
| `_signature_json`   | Signature stylométrique (7 axes) en JSON |
| `_coherence_score`  | Score de cohérence avec la moyenne du dataset (0–100) |
| `_trigrams_json`    | Distribution des trigrammes POS en JSON |
| `_lexical_density`  | Densité lexicale (N+V+Adj+Adv) / tokens |
| `_weak_verb_ratio`  | Proportion de verbes faibles (être, avoir, faire, aller, dire) |
| `_syntax_contrast`  | Contraste stylistique input ↔ output (0–1) |
| `_nb_sentences`     | Nombre de phrases (output) |
| `_punct_exp`        | Ponctuation expressive : "n,m,p" (tirets —, ..., :) |
| `_stop_ratio_out`   | Proportion de mots-outils dans l'output |

Voir [docs/stylometrie_finetuning.md](docs/stylometrie_finetuning.md) pour le rôle de chaque indicateur lors du fine-tuning.

---

## ✨ Fonctionnalités

### Onglet « Nouvelle Entrée »

Formulaire de saisie (type, forme, ton, support, brouillon, prose, statut). Identique à Gestion & Édition pour l'analyse :

- **Corriger l'orthographe** : appel à l'API LanguageTool avant enregistrement.
- **Génération par LLM** : deux boutons — « Générer le brouillon depuis la prose » et « Générer la prose depuis le brouillon » (OpenRouter, type / forme / ton / support pris en compte).
- **Vérifier ma prose** : analyse linguistique (spaCy) avec métriques, radar, conseils.
- **Wiktionnaire** : champ « Mot à vérifier » + recherche (API Wikimedia) pour afficher définitions, synonymes, antonymes, vocabulaire apparenté et anagrammes.
- **Enregistrer l'entrée** : crée une nouvelle ligne dans le Sheet avec calcul du cache.

### Onglet « Gestion & Édition »

Navigation fiche par fiche avec filtrage par statut.

- **Corriger l'orthographe** : bouton sous le champ *Prose (Output)*. Uniquement corrections orthographe/grammaire (LanguageTool, pas de réécriture). Gestion du timeout et des erreurs réseau.
- **Génération par LLM** : mêmes boutons qu’en Nouvelle Entrée pour générer le brouillon à partir de la prose ou l’inverse (paramètres de la fiche utilisés).
- **Vérifier ma prose** : calcul des indicateurs linguistiques (amplification, TTR, longueur phrases, répétitions, Baguette-Touch, radar stylistique, conseils).
- **Wiktionnaire** : même outil que dans Nouvelle Entrée pour consulter définitions, synonymes, antonymes, vocabulaire apparenté et anagrammes pendant l’édition.
- **Enregistrer les modifications** : écriture dans le Sheet + mise à jour du cache si une vérification a été faite.

### Onglet « Tableau de bord »

Vue d'ensemble du dataset, entièrement basée sur le **cache** (sauf pour la gestion du cache).

**Cache stylométrique (générer / écraser / enregistrer)**
- **Générer le cache (fiches sans cache)** : remplit le cache uniquement pour les fiches « Fait et validé » qui n'en ont pas encore (sans toucher aux autres).
- **Écraser tout le cache et enregistrer** : recalcule tout le cache des fiches validées et met à jour le Google Sheet (avec case à cocher de confirmation).

**Section 1 — Composition**
- Métriques rapides (total / validées / en cours / à faire), barre de progression.
- Distribution des statuts et types (bar charts).
- Expander détaillant formes, tons et supports.

**Section 2 — Qualité stylistique**
- Score santé global (0–100), cohérence moyenne, TTR moyen, ratio moyen.
- Histogrammes de distribution : ratio, TTR, longueur des phrases.
- Histogramme des scores de cohérence avec zones colorées (rouge < 45, orange 45–65, vert > 65).

**Section 3 — Stylométrie globale**
- Radar de la signature moyenne du dataset avec bandes d'erreur (±σ).
- Tableau de dispersion par axe stylistique.
- Top 15 constructions grammaticales (trigrammes POS).
- Courbe d'évolution de la cohérence dans le temps.

**Section 4 — Alertes qualité**
- Fiches problématiques identifiées depuis le cache : cohérence critique (< 45), expansion faible (ratio < 1.5), vocabulaire répétitif (TTR < 0.50).
- Bar chart des alertes par type + tableau détaillé avec ID, type, forme, ton.

### Sidebar

Statistiques par statut. **Export Fine-tuning** : choix du **format d'export JSONL** (LFM2-24B-A2B, PleIAs/Baguettotron, Mistral Small Creative), option **Inclure indicateurs stylométriques**, puis boutons **Télécharger CSV** et **Télécharger JSONL** (même taille, CSS harmonisé).

---

## 📤 Export (CSV et JSONL)

Les deux exports ne concernent que les lignes dont le **statut** est **« Fait et validé »**.

- **CSV** : export tabulaire brut (analyse, tableaux, etc.).
- **JSONL** : format dépend du **modèle cible** choisi dans la sidebar :
  - **LFM2-24B-A2B** : structure `messages` (system optionnel, user, assistant), une ligne JSON par fiche. Option stylométrie → message system.
  - **PleIAs/Baguettotron** : ChatML avec balise `<think>` (trace forme/ton) et marqueurs d'entropie `<H≈0.3>` (Normalisation) ou `<H≈1.5>` (Expansion). Option stylométrie → ligne « Stylo » dans la trace.
  - **Mistral Small Creative** : structure `messages` (user, assistant). Option stylométrie → préfixe dans le message user.

Si **Inclure indicateurs stylométriques** est coché, les colonnes de cache (TTR, longueur de phrase, densité lexicale, etc.) sont injectées dans l'export selon le format (voir [docs/stylometrie_finetuning.md](docs/stylometrie_finetuning.md)).

---

## 🛡️ Contrôle d'accès

Pour limiter l'accès à l'app sur Streamlit Cloud : dépôt GitHub en **privé**, puis dans les paramètres de l'app, onglet **Sharing**, désactiver l'accès public et ajouter les adresses e‑mail autorisées (connexion Google requise).

---

## 🔧 Dépannage

- **503 / Google Sheets indisponible** : l'app réessaie automatiquement (retry + backoff exponentiel, 4 tentatives). Si l'erreur persiste, réessayer plus tard.
- **spaCy non disponible après déploiement** : faire **Reboot** ou **Clear cache and redeploy** dans les paramètres de l'app. Utiliser **Python 3.12** (Advanced settings) pour éviter les incompatibilités blis/NumPy sous 3.13.
- **OOM (mémoire)** : spaCy ne tourne que sur la fiche en cours (Vérifier / Enregistrer) ; le Tableau de bord n'appelle jamais spaCy. Le bloc édition est dans un fragment Streamlit pour limiter les rechargements.
- **Dashboard vide** : les indicateurs stylistiques nécessitent que le cache soit rempli. Soit utiliser l'onglet Gestion & Édition (« Vérifier ma prose » puis « Enregistrer »), soit dans le Tableau de bord ouvrir « Cache stylométrique » et cliquer sur « Générer le cache (fiches sans cache) » ou « Écraser tout le cache et enregistrer ».

---

## 🧱 Architecture interne

Les modules sont conçus pour être orthogonaux :

| Module | Responsabilité | Dépendances |
|--------|---------------|-------------|
| `database.py` | Accès données, cache, helpers DataFrame | `pandas`, `json` |
| `nlp_engine.py` | Calculs analytiques (insights, stylométrie, cohérence, LanguageTool) | `pandas`, `requests` — **sans Streamlit** |
| `export_utils.py` | Export JSONL multi-modèles (LFM2, Baguettotron, Mistral) | `pandas`, `json`, `database.py` |
| `ui_components.py` | Rendu Streamlit, état session, graphiques | tous les modules ci-dessus, `streamlit`, `plotly` |

`nlp_engine.py` ne contient aucun import Streamlit — il est testable indépendamment de l'app.

Les seuils des paliers d'interprétation sont centralisés dans `_PALIERS` (table de données) pour éviter toute duplication. La constante `STATUT_VALIDE` est déclarée une seule fois dans `database.py` et importée partout.
