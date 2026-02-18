# ✒️ Baguettotron Dataset Studio

Interface de curation de données pour constituer des jeux de données de **fine-tuning** stylistique (format Instruct) du modèle **Baguettotron** (PleIAs). Transformation de notes brutes en prose littéraire, avec catégorisation forme / ton / support et exports prêts pour l’entraînement.

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
- [Contrôle d’accès](#-contrôle-daccès)
- [Dépannage](#-dépannage)

---

## 🏗️ Architecture et stack

| Composant        | Technologie |
|------------------|------------|
| Interface        | [Streamlit](https://streamlit.io/), déploiement possible sur Streamlit Community Cloud |
| Données          | [Google Sheets](https://www.google.com/sheets/about/) (API Sheets + Drive) |
| Connexion        | `st-gsheets-connection` (authentification par compte de service) |
| NLP / analyse    | [spaCy](https://spacy.io/) `fr_core_news_sm` (métriques, cohérence, cache) |
| Correction FR    | [LanguageTool](https://languagetool.org/) (API publique HTTP, pas de Java) |
| Visualisation    | [Plotly](https://plotly.com/python/) (radar, tendances) |

Les appels à l’API Google Sheets sont retentés en cas d’erreur temporaire (503, 429, etc.) avec backoff exponentiel.

---

## 📁 Structure du projet

```
dataset_style/
├── main.py              # Point d’entrée Streamlit, chargement des données, onglets
├── requirements.txt     # Dépendances Python (Streamlit, spaCy, requests, etc.)
├── runtime.txt          # Version Python pour le déploiement Cloud
├── README.md
└── src/
    ├── __init__.py
    ├── database.py      # Connexion Sheets, load_data (retry), update_data, cache (colonnes _*)
    ├── export_utils.py  # Conversion dataset → JSONL Baguettotron (ChatML, <think>, <H≈…>)
    ├── nlp_engine.py    # spaCy, insights linguistiques, corriger_texte_fr (LanguageTool), cohérence
    └── ui_components.py # Sidebar, formulaire ajout, onglet édition (analyse, graphiques, boutons)
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

L’app s’ouvre dans le navigateur. Une configuration Google Sheets (projet Cloud, compte de service, secrets) est nécessaire pour charger et enregistrer les données.

---

## 🔑 Configuration Google Sheets

1. **Créer un projet** dans la [Google Cloud Console](https://console.cloud.google.com/).
2. **Activer les API** : **Google Sheets** et **Google Drive**.
3. **Compte de service** : *Identifiants* → *Créer des identifiants* → *Compte de service*. Dans l’onglet *Clés* du compte, *Ajouter une clé* → *Créer une nouvelle clé* → **JSON**.
4. **Télécharger** le fichier JSON (clés secrètes).
5. **Partager le Google Sheet** avec l’adresse e‑mail du compte de service (ex. `xxx@project-id.iam.gserviceaccount.com`) en **Éditeur**.

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

### Sur Streamlit Community Cloud

*App Settings* → *Secrets* : coller le contenu du `secrets.toml` ci‑dessus.

---

## 📊 Structure du dataset

Chaque ligne du Sheet correspond à une fiche de curation :

| Champ   | Rôle |
|--------|------|
| `id`   | Identifiant unique |
| `type` | **Normalisation** ou **Expansion** |
| `forme`| Narration, Description, Portrait, Dialogue, Monologue intérieur, Réflexion, Scène |
| `ton`  | Neutre, Lyrique, Mélancolique, Tendu, Sardonique, Chaleureux, Clinique |
| `support` | Narratif, Épistolaire, Instantané, Formel, Journal intime |
| `input`  | Brouillon / note brute |
| `output` | Prose finale stylisée |
| `statut` | A faire, En cours, A relire, **Fait et validé** |
| `notes`  | Notes libres |

Colonnes de **cache** (remplies par l’app à l’analyse / sauvegarde) : `_ratio`, `_richesse`, `_ttr`, `_long_phrases`, `_signature_json`, `_coherence_score`, `_trigrams_json`.

---

## ✨ Fonctionnalités

- **Onglet « Nouvelle Entrée »** : formulaire (type, forme, ton, support, brouillon, prose, statut). Envoi d’une nouvelle ligne vers le Sheet.
- **Onglet « Gestion & Édition »** : sélection d’une fiche par ID, édition de tous les champs.
  - **Vérifier ma prose** : calcul des indicateurs linguistiques (amplification, TTR, longueur des phrases, répétitions, conseils).
  - **Corriger l’orthographe** : bouton sous le champ *Prose (Output)*. Appel à l’API LanguageTool (français) ; uniquement corrections orthographe/grammaire, pas de réécriture. Gestion du timeout et des erreurs réseau (messages dans l’interface).
  - **Enregistrer les modifications** : écriture dans le Sheet (et mise à jour du cache si une vérification a été faite).
- **Sidebar** : statistiques par statut, exports **Télécharger CSV** et **Télécharger JSONL** (largeur/hauteur harmonisées), rappel sur le format JSONL.

En cas d’indisponibilité temporaire de l’API Google (503, etc.), un message d’erreur explicite est affiché et un retry automatique est effectué au chargement des données.

---

## 📤 Export (CSV et JSONL)

Les deux exports ne concernent que les lignes dont le **statut** est **« Fait et validé »**.

- **CSV** : export tabulaire brut (analyse, tableaux, etc.).
- **JSONL Baguettotron** : format ChatML pour fine-tuning, avec :
  - balises de raisonnement (forme/ton) et marqueurs d’entropie `<H≈0.3>` (Normalisation) ou `<H≈1.5>` (Expansion).

---

## 🛡️ Contrôle d’accès

Pour limiter l’accès à l’app sur Streamlit Cloud : dépôt GitHub en **privé**, puis dans les paramètres de l’app, onglet **Sharing**, désactiver l’accès public et ajouter les adresses e‑mail autorisées (connexion Google requise).

---

## 🔧 Dépannage

- **503 / Google Sheets indisponible** : l’app réessaie automatiquement (retry + backoff). Si l’erreur persiste, réessayer plus tard.
- **spaCy non disponible après déploiement** : dans les paramètres de l’app sur Streamlit Cloud, faire **Reboot** ou **Clear cache and redeploy**. Utiliser **Python 3.12** (Advanced settings) pour éviter les soucis avec blis sous 3.13.
- **OOM (mémoire)** : l’audit et le radar s’appuient sur les colonnes cache du Sheet ; spaCy ne tourne que sur la fiche en cours (Vérifier / Enregistrer). Le bloc édition est dans un fragment pour limiter les rechargements.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>