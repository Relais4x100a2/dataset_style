# ✒️ Baguettotron Dataset Studio

Interface de curation de données pour constituer des jeux de données de **fine-tuning** stylistique (format Instruct) du modèle **Baguettotron** (PleIAs). Transformation de notes brutes en prose littéraire, avec catégorisation forme / ton / support et exports prêts pour l'entraînement.

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
    ├── export_utils.py  # Conversion dataset → JSONL Baguettotron (ChatML, <think>, <H≈…>)
    ├── nlp_engine.py    # Insights linguistiques, stylométrie, cohérence, LanguageTool — sans Streamlit
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

Colonnes de **cache** (calculées automatiquement à la sauvegarde) :

| Colonne             | Contenu |
|---------------------|---------|
| `_ratio`            | Ratio d'amplification (nb mots output / input) |
| `_ttr`              | Type-Token Ratio (diversité du vocabulaire) |
| `_long_phrases`     | Longueur moyenne des phrases (mots) |
| `_signature_json`   | Signature stylométrique (7 axes) en JSON |
| `_coherence_score`  | Score de cohérence avec la moyenne du dataset (0–100) |
| `_trigrams_json`    | Distribution des trigrammes POS en JSON |

---

## ✨ Fonctionnalités

### Onglet « Nouvelle Entrée »

Formulaire de saisie (type, forme, ton, support, brouillon, prose, statut). Identique à Gestion & Édition pour l'analyse :

- **Corriger l'orthographe** : appel à l'API LanguageTool avant enregistrement.
- **Vérifier ma prose** : analyse linguistique (spaCy) avec métriques, radar, conseils.
- **Enregistrer l'entrée** : crée une nouvelle ligne dans le Sheet avec calcul du cache.

### Onglet « Gestion & Édition »

Navigation fiche par fiche avec filtrage par statut.

- **Corriger l'orthographe** : bouton sous le champ *Prose (Output)*. Uniquement corrections orthographe/grammaire (LanguageTool, pas de réécriture). Gestion du timeout et des erreurs réseau.
- **Vérifier ma prose** : calcul des indicateurs linguistiques (amplification, TTR, longueur phrases, répétitions, Baguette-Touch, radar stylistique, conseils).
- **Enregistrer les modifications** : écriture dans le Sheet + mise à jour du cache si une vérification a été faite.

### Onglet « Tableau de bord »

Vue d'ensemble du dataset, entièrement basée sur le **cache** (pas de spaCy, rendu instantané).

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

Statistiques par statut, boutons **Télécharger CSV** et **Télécharger JSONL** (même taille, CSS harmonisé).

---

## 📤 Export (CSV et JSONL)

Les deux exports ne concernent que les lignes dont le **statut** est **« Fait et validé »**.

- **CSV** : export tabulaire brut (analyse, tableaux, etc.).
- **JSONL Baguettotron** : format ChatML pour fine-tuning, avec balises de raisonnement (forme/ton) et marqueurs d'entropie `<H≈0.3>` (Normalisation) ou `<H≈1.5>` (Expansion).

---

## 🛡️ Contrôle d'accès

Pour limiter l'accès à l'app sur Streamlit Cloud : dépôt GitHub en **privé**, puis dans les paramètres de l'app, onglet **Sharing**, désactiver l'accès public et ajouter les adresses e‑mail autorisées (connexion Google requise).

---

## 🔧 Dépannage

- **503 / Google Sheets indisponible** : l'app réessaie automatiquement (retry + backoff exponentiel, 4 tentatives). Si l'erreur persiste, réessayer plus tard.
- **spaCy non disponible après déploiement** : faire **Reboot** ou **Clear cache and redeploy** dans les paramètres de l'app. Utiliser **Python 3.12** (Advanced settings) pour éviter les incompatibilités blis/NumPy sous 3.13.
- **OOM (mémoire)** : spaCy ne tourne que sur la fiche en cours (Vérifier / Enregistrer) ; le Tableau de bord n'appelle jamais spaCy. Le bloc édition est dans un fragment Streamlit pour limiter les rechargements.
- **Dashboard vide** : les indicateurs stylistiques nécessitent que le cache soit rempli. Ouvrir l'onglet Gestion & Édition, cliquer « Vérifier ma prose » puis « Enregistrer » sur chaque fiche validée.

---

## 🧱 Architecture interne

Les modules sont conçus pour être orthogonaux :

| Module | Responsabilité | Dépendances |
|--------|---------------|-------------|
| `database.py` | Accès données, cache, helpers DataFrame | `pandas`, `json` |
| `nlp_engine.py` | Calculs analytiques (insights, stylométrie, cohérence, LanguageTool) | `pandas`, `requests` — **sans Streamlit** |
| `export_utils.py` | Conversion JSONL ChatML | `pandas`, `json`, `database.py` |
| `ui_components.py` | Rendu Streamlit, état session, graphiques | tous les modules ci-dessus, `streamlit`, `plotly` |

`nlp_engine.py` ne contient aucun import Streamlit — il est testable indépendamment de l'app.

Les seuils des paliers d'interprétation sont centralisés dans `_PALIERS` (table de données) pour éviter toute duplication. La constante `STATUT_VALIDE` est déclarée une seule fois dans `database.py` et importée partout.
