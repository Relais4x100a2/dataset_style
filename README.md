# ✒️ Baguettotron Dataset Studio

Ce projet est une interface de curation de données conçue pour créer des datasets de **fine-tuning** stylistique (format Instruct) pour le modèle **Baguettotron** (PleIAs). L'outil permet de transformer des notes brutes en prose littéraire tout en catégorisant la forme, le ton et le support.

## 🏗️ Architecture du Projet

* **Frontend :** [Streamlit](https://streamlit.io/) (Déployé sur Streamlit Community Cloud).
* **Base de données :** [Google Sheets](https://www.google.com/sheets/about/) via l'API Google Sheets.
* **Connexion :** `st-gsheets-connection` avec authentification par compte de service.
* **Format d'export :** CSV (brut) et JSONL (Format ChatML avec thinking traces et tokens d'entropie).

---

## 🔑 Configuration de Google Cloud (Le JSON)

Pour que l'application puisse lire/écrire dans votre Google Sheet, suivez ces étapes :

1. **Créer un projet :** Allez sur la [Google Cloud Console](https://console.cloud.google.com/).
2. **Activer les API :** Activez l'**API Google Sheets** et l'**API Google Drive**.
3. **Compte de Service :** * Allez dans `Identifiants` > `Créer des identifiants` > `Compte de service`.
* Une fois créé, allez dans l'onglet `Clés` du compte.
* Cliquez sur `Ajouter une clé` > `Créer une nouvelle clé` > **JSON**.


4. **Téléchargement :** Un fichier `.json` est téléchargé. Il contient vos accès secrets.
5. **Partage du Sheet :** **Indispensable !** Ouvrez votre Google Sheet et partagez-le (bouton Partager) avec l'adresse email du compte de service (ex: `votre-nom@project-id.iam.gserviceaccount.com`) en tant qu'**Éditeur**.

---

## 🔒 Gestion des Secrets

### En Local (Développement)

Créez un fichier `.streamlit/secrets.toml` à la racine de votre projet :

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
spreadsheet = "URL_DE_VOTRE_GOOGLE_SHEET"

```

### Sur Streamlit Community Cloud

1. Allez sur votre dashboard Streamlit.
2. `App Settings` > `Secrets`.
3. Copiez-collez le contenu du fichier `secrets.toml` ci-dessus.

**Important (spaCy / blis) :** L’app utilise spaCy et le modèle français. Si le déploiement échoue avec une erreur du type `Failed building wheel for blis`, c’est souvent dû à Python 3.13. **Lors du déploiement**, dans **Advanced settings**, choisissez **Python 3.12** (le fichier `runtime.txt` à la racine demande déjà 3.12 ; si la plateforme l’ignore, sélectionnez 3.12 manuellement). Puis redéployez (ou créez une nouvelle app avec le même dépôt et Python 3.12).

---

## 🛡️ Contrôle d'accès (Emails spécifiques)

Si vous voulez que seuls certains utilisateurs accèdent à votre application sur Streamlit Cloud :

1. **Dépôt Privé :** Assurez-vous que votre dépôt GitHub est en mode **Privé**.
2. **Invite Only :** Sur Streamlit Cloud, allez dans les paramètres de l'application.
3. Dans l'onglet **"Sharing"**, désactivez l'accès public.
4. Ajoutez manuellement les adresses emails Google des personnes autorisées. Elles devront se connecter avec leur compte Google pour voir l'app.

---

## 🚀 Installation rapide

1. Clonez le dépôt.
2. Installez les dépendances :
```bash
pip install -r requirements.txt

```


3. Lancez l'application :
```bash
streamlit run main.py

```



---

## 📊 Structure du Dataset

* `id` : Identifiant unique de l'entrée.
* `type` : Normalisation ou Normalisation & Expansion.
* `forme` : Narration, Description, Dialogue, etc.
* `ton` : Lyrique, Mélancolique, Tendu, etc.
* `input` : La note brute (brouillon).
* `output` : Le texte stylisé final.

## ✨ Fonctionnalités d'Export

L'application propose deux modes d'export pour les lignes marquées comme **"Fait et validé"** :

* **CSV :** Pour une analyse tabulaire classique. 
* **JSONL Baguettotron :** Génère automatiquement les balises de raisonnement `<think>` (basées sur la Forme et le Ton) et les marqueurs d'entropie `<H≈0.3>` (Normalisation) ou `<H≈1.5>` (Expansion).
