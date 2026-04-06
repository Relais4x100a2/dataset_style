import os

import streamlit as st
from src.database import create_db_engine, load_data
from src.ui_components import (
    render_sidebar,
    render_tab_ajout,
    render_tab_dashboard,
    render_tab_edition,
)


def _database_url() -> str:
    """URL PostgreSQL : priorité à l'environnement (CapRover), puis secret Streamlit."""
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if url:
        return url
    try:
        sec = st.secrets.get("DATABASE_URL")
        if sec:
            return str(sec).strip()
    except (AttributeError, KeyError, TypeError):
        pass
    return ""


# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Dataset Style Studio", layout="wide")


def _hydrate_deployment_env_from_secrets() -> None:
    """Copie des secrets Streamlit vers l'environnement (équivalent CapRover en local)."""
    keys = (
        "LLM_BASE_URL",
        "OLLAMA_BASE_URL",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LANGUAGETOOL_BASE_URL",
        "LLM_TIMEOUT_SECONDS",
    )
    try:
        for key in keys:
            if os.environ.get(key):
                continue
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except (AttributeError, KeyError, TypeError):
        pass


_hydrate_deployment_env_from_secrets()

db_url = _database_url()
if not db_url:
    st.error(
        "Configure **DATABASE_URL** (variable d'environnement sur CapRover, ou clé "
        "`DATABASE_URL` dans `.streamlit/secrets.toml` en local)."
    )
    st.stop()

engine = create_db_engine(db_url)
try:
    df = load_data(engine)
except Exception as e:
    st.error(f"Impossible de charger les données depuis PostgreSQL : {e}")
    st.stop()

# --- DÉFINITION DES OPTIONS (Listes fermées) ---
LISTE_TYPES = ["Normalisation", "Expansion"]
LISTE_FORMES = [
    "Narration",
    "Description",
    "Portrait",
    "Dialogue",
    "Monologue intérieur",
    "Réflexion",
    "Scène",
]
LISTE_TONS = ["Neutre", "Lyrique", "Mélancolique", "Tendu", "Sardonique", "Chaleureux", "Clinique"]
LISTE_SUPPORTS = ["Narratif", "Épistolaire", "Instantané", "Formel", "Journal intime"]
LISTE_STATUTS = ["A faire", "En cours", "A relire", "Fait et validé"]

# --- INTERFACE PRINCIPALE ---
listes = {
    "types": LISTE_TYPES,
    "formes": LISTE_FORMES,
    "tons": LISTE_TONS,
    "supports": LISTE_SUPPORTS,
    "statuts": LISTE_STATUTS,
}

st.title("✒️ Dataset Style Studio")

with st.sidebar:
    render_sidebar(df, engine, listes)

tab1, tab2, tab3 = st.tabs(["➕ Nouvelle Entrée", "📂 Gestion & Édition", "📊 Tableau de bord"])

with tab1:
    render_tab_ajout(df, engine, listes)

with tab2:
    render_tab_edition(df, engine, listes)

with tab3:
    render_tab_dashboard(df, engine, listes)
