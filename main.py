import streamlit as st
from streamlit_gsheets import GSheetsConnection

from src.database import load_data, update_data
from src.ui_components import render_sidebar, render_tab_ajout, render_tab_edition, render_tab_dashboard

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Baguettotron Dataset Studio", layout="wide")

# --- CONNEXION GOOGLE SHEETS ---
conn = st.connection("gsheets", type=GSheetsConnection)
try:
    df = load_data(conn)
except Exception as e:
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code == 503:
        st.error(
            "Le service Google Sheets est temporairement indisponible (503). "
            "Réessayez dans quelques instants."
        )
    elif code == 403:
        st.error("Accès refusé au tableur. Vérifiez les permissions et les identifiants.")
    else:
        st.error(f"Impossible de charger les données : {e}")
    st.stop()

# --- DÉFINITION DES OPTIONS (Listes fermées) ---
# Mise à jour des types selon ta demande
LISTE_TYPES = ["Normalisation", "Expansion"]
LISTE_FORMES = ["Narration", "Description", "Portrait", "Dialogue", "Monologue intérieur", "Réflexion", "Scène"]
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

st.title("✒️ Baguettotron Style Manager")

with st.sidebar:
    render_sidebar(df, conn, listes)

tab1, tab2, tab3 = st.tabs(["➕ Nouvelle Entrée", "📂 Gestion & Édition", "📊 Tableau de bord"])

with tab1:
    render_tab_ajout(df, conn, listes)

with tab2:
    render_tab_edition(df, conn, listes)

with tab3:
    render_tab_dashboard(df, listes)
