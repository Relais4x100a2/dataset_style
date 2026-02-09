import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# 1. Configuration de la page
st.set_page_config(page_title="Style Dataset", layout="wide")

# 2. Connexion à Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

# Fonction pour charger les données (et forcer le rafraîchissement)
def load_data():
    return conn.read(ttl="0") # ttl=0 pour toujours avoir la version du Sheet

df = load_data()

# --- SIDEBAR : Statistiques et Export ---
with st.sidebar:
    st.title("📊 Statistiques")
    if not df.empty:
        st.write(df['Statut'].value_counts())
    
    st.divider()
    st.subheader("🚀 Export pour Fine-Tuning")
    csv = df[df['Statut'] == "Fait et validé"].to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger le Dataset (CSV)", csv, "dataset_baguettotron.csv", "text/csv")

# --- MAIN : Interface CRUD ---
st.title("📝 Gestionnaire de Style - PleIAs")

# Onglets pour séparer l'Ajout et la Gestion
tab1, tab2 = st.tabs(["➕ Ajouter une entrée", "📂 Gérer & Éditer le Dataset"])

with tab1:
    with st.form("new_entry"):
        col1, col2 = st.columns(2)
        with col1:
            task_type = st.selectbox("Type d'entrée", ["Normalisation", "Expansion & Suite"])
            status = st.selectbox("Statut", ["A faire", "En cours", "A relire", "Fait et validé"])
        with col2:
            notes = st.text_input("Note interne (contexte)")
        
        draft_input = st.text_area("Brouillon Synthétique (Input)")
        prose_output = st.text_area("Style Littéraire (Output)")
        
        if st.form_submit_button("Enregistrer dans le Sheet"):
            # Logique pour ajouter la ligne via gspread ou la connexion Streamlit
            st.success("Entrée ajoutée !")

with tab2:
    st.subheader("Édition rapide (Bi-directionnelle)")
    # Le composant magique de Streamlit pour l'édition
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "Statut": st.column_config.SelectboxColumn(
                "Statut",
                options=["A faire", "En cours", "A relire", "Fait et validé"],
                required=True,
            ),
            "Type": st.column_config.SelectboxColumn(
                "Type", options=["Normalisation", "Expansion & Suite"]
            )
        },
        use_container_width=True
    )

    if st.button("Sauvegarder les modifications"):
        conn.update(data=edited_df)
        st.success("Google Sheet mis à jour !")
