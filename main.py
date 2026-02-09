import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime
import uuid

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Baguettotron Dataset Studio", layout="wide")

# --- CONNEXION GOOGLE SHEETS ---
conn = st.connection("gsheets", type=GSheetsConnection)

def load_data():
    # On force le rafraîchissement pour éviter le cache lors des éditions bi-directionnelles
    return conn.read(ttl="0")

df = load_data()

# --- DEFINITION DES OPTIONS (Listes fermées) ---
LISTE_TYPES = ["Normalisation", "Expansion & Suite"]
LISTE_FORMES = ["Narration", "Description", "Portrait", "Dialogue", "Monologue intérieur", "Réflexion", "Scène"]
LISTE_TONS = ["Neutre", "Lyrique", "Mélancolique", "Tendu", "Sardonique", "Chaleureux", "Clinique"]
LISTE_SUPPORTS = ["Narratif", "Épistolaire", "Instantané", "Formel", "Journal intime"]
LISTE_STATUTS = ["A faire", "En cours", "A relire", "Fait et validé"]

# --- SIDEBAR : STATISTIQUES ET EXPORT ---
with st.sidebar:
    st.title("📊 Dataset Status")
    if not df.empty and "statut" in df.columns:
        st.write(df['statut'].value_counts())
    
    st.divider()
    st.subheader("🚀 Export Fine-tuning")
    if not df.empty:
        # On n'exporte que ce qui est validé
        df_export = df[df['statut'] == "Fait et validé"]
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger JSONL/CSV Prêt",
            data=csv,
            file_name=f"dataset_baguettotron_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    st.info("L'export ne contient que les lignes marquées 'Fait et validé'.")

# --- INTERFACE PRINCIPALE ---
st.title("✒️ Baguettotron Style Manager")
st.markdown("Structurez votre dataset de style pour PleIAs.")

tab1, tab2 = st.tabs(["➕ Nouvelle Entrée", "📂 Gestion & Édition"])

# --- TAB 1 : FORMULAIRE D'AJOUT ---
with tab1:
    with st.form("ajout_form", clear_on_submit=True):
        st.subheader("Paramètres de Style")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val_type = st.selectbox("Type", LISTE_TYPES)
        with c2:
            val_forme = st.selectbox("Forme", LISTE_FORMES)
        with c3:
            val_ton = st.selectbox("Ton", LISTE_TONS)
        with c4:
            val_support = st.selectbox("Support", LISTE_SUPPORTS)
        
        st.divider()
        st.subheader("Contenu Littéraire")
        val_input = st.text_area("Brouillon Synthétique (Input)", placeholder="jhon arrive chateu, triste...")
        val_output = st.text_area("Prose Développée (Output)", placeholder="John poussa les lourdes portes...")
        
        st.divider()
        c5, c6 = st.columns(2)
        with c5:
            val_statut = st.selectbox("Statut initial", LISTE_STATUTS)
        with c6:
            val_notes = st.text_input("Notes libres / Contexte")

        submit = st.form_submit_button("Enregistrer l'entrée")

        if submit:
            if val_input and val_output:
                # Création de la nouvelle ligne
                new_row = pd.DataFrame([{
                    "id": str(uuid.uuid4())[:8],
                    "date": datetime.now().strftime("%d/%m/%Y"),
                    "type": val_type,
                    "forme": val_forme,
                    "ton": val_ton,
                    "support": val_support,
                    "input": val_input,
                    "output": val_output,
                    "statut": val_statut,
                    "notes": val_notes
                }])
                
                # Mise à jour du Google Sheet
                updated_df = pd.concat([df, new_row], ignore_index=True)
                conn.update(data=updated_df)
                st.success("Entrée enregistrée avec succès !")
                st.rerun()
            else:
                st.error("L'input et l'output sont obligatoires.")

# --- TAB 2 : EDITION BI-DIRECTIONNELLE ---
with tab2:
    st.subheader("Base de données complète")
    st.info("Vous pouvez modifier les cellules directement ci-dessous. N'oubliez pas de sauvegarder.")

    # Configuration de l'éditeur de données
    edited_df = st.data_editor(
        df,
        num_rows="dynamic", # Permet de supprimer des lignes
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("ID", disabled=True),
            "date": st.column_config.TextColumn("Date", disabled=True),
            "type": st.column_config.SelectboxColumn("Type", options=LISTE_TYPES),
            "forme": st.column_config.SelectboxColumn("Forme", options=LISTE_FORMES),
            "ton": st.column_config.SelectboxColumn("Ton", options=LISTE_TONS),
            "support": st.column_config.SelectboxColumn("Support", options=LISTE_SUPPORTS),
            "statut": st.column_config.SelectboxColumn("Statut", options=LISTE_STATUTS),
            "input": st.column_config.TextColumn("Input", width="medium"),
            "output": st.column_config.TextColumn("Output", width="large"),
            "notes": st.column_config.TextColumn("Notes", width="small"),
        },
        hide_index=True,
    )

    if st.button("💾 Sauvegarder les modifications vers Google Sheets"):
        try:
            conn.update(data=edited_df)
            st.success("Google Sheet mis à jour !")
        except Exception as e:
            st.error(f"Erreur lors de la mise à jour : {e}")
