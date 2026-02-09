import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime
import uuid
import json 
import io   

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Baguettotron Dataset Studio", layout="wide")

# --- CONNEXION GOOGLE SHEETS ---
conn = st.connection("gsheets", type=GSheetsConnection)

def load_data():
    # On force le rafraîchissement (ttl=0)
    data = conn.read(ttl="0")
    # NETTOYAGE CRUCIAL : On force tout en texte pour éviter l'erreur FLOAT sur les colonnes vides
    data = data.astype(str).replace(['nan', 'None', '<NA>'], '')
    return data

df = load_data()

# --- FONCTION D'EXPORT BAGUETTOTRON (JSONL) ---
def convert_to_baguettotron_jsonl(df):
    jsonl_output = io.StringIO()
    # On ne prend que ce qui est validé
    df_valid = df[df['statut'] == "Fait et validé"]
    
    for _, row in df_valid.iterrows():
        # 1. Détermination de l'entropie selon le type
        h_token = "<H≈0.3>" if row['type'] == "Normalisation" else "<H≈1.5>"
        
        # 2. Construction de la trace de pensée (Thinking Trace)
        # Format: Forme → Ton ※ Mots-clés de l'input ∴ Type
        short_input = " ".join(row['input'].split()[:5]) + "..." # Extrait court pour la trace
        trace = f"{row['forme']} → {row['ton']} ※ {short_input} ∴ {row['type']}"
        
        # 3. Construction de l'instruction (User)
        instruction = f"Réécris ce brouillon. Forme : {row['forme']}. Ton : {row['ton']}. Support : {row['support']}."
        
        # 4. Formatage ChatML complet
        prompt = f"<|im_start|>user\n{instruction}\n\nBrouillon : {row['input']}<|im_end|>\n<|im_start|>assistant"
        response = f"<think>\n{trace}\n</think>\n{h_token} {row['output']}<|im_end|>"
        
        # Structure finale JSONL
        entry = {
            "text": f"{prompt}{response}"
        }
        jsonl_output.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    return jsonl_output.getvalue()

# --- DÉFINITION DES OPTIONS (Listes fermées) ---
# Mise à jour des types selon ta demande
LISTE_TYPES = ["Normalisation", "Normalisation & Expansion"]
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
        # Export CSV (Standard)
        csv = df[df['statut'] == "Fait et validé"].to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger CSV", csv, "dataset_brut.csv", "text/csv")
        
        # Export JSONL (Spécifique Baguettotron)
        jsonl_data = convert_to_baguettotron_jsonl(df)
        st.download_button(
            label="✨ Télécharger JSONL Baguettotron",
            data=jsonl_data,
            file_name=f"baguettotron_train_{datetime.now().strftime('%Y%m%d')}.jsonl",
            mime="application/jsonl"
        )
    
    st.info("Le format JSONL inclut les balises <think> et <H≈X.X> de PleIAs. L'export ne contient que les lignes 'Fait et validé'.")

# --- INTERFACE PRINCIPALE ---
st.title("✒️ Baguettotron Style Manager")

tab1, tab2 = st.tabs(["➕ Nouvelle Entrée", "📂 Gestion & Édition"])

# --- TAB 1 : FORMULAIRE D'AJOUT ---
with tab1:
    with st.form("ajout_form", clear_on_submit=True):
        st.subheader("Paramètres de Style")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val_type = st.selectbox("Type", LISTE_TYPES, help="Normalisation = Transcription simple | Expansion = Développement ou suite")
        with c2:
            val_forme = st.selectbox("Forme", LISTE_FORMES)
        with c3:
            val_ton = st.selectbox("Ton", LISTE_TONS)
        with c4:
            val_support = st.selectbox("Support", LISTE_SUPPORTS)
        
        st.divider()
        st.subheader("Contenu Littéraire")
        val_input = st.text_area("Brouillon Synthétique (Input)", placeholder="Note brute avec fautes...")
        val_output = st.text_area("Prose Développée (Output)", placeholder="Texte final dans votre style...")
        
        st.divider()
        c5, c6 = st.columns(2)
        with c5:
            val_statut = st.selectbox("Statut initial", LISTE_STATUTS)
        with c6:
            val_notes = st.text_input("Notes libres / Contexte")

        submit = st.form_submit_button("Enregistrer l'entrée")

        if submit:
            if val_input and val_output:
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
                updated_df = pd.concat([df, new_row], ignore_index=True)
                conn.update(data=updated_df)
                st.success("Entrée enregistrée !")
                st.rerun()
            else:
                st.error("L'input et l'output sont obligatoires.")

# --- TAB 2 : EDITION BI-DIRECTIONNELLE ---
with tab2:
    st.subheader("Base de données complète")
    
    # On s'assure que toutes les colonnes attendues existent pour éviter les erreurs d'affichage
    expected_cols = ["id", "date", "type", "forme", "ton", "support", "input", "output", "statut", "notes"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    # Configuration de l'éditeur de données corrigée
    edited_df = st.data_editor(
        df,
        num_rows="dynamic", 
        width="stretch", # Correction de l'erreur use_container_width
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
    )

    if st.button("💾 Sauvegarder les modifications"):
        try:
            conn.update(data=edited_df)
            st.success("Google Sheet mis à jour !")
        except Exception as e:
            st.error(f"Erreur de sauvegarde : {e}")
