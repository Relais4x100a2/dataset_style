import logging
from collections import Counter, defaultdict

import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime
import uuid
import json
import io

logger = logging.getLogger(__name__)

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


# --- MODÈLE NLP (spaCy) ---
@st.cache_resource
def load_nlp():
    """
    Charge le modèle spaCy français une seule fois. Retourne None si absent ou
    en cas d'erreur (ex. incompatibilité binaire numpy/thinc sur Cloud).
    """
    try:
        import spacy
        return spacy.load("fr_core_news_sm")
    except OSError as e:
        logger.warning("Modèle spaCy fr_core_news_sm non trouvé: %s", e)
        return None
    except (ValueError, ImportError, Exception) as e:
        logger.warning("spaCy non disponible sur cet environnement.")
        logger.debug("Détail: %s", e)
        return None


def get_linguistic_insights(
    text_in: str, text_out: str, nlp, seuil_repetition: int = 3
) -> dict | None:
    """
    Analyse linguistique input/output : ratio d'expansion, entités perdues,
    richesse lexicale, TTR, mots répétés, longueur moyenne des phrases.
    Retourne None si nlp est None ou textes vides.
    """
    if nlp is None or not (text_in and text_out):
        return None
    doc_in = nlp(text_in)
    doc_out = nlp(text_out)
    tokens_in = [t for t in doc_in if not t.is_punct]
    tokens_out = [t for t in doc_out if not t.is_punct]
    len_in = len(tokens_in)
    len_out = len(tokens_out)
    ratio = len_out / max(1, len_in)

    ents_in = {ent.text.lower() for ent in doc_in.ents}
    ents_out = {ent.text.lower() for ent in doc_out.ents}
    perdues = ents_in - ents_out

    lemmes_out = {t.lemma_.lower() for t in doc_out if not t.is_punct}
    richesse = len(lemmes_out) / max(1, len_out)

    # TTR (Type-Token Ratio) et mots répétés (hors stop words : mots porteurs de sens)
    types_out = len(lemmes_out)
    ttr = types_out / max(1, len_out)
    comptage = Counter(
        t.lemma_.lower() for t in doc_out if not t.is_punct and not t.is_stop
    )
    mots_repetes = [lem for lem, n in comptage.items() if n >= seuil_repetition]

    # Longueur moyenne des phrases (en mots)
    sents = list(doc_out.sents)
    long_phrases = [len([t for t in s if not t.is_punct]) for s in sents]
    long_moy_phrases = sum(long_phrases) / max(1, len(long_phrases))

    return {
        "ratio": ratio,
        "perdues": perdues,
        "richesse": richesse,
        "mots_in": len_in,
        "mots_out": len_out,
        "ttr": ttr,
        "mots_repetes": mots_repetes,
        "long_moy_phrases": long_moy_phrases,
    }


@st.cache_data(ttl=300)
def compute_audit_global(data_json: str) -> list[dict]:
    """
    Calcule l'audit global sur les lignes validées. Mis en cache par contenu
    (recalcul uniquement si les données changent). TTL 5 min pour limiter la mémoire.
    """
    nlp = load_nlp()
    if nlp is None:
        return []
    df_audit = pd.read_json(data_json)
    rows_audit = []
    for _, row in df_audit.iterrows():
        ins = get_linguistic_insights(
            row.get("input", ""), row.get("output", ""), nlp
        )
        if ins is None:
            continue
        alertes = []
        if "Expansion" in str(row.get("type", "")) and ins["ratio"] < 2:
            alertes.append("Expansion faible")
        if ins["perdues"]:
            alertes.append("Entités perdues")
        if ins["ttr"] < 0.5 and ins["mots_out"] > 20:
            alertes.append("Répétitions")
        rows_audit.append({
            "id": row.get("id", ""),
            "type": row.get("type", ""),
            "ratio": round(ins["ratio"], 1),
            "richesse": f"{ins['richesse']:.0%}",
            "entités perdues": "oui" if ins["perdues"] else "non",
            "moy. mots/phrase": round(ins["long_moy_phrases"], 0),
            "TTR": round(ins["ttr"], 2),
            "alertes": " ; ".join(alertes) if alertes else "—",
        })
    return rows_audit


# --- FONCTION D'EXPORT BAGUETTOTRON (JSONL) ---
LABELS_NER_FR = {"LOC": "Lieux", "PER": "Personnes", "ORG": "Organisations", "MISC": "Autres", "GPE": "Lieux"}


def _trace_avec_ner(input_text: str, forme: str, ton: str, type_row: str, nlp) -> str:
    """Construit une trace de pensée enrichie avec les entités NER du brouillon."""
    if not input_text or nlp is None:
        return f"{forme} → {ton} ※ {' '.join(input_text.split()[:5])}... ∴ {type_row}"
    doc = nlp(input_text)
    entites_par_label: dict[str, set[str]] = defaultdict(set)
    for ent in doc.ents:
        label_fr = LABELS_NER_FR.get(ent.label_, ent.label_)
        entites_par_label[label_fr].add(ent.text)
    if not entites_par_label:
        return f"{forme} → {ton} ※ {' '.join(input_text.split()[:5])}... ∴ {type_row}"
    parties = [f"{label}: [{', '.join(sorted(ents))}]" for label, ents in sorted(entites_par_label.items())]
    return f"{forme} → {ton} ※ {' '.join(parties)} ∴ {type_row}"


def convert_to_baguettotron_jsonl(df: pd.DataFrame, nlp=None):
    jsonl_output = io.StringIO()
    df_valid = df[df['statut'] == "Fait et validé"]
    for _, row in df_valid.iterrows():
        h_token = "<H≈0.3>" if row['type'] == "Normalisation" else "<H≈1.5>"
        trace = _trace_avec_ner(
            row.get("input", ""), row["forme"], row["ton"], row["type"], nlp
        )
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
        # Export JSONL (trace enrichie NER si spaCy disponible)
        nlp_export = load_nlp()
        jsonl_data = convert_to_baguettotron_jsonl(df, nlp_export)
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

# --- TAB 2 : NAVIGATION & ÉDITION DE FICHES BI-DIRECTIONNELLE ---
with tab2:
    if df.empty:
        st.warning("Le dataset est vide.")
    else:
        nlp = load_nlp()

        # --- AUDIT GLOBAL (Fait et validé), mis en cache ---
        df_valid = df[df["statut"] == "Fait et validé"]
        if not df_valid.empty and nlp is not None:
            with st.expander("📋 Résumé audit dataset (Fait et validé)", expanded=False):
                data_key = df_valid[["id", "input", "output", "type"]].to_json()
                rows_audit = compute_audit_global(data_key)
                if rows_audit:
                    st.dataframe(pd.DataFrame(rows_audit), use_container_width=True)
                else:
                    st.info("Aucune fiche analysable (input/output vides).")
        elif not df_valid.empty and nlp is None:
            st.warning(
                "Fonctions linguistiques (spaCy) non disponibles sur cet environnement. "
                "L'audit et l'export JSONL utilisent la version sans NER."
            )

        # 1. FILTRAGE
        st.subheader("🔍 Filtrer les fiches")
        filtre_statut = st.multiselect(
            "Statuts à afficher :", 
            LISTE_STATUTS, 
            default=LISTE_STATUTS
        )
        
        df_view = df[df['statut'].isin(filtre_statut)].reset_index(drop=True)

        if df_view.empty:
            st.info("Aucune fiche trouvée.")
        else:
            # 2. NAVIGATION
            if 'index_fiche' not in st.session_state:
                st.session_state.index_fiche = 0
            
            # Ajustement de l'index si on filtre
            st.session_state.index_fiche = min(st.session_state.index_fiche, len(df_view) - 1)

            c_nav1, c_nav2, c_nav3 = st.columns([1, 2, 1])
            with c_nav1:
                if st.button("⬅️ Précédent") and st.session_state.index_fiche > 0:
                    st.session_state.index_fiche -= 1
                    st.rerun() # On force le rafraîchissement immédiat
            with c_nav2:
                st.markdown(f"<center><h3>Fiche {st.session_state.index_fiche + 1} / {len(df_view)}</h3></center>", unsafe_allow_html=True)
            with c_nav3:
                if st.button("Suivant ➡️") and st.session_state.index_fiche < len(df_view) - 1:
                    st.session_state.index_fiche += 1
                    st.rerun()

            # 3. RÉCUPÉRATION DE LA DONNÉE
            current_row = df_view.iloc[st.session_state.index_fiche]
            row_id = current_row['id'] # On utilise l'ID pour verrouiller le contenu

            st.divider()

            # 4. FORMULAIRE AVEC KEYS DYNAMIQUES
            # En ajoutant row_id à la key, Streamlit recharge le contenu à chaque changement
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            
            # On utilise .get() ou des index sécurisés
            try:
                idx_type = LISTE_TYPES.index(current_row['type'])
                idx_forme = LISTE_FORMES.index(current_row['forme'])
                idx_ton = LISTE_TONS.index(current_row['ton'])
                idx_supp = LISTE_SUPPORTS.index(current_row['support'])
                idx_statut = LISTE_STATUTS.index(current_row['statut'])
            except (ValueError, KeyError):
                idx_type = idx_forme = idx_ton = idx_supp = idx_statut = 0

            edit_type = col_e1.selectbox("Type", LISTE_TYPES, index=idx_type, key=f"type_{row_id}")
            edit_forme = col_e2.selectbox("Forme", LISTE_FORMES, index=idx_forme, key=f"forme_{row_id}")
            edit_ton = col_e3.selectbox("Ton", LISTE_TONS, index=idx_ton, key=f"ton_{row_id}")
            edit_support = col_e4.selectbox("Support", LISTE_SUPPORTS, index=idx_supp, key=f"supp_{row_id}")

            edit_input = st.text_area("Brouillon (Input)", value=current_row['input'], height=150, key=f"in_{row_id}")
            edit_output = st.text_area("Prose (Output)", value=current_row['output'], height=350, key=f"out_{row_id}")

            col_e5, col_e6 = st.columns([1, 2])
            edit_statut = col_e5.selectbox("Statut", LISTE_STATUTS, index=idx_statut, key=f"stat_{row_id}")
            edit_notes = col_e6.text_input("Notes libres", value=current_row['notes'], key=f"note_{row_id}")

            # --- PANNEAU DIAGNOSTICS LINGUISTIQUES ---
            st.divider()
            with st.expander("🔍 Diagnostics Linguistiques (spaCy)", expanded=True):
                if nlp is None:
                    st.warning(
                        "Fonctions linguistiques (spaCy) non disponibles. Export et diagnostics utilisent la version simple."
                    )
                else:
                    stats = get_linguistic_insights(edit_input, edit_output, nlp)
                    if stats:
                        c_st1, c_st2, c_st3 = st.columns(3)
                        with c_st1:
                            st.metric("Ratio d'expansion", f"x{stats['ratio']:.1f}")
                            st.caption(f"Brouillon : {stats['mots_in']} mots | Prose : {stats['mots_out']} mots")
                        with c_st2:
                            if stats["perdues"]:
                                st.error(f"⚠️ Oubli potentiel : {', '.join(stats['perdues'])}")
                            else:
                                st.success("✅ Entités préservées")
                        with c_st3:
                            st.metric("Diversité lexicale", f"{stats['richesse']:.0%}")
                            st.metric("Moy. mots/phrase", f"{stats['long_moy_phrases']:.0f}")

                        if stats["mots_repetes"]:
                            st.caption(f"Répétitions (≥3×) : {', '.join(stats['mots_repetes'][:10])}{'…' if len(stats['mots_repetes']) > 10 else ''}")
                        st.caption(f"TTR (types/tokens) : {stats['ttr']:.2f}")

                        if edit_type == "Normalisation & Expansion" and stats["ratio"] < 2:
                            st.warning(
                                "💡 Conseil : pour une « Expansion », essayez de développer davantage."
                            )
                    else:
                        st.info("Remplissez l’Input et l’Output pour voir l’analyse.")

            # 5. SAUVEGARDE
            if st.button("💾 Enregistrer les modifications", type="primary", use_container_width=True):
                # On met à jour le DF original
                df.loc[df['id'] == row_id, ['type', 'forme', 'ton', 'support', 'input', 'output', 'statut', 'notes']] = [
                    edit_type, edit_forme, edit_ton, edit_support, edit_input, edit_output, edit_statut, edit_notes
                ]
                conn.update(data=df)
                st.success(f"Fiche {row_id} mise à jour !")
