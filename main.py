import logging
from collections import Counter

import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import plotly.graph_objects as go
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
    Analyse linguistique input/output : ratio d'expansion, richesse lexicale,
    TTR, mots répétés, longueur moyenne des phrases.
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

    lemmes_out = {t.lemma_.lower() for t in doc_out if not t.is_punct}
    richesse = len(lemmes_out) / max(1, len_out)

    # TTR (Type-Token Ratio) et mots répétés (hors stop words)
    ttr = len(lemmes_out) / max(1, len_out)
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
        "richesse": richesse,
        "mots_in": len_in,
        "mots_out": len_out,
        "ttr": ttr,
        "mots_repetes": mots_repetes,
        "long_moy_phrases": long_moy_phrases,
    }


def _interpretation_palier(indicateur: str, value: float) -> str:
    """Retourne l'interprétation du palier correspondant à la valeur."""
    if indicateur == "ratio":
        if value < 1.3:
            return "Tu restes proche du brouillon."
        if value < 2.0:
            return "Tu développes."
        if value < 2.5:
            return "Tu as bien développé l'idée."
        return "Tu déploies beaucoup."
    if indicateur == "ttr":
        if value < 0.50:
            return "Vocabulaire répétitif."
        if value < 0.65:
            return "Vocabulaire correct."
        if value < 0.80:
            return "Vocabulaire soutenu."
        return "Vocabulaire très riche."
    if indicateur == "moy_phrases":
        if value < 10:
            return "Rythme vif, phrases courtes."
        if value < 18:
            return "Rythme équilibré."
        if value < 25:
            return "Rythme ample."
        return "Phrases très longues."
    return ""


def _palier_details(indicateur: str, value: float) -> tuple[str, str]:
    """Retourne (niveau, interpretation) pour un indicateur."""
    if indicateur == "ratio":
        if value < 1.3:
            return "Minimal", _interpretation_palier(indicateur, value)
        if value < 2.0:
            return "Progressif", _interpretation_palier(indicateur, value)
        if value < 2.5:
            return "Solide", _interpretation_palier(indicateur, value)
        return "Amplifié", _interpretation_palier(indicateur, value)
    if indicateur == "ttr":
        if value < 0.50:
            return "Bas", _interpretation_palier(indicateur, value)
        if value < 0.65:
            return "Intermédiaire", _interpretation_palier(indicateur, value)
        if value < 0.80:
            return "Élevé", _interpretation_palier(indicateur, value)
        return "Très élevé", _interpretation_palier(indicateur, value)
    if indicateur == "moy_phrases":
        if value < 10:
            return "Court", _interpretation_palier(indicateur, value)
        if value < 18:
            return "Équilibré", _interpretation_palier(indicateur, value)
        if value < 25:
            return "Ample", _interpretation_palier(indicateur, value)
        return "Très ample", _interpretation_palier(indicateur, value)
    return "—", ""


def _normalize_signature(indicateur: str, value: float) -> float:
    """Normalise la signature stylométrique sur une échelle [0,1] stable par axe."""
    bornes: dict[str, tuple[float, float]] = {
        "Noms & adjectifs": (0.0, 0.60),
        "Verbes d'action": (0.0, 0.40),
        "Nuances (adverbes)": (0.0, 0.30),
        "Ponctuation": (0.0, 0.35),
        "Longueur des mots": (3.0, 10.0),
        "Participes vs conjugués": (0.0, 1.0),
        "Déterminants définis": (0.0, 1.0),
    }
    mn, mx = bornes.get(indicateur, (0.0, 1.0))
    if mx - mn < 1e-6:
        return 0.0
    return min(1.0, max(0.0, (value - mn) / (mx - mn)))


def _coherence_level(score: int) -> tuple[str, str]:
    """Retourne (label, tonalité streamlit) pour un score de cohérence."""
    if score >= 80:
        return "Excellent", "success"
    if score >= 65:
        return "Bon", "info"
    if score >= 45:
        return "À surveiller", "warning"
    return "Critique", "error"


def _compute_coherence_score(
    sig_fiche: dict[str, float], sig_dataset: dict[str, float], mots_repetes: list[str]
) -> tuple[int, dict[str, float]]:
    """Calcule un score global de cohérence (0-100) à partir des écarts stylométriques."""
    deltas: dict[str, float] = {}
    for k in sig_fiche:
        nf = _normalize_signature(k, sig_fiche[k])
        nd = _normalize_signature(k, sig_dataset[k])
        deltas[k] = abs(nf - nd)
    avg_delta = sum(deltas.values()) / max(1, len(deltas))
    base_score = 100 * (1 - avg_delta)
    rep_penalty = min(20, max(0, len(mots_repetes) - 1) * 2)
    final_score = int(max(0, min(100, round(base_score - rep_penalty))))
    return final_score, deltas


def _prioritized_actions(
    stats: dict, deltas: dict[str, float], max_actions: int = 3
) -> list[str]:
    """Génère des conseils d'écriture concrets à partir des métriques courantes."""
    actions: list[str] = []
    if stats["ratio"] < 1.3:
        actions.append(
            "Ta prose reste très proche du brouillon — essaie d'ajouter des "
            "détails, des images ou des précisions pour enrichir le texte."
        )
    elif stats["ratio"] > 3.0:
        actions.append(
            "Tu développes beaucoup — vérifie que chaque ajout apporte du sens, "
            "sinon élague les passages redondants."
        )

    if stats["ttr"] < 0.50:
        actions.append(
            "Plusieurs mots reviennent souvent — cherche des synonymes ou "
            "reformule pour diversifier le vocabulaire."
        )
    elif stats["ttr"] > 0.85:
        actions.append(
            "Le vocabulaire est très varié — assure-toi que le registre reste "
            "cohérent d'une fiche à l'autre."
        )

    if stats["long_moy_phrases"] < 10:
        actions.append(
            "Tes phrases sont courtes — essaie d'en relier certaines pour "
            "obtenir un rythme plus fluide."
        )
    elif stats["long_moy_phrases"] > 25:
        actions.append(
            "Tes phrases sont longues — découpe-en quelques-unes pour "
            "faciliter la lecture."
        )

    if stats["mots_repetes"]:
        sample = ", ".join(f"« {m} »" for m in stats["mots_repetes"][:3])
        actions.append(
            f"Les mots {sample} reviennent souvent — remplace-en "
            "certains par des synonymes."
        )

    top_deltas = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:2]
    for axis, delta in top_deltas:
        if delta > 0.35:
            actions.append(
                f"Ta fiche s'écarte de la moyenne du dataset sur « {axis} » — "
                "rapproche-toi du style général pour garder la cohérence."
            )

    dedup = list(dict.fromkeys(actions))
    return dedup[:max_actions]


_POS_FR: dict[str, str] = {
    "ADJ": "Adj",
    "ADP": "Prép",
    "ADV": "Adv",
    "AUX": "Aux",
    "CCONJ": "Conj",
    "DET": "Dét",
    "INTJ": "Intj",
    "NOUN": "Nom",
    "NUM": "Num",
    "PART": "Part",
    "PRON": "Pron",
    "PROPN": "NomPr",
    "PUNCT": "Ponct",
    "SCONJ": "SubConj",
    "SYM": "Sym",
    "VERB": "Verbe",
    "X": "Autre",
}


def _translate_trigram(trigram: str) -> str:
    """Traduit un trigramme POS (ex. 'DET-NOUN-VERB') en français lisible."""
    return " · ".join(_POS_FR.get(tag, tag) for tag in trigram.split("-"))


def get_pos_trigrams(text: str, nlp) -> Counter | None:
    """Extrait les trigrammes POS d'un texte.

    Args:
        text: Texte à analyser.
        nlp: Pipeline spaCy chargé.

    Returns:
        Counter des trigrammes (ex. "DET-NOUN-VERB") ou None.
    """
    if nlp is None or not text:
        return None
    doc = nlp(text)
    tags = [t.pos_ for t in doc if not t.is_space]
    if len(tags) < 3:
        return None
    trigrams = [f"{tags[i]}-{tags[i + 1]}-{tags[i + 2]}" for i in range(len(tags) - 2)]
    return Counter(trigrams)


@st.cache_data(ttl=300)
def compute_avg_pos_trigrams(data_json: str) -> Counter | None:
    """Distribution agrégée des trigrammes POS sur les outputs validés."""
    nlp = load_nlp()
    if nlp is None:
        return None
    df_audit = pd.read_json(io.StringIO(data_json))
    total: Counter = Counter()
    for _, row in df_audit.iterrows():
        tri = get_pos_trigrams(str(row.get("output", "")), nlp)
        if tri:
            total += tri
    return total if total else None


def get_stylometric_signature(text: str, nlp) -> dict[str, float] | None:
    """Signature stylométrique (ADN stylistique).

    Ratios POS, ponctuation, longueur moyenne des mots, ratio participes /
    formes verbales, ratio déterminants définis / total déterminants.
    Retourne None si nlp absent ou texte vide.
    """
    if nlp is None or not text:
        return None
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    nb_tokens = max(1, len(tokens))
    counts = Counter(t.pos_ for t in doc)
    nb_punct = len([t for t in doc if t.is_punct])

    # Participes vs formes conjuguées (finies)
    verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]
    participes = sum(1 for t in verbs if "Part" in t.morph.get("VerbForm", []))
    conjugues = sum(1 for t in verbs if "Fin" in t.morph.get("VerbForm", []))
    ratio_part = participes / max(1, participes + conjugues)

    # Déterminants définis vs indéfinis
    dets = [t for t in doc if t.pos_ == "DET"]
    dets_def = sum(1 for t in dets if "Def" in t.morph.get("Definite", []))
    dets_indef = sum(1 for t in dets if "Ind" in t.morph.get("Definite", []))
    ratio_def = dets_def / max(1, dets_def + dets_indef)

    return {
        "Noms & adjectifs": (counts.get("NOUN", 0) + counts.get("ADJ", 0)) / nb_tokens,
        "Verbes d'action": counts.get("VERB", 0) / nb_tokens,
        "Nuances (adverbes)": counts.get("ADV", 0) / nb_tokens,
        "Ponctuation": nb_punct / nb_tokens,
        "Longueur des mots": sum(len(t.text) for t in tokens) / nb_tokens,
        "Participes vs conjugués": ratio_part,
        "Déterminants définis": ratio_def,
    }


@st.cache_data(ttl=300)
def compute_avg_stylometric_signature(data_json: str) -> dict[str, float] | None:
    """Moyenne des signatures stylométriques sur les outputs validés. Pour le radar de cohérence."""
    nlp = load_nlp()
    if nlp is None:
        return None
    df_audit = pd.read_json(io.StringIO(data_json))
    sigs = []
    for _, row in df_audit.iterrows():
        s = get_stylometric_signature(str(row.get("output", "")), nlp)
        if s:
            sigs.append(s)
    if not sigs:
        return None
    keys = list(sigs[0].keys())
    return {k: sum(s[k] for s in sigs) / len(sigs) for k in keys}


@st.cache_data(ttl=300)
def compute_audit_global(data_json: str) -> list[dict]:
    """
    Calcule l'audit global sur les lignes validées. Mis en cache par contenu
    (recalcul uniquement si les données changent). TTL 5 min pour limiter la mémoire.
    """
    nlp = load_nlp()
    if nlp is None:
        return []
    df_audit = pd.read_json(io.StringIO(data_json))
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
        if ins["ttr"] < 0.5 and ins["mots_out"] > 20:
            alertes.append("Répétitions")
        rows_audit.append({
            "id": row.get("id", ""),
            "type": row.get("type", ""),
            "ratio": round(ins["ratio"], 1),
            "richesse": f"{ins['richesse']:.0%}",
            "moy. mots/phrase": round(ins["long_moy_phrases"], 0),
            "TTR": round(ins["ttr"], 2),
            "alertes": " ; ".join(alertes) if alertes else "—",
        })
    return rows_audit


# --- FONCTION D'EXPORT BAGUETTOTRON (JSONL) ---
def convert_to_baguettotron_jsonl(df: pd.DataFrame):
    jsonl_output = io.StringIO()
    df_valid = df[df['statut'] == "Fait et validé"]
    for _, row in df_valid.iterrows():
        h_token = "<H≈0.3>" if row['type'] == "Normalisation" else "<H≈1.5>"
        short_input = " ".join(str(row.get("input", "")).split()[:5]) + "..."
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
LISTE_TYPES = ["Normalisation", "Expansion"]
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

# --- TAB 2 : NAVIGATION & ÉDITION DE FICHES BI-DIRECTIONNELLE ---
with tab2:
    if df.empty:
        st.warning("Le dataset est vide.")
    else:
        # Chargement spaCy à la demande pour éviter timeout/OOM au démarrage (Streamlit Cloud)
        if "spacy_activated" not in st.session_state:
            st.session_state.spacy_activated = False
        st.session_state.spacy_activated = st.checkbox(
            "Activer les diagnostics linguistiques (spaCy)",
            value=st.session_state.spacy_activated,
            help="Charge le modèle français (~50 Mo). À activer uniquement quand vous en avez besoin.",
        )
        nlp = load_nlp() if st.session_state.spacy_activated else None

        # --- AUDIT GLOBAL (Fait et validé), mis en cache ---
        df_valid = df[df["statut"] == "Fait et validé"]
        if not df_valid.empty and nlp is not None:
            with st.expander("📋 Résumé audit dataset (Fait et validé)", expanded=False):
                data_key = df_valid[["id", "input", "output", "type"]].to_json()
                rows_audit = compute_audit_global(data_key)
                if rows_audit:
                    st.dataframe(pd.DataFrame(rows_audit), width="stretch")
                else:
                    st.info("Aucune fiche analysable (input/output vides).")
        elif not df_valid.empty and nlp is None:
            st.info("Cochez « Activer les diagnostics linguistiques (spaCy) » ci‑dessus pour afficher l'audit et le radar.")

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
            with st.expander("🔍 Analyse de ta prose", expanded=True):
                if nlp is None:
                    if st.session_state.spacy_activated:
                        st.warning(
                            "Le modèle d'analyse n'a pas pu se charger "
                            "(environnement ou dépendances). L'export et "
                            "l'édition fonctionnent normalement."
                        )
                    else:
                        st.info(
                            "Cochez « Activer les diagnostics linguistiques » "
                            "en haut de l'onglet pour voir comment améliorer "
                            "ta prose."
                        )
                else:
                    stats = get_linguistic_insights(edit_input, edit_output, nlp)
                    if stats:
                        # Introduction pédagogique
                        st.caption(
                            "Ces indicateurs t'aident à écrire une prose "
                            "cohérente et à construire un dataset de qualité. "
                            "L'objectif : que chaque fiche ait un style proche "
                            "de la moyenne du dataset."
                        )

                        # ═══════════════════════════════════════
                        # SECTION 1 — Ton écriture en un coup d'œil
                        # ═══════════════════════════════════════
                        st.markdown("#### Ton écriture en un coup d'œil")
                        c_st1, c_st2, c_st3 = st.columns(3)
                        with c_st1:
                            st.metric(
                                "Amplification",
                                f"x{stats['ratio']:.1f}",
                                help=(
                                    "Combien de fois ta prose est plus longue "
                                    "que le brouillon. x1 = identique, "
                                    "x2 = deux fois plus long."
                                ),
                            )
                            st.caption(
                                f"Brouillon : {stats['mots_in']} mots — "
                                f"Prose : {stats['mots_out']} mots"
                            )
                        with c_st2:
                            st.metric(
                                "Variété du vocabulaire",
                                f"{stats['ttr']:.0%}",
                                help=(
                                    "Pourcentage de mots différents (lemmes) "
                                    "dans ta prose. Plus c'est haut, plus ton "
                                    "vocabulaire est riche."
                                ),
                            )
                        with c_st3:
                            st.metric(
                                "Longueur des phrases",
                                f"{stats['long_moy_phrases']:.0f} mots",
                                help=(
                                    "Nombre moyen de mots par phrase. "
                                    "10-18 = rythme équilibré, <10 = vif, "
                                    ">25 = ample."
                                ),
                            )

                        if stats["mots_repetes"]:
                            mots_list = ", ".join(
                                f"**{m}**"
                                for m in stats["mots_repetes"][:8]
                            )
                            suffix = (
                                "..." if len(stats["mots_repetes"]) > 8 else ""
                            )
                            st.warning(
                                f"Mots qui reviennent souvent (3 fois ou "
                                f"plus) : {mots_list}{suffix} — pense à "
                                "varier.",
                                icon="🔁",
                            )

                        # Interprétation — un verdict clair par indicateur
                        ratio_lvl, ratio_txt = _palier_details(
                            "ratio", stats["ratio"]
                        )
                        ttr_lvl, ttr_txt = _palier_details(
                            "ttr", stats["ttr"]
                        )
                        rythme_lvl, rythme_txt = _palier_details(
                            "moy_phrases", stats["long_moy_phrases"]
                        )

                        st.markdown("##### Ce que ça veut dire")
                        f1, f2, f3 = st.columns(3)
                        with f1:
                            st.info(
                                f"**Amplification x{stats['ratio']:.1f}**\n\n"
                                f"Niveau : **{ratio_lvl}**\n\n{ratio_txt}"
                            )
                        with f2:
                            st.info(
                                f"**Variété {stats['ttr']:.0%}**\n\n"
                                f"Niveau : **{ttr_lvl}**\n\n{ttr_txt}"
                            )
                        with f3:
                            st.info(
                                f"**{stats['long_moy_phrases']:.0f} "
                                f"mots/phrase**\n\n"
                                f"Niveau : **{rythme_lvl}**\n\n{rythme_txt}"
                            )

                        # ═══════════════════════════════════════
                        # SECTION 2 — Empreinte stylistique
                        # ═══════════════════════════════════════
                        sig_fiche = get_stylometric_signature(edit_output, nlp)
                        if not df_valid.empty:
                            data_key = df_valid[
                                ["id", "input", "output", "type"]
                            ].to_json()
                            sig_dataset = compute_avg_stylometric_signature(
                                data_key
                            )
                        else:
                            sig_dataset = None
                        if sig_fiche and sig_dataset:
                            st.markdown("#### Empreinte stylistique")
                            st.caption(
                                "Ce radar compare ta fiche (bleu) à la "
                                "moyenne du dataset (orange). Plus les "
                                "formes se superposent, plus ton style est "
                                "cohérent avec l'ensemble."
                            )
                            categories = list(sig_fiche.keys())
                            v_fiche = [sig_fiche[k] for k in categories]
                            v_dataset = [sig_dataset[k] for k in categories]
                            r_fiche_norm = [
                                _normalize_signature(k, v)
                                for k, v in zip(categories, v_fiche)
                            ]
                            r_dataset_norm = [
                                _normalize_signature(k, v)
                                for k, v in zip(categories, v_dataset)
                            ]
                            theta = categories + [categories[0]]
                            r_fiche = r_fiche_norm + [r_fiche_norm[0]]
                            r_dataset = r_dataset_norm + [r_dataset_norm[0]]
                            fig = go.Figure()
                            fig.add_trace(
                                go.Scatterpolar(
                                    r=r_fiche,
                                    theta=theta,
                                    name="Ta fiche",
                                    fill="toself",
                                    line=dict(color="rgb(0,120,200)"),
                                )
                            )
                            fig.add_trace(
                                go.Scatterpolar(
                                    r=r_dataset,
                                    theta=theta,
                                    name="Moyenne dataset",
                                    fill="toself",
                                    line=dict(
                                        color="rgb(200,80,0)", dash="dash"
                                    ),
                                )
                            )
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True, range=[0, 1]
                                    )
                                ),
                                showlegend=True,
                                title="Radar de signature stylistique",
                                height=420,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            with st.expander(
                                "Détails chiffrés du radar", expanded=False
                            ):
                                df_comp = pd.DataFrame(
                                    {
                                        "Axe": categories,
                                        "Ta fiche": [
                                            round(v, 3) for v in v_fiche
                                        ],
                                        "Moy. dataset": [
                                            round(v, 3) for v in v_dataset
                                        ],
                                        "Écart (%)": [
                                            round(
                                                (
                                                    (v_fiche[i] - v_dataset[i])
                                                    / max(1e-6, v_dataset[i])
                                                )
                                                * 100,
                                                1,
                                            )
                                            for i in range(len(categories))
                                        ],
                                    }
                                )
                                st.dataframe(
                                    df_comp,
                                    width="stretch",
                                    hide_index=True,
                                )

                            # ═══════════════════════════════════════
                            # SECTION 3 — Constructions de phrases
                            # ═══════════════════════════════════════
                            st.markdown(
                                "#### Tes constructions de phrases favorites"
                            )
                            st.caption(
                                "Quels enchaînements grammaticaux reviennent "
                                "le plus dans ta prose ? Par exemple "
                                "« Dét · Nom · Verbe » = un déterminant suivi "
                                "d'un nom puis d'un verbe."
                            )
                            tri_fiche = get_pos_trigrams(edit_output, nlp)
                            if not df_valid.empty:
                                tri_dataset = compute_avg_pos_trigrams(
                                    data_key
                                )
                            else:
                                tri_dataset = None
                            if tri_fiche:
                                total_tri_fiche = sum(tri_fiche.values())
                                top5 = tri_fiche.most_common(5)
                                rows_tri: list[dict] = []
                                for gram, count in top5:
                                    pct_fiche = (
                                        count / max(1, total_tri_fiche) * 100
                                    )
                                    if tri_dataset:
                                        total_tri_ds = sum(
                                            tri_dataset.values()
                                        )
                                        pct_ds = (
                                            tri_dataset.get(gram, 0)
                                            / max(1, total_tri_ds)
                                            * 100
                                        )
                                        delta_pct = round(
                                            pct_fiche - pct_ds, 1
                                        )
                                        delta_str = (
                                            f"+{delta_pct}"
                                            if delta_pct >= 0
                                            else str(delta_pct)
                                        )
                                    else:
                                        pct_ds = None
                                        delta_str = "—"
                                    rows_tri.append(
                                        {
                                            "Construction": _translate_trigram(
                                                gram
                                            ),
                                            "Occurrences": count,
                                            "Ta fiche (%)": f"{pct_fiche:.1f}",
                                            "Dataset (%)": (
                                                f"{pct_ds:.1f}"
                                                if pct_ds is not None
                                                else "—"
                                            ),
                                            "Écart": delta_str,
                                        }
                                    )
                                st.dataframe(
                                    pd.DataFrame(rows_tri),
                                    width="stretch",
                                    hide_index=True,
                                )
                                st.caption(
                                    "Écart = différence en points de "
                                    "pourcentage entre ta fiche et la "
                                    "moyenne du dataset. Un écart élevé "
                                    "signale un tic syntaxique."
                                )
                            else:
                                st.caption(
                                    "Texte trop court pour analyser les "
                                    "constructions de phrases."
                                )

                            # ═══════════════════════════════════════
                            # SECTION 4 — Score + conseils
                            # ═══════════════════════════════════════
                            score, deltas = _compute_coherence_score(
                                sig_fiche,
                                sig_dataset,
                                stats["mots_repetes"],
                            )
                            level_label, tone = _coherence_level(score)
                            st.markdown("#### Cohérence avec le dataset")
                            st.caption(
                                "Ce score mesure à quel point ta fiche "
                                "ressemble au style moyen de ton dataset. "
                                "100 = parfaitement aligné."
                            )
                            c_score1, c_score2 = st.columns([1, 2])
                            with c_score1:
                                st.metric(
                                    "Score",
                                    f"{score} / 100",
                                    help=(
                                        "100 = ta fiche est parfaitement "
                                        "alignée avec le style moyen du "
                                        "dataset."
                                    ),
                                )
                            with c_score2:
                                _MSG = {
                                    "success": (
                                        f"**{level_label}** — ton style "
                                        "est bien aligné avec le dataset. "
                                        "Continue comme ça !"
                                    ),
                                    "info": (
                                        f"**{level_label}** — quelques "
                                        "petits écarts, rien de bloquant. "
                                        "Consulte les conseils."
                                    ),
                                    "warning": (
                                        f"**{level_label}** — ta fiche "
                                        "s'éloigne du style général. "
                                        "Lis les conseils ci-dessous."
                                    ),
                                    "error": (
                                        f"**{level_label}** — gros écart "
                                        "de style. Relis le texte et "
                                        "ajuste selon les conseils."
                                    ),
                                }
                                getattr(st, tone)(
                                    _MSG.get(tone, "")
                                )

                            actions = _prioritized_actions(stats, deltas)
                            if actions:
                                st.markdown("#### Conseils pour cette fiche")
                                for i, action in enumerate(
                                    actions, start=1
                                ):
                                    st.write(f"{i}. {action}")
                        elif sig_fiche:
                            st.caption(
                                "Ajoute des fiches « Fait et validé » "
                                "pour débloquer le radar et la comparaison "
                                "avec le dataset."
                            )

                        if edit_type == "Expansion" and stats["ratio"] < 2:
                            st.warning(
                                "Pour une fiche de type « Expansion », "
                                "essaie de développer davantage le "
                                "brouillon (au moins x2).",
                                icon="💡",
                            )
                    else:
                        st.info(
                            "Remplis le Brouillon et la Prose pour voir "
                            "l'analyse de ton écriture."
                        )

            # 5. SAUVEGARDE
            if st.button("💾 Enregistrer les modifications", type="primary", width="stretch"):
                # On met à jour le DF original
                df.loc[df['id'] == row_id, ['type', 'forme', 'ton', 'support', 'input', 'output', 'statut', 'notes']] = [
                    edit_type, edit_forme, edit_ton, edit_support, edit_input, edit_output, edit_statut, edit_notes
                ]
                conn.update(data=df)
                st.success(f"Fiche {row_id} mise à jour !")
