"""
Composants UI Streamlit : sidebar, formulaire ajout, onglet édition (fragment + graphiques).
"""
from datetime import datetime
import uuid

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

from src.database import (
    CACHE_COLUMNS,
    STATUT_VALIDE,
    update_data,
    audit_rows_from_cache,
    avg_signature_from_cache,
    avg_trigrams_from_cache,
    dataset_cache_stats,
    flag_problematic_rows,
)
from src.export_utils import convert_to_baguettotron_jsonl
from src.nlp_engine import (
    corriger_texte_fr,
    get_linguistic_insights,
    get_stylometric_signature,
    get_pos_trigrams,
    get_baguette_touch,
    syntax_contrast_score,
    palier_details,
    normalize_signature,
    coherence_level,
    compute_coherence_score,
    prioritized_actions,
    translate_trigram,
    compute_row_cache,
    signature_variance,
)


@st.cache_resource
def load_nlp():
    """Charge le modèle spaCy français une seule fois (cache Streamlit)."""
    import logging
    _log = logging.getLogger(__name__)
    try:
        import spacy
        return spacy.load("fr_core_news_sm")
    except OSError as e:
        _log.warning("Modèle spaCy fr_core_news_sm non trouvé: %s", e)
        return None
    except (ValueError, ImportError, Exception) as e:
        _log.warning("spaCy non disponible sur cet environnement.")
        _log.debug("Détail: %s", e)
        return None


def render_sidebar(df, conn, listes):
    """Sidebar : stats, export CSV, export JSONL."""
    st.title("📊 Dataset Status")
    if not df.empty and "statut" in df.columns:
        st.write(df['statut'].value_counts())
        
    st.divider()
    st.subheader("🚀 Export Fine-tuning")
    if not df.empty:
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] .stDownloadButton button {
                min-height: 52px;
                width: 100%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        csv = df[df["statut"] == STATUT_VALIDE].to_csv(index=False).encode("utf-8")
        jsonl_data = convert_to_baguettotron_jsonl(df)
        col_csv, col_jsonl = st.columns(2)
        with col_csv:
            st.download_button("Télécharger CSV", csv, "dataset_brut.csv", "text/csv", key="dl_csv")
        with col_jsonl:
            st.download_button(
                label="Télécharger JSONL",
                data=jsonl_data,
                file_name=f"baguettotron_train_{datetime.now().strftime('%Y%m%d')}.jsonl",
                mime="application/jsonl",
                key="dl_jsonl",
                help="Format Baguettotron (ChatML, <think>, entropie)",
            )
        
    st.info("Le format JSONL inclut les balises <think> et <H≈X.X> de PleIAs. L'export ne contient que les lignes 'Fait et validé'.")


def _run_correction_ortho(output_text: str, pending_key: str) -> bool:
    """
    Lance la correction orthographique (LanguageTool) et stocke le résultat
    dans session_state[pending_key]. Affiche warning/error en cas de problème.

    Returns:
        True si la correction a été appliquée (le caller doit alors faire st.rerun()),
        False sinon.
    """
    if not (output_text and output_text.strip()):
        st.warning("Le champ Prose (Output) est vide. Saisis un texte à corriger.")
        return False
    try:
        corrected = corriger_texte_fr(output_text)
        st.session_state[pending_key] = corrected
        st.success("Orthographe et grammaire corrigées. Le champ Output a été mis à jour.")
        return True
    except requests.Timeout:
        st.error("Délai dépassé : le service LanguageTool n'a pas répondu. Réessaie dans un instant.")
        return False
    except requests.RequestException as e:
        st.error(f"Erreur réseau ou service indisponible : {e}")
        return False
    except (ValueError, Exception) as e:
        st.error(f"Impossible de corriger : {e}")
        return False


def _render_analyse_prose(
    input_text: str,
    output_text: str,
    type_value: str,
    nlp,
    df_valid: pd.DataFrame,
    verifier_flag_key: str = "verifier_clique",
) -> None:
    """Contenu réutilisable du bloc « Analyse de ta prose » (édition et nouvelle entrée)."""
    if nlp is None:
        if st.session_state.get(verifier_flag_key):
            st.warning(
                "Le modèle d'analyse n'a pas pu se charger "
                "(environnement ou dépendances). L'export et "
                "l'édition fonctionnent normalement."
            )
        else:
            st.info(
                "Clique sur « Vérifier ma prose » ci-dessus "
                "pour lancer l'analyse."
            )
        return
    stats = get_linguistic_insights(input_text, output_text, nlp)
    if not stats:
        st.info(
            "Remplis le Brouillon et la Prose pour voir "
            "l'analyse de ton écriture."
        )
        return
    st.caption(
        "Ces indicateurs t'aident à écrire une prose "
        "cohérente et à construire un dataset de qualité. "
        "L'objectif : que chaque fiche ait un style proche "
        "de la moyenne du dataset."
    )
    st.markdown("#### Ton écriture en un coup d'œil")
    c_st1, c_st2, c_st3 = st.columns(3)
    with c_st1:
        st.metric("Amplification", f"x{stats['ratio']:.1f}", help="Combien de fois ta prose est plus longue que le brouillon. x1 = identique, x2 = deux fois plus long.")
        st.caption(f"Brouillon : {stats['mots_in']} mots — Prose : {stats['mots_out']} mots")
    with c_st2:
        st.metric("Variété du vocabulaire", f"{stats['ttr']:.0%}", help="Pourcentage de mots différents (lemmes) dans ta prose.")
    with c_st3:
        st.metric("Longueur des phrases", f"{stats['long_moy_phrases']:.0f} mots", help="10-18 = rythme équilibré, <10 = vif, >25 = ample.")
    if stats["mots_repetes"]:
        mots_list = ", ".join(f"**{m}**" for m in stats["mots_repetes"][:8])
        suffix = "..." if len(stats["mots_repetes"]) > 8 else ""
        st.warning(f"Mots qui reviennent souvent (3 fois ou plus) : {mots_list}{suffix} — pense à varier.", icon="🔁")
    ratio_lvl, ratio_txt = palier_details("ratio", stats["ratio"])
    ttr_lvl, ttr_txt = palier_details("ttr", stats["ttr"])
    rythme_lvl, rythme_txt = palier_details("moy_phrases", stats["long_moy_phrases"])
    st.markdown("##### Ce que ça veut dire")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.info(f"**Amplification x{stats['ratio']:.1f}**\n\nNiveau : **{ratio_lvl}**\n\n{ratio_txt}")
    with f2:
        st.info(f"**Variété {stats['ttr']:.0%}**\n\nNiveau : **{ttr_lvl}**\n\n{ttr_txt}")
    with f3:
        st.info(f"**{stats['long_moy_phrases']:.0f} mots/phrase**\n\nNiveau : **{rythme_lvl}**\n\n{rythme_txt}")
    st.markdown("##### Grain littéraire (Baguette-Touch)")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("Mots vides (brouillon)", f"{stats.get('stop_ratio_in', 0):.0%}", help="Le brouillon doit rester brut (ratio bas).")
        st.metric("Mots vides (prose)", f"{stats.get('stop_ratio_out', 0):.0%}", help="Prose stylisée : ~40–50 % est normal.")
    with b2:
        bag = get_baguette_touch(output_text, nlp)
        if bag:
            pe = bag["punct_exp"]
            st.caption(f"Ponctuation expressive : — {pe['tiret_cadratin']}× · ... {pe['points_suspension']}× · : {pe['deux_points']}×")
    with b3:
        if bag and bag["weak_verbs"]:
            wv = ", ".join(f"{v} ({n}×)" for v, n in bag["weak_verbs"])
            st.warning(f"Verbes faibles : {wv}. Remplace par des verbes plus précis.", icon="✒️")
        elif bag:
            st.success("Peu de verbes faibles détectés.")
    contrast = syntax_contrast_score(input_text, output_text, nlp)
    st.metric("Contraste syntaxique (Input vs Output)", f"{contrast:.0%}", help="Élevé = ta prose transforme bien le brouillon.")
    if contrast < 0.2:
        st.warning("L'output ressemble trop à l'input. Varie les structures pour que le modèle apprenne.", icon="📐")
    sig_fiche = get_stylometric_signature(output_text, nlp)
    sig_dataset = avg_signature_from_cache(df_valid) if not df_valid.empty else None
    if sig_fiche and sig_dataset:
        st.markdown("#### Empreinte stylistique")
        st.caption("Ce radar compare ta fiche (bleu) à la moyenne du dataset (orange).")
        categories = list(sig_fiche.keys())
        v_fiche = [sig_fiche[k] for k in categories]
        v_dataset = [sig_dataset[k] for k in categories]
        r_fiche_norm = [normalize_signature(k, v) for k, v in zip(categories, v_fiche)]
        r_dataset_norm = [normalize_signature(k, v) for k, v in zip(categories, v_dataset)]
        theta = categories + [categories[0]]
        r_fiche = r_fiche_norm + [r_fiche_norm[0]]
        r_dataset = r_dataset_norm + [r_dataset_norm[0]]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=r_fiche, theta=theta, name="Ta fiche", fill="toself", line=dict(color="rgb(0,120,200)")))
        fig.add_trace(go.Scatterpolar(r=r_dataset, theta=theta, name="Moyenne dataset", fill="toself", line=dict(color="rgb(200,80,0)", dash="dash")))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Radar de signature stylistique", height=420)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Détails chiffrés du radar", expanded=False):
            df_comp = pd.DataFrame({"Axe": categories, "Ta fiche": [round(v, 3) for v in v_fiche], "Moy. dataset": [round(v, 3) for v in v_dataset], "Écart (%)": [round((v_fiche[i] - v_dataset[i]) / max(1e-6, v_dataset[i]) * 100, 1) for i in range(len(categories))]})
            st.dataframe(df_comp, width="stretch", hide_index=True)
        st.markdown("#### Tes constructions de phrases favorites")
        st.caption("Quels enchaînements grammaticaux reviennent le plus dans ta prose ?")
        tri_fiche = get_pos_trigrams(output_text, nlp)
        tri_dataset = avg_trigrams_from_cache(df_valid) if not df_valid.empty else None
        if tri_fiche:
            total_tri_fiche = sum(tri_fiche.values())
            top5 = tri_fiche.most_common(5)
            rows_tri = []
            for gram, count in top5:
                pct_fiche = count / max(1, total_tri_fiche) * 100
                pct_ds = (tri_dataset.get(gram, 0) / max(1, sum(tri_dataset.values())) * 100) if tri_dataset else None
                delta_pct = round(pct_fiche - (pct_ds or 0), 1)
                delta_str = f"+{delta_pct}" if delta_pct >= 0 else str(delta_pct)
                rows_tri.append({"Construction": translate_trigram(gram), "Occurrences": count, "Ta fiche (%)": f"{pct_fiche:.1f}", "Dataset (%)": f"{pct_ds:.1f}" if pct_ds is not None else "—", "Écart": delta_str})
            st.dataframe(pd.DataFrame(rows_tri), width="stretch", hide_index=True)
        else:
            st.caption("Texte trop court pour analyser les constructions de phrases.")
        score, deltas = compute_coherence_score(sig_fiche, sig_dataset, stats["mots_repetes"])
        level_label, tone = coherence_level(score)
        st.markdown("#### Cohérence avec le dataset")
        st.caption("Ce score mesure à quel point ta fiche ressemble au style moyen de ton dataset. 100 = parfaitement aligné.")
        c_score1, c_score2 = st.columns([1, 2])
        with c_score1:
            st.metric("Score", f"{score} / 100", help="100 = ta fiche est parfaitement alignée avec le style moyen du dataset.")
        with c_score2:
            _MSG = {"success": f"**{level_label}** — ton style est bien aligné avec le dataset. Continue comme ça !", "info": f"**{level_label}** — quelques petits écarts, rien de bloquant. Consulte les conseils.", "warning": f"**{level_label}** — ta fiche s'éloigne du style général. Lis les conseils ci-dessous.", "error": f"**{level_label}** — gros écart de style. Relis le texte et ajuste selon les conseils."}
            getattr(st, tone)(_MSG.get(tone, ""))
        actions = prioritized_actions(stats, deltas)
        if actions:
            st.markdown("#### Conseils pour cette fiche")
            for i, action in enumerate(actions, start=1):
                st.write(f"{i}. {action}")
        scores_trend = []
        for _, row in df_valid.tail(10).iterrows():
            sc = row.get("_coherence_score", "")
            if sc and str(sc).strip():
                try:
                    scores_trend.append(float(sc))
                except (ValueError, TypeError):
                    pass
        if len(scores_trend) >= 2:
            st.markdown("#### Évolution de la cohérence")
            fig_trend = go.Figure(go.Scatter(y=scores_trend, mode="lines+markers", line=dict(color="rgb(0,120,200)")))
            fig_trend.update_layout(height=180, margin=dict(t=20, b=20, l=40, r=20), yaxis=dict(range=[0, 100], title="Score"), xaxis=dict(title="Fiche (récente →)"), showlegend=False)
            st.plotly_chart(fig_trend, use_container_width=True)
    elif sig_fiche:
        st.caption("Ajoute des fiches « Fait et validé » pour débloquer le radar et la comparaison avec le dataset.")
    if type_value == "Expansion" and stats["ratio"] < 2:
        st.warning("Pour une fiche de type « Expansion », essaie de développer davantage le brouillon (au moins x2).", icon="💡")


_AJOUT_DEFAULTS = [
    ("ajout_in",      lambda listes: ""),
    ("ajout_out",     lambda listes: ""),
    ("ajout_notes",   lambda listes: ""),
    ("ajout_type",    lambda listes: listes["types"][0]),
    ("ajout_forme",   lambda listes: listes["formes"][0]),
    ("ajout_ton",     lambda listes: listes["tons"][0]),
    ("ajout_support", lambda listes: listes["supports"][0]),
    ("ajout_statut",  lambda listes: listes["statuts"][0]),
]
_AJOUT_KEYS = [k for k, _ in _AJOUT_DEFAULTS]


def render_tab_ajout(df, conn, listes):
    """Formulaire d'ajout d'une nouvelle entrée (mêmes outils que Gestion & Édition : correction ortho, vérification prose)."""
    for key, default_fn in _AJOUT_DEFAULTS:
        st.session_state.setdefault(key, default_fn(listes))
    st.session_state.setdefault("verifier_ajout_clique", False)

    def _idx(lst: list, val) -> int:
        try:
            return lst.index(val) if val in lst else 0
        except (ValueError, TypeError):
            return 0

    st.subheader("Paramètres de Style")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.selectbox("Type", listes["types"], index=_idx(listes["types"], st.session_state["ajout_type"]), key="ajout_type", help="Normalisation = Transcription simple | Expansion = Développement ou suite")
    with c2:
        st.selectbox("Forme", listes["formes"], index=_idx(listes["formes"], st.session_state["ajout_forme"]), key="ajout_forme")
    with c3:
        st.selectbox("Ton", listes["tons"], index=_idx(listes["tons"], st.session_state["ajout_ton"]), key="ajout_ton")
    with c4:
        st.selectbox("Support", listes["supports"], index=_idx(listes["supports"], st.session_state["ajout_support"]), key="ajout_support")

    st.divider()
    st.subheader("Contenu Littéraire")
    if "pending_correction_ajout" in st.session_state:
        st.session_state["ajout_out"] = st.session_state.pop("pending_correction_ajout")
    val_input = st.text_area("Brouillon Synthétique (Input)", value=st.session_state["ajout_in"], height=150, key="ajout_in", placeholder="Note brute avec fautes...")
    val_output = st.text_area("Prose Développée (Output)", value=st.session_state["ajout_out"], height=350, key="ajout_out", placeholder="Texte final dans votre style...")

    if st.button("🪄 Corriger l'orthographe", key="correct_ortho_ajout", help="Correction orthographe/grammaire (LanguageTool, français). Ne modifie pas le style."):
        if _run_correction_ortho(val_output, "pending_correction_ajout"):
            st.rerun()

    if st.button("🔍 Vérifier ma prose", key="verifier_ajout_btn", help="Lancer l'analyse linguistique (spaCy) sur le brouillon et la prose."):
        st.session_state["verifier_ajout_clique"] = True
        st.rerun()

    df_valid = df[df["statut"] == STATUT_VALIDE]
    nlp_ajout = load_nlp() if st.session_state["verifier_ajout_clique"] else None
    with st.expander("🔍 Analyse de ta prose", expanded=st.session_state["verifier_ajout_clique"]):
        _render_analyse_prose(
            st.session_state["ajout_in"],
            st.session_state["ajout_out"],
            st.session_state["ajout_type"],
            nlp_ajout,
            df_valid,
            verifier_flag_key="verifier_ajout_clique",
        )

    st.divider()
    c5, c6 = st.columns(2)
    with c5:
        st.selectbox("Statut initial", listes["statuts"], index=_idx(listes["statuts"], st.session_state["ajout_statut"]), key="ajout_statut")
    with c6:
        st.text_input("Notes libres / Contexte", value=st.session_state["ajout_notes"], key="ajout_notes")

    if st.button("💾 Enregistrer l'entrée", type="primary", key="submit_ajout"):
        val_input = st.session_state.get("ajout_in", "")
        val_output = st.session_state.get("ajout_out", "")
        if not (val_input and val_output):
            st.error("L'input et l'output sont obligatoires.")
        else:
            new_row = pd.DataFrame([{
                "id": str(uuid.uuid4())[:8],
                "date": datetime.now().strftime("%d/%m/%Y"),
                "type": st.session_state.get("ajout_type", listes["types"][0]),
                "forme": st.session_state.get("ajout_forme", listes["formes"][0]),
                "ton": st.session_state.get("ajout_ton", listes["tons"][0]),
                "support": st.session_state.get("ajout_support", listes["supports"][0]),
                "input": val_input,
                "output": val_output,
                "statut": st.session_state.get("ajout_statut", listes["statuts"][0]),
                "notes": st.session_state.get("ajout_notes", ""),
                **{c: "" for c in CACHE_COLUMNS},
            }])
            updated_df = pd.concat([df, new_row], ignore_index=True)
            update_data(conn, updated_df)
            for k in _AJOUT_KEYS:
                st.session_state.pop(k, None)
            st.session_state["verifier_ajout_clique"] = False
            st.success("Entrée enregistrée !")
            st.rerun()


def render_tab_edition(df, conn, listes):
    """Onglet Gestion & Édition : navigation, fragment édition + analyse."""
    if df.empty:
        st.warning("Le dataset est vide.")
    else:
        df_valid = df[df["statut"] == STATUT_VALIDE]
    
        # --- Barre de progression (fiches validées / total) ---
        total_fiches = len(df)
        validées = len(df_valid)
        if total_fiches > 0:
            st.progress(
                validées / total_fiches,
                text=f"📊 {validées} fiche(s) validée(s) / {total_fiches} total",
            )
    
        # --- Audit global : uniquement depuis le cache (pas de boucle spaCy) ---
        rows_audit = audit_rows_from_cache(df_valid) if not df_valid.empty else []
        if not df_valid.empty:
            with st.expander("📋 Résumé audit dataset (Fait et validé)", expanded=False):
                if rows_audit:
                    st.dataframe(pd.DataFrame(rows_audit), width="stretch")
                else:
                    st.info(
                        "Enregistre des fiches (bouton « Sauvegarder ») pour remplir "
                        "l'audit à partir des colonnes cache. Aucune analyse lourde au chargement."
                    )

        st.session_state.setdefault("verifier_clique", False)

        # 1. FILTRAGE
        st.subheader("🔍 Filtrer les fiches")
        filtre_statut = st.multiselect(
            "Statuts à afficher :", 
            listes["statuts"], 
            default=listes["statuts"]
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
            row_id = current_row["id"]
    
            st.divider()
    
            @st.fragment
            def _bloc_edition_et_analyse():
                nlp = load_nlp() if st.session_state["verifier_clique"] else None
                col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                try:
                    idx_type = listes["types"].index(current_row["type"])
                    idx_forme = listes["formes"].index(current_row["forme"])
                    idx_ton = listes["tons"].index(current_row["ton"])
                    idx_supp = listes["supports"].index(current_row["support"])
                    idx_statut = listes["statuts"].index(current_row["statut"])
                except (ValueError, KeyError):
                    idx_type = idx_forme = idx_ton = idx_supp = idx_statut = 0

                edit_type = col_e1.selectbox("Type", listes["types"], index=idx_type, key=f"type_{row_id}")
                edit_forme = col_e2.selectbox("Forme", listes["formes"], index=idx_forme, key=f"forme_{row_id}")
                edit_ton = col_e3.selectbox("Ton", listes["tons"], index=idx_ton, key=f"ton_{row_id}")
                edit_support = col_e4.selectbox("Support", listes["supports"], index=idx_supp, key=f"supp_{row_id}")
    
                edit_input = st.text_area("Brouillon (Input)", value=current_row["input"], height=150, key=f"in_{row_id}")
                # Appliquer une correction en attente avant d'instancier le widget (Streamlit interdit de modifier la clé après)
                pending_key = f"pending_correction_{row_id}"
                if pending_key in st.session_state:
                    st.session_state[f"out_{row_id}"] = st.session_state.pop(pending_key)
                edit_output = st.text_area("Prose (Output)", value=current_row["output"], height=350, key=f"out_{row_id}")

                if st.button("🪄 Corriger l'orthographe", key=f"correct_ortho_{row_id}", help="Correction orthographe/grammaire (LanguageTool, français). Ne modifie pas le style."):
                    if _run_correction_ortho(edit_output, pending_key):
                        st.rerun()

                if st.button("🔍 Vérifier ma prose", key=f"verifier_btn_{row_id}", type="secondary", help="Lancer l'analyse linguistique (spaCy) sur le brouillon et la prose."):
                    st.session_state["verifier_clique"] = True
                    st.rerun()

                col_e5, col_e6 = st.columns([1, 2])
                edit_statut = col_e5.selectbox("Statut", listes["statuts"], index=idx_statut, key=f"stat_{row_id}")
                edit_notes = col_e6.text_input("Notes libres", value=current_row["notes"], key=f"note_{row_id}")
    
                # --- PANNEAU DIAGNOSTICS LINGUISTIQUES ---
                st.divider()
                with st.expander("🔍 Analyse de ta prose", expanded=True):
                    _render_analyse_prose(edit_input, edit_output, edit_type, nlp, df_valid)
                # 5. SAUVEGARDE (met à jour le cache pour cette ligne si "Vérifier" a été cliqué)
                if st.button("💾 Enregistrer les modifications", type="primary", width="stretch"):
                    cols_main = [
                        "type", "forme", "ton", "support",
                        "input", "output", "statut", "notes",
                    ]
                    df.loc[df["id"] == row_id, cols_main] = [
                        edit_type, edit_forme, edit_ton, edit_support,
                        edit_input, edit_output, edit_statut, edit_notes,
                    ]
                    if st.session_state["verifier_clique"]:
                        nlp_save = load_nlp()
                        cache_vals = compute_row_cache(
                            edit_input,
                            edit_output,
                            nlp_save,
                            df_valid,
                            row_id,
                            CACHE_COLUMNS,
                            avg_signature_from_cache,
                        )
                        for col, val in cache_vals.items():
                            df.loc[df["id"] == row_id, col] = val
                    update_data(conn, df)
                    st.success(f"Fiche {row_id} mise à jour !")
                    st.rerun()
            _bloc_edition_et_analyse()


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 3 — TABLEAU DE BORD
# ─────────────────────────────────────────────────────────────────────────────

def _render_overview(df: pd.DataFrame, listes: dict) -> None:
    """Section 1 — Composition du dataset (métadonnées, aucun cache requis)."""
    df_valid = df[df["statut"] == STATUT_VALIDE]
    total = len(df)
    n_valide = len(df_valid)
    n_cours = len(df[df["statut"] == "En cours"])
    n_relire = len(df[df["statut"] == "A relire"])
    n_todo = len(df[df["statut"] == "A faire"])

    st.subheader("Composition du dataset")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total fiches", total)
    m2.metric("Validées", n_valide)
    m3.metric("En cours", n_cours)
    m4.metric("A relire", n_relire)
    m5.metric("A faire", n_todo)

    if total > 0:
        st.progress(n_valide / total, text=f"{n_valide}/{total} fiches validées ({n_valide/total:.0%})")

    if df.empty:
        return

    col_left, col_right = st.columns(2)

    with col_left:
        counts_statut = df["statut"].value_counts().reset_index()
        counts_statut.columns = ["Statut", "Nombre"]
        fig_statut = go.Figure(go.Bar(
            x=counts_statut["Nombre"],
            y=counts_statut["Statut"],
            orientation="h",
            text=counts_statut["Nombre"],
            textposition="auto",
        ))
        fig_statut.update_layout(
            title="Répartition par statut",
            height=220,
            margin=dict(t=40, b=20, l=10, r=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_statut, use_container_width=True)

    with col_right:
        counts_type = df["type"].value_counts().reset_index()
        counts_type.columns = ["Type", "Nombre"]
        fig_type = go.Figure(go.Bar(
            x=counts_type["Nombre"],
            y=counts_type["Type"],
            orientation="h",
            text=counts_type["Nombre"],
            textposition="auto",
        ))
        fig_type.update_layout(
            title="Répartition par type",
            height=220,
            margin=dict(t=40, b=20, l=10, r=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with st.expander("Détail formes / tons / supports", expanded=False):
        for dim, label in [("forme", "Formes"), ("ton", "Tons"), ("support", "Supports")]:
            counts = df[dim].value_counts().reset_index()
            counts.columns = [label, "Nombre"]
            fig = go.Figure(go.Bar(
                x=counts["Nombre"],
                y=counts[label],
                orientation="h",
                text=counts["Nombre"],
                textposition="auto",
            ))
            fig.update_layout(
                title=label,
                height=max(160, len(counts) * 30 + 60),
                margin=dict(t=40, b=10, l=10, r=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_quality_panel(df_valid: pd.DataFrame, stats: dict) -> None:
    """Section 2 — Qualité stylistique (lit uniquement le cache, pas de spaCy)."""
    st.subheader("Qualité stylistique")
    n = stats["n"]

    health = stats["health_score"]
    _, health_tone = coherence_level(health)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Score santé dataset", f"{health} / 100",
        help="0.4×cohérence + 0.3×TTR normalisé + 0.3×% fiches sans alerte",
    )
    m2.metric(
        "Cohérence moyenne",
        f"{stats['coherence']['mean']:.0f} / 100",
        help=f"Écart-type : {stats['coherence']['std']:.1f}",
    )
    m3.metric(
        "TTR moyen",
        f"{stats['ttr']['mean']:.0%}",
        help=f"Écart-type : {stats['ttr']['std']:.2f}",
    )
    m4.metric(
        "Ratio moyen",
        f"x{stats['ratio']['mean']:.1f}",
        help=f"Écart-type : {stats['ratio']['std']:.2f}",
    )

    # Histogrammes triples : ratio / TTR / longueur phrases
    c1, c2, c3 = st.columns(3)
    for col, key, label, xrange in [
        (c1, "ratio",   "Ratio d'amplification", None),
        (c2, "ttr",     "TTR (diversité vocabulaire)", [0, 1]),
        (c3, "phrases", "Moy. mots / phrase", None),
    ]:
        with col:
            fig = go.Figure(go.Histogram(x=stats[key]["values"], nbinsx=12))
            fig.update_layout(
                title=label,
                height=220,
                margin=dict(t=40, b=20, l=10, r=10),
                xaxis=dict(range=xrange) if xrange else {},
            )
            st.plotly_chart(fig, use_container_width=True)

    # Histogramme cohérence avec zones colorées
    scores = stats["coherence"]["values"]
    fig_coh = go.Figure()
    fig_coh.add_vrect(x0=0,  x1=45,  fillcolor="red",    opacity=0.08, line_width=0)
    fig_coh.add_vrect(x0=45, x1=65,  fillcolor="orange",  opacity=0.08, line_width=0)
    fig_coh.add_vrect(x0=65, x1=100, fillcolor="green",  opacity=0.08, line_width=0)
    fig_coh.add_trace(go.Histogram(x=scores, nbinsx=15, name="Cohérence"))
    fig_coh.add_vline(
        x=stats["coherence"]["mean"],
        line_dash="dash", line_color="white",
        annotation_text=f"Moy. {stats['coherence']['mean']:.0f}",
        annotation_position="top right",
    )
    fig_coh.update_layout(
        title=f"Distribution des scores de cohérence ({n} fiches)",
        height=280,
        margin=dict(t=40, b=20, l=10, r=10),
        xaxis=dict(range=[0, 100], title="Score"),
        yaxis=dict(title="Nb fiches"),
        showlegend=False,
    )
    st.plotly_chart(fig_coh, use_container_width=True)


def _render_stylometry_panel(df_valid: pd.DataFrame) -> None:
    """Section 3 — Stylométrie globale (radar moyen + variance + trigrammes + trend)."""
    st.subheader("Stylométrie globale")

    sig_avg = avg_signature_from_cache(df_valid)
    sig_std = signature_variance(df_valid)

    if sig_avg:
        categories = list(sig_avg.keys())
        r_avg = [normalize_signature(k, sig_avg[k]) for k in categories]

        col_radar, col_tri = st.columns([1, 1])

        with col_radar:
            fig_radar = go.Figure()
            theta = categories + [categories[0]]
            r_plot = r_avg + [r_avg[0]]

            # Bandes d'erreur si variance disponible
            if sig_std:
                r_upper = [
                    min(1.0, normalize_signature(k, sig_avg[k] + sig_std[k]))
                    for k in categories
                ]
                r_lower = [
                    max(0.0, normalize_signature(k, sig_avg[k] - sig_std[k]))
                    for k in categories
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_upper + [r_upper[0]],
                    theta=theta,
                    fill=None,
                    line=dict(color="rgba(0,120,200,0.2)", width=0),
                    showlegend=False,
                    name="Zone ±σ",
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_lower + [r_lower[0]],
                    theta=theta,
                    fill="tonext",
                    fillcolor="rgba(0,120,200,0.12)",
                    line=dict(color="rgba(0,120,200,0.2)", width=0),
                    showlegend=False,
                    name="Zone ±σ",
                ))

            fig_radar.add_trace(go.Scatterpolar(
                r=r_plot,
                theta=theta,
                name="Signature moyenne",
                fill="toself",
                line=dict(color="rgb(0,120,200)"),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title="Signature stylistique moyenne (zone = ±σ)",
                height=380,
                margin=dict(t=60, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            if sig_std:
                with st.expander("Dispersion par axe (écart-type)", expanded=False):
                    rows_var = [
                        {"Axe": k, "Moy.": round(sig_avg[k], 3), "σ": round(sig_std[k], 4)}
                        for k in categories
                    ]
                    st.dataframe(pd.DataFrame(rows_var), hide_index=True, width="stretch")

        with col_tri:
            tri_total = avg_trigrams_from_cache(df_valid)
            if tri_total:
                top15 = tri_total.most_common(15)
                labels = [translate_trigram(g) for g, _ in top15]
                values = [c for _, c in top15]
                fig_tri = go.Figure(go.Bar(
                    x=values[::-1],
                    y=labels[::-1],
                    orientation="h",
                    text=values[::-1],
                    textposition="auto",
                ))
                fig_tri.update_layout(
                    title="Top 15 constructions grammaticales (POS)",
                    height=380,
                    margin=dict(t=60, b=20, l=10, r=10),
                )
                st.plotly_chart(fig_tri, use_container_width=True)
            else:
                st.info("Aucun trigramme POS disponible. Enregistre des fiches en cliquant sur « Vérifier ma prose ».")
    else:
        st.info("Signature stylométrique non disponible. Clique sur « Vérifier ma prose » dans l'onglet Gestion & Édition pour remplir le cache.")

    # Évolution temporelle de la cohérence
    scores_trend: list[float] = []
    ids_trend: list[str] = []
    for _, row in df_valid.iterrows():
        sc = row.get("_coherence_score", "") or ""
        if not sc:
            continue
        try:
            scores_trend.append(float(sc))
            ids_trend.append(str(row.get("id", "")))
        except (ValueError, TypeError):
            continue

    if len(scores_trend) >= 3:
        st.markdown("#### Évolution de la cohérence (ordre de saisie)")
        st.caption("Chaque point représente une fiche validée dans l'ordre des lignes. Si la courbe descend, le style dérive.")
        fig_trend = go.Figure()
        fig_trend.add_hrect(y0=0,  y1=45,  fillcolor="red",   opacity=0.07, line_width=0)
        fig_trend.add_hrect(y0=45, y1=65,  fillcolor="orange", opacity=0.07, line_width=0)
        fig_trend.add_hrect(y0=65, y1=100, fillcolor="green", opacity=0.07, line_width=0)
        fig_trend.add_trace(go.Scatter(
            y=scores_trend,
            x=list(range(1, len(scores_trend) + 1)),
            mode="lines+markers",
            line=dict(color="rgb(0,120,200)"),
            hovertext=ids_trend,
            hovertemplate="Fiche %{hovertext}<br>Score : %{y}<extra></extra>",
        ))
        fig_trend.update_layout(
            height=240,
            margin=dict(t=20, b=20, l=40, r=20),
            yaxis=dict(range=[0, 100], title="Score cohérence"),
            xaxis=dict(title="Fiche (ordre dataset)"),
            showlegend=False,
        )
        st.plotly_chart(fig_trend, use_container_width=True)


def _render_alerts_panel(problematic: list[dict]) -> None:
    """Section 4 — Alertes qualité issues du cache."""
    st.subheader("Alertes qualité")
    if not problematic:
        st.success("Aucune fiche problématique détectée. Bon travail !")
        return

    # Résumé par type d'alerte
    from collections import Counter as _Counter
    alerte_counts: _Counter = _Counter()
    for row in problematic:
        for a in row["alertes"]:
            alerte_counts[a] += 1

    col_bar, col_info = st.columns([1, 2])
    with col_bar:
        labels = list(alerte_counts.keys())
        values = [alerte_counts[k] for k in labels]
        fig_al = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            text=values,
            textposition="auto",
        ))
        fig_al.update_layout(
            title="Alertes par type",
            height=max(160, len(labels) * 50 + 60),
            margin=dict(t=40, b=10, l=10, r=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_al, use_container_width=True)

    with col_info:
        st.caption(
            f"**{len(problematic)} fiche(s) concernée(s)** sur les fiches validées avec cache. "
            "Ouvre l'onglet Gestion & Édition pour corriger."
        )
        for alerte, count in alerte_counts.items():
            level_label, tone = coherence_level(0 if alerte == "Cohérence critique" else 50)
            st.markdown(f"- **{alerte}** : {count} fiche(s)")

    # Tableau détaillé
    rows_display = []
    for r in problematic:
        rows_display.append({
            "ID": r["id"],
            "Type": r["type"],
            "Forme": r["forme"],
            "Ton": r["ton"],
            "Alertes": " · ".join(r["alertes"]),
        })
    st.dataframe(
        pd.DataFrame(rows_display),
        hide_index=True,
        width="stretch",
        column_config={
            "ID": st.column_config.TextColumn(width="small"),
            "Alertes": st.column_config.TextColumn(width="large"),
        },
    )


def render_tab_dashboard(df: pd.DataFrame, listes: dict) -> None:
    """Onglet Tableau de bord : composition, qualité, stylométrie, alertes.

    Entièrement basé sur le cache — aucun appel spaCy ni LanguageTool.
    """
    df_valid = df[df["statut"] == STATUT_VALIDE]

    n_total = len(df)
    n_valid = len(df_valid)

    if n_total == 0:
        st.info("Le dataset est vide. Commence à ajouter des fiches dans l'onglet « Nouvelle Entrée ».")
        return

    # Section 1 — Composition (pas besoin du cache)
    _render_overview(df, listes)

    st.divider()

    if n_valid == 0:
        st.info("Aucune fiche validée. Passe des fiches au statut « Fait et validé » pour débloquer les indicateurs qualité et stylistiques.")
        return

    # Calcul unique des stats cache — passé aux sections qui en ont besoin
    stats = dataset_cache_stats(df_valid)

    if stats is None:
        st.warning(
            "Le cache n'est pas encore rempli. Ouvre l'onglet Gestion & Édition, "
            "clique « Vérifier ma prose » puis « Enregistrer les modifications » pour chaque fiche validée."
        )
        return

    # Section 2 — Qualité
    _render_quality_panel(df_valid, stats)

    st.divider()

    # Section 3 — Stylométrie
    _render_stylometry_panel(df_valid)

    st.divider()

    # Section 4 — Alertes (flag_problematic_rows réutilise le cache, déjà chargé dans dataset_cache_stats)
    problematic = flag_problematic_rows(df_valid)
    _render_alerts_panel(problematic)
