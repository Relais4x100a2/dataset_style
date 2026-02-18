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
    update_data,
    audit_rows_from_cache,
    avg_signature_from_cache,
    avg_trigrams_from_cache,
)
from src.export_utils import convert_to_baguettotron_jsonl
from src.nlp_engine import (
    load_nlp,
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
)


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
        csv = df[df['statut'] == "Fait et validé"].to_csv(index=False).encode('utf-8')
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


def render_tab_ajout(df, conn, listes):
    """Formulaire d'ajout d'une nouvelle entrée."""
    with st.form("ajout_form", clear_on_submit=True):
        st.subheader("Paramètres de Style")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val_type = st.selectbox("Type", listes["types"], help="Normalisation = Transcription simple | Expansion = Développement ou suite")
        with c2:
            val_forme = st.selectbox("Forme", listes["formes"])
        with c3:
            val_ton = st.selectbox("Ton", listes["tons"])
        with c4:
            val_support = st.selectbox("Support", listes["supports"])
            
        st.divider()
        st.subheader("Contenu Littéraire")
        val_input = st.text_area("Brouillon Synthétique (Input)", placeholder="Note brute avec fautes...")
        val_output = st.text_area("Prose Développée (Output)", placeholder="Texte final dans votre style...")
            
        st.divider()
        c5, c6 = st.columns(2)
        with c5:
            val_statut = st.selectbox("Statut initial", listes["statuts"])
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
                    "notes": val_notes,
                    **{c: "" for c in CACHE_COLUMNS},
                }])
                updated_df = pd.concat([df, new_row], ignore_index=True)
                update_data(conn, updated_df)
                st.success("Entrée enregistrée !")
                st.rerun()
            else:
                st.error("L'input et l'output sont obligatoires.")


def render_tab_edition(df, conn, listes):
    """Onglet Gestion & Édition : navigation, fragment édition + analyse."""
    if df.empty:
        st.warning("Le dataset est vide.")
    else:
        df_valid = df[df["statut"] == "Fait et validé"]
    
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
        if rows_audit:
            with st.expander("📋 Résumé audit dataset (Fait et validé)", expanded=False):
                st.dataframe(pd.DataFrame(rows_audit), width="stretch")
        elif not df_valid.empty:
            with st.expander("📋 Résumé audit dataset (Fait et validé)", expanded=False):
                st.info(
                    "Enregistre des fiches (bouton « Sauvegarder ») pour remplir "
                    "l'audit à partir des colonnes cache. Aucune analyse lourde au chargement."
                )
    
        # --- Vérifier ma prose : charge spaCy uniquement au clic ---
        if "verifier_clique" not in st.session_state:
            st.session_state.verifier_clique = False
        col_verif, _ = st.columns([1, 3])
        with col_verif:
            if st.button("🔍 Vérifier ma prose", type="secondary"):
                st.session_state.verifier_clique = True
                st.rerun()
        st.caption(
            "Clique pour lancer l'analyse linguistique sur la fiche affichée. "
            "« Sauvegarder » enregistre en base et met à jour le cache."
        )
        nlp = load_nlp() if st.session_state.verifier_clique else None
    
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
                    if not (edit_output and edit_output.strip()):
                        st.warning("Le champ Prose (Output) est vide. Saisis un texte à corriger.")
                    else:
                        try:
                            corrected = corriger_texte_fr(edit_output)
                            st.session_state[pending_key] = corrected
                            st.success("Orthographe et grammaire corrigées. Le champ Output a été mis à jour.")
                            st.rerun()
                        except requests.Timeout:
                            st.error("Délai dépassé : le service LanguageTool n'a pas répondu. Réessaie dans un instant.")
                        except requests.RequestException as e:
                            st.error(f"Erreur réseau ou service indisponible : {e}")
                        except (ValueError, Exception) as e:
                            st.error(f"Impossible de corriger : {e}")

                col_e5, col_e6 = st.columns([1, 2])
                edit_statut = col_e5.selectbox("Statut", listes["statuts"], index=idx_statut, key=f"stat_{row_id}")
                edit_notes = col_e6.text_input("Notes libres", value=current_row["notes"], key=f"note_{row_id}")
    
                # --- PANNEAU DIAGNOSTICS LINGUISTIQUES ---
                st.divider()
                with st.expander("🔍 Analyse de ta prose", expanded=True):
                    if nlp is None:
                        if st.session_state.verifier_clique:
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
                            ratio_lvl, ratio_txt = palier_details(
                                "ratio", stats["ratio"]
                            )
                            ttr_lvl, ttr_txt = palier_details(
                                "ttr", stats["ttr"]
                            )
                            rythme_lvl, rythme_txt = palier_details(
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
    
                            # --- Baguette-Touch : stop-words, ponctuation, verbes faibles ---
                            st.markdown("##### Grain littéraire (Baguette-Touch)")
                            b1, b2, b3 = st.columns(3)
                            with b1:
                                st.metric(
                                    "Mots vides (brouillon)",
                                    f"{stats.get('stop_ratio_in', 0):.0%}",
                                    help="Le brouillon doit rester brut (ratio bas).",
                                )
                                st.metric(
                                    "Mots vides (prose)",
                                    f"{stats.get('stop_ratio_out', 0):.0%}",
                                    help="Prose stylisée : ~40–50 % est normal.",
                                )
                            with b2:
                                bag = get_baguette_touch(edit_output, nlp)
                                if bag:
                                    pe = bag["punct_exp"]
                                    st.caption(
                                        f"Ponctuation expressive : — {pe['tiret_cadratin']}× · "
                                        f"... {pe['points_suspension']}× · : {pe['deux_points']}×"
                                    )
                            with b3:
                                if bag and bag["weak_verbs"]:
                                    wv = ", ".join(
                                        f"{v} ({n}×)" for v, n in bag["weak_verbs"]
                                    )
                                    st.warning(
                                        f"Verbes faibles : {wv}. Remplace par des verbes plus précis.",
                                        icon="✒️",
                                    )
                                elif bag:
                                    st.success("Peu de verbes faibles détectés.")
    
                            # Score de contrastes (delta syntaxique Input / Output)
                            contrast = syntax_contrast_score(
                                edit_input, edit_output, nlp
                            )
                            st.metric(
                                "Contraste syntaxique (Input vs Output)",
                                f"{contrast:.0%}",
                                help="Élevé = ta prose transforme bien le brouillon. "
                                "Bas = structures trop proches.",
                            )
                            if contrast < 0.2:
                                st.warning(
                                    "L’output ressemble trop à l’input. "
                                    "Varie les structures pour que le modèle apprenne.",
                                    icon="📐",
                                )
    
                            # ═══════════════════════════════════════
                            # SECTION 2 — Empreinte stylistique
                            # ═══════════════════════════════════════
                            sig_fiche = get_stylometric_signature(edit_output, nlp)
                            sig_dataset = (
                                avg_signature_from_cache(df_valid)
                                if not df_valid.empty
                                else None
                            )
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
                                    normalize_signature(k, v)
                                    for k, v in zip(categories, v_fiche)
                                ]
                                r_dataset_norm = [
                                    normalize_signature(k, v)
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
                                tri_dataset = (
                                    avg_trigrams_from_cache(df_valid)
                                    if not df_valid.empty
                                    else None
                                )
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
                                                "Construction": translate_trigram(
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
                                score, deltas = compute_coherence_score(
                                    sig_fiche,
                                    sig_dataset,
                                    stats["mots_repetes"],
                                )
                                level_label, tone = coherence_level(score)
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
    
                                actions = prioritized_actions(stats, deltas)
                                if actions:
                                    st.markdown("#### Conseils pour cette fiche")
                                    for i, action in enumerate(
                                        actions, start=1
                                    ):
                                        st.write(f"{i}. {action}")
    
                                # Trend de cohérence (10 dernières fiches validées)
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
                                    st.caption(
                                        "Dernières fiches validées. Si la courbe descend, "
                                        "tu dérives peut-être du style."
                                    )
                                    fig_trend = go.Figure(
                                        go.Scatter(
                                            y=scores_trend,
                                            mode="lines+markers",
                                            line=dict(color="rgb(0,120,200)"),
                                        )
                                    )
                                    fig_trend.update_layout(
                                        height=180,
                                        margin=dict(t=20, b=20, l=40, r=20),
                                        yaxis=dict(
                                            range=[0, 100],
                                            title="Score",
                                        ),
                                        xaxis=dict(title="Fiche (récente →)"),
                                        showlegend=False,
                                    )
                                    st.plotly_chart(
                                        fig_trend, use_container_width=True
                                    )
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
                    if st.session_state.verifier_clique:
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
