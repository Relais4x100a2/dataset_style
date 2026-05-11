"""
UI multi-utilisateur / multi-projet.
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st
from sqlalchemy.engine import Engine

from src.auth import CurrentUser, create_invitation_link, logout, revoke_account_with_saga
from src.database import (
    CACHE_COLUMNS,
    STATUT_VALIDE,
    ProjectSettings,
    count_active_memberships,
    count_owned_projects,
    count_users_for_admin,
    create_project,
    delete_project_as_admin,
    detach_memberships_as_super_admin,
    get_project_settings,
    list_accounts_for_super_admin,
    list_projects_for_user,
    list_quarantined_deprovision_ops,
    list_recent_deprovision_ops,
    replay_quarantined_operation,
    require_role,
    update_project_entries,
    update_project_settings_as_admin,
)
from src.export_utils import ExportScope, convert_to_jsonl, dataframe_for_export
from src.flash_messages import schedule_post_rerun_flash
from src.llm_generate import generate_input_from_output, generate_output_from_input
from src.mailer import send_account_link_email
from src.nlp_engine import (
    RowNlpCacheResult,
    avg_signature_from_cache,
    coherence_level,
    coherence_score_bucket_table,
    compute_row_cache,
    corriger_texte_fr,
    curator_advices_after_save,
    dataframe_for_dashboard_scope,
    list_parsed_coherence_scores,
    mean_syntax_contrast_parsed,
    outliers_low_coherence_table,
    parse_persisted_coherence_score,
    row_nlp_feedback_bundle_after_persist,
    signature_variance,
)
from src.presets import (
    DIMENSION_KEYS,
    PRESETS,
    available_presets,
    dumps_custom_presets,
    dumps_dimensions_override,
    load_active_dimensions,
    preset_dimensions,
)
from src.project_entries_cache import (
    cached_load_project_entries,
    invalidate_project_entries_cache,
)

logger = logging.getLogger(__name__)

_SESSION_EDITION_LAST_ENTRY_ID = "edition_last_entry_id"


def _post_save_stylometric_session_key(project_id: str) -> str:
    """Clé session pour afficher le feedback stylométrique après ``st.rerun()``."""
    return f"post_save_stylometric_{project_id}"


@st.cache_resource(show_spinner="Chargement du modèle linguistique (spaCy)…")
def _load_fr_core_nlp():
    """Charge ``fr_core_news_sm`` ; ``None`` si indisponible.

    Grâce à ``@st.cache_resource``, un seul pipeline spaCy est conservé par processus
    Streamlit : les appels depuis plusieurs onglets ou actions réutilisent la même
    instance en mémoire.
    """
    try:
        import spacy

        return spacy.load("fr_core_news_sm")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Impossible de charger fr_core_news_sm: %s", exc)
        return None


def _ensure_cache_columns_on_df(df: pd.DataFrame) -> pd.DataFrame:
    """Garantit la présence des colonnes de cache NLP (DataFrame issu d'anciennes données)."""
    out = df.copy()
    for col in CACHE_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out


def _clear_post_save_stylometric_feedback(project_id: str) -> None:
    """Supprime le feedback stylométrique en session (échec de persistance ou reset)."""
    st.session_state.pop(_post_save_stylometric_session_key(project_id), None)


def _store_post_save_stylometric_feedback(project_id: str, pkg: RowNlpCacheResult) -> None:
    """Enregistre le feedback à afficher au prochain run (après ``st.rerun()``)."""
    advices = curator_advices_after_save(pkg.advice_stats, pkg.coherence_deltas)
    score = pkg.coherence_score
    ttr = (pkg.cache.get("_ttr") or "").strip() or "—"
    contrast = (pkg.cache.get("_syntax_contrast") or "").strip() or "—"
    if score is not None:
        level, tone = coherence_level(int(score))
    else:
        level, tone = ("Non calculé", "warning")
    st.session_state[_post_save_stylometric_session_key(project_id)] = {
        "score": score,
        "ttr": ttr,
        "contrast": contrast,
        "level": level,
        "tone": tone,
        "advices": advices,
    }


def _render_post_save_stylometric_feedback(project_id: str) -> None:
    """Affiche puis consomme le feedback stylométrique post-sauvegarde (session)."""
    key = _post_save_stylometric_session_key(project_id)
    payload = st.session_state.pop(key, None)
    if not payload:
        return
    st.markdown("#### Retour stylistique (ligne enregistrée)")
    m1, m2, m3 = st.columns(3)
    score = payload.get("score")
    with m1:
        if score is None:
            st.metric("Score de cohérence", "—")
        else:
            st.metric("Score de cohérence", f"{int(score)}/100")
    with m2:
        st.metric("TTR", str(payload.get("ttr", "—")))
    with m3:
        st.metric("Contraste syntaxique", str(payload.get("contrast", "—")))
    tone = str(payload.get("tone", "info"))
    level = str(payload.get("level", ""))
    if tone == "success":
        st.success(f"Qualité perçue : **{level}**")
    elif tone == "warning":
        st.warning(f"Qualité perçue : **{level}**")
    elif tone == "error":
        st.error(f"Qualité perçue : **{level}**")
    else:
        st.info(f"Qualité perçue : **{level}**")
    advices = payload.get("advices") or []
    if advices:
        st.info("\n\n".join(str(a) for a in advices[:3]), icon="💡")


def sync_edition_output_widget_state(
    session: MutableMapping[str, Any],
    entry_id: str,
    row_output: str,
) -> str:
    """Keep the generated-output draft in session for the selected entry.

    Resets from ``row_output`` when the user selects another entry; preserves
    an in-session draft (for example after orthographic correction) when the
    same entry remains selected.

    Args:
        session: Streamlit ``st.session_state`` or any mutable mapping (tests).
        entry_id: Stable entry identifier (``id`` column).
        row_output: Persisted output text for the row from the dataframe.

    Returns:
        Session state key for ``st.text_area(..., key=...)`` for generated text.
    """
    widget_key = f"edit_output_{entry_id}"
    if session.get(_SESSION_EDITION_LAST_ENTRY_ID) != entry_id:
        session[_SESSION_EDITION_LAST_ENTRY_ID] = entry_id
        session[widget_key] = row_output
    elif widget_key not in session:
        session[widget_key] = row_output
    return widget_key


def new_entry_session_keys(project_id: str) -> dict[str, str]:
    """Build stable ``session_state`` keys for the « Nouvelle entrée » tab.

    Keys are scoped by ``project_id`` so drafts from one project never leak
    into another when the user switches context.

    Args:
        project_id: Active project identifier.

    Returns:
        Mapping of logical field names to session keys (``input``, ``output``,
        dimension keys, ``statut``, ``notes``).
    """
    prefix = f"new_entry_{project_id}"
    return {
        "input": f"{prefix}_input",
        "output": f"{prefix}_output",
        "type": f"{prefix}_type",
        "structure": f"{prefix}_structure",
        "ton": f"{prefix}_ton",
        "format": f"{prefix}_format",
        "public": f"{prefix}_public",
        "statut": f"{prefix}_statut",
        "notes": f"{prefix}_notes",
    }


def ensure_new_entry_widget_keys_initialized(
    session: MutableMapping[str, Any],
    project_id: str,
    dimensions: dict[str, list[str]],
) -> dict[str, str]:
    """Ensure new-entry widget keys exist with safe defaults.

    If a stored select value is no longer in the active preset options (e.g.
    after a dimensions change), it is reset to the first available option.

    Args:
        session: Streamlit ``st.session_state`` or any mutable mapping (tests).
        project_id: Active project identifier.
        dimensions: Preset dimension lists (``types``, ``structures``, …).

    Returns:
        The same mapping as :func:`new_entry_session_keys`.
    """
    keys = new_entry_session_keys(project_id)
    dim_pairs: tuple[tuple[str, str], ...] = (
        ("type", "types"),
        ("structure", "structures"),
        ("ton", "tons"),
        ("format", "formats"),
        ("public", "publics"),
        ("statut", "statuts"),
    )
    for short, dim_list_key in dim_pairs:
        key = keys[short]
        options = [str(x) for x in (dimensions.get(dim_list_key) or []) if str(x).strip()]
        if key not in session:
            session[key] = options[0] if options else ""
        elif options and str(session[key]) not in options:
            session[key] = options[0]
    if keys["input"] not in session:
        session[keys["input"]] = ""
    if keys["output"] not in session:
        session[keys["output"]] = ""
    if keys["notes"] not in session:
        session[keys["notes"]] = ""
    return keys


def new_entry_missing_required_body_message(input_text: str, output_text: str) -> str | None:
    """Validate that both body fields are non-empty before persisting.

    Args:
        input_text: Draft (brouillon) content.
        output_text: Generated text content.

    Returns:
        A short French error message if validation fails, otherwise ``None``.
    """
    if not str(input_text).strip() or not str(output_text).strip():
        return "Brouillon/Texte généré obligatoires."
    return None


def _current_project_id() -> str:
    return str(st.session_state.get("project_id") or "")


def _show_action_error(prefix: str, exc: Exception) -> None:
    if isinstance(exc, PermissionError):
        st.error(str(exc))
    else:
        logger.exception("Erreur inattendue (%s)", prefix, exc_info=exc)
        st.error(f"{prefix}: erreur inattendue.")


def _safe_index(options: list[str], value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _select_with_legacy(
    label: str,
    options: list[str],
    current_value: str,
    key: str,
    disabled: bool = False,
    show_warning: bool = True,
) -> str:
    clean_options = [str(item) for item in options if str(item).strip()]
    current = str(current_value or "").strip()
    label_to_value = {item: item for item in clean_options}
    display_options = list(clean_options)
    index = _safe_index(clean_options, current) if clean_options else 0

    if current and current not in label_to_value:
        legacy_label = f"[obsolète] {current}"
        display_options.insert(0, legacy_label)
        label_to_value[legacy_label] = current
        index = 0
        if show_warning:
            st.warning(
                "Cette valeur existe dans vos données mais plus dans le preset actif.",
                icon="⚠️",
            )

    if not display_options:
        st.error(f"{label}: aucune option disponible.")
        return current

    selected_label = st.selectbox(
        label,
        display_options,
        index=index,
        key=key,
        disabled=disabled,
    )
    return label_to_value[selected_label]


def _persist_settings(
    user: CurrentUser,
    engine: Engine,
    project_id: str,
    settings: ProjectSettings,
    success_message: str,
) -> None:
    try:
        update_project_settings_as_admin(engine, project_id, user.user_id, settings)
        st.success(success_message)
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        _show_action_error("Mise à jour réglages impossible", exc)


def _render_project_create_form(
    user: CurrentUser,
    engine: Engine,
    *,
    key_prefix: str,
    label: str = "Nouveau projet",
) -> None:
    """
    Formulaire de création projet.

    Contrat: gère tout le flux (messages + rerun), sans valeur de retour.
    """
    with st.form(f"{key_prefix}_create_project_form"):
        pname = st.text_input(label, key=f"{key_prefix}_new_project_name_input")
        submit = st.form_submit_button("Créer", key=f"{key_prefix}_create_project_submit")
    if not submit:
        return
    if not pname.strip():
        st.error("Nom du projet requis.")
        return
    try:
        pid = create_project(engine, user.user_id, pname.strip())
        st.session_state["project_id"] = pid
        st.success("Projet créé.")
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        _show_action_error("Création projet impossible", exc)


def _render_project_settings_form(
    user: CurrentUser,
    engine: Engine,
    project_id: str,
    role: str,
    *,
    key_prefix: str,
) -> None:
    """
    Formulaire de réglages projet.

    Contrat: gère tout le flux (messages + rerun), sans valeur de retour.
    """
    settings = get_project_settings(engine, project_id)
    disabled = role != "admin"
    with st.form(f"{key_prefix}_project_settings_form"):
        llm_base_url = st.text_input(
            "LLM base URL",
            value=settings.llm_base_url,
            disabled=disabled,
            key=f"{key_prefix}_settings_llm_base_url_input",
        )
        llm_model = st.text_input(
            "LLM model",
            value=settings.llm_model,
            disabled=disabled,
            key=f"{key_prefix}_settings_llm_model_input",
        )
        llm_api_key = st.text_input(
            "LLM API key",
            value=settings.llm_api_key,
            type="password",
            disabled=disabled,
            key=f"{key_prefix}_settings_llm_api_key_input",
        )
        llm_timeout_seconds = st.text_input(
            "LLM timeout (s)",
            value=settings.llm_timeout_seconds,
            disabled=disabled,
            key=f"{key_prefix}_settings_llm_timeout_input",
        )
        languagetool_base_url = st.text_input(
            "LanguageTool base URL",
            value=settings.languagetool_base_url,
            disabled=disabled,
            key=f"{key_prefix}_settings_languagetool_base_url_input",
        )
        save = st.form_submit_button(
            "Enregistrer réglages",
            disabled=disabled,
            key=f"{key_prefix}_settings_save_submit",
        )
    if save:
        next_settings = replace(
            settings,
            llm_base_url=llm_base_url.strip(),
            llm_model=llm_model.strip(),
            llm_api_key=llm_api_key.strip(),
            llm_timeout_seconds=llm_timeout_seconds.strip(),
            languagetool_base_url=languagetool_base_url.strip(),
        )
        _persist_settings(user, engine, project_id, next_settings, "Réglages projet enregistrés.")
    if disabled:
        st.info("Seul un admin peut modifier les réglages projet.")


def _render_single_dimension_editor(
    user: CurrentUser,
    engine: Engine,
    project_id: str,
    settings: ProjectSettings,
    dimensions: dict[str, list[str]],
    dim_key: str,
    label: str,
    *,
    key_prefix: str,
) -> None:
    values = list(dimensions.get(dim_key, []))
    st.caption(label)
    if values:
        st.write(" · ".join(values))
    else:
        st.write("—")

    new_value = st.text_input(
        f"Ajouter une valeur ({label})",
        key=f"{key_prefix}_{dim_key}_add_input",
    )
    if st.button("Ajouter", key=f"{key_prefix}_{dim_key}_add_btn"):
        candidate = new_value.strip()
        if not candidate:
            st.error("Valeur vide.")
            return
        if candidate in values:
            st.info("Valeur déjà présente.")
            return
        next_dimensions = deepcopy(dimensions)
        next_dimensions[dim_key] = values + [candidate]
        next_settings = replace(
            settings,
            dimensions_override_json=dumps_dimensions_override(next_dimensions),
        )
        _persist_settings(user, engine, project_id, next_settings, f"{label}: valeur ajoutée.")

    to_remove = st.multiselect(
        f"Supprimer des valeurs ({label})",
        values,
        key=f"{key_prefix}_{dim_key}_remove_select",
    )
    if st.button("Supprimer la sélection", key=f"{key_prefix}_{dim_key}_remove_btn"):
        if not to_remove:
            st.info("Aucune valeur sélectionnée.")
            return
        next_dimensions = deepcopy(dimensions)
        next_dimensions[dim_key] = [item for item in values if item not in set(to_remove)]
        if dim_key == "statuts" and not next_dimensions[dim_key]:
            st.error("La liste des statuts ne peut pas être vide.")
            return
        next_settings = replace(
            settings,
            dimensions_override_json=dumps_dimensions_override(next_dimensions),
        )
        _persist_settings(user, engine, project_id, next_settings, f"{label}: valeurs supprimées.")


def _render_dimensions_section(
    user: CurrentUser,
    engine: Engine,
    project_id: str,
    role: str,
    *,
    key_prefix: str,
) -> None:
    st.markdown("### Dimensions du texte")
    st.caption("Ces dimensions sont enregistrées pour ce projet uniquement.")
    settings = get_project_settings(engine, project_id)
    active_key, custom_presets, dimensions = load_active_dimensions(settings)
    presets_map = available_presets(custom_presets)
    preset_keys = list(presets_map.keys())
    selected_key = st.selectbox(
        "Preset",
        preset_keys,
        index=_safe_index(preset_keys, active_key),
        format_func=lambda k: str(presets_map[k].get("label") or k),
        key=f"{key_prefix}_preset_select",
        disabled=role != "admin",
    )

    if role != "admin":
        st.info("Seul un admin peut modifier les dimensions.")
        return

    if st.button("Charger le preset", key=f"{key_prefix}_preset_apply_btn"):
        target_dims = preset_dimensions(presets_map[selected_key])
        next_settings = replace(
            settings,
            active_preset_key=selected_key,
            dimensions_override_json=dumps_dimensions_override(target_dims),
        )
        _persist_settings(user, engine, project_id, next_settings, "Preset appliqué au projet.")

    if st.button("Réinitialiser", key=f"{key_prefix}_preset_reset_btn"):
        target_dims = preset_dimensions(presets_map[selected_key])
        next_settings = replace(
            settings,
            active_preset_key=selected_key,
            dimensions_override_json=dumps_dimensions_override(target_dims),
        )
        _persist_settings(user, engine, project_id, next_settings, "Dimensions réinitialisées.")

    settings = get_project_settings(engine, project_id)
    active_key, custom_presets, dimensions = load_active_dimensions(settings)

    labels = {
        "types": "Type de transformation",
        "structures": "Structure textuelle",
        "tons": "Tonalité textuelle",
        "formats": "Format de sortie",
        "publics": "Public cible",
        "statuts": "Statut",
    }
    for dim_key in DIMENSION_KEYS:
        _render_single_dimension_editor(
            user,
            engine,
            project_id,
            settings,
            dimensions,
            dim_key,
            labels[dim_key],
            key_prefix=key_prefix,
        )
        st.divider()

    st.markdown("#### Enregistrer comme preset")
    custom_name = st.text_input("Nom du preset", key=f"{key_prefix}_custom_preset_name_input")
    custom_label = st.text_input("Label du preset", key=f"{key_prefix}_custom_preset_label_input")
    if st.button("Enregistrer comme preset", key=f"{key_prefix}_custom_preset_save_btn"):
        preset_key = custom_name.strip().lower().replace(" ", "_")
        if not preset_key:
            st.error("Nom de preset requis.")
            return
        if preset_key in PRESETS:
            st.error("Ce nom est réservé par un preset par défaut.")
            return
        saved_label = custom_label.strip() or preset_key
        updated_custom = deepcopy(custom_presets)
        updated_custom[preset_key] = {"label": saved_label, **dimensions}
        next_settings = replace(
            settings,
            active_preset_key=preset_key,
            custom_presets_json=dumps_custom_presets(updated_custom),
            dimensions_override_json=dumps_dimensions_override(dimensions),
        )
        _persist_settings(
            user, engine, project_id, next_settings, "Preset personnalisé enregistré."
        )


def _render_project_delete_guarded_form(
    user: CurrentUser,
    engine: Engine,
    project_id: str,
    project_name: str,
    role: str,
    *,
    key_prefix: str,
) -> None:
    """
    Formulaire de suppression projet avec double confirmation.

    Contrat: gère tout le flux (messages + rerun), sans valeur de retour.
    """
    if role != "admin":
        st.info("Seul un admin peut supprimer le projet.")
        return

    confirm = st.checkbox(
        "Je confirme vouloir supprimer ce projet",
        key=f"{key_prefix}_proj_delete_confirm_checkbox",
    )
    typed_name = st.text_input(
        f"Retape le nom du projet ({project_name}) pour confirmer",
        key=f"{key_prefix}_proj_delete_confirm_text",
    )
    if st.button(
        "Supprimer ce projet",
        key=f"{key_prefix}_proj_delete_btn",
        type="secondary",
        disabled=not confirm,
    ):
        if typed_name.strip() != project_name:
            st.error("Nom du projet incorrect. Suppression annulée.")
            return
        try:
            delete_project_as_admin(engine, project_id, user.user_id)
            st.session_state.pop("project_id", None)
            schedule_post_rerun_flash(st.session_state, "Projet supprimé.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            _show_action_error("Suppression projet impossible", exc)


def render_sidebar(
    user: CurrentUser,
    engine: Engine,
) -> tuple[str, str]:
    """Sidebar minimale: compte, projet courant, rôle."""
    st.title("👤 Compte")
    badge = " · super admin" if user.is_super_admin else ""
    st.caption(f"{user.display_name} · {user.email}{badge}")
    if st.button("Se déconnecter", key="sb_logout_btn"):
        logout()
        st.rerun()

    st.divider()
    st.subheader("Projet courant")
    projects = list_projects_for_user(engine, user.user_id)
    if not projects:
        st.warning("Aucun projet. Crée le premier projet.")
        _render_project_create_form(
            user,
            engine,
            key_prefix="sb_first",
            label="Nom du projet",
        )
        return "", ""

    options = {f"{p.name} ({p.role})": p for p in projects}
    labels = list(options.keys())
    current_pid = _current_project_id()
    idx = 0
    for i, p in enumerate(projects):
        if p.project_id == current_pid:
            idx = i
            break
    chosen_label = st.selectbox("Projet", labels, index=idx, key="sb_project_select")
    chosen = options[chosen_label]
    st.session_state["project_id"] = chosen.project_id
    st.session_state["project_role"] = chosen.role
    st.caption(f"Rôle: {chosen.role}")
    return chosen.project_id, chosen.role


def render_tab_projects(
    user: CurrentUser,
    role: str,
    project_id: str,
    engine: Engine,
) -> None:
    """Gestion des projets (1 projet = 1 utilisateur)."""
    st.subheader("Projets")
    projects = list_projects_for_user(engine, user.user_id)
    current = next((p for p in projects if p.project_id == project_id), None)
    if current is None:
        st.error("Projet introuvable. Sélectionne un projet valide.")
        return

    st.markdown("### Projet")
    st.text_input("Projet courant", value=current.name, disabled=True, key="proj_current_name")
    _render_project_create_form(user, engine, key_prefix="proj_tab")

    st.markdown("### Zone sensible")
    _render_project_delete_guarded_form(
        user,
        engine,
        project_id,
        current.name,
        role,
        key_prefix="proj_tab",
    )


def _saga_max_retries() -> int:
    raw = (os.environ.get("ACCOUNT_SAGA_MAX_RETRIES") or "5").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 5
    return max(1, min(value, 20))


def render_tab_account(user: CurrentUser, engine: Engine) -> None:
    """Espace personnel: suppression de son propre compte."""
    st.subheader("Mon compte")
    st.caption("Suppression possible uniquement sans projet propriétaire ni membership active.")
    owned_count = count_owned_projects(engine, user.user_id)
    membership_count = count_active_memberships(engine, user.user_id)
    st.write(f"Projets possédés: **{owned_count}**")
    st.write(f"Memberships actives: **{membership_count}**")

    confirm = st.checkbox(
        "Je confirme vouloir supprimer mon compte", key="account_self_delete_confirm"
    )
    typed_email = st.text_input(
        f"Retape ton email ({user.email}) pour confirmer",
        key="account_self_delete_email_input",
    )
    if st.button(
        "Supprimer mon compte",
        type="secondary",
        disabled=not confirm,
        key="account_self_delete_btn",
    ):
        if typed_email.strip().lower() != user.email.strip().lower():
            st.error("Email de confirmation invalide.")
            return
        op_id = f"op_{uuid.uuid4().hex[:20]}"
        try:
            revoke_account_with_saga(
                engine,
                actor_user_id=user.user_id,
                target_user_id=user.user_id,
                operation_id=op_id,
                max_retries=_saga_max_retries(),
                detach_memberships=False,
            )
            schedule_post_rerun_flash(st.session_state, "Compte supprimé.")
            logout()
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            _show_action_error("Suppression de compte impossible", exc)


def render_tab_super_admin(user: CurrentUser, engine: Engine) -> None:
    """Administration globale des comptes."""
    st.subheader("Super Admin")
    if not user.is_super_admin:
        st.info("Accès réservé aux super admins.")
        return

    st.markdown("### Inviter un utilisateur")
    with st.form("super_admin_invite_form"):
        invite_email = st.text_input("Email utilisateur", key="sa_invite_email_input")
        invite_submit = st.form_submit_button("Envoyer l'invitation")
    if invite_submit:
        try:
            invite_link = create_invitation_link(engine, user.user_id, invite_email)
            delivery = send_account_link_email(
                to_email=invite_email.strip().lower(),
                subject="Invitation Dataset Style Studio",
                intro="Tu as été invité. Clique sur le lien pour définir ton mot de passe.",
                link=invite_link,
            )
            if delivery.mode == "smtp":
                st.success("Invitation envoyée par email.")
            else:
                st.warning("Mode dev: partage le lien affiché au destinataire.")
                st.code(delivery.preview)
        except Exception as exc:  # noqa: BLE001
            _show_action_error("Invitation impossible", exc)

    st.markdown("### Comptes")
    page_size = st.selectbox(
        "Taille de page",
        [10, 25, 50, 100],
        index=1,
        key="sa_page_size_select",
    )
    total_users = count_users_for_admin(engine)
    total_pages = max(1, math.ceil(total_users / page_size))
    page_idx = st.number_input(
        f"Page (1-{total_pages})",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key="sa_page_number_input",
    )
    offset = (int(page_idx) - 1) * int(page_size)
    rows = list_accounts_for_super_admin(
        engine,
        user.user_id,
        limit=int(page_size),
        offset=int(offset),
    )
    accounts_df = pd.DataFrame(
        [
            {
                "user_id": row.user_id,
                "email": row.email,
                "super_admin": row.is_super_admin,
                "nb_projets": row.project_count,
                "derniere_connexion": row.last_login_at or "—",
                "entrees_total": row.entries_total,
                "entrees_validees": row.entries_validated,
            }
            for row in rows
        ]
    )
    if accounts_df.empty:
        st.info("Aucun compte actif.")
    else:
        st.dataframe(accounts_df, hide_index=True, width="stretch")

    st.markdown("### Opérations compte")
    if not rows:
        return
    choices = {f"{row.email} ({row.user_id})": row for row in rows}
    selected_label = st.selectbox("Compte cible", list(choices.keys()), key="sa_target_select")
    target = choices[selected_label]
    owner_count = count_owned_projects(engine, target.user_id)
    membership_count = count_active_memberships(engine, target.user_id)
    st.caption(f"Bloquants suppression: projets={owner_count}, memberships={membership_count}")

    if membership_count > 0:
        st.warning(
            f"Action destructive: retirer {membership_count} membership(s) du compte {target.email}.",
            icon="⚠️",
        )
        detach_confirm = st.checkbox(
            "Je confirme le detach complet des memberships",
            key="sa_detach_confirm_checkbox",
        )
        detach_typed_email = st.text_input(
            f"Retape l'email cible ({target.email})",
            key="sa_detach_email_confirm_input",
        )
        if st.button(
            "Detach memberships",
            key="sa_detach_memberships_btn",
            type="secondary",
            disabled=not detach_confirm,
        ):
            if detach_typed_email.strip().lower() != target.email.strip().lower():
                st.error("Email de confirmation invalide.")
                return
            try:
                removed = detach_memberships_as_super_admin(engine, user.user_id, target.user_id)
                schedule_post_rerun_flash(st.session_state, f"Memberships détachées: {removed}")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                _show_action_error("Detach memberships impossible", exc)

    confirm_delete = st.checkbox(
        "Je confirme vouloir supprimer ce compte",
        key="sa_delete_confirm_checkbox",
    )
    if st.button(
        "Supprimer le compte",
        type="secondary",
        disabled=not confirm_delete,
        key="sa_delete_account_btn",
    ):
        op_id = f"op_{uuid.uuid4().hex[:20]}"
        try:
            revoke_account_with_saga(
                engine,
                actor_user_id=user.user_id,
                target_user_id=target.user_id,
                operation_id=op_id,
                max_retries=_saga_max_retries(),
                detach_memberships=False,
            )
            schedule_post_rerun_flash(st.session_state, "Compte supprimé.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            _show_action_error("Suppression compte impossible", exc)

    st.markdown("### Monitoring saga comptes")
    monitor_rows = list_recent_deprovision_ops(engine, user.user_id, limit=100)
    if monitor_rows:
        monitor_df = pd.DataFrame(
            [
                {
                    "operation_id": row.operation_id,
                    "target_user_id": row.target_user_id,
                    "state": row.state,
                    "retry_count": row.retry_count,
                    "next_retry_at": row.next_retry_at or "—",
                    "quarantined_at": row.quarantined_at or "—",
                    "last_error": row.last_error[:120],
                }
                for row in monitor_rows
            ]
        )
        state_counts = monitor_df["state"].value_counts().to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pending", int(state_counts.get("pending", 0)))
        c2.metric("Provider done", int(state_counts.get("provider_done", 0)))
        c3.metric("Failed", int(state_counts.get("failed", 0)))
        c4.metric("Quarantined", int(state_counts.get("quarantined", 0)))
        st.dataframe(monitor_df, hide_index=True, width="stretch")
    else:
        st.info("Aucune opération de saga récente.")

    st.markdown("### DLQ (quarantaine)")
    dlq_rows = list_quarantined_deprovision_ops(engine, user.user_id, limit=50)
    if not dlq_rows:
        st.info("Aucune opération en quarantaine.")
        return
    dlq_df = pd.DataFrame(
        [
            {
                "operation_id": row.operation_id,
                "target_user_id": row.target_user_id,
                "retry_count": row.retry_count,
                "quarantined_at": row.quarantined_at or "—",
                "last_error": row.last_error[:180],
            }
            for row in dlq_rows
        ]
    )
    st.dataframe(dlq_df, hide_index=True, width="stretch")
    selected_op = st.selectbox(
        "Opération DLQ",
        [row.operation_id for row in dlq_rows],
        key="sa_dlq_operation_select",
    )
    replay_confirm = st.checkbox(
        "Je confirme le replay de l'opération DLQ",
        key="sa_dlq_replay_confirm_checkbox",
    )
    if st.button(
        "Replay opération",
        key="sa_dlq_replay_btn",
        disabled=not replay_confirm,
    ):
        try:
            replay_quarantined_operation(engine, user.user_id, selected_op)
            schedule_post_rerun_flash(st.session_state, "Opération remise en file d'attente.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            _show_action_error("Replay DLQ impossible", exc)


def render_tab_settings_export(
    user: CurrentUser,
    role: str,
    project_id: str,
    df: pd.DataFrame,
    engine: Engine,
) -> None:
    """Réglages projet + export dataset."""
    st.subheader("Réglages & Export")
    st.markdown("### Réglages projet")
    _render_project_settings_form(user, engine, project_id, role, key_prefix="settings_tab")
    _render_dimensions_section(user, engine, project_id, role, key_prefix="dims_tab")

    st.markdown("### Export")
    if df.empty:
        st.info("Aucune donnée à exporter.")
        return

    _scope_labels: dict[ExportScope, str] = {
        "validated_only": "Validées seulement",
        "full_dataset": "Tout le dataset",
    }
    export_scope: ExportScope = st.radio(
        "Périmètre d'export",
        ("validated_only", "full_dataset"),
        format_func=lambda k: _scope_labels[k],
        key="export_scope_radio",
        help=(
            "Choisissez si les téléchargements incluent uniquement les fiches "
            "« Fait et validé » ou l'ensemble des lignes du projet."
        ),
    )
    st.caption(
        "« Tout le dataset » inclut les brouillons et les fiches dont le statut "
        "n'est pas « Fait et validé » (par ex. « À faire »), pas seulement les entrées finalisées."
    )

    df_export = dataframe_for_export(df, export_scope)
    csv = df_export.to_csv(index=False).encode("utf-8")
    export_format = st.selectbox(
        "Format JSONL",
        ["lfm2", "baguettotron", "mistral"],
        key="export_format_select",
    )
    jsonl_data = convert_to_jsonl(
        df,
        export_format,
        include_stylometry=True,
        scope=export_scope,
    )
    st.download_button(
        "Télécharger CSV",
        csv,
        "dataset.csv",
        "text/csv",
        key="export_csv_download_btn",
    )
    st.download_button(
        "Télécharger JSONL",
        jsonl_data,
        f"dataset_{export_format}.jsonl",
        "application/jsonl",
        key="export_jsonl_download_btn",
    )


def _llm_env(settings: ProjectSettings) -> None:
    pairs = {
        "LLM_BASE_URL": settings.llm_base_url,
        "LLM_MODEL": settings.llm_model,
        "LLM_API_KEY": settings.llm_api_key,
        "LLM_TIMEOUT_SECONDS": settings.llm_timeout_seconds,
        "LANGUAGETOOL_BASE_URL": settings.languagetool_base_url,
    }
    for key, value in pairs.items():
        if value:
            os.environ[key] = value


def render_tab_ajout(
    user: CurrentUser,
    role: str,
    project_id: str,
    project_settings: ProjectSettings,
    df: pd.DataFrame,
    engine: Engine,
    dimensions: dict[str, list[str]],
) -> None:
    """Ajout d'entrée (collaborator/admin).

    Draft and generated text are bound to ``st.session_state`` (per project) so
    LLM generation updates the same buffers that « Enregistrer » persists,
    without the anti-pattern of mixing ``st.form`` submit with out-of-band
    writes to unrelated session keys.
    """
    st.subheader("Nouvelle entrée")
    _render_post_save_stylometric_feedback(project_id)
    if role == "viewer":
        st.info("Lecture seule (viewer).")
        return
    _llm_env(project_settings)
    # Legacy keys from the old form+generation split; drop to avoid stale reads.
    st.session_state.pop("new_generated_output", None)
    st.session_state.pop("new_generated_input", None)

    keys = ensure_new_entry_widget_keys_initialized(st.session_state, project_id, dimensions)

    type_ = st.selectbox("Type de transformation", dimensions["types"], key=keys["type"])
    structure = st.selectbox("Structure textuelle", dimensions["structures"], key=keys["structure"])
    ton = st.selectbox("Tonalité textuelle", dimensions["tons"], key=keys["ton"])
    format_ = st.selectbox("Format de sortie", dimensions["formats"], key=keys["format"])
    public = st.selectbox("Public cible", dimensions["publics"], key=keys["public"])

    st.text_area("Brouillon", height=120, key=keys["input"])
    st.text_area("Texte généré", height=220, key=keys["output"])

    statut = st.selectbox("Statut", dimensions["statuts"], key=keys["statut"])
    notes = st.text_input("Notes", key=keys["notes"])

    col1, col2, col3 = st.columns(3)
    gen_out = col1.button("Générer texte", key=f"{keys['input']}_btn_gen_out")
    gen_in = col2.button("Générer brouillon", key=f"{keys['output']}_btn_gen_in")
    save = col3.button("Enregistrer", type="primary", key=f"{keys['input']}_btn_save")

    input_text = str(st.session_state.get(keys["input"], ""))
    output_text = str(st.session_state.get(keys["output"], ""))

    if gen_out and input_text.strip():
        try:
            with st.spinner("Génération en cours..."):
                generated = generate_output_from_input(
                    api_key=project_settings.llm_api_key,
                    input_text=input_text,
                    type_=type_,
                    structure=structure,
                    ton=ton,
                    format_=format_,
                    public=public,
                    model=project_settings.llm_model or None,
                )
            if generated:
                st.session_state[keys["output"]] = generated
                st.toast("Texte généré.")
                # Rerun so text widgets render after state update (same-run widget
                # ordering would otherwise keep stale output on screen).
                st.rerun()
            else:
                st.error("La génération a échoué. Vérifiez vos paramètres LLM puis réessayez.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur génération texte", exc_info=exc)
            st.error("Génération impossible: erreur inattendue côté service.")
    if gen_in and output_text.strip():
        try:
            with st.spinner("Génération en cours..."):
                generated = generate_input_from_output(
                    api_key=project_settings.llm_api_key,
                    output=output_text,
                    type_=type_,
                    structure=structure,
                    ton=ton,
                    format_=format_,
                    public=public,
                    model=project_settings.llm_model or None,
                )
            if generated:
                st.session_state[keys["input"]] = generated
                st.toast("Brouillon généré.")
                st.rerun()
            else:
                st.error("La génération a échoué. Vérifiez vos paramètres LLM puis réessayez.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur génération brouillon", exc_info=exc)
            st.error("Génération impossible: erreur inattendue côté service.")
    if save:
        input_save = str(st.session_state.get(keys["input"], ""))
        output_save = str(st.session_state.get(keys["output"], ""))
        body_err = new_entry_missing_required_body_message(input_save, output_save)
        if body_err:
            st.error(body_err)
            return
        type_save = str(st.session_state.get(keys["type"], type_))
        structure_save = str(st.session_state.get(keys["structure"], structure))
        ton_save = str(st.session_state.get(keys["ton"], ton))
        format_save = str(st.session_state.get(keys["format"], format_))
        public_save = str(st.session_state.get(keys["public"], public))
        statut_save = str(st.session_state.get(keys["statut"], statut))
        notes_save = str(st.session_state.get(keys["notes"], notes))
        new_row = pd.DataFrame(
            [
                {
                    "id": str(uuid.uuid4())[:8],
                    "project_id": project_id,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "type": type_save,
                    "structure": structure_save,
                    "ton": ton_save,
                    "format": format_save,
                    "public": public_save,
                    "input": input_save,
                    "output": output_save,
                    "statut": statut_save,
                    "notes": notes_save,
                    **{c: "" for c in CACHE_COLUMNS},
                }
            ]
        )
        require_role(engine, project_id, user.user_id, ("admin", "collaborator"))
        df_base = _ensure_cache_columns_on_df(df)
        new_row = _ensure_cache_columns_on_df(new_row)
        combined = pd.concat([df_base, new_row], ignore_index=True)
        row_id = str(new_row.iloc[0]["id"])
        nlp_model = _load_fr_core_nlp()
        pkg = compute_row_cache(
            input_save,
            output_save,
            nlp_model,
            combined,
            row_id,
            CACHE_COLUMNS,
            avg_signature_from_cache,
        )
        for col, val in pkg.cache.items():
            new_row.at[0, col] = val
        to_persist = pd.concat([df_base, new_row], ignore_index=True)
        try:
            update_project_entries(engine, project_id, to_persist, user.user_id)
            invalidate_project_entries_cache()
            df_loaded = cached_load_project_entries(engine, project_id, user.user_id)
            fb = row_nlp_feedback_bundle_after_persist(df_loaded, row_id, nlp_model, CACHE_COLUMNS)
            _store_post_save_stylometric_feedback(project_id, fb)
        except Exception as exc:  # noqa: BLE001
            _clear_post_save_stylometric_feedback(project_id)
            _show_action_error("Enregistrement impossible", exc)
            return
        st.session_state[keys["input"]] = ""
        st.session_state[keys["output"]] = ""
        st.session_state[keys["notes"]] = ""
        st.success("Entrée enregistrée.")
        st.rerun()


def render_tab_edition(
    user: CurrentUser,
    role: str,
    project_id: str,
    project_settings: ProjectSettings,
    df: pd.DataFrame,
    engine: Engine,
    dimensions: dict[str, list[str]],
) -> None:
    """Edition des entrées du projet."""
    st.subheader("Gestion & édition")
    _render_post_save_stylometric_feedback(project_id)
    _llm_env(project_settings)
    if df.empty:
        st.info("Aucune entrée.")
        return
    if "structure" not in df.columns and "forme" in df.columns:
        df["structure"] = df["forme"]
    if "format" not in df.columns and "support" in df.columns:
        df["format"] = df["support"]
    if "public" not in df.columns:
        df["public"] = ""
    options = [f"{row['id']} · {row['type']} · {row['statut']}" for _, row in df.iterrows()]
    idx = st.selectbox("Entrée", list(range(len(options))), format_func=lambda i: options[i])
    row = df.iloc[int(idx)].copy()
    disabled = role == "viewer"
    entry_id = str(row["id"])
    output_widget_key = sync_edition_output_widget_state(
        st.session_state,
        entry_id,
        str(row.get("output", "") or ""),
    )
    legacy_fields: list[str] = []
    legacy_candidates = [
        ("Type de transformation", "type", "types"),
        ("Structure textuelle", "structure", "structures"),
        ("Tonalité textuelle", "ton", "tons"),
        ("Format de sortie", "format", "formats"),
        ("Public cible", "public", "publics"),
        ("Statut", "statut", "statuts"),
    ]
    for label, row_key, dim_key in legacy_candidates:
        value = str(row.get(row_key, "") or "").strip()
        if value and value not in dimensions[dim_key]:
            legacy_fields.append(label)
    if legacy_fields:
        st.warning(
            f"{len(legacy_fields)} champ(s) obsolète(s) détecté(s): {', '.join(legacy_fields)}. "
            "Cette valeur existe dans vos données mais plus dans le preset actif.",
            icon="⚠️",
        )
    with st.form("edit_entry_form"):
        row["type"] = _select_with_legacy(
            "Type de transformation",
            dimensions["types"],
            str(row["type"]),
            key="edit_type_select",
            disabled=disabled,
            show_warning=False,
        )
        row["structure"] = _select_with_legacy(
            "Structure textuelle",
            dimensions["structures"],
            str(row["structure"]),
            key="edit_structure_select",
            disabled=disabled,
            show_warning=False,
        )
        row["ton"] = _select_with_legacy(
            "Tonalité textuelle",
            dimensions["tons"],
            str(row["ton"]),
            key="edit_ton_select",
            disabled=disabled,
            show_warning=False,
        )
        row["format"] = _select_with_legacy(
            "Format de sortie",
            dimensions["formats"],
            str(row["format"]),
            key="edit_format_select",
            disabled=disabled,
            show_warning=False,
        )
        row["public"] = _select_with_legacy(
            "Public cible",
            dimensions["publics"],
            str(row["public"]),
            key="edit_public_select",
            disabled=disabled,
            show_warning=False,
        )
        row["input"] = st.text_area("Brouillon", value=row["input"], height=140, disabled=disabled)
        row["output"] = st.text_area(
            "Texte généré",
            height=240,
            disabled=disabled,
            key=output_widget_key,
        )
        row["statut"] = _select_with_legacy(
            "Statut",
            dimensions["statuts"],
            str(row["statut"]),
            key="edit_statut_select",
            disabled=disabled,
            show_warning=False,
        )
        row["notes"] = st.text_input("Notes", value=row["notes"], disabled=disabled)
        col1, col2 = st.columns(2)
        fix = col1.form_submit_button("Corriger output", disabled=disabled)
        save = col2.form_submit_button("Sauvegarder", disabled=disabled, type="primary")
    if fix:
        try:
            source_text = str(st.session_state.get(output_widget_key, "") or "")
            corrected = corriger_texte_fr(
                source_text,
                languagetool_base_url=project_settings.languagetool_base_url or None,
            )
            st.session_state[output_widget_key] = corrected
            st.toast("Correction orthographique appliquée au texte généré.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Correction impossible: {exc}")
    if save:
        require_role(engine, project_id, user.user_id, ("admin", "collaborator"))
        out = _ensure_cache_columns_on_df(df.copy())
        for col in [
            "type",
            "structure",
            "ton",
            "format",
            "public",
            "input",
            "output",
            "statut",
            "notes",
        ]:
            out.loc[out["id"] == row["id"], col] = str(row[col])
        nlp_model = _load_fr_core_nlp()
        pkg = compute_row_cache(
            str(row["input"]),
            str(row["output"]),
            nlp_model,
            out,
            str(row["id"]),
            CACHE_COLUMNS,
            avg_signature_from_cache,
        )
        for col, val in pkg.cache.items():
            out.loc[out["id"].astype(str) == str(row["id"]), col] = val
        entry_id_save = str(row["id"])
        try:
            update_project_entries(engine, project_id, out, user.user_id)
            invalidate_project_entries_cache()
            df_loaded = cached_load_project_entries(engine, project_id, user.user_id)
            fb = row_nlp_feedback_bundle_after_persist(
                df_loaded, entry_id_save, nlp_model, CACHE_COLUMNS
            )
            _store_post_save_stylometric_feedback(project_id, fb)
        except Exception as exc:  # noqa: BLE001
            _clear_post_save_stylometric_feedback(project_id)
            _show_action_error("Sauvegarde impossible", exc)
            return
        st.success("Entrée mise à jour.")
        st.rerun()


def render_tab_dashboard(df: pd.DataFrame, role: str) -> None:
    st.subheader("Tableau de bord")
    st.caption(f"Rôle projet: {role}")
    if df.empty:
        st.info("Aucune donnée.")
        return
    df_view = df.copy()
    if "structure" not in df_view.columns and "forme" in df_view.columns:
        df_view["structure"] = df_view["forme"]
    if "format" not in df_view.columns and "support" in df_view.columns:
        df_view["format"] = df_view["support"]
    if "public" not in df_view.columns:
        df_view["public"] = ""
    for cache_col in ("_coherence_score", "_syntax_contrast", "_signature_json"):
        if cache_col not in df_view.columns:
            df_view[cache_col] = ""

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", len(df_view))
    c2.metric("Validées", int((df_view["statut"] == STATUT_VALIDE).sum()))
    c3.metric("Types", int(df_view["type"].nunique()))

    scope_key = "dashboard_stylometry_scope"
    if scope_key not in st.session_state:
        st.session_state[scope_key] = "validated"
    scope_choice = st.radio(
        "Périmètre des indicateurs (distribution, contraste syntaxique, outliers)",
        options=("validated", "all"),
        format_func=lambda x: (
            "Fiches validées uniquement (aligné export JSONL)"
            if x == "validated"
            else "Tout le projet (tous statuts)"
        ),
        index=0 if st.session_state.get(scope_key, "validated") == "validated" else 1,
        horizontal=True,
        key=scope_key,
    )
    validated_only = scope_choice == "validated"
    scope_df = dataframe_for_dashboard_scope(
        df_view,
        validated_only=validated_only,
        validated_label=STATUT_VALIDE,
    )

    if validated_only:
        st.caption(
            "Les métriques ci-dessous (sauf la variance par axe) utilisent uniquement les "
            "fiches au statut validé, comme le fichier JSONL exporté."
        )
    else:
        st.caption(
            "Vue « tout le projet » : les brouillons sont inclus — ne confondez pas cette "
            "vue avec le périmètre exporté (validées seulement)."
        )

    st.markdown("#### Distribution des scores de cohérence")
    scores = list_parsed_coherence_scores(scope_df)
    n_scope = len(scope_df)
    if "_coherence_score" in scope_df.columns:
        missing_scores = sum(
            1
            for v in scope_df["_coherence_score"].tolist()
            if parse_persisted_coherence_score(v) is None
        )
    else:
        missing_scores = n_scope
    if not scores:
        st.info("Aucun score de cohérence numérique sur ce périmètre.")
    else:
        bucket_df = coherence_score_bucket_table(scores)
        st.bar_chart(bucket_df.set_index("Tranche (score)"), width="stretch", horizontal=False)
        if missing_scores:
            st.caption(
                f"{missing_scores} entrée(s) sans score de cohérence exploitable sur "
                f"{n_scope} dans ce périmètre."
            )

    st.markdown("#### Écart-type par axe stylistique (fiches validées)")
    df_valid = dataframe_for_dashboard_scope(
        df_view,
        validated_only=True,
        validated_label=STATUT_VALIDE,
    )
    var_axes = signature_variance(df_valid)
    if var_axes is None:
        st.info(
            "Variance par axe indisponible : il faut au moins deux signatures "
            "stylométriques exploitables parmi les fiches validées."
        )
    else:
        st.caption(
            "Indicateur calculé exclusivement sur les fiches validées, conformément au "
            "contrat analytique de signature_variance()."
        )
        var_frame = (
            pd.DataFrame({"Axe": list(var_axes.keys()), "Écart-type": list(var_axes.values())})
            .sort_values("Écart-type", ascending=False)
            .reset_index(drop=True)
        )
        st.bar_chart(var_frame.set_index("Axe"), width="stretch", horizontal=False)

    st.markdown("#### Entrées aux scores de cohérence les plus bas (outliers)")
    out_tbl = outliers_low_coherence_table(scope_df, limit=15)
    if out_tbl.empty:
        st.info("Aucune entrée avec score de cohérence numérique sur ce périmètre.")
    else:
        disp = out_tbl.rename(
            columns={
                "id": "Identifiant",
                "statut": "Statut",
                "type": "Type",
                "score_coherence": "Score cohérence (0–100)",
            }
        )
        st.dataframe(disp, hide_index=True, width="stretch")
        st.caption(
            "Retrouvez une fiche via son identifiant dans l'onglet « Gestion & édition » "
            "(liste déroulante ou navigation entre fiches)."
        )

    st.markdown("#### Moyenne du contraste syntaxique")
    m_contrast = mean_syntax_contrast_parsed(scope_df)
    if m_contrast is None:
        st.info("Aucune valeur de contraste syntaxique exploitable sur ce périmètre.")
    else:
        st.metric(
            "Moyenne (0–1, plus haut = plus de transformation)",
            f"{m_contrast:.3f}",
        )
        st.caption(
            "Moyenne des cellules `_syntax_contrast` parseables uniquement ; les entrées "
            "sans valeur sont exclues du calcul."
        )

    st.markdown("#### Aperçu des entrées")
    st.dataframe(
        df_view[["id", "date", "type", "structure", "ton", "format", "public", "statut"]],
        hide_index=True,
        width="stretch",
    )
