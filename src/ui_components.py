"""
UI multi-utilisateur / multi-projet.
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from typing import Any, Literal

import pandas as pd
import requests
import streamlit as st
from sqlalchemy.engine import Engine

from src.auth import CurrentUser, create_invitation_link, logout, revoke_account_with_saga
from src.corpus_stylometry_alerts_fr import (
    TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR,
    dashboard_stylometry_glossary_markdown_fr,
    trivial_syntax_contrast_missing_cache_caption_fr,
    trivial_syntax_pair_curator_warning_fr,
    trivial_syntax_pair_threshold_rule_sentence_fr,
)
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
from src.empty_project_onboarding import (
    NO_PROJECT_CREATION_DISABLED_MESSAGE,
    ONBOARDING_MAIN_CREATE_FORM_KEY_PREFIX,
    ONBOARDING_PRIMARY_SUBMIT_LABEL_FR,
    PRODUCT_RULE_ISSUE_11_CREATION_PATHS_FR,
    SIDEBAR_CONTEXT_HINT_FR,
    STYLOMETRIC_VALUE_SENTENCE_FR,
    TAB_PROJECTS_ACTIONS_CREATE_FORM_KEY_PREFIX,
    is_self_service_project_creation_allowed,
    onboarding_steps_when_creation_allowed,
)
from src.export_utils import ExportScope, convert_to_jsonl
from src.flash_messages import schedule_post_rerun_flash
from src.llm_generate import generate_input_from_output, generate_output_from_input
from src.mailer import send_account_link_email
from src.nlp_engine import (
    CURATOR_MESSAGE_ADVICE_BALANCED,
    DASHBOARD_COHERENCE_SCORE_MAX_ROWS_FULL_SCAN,
    DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT,
    RowNlpCacheResult,
    avg_signature_from_cache,
    coherence_score_bucket_table,
    compute_row_cache,
    corriger_texte_fr,
    count_trivial_syntax_contrast_entries,
    dataframe_for_coherence_distribution_scan,
    dataframe_for_dashboard_scope,
    edition_entry_k_of_n,
    edition_nav_neighbor_entry_id,
    edition_pick_revision_stats,
    edition_statut_filter_options,
    filter_edition_entries_dataframe,
    is_persisted_syntax_contrast_trivially_low,
    list_parsed_coherence_scores,
    mean_syntax_contrast_parsed,
    outliers_low_coherence_table,
    parse_persisted_syntax_contrast,
    post_save_stylometric_session_payload,
    row_nlp_feedback_bundle_after_persist,
    signature_variance,
    summarize_parsed_coherence_scores,
    trivial_syntax_contrast_entries_table,
)
from src.post_save_feedback_display import (
    post_save_freshness_caption_fr,
    post_save_stylistic_metric_labels_fr,
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
from src.project_session import MembershipProject, resolve_active_project
from src.services.dashboard_stylometry_service import (
    count_rows_missing_parseable_coherence_score,
    project_dataset_headline_metrics,
)
from src.services.edition_filters_service import (
    build_edition_score_filter_spec,
    coherence_bucket_label_fr,
)
from src.services.edition_sequential_navigation import (
    edition_nav_boundary_caption_fr,
    edition_nav_singleton_filtered_caption_fr,
    edition_nav_unsaved_changes_notice_fr,
)
from src.services.export_scope_service import summarize_export_perimeter
from src.services.project_dataframe_view import prepare_for_dashboard_tab, prepare_for_edition_tab
from src.super_admin_ui_texts import (
    SUPER_ADMIN_ACCOUNT_MANAGEMENT_HUB_TITLE,
    SUPER_ADMIN_ACCOUNTS_SECTION_TITLE,
    SUPER_ADMIN_ACTIONS_SECTION_TITLE,
    SUPER_ADMIN_DLQ_SECTION_TITLE,
    SUPER_ADMIN_INVITE_SECTION_TITLE,
    SUPER_ADMIN_SAGA_SECTION_TITLE,
    SUPER_ADMIN_TECH_EXPANDER_CAPTION,
    SUPER_ADMIN_TECH_EXPANDER_TITLE,
    SUPER_ADMIN_WORKFLOW_HINT,
    button_detach_memberships,
    button_replay_quarantined,
    error_delete_target_account_failed,
    error_detach_memberships_failed,
    flash_memberships_detached,
    saga_metric_label,
    selectbox_dlq_operation,
    selectbox_target_account,
    super_admin_accounts_table_column_labels,
    super_admin_tab_labels,
    super_admin_warning_detach_memberships,
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
    """Garantit colonnes de cache NLP + alias legacy (voir ``prepare_for_edition_tab``)."""
    return prepare_for_edition_tab(df)


def _clear_post_save_stylometric_feedback(project_id: str) -> None:
    """Supprime le feedback stylométrique en session (échec de persistance ou reset)."""
    st.session_state.pop(_post_save_stylometric_session_key(project_id), None)


def _store_post_save_stylometric_feedback(project_id: str, pkg: RowNlpCacheResult) -> None:
    """Enregistre le feedback à afficher au prochain run (après ``st.rerun()``)."""
    st.session_state[_post_save_stylometric_session_key(project_id)] = (
        post_save_stylometric_session_payload(pkg)
    )


def _render_post_save_stylometric_feedback(project_id: str) -> None:
    """Affiche puis consomme le feedback stylométrique post-sauvegarde (session)."""
    key = _post_save_stylometric_session_key(project_id)
    payload = st.session_state.pop(key, None)
    if not payload:
        return
    labels = post_save_stylistic_metric_labels_fr()
    st.markdown("#### Retour stylistique (ligne enregistrée)")
    st.caption(post_save_freshness_caption_fr(synchronous_before_commit=True))
    m1, m2, m3 = st.columns(3)
    score = payload.get("score")
    with m1:
        if score is None:
            st.metric(labels["coherence_score"], "—")
        else:
            st.metric(labels["coherence_score"], f"{int(score)}/100")
    with m2:
        st.metric(labels["ttr"], str(payload.get("ttr", "—")))
    with m3:
        st.metric(labels["syntax_contrast"], str(payload.get("contrast", "—")))
    if payload.get("syntax_contrast_trivially_low"):
        st.warning(
            trivial_syntax_pair_curator_warning_fr(
                contrast_raw_display=str(payload.get("contrast", "") or "").strip() or None
            ),
            icon="⚠️",
        )
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
    st.markdown("##### Conseils prioritaires")
    advices = payload.get("advices") or []
    if advices:
        st.info("\n\n".join(str(a) for a in advices[:3]), icon="💡")
    else:
        st.info(CURATOR_MESSAGE_ADVICE_BALANCED, icon="💡")


def render_post_save_stylometric_feedback_banner(project_id: str) -> None:
    """Affiche le bloc post-sauvegarde une seule fois par exécution (évite multi-onglets).

    ``st.tabs`` exécute chaque corps d'onglet à chaque rerun : appeler ce rendu depuis
    un seul point (``main``) garantit que le payload session n'est pas consommé par
    l'onglet « Nouvelle entrée » avant l'onglet « Édition ».
    """
    _render_post_save_stylometric_feedback(project_id)


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


def edition_output_correction_notice_session_key(project_id: str) -> str:
    """Return ``session_state`` key for a one-shot post-correction UI notice.

    Args:
        project_id: Active project identifier (scopes the flag per project).

    Returns:
        A stable key whose value is the ``entry_id`` last orthographically corrected.
    """
    return f"_edition_output_corrected_notice_{project_id}"


def read_edition_output_text_for_persist(
    session: MutableMapping[str, Any],
    output_widget_key: str,
    fallback_row_output: str,
) -> str:
    """Resolve « texte généré » for persistence from the live widget buffer.

    Keyed ``st.text_area`` values live under ``output_widget_key``; after an
    in-session correction, the dataframe row snapshot can lag behind that buffer.
    Callers should prefer this helper when saving or recomputing NLP cache.

    Args:
        session: Streamlit ``st.session_state`` or any mutable mapping (tests).
        output_widget_key: Same key passed to ``st.text_area(..., key=...)``.
        fallback_row_output: Persisted output from the loaded row if the buffer
            is absent (defensive).

    Returns:
        Full generated text string to treat as authoritative for save/cache.
    """
    if output_widget_key in session:
        return str(session[output_widget_key] or "")
    return str(fallback_row_output or "")


def _sanitize_new_entry_session_user_id(raw_user_id: str) -> str:
    """Normalize a user id for use inside Streamlit session-state key strings.

    Args:
        raw_user_id: Authenticated user identifier (may contain punctuation).

    Returns:
        A non-empty ASCII-ish token safe for ``session_state`` keys.
    """
    s = (raw_user_id or "").strip()
    if not s:
        return "anonymous"
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    return out[:80] if out else "anonymous"


def _legacy_new_entry_storage_keys(project_id: str) -> dict[str, str]:
    """Return pre-user-scope keys (``new_entry_{project}_*``) for one-off migration."""
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


def _discard_legacy_new_entry_keys_for_project(
    session: MutableMapping[str, Any],
    legacy: dict[str, str],
) -> None:
    """Remove pre-user-scope keys for this project without copying them anywhere.

    Legacy keys did not record which account produced the text. Copying them into
    the current user's scoped buffers would be unsafe when another account signs
    in on the same browser session without a full session reset.

    The legacy pending-clear flag ``_pending_clear_new_entry_{project_id}`` is
    left intact so ``render_tab_ajout`` can still consume it once.

    Args:
        session: Streamlit ``st.session_state`` or any mutable mapping (tests).
        legacy: Keys from :func:`_legacy_new_entry_storage_keys`.
    """
    for key in legacy.values():
        session.pop(key, None)


def new_entry_session_keys(project_id: str, user_id: str) -> dict[str, str]:
    """Build stable ``session_state`` keys for the « Nouvelle entrée » tab.

    Keys are scoped by ``project_id`` and ``user_id`` so drafts do not collide
    across projects or between accounts **when each account uses a clean session
    or goes through normal logout / login**.

    Streamlit ``session_state`` is per browser tab/session: pre-user-scope legacy
    keys are discarded instead of being reassigned to the current user (see
    :func:`_discard_legacy_new_entry_keys_for_project`). On logout or when the
    authenticated user id changes, ``src.auth`` (``logout`` / ``_set_user``) purges
    all ``new_entry_*`` / ``_pending_clear_new_entry_*`` keys via
    ``purge_all_new_entry_session_state`` (see ``src.new_entry_session_state``).

    Args:
        project_id: Active project identifier.
        user_id: Authenticated user identifier (stable primary key).

    Returns:
        Mapping of logical field names to session keys (``input``, ``output``,
        dimension keys, ``statut``, ``notes``).
    """
    uid = _sanitize_new_entry_session_user_id(user_id)
    prefix = f"new_entry_{project_id}_u_{uid}"
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
    user_id: str,
    dimensions: dict[str, list[str]],
) -> dict[str, str]:
    """Ensure new-entry widget keys exist with safe defaults.

    If a stored select value is no longer in the active preset options (e.g.
    after a dimensions change), it is reset to the first available option.

    Args:
        session: Streamlit ``st.session_state`` or any mutable mapping (tests).
        project_id: Active project identifier.
        user_id: Authenticated user identifier (stable primary key).
        dimensions: Preset dimension lists (``types``, ``structures``, …).

    Returns:
        The same mapping as :func:`new_entry_session_keys`.
    """
    keys = new_entry_session_keys(project_id, user_id)
    legacy = _legacy_new_entry_storage_keys(project_id)
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
    _discard_legacy_new_entry_keys_for_project(session, legacy)
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


def new_entry_pending_clear_session_key(project_id: str, user_id: str) -> str:
    """Return the ``session_state`` flag used to clear buffers after a successful save.

    Clearing must happen on the next run *before* ``text_area`` widgets are created,
    because Streamlit forbids assigning to a widget key after that widget was
    instantiated in the same script run.

    Args:
        project_id: Active project identifier.
        user_id: Authenticated user identifier (stable primary key).

    Returns:
        A session key scoped by ``project_id`` and ``user_id``.
    """
    uid = _sanitize_new_entry_session_user_id(user_id)
    return f"_pending_clear_new_entry_{project_id}_u_{uid}"


def _legacy_new_entry_pending_clear_session_key(project_id: str) -> str:
    """Session flag used before user id was included in key names (migration only)."""
    return f"_pending_clear_new_entry_{project_id}"


def commit_new_entry_llm_result(
    session: MutableMapping[str, Any],
    keys: Mapping[str, str],
    *,
    target: Literal["input", "output"],
    text: str,
) -> None:
    """Write LLM output into the canonical new-entry body buffer.

    Args:
        session: Streamlit session state or any mutable mapping (tests).
        keys: Mapping returned by :func:`new_entry_session_keys`.
        target: Whether to update the draft (``input``) or generated (``output``) field.
        text: Full text to store.
    """
    if target == "input":
        session[keys["input"]] = text
    else:
        session[keys["output"]] = text


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
        schedule_post_rerun_flash(st.session_state, success_message, level="success")
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        _show_action_error("Mise à jour réglages impossible", exc)


def _render_project_create_form(
    user: CurrentUser,
    engine: Engine,
    *,
    key_prefix: str,
    label: str = "Nouveau projet",
    submit_label: str = "Créer",
) -> None:
    """
    Formulaire de création projet.

    Contrat: gère tout le flux (messages + rerun), sans valeur de retour.
    """
    with st.form(f"{key_prefix}_create_project_form"):
        pname = st.text_input(label, key=f"{key_prefix}_new_project_name_input")
        submit = st.form_submit_button(submit_label, key=f"{key_prefix}_create_project_submit")
    if not submit:
        return
    if not pname.strip():
        st.error("Nom du projet requis.")
        return
    try:
        pid = create_project(engine, user.user_id, pname.strip())
        invalidate_project_entries_cache()
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
            invalidate_project_entries_cache()
            st.session_state.pop("project_id", None)
            schedule_post_rerun_flash(st.session_state, "Projet supprimé.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            _show_action_error("Suppression projet impossible", exc)


def render_needs_active_project_tab_notice(*, target_workflow_tab_title_fr: str) -> None:
    """Placeholder when a workflow tab is rendered before any project exists (issue-028).

    Streamlit runs every tab body on each rerun; this avoids loading project-scoped
    data without a valid ``project_id``.

    Args:
        target_workflow_tab_title_fr: Label of the gated tab (from ``tab_layout``).
    """
    st.info(
        "Crée d’abord un projet depuis l’onglet **Projets**, puis reviens dans "
        f"**{target_workflow_tab_title_fr}**.",
        icon="ℹ️",
    )


def render_no_project_onboarding(user: CurrentUser, engine: Engine) -> None:
    """Welcome + first-project flow inside the **Projets** tab (issue-008, issue-025, issue-028).

    Shows value proposition, the issue-11 / issue-28 product rule, step 1 with an
    on-page creation form (``ONBOARDING_MAIN_*`` keys), then narrative steps 2–3 that
    reference real tab titles from ``main_tab_labels``.

    When ``DISABLE_SELF_SERVICE_PROJECT_CREATION`` is set, shows guidance without
    a creation form (invitation-only deployments).

    Args:
        user: Authenticated user.
        engine: SQLAlchemy engine.
    """
    st.markdown("## Bienvenue dans Dataset Style Studio")
    st.markdown(STYLOMETRIC_VALUE_SENTENCE_FR)
    st.caption(PRODUCT_RULE_ISSUE_11_CREATION_PATHS_FR)
    st.caption(SIDEBAR_CONTEXT_HINT_FR)
    if not is_self_service_project_creation_allowed():
        st.warning(NO_PROJECT_CREATION_DISABLED_MESSAGE, icon="🔒")
        st.info(
            "Tu peux ouvrir le menu **☰** en haut à gauche pour te déconnecter ou "
            "consulter ton compte.",
            icon="ℹ️",
        )
        return
    st.info(
        "Tu n’as pas encore de projet : l’étape 1 te fait créer le premier dans cet onglet ; "
        "les étapes 2 et 3 décrivent la suite une fois le studio chargé.",
        icon="👋",
    )
    steps = onboarding_steps_when_creation_allowed()
    st.markdown(steps[0].body_markdown)
    _render_project_create_form(
        user,
        engine,
        key_prefix=ONBOARDING_MAIN_CREATE_FORM_KEY_PREFIX,
        label="Nom du projet",
        submit_label=ONBOARDING_PRIMARY_SUBMIT_LABEL_FR,
    )
    for step in steps[1:]:
        st.markdown(step.body_markdown)


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
        st.session_state.pop("project_id", None)
        st.session_state.pop("project_role", None)
        if not is_self_service_project_creation_allowed():
            st.warning(NO_PROJECT_CREATION_DISABLED_MESSAGE)
            return "", ""
        st.warning("Aucun projet pour l’instant.")
        st.caption(
            "Crée ton premier projet depuis l’onglet **Projets** (zone principale, "
            "bandeau d’onglets). Ce menu latéral reste dédié au **contexte** : compte "
            "et, plus tard, choix du projet courant."
        )
        return "", ""

    summaries = [MembershipProject(p.project_id, p.role) for p in projects]
    pid, role = resolve_active_project(_current_project_id(), summaries)
    st.session_state["project_id"] = pid
    st.session_state["project_role"] = role

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
    _render_project_create_form(
        user,
        engine,
        key_prefix=TAB_PROJECTS_ACTIONS_CREATE_FORM_KEY_PREFIX,
    )

    st.markdown("### Zone sensible")
    _render_project_delete_guarded_form(
        user,
        engine,
        project_id,
        current.name,
        role,
        key_prefix=TAB_PROJECTS_ACTIONS_CREATE_FORM_KEY_PREFIX,
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


def _render_super_admin_technical_panel(user: CurrentUser, engine: Engine) -> None:
    """Saga, quarantaine et relance manuelle (onglet secondaire, issue-012)."""
    st.markdown(f"### {SUPER_ADMIN_TECH_EXPANDER_TITLE}")
    st.caption(SUPER_ADMIN_TECH_EXPANDER_CAPTION)
    st.markdown(f"#### {SUPER_ADMIN_SAGA_SECTION_TITLE}")
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
        c1.metric(saga_metric_label("pending"), int(state_counts.get("pending", 0)))
        c2.metric(
            saga_metric_label("provider_done"),
            int(state_counts.get("provider_done", 0)),
        )
        c3.metric(saga_metric_label("failed"), int(state_counts.get("failed", 0)))
        c4.metric(
            saga_metric_label("quarantined"),
            int(state_counts.get("quarantined", 0)),
        )
        st.dataframe(monitor_df, hide_index=True, width="stretch")
    else:
        st.info("Aucune opération de suppression récente à afficher.")

    st.markdown(f"#### {SUPER_ADMIN_DLQ_SECTION_TITLE}")
    dlq_rows = list_quarantined_deprovision_ops(engine, user.user_id, limit=50)
    if not dlq_rows:
        st.info("Aucune opération en quarantaine.")
    else:
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
            selectbox_dlq_operation(),
            [row.operation_id for row in dlq_rows],
            key="sa_dlq_operation_select",
        )
        replay_confirm = st.checkbox(
            "Je confirme la relance du traitement bloqué",
            key="sa_dlq_replay_confirm_checkbox",
        )
        if st.button(
            button_replay_quarantined(),
            key="sa_dlq_replay_btn",
            disabled=not replay_confirm,
        ):
            try:
                replay_quarantined_operation(engine, user.user_id, selected_op)
                schedule_post_rerun_flash(
                    st.session_state,
                    "Opération remise en file d'attente.",
                    channel="super_admin",
                )
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                _show_action_error("Relance du traitement impossible", exc)


def render_tab_super_admin(user: CurrentUser, engine: Engine) -> None:
    """Administration globale des comptes."""
    st.subheader("Super Admin")
    if not user.is_super_admin:
        st.info("Accès réservé aux super admins.")
        return

    st.caption(SUPER_ADMIN_WORKFLOW_HINT)
    tab_accounts, tab_tech = st.tabs(super_admin_tab_labels())
    with tab_accounts:
        _render_super_admin_accounts_panel(user, engine)
    with tab_tech:
        _render_super_admin_technical_panel(user, engine)


def _render_super_admin_accounts_panel(user: CurrentUser, engine: Engine) -> None:
    """Invitation, liste des comptes et actions courantes (issue-012, issue-029)."""
    st.markdown(f"## {SUPER_ADMIN_ACCOUNT_MANAGEMENT_HUB_TITLE}")
    st.markdown(f"### {SUPER_ADMIN_INVITE_SECTION_TITLE}")
    with st.form("super_admin_invite_form"):
        invite_email = st.text_input(
            "E-mail du collaborateur à inviter",
            key="sa_invite_email_input",
        )
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

    st.markdown(f"### {SUPER_ADMIN_ACCOUNTS_SECTION_TITLE}")
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
    ).rename(columns=super_admin_accounts_table_column_labels())
    if accounts_df.empty:
        st.info("Aucun compte actif.")
    else:
        st.dataframe(accounts_df, hide_index=True, width="stretch")

    st.markdown(f"### {SUPER_ADMIN_ACTIONS_SECTION_TITLE}")
    if not rows:
        return
    choices = {f"{row.email} ({row.user_id})": row for row in rows}
    selected_label = st.selectbox(
        selectbox_target_account(), list(choices.keys()), key="sa_target_select"
    )
    target = choices[selected_label]
    owner_count = count_owned_projects(engine, target.user_id)
    membership_count = count_active_memberships(engine, target.user_id)
    st.caption(
        "Freins avant suppression du compte : "
        f"projets dont ce compte est propriétaire = **{owner_count}**, "
        f"accès à des projets d'autrui (collaboration) = **{membership_count}**."
    )

    if membership_count > 0:
        st.warning(
            super_admin_warning_detach_memberships(
                membership_count=membership_count,
                email=target.email,
            ),
            icon="⚠️",
        )
        detach_confirm = st.checkbox(
            "Je confirme le retrait complet des accès aux projets partagés",
            key="sa_detach_confirm_checkbox",
        )
        detach_typed_email = st.text_input(
            f"Retape l'email cible ({target.email})",
            key="sa_detach_email_confirm_input",
        )
        if st.button(
            button_detach_memberships(),
            key="sa_detach_memberships_btn",
            type="secondary",
            disabled=not detach_confirm,
        ):
            if detach_typed_email.strip().lower() != target.email.strip().lower():
                st.error("Email de confirmation invalide.")
                return
            try:
                removed = detach_memberships_as_super_admin(engine, user.user_id, target.user_id)
                schedule_post_rerun_flash(
                    st.session_state,
                    flash_memberships_detached(removed),
                    channel="super_admin",
                )
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                _show_action_error(error_detach_memberships_failed(), exc)

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
            invalidate_project_entries_cache()
            schedule_post_rerun_flash(
                st.session_state,
                "Compte supprimé.",
                channel="super_admin",
            )
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            _show_action_error(error_delete_target_account_failed(), exc)


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

    perimeter = summarize_export_perimeter(df, export_scope)
    df_export = perimeter.dataframe
    row_n = perimeter.row_count
    st.metric("Fiches dans le périmètre", row_n)
    st.caption(perimeter.recap_caption)
    if perimeter.recap_warning:
        st.warning(perimeter.recap_warning)

    export_format = st.selectbox(
        "Format JSONL",
        ["lfm2", "baguettotron", "mistral"],
        key="export_format_select",
    )
    csv = df_export.to_csv(index=False).encode("utf-8")
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


def _new_entry_llm_generate_clicked(
    project_id: str,
    user_id: str,
    project_settings: ProjectSettings,
    dimensions: dict[str, list[str]],
    mode: Literal["draft_to_output", "output_to_draft"],
) -> None:
    """Run LLM generation during the pre-script callback phase (before widgets run).

    Streamlit applies widget state from the client, then invokes ``on_click``
    callbacks, then runs the script. Updating ``session_state`` here avoids
    ``StreamlitAPIException`` when assigning to ``text_area`` keys after those
    widgets were already instantiated. After a successful write, ``st.rerun()``
    forces a fresh pass so the text areas immediately show the committed LLM
    text (avoids stale widget state masking the new value).
    """
    _llm_env(project_settings)
    keys = ensure_new_entry_widget_keys_initialized(
        st.session_state, project_id, user_id, dimensions
    )
    type_ = str(st.session_state.get(keys["type"], ""))
    structure = str(st.session_state.get(keys["structure"], ""))
    ton = str(st.session_state.get(keys["ton"], ""))
    format_ = str(st.session_state.get(keys["format"], ""))
    public = str(st.session_state.get(keys["public"], ""))
    if mode == "draft_to_output":
        input_text = str(st.session_state.get(keys["input"], ""))
        if not input_text.strip():
            return
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
                commit_new_entry_llm_result(
                    st.session_state, keys, target="output", text=str(generated)
                )
                st.toast("Texte généré.")
                st.rerun()
            else:
                st.error("La génération a échoué. Vérifiez vos paramètres LLM puis réessayez.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur génération texte", exc_info=exc)
            st.error("Génération impossible: erreur inattendue côté service.")
        return

    output_text = str(st.session_state.get(keys["output"], ""))
    if not output_text.strip():
        return
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
            commit_new_entry_llm_result(st.session_state, keys, target="input", text=str(generated))
            st.toast("Brouillon généré.")
            st.rerun()
        else:
            st.error("La génération a échoué. Vérifiez vos paramètres LLM puis réessayez.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erreur génération brouillon", exc_info=exc)
        st.error("Génération impossible: erreur inattendue côté service.")


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

    Draft and generated text use ``st.session_state`` (per project and user) as
    the single source of truth. LLM actions use ``st.button(..., on_click=…)``
    so buffers are updated before ``text_area`` widgets run, which matches
    Streamlit's rule forbidding writes to widget keys after those widgets were
    instantiated in the same script run. A successful generation triggers
    ``st.rerun()`` so the refreshed text areas always reflect the committed LLM
    text. « Enregistrer » reads the same keys from ``session_state`` at click
    time.
    """
    st.subheader("Nouvelle entrée")
    if role == "viewer":
        st.info("Lecture seule (viewer).")
        return
    _llm_env(project_settings)
    # Legacy keys from the old form+generation split; drop to avoid stale reads.
    st.session_state.pop("new_generated_output", None)
    st.session_state.pop("new_generated_input", None)

    keys = ensure_new_entry_widget_keys_initialized(
        st.session_state, project_id, user.user_id, dimensions
    )
    if st.session_state.pop(
        new_entry_pending_clear_session_key(project_id, user.user_id), None
    ) or st.session_state.pop(_legacy_new_entry_pending_clear_session_key(project_id), None):
        st.session_state[keys["input"]] = ""
        st.session_state[keys["output"]] = ""
        st.session_state[keys["notes"]] = ""

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
    col1.button(
        "Générer texte",
        key=f"{keys['input']}_btn_gen_out",
        on_click=_new_entry_llm_generate_clicked,
        kwargs={
            "project_id": project_id,
            "user_id": user.user_id,
            "project_settings": project_settings,
            "dimensions": dimensions,
            "mode": "draft_to_output",
        },
    )
    col2.button(
        "Générer brouillon",
        key=f"{keys['output']}_btn_gen_in",
        on_click=_new_entry_llm_generate_clicked,
        kwargs={
            "project_id": project_id,
            "user_id": user.user_id,
            "project_settings": project_settings,
            "dimensions": dimensions,
            "mode": "output_to_draft",
        },
    )
    save = col3.button("Enregistrer", type="primary", key=f"{keys['input']}_btn_save")

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
        to_persist: pd.DataFrame
        try:
            with st.spinner("Analyse linguistique et enregistrement en base…"):
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
                update_project_entries(engine, project_id, to_persist, user.user_id)
                invalidate_project_entries_cache()
                df_loaded = cached_load_project_entries(engine, project_id, user.user_id)
                fb = row_nlp_feedback_bundle_after_persist(
                    df_loaded, row_id, nlp_model, CACHE_COLUMNS
                )
                _store_post_save_stylometric_feedback(project_id, fb)
        except Exception as exc:  # noqa: BLE001
            _clear_post_save_stylometric_feedback(project_id)
            _show_action_error("Enregistrement impossible", exc)
            return
        st.session_state[new_entry_pending_clear_session_key(project_id, user.user_id)] = True
        st.success("Entrée enregistrée.")
        st.rerun()


def _render_edition_entry_change_confirm_dialog(
    *,
    entry_widget_key: str,
    committed_key: str,
    pending_key: str,
    target_entry_id: str,
) -> None:
    """Affiche la boîte de dialogue de confirmation avant changement de fiche (issue 032)."""

    @st.dialog("Changer de fiche")
    def _dialog_body() -> None:
        st.markdown(edition_nav_unsaved_changes_notice_fr())
        ok_key = f"{pending_key}_confirm_ok"
        cancel_key = f"{pending_key}_confirm_cancel"
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Continuer", type="primary", key=ok_key):
                tid = str(st.session_state.pop(pending_key, "") or target_entry_id)
                st.session_state[entry_widget_key] = tid
                st.session_state[committed_key] = tid
                st.rerun()
        with c2:
            if st.button("Annuler", key=cancel_key):
                st.session_state.pop(pending_key, None)
                st.rerun()

    _dialog_body()


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
    _llm_env(project_settings)
    if df.empty:
        st.info("Aucune entrée.")
        return
    basis = prepare_for_edition_tab(df)
    st.markdown("##### Filtres de la liste d'entrées")
    statut_labels = edition_statut_filter_options(dimensions["statuts"], basis)
    statut_choice_key = f"edition_filter_statut_{project_id}"
    statut_choices_ui = ["Tous"] + statut_labels
    raw_stat_choice = st.selectbox(
        "Filtrer par statut",
        statut_choices_ui,
        key=statut_choice_key,
    )
    statut_filter_val: str | None = None if raw_stat_choice == "Tous" else str(raw_stat_choice)

    score_mode_key = f"edition_filter_score_mode_{project_id}"
    score_mode_choice = st.selectbox(
        "Filtrer par score de cohérence (_coherence_score)",
        options=("all", "below", "bucket", "na_only"),
        format_func=lambda m: {
            "all": "Tous les scores (aucun filtre)",
            "below": "Strictement sous un seuil (score < seuil)",
            "bucket": "Tranche de 10 points (comme le tableau de bord)",
            "na_only": "Score non calculé uniquement (N/A)",
        }[m],
        key=score_mode_key,
    )
    threshold_lt = 50
    bucket_decile_pick = 0
    if score_mode_choice == "below":
        thr_key = f"edition_filter_score_threshold_{project_id}"
        threshold_lt = int(
            st.number_input(
                "Seuil exclusif (conserver les fiches avec score < seuil, 0–100)",
                min_value=0,
                max_value=100,
                value=50,
                step=1,
                key=thr_key,
            )
        )
    elif score_mode_choice == "bucket":
        b_key = f"edition_filter_score_bucket_{project_id}"
        bucket_decile_pick = int(
            st.selectbox(
                "Tranche",
                options=list(range(10)),
                format_func=coherence_bucket_label_fr,
                key=b_key,
            )
        )
    include_na_key = f"edition_filter_include_na_score_{project_id}"
    include_na_scores = False
    if score_mode_choice not in ("all", "na_only"):
        include_na_scores = st.checkbox(
            "Inclure les fiches sans score exploitable (N/A)",
            value=False,
            key=include_na_key,
            help=(
                "N/A : cellule vide ou non numérique après lecture ; le score est dérivé "
                "uniquement via parse_persisted_coherence_score (même règle que le tableau de bord)."
            ),
        )
        st.caption(
            "Règle N/A : si la case est décochée, les entrées sans score exploitable sont "
            "exclues dès qu'un filtre sur le score (autre que « Tous les scores ») est actif."
        )
    score_spec = build_edition_score_filter_spec(
        score_mode_choice,
        threshold_lt=threshold_lt,
        bucket_decile=bucket_decile_pick,
        include_na=include_na_scores,
    )

    df_pick = filter_edition_entries_dataframe(
        basis,
        statut_label=statut_filter_val,
        score_spec=score_spec,
    )
    if df_pick.empty:
        st.warning(
            "Aucune entrée ne correspond aux filtres. Élargissez les critères : "
            "passez le statut à « Tous », le score à « Tous les scores » ou à une autre "
            "tranche, désactivez « Score non calculé uniquement », ou cochez « Inclure les "
            "fiches sans score exploitable » lorsque vous filtrez par seuil ou tranche."
        )
        return
    n_pick, n_basis = len(df_pick), len(basis)
    if n_pick == 1:
        st.caption(f"1 entrée affichée sur {n_basis} au total.")
    else:
        st.caption(f"{n_pick} entrées affichées sur {n_basis} au total.")

    entry_ids = df_pick["id"].astype(str).tolist()
    id_to_label: dict[str, str] = {}
    for _, r in df_pick.iterrows():
        eid = str(r["id"])
        id_to_label[eid] = f"{eid} · {r['type']} · {r['statut']}"
    entry_widget_key = f"edition_entry_select_{project_id}"
    committed_key = f"edition_entry_committed_{project_id}"
    pending_key = f"edition_nav_pending_target_{project_id}"
    if entry_widget_key in st.session_state and st.session_state[entry_widget_key] not in entry_ids:
        del st.session_state[entry_widget_key]
    if committed_key in st.session_state and st.session_state[committed_key] not in entry_ids:
        del st.session_state[committed_key]
    if pending_key in st.session_state and st.session_state[pending_key] not in entry_ids:
        del st.session_state[pending_key]
    if entry_widget_key not in st.session_state:
        st.session_state[entry_widget_key] = entry_ids[0]
    if committed_key not in st.session_state:
        st.session_state[committed_key] = st.session_state[entry_widget_key]
    pending_raw = st.session_state.get(pending_key)
    pending_target = str(pending_raw) if pending_raw is not None and str(pending_raw) else ""
    if not pending_target:
        wid = str(st.session_state[entry_widget_key])
        com = str(st.session_state[committed_key])
        if wid != com:
            st.session_state[pending_key] = wid
            st.session_state[entry_widget_key] = com
            st.rerun()
    committed_id = str(st.session_state[committed_key])
    nav_prev_key = f"edition_nav_prev_{project_id}"
    nav_next_key = f"edition_nav_next_{project_id}"
    can_prev = edition_nav_neighbor_entry_id(entry_ids, committed_id, direction="prev") is not None
    can_next = edition_nav_neighbor_entry_id(entry_ids, committed_id, direction="next") is not None
    if pending_target:
        _render_edition_entry_change_confirm_dialog(
            entry_widget_key=entry_widget_key,
            committed_key=committed_key,
            pending_key=pending_key,
            target_entry_id=pending_target,
        )
    c_prev, c_sel, c_next = st.columns([1, 8, 1])
    with c_prev:
        if st.button(
            "Précédent",
            key=nav_prev_key,
            disabled=not can_prev,
            help="Fiche précédente (ordre de la liste filtrée, tri stable sur l'identifiant)",
            width="content",
        ):
            nid = edition_nav_neighbor_entry_id(entry_ids, committed_id, direction="prev")
            if nid is not None:
                st.session_state[pending_key] = nid
                st.rerun()
    with c_sel:
        chosen_id = str(
            st.selectbox(
                "Entrée",
                options=entry_ids,
                format_func=lambda eid: id_to_label[str(eid)],
                key=entry_widget_key,
            )
        )
    with c_next:
        if st.button(
            "Suivant",
            key=nav_next_key,
            disabled=not can_next,
            help="Fiche suivante (ordre de la liste filtrée, tri stable sur l'identifiant)",
            width="content",
        ):
            nid = edition_nav_neighbor_entry_id(entry_ids, committed_id, direction="next")
            if nid is not None:
                st.session_state[pending_key] = nid
                st.rerun()
    hint_prev = edition_nav_boundary_caption_fr("prev", can_navigate=can_prev)
    hint_next = edition_nav_boundary_caption_fr("next", can_navigate=can_next)
    hint_single = edition_nav_singleton_filtered_caption_fr(n_filtered=n_pick)
    if hint_single:
        st.caption(hint_single)
    else:
        cap_prev, _, cap_next = st.columns([1, 8, 1])
        with cap_prev:
            if hint_prev:
                st.caption(hint_prev)
        with cap_next:
            if hint_next:
                st.caption(hint_next)
    k_pos, n_filtered = edition_entry_k_of_n(entry_ids, chosen_id)
    rev_pick = edition_pick_revision_stats(df_pick)
    m_pos, m_rev, m_ok, m_other = st.columns(4)
    with m_pos:
        st.metric("Entrée (liste filtrée)", f"{k_pos} / {n_filtered}")
    with m_rev:
        st.metric("À réviser", rev_pick.needing_review)
    with m_ok:
        st.metric("Validées (liste filtrée)", rev_pick.validated)
    with m_other:
        st.metric("Autres statuts", rev_pick.other)
    st.caption(
        "Compteurs ci-dessus : périmètre **liste filtrée** uniquement. "
        "« À réviser » : fiches en statut « A faire » ou « En cours ». "
        "Le libellé « N entrée(s) affichée(s) sur … au total » (au-dessus) indique la taille "
        "de la liste courante par rapport au projet."
    )
    st.caption(
        "Navigation dans l'ordre de la **liste filtrée** (tri stable sur l'identifiant). "
        "Un dialogue de confirmation s'affiche avant tout changement de fiche pour "
        "rappeler le risque de perte des modifications non sauvegardées du formulaire."
    )
    row = df_pick.loc[df_pick["id"].astype(str) == chosen_id].iloc[0].copy()
    disabled = role == "viewer"
    entry_id = str(row["id"])
    output_widget_key = sync_edition_output_widget_state(
        st.session_state,
        entry_id,
        str(row.get("output", "") or ""),
    )
    notice_key = edition_output_correction_notice_session_key(project_id)
    if st.session_state.pop(notice_key, None) == entry_id:
        st.info(
            "Le champ « Texte généré » affiche la correction : vous pouvez l'ajuster "
            "ou l'annuler manuellement avant enregistrement.",
            icon="✏️",
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
    sc_cell = row.get("_syntax_contrast")
    if is_persisted_syntax_contrast_trivially_low(sc_cell):
        st.warning(
            trivial_syntax_pair_curator_warning_fr(
                contrast_raw_display=str(sc_cell or "").strip() or None
            ),
            icon="⚠️",
        )
    elif parse_persisted_syntax_contrast(sc_cell) is None:
        st.caption(trivial_syntax_contrast_missing_cache_caption_fr())
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
            st.session_state[notice_key] = entry_id
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Correction impossible: {exc}")
    if save:
        require_role(engine, project_id, user.user_id, ("admin", "collaborator"))
        row["output"] = read_edition_output_text_for_persist(
            st.session_state,
            output_widget_key,
            str(row.get("output", "") or ""),
        )
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
        entry_id_save = str(row["id"])
        try:
            with st.spinner("Analyse linguistique et enregistrement en base…"):
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
    df_view = prepare_for_dashboard_tab(df)
    total_rows, validated_rows, n_types = project_dataset_headline_metrics(
        df_view,
        validated_status_label=STATUT_VALIDE,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total_rows)
    c2.metric("Validées", validated_rows)
    c3.metric("Types", n_types)

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

    with st.expander(
        "Seuils et définitions (stylométrie du corpus)",
        expanded=False,
    ):
        st.markdown(dashboard_stylometry_glossary_markdown_fr())

    st.markdown("#### Distribution des scores de cohérence")
    work_df, used_sample, n_scope = dataframe_for_coherence_distribution_scan(
        scope_df,
        max_rows_without_sampling=DASHBOARD_COHERENCE_SCORE_MAX_ROWS_FULL_SCAN,
    )
    if used_sample and n_scope > 0:
        st.caption(
            f"Performance : ce périmètre compte {n_scope} entrées ; la distribution et la "
            f"synthèse ci-dessous sont calculées sur un échantillon aléatoire de {len(work_df)} "
            "lignes (scores lus via le même parseur que l'export et les filtres)."
        )
    scores = list_parsed_coherence_scores(work_df)
    missing_scores = count_rows_missing_parseable_coherence_score(work_df)
    if not scores:
        st.info("Aucun score de cohérence numérique sur ce périmètre.")
    else:
        summary = summarize_parsed_coherence_scores(scores)
        if summary is not None:
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Moyenne", f"{summary.mean:.2f}")
            sm2.metric("Médiane", f"{summary.median:.2f}")
            sm3.metric("Minimum", str(summary.minimum))
        bucket_df = coherence_score_bucket_table(scores)
        st.bar_chart(bucket_df.set_index("Tranche (score)"), width="stretch", horizontal=False)
        if missing_scores:
            scope_label = (
                f"{len(work_df)} lignes de l'échantillon"
                if used_sample
                else f"{n_scope} dans ce périmètre"
            )
            st.caption(
                f"{missing_scores} entrée(s) sans score de cohérence exploitable sur {scope_label}."
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
        st.dataframe(var_frame, hide_index=True, width="stretch")

    st.markdown("#### Entrées aux scores de cohérence les plus bas (outliers)")
    out_tbl = outliers_low_coherence_table(scope_df, limit=DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT)
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
            "(liste déroulante ou navigation entre fiches). "
            f"Jusqu'à {DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT} entrées les plus basses sur ce périmètre."
        )

    st.markdown(
        f"#### {TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR} (contraste syntaxique brouillon ↔ généré)"
    )
    n_trivial = count_trivial_syntax_contrast_entries(scope_df)
    st.metric(
        f"Nombre — {TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR}",
        int(n_trivial),
        help=trivial_syntax_pair_threshold_rule_sentence_fr(),
    )
    st.caption(
        "Comptage sur cellules `_syntax_contrast` persistées et parseables uniquement. "
        "Indicateur syntaxique (motifs grammaticaux), pas une mesure sémantique."
    )
    trivial_tbl = trivial_syntax_contrast_entries_table(
        scope_df, limit=DASHBOARD_STYLOMETRY_ALERT_TABLE_LIMIT
    )
    if trivial_tbl.empty:
        st.info(
            f"Aucune entrée ne remplit le critère « {TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR} » "
            "sur ce périmètre (voir l'expander « Seuils et définitions » ou l'info-bulle du compteur)."
        )
    else:
        show_tbl = trivial_tbl.copy()
        show_tbl.insert(3, "alerte", TRIVIAL_SYNTAX_PAIR_BUSINESS_LABEL_FR)
        disp_trivial = show_tbl.rename(
            columns={
                "id": "Identifiant",
                "statut": "Statut",
                "type": "Type",
                "alerte": "Alerte",
                "syntax_contrast": "Valeur `_syntax_contrast`",
            }
        )
        st.dataframe(disp_trivial, hide_index=True, width="stretch")
        st.caption(
            "Les lignes sans mesure exploitable n'apparaissent pas ici (distinct d'un score bas mesuré)."
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
