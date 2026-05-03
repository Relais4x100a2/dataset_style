"""
UI multi-utilisateur / multi-projet.
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from copy import deepcopy
from dataclasses import replace
from datetime import datetime

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
from src.export_utils import convert_to_jsonl
from src.llm_generate import generate_input_from_output, generate_output_from_input
from src.mailer import send_account_link_email
from src.nlp_engine import corriger_texte_fr
from src.presets import (
    DIMENSION_KEYS,
    PRESETS,
    available_presets,
    dumps_custom_presets,
    dumps_dimensions_override,
    load_active_dimensions,
    preset_dimensions,
)

logger = logging.getLogger(__name__)


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
        _persist_settings(user, engine, project_id, next_settings, "Preset personnalisé enregistré.")


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
            st.success("Projet supprimé.")
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

    confirm = st.checkbox("Je confirme vouloir supprimer mon compte", key="account_self_delete_confirm")
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
            logout()
            st.success("Compte supprimé.")
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
                st.success(f"Memberships détachées: {removed}")
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
            st.success("Compte supprimé.")
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
            st.success("Opération remise en file d'attente.")
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
    csv = df[df["statut"] == STATUT_VALIDE].to_csv(index=False).encode("utf-8")
    export_format = st.selectbox(
        "Format JSONL",
        ["lfm2", "baguettotron", "mistral"],
        key="export_format_select",
    )
    jsonl_data = convert_to_jsonl(df, export_format, include_stylometry=True)
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
    """Ajout d'entrée (collaborator/admin)."""
    st.subheader("Nouvelle entrée")
    if role == "viewer":
        st.info("Lecture seule (viewer).")
        return
    _llm_env(project_settings)
    with st.form("new_entry_form"):
        type_ = st.selectbox("Type de transformation", dimensions["types"])
        structure = st.selectbox("Structure textuelle", dimensions["structures"])
        ton = st.selectbox("Tonalité textuelle", dimensions["tons"])
        format_ = st.selectbox("Format de sortie", dimensions["formats"])
        public = st.selectbox("Public cible", dimensions["publics"])
        input_text = st.text_area("Brouillon", height=120)
        output_text = st.text_area("Texte généré", height=220)
        statut = st.selectbox("Statut", dimensions["statuts"])
        notes = st.text_input("Notes")
        col1, col2, col3 = st.columns(3)
        gen_out = col1.form_submit_button("Générer texte")
        gen_in = col2.form_submit_button("Générer brouillon")
        save = col3.form_submit_button("Enregistrer", type="primary")
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
                st.session_state["new_generated_output"] = generated
                st.toast("Texte généré.")
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
                st.session_state["new_generated_input"] = generated
                st.toast("Brouillon généré.")
            else:
                st.error("La génération a échoué. Vérifiez vos paramètres LLM puis réessayez.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur génération brouillon", exc_info=exc)
            st.error("Génération impossible: erreur inattendue côté service.")
    if save:
        if not input_text.strip() or not output_text.strip():
            st.error("Brouillon/Texte généré obligatoires.")
            return
        new_row = pd.DataFrame(
            [
                {
                    "id": str(uuid.uuid4())[:8],
                    "project_id": project_id,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "type": type_,
                    "structure": structure,
                    "ton": ton,
                    "format": format_,
                    "public": public,
                    "input": input_text,
                    "output": output_text,
                    "statut": statut,
                    "notes": notes,
                    **{c: "" for c in CACHE_COLUMNS},
                }
            ]
        )
        require_role(engine, project_id, user.user_id, ("admin", "collaborator"))
        update_project_entries(engine, project_id, pd.concat([df, new_row], ignore_index=True))
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
        row["output"] = st.text_area("Texte généré", value=row["output"], height=240, disabled=disabled)
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
            corrected = corriger_texte_fr(
                str(row["output"]),
                languagetool_base_url=project_settings.languagetool_base_url or None,
            )
            st.info(corrected[:1500] + ("..." if len(corrected) > 1500 else ""))
        except requests.RequestException as exc:
            st.error(f"Correction impossible: {exc}")
    if save:
        require_role(engine, project_id, user.user_id, ("admin", "collaborator"))
        out = df.copy()
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
        update_project_entries(engine, project_id, out)
        st.success("Entrée mise à jour.")
        st.rerun()


def render_tab_dashboard(df: pd.DataFrame, role: str) -> None:
    st.subheader("Tableau de bord")
    st.caption(f"Rôle projet: {role}")
    if df.empty:
        st.info("Aucune donnée.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", len(df))
    c2.metric("Validées", int((df["statut"] == STATUT_VALIDE).sum()))
    c3.metric("Types", int(df["type"].nunique()))
    if "structure" not in df.columns and "forme" in df.columns:
        df = df.copy()
        df["structure"] = df["forme"]
    if "format" not in df.columns and "support" in df.columns:
        df = df.copy()
        df["format"] = df["support"]
    if "public" not in df.columns:
        df = df.copy()
        df["public"] = ""
    st.dataframe(
        df[["id", "date", "type", "structure", "ton", "format", "public", "statut"]],
        hide_index=True,
        width="stretch",
    )
