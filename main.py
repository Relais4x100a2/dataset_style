import logging
import os

import pandas as pd
import streamlit as st
from src.auth import bootstrap_first_admin, render_auth_gate
from src.config import initialize_runtime_config
from src.database import create_db_engine, ensure_schema, get_project_settings
from src.db_startup import (
    DbFailureCategory,
    classify_database_startup_error,
    effective_database_url,
    is_development_ui,
    technical_hint_for_dev,
    user_facing_summary,
)
from src.presets import load_active_dimensions
from src.project_entries_cache import cached_load_project_entries
from src.tab_layout import EXPECTED_WORKFLOW_TAB_ORDER, main_tab_labels
from src.ui_components import (
    render_needs_active_project_tab_notice,
    render_no_project_onboarding,
    render_post_save_stylometric_feedback_banner,
    render_sidebar,
    render_tab_account,
    render_tab_ajout,
    render_tab_dashboard,
    render_tab_edition,
    render_tab_projects,
    render_tab_settings_export,
    render_tab_super_admin,
)

logger = logging.getLogger(__name__)


def _read_streamlit_database_url() -> str | None:
    """Return database URL from Streamlit secrets when the process env has none."""
    try:
        sec = st.secrets.get("DATABASE_URL")
        if sec:
            return str(sec).strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _render_database_unavailable(
    category: DbFailureCategory,
    *,
    exc: BaseException | None = None,
) -> None:
    """Affiche un message utilisateur compréhensible ; journalise le détail côté serveur."""
    summary = user_facing_summary(category)
    st.error(summary)
    if is_development_ui():
        hint = technical_hint_for_dev(exc, category=category)
        with st.expander("Détails techniques (mode développement)"):
            st.code(hint)


try:
    initialize_runtime_config()
except ValueError as exc:
    logger.exception("Runtime configuration failed (APP_CONFIG_JSON or derived settings)")
    st.set_page_config(page_title="Dataset Style Studio", layout="wide")
    _render_database_unavailable("invalid_config", exc=exc)
    st.stop()

st.set_page_config(page_title="Dataset Style Studio", layout="wide")

db_url = effective_database_url(os.environ, _read_streamlit_database_url())
if not db_url:
    logger.error(
        "No database URL resolved after runtime configuration was applied "
        "(process environment after JSON merge and PostgreSQL derivation, "
        "plus optional Streamlit secrets)."
    )
    _render_database_unavailable("missing_url", exc=None)
    st.stop()

try:
    engine = create_db_engine(db_url)
    ensure_schema(engine)
except Exception as exc:
    logger.exception("Database engine creation or schema initialization failed")
    category = classify_database_startup_error(exc)
    _render_database_unavailable(category, exc=exc)
    st.stop()

bootstrap_first_admin(engine)

user = render_auth_gate(engine)
if not user:
    st.stop()

st.title("Dataset Style Studio · Multi-projet")
with st.sidebar:
    project_id, role = render_sidebar(user, engine)

# Issue-028: keep the tab strip mounted even without ``project_id`` so « Projets » can host
# first-project creation; isolate dataset/settings loads behind a project guard.
if project_id:
    df = cached_load_project_entries(engine, project_id, user.user_id)
    project_settings = get_project_settings(engine, project_id)
    _, _, dimensions = load_active_dimensions(project_settings)
else:
    df = pd.DataFrame()
    project_settings = None
    dimensions = {
        "types": [],
        "structures": [],
        "tons": [],
        "formats": [],
        "publics": [],
        "statuts": [],
    }

if project_id:
    # Issue-021: single render site — each ``st.tabs`` body runs every rerun, so a
    # per-tab renderer would pop session feedback before the curator's active tab runs.
    render_post_save_stylometric_feedback_banner(project_id)

# Issue-007 / issue-024: tab strip follows the curator workflow; titles come from
# ``src.tab_layout`` (``EXPECTED_WORKFLOW_TAB_ORDER`` + ``Mon compte`` [+ Super Admin]).
# Bodies must stay aligned with ``main_tab_labels`` slot order:
# tab1→projects | tab2→settings_export | tab3→ajout | tab4→edition |
# tab5→dashboard | tab6→account | extra_tabs[0]→super_admin (if enabled).
tab_labels = main_tab_labels(include_super_admin=user.is_super_admin)
tabs = st.tabs(tab_labels)
tab1, tab2, tab3, tab4, tab5, tab6, *extra_tabs = tabs
with tab1:
    if not project_id:
        render_no_project_onboarding(user, engine)
    else:
        render_tab_projects(user, role, project_id, engine)
with tab2:
    if not project_id:
        render_needs_active_project_tab_notice(
            target_workflow_tab_title_fr=EXPECTED_WORKFLOW_TAB_ORDER[1],
        )
    else:
        render_tab_settings_export(user, role, project_id, df, engine)
with tab3:
    if not project_id:
        render_needs_active_project_tab_notice(
            target_workflow_tab_title_fr=EXPECTED_WORKFLOW_TAB_ORDER[2],
        )
    else:
        render_tab_ajout(user, role, project_id, project_settings, df, engine, dimensions)
with tab4:
    if not project_id:
        render_needs_active_project_tab_notice(
            target_workflow_tab_title_fr=EXPECTED_WORKFLOW_TAB_ORDER[3],
        )
    else:
        render_tab_edition(user, role, project_id, project_settings, df, engine, dimensions)
with tab5:
    if not project_id:
        render_needs_active_project_tab_notice(
            target_workflow_tab_title_fr=EXPECTED_WORKFLOW_TAB_ORDER[4],
        )
    else:
        render_tab_dashboard(df, role)
with tab6:
    render_tab_account(user, engine)
if user.is_super_admin and extra_tabs:
    with extra_tabs[0]:
        render_tab_super_admin(user, engine)
