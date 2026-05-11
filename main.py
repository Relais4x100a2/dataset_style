import logging
import os

import streamlit as st
from src.auth import bootstrap_first_admin, render_auth_gate
from src.config import initialize_runtime_config
from src.database import create_db_engine, ensure_schema, get_project_settings, load_project_entries
from src.db_startup import (
    DbFailureCategory,
    classify_database_startup_error,
    is_development_ui,
    technical_hint_for_dev,
    user_facing_summary,
)
from src.presets import load_active_dimensions
from src.tab_layout import main_tab_labels
from src.ui_components import (
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

initialize_runtime_config()


def _database_url() -> str:
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if url:
        return url
    try:
        sec = st.secrets.get("DATABASE_URL")
        if sec:
            return str(sec).strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


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


st.set_page_config(page_title="Dataset Style Studio", layout="wide")

db_url = _database_url()
if not db_url:
    logger.error(
        "Database URL missing after runtime config "
        "(set DATABASE_URL, APP_CONFIG_JSON, POSTGRES_*-derived URL, or Streamlit secrets)."
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

if not project_id:
    st.info("Crée un projet pour commencer.")
    st.stop()

df = load_project_entries(engine, project_id, user.user_id)
project_settings = get_project_settings(engine, project_id)
_, _, dimensions = load_active_dimensions(project_settings)

tab_labels = main_tab_labels(include_super_admin=user.is_super_admin)
tabs = st.tabs(tab_labels)
tab1, tab2, tab3, tab4, tab5, tab6, *extra_tabs = tabs
with tab1:
    render_tab_projects(user, role, project_id, engine)
with tab2:
    render_tab_settings_export(user, role, project_id, df, engine)
with tab3:
    render_tab_ajout(user, role, project_id, project_settings, df, engine, dimensions)
with tab4:
    render_tab_edition(user, role, project_id, project_settings, df, engine, dimensions)
with tab5:
    render_tab_dashboard(df, role)
with tab6:
    render_tab_account(user, engine)
if user.is_super_admin and extra_tabs:
    with extra_tabs[0]:
        render_tab_super_admin(user, engine)
