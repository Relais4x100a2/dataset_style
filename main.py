import os

import streamlit as st
from src.auth import render_auth_gate
from src.config import initialize_runtime_config
from src.database import create_db_engine, ensure_schema, get_project_settings, load_project_entries
from src.presets import load_active_dimensions
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


st.set_page_config(page_title="Dataset Style Studio", layout="wide")

db_url = _database_url()
if not db_url:
    st.error("Variable DATABASE_URL requise.")
    st.stop()

engine = create_db_engine(db_url)
ensure_schema(engine)

user = render_auth_gate(engine)
if not user:
    st.stop()

st.title("Dataset Style Studio · Multi-projet")
with st.sidebar:
    project_id, role = render_sidebar(user, engine)

if not project_id:
    st.info("Crée un projet pour commencer.")
    st.stop()

df = load_project_entries(engine, project_id)
project_settings = get_project_settings(engine, project_id)
_, _, dimensions = load_active_dimensions(project_settings)

tab_labels = [
    "Nouvelle entrée",
    "Gestion & édition",
    "Tableau de bord",
    "Projets",
    "Réglages & Export",
    "Mon compte",
]
if user.is_super_admin:
    tab_labels.append("Super Admin")
tabs = st.tabs(tab_labels)
tab1, tab2, tab3, tab4, tab5, tab6, *extra_tabs = tabs
with tab1:
    render_tab_ajout(user, role, project_id, project_settings, df, engine, dimensions)
with tab2:
    render_tab_edition(user, role, project_id, project_settings, df, engine, dimensions)
with tab3:
    render_tab_dashboard(df, role)
with tab4:
    render_tab_projects(user, role, project_id, engine)
with tab5:
    render_tab_settings_export(user, role, project_id, df, engine)
with tab6:
    render_tab_account(user, engine)
if user.is_super_admin and extra_tabs:
    with extra_tabs[0]:
        render_tab_super_admin(user, engine)
