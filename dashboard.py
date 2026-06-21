"""
GDIZ Bénin — Supervision Maintenance Prédictive
Usage : streamlit run dashboard.py
"""

import streamlit as st

from ui.constants import _DARK, _LIGHT
from ui.styles import apply_styles
from ui.tabs import tab_about, tab_general, tab_machine

# Configuration de la page
st.set_page_config(
    page_title="GDIZ · Supervision",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Thème
if "theme" not in st.session_state:
    st.session_state.theme = "light"

DARK: bool = st.session_state.theme == "dark"
C = _DARK if DARK else _LIGHT

apply_styles(C, DARK)

# Entête
col_title, col_toggle = st.columns([10, 2])
with col_title:
    st.markdown("## ⚙  GDIZ Bénin · Supervision Maintenance Prédictive")
    st.markdown(
        f"<p style='color:{C['faint']};font-size:.77rem;margin-top:-.35rem;margin-bottom:.3rem;'>"
        "30 machines · Textile · Données 30 min</p>",
        unsafe_allow_html=True,
    )
with col_toggle:
    toggled = st.toggle("LightDark", value=DARK, key="theme_toggle", label_visibility="collapsed")
    if toggled != DARK:
        st.session_state.theme = "dark" if toggled else "light"
        st.rerun()

# Onglets
tab1, tab2, tab3 = st.tabs(
    [
        ":material/dashboard: Dashboard Général",
        ":material/build: Vue Machine",
        ":material/psychology: Modèle & Architecture",
    ]
)
with tab1:
    tab_general.render(C)

with tab2:
    tab_machine.render(C)

with tab3:
    tab_about.render(C)
