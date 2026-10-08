"""Streamlit entrypoint for the Employee Attrition Analytics application."""

import streamlit as st

from attrition.app_pages import render_methodology, render_overview

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide",
)

pages = [
    st.Page(
        render_overview,
        title="Overview",
        icon=":material/home:",
        url_path="overview",
        default=True,
    ),
    st.Page(
        render_methodology,
        title="Model & Methodology",
        icon=":material/model_training:",
        url_path="methodology",
    ),
]

with st.sidebar:
    st.markdown("### Employee Attrition Analytics")
    st.caption("Reproducible ML • calibrated risk • explainability")

navigation = st.navigation(pages)
navigation.run()
