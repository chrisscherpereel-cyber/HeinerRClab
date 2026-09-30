"""Heiner's Reliability Condition - agent-based simulation lab.

Run with:  streamlit run app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.common import render_sidebar  # noqa: E402

st.set_page_config(page_title="Heiner_RC_Lab", page_icon="⚖️", layout="wide")

pages = {
    "Theory": [
        st.Page("app_pages/theory.py", title="Heiner's reliability condition", icon="📘", default=True),
    ],
    "Simulate": [
        st.Page("app_pages/market_lab.py", title="Market lab (single run)", icon="🏭"),
        st.Page("app_pages/hypotheses.py", title="Hypothesis tests", icon="🧪"),
        st.Page("app_pages/rc_validation.py", title="Does the RC predict performance?", icon="🎯"),
        st.Page("app_pages/dynamic_rc.py", title="Dynamic RC (Heiner 1989)", icon="⏱️"),
        st.Page("app_pages/uncertainty.py", title="Risk vs Knightian uncertainty", icon="🌪️"),
        st.Page("app_pages/cd_gap.py", title="CD-gap explorer", icon="🧭"),
        st.Page("app_pages/evolution.py", title="Endogenous flexibility", icon="🧬"),
        st.Page("app_pages/theories.py", title="Competing theories", icon="🏆"),
        st.Page("app_pages/designer.py", title="Experiment designer", icon="🛠️"),
    ],
    "Reference": [
        st.Page("app_pages/model_docs.py", title="Model & methods", icon="📐"),
    ],
}

nav = st.navigation(pages)
render_sidebar()
nav.run()
