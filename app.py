"""Decision making under uncertainty: an agent-based laboratory comparing eight theories.

Run with:  streamlit run app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.common import render_sidebar  # noqa: E402

st.set_page_config(page_title="Decision under uncertainty lab", page_icon="⚖️", layout="wide")

pages = {
    "Theories": [
        st.Page("app_pages/theory_overview.py", title="Overview", icon="🗺️", default=True),
        st.Page("app_pages/theory_heiner.py", title="Heiner: reliability condition", icon="📘"),
        st.Page("app_pages/theory_optimiser.py", title="Neoclassical optimisation", icon="📈"),
        st.Page("app_pages/theory_options.py", title="Real options", icon="🔀"),
        st.Page("app_pages/theory_cobweb.py", title="Cobweb & adaptive expectations", icon="🕸️"),
        st.Page("app_pages/theory_heuristics.py", title="Simple heuristics", icon="🎯"),
        st.Page("app_pages/theory_satisficing.py", title="Satisficing", icon="🎚️"),
        st.Page("app_pages/theory_rl.py", title="Reinforcement learning", icon="🧠"),
        st.Page("app_pages/theory_imitation.py", title="Imitation & selection", icon="🧬"),
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
        st.Page("app_pages/arena.py", title="Agent tournament", icon="🤖"),
        st.Page("app_pages/mechanisms.py", title="Mechanisms", icon="🔬"),
        st.Page("app_pages/rule_choice.py", title="Rule choice (emergence)", icon="🔄"),
        st.Page("app_pages/designer.py", title="Experiment designer", icon="🛠️"),
    ],
    "Validate & generalise": [
        st.Page("app_pages/field_patterns.py", title="Field patterns", icon="📊"),
        st.Page("app_pages/calibration.py", title="Calibration to experiments", icon="📏"),
        st.Page("app_pages/empirical.py", title="Empirical validation (public data)", icon="🗄️"),
        st.Page("app_pages/play_market.py", title="Play the market", icon="🎮"),
        st.Page("app_pages/experiment_analysis.py", title="Experiment analysis", icon="🧾"),
        st.Page("app_pages/generalisation.py", title="Generalisation", icon="🌐"),
        st.Page("app_pages/tracking.py", title="Solvable benchmark (Muth–Kalman)", icon="📐"),
    ],
    "Reference": [
        st.Page("app_pages/literature.py", title="Research & contribution", icon="📚"),
        st.Page("app_pages/model_docs.py", title="Model & methods", icon="📐"),
    ],
}

nav = st.navigation(pages)
render_sidebar()
nav.run()
