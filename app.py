"""Decision making under uncertainty: an agent-based laboratory comparing nine theories.

Run with:  streamlit run app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.common import render_sidebar  # noqa: E402

st.set_page_config(page_title="Decision under uncertainty lab", page_icon="⚖️", layout="wide")

SECTIONS = {
    "Start": [
        st.Page("app_pages/start.py", title="Start", icon="🧭", default=True),
        st.Page("app_pages/hypotheses.py", title="Hypothesis tests", icon="🧪"),
        st.Page("app_pages/uncertainty.py", title="Risk vs structural change", icon="🌪️"),
    ],
    "Experiment": [
        st.Page("app_pages/experiment.py", title="Experiment", icon="🛠️"),
        st.Page("app_pages/market_lab.py", title="Market lab (single run)", icon="🏭"),
        st.Page("app_pages/designer.py", title="Experiment designer (market sweeps)", icon="📐"),
        st.Page("app_pages/arena.py", title="Agent tournament", icon="🤖"),
        st.Page("app_pages/rule_choice.py", title="Rule choice (emergence)", icon="🔄"),
        st.Page("app_pages/evolution.py", title="Endogenous flexibility", icon="🧬"),
        st.Page("app_pages/theories.py", title="Competing theories", icon="🏆"),
        st.Page("app_pages/generalisation.py", title="Generalization tasks", icon="🌐"),
        st.Page("app_pages/learnability.py", title="When can reliability be learned?", icon="⏳"),
        st.Page("app_pages/nk.py", title="NK landscapes (complexity)", icon="🏔️"),
    ],
    "Results": [
        st.Page("app_pages/results.py", title="Results", icon="📊"),
        st.Page("app_pages/mechanisms.py", title="Mechanisms: oracle vs learned", icon="🔬"),
        st.Page("app_pages/reliability_gates.py", title="Reliability gates (extension)", icon="🚦"),
        st.Page("app_pages/dynamic_rc.py", title="Dynamic RC (1989)", icon="⏱️"),
        st.Page("app_pages/cd_gap.py", title="CD-gap explorer", icon="🧭"),
    ],
    "Validation": [
        st.Page("app_pages/validation_hub.py", title="Validation", icon="✅"),
        st.Page("app_pages/tracking.py", title="Heiner vs optimal filtering (Muth–Kalman)", icon="📐"),
        st.Page("app_pages/benchmarks.py", title="Decision benchmarks (Bayes, robust, bandit)", icon="🧮"),
        st.Page("app_pages/special_tests.py", title="Signature tests by theory", icon="🔎"),
        st.Page("app_pages/rc_validation.py", title="Does the RC predict performance?", icon="🎯"),
        st.Page("app_pages/field_patterns.py", title="Field patterns", icon="📊"),
        st.Page("app_pages/calibration.py", title="Calibration to experiments", icon="📏"),
        st.Page("app_pages/empirical.py", title="Empirical validation (public data)", icon="🗄️"),
        st.Page("app_pages/play_market.py", title="Human experiments: play the market", icon="🎮"),
        st.Page("app_pages/experiment_analysis.py", title="Human experiments: analysis", icon="🧾"),
    ],
    "Reference": [
        st.Page("app_pages/reference_hub.py", title="Reference", icon="📚"),
        st.Page("app_pages/theory_overview.py", title="Theories: overview", icon="🗺️"),
        st.Page("app_pages/theory_heiner.py", title="Heiner: reliability condition", icon="📘"),
        st.Page("app_pages/theory_optimiser.py", title="Neoclassical optimization", icon="📈"),
        st.Page("app_pages/theory_options.py", title="Real options", icon="🔀"),
        st.Page("app_pages/theory_cobweb.py", title="Cobweb & adaptive expectations", icon="🕸️"),
        st.Page("app_pages/theory_heuristics.py", title="Simple heuristics", icon="🎯"),
        st.Page("app_pages/theory_satisficing.py", title="Satisficing", icon="🎚️"),
        st.Page("app_pages/theory_rl.py", title="Reinforcement learning", icon="🧠"),
        st.Page("app_pages/theory_imitation.py", title="Imitation & selection", icon="🧬"),
        st.Page("app_pages/theory_ecology.py", title="Organizational ecology", icon="🏛️"),
        st.Page("app_pages/agents_reference.py", title="Agents as implemented", icon="🤖"),
        st.Page("app_pages/information.py", title="Information & feedback", icon="👁️"),
        st.Page("app_pages/model_docs.py", title="Model & methods", icon="📐"),
        st.Page("app_pages/literature.py", title="Bibliography & contribution", icon="📚"),
    ],
}

nav = st.navigation(SECTIONS, position="hidden")
render_sidebar(SECTIONS, nav)
st.session_state["_info_engine"] = ("market", None)   # pages on other engines override this
nav.run()
