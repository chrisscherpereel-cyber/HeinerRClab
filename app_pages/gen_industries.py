import numpy as np
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.theories import THEORIES
from ui.common import (agent_specs, base_scenario, cached_homogeneous, download, environment_builder, envs_json,
                       reps, show_errors, spec_json, style, theory_color, theory_picker, to_json)

st.title("Single-theory industries")
st.markdown(
    "What kind of **industry** does each theory produce? Every firm in the industry follows the same theory. "
    "This tests the theories' predictions about industrial organization: do markets of optimisers settle at "
    "the Cournot–Nash equilibrium? Do inert or reliability-constrained industries keep prices away from equilibrium "
    "and margins higher (Heiner; the paper's closing conjecture)? Do imitators or biased agents destabilise prices?")
base = base_scenario()
if not show_errors(base):
    st.stop()
with st.form("homo"):
    theories = theory_picker("hi_theories")
    c1, c2 = st.columns(2)
    n_firms = c1.slider("Firms in each industry", 2, 10, 4)
    n_reps = c2.number_input("Replications", 2, 100, max(5, min(reps(), 10)))
    envs = environment_builder("hi_env", "2, 10, 25", "0, 0.05")
    go_btn = st.form_submit_button("Run industries", type="primary")
if not theories:
    st.stop()
specs = agent_specs(theories)
key = (to_json(base), spec_json(specs), int(n_firms), envs_json(envs), int(n_reps))
if go_btn:
    st.session_state["hi_key"] = key
if st.session_state.get("hi_key") != key:
    st.stop()
with st.spinner("Simulating industries…"):
    H = cached_homogeneous(*key)

env_order = [e.name for e in envs]
metrics = [("industry_profit", "Industry profit per period"), ("avg_margin", "Average margin P − c"),
           ("nash_gap", "Mean |P − P_Nash| (distance from equilibrium)"),
           ("price_minus_nash", "Mean P − P_Nash (> 0: above equilibrium)"),
           ("price_change_rms", "Price volatility (RMS change)")]
agg = H.groupby(["theory", "environment"]).agg(**{m: (m, "mean") for m, _ in metrics},
                                                **{m + "_sd": (m, "std") for m, _ in metrics},
                                                n=("rep", "size")).reset_index()
cols = st.columns(2)
for k, (m, title) in enumerate(metrics):
    fig = go.Figure()
    for th in theories:
        g = agg[agg["theory"] == th].set_index("environment").reindex(env_order)
        fig.add_trace(go.Bar(x=env_order, y=g[m], name=THEORIES[th].label, marker_color=theory_color(th),
                             error_y=dict(type="data", array=1.96 * g[m + "_sd"] / np.sqrt(g["n"]), thickness=1, width=3),
                             hovertemplate=f"{THEORIES[th].label}<br>%{{x}}<br>{title}: %{{y:,.2f}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_xaxes(type="category", title=None); fig.update_yaxes(title=title)
    cols[k % 2].plotly_chart(style(fig, 380, title))
st.caption("P_Nash is the Cournot–Nash price for the realised cost and the true demand curve, the static equilibrium "
           "benchmark of neoclassical IO.")
download(H, "single_theory_industries.csv")
