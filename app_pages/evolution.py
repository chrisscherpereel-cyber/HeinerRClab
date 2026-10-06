import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.analysis import ols
from ui.common import (CAT, base_scenario, behaviour_horizon, cached_evolution, download, fmt_p, hypothesis_card, measurement, reps,
                       show_errors, style, to_json, verdict)
from ui.common import evidence_note

st.title("Endogenous flexibility: what do firms choose?")
evidence_note("simulation")
hypothesis_card(
    "EVO",
    "Every *K* periods each firm imitates the most profitable rival's flexibility φ with some probability and then "
    "experiments (mutation). This is social learning, with no optimization. The question is which flexibility the "
    "industry evolves toward as cost volatility rises.",
    notes={"heiner": "If firms can choose their flexibility, decision errors push them toward constrained behavior as "
                     "uncertainty rises: they compete less aggressively without colluding, and the industry settles at "
                     "positive margins suited to its uncertainty.",
           "options": "More volatility raises the value of flexibility, so selection should favor flexible firms."})

base = base_scenario()
if not show_errors(base):
    st.stop()
H, cont = measurement()
H = behaviour_horizon(base)

with st.form("evo"):
    c = st.columns(4)
    deltas_txt = c[0].text_input("Volatility levels Δ", "3, 10, 20, 30",
                                 help="Comma-separated cost volatilities; flexibility evolves separately at each level.")
    periods = c[1].number_input("Periods", 500, 50000, max(base.periods, 5000), step=500,
                                help="Length of each run. Evolution needs many revision rounds, so runs are long.")
    every = c[2].number_input("Revise every K periods", 5, 1000, 50,
                              help="Periods between flexibility revisions. Profits are compared over each K-period window.")
    n_reps = c[3].number_input("Replications", 1, 100, max(reps() // 2, 6),
                               help="Independent runs per volatility level.")
    c = st.columns(4)
    imit = c[0].slider("Imitation probability", 0.0, 1.0, 0.5,
                       help="Chance that, at each revision, a firm copies the φ of the most profitable firm.")
    mut = c[1].slider("Mutation s.d.", 0.0, 0.3, 0.03, 0.005,
                      help="Standard deviation of the random experiment added to every firm's φ at each revision.")
    fmax = c[2].number_input("Max φ", 0.1, 10.0, 2.0 if base.firms[0].rule == "Bertrand" else 1.0, step=0.1,
                             help="Upper limit on evolved flexibility (Cournot is always capped at 1); the lower limit is 0.")
    start_same = c[3].toggle("All firms start at the same φ", False,
                             help="Off = firms start at the sidebar's φ ladder. On = everyone starts at its mean.")
    run = st.form_submit_button("Run evolution", type="primary")

try:
    deltas = [float(x) for x in deltas_txt.split(",") if x.strip()]
except ValueError:
    st.error("Could not parse volatility levels.")
    st.stop()
scn = base.copy(periods=int(periods))
scn.evolution.every, scn.evolution.imitation_prob, scn.evolution.mutation_sd = int(every), float(imit), float(mut)
scn.evolution.flex_max = float(fmax)
if start_same:
    mean = float(np.mean([f.flex for f in scn.firms]))
    for f in scn.firms:
        f.flex = mean
key = (to_json(scn), tuple(deltas), int(n_reps), H, cont)
if run:
    st.session_state["evo_key"] = key
if st.session_state.get("evo_key") != key:
    st.stop()
if H > 1:
    st.caption(f"ORACLE treatment: Adaptive firms learn from the researcher's counterfactual over H = {H} periods, "
               "released once those periods have occurred. Evolution itself uses realized profits.")
with st.spinner("Evolving industries…"):
    paths, summ = cached_evolution(*key)

c1, c2 = st.columns(2)
f1, f2 = go.Figure(), go.Figure()
for k, (d, g) in enumerate(paths.groupby("delta")):
    col = CAT[k % len(CAT)]
    f1.add_trace(go.Scatter(x=np.r_[g["period"], g["period"][::-1]], y=np.r_[g["flex_hi"], g["flex_lo"][::-1]],
                            fill="toself", fillcolor=col, opacity=0.12, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f1.add_trace(go.Scatter(x=g["period"], y=g["flex_mean"], name=f"Δ = {d:g}", line=dict(color=col, width=2)))
    f2.add_trace(go.Scatter(x=g["period"], y=g["margin"].rolling(5, min_periods=1).mean(), name=f"Δ = {d:g}",
                            line=dict(color=col, width=2)))
f1.update_xaxes(title="Period"); f1.update_yaxes(title="Industry mean φ (band = 10th–90th pct across runs)")
f2.update_xaxes(title="Period"); f2.update_yaxes(title="Margin P − c (smoothed)")
f1.update_layout(hovermode="x unified"); f2.update_layout(hovermode="x unified")
c1.plotly_chart(style(f1, 380, "Evolution of flexibility"))
c2.plotly_chart(style(f2, 380, "Industry margin"))

agg = summ.groupby("delta").agg(flex_start=("flex_start", "mean"), flex_end=("flex_end", "mean"),
                                flex_end_sd=("flex_end", "std"), margin_first=("margin_first_q", "mean"),
                                margin_last=("margin_last_q", "mean"), sc_cost=("sc_cost", "mean")).reset_index()
st.dataframe(agg, hide_index=True, width="stretch", column_config={
    "delta": "Δ", "flex_start": st.column_config.NumberColumn("φ start", format="%.3f"),
    "flex_end": st.column_config.NumberColumn("φ end", format="%.3f"),
    "flex_end_sd": st.column_config.NumberColumn("s.d. across runs", format="%.3f"),
    "margin_first": st.column_config.NumberColumn("Margin, first quarter", format="%.2f"),
    "margin_last": st.column_config.NumberColumn("Margin, last quarter", format="%.2f"),
    "sc_cost": st.column_config.NumberColumn("Serial corr. of cost", format="%.3f")})

if summ["delta"].nunique() > 1:
    tab, r2 = ols(summ["flex_end"].to_numpy(float), [summ["delta"].to_numpy(float)], ["Δ"])
    coef, p = tab.loc[1, "coef"], tab.loc[1, "p"]
    st.markdown(f"Regression of evolved φ on Δ: coefficient **{coef:+.4f}**, p = {fmt_p(p)}, R² = {r2:.3f}.")
    if coef < 0 and p < 0.05:
        verdict("support", "Industries facing more volatile costs **evolve toward rigidity**, as the reliability condition predicts.")
    elif coef > 0 and p < 0.05:
        verdict("reject", "Industries facing more volatile costs evolve toward **more** flexibility, as the traditional view predicts.")
    else:
        verdict("neutral", "Evolved flexibility does not depend significantly on volatility in this setting.")
st.caption("Tip: the effect is sharpest in low-margin markets with the margin-feedback rule (e.g. m* = 1), where mistakes are costly. "
           "In Cournot markets rigidity tends to win at every volatility level.")
download(summ, "evolution_runs.csv")
