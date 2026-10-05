import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.general import STANDARD_ENVS
from heiner_abm.theories import GENERAL_THEORIES, THEORIES
from ui.common import (DIVERGING, agent_specs, base_scenario, cached_scorecard, download, reps, show_errors,
                       spec_json, style, theory_color, to_json)

st.title("Theory scorecard")
st.markdown(
    "Every theory makes a falsifiable claim about decision making under uncertainty. The scorecard tests **each "
    "theory's own claim** in one common battery of simulations in which all theories act as agents:\n\n"
    "1. a **tournament** of all eight theories in three environments: *calm* (Δ = 2), *risk* (Δ = 25) and "
    "*Knightian* (Δ = 5 with unannounced demand-regime shifts, λ = 0.05);\n"
    "2. the same tournament with the biased agent replaced by its **debiased twin** (a paired control);\n"
    "3. **single-theory industries** (equilibrium outcomes);\n"
    "4. **ecological selection** with exit, survivor-weighted entry and mutation.\n\n"
    "All conditions share common random numbers. Differences are paired across replications, with 95% CIs.")
base = base_scenario()
if not show_errors(base):
    st.stop()
with st.form("sc"):
    c = st.columns(3)
    n_reps = c[0].number_input("Replications", 4, 100, max(10, min(reps(), 20)))
    periods = c[1].number_input("Periods in the selection runs", 500, 10000, 2000, 500)
    cap0 = c[2].number_input("Initial capital (selection)", 1000.0, 1e6, 20000.0, 1000.0)
    go_btn = st.form_submit_button("Run scorecard", type="primary")
specs = agent_specs(GENERAL_THEORIES)
key = (to_json(base), spec_json(specs), int(n_reps), int(periods), float(cap0))
if go_btn:
    st.session_state["sc_key"] = key
if st.session_state.get("sc_key") != key:
    st.info("The scorecard uses the agent parameters in the sidebar (Agents → each theory) and the base market. "
            f"Runs ≈ {int(n_reps) * 3 * 2 + 4 * 3 * max(3, int(n_reps) // 2) + 2 * max(4, int(n_reps) // 2):,} "
            "markets.", icon="ℹ️")
    st.stop()
with st.spinner("Running the battery (tournament, control, industries, selection)…"):
    sc, det = cached_scorecard(*key)

n_sup = (sc["verdict"] == "Supported").sum()
st.metric("Claims supported", f"{n_sup} of {len(sc)}")
for th in GENERAL_THEORIES:
    rows = sc[sc["theory"] == th]
    if rows.empty:
        continue
    with st.container(border=True):
        st.markdown(f"<span style='color:{theory_color(th)};font-size:1.2em'>■</span> **{THEORIES[th].label}** "
                    f"· <span style='opacity:0.7'>{THEORIES[th].literature}</span>", unsafe_allow_html=True)
        for _, r in rows.iterrows():
            ok = r["verdict"] == "Supported"
            ci = (f" (95% CI {r['ci_low']:,.2f} to {r['ci_high']:,.2f}, p = {r['p_value']:.3f})"
                  if pd.notna(r["ci_low"]) else "")
            (st.success if ok else st.error)(
                f"**{r['verdict']}:** {r['claim']}\n\n{r['statistic']}: **{r['estimate']:,.2f}**{ci}"
                + (f"\n\n{r['note']}" if isinstance(r["note"], str) and r["note"] else ""),
                icon="✅" if ok else "❌")

st.subheader("Evidence: tournament")
T = det["tournament"]
env_order = [e.name for e in STANDARD_ENVS]
pv = T.groupby(["theory", "environment"])["rel_profit"].mean().unstack().reindex(index=GENERAL_THEORIES, columns=env_order)
lim = float(np.nanmax(np.abs(pv.to_numpy()))) or 1.0
fig = go.Figure(go.Heatmap(z=pv.to_numpy(), x=env_order, y=[THEORIES[k].label for k in pv.index], colorscale=DIVERGING,
                           zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2, texttemplate="%{z:,.0f}",
                           hovertemplate="%{y}<br>%{x}<br>relative profit %{z:,.1f}<extra></extra>"))
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style(fig, 420, "Relative profit per period by theory and environment"))

st.subheader("Evidence: selection")
p = det["selection_paths"]
cols = st.columns(2)
for k, (e, g) in enumerate(p.groupby("environment", sort=False)):
    f = go.Figure()
    for th in GENERAL_THEORIES:
        d = g[g["theory"] == th]
        f.add_trace(go.Scatter(x=d["period"], y=d["share"], name=THEORIES[th].label, mode="lines", stackgroup="one",
                               line=dict(color=theory_color(th), width=1.5)))
    f.update_yaxes(range=[0, 1], tickformat=".0%", title="Share of firms"); f.update_xaxes(title="Period")
    cols[k % 2].plotly_chart(style(f, 340, e))

st.subheader("How to read this")
st.markdown(
    "* Each claim is the theory's *own* prediction, stated before running. A theory can be supported on its own "
    "terms and still lose the tournament, and vice versa.\n"
    "* Agents use literature-default parameters. Results depend on them, so explore sensitivity in the sidebar.\n"
    "* The neoclassical claim ('optimisers earn the most everywhere') is the traditional hypothesis the paper "
    "tests. Heiner's claims concern *relative* performance as uncertainty rises and the sign of the "
    "flexibility–performance relationship.")
download(sc, "scorecard.csv")
