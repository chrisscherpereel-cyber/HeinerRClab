import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from heiner_abm.analysis import mean_ci
from heiner_abm.theories import THEORIES, SelectionParams
from ui.common import (DIVERGING, agent_specs, base_scenario, cached_tournament, download, environment_builder,
                       envs_json, reps, show_errors, spec_json, style, theory_color, theory_picker, to_json, verdict)

st.title("Theory tournament")
st.markdown(
    "All selected theories compete **in the same market**: one firm per theory, with the same information, "
    "the same cost shocks and the same demand regime shifts. Each environment is replicated with common "
    "random numbers. The outcome is each theory's **relative profit**, its profit minus the market average. "
    "This tests every theory on equal terms: which way of deciding under uncertainty performs best, and where?")
with st.expander("Fairness conventions"):
    st.markdown(
        "* Every agent observes last price and quantity, all firms' last outputs and profits, last costs, and its "
        "(possibly outdated) demand model. Competence κ and noise σ apply equally to all.\n"
        "* Initial output is scaled so total initial capacity equals the 4-firm base market.\n"
        "* Flexibility is free unless you set flexibility costs; fixed costs apply equally to all.\n"
        "* Parameters are literature defaults; change them in the sidebar under **Agents (theories)**.")

base = base_scenario()
if not show_errors(base):
    st.stop()
with st.form("tourn"):
    theories = theory_picker("tn_theories")
    copies = st.slider("Firms per theory", 1, 3, 1, help="More than one copy lets a theory compete against itself.")
    envs = environment_builder("tn_env")
    c1, c2, c3 = st.columns(3)
    n_reps = c1.number_input("Replications per environment", 2, 200, max(5, min(reps(), 20)))
    use_sel = c2.toggle("Exit when capital runs out", False,
                        help="Firms whose capital (initial endowment + cumulative profit) turns negative exit and are "
                             "replaced by a fresh firm of the same theory. Counts failures.")
    cap0 = c3.number_input("Initial capital", 1000.0, 1e6, 20000.0, 1000.0)
    st.caption(f"{len(envs) * int(n_reps):,} markets × {base.periods:,} periods, "
               f"{len(theories) * copies} firms per market.")
    go_btn = st.form_submit_button("Run tournament", type="primary")
if not theories:
    st.stop()
specs = [s for s in agent_specs(theories) for _ in range(copies)]
sel = SelectionParams(enabled=use_sel, capital0=float(cap0), mode="same")
key = (to_json(base), spec_json(specs), envs_json(envs), int(n_reps), json.dumps(asdict(sel)))
if go_btn:
    st.session_state["tn_key"] = key
if st.session_state.get("tn_key") != key:
    st.stop()
with st.spinner("Running the tournament…"):
    T, I = cached_tournament(*key)

env_order = [e.name for e in envs]
agg = T.groupby(["environment", "theory"]).agg(rel=("rel_profit", "mean"), sd=("rel_profit", "std"), n=("rel_profit", "size"),
                                               profit=("avg_profit", "mean"), act=("avg_absdq", "mean"),
                                               exits=("exits", "mean")).reset_index()
agg["ci"] = 1.96 * agg["sd"] / np.sqrt(agg["n"])
labels = {k: THEORIES[k].label for k in theories}

# ---------------------------------------------------------------- heatmap theory × environment
pv = agg.pivot(index="theory", columns="environment", values="rel").reindex(index=theories, columns=env_order)
lim = float(np.nanmax(np.abs(pv.to_numpy()))) or 1.0
fig = go.Figure(go.Heatmap(z=pv.to_numpy(), x=env_order, y=[labels[k] for k in pv.index], colorscale=DIVERGING,
                           zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2, texttemplate="%{z:,.0f}",
                           hovertemplate="%{y}<br>%{x}<br>relative profit %{z:,.1f}<extra></extra>"))
fig.update_xaxes(type="category", title=None); fig.update_yaxes(autorange="reversed", title=None)
st.plotly_chart(style(fig, 70 + 42 * len(theories), "Relative profit per period (blue = above market average)"))

wins = agg.loc[agg.groupby("environment")["rel"].idxmax()].set_index("environment").reindex(env_order)
st.markdown("**Winner by environment:** " + " · ".join(f"{e}: **{labels[w]}**" for e, w in wins["theory"].items()))

# ---------------------------------------------------------------- one environment in detail
st.subheader("One environment in detail")
e_sel = st.selectbox("Environment", env_order)
a = agg[agg["environment"] == e_sel].sort_values("rel")
c1, c2 = st.columns(2)
fb = go.Figure(go.Bar(x=a["rel"], y=[labels[k] for k in a["theory"]], orientation="h",
                      marker_color=[theory_color(k) for k in a["theory"]],
                      error_x=dict(type="data", array=a["ci"], thickness=1.2, width=4),
                      hovertemplate="%{y}<br>relative profit %{x:,.1f}<extra></extra>"))
fb.add_vline(x=0, line=dict(color="rgba(128,128,128,0.8)", width=1))
fb.update_xaxes(title="Relative profit per period (95% CI)")
c1.plotly_chart(style(fb, 60 + 38 * len(a), "Who earns more than the market?"))
fs = go.Figure()
for _, r in a.iterrows():
    fs.add_trace(go.Scatter(x=[r["act"]], y=[r["rel"]], mode="markers+text", text=[labels[r["theory"]].split(" ")[0]],
                            textposition="top center", name=labels[r["theory"]], showlegend=False,
                            marker=dict(size=13, color=theory_color(r["theory"]), line=dict(color="white", width=2)),
                            hovertemplate=f"{labels[r['theory']]}<br>activity %{{x:.2f}}<br>relative profit %{{y:,.1f}}<extra></extra>"))
fs.add_hline(y=0, line=dict(color="rgba(128,128,128,0.8)", width=1))
fs.update_xaxes(title="Behavioural activity: mean |Δq| per period", type="log")
fs.update_yaxes(title="Relative profit per period")
rho = stats.spearmanr(a["act"], a["rel"])[0] if len(a) > 2 else np.nan
c2.plotly_chart(style(fs, 60 + 38 * len(a), f"Flexibility vs performance across theories (ρ = {rho:.2f})"))

rhos = {e: stats.spearmanr(g["act"], g["rel"])[0] for e, g in agg.groupby("environment") if len(g) > 2}
rr = pd.Series(rhos).reindex(env_order)
st.markdown("**Across-theory rank correlation between activity and profit** (Heiner predicts it turns negative as "
            "uncertainty rises): " + " · ".join(f"{e}: ρ = {v:.2f}" for e, v in rr.items()))
if len(rr.dropna()) >= 2 and rr.iloc[-1] < rr.iloc[0]:
    verdict("support", "Behavioural activity is rewarded less as the environment becomes harder, as Heiner predicts "
            "for flexible versus constrained decision makers.")
elif len(rr.dropna()) >= 2:
    verdict("neutral", "The activity–profit relationship does not weaken across these environments.")

if use_sel:
    st.subheader("Failures")
    ex = agg.pivot(index="theory", columns="environment", values="exits").reindex(index=theories, columns=env_order)
    ex.index = [labels[k] for k in ex.index]
    st.dataframe(ex.style.format("{:.2f}"), width="stretch")
    st.caption("Mean number of exits per firm slot and run (after burn-in).")

st.subheader("Industry outcomes")
st.dataframe(I.groupby("environment")[["avg_margin", "price_change_rms", "nash_gap", "industry_profit", "exits"]]
             .mean().reindex(env_order), width="stretch")
c1, c2 = st.columns(2)
with c1:
    download(T, "tournament_firms.csv", "Download theory-level results")
with c2:
    download(I, "tournament_industry.csv", "Download industry results")
