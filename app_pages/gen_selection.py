import json
from dataclasses import asdict

import plotly.graph_objects as go
import streamlit as st

from heiner_abm.theories import THEORIES, SelectionParams
from ui.common import (agent_specs, base_scenario, cached_selection, download, environment_builder, envs_json, reps,
                       show_errors, spec_json, style, theory_color, theory_picker, to_json, verdict)

st.title("Ecological selection of decision forms")
st.markdown(
    "Organisational ecology argues that industries adapt mainly through **selection**: firms rarely change, "
    "but poorly fitted ones fail and are replaced. Here every firm has capital (an initial endowment plus cumulative "
    "profit). A firm whose capital turns negative **exits**, and an entrant takes its slot. The entrant's form "
    "(theory) is drawn from the surviving firms in that market, so successful forms reproduce, with a small "
    "mutation probability of any form. The population shares show which ways of deciding the market selects for, "
    "in calm, risky and Knightian environments.")
base = base_scenario()
if not show_errors(base):
    st.stop()
with st.form("sel"):
    theories = theory_picker("sl_theories")
    copies = st.slider("Initial firms per theory", 1, 3, 1)
    envs = environment_builder("sl_env", "2, 25", "0, 0.05")
    c = st.columns(4)
    n_reps = c[0].number_input("Replications", 2, 100, max(6, min(reps(), 12)))
    periods = c[1].number_input("Periods", 500, 20000, 3000, 500)
    cap0 = c[2].number_input("Initial capital", 1000.0, 1e6, 20000.0, 1000.0)
    mut = c[3].slider("Mutation probability", 0.0, 0.5, 0.02, 0.01)
    mode = st.radio("Entrants are…", ["survivors", "same"], horizontal=True,
                    format_func=lambda m: {"survivors": "drawn from surviving forms (selection)",
                                           "same": "the same theory (count failures only)"}[m])
    go_btn = st.form_submit_button("Run selection", type="primary")
if not theories:
    st.stop()
specs = [s for s in agent_specs(theories) for _ in range(copies)]
sel = SelectionParams(enabled=True, capital0=float(cap0), mode=mode, mutation=float(mut))
key = (to_json(base), spec_json(specs), envs_json(envs), int(n_reps), json.dumps(asdict(sel)), int(periods))
if go_btn:
    st.session_state["sl_key"] = key
if st.session_state.get("sl_key") != key:
    st.stop()
with st.spinner("Running selection…"):
    paths, summ = cached_selection(*key)

env_order = [e.name for e in envs]
cols = st.columns(2)
for k, e in enumerate(env_order):
    g = paths[paths["environment"] == e]
    fig = go.Figure()
    for th in theories:
        d = g[g["theory"] == th]
        fig.add_trace(go.Scatter(x=d["period"], y=d["share"], name=THEORIES[th].label, mode="lines",
                                 line=dict(color=theory_color(th), width=2), stackgroup="one",
                                 hovertemplate=f"{THEORIES[th].label}<br>period %{{x}}<br>share %{{y:.1%}}<extra></extra>"))
    fig.update_xaxes(title="Period"); fig.update_yaxes(title="Share of firms", range=[0, 1], tickformat=".0%")
    fig.update_layout(hovermode="x unified")
    cols[k % 2].plotly_chart(style(fig, 360, e))

st.subheader("Survival and selection")
tab = summ.copy()
tab["label"] = tab["theory"].map(lambda k: THEORIES[k].label)
tab["share_change"] = tab["final_share"] - tab["initial_share"]
st.dataframe(tab[["environment", "label", "initial_share", "final_share", "share_change", "exit_rate", "avg_profit"]],
             hide_index=True, width="stretch", column_config={
                 "initial_share": st.column_config.NumberColumn("Initial share", format="%.2f"),
                 "final_share": st.column_config.NumberColumn("Final share", format="%.2f"),
                 "share_change": st.column_config.NumberColumn("Change", format="%+.2f"),
                 "exit_rate": st.column_config.NumberColumn("Exits per 1,000 firm-periods", format="%.2f"),
                 "avg_profit": st.column_config.NumberColumn("Avg profit", format="%.0f")})
if mode == "survivors":
    for e in env_order:
        t = tab[tab["environment"] == e].sort_values("share_change", ascending=False)
        if len(t) and t["share_change"].abs().max() > 0:
            verdict("neutral", f"**{e}:** selection favours **{t.iloc[0]['label']}** ({t.iloc[0]['share_change']:+.0%}) "
                    f"and disfavours **{t.iloc[-1]['label']}** ({t.iloc[-1]['share_change']:+.0%}).")
download(tab, "selection_summary.csv")
download(paths, "selection_paths.csv", "Download share paths")
