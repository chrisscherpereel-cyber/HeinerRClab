import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.theories import GENERAL_THEORIES, THEORIES, SelectionParams
from ui.common import (COST_C, PRICE_C, agent_spec, base_scenario, cached_theory_market, download, show_errors,
                       spec_json, style, theory_color, to_json)

st.title("Multi-agent market lab")
st.markdown("One market, one run, with any line-up of theory agents. Watch how each way of deciding responds to the "
            "same cost shocks and demand-regime shifts. The agents are described in the sidebar under "
            "**Agents (theories)**.")
base = base_scenario()
if not show_errors(base):
    st.stop()
opts = GENERAL_THEORIES + ["paper"]
if "gm_lineup" not in st.session_state:
    st.session_state["gm_lineup"] = pd.DataFrame({"theory": [THEORIES[k].label for k in GENERAL_THEORIES]})
with st.expander("Line-up (one row per firm)", expanded=True):
    lab2key = {THEORIES[k].label: k for k in opts}
    ed = st.data_editor(st.session_state["gm_lineup"], num_rows="dynamic", width="stretch", key="gm_editor",
                        column_config={"theory": st.column_config.SelectboxColumn(
                            "Theory", options=[THEORIES[k].label for k in opts], required=True)})
    c = st.columns(4)
    delta = c[0].number_input("Cost volatility Δ", 0.0, 60.0, float(base.market.delta), 0.5)
    hazard = c[1].number_input("Regime-shift hazard λ", 0.0, 0.2, float(base.structural.hazard if base.structural.enabled else 0.0), 0.005)
    seed = c[2].number_input("Seed", 0, 10**7, int(base.seed))
    use_sel = c[3].toggle("Exit when capital runs out", False)
keys_ = [lab2key[x] for x in ed["theory"].dropna() if x in lab2key]
if not keys_:
    st.warning("Add at least one firm.")
    st.stop()
specs = [agent_spec(k) for k in keys_]
scn = base.copy(seed=int(seed))
scn.market.delta = float(delta)
scn.structural.enabled = hazard > 0
scn.structural.hazard = float(hazard)
scn.firm_globals.q0 = round(base.firm_globals.q0 * 4 / len(specs))
sel = SelectionParams(enabled=use_sel, mode="same")
with st.spinner("Running…"):
    out = cached_theory_market(to_json(scn), spec_json(specs), json.dumps(asdict(sel)))

T = len(out["price"]); t = np.arange(T); burn = scn.burn_in
tab = out["table"]
names = [f"{i + 1}. {THEORIES[k].label}" for i, k in enumerate(keys_)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=out["cost"], name="Raw-material cost", line=dict(color=COST_C, width=1.2)))
fig.add_trace(go.Scatter(x=t, y=out["nash"], name="Cournot–Nash price", line=dict(color="#8a8984", width=1.2, dash="dot")))
fig.add_trace(go.Scatter(x=t, y=out["price"], name="Market price", line=dict(color=PRICE_C, width=2)))
for ts in np.flatnonzero(out["shifts"]):
    fig.add_vline(x=int(ts), line=dict(color="rgba(74,58,167,0.4)", width=1, dash="dot"))
fig.update_xaxes(title="Period", rangeslider=dict(visible=True, thickness=0.05)); fig.update_yaxes(title="Price / cost")
fig.update_layout(hovermode="x unified")
st.plotly_chart(style(fig, 380, "Price, cost and the static equilibrium benchmark (dotted lines = regime shifts)"))
c1, c2 = st.columns(2)
fq, fc = go.Figure(), go.Figure()
cum = np.cumsum(np.where(t[:, None] >= burn, out["profit"], 0.0), axis=0)
for i, k in enumerate(keys_):
    col = theory_color(k)
    dash = None if keys_.index(k) == i else "dash"
    fq.add_trace(go.Scatter(x=t, y=out["q"][:, i], name=names[i], line=dict(color=col, width=1.6, dash=dash)))
    fc.add_trace(go.Scatter(x=t, y=cum[:, i], name=names[i], line=dict(color=col, width=2, dash=dash)))
for f, yt in ((fq, "Output q"), (fc, "Cumulative profit after burn-in")):
    f.update_xaxes(title="Period"); f.update_yaxes(title=yt); f.update_layout(hovermode="x unified")
c1.plotly_chart(style(fq, 380, "Output by agent"))
c2.plotly_chart(style(fc, 380, "Cumulative profit by agent"))
summary = pd.DataFrame(dict(firm=names, avg_profit=out["profit"][burn:].mean(0),
                            avg_abs_dq=np.abs(np.diff(out["q"][burn - 1:], axis=0)).mean(0),
                            avg_q=out["q"][burn:].mean(0), final_capital=out["capital"][-1]))
st.dataframe(summary.sort_values("avg_profit", ascending=False), hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.1f") for c in summary.columns if c != "firm"})
st.caption(f"Industry: mean |P − P_Nash| = {out['industry']['nash_gap'].iloc[0]:.2f}, price volatility "
           f"{out['industry']['price_change_rms'].iloc[0]:.2f}, average margin {out['industry']['avg_margin'].iloc[0]:.2f}.")
download(pd.DataFrame({"period": t, "price": out["price"], "cost": out["cost"], "nash_price": out["nash"],
                       **{f"q_{n}": out["q"][:, i] for i, n in enumerate(names)}}), "multi_agent_market.csv")
