import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from heiner_abm.params import FirmSpec, RULE_TYPES, SELECTION_RULES
from ui.common import (CAT, COST_C, PRICE_C, base_scenario, cached_single, download, firm_colors, measure_opts, measurement,
                       show_errors, style, to_json)

st.title("Market lab: one market, every decision")
st.caption("A single cobweb market populated by firm agents. Edit the firms below (they start from the sidebar "
           "setup), run, and inspect each firm's reliability: when it deviated from rule B, whether that was a "
           "preferred exception, and what its type I and type II errors cost.")

base = base_scenario()
sig = json.dumps([asdict(f) for f in base.firms])
if st.session_state.get("lab_sig") != sig or "lab_firms" not in st.session_state:
    st.session_state["lab_sig"] = sig
    st.session_state.pop("lab_editor", None)
    st.session_state["lab_firms"] = pd.DataFrame([dict(
        label=f"Firm {i + 1}", rule=f.rule, flex=f.flex, selection=f.selection, threshold=f.threshold,
        desired_margin=f.desired_margin, foresight=f.foresight, noise=f.noise) for i, f in enumerate(base.firms)])

with st.expander("Firms in this market (editable)", expanded=True):
    st.caption("Mix rules freely: e.g. a rigid 'Never' firm against flexible rivals, or SR1 vs SR2 selection rules. "
               "Changing the sidebar firm setup resets this table.")
    edited = st.data_editor(
        st.session_state["lab_firms"], num_rows="dynamic", width="stretch", key="lab_editor",
        column_config={
            "label": st.column_config.TextColumn("Label"),
            "rule": st.column_config.SelectboxColumn("Rule", options=list(RULE_TYPES), required=True),
            "flex": st.column_config.NumberColumn("Flexibility φ", min_value=0.0, max_value=10.0, step=0.05,
                                                  format="%.2f"),
            "selection": st.column_config.SelectboxColumn("Selection rule", options=list(SELECTION_RULES),
                                                          required=True),
            "threshold": st.column_config.NumberColumn("θ", min_value=0.0, step=1.0),
            "desired_margin": st.column_config.NumberColumn("m*", step=0.5),
            "foresight": st.column_config.NumberColumn("κ", min_value=0.0, max_value=1.0, step=0.05),
            "noise": st.column_config.NumberColumn("σ", min_value=0.0, step=0.5),
        })
    c1, c2, c3 = st.columns([1, 1, 2])
    seed = c1.number_input("Seed for this run", 0, 10_000_000, int(base.seed), key="lab_seed")
    evolve = c2.toggle("Let firms evolve φ", value=False,
                       help="Imitate the most profitable rival's flexibility every 50 periods (see Endogenous flexibility).")

edited = edited.dropna(subset=["rule", "selection"]).reset_index(drop=True)
if edited.empty:
    st.warning("Add at least one firm.")
    st.stop()
firms = [FirmSpec(rule=r.rule, flex=float(r.flex), selection=r.selection, threshold=float(r.threshold),
                  desired_margin=float(r.desired_margin), foresight=float(r.foresight), noise=float(r.noise),
                  label=str(r.label) if isinstance(r.label, str) and r.label else f"Firm {i + 1}")
         for i, r in edited.fillna({"threshold": 25, "desired_margin": 5, "foresight": 0, "noise": 0,
                                    "flex": 0.5}).iterrows()]
scn = base.copy(firms=firms, seed=int(seed))
scn.evolution.enabled = bool(evolve)
if not show_errors(scn):
    st.stop()
H, cont = measurement()
with st.spinner("Running the market…"):
    out = cached_single(to_json(scn), H, cont, *measure_opts())
F, M = out["firms"], out["market"].iloc[0]
hist, P, C, Q = out["hist"], out["price"], out["cost"], out["quantity"]
burn, T, N = scn.burn_in, scn.periods, len(firms)
labels = [f.label for f in firms]
order = np.argsort([f.flex for f in firms], kind="stable")
colors = [None] * N
for rank, i in enumerate(order):
    colors[i] = firm_colors(N)[rank]

# ---------------- headline metrics -----------------
m = st.columns(6)
m[0].metric("Avg price", f"{M.avg_price:.2f}")
m[1].metric("Avg raw-material cost", f"{M.avg_cost:.2f}")
m[2].metric("Avg margin P − c", f"{M.avg_margin:.2f}")
m[3].metric("Serial corr. of cost", f"{M.sc_cost:.3f}", help="The paper's uncertainty proxy: lower = more uncertain.")
m[4].metric("Serial corr. of price", f"{M.sc_price:.3f}")
m[5].metric("Regime", M.regime.split(" (")[0], help=M.regime)

tab_ts, tab_rc, tab_dec, tab_dist = st.tabs(["Market dynamics", "Reliability scoreboard", "Decision inspector",
                                            "Distributions"])

with tab_ts:
    t = np.arange(T)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=C, name="Raw-material cost c", line=dict(color=COST_C, width=1.5)))
    fig.add_trace(go.Scatter(x=t, y=P, name="Market price P", line=dict(color=PRICE_C, width=2)))
    fig.add_vrect(x0=0, x1=burn, fillcolor="rgba(128,128,128,0.12)", line_width=0)
    for ts in np.flatnonzero(out["shifts"]):
        fig.add_vline(x=int(ts), line=dict(color="rgba(74,58,167,0.45)", width=1, dash="dot"))
    if out["shifts"].any():
        st.caption(f"{int(out['shifts'].sum())} unannounced demand-regime shifts (dotted lines). Cournot firms use an "
                   f"outdated demand model for L = {scn.structural.belief_lag} periods after each shift.")
    fig.update_xaxes(title="Period", rangeslider=dict(visible=True, thickness=0.06))
    fig.update_yaxes(title="Price / cost")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 380, "Price follows raw-material cost (shaded = burn-in)"))
    c1, c2 = st.columns(2)
    fq = go.Figure()
    fc = go.Figure()
    cum = np.cumsum(np.where(t[:, None] > burn, hist["profit"], 0.0), axis=0)
    for i in order[::-1]:
        fq.add_trace(go.Scatter(x=t, y=hist["q"][:, i], name=f"{labels[i]} (φ={firms[i].flex:.2f})",
                                line=dict(color=colors[i], width=1.5)))
        fc.add_trace(go.Scatter(x=t, y=cum[:, i], name=f"{labels[i]} (φ={firms[i].flex:.2f})",
                                line=dict(color=colors[i], width=2)))
    fq.update_layout(hovermode="x unified"); fc.update_layout(hovermode="x unified")
    fq.update_xaxes(title="Period"); fq.update_yaxes(title="Production q")
    fc.update_xaxes(title="Period"); fc.update_yaxes(title="Cumulative profit (after burn-in)")
    c1.plotly_chart(style(fq, 360, "Production by firm"))
    c2.plotly_chart(style(fc, 360, "Cumulative profit by firm"))
    if evolve:
        fe = go.Figure()
        for i in order[::-1]:
            fe.add_trace(go.Scatter(x=t, y=out["flex_path"][:, i], name=labels[i], line=dict(color=colors[i], width=2)))
        fe.update_xaxes(title="Period"); fe.update_yaxes(title="Flexibility φ")
        st.plotly_chart(style(fe, 300, "Evolving flexibility"))

with tab_rc:
    st.markdown(f"Counterfactual horizon **H = {H}**, evaluated firm "
                f"**{'returns to rule B' if cont == 'default' else 'keeps its own rules'}** after each decision. "
                "Change both in the sidebar under *Simulation & measurement*.")
    tbl = F.copy()
    tbl["label"] = labels
    tbl["rank"] = tbl["avg_profit"].rank(ascending=False).astype(int)
    tbl["RC"] = np.where(tbl["rc_holds"].isna(), "n/a", np.where(tbl["rc_holds"] > 0.5, "✅ holds", "❌ fails"))
    show = tbl[["label", "rule", "selection", "flex", "avg_profit", "rank", "avg_abs_dq", "cd_gap", "pi", "r", "w",
                "G", "D", "reliability_ratio", "tolerance_limit", "RC", "realized_adv_per_period"]]
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "label": "Firm", "flex": st.column_config.NumberColumn("φ", format="%.2f"),
        "avg_profit": st.column_config.NumberColumn("Avg profit", format="%.1f"),
        "rank": "Rank", "avg_abs_dq": st.column_config.NumberColumn("Avg |Δq|", format="%.1f"),
        "cd_gap": st.column_config.NumberColumn("CD-gap (cost RMSE)", format="%.2f"),
        "pi": st.column_config.NumberColumn("π", format="%.3f"),
        "r": st.column_config.NumberColumn("r", format="%.3f"), "w": st.column_config.NumberColumn("w", format="%.3f"),
        "G": st.column_config.NumberColumn("G", format="%.1f"), "D": st.column_config.NumberColumn("D", format="%.1f"),
        "reliability_ratio": st.column_config.NumberColumn("r / w", format="%.3f"),
        "tolerance_limit": st.column_config.NumberColumn("Tolerance", format="%.3f"),
        "realized_adv_per_period": st.column_config.NumberColumn("Gain from deviating / period", format="%.1f",
            help="Σ over deviations of (payoff with q*) − (payoff with rule B), over horizon H, per period."),
    })
    st.caption("r/w = ∞ when the firm never deviated wrongly. 'Never' firms have no deviations, so their RC is n/a. "
               "An 'Always' firm has r = w = 1, so its RC reduces to πG > (1−π)D.")
    fin = tbl[np.isfinite(tbl["log_rc_margin"])]
    if len(fin):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fin["label"], y=fin["log_rc_margin"], marker_color=[CAT[0] if v > 0 else CAT[7] for v in fin["log_rc_margin"]],
            hovertemplate="%{x}<br>ln(ratio / tolerance) = %{y:.2f}<extra></extra>", name="RC margin"))
        fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.8)", width=1))
        fig.update_yaxes(title="ln(reliability ratio ÷ tolerance limit)")
        st.plotly_chart(style(fig, 300, "Reliability-condition margin (> 0 = condition holds)"))

with tab_dec:
    c1, c2 = st.columns([1, 3])
    fi = c1.selectbox("Firm", range(N), format_func=lambda i: f"{labels[i]} · {firms[i].selection} · φ={firms[i].flex:.2f}")
    lo, hi = c2.slider("Periods", 1, T - 1, (max(burn, 1), min(T - 1, burn + 150)))
    idx = np.arange(lo, hi + 1)
    dev, opp = hist["deviate"][idx, fi], hist["opportunity"][idx, fi]
    gain = hist["profit_rule"][idx, fi] - hist["profit_default"][idx, fi]
    pe = opp & (gain > 0)
    outcome = np.select([~opp, pe & dev, pe & ~dev, ~pe & dev, ~pe & ~dev],
                        ["no change recommended", "correct deviation", "type I error (missed exception)",
                         "type II error (wrong deviation)", "correctly kept B"], "")
    log = pd.DataFrame(dict(period=idx, cost_prev=C[idx - 1], cost_perceived=hist["c_hat"][idx, fi], cost_actual=C[idx],
                            price=P[idx], q_before=hist["q"][idx - 1, fi], q_recommended=hist["rec"][idx, fi],
                            deviated=dev, q_chosen=hist["q"][idx, fi], gain_rule_vs_B=gain, outcome=outcome,
                            gain_one_period=hist["gain_static"][idx, fi], gain_persistence=hist["gain_persist"][idx, fi],
                            profit=hist["profit"][idx, fi]))
    counts = pd.Series(outcome).value_counts()
    k = st.columns(4)
    k[0].metric("Correct deviations", int(counts.get("correct deviation", 0)))
    k[1].metric("Type I errors", int(counts.get("type I error (missed exception)", 0)))
    k[2].metric("Type II errors", int(counts.get("type II error (wrong deviation)", 0)))
    k[3].metric("Correctly kept B", int(counts.get("correctly kept B", 0)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=hist["q"][idx - 1, fi], name="q before (rule B)", line=dict(color="#9a9893", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=idx, y=hist["q"][idx, fi], name="q chosen", line=dict(color=CAT[0], width=2)))
    styles = {"correct deviation": (CAT[2], "circle"), "type I error (missed exception)": (CAT[3], "triangle-down"),
              "type II error (wrong deviation)": (CAT[7], "x"), "correctly kept B": (CAT[6], "square")}
    for name, (col, sym) in styles.items():
        sel = outcome == name
        if sel.any():
            fig.add_trace(go.Scatter(x=idx[sel], y=hist["rec"][idx[sel], fi], mode="markers", name=f"q* · {name}",
                                     marker=dict(color=col, symbol=sym, size=9, line=dict(color="white", width=1)),
                                     customdata=gain[sel], hovertemplate="period %{x}<br>q* %{y}<br>gain of q* vs B: "
                                     "%{customdata:.1f}<extra>" + name + "</extra>"))
    fig.update_xaxes(title="Period"); fig.update_yaxes(title="Production")
    st.plotly_chart(style(fig, 380, "Recommendations q* and what the firm did"))
    st.dataframe(log, hide_index=True, width="stretch", height=300,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                ("cost_prev", "cost_perceived", "cost_actual", "price", "gain_rule_vs_B", "profit")})
    download(log, f"decisions_firm{fi + 1}.csv", "Download decision log")

with tab_dist:
    c1, c2 = st.columns(2)
    fh = go.Figure()
    fh.add_trace(go.Histogram(x=P[burn:], name="Price", marker_color=PRICE_C, opacity=0.75, nbinsx=40))
    fh.add_trace(go.Histogram(x=C[burn:], name="Raw-material cost", marker_color=COST_C, opacity=0.75, nbinsx=40))
    fh.update_layout(barmode="overlay"); fh.update_xaxes(title="Value"); fh.update_yaxes(title="Periods")
    c1.plotly_chart(style(fh, 340, "Histogram of price & cost (VBA chart)"))
    fs = go.Figure()
    fs.add_trace(go.Scatter(x=F["avg_abs_dq"], y=F["avg_profit"], mode="markers+text", text=labels,
                            textposition="top center", marker=dict(size=12, color=colors, line=dict(color="white", width=2)),
                            hovertemplate="%{text}<br>avg |Δq| %{x:.1f}<br>avg profit %{y:.1f}<extra></extra>",
                            showlegend=False))
    fs.update_xaxes(title="Realised flexibility: avg |Δq| per period"); fs.update_yaxes(title="Average profit")
    c2.plotly_chart(style(fs, 340, "Profit vs realised flexibility (VBA scattergram)"))
    ts = pd.DataFrame(dict(period=np.arange(T), price=P, cost=C, quantity=Q))
    for i in range(N):
        ts[f"q_{labels[i]}"] = hist["q"][:, i]
        ts[f"profit_{labels[i]}"] = hist["profit"][:, i]
    c1, c2 = st.columns(2)
    with c1:
        download(ts, "market_timeseries.csv", "Download time series")
    with c2:
        download(F.assign(label=labels), "firm_reliability.csv", "Download firm reliability table")
