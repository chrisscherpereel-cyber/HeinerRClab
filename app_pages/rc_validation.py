import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.analysis import confusion, ols
from heiner_abm.experiments import EnvRanges
from heiner_abm.params import SELECTION_RULES
from ui.common import (CAT, base_scenario, cached_rc_validation, download, fmt_p, hypothesis_card, measurement,
                       show_errors, style, to_json, verdict)

st.title("Does the reliability condition predict who benefits from flexibility?")
hypothesis_card(
    "The paper's proposed test: firms that satisfy the RC should do better than firms that violate it",
    "We generate many random market environments with heterogeneous firms and selection rules. For **every firm** "
    "we run its market twice with identical shocks: once as specified, and once with that firm locked to rule B "
    "(*Never*, a rigid twin). The **dynamic advantage of flexibility** is the firm's profit minus its rigid twin's. "
    "Separately, the firm's reliability condition is computed from its own decisions in the first run.",
    "RC satisfied → flexibility pays (advantage > 0). RC violated → the rigid twin does better.",
    "Flexibility always pays, so predict 'advantage > 0' for everyone.")

base = base_scenario()
if not show_errors(base):
    st.stop()
H, cont = measurement()

with st.form("rcv"):
    st.markdown("**Environment sampling** (each environment draws its own parameters uniformly from these ranges)")
    c = st.columns(4)
    n_env = c[0].number_input("Environments", 5, 400, 40, step=5)
    n_reps = c[1].number_input("Replications per environment", 1, 20, 2)
    env_seed = c[2].number_input("Sampling seed", 0, 10**6, 12345)
    rules = c[3].multiselect("Production rules", ["Bertrand", "Cournot"], default=[base.firms[0].rule])
    c = st.columns(4)
    delta = c[0].slider("Volatility Δ", 0.5, 40.0, (2.0, 30.0))
    margin = c[1].slider("Desired margin m*", -2.0, 25.0, (0.0, 16.0))
    cmax = c[2].slider("Max raw-material cost", 50.0, 99.0, (60.0, 95.0))
    flex = c[3].slider("Flexibility φ", 0.0, 3.0, (0.1, 1.0))
    c = st.columns(3)
    thr = c[0].slider("Threshold θ", 0.0, 100.0, (5.0, 40.0))
    fore = c[1].slider("Competence κ", 0.0, 1.0, (0.0, 0.0))
    sels = c[2].multiselect("Selection rules", [s for s in SELECTION_RULES if s != "Never"],
                            default=["Always", "Small", "Large"])
    st.caption(f"Measurement: horizon **H = {H}**, evaluated firm "
               f"**{'returns to rule B' if cont == 'default' else 'keeps its own rules'}** (sidebar). "
               f"Cost grows with H × number of firms. Runs = environments × replications × (firms + 1) = "
               f"{int(n_env) * int(n_reps) * (base.n_firms + 1):,} markets.")
    go_btn = st.form_submit_button("Run validation", type="primary")

ranges = EnvRanges(delta=delta, desired_margin=margin, c_max=cmax, flex=flex, threshold=thr, foresight=fore,
                   selections=tuple(sels) or ("Always",), rules=tuple(rules) or ("Bertrand",))
key = (to_json(base), int(n_env), int(n_reps), json.dumps(asdict(ranges)), H, cont, int(env_seed))
if go_btn:
    st.session_state["rcv_key"] = key
if st.session_state.get("rcv_key") != key:
    st.stop()

with st.spinner("Running flexible markets and their rigid twins…"):
    df = cached_rc_validation(*key)
ok = df.dropna(subset=["rc_holds"]).copy()
if ok.empty:
    st.warning("No firm had enough decisions to compute its RC.")
    st.stop()
pred = ok["rc_holds"] > 0.5
act = ok["dynamic_adv"] > 0
acc = (pred == act).mean()
base_acc = act.mean()
bal = 0.5 * ((pred & act).sum() / max(act.sum(), 1) + (~pred & ~act).sum() / max((~act).sum(), 1))

m = st.columns(5)
m[0].metric("Firms evaluated", f"{len(ok):,}")
m[1].metric("Flexibility paid off", f"{act.mean():.0%}")
m[2].metric("RC accuracy", f"{acc:.1%}")
m[3].metric("'Always flexible' accuracy", f"{base_acc:.1%}", help="Traditional prediction: flexibility always pays.")
m[4].metric("RC balanced accuracy", f"{bal:.1%}", help="Mean of hit rates on 'paid off' and 'hurt'; 50% = chance.")

if base_acc < 0.5:
    verdict("support", f"Flexibility **hurt** {1 - base_acc:.0%} of firms relative to their rigid twin. The traditional "
            "claim that flexibility always pays is rejected in these environments.")
if acc > max(base_acc, 1 - base_acc) + 0.02:
    verdict("support", f"The RC classifies firms better ({acc:.0%}) than either naive rule ({max(base_acc, 1 - base_acc):.0%}).")
else:
    verdict("warn", f"At H = {H} the RC does **not** beat the naive rule ({acc:.0%} vs {max(base_acc, 1 - base_acc):.0%}). "
            "Production changes persist, so one-period gains and losses understate their consequences. "
            "Increase the horizon H in the sidebar (and keep 'returns to rule B') and compare.")

c1, c2 = st.columns([1, 1.4])
with c1:
    st.markdown("**Confusion matrix**")
    st.dataframe(confusion(pred, act), width="stretch")
    sub = ok.assign(correct=(pred == act)).groupby("selection").agg(
        firms=("correct", "size"), rc_accuracy=("correct", "mean"), share_paid=("dynamic_adv", lambda s: (s > 0).mean()),
        median_pi=("pi", "median"), median_r=("r", "median"), median_w=("w", "median")).reset_index()
    st.markdown("**By selection rule**")
    st.dataframe(sub, hide_index=True, width="stretch", column_config={
        "rc_accuracy": st.column_config.NumberColumn("RC accuracy", format="%.2f"),
        "share_paid": st.column_config.NumberColumn("Share where flex paid", format="%.2f"),
        "median_pi": st.column_config.NumberColumn("π", format="%.2f"),
        "median_r": st.column_config.NumberColumn("r", format="%.2f"),
        "median_w": st.column_config.NumberColumn("w", format="%.2f")})
with c2:
    x = ok["log_rc_margin"].replace([np.inf, -np.inf], np.nan)
    capped = x.clip(-4, 4)
    fig = go.Figure()
    for k, (sel, g) in enumerate(ok.assign(xm=capped).groupby("selection")):
        fig.add_trace(go.Scatter(x=g["xm"], y=g["dynamic_adv"], mode="markers", name=sel,
                                 marker=dict(size=8, color=CAT[k % len(CAT)], opacity=0.75, line=dict(color="white", width=1)),
                                 customdata=np.stack([g["pi"], g["r"], g["w"], g["delta"]], axis=1),
                                 hovertemplate="ln(ratio/tolerance) %{x:.2f}<br>dynamic advantage %{y:.1f}<br>π %{customdata[0]:.2f}"
                                               " r %{customdata[1]:.2f} w %{customdata[2]:.2f}<br>Δ %{customdata[3]:.1f}<extra>%{fullData.name}</extra>"))
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.8)", width=1))
    fig.add_vline(x=0, line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"))
    fig.update_xaxes(title="RC margin: ln(reliability ratio ÷ tolerance) (capped at ±4)")
    fig.update_yaxes(title="Profit vs rigid twin (per period)")
    st.plotly_chart(style(fig, 420, "Top-right & bottom-left quadrants = RC predicted correctly"))

st.subheader("What drives the advantage of flexibility?")
reg = ok.dropna(subset=["pi"]).copy()
reg["inst_adv"] = reg["realized_adv_per_period"]
X = [reg["inst_adv"].to_numpy(float), reg["delta"].to_numpy(float), reg["avg_margin"].to_numpy(float),
     reg["flex"].to_numpy(float)]
tab, r2 = ols(reg["dynamic_adv"].to_numpy(float), X,
              ["Σ instance gains (RC numerator − denominator) / period", "Volatility Δ", "Avg margin P − c", "φ"])
st.dataframe(tab, hide_index=True, width="stretch",
             column_config={"coef": st.column_config.NumberColumn(format="%.3f"), "se": st.column_config.NumberColumn(format="%.3f"),
                            "t": st.column_config.NumberColumn(format="%.2f"), "p": st.column_config.NumberColumn(format="%.4f")})
st.caption(f"OLS of the dynamic advantage on firm and environment characteristics, R² = {r2:.3f}. "
           "Σ instance gains = πrG − (1−π)wD per opportunity, times the opportunity rate. The RC is exactly the "
           "statement that this quantity is positive.")
download(ok, "rc_validation.csv", "Download firm-level results")

with st.expander("Why the counterfactual horizon matters"):
    st.markdown(
        "The paper proposed judging each deviation by one period of profit. In this market a production change "
        "**persists** (rule B keeps the new level) and rivals **react** to it, so one period of profit misses most "
        "of the deviation's consequences. With H = 1, nearly every firm appears to satisfy the RC, yet many lose to "
        "their rigid twins. Evaluating each deviation over H periods, with the firm returning to rule B afterwards "
        "(Heiner's 'deviate at this instance, otherwise follow B'), makes the per-instance gains add up to "
        "approximately the flexible-vs-rigid profit difference, and the RC's predictive power rises with H. "
        "Run this page at H = 1, 10, 25 and 50 to see the effect.")
