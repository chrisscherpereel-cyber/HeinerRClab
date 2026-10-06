import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.analysis import confusion, ols
from heiner_abm.experiments import MEASURE_LABELS, EnvRanges, prediction_table
from heiner_abm.params import SELECTION_RULES
from heiner_abm.terminology import IDENTITY_NOTE, OOS_GAP_NOTE, rule_option
from ui.common import adaptive_supported
from ui.common import (CAT, base_scenario, cached_rc_validation, download, hypothesis_card, measure_opts,
                       measurement, show_errors, style, to_json, verdict)
from ui.common import evidence_note

st.title("Does the reliability condition predict who benefits from flexibility?")
evidence_note("simulation")
hypothesis_card(
    "RC",
    "We generate many random market environments with heterogeneous firms and selection rules. For **every firm** "
    "we run its market twice with identical shocks: once as specified, and once with that firm locked to rule B "
    "(*Never*, a rigid twin). The **advantage of flexibility** is the firm's profit minus its rigid twin's.\n\n"
    "To avoid circularity, each run is split in two. The RC is **estimated in the first window** from the firm's "
    "own decisions, and used to **predict the advantage in the second window**, which it has never seen. "
    "Predictions are scored with the **AUC** (area under the ROC curve): 0.5 = no better than a coin flip, "
    "1 = perfect. AUC is insensitive to how often flexibility pays in a given sample, and it is the "
    "signal-detection statistic Heiner (1986) used for r (hit rate) and w (false-alarm rate).")

st.caption(IDENTITY_NOTE + " " + OOS_GAP_NOTE)

base = base_scenario()
if not show_errors(base):
    st.stop()
H, cont = measurement()
disc, split = measure_opts()

with st.form("rcv"):
    st.markdown("**Environment sampling** (each environment draws its own parameters uniformly from these ranges)")
    c = st.columns(4)
    n_env = c[0].number_input("Environments", 5, 400, 40, step=5,
                               help="Number of random market environments to draw from the ranges below.")
    n_reps = c[1].number_input("Replications per environment", 1, 20, 2,
                                help="Independent runs (different cost shocks) of each environment.")
    env_seed = c[2].number_input("Sampling seed", 0, 10**6, 12345,
                                  help="Seed for drawing the environments' parameters; the same seed gives the same set.")
    rules = c[3].multiselect("Production rules", ["Bertrand", "Cournot"], default=[base.firms[0].rule],
                             format_func=rule_option,
                             help="Production rules environments may use; each environment picks one at random.")
    c = st.columns(4)
    delta = c[0].slider("Volatility Δ", 0.5, 40.0, (2.0, 30.0),
                        help="Range for cost volatility Δ, the max raw-material cost change per period.")
    margin = c[1].slider("Desired margin m*", -2.0, 25.0, (0.0, 16.0),
                         help="Range for the margin-feedback rule's desired margin (competition intensity).")
    cmax = c[2].slider("Max raw-material cost", 50.0, 99.0, (60.0, 95.0),
                       help="Range for the upper cost bound; lower values make the industry more profitable.")
    flex = c[3].slider("Flexibility φ", 0.0, 3.0, (0.1, 1.0),
                       help="Range from which each firm's flexibility is drawn (Cournot values are capped at 1).")
    c = st.columns(4)
    thr = c[0].slider("Threshold θ", 0.0, 100.0, (5.0, 40.0),
                      help="Range for the |q* − q| threshold used by Small and Large selection rules.")
    fore = c[1].slider("Competence κ", 0.0, 1.0, (0.0, 0.0),
                       help="Range for cost foresight κ (share of the coming cost change anticipated). "
                            "Baseline firms have κ = 0.")
    hazard = c[2].slider("Regime-shift hazard λ (structural uncertainty)", 0.0, 0.15, (0.0, 0.0), step=0.005,
                         help="> 0 adds unannounced demand-regime shifts, so the regime can change between the "
                              "estimation and evaluation windows (a structural-change stress test for the RC).")
    lag = c[3].slider("Model-updating lag L", 0, 100, (20, 20),
                      help="Range for how many periods Cournot firms keep the old demand curve after a regime shift. "
                           "Only matters when the hazard λ range is above 0.")
    _sel_opts = [s for s in SELECTION_RULES if s != "Never" and (s != "Adaptive" or adaptive_supported(base))]
    sels = st.multiselect("Selection rules", _sel_opts,
                          default=[s for s in ("Always", "Small", "Large", "Adaptive") if s in _sel_opts],
                          help="Selection rules firms may be given. Never is excluded because it is the rigid twin "
                               "every firm is compared with.")
    st.caption(f"Measurement: horizon **H = {H}**, discount γ = {disc:g}, estimation window = first "
               f"{split:.0%} of recorded periods (sidebar). Markets simulated = environments × replications × "
               f"(firms + 1) = {int(n_env) * int(n_reps) * (base.n_firms + 1):,}. Cost grows with H.")
    go_btn = st.form_submit_button("Run validation", type="primary")

if not adaptive_supported(base):
    st.caption("The information specification does not support Adaptive firms, so they are not offered here.")
ranges = EnvRanges(delta=delta, desired_margin=margin, c_max=cmax, flex=flex, threshold=thr, foresight=fore,
                   selections=tuple(sels) or ("Always",), rules=tuple(rules) or ("Bertrand",), hazard=hazard,
                   belief_lag=(float(lag[0]), float(lag[1])))
key = (to_json(base), int(n_env), int(n_reps), json.dumps(asdict(ranges)), H, cont, int(env_seed), disc, split)
if go_btn:
    st.session_state["rcv_key"] = key
if st.session_state.get("rcv_key") != key:
    st.stop()

with st.spinner("Running flexible markets and their rigid twins…"):
    df = cached_rc_validation(*key)
if df.empty:
    st.warning("No firms to evaluate.")
    st.stop()

# ------------------------------------------------------------------ headline: AUC table
with st.spinner("Bootstrapping AUC confidence intervals (resampling environments)…"):
    pt = prediction_table(df, n_boot=300)
paid = (df["dyn_adv_eval"] > 0).mean()
full_oos = pt[(pt["predictor"] == MEASURE_LABELS["full"]) & pt["test"].str.startswith("Out")].iloc[0]
static_oos = pt[(pt["predictor"] == MEASURE_LABELS["static"]) & pt["test"].str.startswith("Out")].iloc[0]
m = st.columns(4)
m[0].metric("Firms evaluated", f"{len(df):,}")
m[1].metric("Flexibility beat rigid twin (eval window)", f"{paid:.0%}")
m[2].metric("Out-of-sample AUC, full dynamic RC", f"{full_oos.auc:.3f}", help=f"95% CI {full_oos.lo:.3f}–{full_oos.hi:.3f}")
m[3].metric("Out-of-sample AUC, one-shot RC (H = 1)", f"{static_oos.auc:.3f}",
            help=f"95% CI {static_oos.lo:.3f}–{static_oos.hi:.3f}")

if paid < 0.5:
    verdict("support", f"Flexibility **lost** to the rigid twin for {1 - paid:.0%} of firms in the evaluation window. "
            "So the claim that flexibility always pays does not hold for these simulated firms in these environments.")
if full_oos.lo > 0.5:
    verdict("support", f"The dynamic RC, estimated in the first window, **predicts** the second window's outcome "
            f"out of sample (AUC {full_oos.auc:.3f}, 95% CI {full_oos.lo:.3f}–{full_oos.hi:.3f}, excludes 0.5).")
else:
    verdict("warn", f"The dynamic RC's out-of-sample AUC ({full_oos.auc:.3f}, CI {full_oos.lo:.3f}–{full_oos.hi:.3f}) "
            "is not distinguishable from 0.5 here. Try more environments, a longer horizon H, or longer runs.")
if static_oos.hi < full_oos.lo:
    verdict("support", "The one-shot, one-period RC does significantly worse than the dynamic RC: "
            "the value of a decision lies mostly in its later consequences.")

fig = go.Figure()
order = list(dict.fromkeys(pt["predictor"]))
for k, (test, g) in enumerate(pt.groupby("test", sort=False)):
    g = g.set_index("predictor").reindex(order).dropna(subset=["auc"])
    fig.add_trace(go.Scatter(
        x=g["auc"], y=g.index, mode="markers", name=test,
        marker=dict(size=11, color=CAT[k], symbol="circle" if "Out" in test else "diamond"),
        error_x=dict(type="data", symmetric=False, array=g["hi"] - g["auc"], arrayminus=g["auc"] - g["lo"],
                     thickness=1.5, width=5),
        hovertemplate="%{y}<br>AUC %{x:.3f}<extra>" + test + "</extra>"))
fig.add_vline(x=0.5, line=dict(color="rgba(128,128,128,0.9)", width=1, dash="dot"))
fig.add_annotation(x=0.5, y=1.0, yref="paper", text="coin flip", showarrow=False, yanchor="bottom")
fig.update_xaxes(title="AUC for 'flexibility beat the rigid twin' (95% CI, environments resampled)")
fig.update_yaxes(autorange="reversed", title=None)
st.plotly_chart(style(fig, 440, "How well does each predictor separate winners from losers?"))
st.dataframe(pt, hide_index=True, width="stretch", column_config={
    "auc": st.column_config.NumberColumn("AUC", format="%.3f"),
    "lo": st.column_config.NumberColumn("95% lo", format="%.3f"),
    "hi": st.column_config.NumberColumn("95% hi", format="%.3f")})

# ------------------------------------------------------------------ details for a chosen measure
st.subheader("Look inside one measure")
c1, c2 = st.columns(2)
meas = c1.radio("RC measure", ["full", "persist", "static"], format_func=MEASURE_LABELS.get, horizontal=False,
                help="Which version of the reliability condition to inspect: one-shot (one period), H-period with "
                     "rivals fixed, or H-period with rivals reacting.")
mode = c2.radio("Test", ["oos", "ins"], format_func=lambda x: {"oos": "Out-of-sample (estimate → evaluate)",
                                                                "ins": "In-sample (all periods)"}[x],
                help="Out-of-sample: RC estimated in the first window predicts the advantage in the second. "
                     "In-sample: estimated and evaluated on the same periods (circular, shown for comparison).")
rc_col, mg_col, y_col = ((f"rc_est_{meas}", f"margin_est_{meas}", "dyn_adv_eval") if mode == "oos"
                         else (f"rc_all_{meas}", f"margin_all_{meas}", "dyn_adv_all"))
ok = df.dropna(subset=[rc_col]).copy()
pred, act = ok[rc_col] > 0.5, ok[y_col] > 0
c1, c2 = st.columns([1, 1.4])
with c1:
    st.markdown("**Confusion matrix**")
    st.dataframe(confusion(pred, act), width="stretch")
    st.caption(f"Accuracy {(pred == act).mean():.1%} vs 'always flexible' {act.mean():.1%}. Accuracy depends on the "
               "base rate; the AUC above does not.")
    sub = ok.assign(correct=(pred == act)).groupby("selection").agg(
        firms=("correct", "size"), accuracy=("correct", "mean"), share_paid=(y_col, lambda s: (s > 0).mean()),
        median_pi=("pi", "median"), median_r=("r", "median"), median_w=("w", "median")).reset_index()
    st.markdown("**By selection rule**")
    st.dataframe(sub, hide_index=True, width="stretch", column_config={
        "accuracy": st.column_config.NumberColumn(format="%.2f"),
        "share_paid": st.column_config.NumberColumn("Share where flex paid", format="%.2f"),
        "median_pi": st.column_config.NumberColumn("π", format="%.2f"),
        "median_r": st.column_config.NumberColumn("r", format="%.2f"),
        "median_w": st.column_config.NumberColumn("w", format="%.2f")})
with c2:
    x = ok[mg_col].replace([np.inf, -np.inf], np.nan).clip(-4, 4)
    fig = go.Figure()
    for k, (sel, g) in enumerate(ok.assign(xm=x).groupby("selection")):
        fig.add_trace(go.Scatter(x=g["xm"], y=g[y_col], mode="markers", name=sel,
                                 marker=dict(size=8, color=CAT[k % len(CAT)], opacity=0.75, line=dict(color="white", width=1)),
                                 customdata=np.stack([g["pi"], g["r"], g["w"], g["delta"]], axis=1),
                                 hovertemplate="RC margin %{x:.2f}<br>advantage %{y:.1f}<br>π %{customdata[0]:.2f}"
                                               " r %{customdata[1]:.2f} w %{customdata[2]:.2f}<br>Δ %{customdata[3]:.1f}"
                                               "<extra>%{fullData.name}</extra>"))
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.8)", width=1))
    fig.add_vline(x=0, line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"))
    fig.update_xaxes(title="RC margin ln(ratio ÷ tolerance), capped at ±4")
    fig.update_yaxes(title="Profit vs rigid twin (per period)")
    st.plotly_chart(style(fig, 420, "Top-right & bottom-left = RC predicted correctly"))

if df["hazard"].max() > 0:
    st.subheader("Does structural uncertainty break the RC's predictions?")
    rows = []
    for lab, g in df.assign(regime=np.where(df["n_shifts"] > 0, "Regime shifts occurred", "No regime shifts")).groupby("regime"):
        t = prediction_table(g, n_boot=150)
        t = t[t["predictor"].str.startswith("RC")]
        rows.append(t.assign(environments=lab))
    st.dataframe(pd.concat(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("auc", "lo", "hi")})
    st.caption("Compare the out-of-sample AUC between environments with and without regime shifts. A drop means the "
               "reliability learned in one regime does not carry over to the next: a structural-change limit on the RC.")

st.subheader("What drives the advantage of flexibility?")
reg = df.dropna(subset=[f"inst_adv_est_{meas}"]).copy()
X = [reg[f"inst_adv_est_{meas}"].to_numpy(float), reg["delta"].to_numpy(float), reg["avg_margin"].to_numpy(float),
     reg["flex"].to_numpy(float)]
tab, r2 = ols(reg["dyn_adv_eval"].to_numpy(float), X,
              ["Instance gains in estimation window (RC numerator − denominator)", "Volatility Δ", "Avg margin P − c", "φ"])
st.dataframe(tab, hide_index=True, width="stretch",
             column_config={"coef": st.column_config.NumberColumn(format="%.3f"), "se": st.column_config.NumberColumn(format="%.3f"),
                            "t": st.column_config.NumberColumn(format="%.2f"), "p": st.column_config.NumberColumn(format="%.4f")})
st.caption(f"OLS of the evaluation-window advantage on estimation-window characteristics, R² = {r2:.3f}. The RC is "
           "exactly the statement that the instance gains πrG − (1−π)wD are positive.")
download(df, "rc_validation.csv", "Download firm-level results")
