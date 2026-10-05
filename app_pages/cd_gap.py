import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.analysis import ols
from heiner_abm.experiments import slope_summary
from heiner_abm.params import SELECTION_RULES
from ui.common import (CAT, DIVERGING, SEQUENTIAL, base_scenario, download, fmt_p, hypothesis_card, measurement, reps,
                       run_sweep_ui, show_errors, style, verdict)

st.title("CD-gap explorer: difficulty versus competence")
hypothesis_card(
    "Uncertainty is the gap between difficulty and competence",
    "Difficulty is raised by cost volatility Δ (or perception noise σ). Competence is raised by cost foresight κ, the "
    "share of the coming cost change a firm anticipates. Each firm's **measured CD-gap** is the RMSE of its cost "
    "perception. The grid below crosses the two and records Heiner's quantities for every firm.",
    "As the CD-gap widens, **r(U) falls and w(U) rises**, so the reliability ratio falls, the RC fails more often, "
    "and the payoff to flexibility shrinks. Raising competence restores it.",
    "Information and flexibility are both valuable; there is no systematic trade-off between them.")

base = base_scenario()
if not show_errors(base):
    st.stop()
H, cont = measurement()

c = st.columns(4)
difficulty = c[0].selectbox("Difficulty axis", ["delta", "noise"],
                            format_func=lambda k: {"delta": "Cost volatility Δ", "noise": "Perception noise σ"}[k])
d_rng = c[1].slider("Difficulty range", 0.0, 40.0, (2.0, 30.0) if difficulty == "delta" else (0.0, 12.0),
                    key=f"cd_rng_{difficulty}")
d_steps = c[2].slider("Difficulty steps", 2, 10, 5)
k_steps = c[3].slider("Competence steps (κ from 0 to 1)", 2, 6, 3)
c = st.columns(3)
sel_default = base.firms[0].selection if base.firms[0].selection not in ("Always", "Never") else "Adaptive"
sel = c[0].selectbox("Selection rule for all firms", [s for s in SELECTION_RULES if s != "Never"],
                     index=[s for s in SELECTION_RULES if s != "Never"].index(sel_default),
                     help="With 'Always', r = w = 1 by construction and only π, G and D can move. "
                          "Adaptive or Large rules let r and w respond to uncertainty.")
n_reps = c[1].number_input("Replications per cell", 1, 100, max(4, min(reps(), 10)))
periods = c[2].number_input("Periods per run", 200, 5000, min(base.periods, 800), step=100)

scn = base.copy(periods=int(periods))
for f in scn.firms:
    f.selection = sel
dvals = list(np.round(np.linspace(max(d_rng[0], 0.01 if difficulty == "delta" else 0.0), d_rng[1], d_steps), 3))
kvals = list(np.round(np.linspace(0, 1, k_steps), 3))
key = (difficulty, tuple(dvals), tuple(kvals), sel, int(n_reps), int(periods), H, cont, str(base))
if st.button("Run grid", type="primary"):
    st.session_state["cd_key"] = key
if st.session_state.get("cd_key") != key:
    st.caption(f"{len(dvals) * len(kvals) * int(n_reps):,} markets at H = {H}.")
    st.stop()

firms, mk = run_sweep_ui(scn, difficulty, dvals, int(n_reps), "foresight", kvals, horizon=H, continuation=cont)
summ = slope_summary(firms, mk, [difficulty, "foresight"])
dlab = {"delta": "Cost volatility Δ", "noise": "Perception noise σ"}[difficulty]

metrics = [("r", "r = P(deviate | exception)", SEQUENTIAL, None),
           ("w", "w = P(deviate | no exception)", SEQUENTIAL, None),
           ("pi", "π = P(preferred exception)", SEQUENTIAL, None),
           ("log_rc_margin", "RC margin ln(ratio ÷ tolerance)", DIVERGING, 0.0),
           ("slope", "Payoff to flexibility (profit slope on φ)", DIVERGING, 0.0),
           ("cd_gap", "Measured CD-gap (cost RMSE)", SEQUENTIAL, None)]
if H < 10:
    st.warning(f"The counterfactual horizon is H = {H}. Short horizons judge a deviation by its first few periods "
               "only, although production changes persist, so r, w and the RC margin can be misleading. H ≥ 20 "
               "is recommended (sidebar → Simulation & measurement).", icon="⚠️")
cols = st.columns(2)
for i, (mcol, title, scale, mid) in enumerate(metrics):
    pv = summ.pivot(index="foresight", columns=difficulty, values=mcol)
    z = pv.to_numpy(float)
    kw = {}
    if mid is not None:
        lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1
        kw = dict(zmid=0, zmin=-lim, zmax=lim)
    fig = go.Figure(go.Heatmap(z=z, x=[f"{x:g}" for x in pv.columns], y=[f"{y:g}" for y in pv.index], colorscale=scale,
                               xgap=2, ygap=2, texttemplate="%{z:.3g}", textfont=dict(size=12),
                               hovertemplate=f"{dlab} %{{x}}<br>κ %{{y}}<br>{mcol} %{{z:.3f}}<extra></extra>", **kw))
    fig.update_xaxes(title=dlab, type="category"); fig.update_yaxes(title="Competence κ", type="category")
    cols[i % 2].plotly_chart(style(fig, 340, title))

st.subheader("Reliability against the measured CD-gap")
fd = firms.dropna(subset=["r", "w"]).copy()
fd = fd[fd["opportunities"] > 20]
bins = pd.qcut(fd["cd_gap"], q=min(10, max(2, fd["cd_gap"].nunique() // 3)), duplicates="drop")
b = fd.groupby(bins, observed=True).agg(cd=("cd_gap", "mean"), r=("r", "mean"), w=("w", "mean"), pi=("pi", "mean"),
                                         rc=("rc_holds", "mean")).reset_index(drop=True)
fig = go.Figure()
for col, name, color, dash in [("r", "r(U)", CAT[0], None), ("w", "w(U)", CAT[1], "dash"),
                               ("pi", "π", CAT[2], "dot"), ("rc", "share satisfying RC", CAT[6], "dashdot")]:
    fig.add_trace(go.Scatter(x=b["cd"], y=b[col], name=name, mode="lines+markers", line=dict(color=color, width=2, dash=dash)))
fig.update_xaxes(title="Measured CD-gap U (RMSE of cost perception, decile means)"); fig.update_yaxes(title="Probability", range=[0, 1.02])
st.plotly_chart(style(fig, 380, "r = deviates when it should · w = deviates when it should not · π = exception rate"))

reg = fd.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_rc_margin"])
if len(reg) > 5:
    tr, _ = ols(reg["r"].to_numpy(float), [reg["cd_gap"].to_numpy(float)], ["U"])
    tw, _ = ols(reg["w"].to_numpy(float), [reg["cd_gap"].to_numpy(float)], ["U"])
    tm, _ = ols(reg["log_rc_margin"].to_numpy(float), [reg["cd_gap"].to_numpy(float)], ["U"])
    st.markdown(f"Firm-level regressions on U: **r** slope {tr.loc[1, 'coef']:+.4f} (p {fmt_p(tr.loc[1, 'p'])}), "
                f"**w** slope {tw.loc[1, 'coef']:+.4f} (p {fmt_p(tw.loc[1, 'p'])}), "
                f"**RC margin** slope {tm.loc[1, 'coef']:+.4f} (p {fmt_p(tm.loc[1, 'p'])}).")
    if tm.loc[1, "coef"] < 0 and tm.loc[1, "p"] < 0.05:
        verdict("support", "Reliability relative to the tolerance limit **declines significantly** as the CD-gap widens, as Heiner predicts.")
    elif tm.loc[1, "coef"] > 0 and tm.loc[1, "p"] < 0.05:
        verdict("reject", "Reliability relative to the tolerance limit **rises** with the CD-gap here, contrary to Heiner.")
    else:
        verdict("neutral", "No significant relationship between the measured CD-gap and the RC margin in this grid.")
    if sel == "Always":
        st.caption("With 'Always', r = w = 1 by construction, so the RC margin moves only through π, G and D.")
download(summ, "cd_gap_grid.csv")
