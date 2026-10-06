from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.tracking import (FAMILY_LABELS, QUICK_TRACK, TrackPlan, gains, kalman_gain, loss_of_speed,
                                 optimal_offset, run_tracking, weights)
from ui.common import CAT, download, hypothesis_card, style, verdict, prereg_explainer
from ui.common import evidence_note

st.title("A benchmark solvable on paper")
evidence_note("analytical", "simulation")
st.caption("One firm tracks a moving target that it observes with noise. Here the best adjustment speed can be "
           "derived exactly (Muth 1960; Kalman 1960), so the simulation can be checked against theory. Then the stakes "
           "are made lopsided, the one case in which Heiner's reliability condition and optimal filtering make "
           "different predictions.")

st.header("1 · The model and the exact solution", divider="gray")
st.markdown(
    "The best action follows a random walk, **x**ₜ = xₜ₋₁ + wₜ with Var(w) = q, and the firm observes "
    "**y**ₜ = xₜ + vₜ with Var(v) = r. With partial adjustment aₜ = aₜ₋₁ + φ(yₜ − aₜ₋₁) and squared-error loss, "
    "the expected loss per period is")
st.latex(r"E(\varphi)=\frac{(1-\varphi)^2 q+\varphi^2 r}{\varphi(2-\varphi)},\qquad "
         r"\varphi^*=k=\frac{P}{P+r},\quad P=\frac{q+\sqrt{q^2+4qr}}{2}")
st.markdown("The best speed φ\\* is the steady-state Kalman gain. It depends **only on the signal-to-noise ratio** "
            "q / r: noisier observations call for slower adjustment, faster-moving targets for quicker adjustment.")
c1, c2 = st.columns([1, 2])
snr = c1.select_slider("Signal-to-noise ratio q / r", options=[0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                       value=0.1, key="trk_snr",
                       help="Variance of the target's movement relative to the variance of the observation error.")
k = kalman_gain(snr, 1.0)
c1.metric("Kalman gain (best speed)", f"{k:.3f}", help="The adjustment speed that minimizes expected squared error.")
c1.metric("Minimum loss", f"{loss_of_speed(k, snr, 1.0):.3f}", help="Expected squared error at the best speed (r = 1).")
ph = np.linspace(0.02, 1.0, 99)
fig = go.Figure(go.Scatter(x=ph, y=loss_of_speed(ph, snr, 1.0), mode="lines", line=dict(color=CAT[0]),
                           name="Analytic loss"))
fig.add_vline(x=k, line=dict(color=CAT[1], dash="dash"), annotation_text="Kalman gain")
fig.update_xaxes(title="Adjustment speed φ")
fig.update_yaxes(title="Expected loss per period", type="log")
c2.plotly_chart(style(fig, height=320), width="stretch")

st.header("2 · Where the theories part: lopsided stakes", divider="gray")
hypothesis_card(
    "TRACK",
    "Overshooting the target now costs ρ times as much as undershooting it (weights normalized so that the average "
    "weight is 1, which leaves the loss of any symmetric error unchanged). The information is the same as before. "
    "Optimal filtering says the response to news should not change; the stakes only shift the level of the action "
    "(certainty equivalence with an offset). The reliability condition says the tolerance limit for acting on a "
    "signal rises with the cost of a wrong move, so moves in the costly direction should be restricted.")
rho = st.select_slider("Stakes ratio ρ (cost of overshooting ÷ cost of undershooting)",
                       options=[1, 2, 4, 8, 16, 32], value=8, key="trk_rho",
                       help="How lopsided the stakes are. ρ = 1 is the symmetric case of section 1.")
off, lmin = optimal_offset(rho)
co, cu = weights(rho)
c = st.columns(3)
c[0].metric("Overshoot weight", f"{co:.2f}", help="Loss weight on squared errors above the target.")
c[1].metric("Undershoot weight", f"{cu:.2f}", help="Loss weight on squared errors below the target.")
c[2].metric("Optimal offset", f"{off:+.2f} s.d.",
            help="Under certainty equivalence the firm aims this many posterior standard deviations below its "
                 "estimate. Its loss falls to this share of the filter's loss: " + f"{lmin:.2f}.")
st.caption("Rules compared: " + "; ".join(f"**{v}**" for v in FAMILY_LABELS.values()) +
           ". Speeds and thresholds are tuned on training paths for each stakes ratio and evaluated on separate test "
           "paths with the same random numbers.")

st.header("3 · Plan", divider="gray")
prereg_explainer()
scale = st.radio("Protocol", ["Registered", "Quick check"], horizontal=True, key="trk_scale",
                 help="Registered runs the frozen plan (about 15 seconds). Quick check is a small exploratory run.")
registered = TrackPlan()
plan = registered if scale == "Registered" else replace(registered, **QUICK_TRACK)
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Plan hash:** `{plan.digest}` " + ("· registered plan" if plan.digest == registered.digest
                                                       else f"· **exploratory** (registered: `{registered.digest}`)"))
    st.caption(f"Verification: q / r = {', '.join(f'{s:g}' for s in plan.snrs)}, {plan.verify_reps} paths × "
               f"{plan.verify_periods:,} periods. Stakes: q / r = {plan.stakes_snr:g}, ρ = "
               f"{', '.join(f'{x:g}' for x in plan.rhos)}, {plan.train_reps} training and {plan.test_reps} test paths "
               f"× {plan.periods:,} periods.")
with c2:
    st.download_button("Download plan (.json)", plan.to_json().encode(), file_name=f"tracking_{plan.digest}.json",
                       mime="application/json", width="stretch", help="The frozen plan, including a code hash.")
st.dataframe(pd.DataFrame(plan.hypotheses, columns=["ID", "Hypothesis", "Decision rule"]), hide_index=True,
             width="stretch")
store = st.session_state.setdefault("trk_results", {})
if st.button("Run study", type="primary", key="trk_run"):
    bar = st.progress(0.0, "Starting")
    store[plan.digest] = run_tracking(plan, lambda f, m: bar.progress(min(f, 1.0), m))
    bar.empty()
res = store.get(plan.digest)
if res is None:
    st.info("Run the study to see results.", icon="ℹ️")
    st.stop()

st.header("4 · Results", divider="gray")
if plan.digest != registered.digest:
    st.warning("Exploratory run: the plan differs from the registered one, so these verdicts are not confirmatory.",
               icon="⚠️")
for _, r in res.verdicts.iterrows():
    verdict("support" if r["verdict"] == "supported" else "reject", f"**{r['id']}** · {r['hypothesis']}  \n"
            f"{r['result']}")

st.subheader("Does the simulation reproduce the exact solution?")
fig = go.Figure()
for i, (s, d) in enumerate(res.verify.groupby("snr")):
    fig.add_trace(go.Scatter(x=d["phi"], y=d["analytic"], mode="lines", line=dict(color=CAT[i]),
                             name=f"q/r = {s:g}, analytic"))
    fig.add_trace(go.Scatter(x=d["phi"], y=d["simulated"], mode="markers", marker=dict(color=CAT[i], size=5),
                             name=f"q/r = {s:g}, simulated"))
    fig.add_vline(x=kalman_gain(s, 1.0), line=dict(color=CAT[i], dash="dot"))
fig.update_xaxes(title="Adjustment speed φ")
fig.update_yaxes(title="Loss per period", type="log")
st.plotly_chart(style(fig, height=420), width="stretch")
st.dataframe(res.verify_summary.rename(columns={
    "snr": "q / r", "kalman_gain": "Kalman gain", "simulated_best": "Simulated best speed",
    "analytic_best_on_grid": "Analytic best on grid", "grid_step": "Grid step", "max_rel_gap": "Largest gap",
    "min_loss": "Minimum loss", "posterior_var": "Posterior variance"}), hide_index=True, width="stretch",
    column_config={"Largest gap": st.column_config.NumberColumn(format="percent")})

st.subheader("Lopsided stakes")
g = gains(res, plan)
fig = go.Figure()
for i, f in enumerate(["filter_offset", "speed_asym", "restrict", "speed_sym"]):
    d = g[g["family"] == f]
    rel = d["gain"] / d["base_loss"]
    fig.add_trace(go.Scatter(x=d["rho"], y=rel, mode="lines+markers", name=FAMILY_LABELS[f],
                             line=dict(color=CAT[i]),
                             error_y=dict(type="data", array=(d["hi"] - d["gain"]) / d["base_loss"],
                                          arrayminus=(d["gain"] - d["lo"]) / d["base_loss"], thickness=1, width=3)))
fig.add_hline(y=0, line=dict(color="gray", dash="dot"))
fig.update_xaxes(title="Stakes ratio ρ", type="log")
fig.update_yaxes(title="Loss reduction vs Kalman filter", tickformat=".0%")
st.plotly_chart(style(fig, height=380), width="stretch")
c1, c2 = st.columns(2)
mv = res.stakes.groupby(["rho", "family"])[["up_rate", "down_rate"]].mean().reset_index()
fig = go.Figure()
for col, lab, dash in (("up_rate", "upward moves", "solid"), ("down_rate", "downward moves", "dash")):
    for i, f in enumerate(["filter", "restrict"]):
        d = mv[mv["family"] == f]
        fig.add_trace(go.Scatter(x=d["rho"], y=d[col], mode="lines+markers", name=f"{FAMILY_LABELS[f]}: {lab}",
                                 line=dict(color=CAT[i], dash=dash)))
fig.update_xaxes(title="Stakes ratio ρ", type="log")
fig.update_yaxes(title="Share of periods with a move", tickformat=".0%")
c1.plotly_chart(style(fig, height=380), width="stretch")
ch = res.chosen[res.chosen["family"].isin(["speed_asym", "restrict"])]
fig = go.Figure()
d = ch[ch["family"] == "speed_asym"]
fig.add_trace(go.Scatter(x=d["rho"], y=d["phi_up"], name="Speed up", mode="lines+markers", line=dict(color=CAT[2])))
fig.add_trace(go.Scatter(x=d["rho"], y=d["phi_down"], name="Speed down", mode="lines+markers",
                         line=dict(color=CAT[2], dash="dash")))
fig.add_hline(y=kalman_gain(plan.stakes_snr, 1.0), line=dict(color="gray", dash="dot"),
              annotation_text="Kalman gain")
d = ch[ch["family"] == "restrict"]
fig.add_trace(go.Scatter(x=d["rho"], y=d["th_up"], name="Threshold up (s.d.)", mode="lines+markers", yaxis="y2",
                         line=dict(color=CAT[3])))
fig.add_trace(go.Scatter(x=d["rho"], y=d["th_down"], name="Threshold down (s.d.)", mode="lines+markers", yaxis="y2",
                         line=dict(color=CAT[3], dash="dash")))
fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Threshold (posterior s.d.)", showgrid=False))
fig.update_xaxes(title="Stakes ratio ρ", type="log")
fig.update_yaxes(title="Tuned adjustment speed")
c2.plotly_chart(style(fig, height=380), width="stretch")
st.markdown(
    "**Reading the results.** With symmetric stakes nothing beats the Kalman filter, as both theories predict. As the "
    "stakes become lopsided, the best rule restricts moves in the costly direction (higher thresholds and slower "
    "speeds upward), although the information has not changed. Optimal filtering's prescription for the speed is "
    "unchanged by the stakes, so this is the prediction that is uniquely Heiner's. Its limit is just as clear: an "
    "agent that builds the stakes into its estimate (the filter plus the optimal offset) does better still, so the "
    "restriction is the reliability condition's answer for agents whose flexible rule does not encode the stakes.")
download(res.stakes, f"tracking_{plan.digest}.csv", "Download test-path results (CSV)")
