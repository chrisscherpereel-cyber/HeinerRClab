import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.arena import ALL_DESIGNS
from heiner_abm.experiment import PLAN, by_condition, classify, evaluate_experiment, synthetic_participants
from ui.common import CAT, download, hypothesis_card, style, verdict, prereg_explainer

st.title("Experiment analysis")
st.caption("Pools the files downloaded from Play the market, classifies each participant by the agent design that "
           "best predicts their choices, and tests the pre-registered hypotheses.")

hypothesis_card(
    "EXPER",
    "**Classification:** every agent design ran in shadow mode on the participant's firm and recorded the output it "
    "would have chosen given exactly what the participant saw. The design with the lowest prediction error is the "
    "participant's best-fitting rule. **Predictability (X1):** the share of periods in which the participant changed "
    "output, high versus low uncertainty, paired within participants. **Does restraint pay (X2):** under high "
    "uncertainty, profit relative to rivals against that share, across participants.")

st.header("1 · Plan", divider="gray")
prereg_explainer()
st.markdown(f"**Experiment plan hash:** `{PLAN.digest}` · {PLAN.periods_per_block} periods per block · conditions "
            f"{', '.join(PLAN.conditions)} · rivals {', '.join(ALL_DESIGNS[k].name for k in PLAN.rivals)}")
st.dataframe(pd.DataFrame(PLAN.hypotheses, columns=["ID", "Hypothesis", "Decision rule"]), hide_index=True,
             width="stretch")
st.download_button("Download plan (.json)", PLAN.to_json().encode(), file_name=f"experiment_{PLAN.digest}.json",
                   mime="application/json", help="Register this plan before collecting data.")

st.header("2 · Data", divider="gray")
src = st.radio("Source", ["Upload participants' files", "Synthetic demonstration"], horizontal=True, key="exa_src",
               help="The synthetic demonstration uses simulated participants who follow known agent designs with "
                    "noise. It shows how the analysis works and is not evidence about people.")
truth = None
if src.startswith("Upload"):
    ups = st.file_uploader("CSV files from Play the market", type="csv", accept_multiple_files=True, key="exa_up",
                           help="One file per participant (or a combined file with a participant column).")
    if not ups:
        st.info("Upload one or more files to continue.", icon="ℹ️")
        st.stop()
    df = pd.concat([pd.read_csv(u) for u in ups], ignore_index=True)
    if "participant" not in df.columns or not any(c.startswith("shadow:") for c in df.columns):
        st.error("These files do not look like Play the market downloads (missing participant or shadow columns).")
        st.stop()
    if "plan" in df.columns and set(df["plan"].astype(str)) != {PLAN.digest}:
        st.warning("Some files were collected under a different experiment plan hash. Analyze them separately.",
                   icon="⚠️")
else:
    c1, c2 = st.columns(2)
    n = c1.slider("Synthetic participants", 6, 48, 12, 6, key="exa_n", help="Number of simulated participants.")
    noise = c2.slider("Decision noise", 0.0, 20.0, 4.0, 1.0, key="exa_noise",
                      help="Standard deviation of the noise added to each simulated participant's design.")
    df, truth = synthetic_participants(n, noise)
    st.warning("Synthetic demonstration: simulated participants with known designs. Not evidence about human "
               "behavior.", icon="🧪")
st.caption(f"{df['participant'].nunique()} participants · {len(df)} decisions")

st.header("3 · Pre-registered hypotheses", divider="gray")
for _, r in evaluate_experiment(df).iterrows():
    kind = {"supported": "support", "not supported": "reject"}.get(r["verdict"], "neutral")
    verdict(kind, f"**{r['id']}** · {r['hypothesis']}  \n{r['verdict'].capitalize()}: {r['result']}")

c = by_condition(df)
c1, c2 = st.columns(2)
fig = go.Figure()
for i, cond in enumerate(PLAN.conditions):
    d = c[c["block"] == cond]
    fig.add_trace(go.Box(y=d["deviation_rate"], name=cond, boxpoints="all", jitter=0.4, marker_color=CAT[i]))
fig.update_yaxes(title="Share of periods with a change in output", tickformat=".0%")
fig.update_xaxes(title="Uncertainty")
c1.plotly_chart(style(fig, height=360, showlegend=False), width="stretch")
h = c[c["block"] == "high"]
fig = go.Figure(go.Scatter(x=h["deviation_rate"], y=h["relative_profit"], mode="markers", text=h["participant"],
                           marker=dict(color=CAT[2], size=9),
                           hovertemplate="%{text}<br>changes %{x:.0%}<br>relative profit %{y:,.0f}<extra></extra>"))
if len(h) >= 3 and h["deviation_rate"].std() > 0:
    b, a = np.polyfit(h["deviation_rate"], h["relative_profit"], 1)
    xs = np.linspace(h["deviation_rate"].min(), h["deviation_rate"].max(), 20)
    fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", line=dict(color="gray", dash="dash"), name="OLS"))
fig.update_xaxes(title="Share of periods with a change (high uncertainty)", tickformat=".0%")
fig.update_yaxes(title="Profit − rivals' mean profit per period")
c2.plotly_chart(style(fig, height=360, showlegend=False), width="stretch")

st.header("4 · Which design predicts each participant?", divider="gray")
cl = classify(df)
cl["design"] = cl["best_design"].map(lambda k: ALL_DESIGNS[k].name)
cl["theory"] = cl["best_design"].map(lambda k: ALL_DESIGNS[k].theory)
counts = cl.groupby("theory").size().sort_values()
fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=CAT[0]))
fig.update_xaxes(title="Participants best predicted")
st.plotly_chart(style(fig, height=80 + 34 * len(counts)), width="stretch")
show = cl[["participant", "design", "theory", "best_rmse"]].rename(columns={"best_rmse": "Prediction error (RMSE)"})
if truth is not None:
    show = show.merge(truth, on="participant")
    show["true design"] = show.pop("true_design").map(lambda k: ALL_DESIGNS[k].name)
    st.metric("Recovery of the true design", f"{(cl.merge(truth, on='participant').eval('best_design == true_design')).mean():.0%}",
              help="Share of simulated participants classified as the design that generated them.")
st.dataframe(show, hide_index=True, width="stretch")
st.caption("Designs that make similar predictions are hard to tell apart; inspect the prediction errors of every "
           "design in the download before drawing conclusions about individuals.")
download(cl, "experiment_classification.csv", "Download classification (CSV)")
