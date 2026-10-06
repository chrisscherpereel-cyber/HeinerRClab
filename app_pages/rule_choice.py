from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.registered import TOURNAMENT_PLAN, registered_tuned
from heiner_abm.rulechoice import (QUICK_CHOICE, RESTRICTED, RULE_LABELS, RULES, default_choice_plan, level_env,
                                   run_choice_study)
from ui.common import CAT, download, hypothesis_card, style, verdict, prereg_explainer
from ui.common import evidence_note

st.title("Rule choice: do firms drift toward rule-governed behavior?")
evidence_note("simulation")
st.caption("Firms are no longer assigned a decision rule. Each firm switches between six rules according to how well "
           "each rule has recently performed, with an adjustable intensity of choice (Brock & Hommes 1997). The "
           "question is whether restricted, rule-governed behavior emerges as uncertainty rises, and what that does "
           "to the market.")

hypothesis_card(
    "EMERGE",
    "Every firm carries all six rules. Each rule observes the firm's actual output and proposes an output every "
    "period, so a firm can switch without losing state. Each period a firm revises its rule with a small "
    "probability, choosing rule *k* with logit probability ∝ exp(β·U_k / s), where U_k is the rule's fitness (an "
    "exponentially weighted average profit of the firms currently using it), s a running profit scale and β the "
    "**intensity of choice** (β = 0: random choice; large β: everyone moves to the best rule). Three rules are "
    "**restricted**: they keep the current output unless a condition is met (rule B, an inaction band and the "
    "reliability condition). Three are **flexible**: the filtered best reply, adaptive price expectations and the "
    "target-margin heuristic.")

with st.expander("The uncertainty scale and the rules", icon="ℹ️"):
    lv = pd.DataFrame([dict(level=u, **{k: getattr(level_env(u, 0), k) for k in ("delta", "noise", "hazard")})
                       for u in (0, 0.25, 0.5, 0.75, 1.0)])
    lv.columns = ["Uncertainty level u", "Cost volatility Δ", "Perception error σ", "Demand-shift hazard λ"]
    st.dataframe(lv, hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame([dict(Rule=RULE_LABELS[k], Type="restricted" if k in RESTRICTED else "flexible")
                               for k in RULES]), hide_index=True, width="stretch")
    st.caption("Each rule uses the parameters it was tuned to in the registered agent tournament "
               f"(plan `{TOURNAMENT_PLAN}`), so no rule is handicapped by its settings.")

st.header("1 · Plan", divider="gray")
prereg_explainer()
scale = st.radio("Protocol", ["Registered", "Quick check"], horizontal=True, key="choice_scale",
                 help="Registered runs the frozen plan (5 uncertainty levels × 4 intensities × 8 replications, about "
                      "20 seconds). Quick check is a small exploratory run.")
plan = default_choice_plan(TOURNAMENT_PLAN)
registered = plan
if scale == "Quick check":
    plan = replace(plan, **QUICK_CHOICE)
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Plan hash:** `{plan.digest}` " + ("· registered plan" if plan.digest == registered.digest
                                                       else f"· **exploratory** (registered: `{registered.digest}`)"))
    st.caption(f"Uncertainty levels {', '.join(f'{u:g}' for u in plan.levels)} · intensities of choice β = "
               f"{', '.join(f'{b:g}' for b in plan.betas)} · {plan.reps} replications · {plan.n_firms} firms · "
               f"{plan.periods} periods · revision probability {plan.revision:g} per period · fitness memory "
               f"{plan.memory:g}")
with c2:
    st.download_button("Download plan (.json)", plan.to_json().encode(), file_name=f"rulechoice_{plan.digest}.json",
                       mime="application/json", width="stretch",
                       help="The frozen plan, including a hash of the simulation code.")
st.dataframe(pd.DataFrame(plan.hypotheses, columns=["ID", "Hypothesis", "Decision rule"]), hide_index=True,
             width="stretch")

store = st.session_state.setdefault("choice_results", {})
if st.button("Run study", type="primary", key="choice_run"):
    bar = st.progress(0.0, "Starting")
    store[plan.digest] = run_choice_study(plan, registered_tuned(), lambda f, m: bar.progress(min(f, 1.0), m))
    bar.empty()
res = store.get(plan.digest)
if res is None:
    st.info("Run the study to see results.", icon="ℹ️")
    st.stop()

st.header("2 · Pre-registered hypotheses", divider="gray")
if plan.digest != registered.digest:
    st.warning("Exploratory run: the plan differs from the registered one, so these verdicts are not confirmatory.",
               icon="⚠️")
for _, r in res.verdicts.iterrows():
    verdict("support" if r["verdict"] == "supported" else "reject", f"**{r['id']}** · {r['hypothesis']}  \n"
            f"{r['result']}")

runs = res.runs
st.header("3 · Which rules survive?", divider="gray")
beta = st.select_slider("Intensity of choice β", options=list(plan.betas), value=max(plan.betas), key="choice_beta",
                        help="How strongly firms move toward the best-performing rule. β = 0 is random choice, so "
                             "any difference from equal shares at β = 0 is chance.")
d = runs[runs["beta"] == beta].groupby("level")[[f"share_{k}" for k in RULES]].mean()
fig = go.Figure()
for i, k in enumerate(RULES):
    fig.add_trace(go.Bar(x=[f"{u:g}" for u in d.index], y=d[f"share_{k}"], name=RULE_LABELS[k],
                         marker_color=CAT[i % len(CAT)], marker_pattern_shape="/" if k in RESTRICTED else "",
                         hovertemplate="u = %{x}<br>share %{y:.0%}<extra>" + RULE_LABELS[k] + "</extra>"))
fig.update_layout(barmode="stack")
fig.update_xaxes(title="Uncertainty level u", type="category")
fig.update_yaxes(title="Share of firms (final third of the run)", tickformat=".0%")
st.plotly_chart(style(fig, height=420), width="stretch")
st.caption("Hatched bars are restricted rules. Shares are averaged over the final third of each run and over "
           "replications.")

c1, c2 = st.columns(2)
for col, y, title in ((c1, "restricted_share", "Share of firms using restricted rules"),
                      (c2, "change_rate", "Share of periods in which firms change output")):
    g = runs.groupby(["beta", "level"])[y].agg(["mean", "std", "count"]).reset_index()
    fig = go.Figure()
    for i, (b, gg) in enumerate(g.groupby("beta")):
        ci = 1.96 * gg["std"] / np.sqrt(gg["count"])
        fig.add_trace(go.Scatter(x=gg["level"], y=gg["mean"], name=f"β = {b:g}", mode="lines+markers",
                                 line=dict(color=CAT[i % len(CAT)]),
                                 error_y=dict(type="data", array=ci, thickness=1, width=3)))
    fig.update_xaxes(title="Uncertainty level u")
    fig.update_yaxes(title=title, tickformat=".0%")
    col.plotly_chart(style(fig, height=360), width="stretch")

st.header("4 · Market outcomes", divider="gray")
st.markdown("Does the rule mix that emerges move the market away from equilibrium? Each point is one run.")
c1, c2 = st.columns(2)
for col, y, title in ((c1, "dist_nash", "Mean |P − P_Nash|"), (c2, "price_sd", "Price standard deviation")):
    fig = go.Figure()
    for i, (u, gg) in enumerate(runs.groupby("level")):
        fig.add_trace(go.Scatter(x=gg["restricted_share"], y=gg[y], mode="markers", name=f"u = {u:g}",
                                 marker=dict(color=CAT[i % len(CAT)], size=7, opacity=0.75),
                                 customdata=gg["beta"], hovertemplate="restricted %{x:.0%}<br>" + title +
                                 " %{y:.1f}<br>β = %{customdata:g}<extra></extra>"))
    fig.update_xaxes(title="Share of firms using restricted rules", tickformat=".0%")
    fig.update_yaxes(title=title)
    col.plotly_chart(style(fig, height=360), width="stretch")

if res.paths:
    st.subheader("How the restricted share evolves")
    fig = go.Figure()
    for i, u in enumerate(plan.levels):
        path = res.paths.get((beta, u))
        if path is not None:
            fig.add_trace(go.Scatter(y=path, name=f"u = {u:g}", mode="lines", line=dict(color=CAT[i % len(CAT)])))
    fig.update_xaxes(title="Period")
    fig.update_yaxes(title="Share of firms using restricted rules", tickformat=".0%", range=[0, 1])
    st.plotly_chart(style(fig, height=360), width="stretch")
    st.caption("Averaged over replications, at the intensity of choice chosen above.")

download(runs, f"rulechoice_{plan.digest}.csv", "Download runs (CSV)")
