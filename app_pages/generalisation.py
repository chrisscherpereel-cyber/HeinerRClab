from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.registered import MECHANISMS
from heiner_abm.tasks import (AXIS_LABELS, LAYER_LABELS, LAYERS, QUICK_TASKS, RANGES, TASKS, TaskPlan, boundary,
                              run_tasks, tercile_table, wide)
from ui.common import CAT, download, hypothesis_card, style, verdict, prereg_explainer
from ui.common import evidence_note

st.title("Generalization: beyond the market")
evidence_note("simulation")
st.caption("If restriction pays only where the flexible rule is unreliable in other decision tasks too, the finding "
           "is about decision making, not about cobweb markets. Three tasks with the same structure but a different "
           "payoff structure are tested with the same selection layers and the same boundary test.")

hypothesis_card(
    "GEN",
    "Each task has a **default** (a slow, long-memory estimate that ignores most new information, the analogue of a "
    "firm that keeps its established output), a **flexible alternative** (a fast re-estimate that responds to "
    "every observation) and a controllable gap between **difficulty** (outcome noise, frequency of unannounced "
    "shifts) and **competence** (observation error, how strongly the fast estimate reacts). Every period the agent "
    "decides whether to deviate from the default to the flexible action. No cost is charged for deviating, so any "
    "gain from restriction comes from reliability alone.")

with st.expander("The three tasks and the selection layers", icon="ℹ️"):
    st.markdown(
        "* **Inventory (newsvendor with shifting demand).** Each period the agent sets an order level; unmet demand "
        "costs 4 per unit, leftover stock 1 per unit, so the best order level is the 80% quantile of demand. Mean "
        "demand jumps without warning. Both rules set the order at that quantile of a smoothed demand forecast; the "
        "default smooths slowly, the flexible rule quickly.\n"
        "* **Learning with shifting payoffs.** Two options; one pays 1 more on average, and which one is better "
        "swaps without warning. The agent sees both payoffs every period (full feedback) with error. Both rules "
        "pick the option with the higher smoothed payoff; the default smooths slowly, the flexible rule quickly.\n"
        "* **Irreversible investment.** Each period one project arrives. Its value is the current mean project "
        "quality, which shifts without warning between booms and busts, plus project-specific noise. Investing "
        "commits the firm: a bad project also costs a write-off of half its loss. Both rules invest when the expected "
        "payoff at their estimate of mean quality is positive; the default estimates slowly, the flexible rule "
        "quickly.")
    st.dataframe(pd.DataFrame([dict(Layer=LAYER_LABELS[k], Rule=d) for k, d in (
        ("always", "Always take the flexible action (no restriction)."),
        ("band", "Deviate only when the flexible action differs from the default by more than b times the agent's "
                 "own noise estimate; b is tuned on separate training environments."),
        ("rc_learned", "Reliability condition learned from experience: per signal-strength bin, a running mean of "
                       "the ex post gain of deviating, judged with the agent's own noisy observations; deviate only "
                       "where it is positive."),
        ("rc_oracle", "The same decision with the true mean gain per bin, from a long independent run of the same "
                      "environment (it knows its rule's reliability, not the future)."),
        ("ruleb", "Never deviate from the default."))]), hide_index=True, width="stretch")
    st.markdown("The error-to-signal ratio is defined as in the market: "
                "K = √( E[(flexible − optimum)²] / E[(optimum − default)²] ), the flexible rule's error relative to "
                "how far the optimum lies from the default.")

st.header("1 · Plan", divider="gray")
prereg_explainer()
scale = st.radio("Protocol", ["Registered", "Quick check"], horizontal=True, key="gen_scale",
                 help="Registered runs the frozen plan (120 environments per task, about 25 seconds). Quick check is "
                      "a small exploratory run.")
registered = TaskPlan()
plan = registered if scale == "Registered" else replace(registered, **QUICK_TASKS)
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Plan hash:** `{plan.digest}` " + ("· registered plan" if plan.digest == registered.digest
                                                       else f"· **exploratory** (registered: `{registered.digest}`)"))
    st.caption(f"{plan.n_envs} environments per task (Latin hypercube) · {plan.periods} periods · band tuned on "
               f"{plan.n_train} separate environments · oracle values from {plan.oracle_periods}-period independent "
               "runs")
with c2:
    st.download_button("Download plan (.json)", plan.to_json().encode(), file_name=f"tasks_{plan.digest}.json",
                       mime="application/json", width="stretch", help="The frozen plan, including a code hash.")
st.dataframe(pd.DataFrame(plan.hypotheses, columns=["ID", "Hypothesis", "Decision rule"]), hide_index=True,
             width="stretch")
st.caption(" · ".join(f"**{TASKS[t]}:** " + "; ".join(f"{AXIS_LABELS[k]} {lo:g}–{hi:g}" for k, (lo, hi) in
                                                       RANGES[t].items()) for t in plan.tasks))

store = st.session_state.setdefault("gen_results", {})
if st.button("Run study", type="primary", key="gen_run"):
    bar = st.progress(0.0, "Starting")
    store[plan.digest] = run_tasks(plan, lambda f, m: bar.progress(min(f, 1.0), m))
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

st.header("3 · The boundary in each task", divider="gray")
cols = st.columns(len(plan.tasks))
for col, t in zip(cols, plan.tasks):
    w = wide(res.runs, t)
    bd = boundary(res.runs, t)
    fig = go.Figure()
    for i, v in enumerate(("rc_oracle", "rc_learned", "band")):
        y = w[v] - w["always"]
        fig.add_trace(go.Scatter(x=np.exp(w["lnK"]), y=y, mode="markers", name=LAYER_LABELS[v],
                                 marker=dict(color=CAT[i], size=6, opacity=0.7)))
        xs = np.linspace(w["lnK"].min(), w["lnK"].max(), 30)
        fig.add_trace(go.Scatter(x=np.exp(xs), y=bd[v]["intercept"] + bd[v]["slope"] * xs, mode="lines",
                                 line=dict(color=CAT[i]), showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=0, line=dict(color="gray", dash="dot"))
    fig.add_vline(x=1, line=dict(color="gray", dash="dot"))
    fig.update_xaxes(title="Error-to-signal ratio K (log scale)", type="log")
    fig.update_yaxes(title="Gain over always deviating, per period")
    col.markdown(f"**{TASKS[t]}**")
    col.plotly_chart(style(fig, height=380), width="stretch")
    col.caption(f"Break-even K* ≈ {bd['rc_oracle']['K_star']:.2f} (oracle), {bd['rc_learned']['K_star']:.2f} "
                f"(learned). Band width b = {res.bands[t]:g}.")

st.subheader("Gain by tercile of K")
tt = pd.concat([tercile_table(res.runs, t, plan.n_boot) for t in plan.tasks], ignore_index=True)
tt["layer"] = tt["layer"].map(LAYER_LABELS)
tt["task"] = tt["task"].map(TASKS)
st.dataframe(tt.rename(columns={"task": "Task", "tercile": "Tercile", "K_range": "K range", "layer": "Layer",
                                "gain": "Gain over always", "lo": "95% CI low", "hi": "95% CI high", "n": "n"}),
             hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.3f")
                            for c in ("Gain over always", "95% CI low", "95% CI high")})

st.header("4 · Comparison with the market", divider="gray")
st.markdown("In the market (Mechanisms page), " + MECHANISMS["heiner"][0].lower() + MECHANISMS["heiner"][1:])
st.markdown("The same ordering appears in every task: restriction pays only where K is high and is worth nothing "
            "where K is low. The break-even from the linear fit lies below K = 1 (in the investment task the fit is "
            "poor, and the gain by tercile turns positive between K ≈ 0.9 and 1.6). The cost of learning the "
            "reliability depends on the task: large in the market, small or absent in these tasks, where the ex "
            "post gain of a deviation can be judged exactly from what the agent observes.")

st.header("5 · Where in the uncertainty space?", divider="gray")
t = st.selectbox("Task", plan.tasks, format_func=TASKS.get, key="gen_task", help="Task to map.")
c1, c2 = st.columns(2)
x = c1.selectbox("Horizontal axis", list(RANGES[t]), index=0, format_func=AXIS_LABELS.get, key="gen_x",
                 help="Difficulty or competence axis.")
y = c2.selectbox("Vertical axis", list(RANGES[t]), index=2, format_func=AXIS_LABELS.get, key="gen_y",
                 help="Difficulty or competence axis.")
w = wide(res.runs, t)
g = w["rc_oracle"] - w["always"]
lim = float(np.abs(g).quantile(0.95)) or 1.0
fig = go.Figure(go.Scatter(x=w[x], y=w[y], mode="markers",
                           marker=dict(color=g, colorscale="RdBu", cmin=-lim, cmax=lim, size=10,
                                       colorbar=dict(title="Gain")),
                           customdata=np.stack([g, np.exp(w["lnK"])], 1),
                           hovertemplate="gain %{customdata[0]:.3f}<br>K %{customdata[1]:.2f}<extra></extra>"))
fig.update_xaxes(title=AXIS_LABELS[x])
fig.update_yaxes(title=AXIS_LABELS[y])
st.plotly_chart(style(fig, height=420), width="stretch")
st.caption("Each point is one environment; color is the oracle's gain from restriction over always deviating "
           "(blue: restriction pays).")
download(res.runs, f"generalisation_{plan.digest}.csv", "Download runs (CSV)")
