from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.arena import KEYS, PREREG, QUICK, THEORY_NAMES, tune
from heiner_abm.mechanisms import (AXES, MAP_RANGES, QUICK_STUDY, TARGETS, VARIANT_LABELS, StudyPlan, default_plan,
                                   gain_map, run_study, theory_maps)
from ui.common import CAT, DIVERGING, download, hypothesis_card, style, verdict, prereg_explainer
from ui.common import evidence_note

st.title("Mechanisms: when does restricting flexibility pay?")
evidence_note("simulation")
st.caption("Two experiments built on the agent tournament. The first separates Heiner's principle (restrict "
           "deviations when they are unreliable) from the cost of applying it. The second maps, across separate "
           "sources of uncertainty, where restriction pays and which decision rule does best.")

hypothesis_card(
    "MECH1",
    "A focal firm aims at the same target as the always-adjusting rule and differs only in **when** it moves. The "
    "simulator knows the true consequences of every decision, so it can build an **oracle**: the reliability "
    "condition with the true expected gain of each kind of deviation, estimated from a long independent run of the "
    "same environment (it knows its rule's reliability, not the future). Comparing the oracle with learners that "
    "judge their past decisions with their own model, or with the true model, splits the cost of applying the "
    "principle into estimation from limited experience and judging with a misspecified model.")
hypothesis_card(
    "MECH2",
    "Environments are drawn by Latin hypercube over separate sources of uncertainty: risk (cost volatility), "
    "perception error, model misspecification (shift hazard and model-updating lag), competence and stakes. The "
    "gain from restriction is related to the measured error-to-signal ratio K of the flexible rule and to each "
    "source of uncertainty, and a quadratic metamodel maps which theory's agent does best where.")

# ------------------------------------------------------------------------------------------------ plan
st.header("1 · Plan", divider="gray")
prereg_explainer()
scale = st.radio("Protocol", ["Registered", "Quick check"], horizontal=True, key="mech_scale",
                 help="Registered runs the frozen study plan (about 30 seconds, plus tuning the rivals if the "
                      "registered tournament has not been run in this session). Quick check is a small exploratory "
                      "run.")
t_plan = PREREG if scale == "Registered" else QUICK
plan = default_plan(t_plan.digest)
if scale == "Quick check":
    plan = replace(plan, **QUICK_STUDY)
registered = default_plan(PREREG.digest)
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Study plan hash:** `{plan.digest}` "
                + ("· registered plan" if plan.digest == registered.digest
                   else f"· **exploratory** (registered: `{registered.digest}`)"))
    st.caption(f"{plan.n_envs} environments (Latin hypercube) · {plan.periods} periods · oracle values from "
               f"{plan.oracle_periods}-period independent runs · memories {', '.join(f'{m:g}' for m in plan.memories)}"
               f" · background rivals tuned under tournament plan `{plan.tournament_plan}`")
with c2:
    st.download_button("Download plan (.json)", plan.to_json().encode(), file_name=f"study_{plan.digest}.json",
                       mime="application/json", width="stretch",
                       help="File with a registry (for example OSF) before running a confirmatory study.")
st.dataframe(pd.DataFrame(plan.hypotheses, columns=["ID", "Hypothesis", "Decision rule"]), hide_index=True,
             width="stretch")
st.caption("Uncertainty axes: " + "; ".join(f"{AXES[k]} {lo:g}–{hi:g}" for k, (lo, hi) in plan.ranges))

store = st.session_state.setdefault("mech_results", {})
if st.button("Run study", type="primary", key="mech_run"):
    bar = st.progress(0.0, "Starting")
    arena_res = st.session_state.get("arena_results", {}).get(t_plan.digest)
    if arena_res is not None:
        tuned = arena_res.tuned
    else:
        tuned, _ = tune(t_plan, lambda f, m: bar.progress(0.5 * min(f, 1.0), "Tuning rivals · " + m))
    store[plan.digest] = run_study(plan, tuned, lambda f, m: bar.progress(0.5 + 0.5 * min(f, 1.0), m))
    bar.empty()
res = store.get(plan.digest)
if res is None:
    st.info("Run the study to see results. If the agent tournament has been run for the same plan in this session, "
            "its tuned rivals are reused.", icon="ℹ️")
    st.stop()

# ------------------------------------------------------------------------------------------------ verdicts
st.header("2 · Pre-registered hypotheses", divider="gray")
if res.exploratory:
    st.warning("Exploratory run: the plan differs from the registered one, so these verdicts are not confirmatory.",
               icon="⚠️")
for _, r in res.verdicts.iterrows():
    verdict("support" if r["verdict"] == "supported" else "reject",
            f"**{r['id']}** · {r['hypothesis']} **{r['verdict'].capitalize()}**: {r['result']}.")

# ------------------------------------------------------------------------------------------------ principle
st.header("3 · Principle versus implementation", divider="gray")
eff = res.effects
main = eff[eff["variant"].isin(["band", "rc_learned", "rc_true_model", "rc_oracle"])]
fig = go.Figure()
for k, target in enumerate(TARGETS):
    d = main[main["target"] == target]
    fig.add_trace(go.Bar(name=f"{target.capitalize()}-based target", y=d["label"], x=d["diff"], orientation="h",
                         marker_color=CAT[k], error_x=dict(type="data", symmetric=False, array=d["hi"] - d["diff"],
                                                           arrayminus=d["diff"] - d["lo"], thickness=1.2, width=4)))
fig.add_vline(x=0, line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"))
fig.update_layout(barmode="group")
fig.update_xaxes(title="Profit per period vs always adjusting toward the same target (95% CI)")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style(fig, 360, "What each way of deciding when to move adds to the same flexible rule"))
st.markdown("**Decomposition of the cost of applying the reliability condition**")
st.dataframe(res.decomposition, hide_index=True, width="stretch", column_config={
    "target": "Target", "component": "Component",
    "diff": st.column_config.NumberColumn("Profit per period", format="%+.0f"),
    "lo": st.column_config.NumberColumn("95% lo", format="%.0f"),
    "hi": st.column_config.NumberColumn("95% hi", format="%.0f"), "p": st.column_config.NumberColumn("p", format="%.3f")})
st.caption("Total cost of learning = oracle − learned = (oracle − true-model learner: estimation from limited, noisy "
           "experience) + (true-model learner − learned: judging past decisions with a misspecified demand model).")

mem = eff[eff["variant"].str.startswith("rc_memory_")].assign(memory=lambda d: d["variant"].str[10:].astype(float))
fm = go.Figure()
for k, target in enumerate(TARGETS):
    d = mem[mem["target"] == target].sort_values("memory")
    eff_n = 1 / (1 - d["memory"])
    fm.add_trace(go.Scatter(x=eff_n, y=d["diff"], mode="lines+markers", name=f"{target.capitalize()}-based, learned",
                            line=dict(color=CAT[k], width=2),
                            error_y=dict(type="data", symmetric=False, array=d["hi"] - d["diff"],
                                         arrayminus=d["diff"] - d["lo"], thickness=1, width=3)))
    o = eff[(eff["target"] == target) & (eff["variant"] == "rc_oracle")]
    if len(o):
        fm.add_hline(y=float(o["diff"].iloc[0]), line=dict(color=CAT[k], width=1.5, dash="dash"),
                     annotation_text=f"oracle ({target})", annotation_position="top left")
fm.add_hline(y=0, line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"))
fm.update_xaxes(title="Effective experience pooled (1 / (1 − memory), decisions)", type="log")
fm.update_yaxes(title="Profit vs always")
st.plotly_chart(style(fm, 340, "Does more experience close the gap to the oracle?"))
st.caption("Longer memory pools more experience (less noise) but adapts more slowly when the environment shifts: the "
           "agent faces its own reliability trade-off when estimating reliability.")

# ------------------------------------------------------------------------------------------------ boundary
st.header("4 · The boundary: restriction pays where the flexible rule is unreliable", divider="gray")
bd = res.boundary
data = bd["data"]
vsel = st.radio("Selection rule", ["rc_oracle", "rc_learned", "band"], horizontal=True, key="mech_bvar",
                format_func=lambda v: VARIANT_LABELS[v],
                help="Which restriction's gain over always adjusting to plot against the flexible rule's K.")
fb = go.Figure()
for k, target in enumerate(TARGETS):
    d = data[data["target"] == target]
    fb.add_trace(go.Scatter(x=np.exp(d["lnK"]), y=d[f"gain_{vsel}"], mode="markers", name=f"{target}-based",
                            marker=dict(color=CAT[k], size=6, opacity=0.6)))
    b = bd[(target, vsel)]
    xs = np.linspace(d["lnK"].min(), d["lnK"].max(), 50)
    fb.add_trace(go.Scatter(x=np.exp(xs), y=b["intercept"] + b["slope"] * xs, mode="lines", showlegend=False,
                            line=dict(color=CAT[k], width=2)))
fb.add_hline(y=0, line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"))
fb.update_xaxes(title="Error-to-signal ratio K of the always-adjusting rule (log scale)", type="log")
fb.update_yaxes(title="Profit gain from restriction")
st.plotly_chart(style(fb, 380, f"{VARIANT_LABELS[vsel]} − always, by the flexible rule's reliability"))
st.dataframe(pd.DataFrame([dict(target=t, rule=VARIANT_LABELS[v], slope_per_lnK=b["slope"], p=b["p"], r2=b["r2"],
                                K_star=b["K_star"]) for (t, v), b in ((k, x) for k, x in bd.items() if isinstance(k, tuple))]),
             hide_index=True, width="stretch", column_config={
                 "target": "Target", "rule": "Selection rule",
                 "slope_per_lnK": st.column_config.NumberColumn("Slope per ln K", format="%+.0f"),
                 "p": st.column_config.NumberColumn("p", format="%.3f"), "r2": st.column_config.NumberColumn("R²", format="%.2f"),
                 "K_star": st.column_config.NumberColumn("Break-even K*", format="%.2f",
                                                         help="K above which the fitted gain is positive (rising "
                                                              "slopes only).")})

st.subheader("Which source of uncertainty drives the gain?")
ty = res.types
tsel = st.radio("Target", list(TARGETS), horizontal=True, key="mech_ttarget", format_func=lambda t: f"{t}-based",
                help="Target of the always-adjusting rule and of the restrictions compared with it.")
ft = go.Figure()
for k, v in enumerate(["rc_oracle", "rc_learned", "band"]):
    d = ty[(ty["target"] == tsel) & (ty["variant"] == v)]
    ft.add_trace(go.Bar(name=VARIANT_LABELS[v], y=d["label"], x=d["coef"], orientation="h", marker_color=CAT[k],
                        error_x=dict(type="data", symmetric=False, array=d["hi"] - d["coef"],
                                     arrayminus=d["coef"] - d["lo"], thickness=1.2, width=4)))
ft.add_vline(x=0, line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"))
ft.update_layout(barmode="group")
ft.update_yaxes(autorange="reversed")
ft.update_xaxes(title="Standardized coefficient on the gain from restriction (95% bootstrap CI)")
st.plotly_chart(style(ft, 420, "Sources of uncertainty and the value of restriction"))

# ------------------------------------------------------------------------------------------------ maps
st.header("5 · Maps: which decision rule works where?", divider="gray")
c1, c2 = st.columns(2)
ax_list = list(MAP_RANGES)
x_ax = c1.selectbox("Horizontal axis", ax_list, index=ax_list.index("delta"), format_func=AXES.get, key="mech_x",
                    help="First uncertainty axis of the map; the others are held at their medians.")
y_ax = c2.selectbox("Vertical axis", [a for a in ax_list if a != x_ax], format_func=AXES.get, key="mech_y",
                    index=[a for a in ax_list if a != x_ax].index("hazard") if x_ax != "hazard" else 0,
                    help="Second uncertainty axis of the map.")
gx, gy, preds, win, r2 = theory_maps(res.theories, x_ax, y_ax)
names = [THEORY_NAMES[k] for k in KEYS]
present = sorted(set(win.ravel()))
cs = [[i / max(len(present) - 1, 1), CAT[j % len(CAT)]] for i, j in enumerate(present)]
zmap = np.vectorize({j: i for i, j in enumerate(present)}.get)(win)
fw = go.Figure(go.Heatmap(z=zmap, x=gx, y=gy, colorscale=cs if len(present) > 1 else [[0, CAT[present[0]]], [1, CAT[present[0]]]],
                          showscale=False, text=np.vectorize(lambda j: names[j])(win),
                          hovertemplate="%{text}<extra></extra>"))
for j in present:
    fw.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=names[j],
                            marker=dict(size=10, color=CAT[j % len(CAT)], symbol="square")))
fw.update_xaxes(title=AXES[x_ax]); fw.update_yaxes(title=AXES[y_ax])
st.plotly_chart(style(fw, 420, "Predicted best theory (profit relative to the market mean)"))
weak = [THEORY_NAMES[k] for k, v in r2.items() if v < 0.2]
st.caption("Cross-validated R² of each theory's metamodel: " + ", ".join(f"{THEORY_NAMES[k]} {v:.2f}" for k, v in r2.items())
           + ". " + ("Maps for metamodels with low cross-validated R² mostly reflect noise; read them with care."
                     if weak else ""))
c1, c2 = st.columns(2)
g_target = c1.radio("Target", list(TARGETS), horizontal=True, key="mech_gt", format_func=lambda t: f"{t}-based",
                    help="Target for the gain-from-restriction map.")
g_var = c2.radio("Restriction", ["rc_oracle", "rc_learned", "band"], horizontal=True, key="mech_gv",
                 format_func=lambda v: VARIANT_LABELS[v], help="Which restriction's gain over always to map.")
gx, gy, z, r2g = gain_map(res.focal, g_target, g_var, x_ax, y_ax)
lim = float(np.nanmax(np.abs(z))) or 1.0
fg = go.Figure(go.Heatmap(z=z, x=gx, y=gy, zmin=-lim, zmax=lim, colorscale=DIVERGING,
                          colorbar=dict(title="gain vs<br>always"),
                          hovertemplate="%{x:.3g}, %{y:.3g}<br>gain %{z:+.0f}<extra></extra>"))
fg.add_trace(go.Contour(z=z, x=gx, y=gy, contours=dict(start=0, end=0, size=1, coloring="none"),
                        line=dict(color="black", width=2), showscale=False, hoverinfo="skip"))
fg.update_xaxes(title=AXES[x_ax]); fg.update_yaxes(title=AXES[y_ax])
st.plotly_chart(style(fg, 420, f"Where {VARIANT_LABELS[g_var].lower()} beats always (blue; black line = break-even)"))
st.caption(f"Cross-validated R² of this metamodel: {r2g:.2f}.")

c1, c2, c3 = st.columns(3)
with c1:
    download(res.focal, f"mechanisms_focal_{res.plan_hash}.csv", "Focal-firm data")
with c2:
    download(res.theories, f"mechanisms_theories_{res.plan_hash}.csv", "Theory map data")
with c3:
    download(res.oracle_values, f"mechanisms_oracle_{res.plan_hash}.csv", "Oracle values")

st.header("6 · Limits", divider="gray")
st.markdown(
    "* The oracle knows the average reliability of each kind of deviation in its environment, estimated under the "
    "true-model learner's behavior. It is an upper benchmark for the principle, not the best conceivable policy.\n"
    "* Heiner's comparison values one deviation followed by rule B. A rule that keeps adjusting values decisions "
    "differently, so even a perfectly informed reliability condition need not beat always adjusting.\n"
    "* Maps come from a quadratic metamodel; cells far from sampled environments, and models with low "
    "cross-validated R², are extrapolations.")
