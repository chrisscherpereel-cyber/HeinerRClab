import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.ecology import ECO_DESIGNS, ECO_NAME, INERT_FORMS, QUICK_ECO, EcoPlan, run_ecology_study
from heiner_abm.registered import TOURNAMENT_PLAN, registered_tuned
from heiner_abm.rulechoice import level_env
from ui.common import CAT, download, prereg_explainer, style, verdict

st.title("Organizational ecology: selection through exit and entry")
st.caption("Organizational ecology (Hannan & Freeman) explains industry change by selection rather than adaptation: "
           "organizations are structurally inert, poorly fitted ones fail, and new ones take their place. Here the "
           "theory competes as agents, tuned with the same budget as every other theory. Its selection mechanism "
           "then runs on the whole population of decision forms.")

with st.expander("The ecology agents and the selection mechanism", icon="ℹ️"):
    st.dataframe(pd.DataFrame([dict(Design=d.name, Rule=d.rule,
                                    Parameters=", ".join(f"{n} ∈ [{s.lo:g}, {s.hi:g}]" for n, s in d.SPACE.items()))
                               for d in ECO_DESIGNS.values()]), hide_index=True, width="stretch")
    st.markdown(
        "**Selection.** Every firm starts with the arena's survival buffer (20 periods of Nash profit) and accumulates "
        "profit. A firm whose capital turns negative exits. An entrant takes its slot with the incumbents' mean output "
        "and a fresh buffer. The entrant's form is copied from a randomly drawn surviving firm in the same market, or, "
        "with a small mutation probability, drawn uniformly from all forms. Every form runs in every slot each period "
        "on the slot's actual history, so entrants start with a warm state (as in the rule-choice study).\n\n"
        "**Inert forms** are structural inertia, crisis reorganization and rule B. All other tournament designs "
        "count as flexible.\n\n"
        "**Tuning.** Both ecology designs get the arena's equal budget of Latin-hypercube candidates on separate "
        "training environments, in mixed markets against every theory's registered tuned design. The better design "
        "enters. Other theories keep the parameters they were tuned to in the registered tournament "
        f"(plan `{TOURNAMENT_PLAN}`). The registered tournament itself is unchanged.")
    lv = pd.DataFrame([dict(level=u, **{k: getattr(level_env(u, 0), k) for k in ("delta", "noise", "hazard")})
                       for u in EcoPlan().levels])
    lv.columns = ["Uncertainty level u", "Cost volatility Δ", "Perception error σ", "Demand-shift hazard λ"]
    st.dataframe(lv, hide_index=True, width="stretch")

st.header("1 · Plan", divider="gray")
prereg_explainer()
scale = st.radio("Protocol", ["Full plan", "Quick check"], horizontal=True, key="eco_scale",
                 help="Full plan: equal-budget tuning, held-out head-to-head and 4 uncertainty levels × 8 replications "
                      "of 3,000-period selection runs (about 15 seconds). Quick check is a small exploratory run.")
plan = EcoPlan(tournament_plan=TOURNAMENT_PLAN)
full = plan
if scale == "Quick check":
    plan = EcoPlan(**{**QUICK_ECO.__dict__, "tournament_plan": TOURNAMENT_PLAN})
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Plan hash:** `{plan.digest}`" + ("" if plan.digest == full.digest else
                                                     f" · **quick check** (full plan: `{full.digest}`)"))
    st.caption(f"Tuning budget {plan.budget} per design on {plan.n_train} training environments · head-to-head on "
               f"{plan.n_test} held-out environments · selection at u = {', '.join(f'{u:g}' for u in plan.levels)} × "
               f"{plan.reps} replications · {plan.copies} initial firm(s) per form · {plan.periods} periods · mutation "
               f"{plan.mutation:g}")
    st.caption("This study has not yet been run as a registered study, so its verdicts are exploratory until a run "
               "is recorded under this plan hash.")
with c2:
    st.download_button("Download plan (.json)", plan.to_json().encode(), file_name=f"ecology_{plan.digest}.json",
                       mime="application/json", width="stretch")
st.dataframe(pd.DataFrame(plan.hypotheses, columns=["ID", "Hypothesis", "Decision rule"]), hide_index=True,
             width="stretch")

store = st.session_state.setdefault("eco_results", {})
if st.button("Run study", type="primary", key="eco_run"):
    bar = st.progress(0.0, "Starting")
    store[plan.digest] = run_ecology_study(plan, registered_tuned(), progress=lambda f, m: bar.progress(min(f, 1.0), m))
    bar.empty()
res = store.get(plan.digest)
if res is None:
    st.info("Run the study to see results.", icon="ℹ️")
    st.stop()

st.header("2 · Hypotheses", divider="gray")
for _, r in res.verdicts.iterrows():
    verdict("support" if r["verdict"] == "supported" else "reject",
            f"**{r['id']}** · {r['hypothesis']}  \n{r['detail']}")
st.caption("S1 and S3 are organizational ecology's own predictions. S2 sets Heiner's prediction (a selection "
           "advantage for rigid forms that grows with uncertainty) against ecology's (inertia selected everywhere, "
           "no volatility gradient).")

st.header("3 · Selection dynamics", divider="gray")
forms = list(dict.fromkeys(res.paths["form"]))
colors = {f: CAT[k % len(CAT)] for k, f in enumerate(forms)}
cols = st.columns(2)
for k, (u, g) in enumerate(res.paths.groupby("level")):
    fig = go.Figure()
    for f in forms:
        d = g[g["form"] == f]
        nm = d["name"].iloc[0] + (" (inert)" if f in INERT_FORMS else "")
        fig.add_trace(go.Scatter(x=d["period"], y=d["share"], name=nm, mode="lines", stackgroup="one",
                                 line=dict(color=colors[f], width=1.2),
                                 hovertemplate=f"{nm}<br>period %{{x}}<br>share %{{y:.1%}}<extra></extra>"))
    fig.update_xaxes(title="Period"); fig.update_yaxes(title="Share of firms", range=[0, 1], tickformat=".0%")
    fig.update_layout(hovermode="x unified")
    cols[k % 2].plotly_chart(style(fig, 360, f"Uncertainty level u = {u:g}"))
agg = res.runs.groupby("level")[["inert_start", "inert_end", "inert_change", "exits_inert", "exits_flexible",
                                 "total_exits"]].mean().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=agg["level"], y=agg["inert_change"], mode="lines+markers", name="Change in inert share",
                         line=dict(color=CAT[0], width=2)))
fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.8)", width=1))
fig.update_xaxes(title="Uncertainty level u"); fig.update_yaxes(title="End-of-run minus start-of-run inert share")
st.plotly_chart(style(fig, 320, "Does selection favor inert forms more as uncertainty rises? (S2)"))
st.dataframe(agg, hide_index=True, width="stretch", column_config={
    "level": "Uncertainty u", "inert_start": st.column_config.NumberColumn("Inert share, start", format="%.2f"),
    "inert_end": st.column_config.NumberColumn("Inert share, end", format="%.2f"),
    "inert_change": st.column_config.NumberColumn("Change", format="%+.2f"),
    "exits_inert": st.column_config.NumberColumn("Exits / 1,000 firm-periods, inert", format="%.2f"),
    "exits_flexible": st.column_config.NumberColumn("Exits / 1,000 firm-periods, flexible", format="%.2f"),
    "total_exits": st.column_config.NumberColumn("Exits per run", format="%.1f")})

st.header("4 · Head-to-head with every theory", divider="gray")
h = res.head_to_head.groupby("name").agg(rank=("rank", "mean"), profit=("profit", "mean"), ruined=("ruined", "mean"),
                                         change_rate=("change_rate", "mean")).sort_values("rank").reset_index()
st.dataframe(h, hide_index=True, width="stretch", column_config={
    "name": "Theory", "rank": st.column_config.NumberColumn("Mean profit rank (1 = best)", format="%.2f"),
    "profit": st.column_config.NumberColumn("Mean profit per period", format="%.1f"),
    "ruined": st.column_config.NumberColumn("Share ruined", format="%.2f"),
    "change_rate": st.column_config.NumberColumn("Share of periods with an output change", format="%.2f")})
st.caption(f"Held-out mixed markets: one firm per theory with its registered tuned design, plus {ECO_NAME}'s selected "
           f"design ({ECO_DESIGNS[res.design].name}).")
st.markdown("**Tuning result:**")
st.dataframe(res.tuning, hide_index=True, width="stretch")
c1, c2 = st.columns(2)
with c1:
    download(res.runs, "ecology_selection_runs.csv", "Download selection runs")
with c2:
    download(res.head_to_head, "ecology_head_to_head.csv", "Download head-to-head")
