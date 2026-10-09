import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm import learnability as L
from heiner_abm import learnability_market as LM
from heiner_abm.registered import LEARN_PLAN
from ui.common import CAT, download, evidence_note, set_engine, style, verdict

st.title("When can reliability be learned before the environment changes?")
evidence_note("simulation")
set_engine("learnability")
st.warning("**Proposed extension, frozen in the repository, not externally preregistered.** The question, the "
           "learnability ratio and the policies' operationalizations are this laboratory's own. The specification and "
           "code are fixed by a hash; no registration with an external registry (such as OSF) has taken place.",
           icon="🧩")
st.markdown(
    "An agent can only learn whether adapting is reliable from the informative feedback it gets **within a regime**. "
    "The proposed **learnability ratio** R compares the observations needed to determine the sign of the advantage of "
    "adapting (with 90% one-sided confidence) with the informative observations available in a regime. R < 1: the "
    "advantage can typically be learned before the environment changes; R > 1: it cannot. Whether R explains how "
    "policies perform beyond volatility and observation noise is tested, not assumed.")

with st.expander("Design (from the frozen specification)", expanded=False):
    st.markdown(L.__doc__.split("Task\n", 1)[1].replace("\n    ", "\n").replace("    ", " "))
    st.markdown("**Market replication.** " + LM.__doc__.split("\n\n", 1)[1])

plan_full = L.LEARN_PREREG
st.subheader("Registration")
c1, c2 = st.columns([2, 1])
same = plan_full.digest == LEARN_PLAN
c1.markdown(f"Frozen plan `{LEARN_PLAN}` · current code `{plan_full.digest}` "
            + ("(identical)" if same else "(**different: results would be exploratory**)"))
c2.download_button("Registration document (Markdown)", L.registration_document(plan_full, LEARN_PLAN).encode(),
                   file_name=f"learnability_registration_{plan_full.digest}.md", mime="text/markdown",
                   width="stretch")
st.dataframe(pd.DataFrame([dict(ID=h[0], Hypothesis=h[1], Decision_rule=h[2]) for h in plan_full.hypotheses]
                          + [dict(ID=c[0], Hypothesis=c[1], Decision_rule=c[2]) for c in plan_full.negative_controls]),
             hide_index=True, width="stretch")

scale = st.radio("Protocol", ["Quick check (exploratory)", "Registered plan"], horizontal=True, key="lrn_scale",
                 help="The registered plan runs the frozen specification (about 8 minutes). The quick check is a small "
                      "exploratory run with a different hash; it is not evidence for or against the hypotheses.")
quick = scale.startswith("Quick")
plan = replace(plan_full, **L.QUICK_LEARN) if quick else plan_full
mplan = LM.MarketPlan(**(LM.QUICK_MARKET if quick else {}))


@st.cache_data(show_spinner=False, max_entries=2)
def cached_study(plan_js: str, mplan_js: str, quick: bool):
    p = replace(L.LEARN_PREREG, **L.QUICK_LEARN) if quick else L.LEARN_PREREG
    return L.run_study(p, market_plan=LM.MarketPlan(**json.loads(mplan_js)))


if st.button("Run study", type="primary", key="lrn_run"):
    st.session_state["lrn_key"] = (plan.digest, quick)
if st.session_state.get("lrn_key") != (plan.digest, quick):
    st.info("Press **Run study**.")
    st.stop()
with st.spinner("Tuning on training configurations, measuring R on pilot paths, testing on untouched paths…"):
    res = cached_study(plan.to_json(), json.dumps(asdict(mplan)), quick)

st.header("Results", divider="gray")
if res.plan_hash != LEARN_PLAN:
    st.info(f"Exploratory run (plan `{res.plan_hash}`, registered `{LEARN_PLAN}`).")
st.caption(f"Tuned on training configurations only: flexible forecast gain {res.hyperparameters['gain']:g}, band "
           f"b = {res.hyperparameters['band']['b']:g}, confidence-sensitive gate {res.hyperparameters['gate']}, "
           f"change detection {res.hyperparameters['bocpd']}, robust order {res.hyperparameters['dro']}.")
with st.expander("Tuning budget: the same number of candidate settings for every policy that has hyperparameters"):
    st.markdown("Each policy below was given the same number of candidate settings from the frozen plan, scored on "
                "the same training configurations and paths; candidate 0 is the registered starting value, so the "
                "search can only help. *Retain the default*, the *estimated-gain gate* and the *ORACLE* have no free "
                "hyperparameters and receive none. Policies are compared on equivalent information, adaptation costs "
                "and tuning budgets.")
    if len(res.tuning):
        st.dataframe(res.tuning.round(3), hide_index=True, width="stretch")
    n = sorted(set(res.tuning["n_candidates"])) if len(res.tuning) else []
    if len(n) > 1:
        st.warning(f"Unequal budgets in this run: {n}. Results are not a like-for-like comparison.", icon="⚠️")
for _, r in res.verdicts.iterrows():
    verdict("support" if r["verdict"] == "supported" else "neutral", f"**{r['id']}** {r['verdict']}: {r['result']}")
st.subheader("Negative controls")
for _, r in res.controls.iterrows():
    verdict("support" if r["passed"] else "reject", f"**{r['id']}** {'passed' if r['passed'] else 'FAILED'}: "
            f"{r['result']}")

st.subheader("Advantage of the confidence-sensitive gate against the learnability ratio")
st.caption("Each point is one test configuration (untouched paths): paired net-payoff difference per period between "
           "the gate and the better fixed rule (chosen on pilot paths), with its 95% interval. R was measured on "
           "independent pilot paths.")
f = go.Figure()
for k, (name, df) in enumerate((("Jump family (tuning family, new paths)", res.tests),
                                ("New families (switching, drifting)", res.new_families))):
    if len(df):
        f.add_trace(go.Scatter(x=df["log10_ratio"], y=df["adv_gate_lcb"], mode="markers", name=name,
                               marker=dict(color=CAT[k], size=8),
                               error_y=dict(type="data", symmetric=False, array=df["adv_gate_lcb_hi"] - df["adv_gate_lcb"],
                                            arrayminus=df["adv_gate_lcb"] - df["adv_gate_lcb_lo"], thickness=1)))
f.add_hline(y=0, line=dict(color="gray", dash="dot"))
f.add_vline(x=0, line=dict(color="gray", dash="dot"))
f.update_xaxes(title="log10 learnability ratio R (R < 1 left of 0)")
f.update_yaxes(title="Gate − better fixed rule (net payoff / period)")
st.plotly_chart(style(f, 380))

st.subheader("One factor at a time (paired effects against always adapting, 95% intervals)")
pols = [p for p in L.POLICIES if p != "always"]
sw = res.sweeps
show = sw[["factor", "level", "log10_ratio"] + [f"vs_always_{p}" for p in pols]].rename(
    columns={f"vs_always_{p}": L.POLICY_LABELS[p] for p in pols})
st.dataframe(show.round(2), hide_index=True, width="stretch")
fac = st.selectbox("Factor", list(dict(plan.sweeps)), key="lrn_factor")
d = sw[sw["factor"] == fac]
f = go.Figure()
for k, p in enumerate(["default", "band", "gate_gain", "gate_lcb", "bocpd", "dro"]):
    f.add_trace(go.Bar(x=d["level"], y=d[f"vs_always_{p}"], name=L.POLICY_LABELS[p], marker_color=CAT[k],
                       error_y=dict(type="data", symmetric=False, array=d[f"vs_always_{p}_hi"] - d[f"vs_always_{p}"],
                                    arrayminus=d[f"vs_always_{p}"] - d[f"vs_always_{p}_lo"], thickness=1)))
f.update_xaxes(title=fac); f.update_yaxes(title="Net payoff − always adapting")
st.plotly_chart(style(f, 360, barmode="group"))

if res.market is not None:
    st.subheader("Market replication")
    st.caption(f"Tuned on training configurations, same budget each: inaction-band threshold {res.market['theta']:g} "
               f"of {len(res.market['band_scores'])} candidates; confidence-sensitive gate (confidence, minimum "
               f"evidence) = {tuple(res.market['gate_hp'])} of {max(len(res.market['gate_scores']), 1)} candidates. "
               "Not applicable in the market: "
               + "; ".join(f"{k}: {v}" for k, v in res.market["not_applicable"].items()))
    st.dataframe(res.market["tests"].round(2), hide_index=True, width="stretch")

st.subheader("Secondary outcomes (jump-family test paths, means with 95% intervals)")
sec = L.secondary_table(res.paths, "test", plan.n_boot)
st.dataframe(sec[["label"] + [c for c in sec.columns if c in L.SECONDARY]].round(3), hide_index=True,
             width="stretch")
st.caption("Gross payoff before any charge, each adaptation charge separately, regret against perfect information, "
           "downside loss (CVaR 5% of per-period net payoff), the two behavioral rates kept apart (departure from "
           "the default and adjustment of the order, which are different events whenever the default moves), "
           "adjustment magnitude, calibration of the gates' predicted advantage, missed opportunities and recovery "
           "delay; intervals are in the download. Small differences in mean payoff or rank are not evidence that "
           "one theory is superior; read the intervals.")
with st.expander("The three adaptation charges, and why they are not interchangeable"):
    st.markdown(f"""
- **Default-departure overhead** (`cost`): charged every period the order differs from *that period's default*.
- **Fixed switching cost** (`switch_cost`): charged every period the order differs from the *previous period's order*.
- **Magnitude cost** (`magnitude_cost`): charged per unit of |S_t − S_(t−1)|.

These are separate economic frictions, not three settings of one. A policy that holds a constant non-default order
pays the overhead every period and nothing to switch; one that tracks a moving default pays no overhead but pays for
every change. The first period has no predecessor, so it carries no switching or magnitude charge. Two orders count
as different when they differ by more than {L.ACTION_TOL:g}. Setting all three to zero is the zero-cost control, and
`cost_model = "legacy_departure_only"` restores the single charge used before 9 October 2026.

The other tasks differ, and the difference is a property of the task rather than of the construct: in the **market**
the default is *keep last output*, so departing from it and changing the order are the same event and only the
magnitude charge is separately identified; the **NK** task has no default to depart from at all, so its `cost` is a
magnitude charge per changed component, alongside a distinct fixed `switch_cost`.
""")
download(res.tests, "learnability_tests.csv", "Download test configurations (CSV)")
download(res.new_families, "learnability_new_families.csv", "Download new-family configurations (CSV)")
download(res.sweeps, "learnability_sweeps.csv", "Download one-factor sweeps (CSV)")
download(sec, "learnability_secondary.csv", "Download secondary outcomes (CSV)")
download(res.paths, "learnability_paths.csv", "Download per-path outcomes (CSV)")
if res.market is not None:
    download(res.market["tests"], "learnability_market.csv", "Download market replication (CSV)")
