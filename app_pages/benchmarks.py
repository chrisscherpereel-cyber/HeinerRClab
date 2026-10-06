import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm import bench_bandit as BB
from heiner_abm import bench_inventory as BI
from heiner_abm.bench_checks import run_checks
from heiner_abm.literature import REFERENCES, cite
from ui.common import CAT, download, evidence_note, set_engine, style

st.title("Decision benchmarks")
evidence_note("analytical", "simulation")
st.caption("Three established decision methods, run in two tasks of the Generalization page under a common protocol: "
           "**(A)** Bayesian learning with change detection and **(B)** distributionally robust optimization in the "
           "inventory (newsvendor) task, and **(C)** nonstationary bandit learning in a chosen-action-feedback version "
           "of the learning task. Hyperparameters are tuned on training environments, the tuned configuration is "
           "chosen on validation environments, and results are reported on separate test environments. Correctly "
           "specified models, oracles and the full-feedback reference are shown beside the rankings, never in them.")


@st.cache_data(show_spinner=False)
def cached_checks():
    return run_checks()


st.header("1 · Correctness checks", divider="gray")
st.caption("Each method is first checked on cases with a known answer: analytic solutions or exhaustive enumeration. "
           "The comparisons below are disabled unless every check passes.")
checks = cached_checks()
st.dataframe(checks.assign(passed=checks["passed"].map({True: "✅", False: "❌"})), hide_index=True, width="stretch",
             column_config={"error": st.column_config.NumberColumn(format="%.2e"),
                            "tolerance": st.column_config.NumberColumn(format="%.0e")})
if not checks["passed"].all():
    st.error("A correctness check failed; the comparisons are disabled.", icon="⛔")
    st.stop()


def families_table(fams) -> pd.DataFrame:
    return pd.DataFrame([{"Agent": f.name, "Role": f.role, "Tuned": ", ".join(f.space) or "—",
                          "Observes": f.feedback, "Assumptions": f.assumptions, "Compute": f.compute,
                          "Sources": cite(*f.sources) if f.sources else "—"} for f in fams.values()])


def show_results(out, fams, prefix: str, engine: str, label: str):
    rk = out["ranking"]
    comp = rk[rk["role"] == "competitor"]
    other = rk[rk["role"] != "competitor"]
    cols = {"rank": "Rank", "agent": "Agent", "mean_payoff": "Test payoff / period", "ci": "± (95%)",
            "vs_baseline": f"vs {label}", "vs_baseline_ci": "± ", "config": "Tuned configuration",
            "runs": "Environment runs"}
    st.markdown("**Ranking on the test environments** (competitors only)")
    st.dataframe(comp[list(cols)].rename(columns=cols).round(3), hide_index=True, width="stretch")
    st.markdown("**Beside the ranking:** correctly specified benchmark, oracle and references (different "
                "information; not ranked)")
    st.dataframe(other[["agent", "role", "mean_payoff", "ci", "vs_baseline", "config"]].rename(columns={
        "agent": "Agent", "role": "Role", "mean_payoff": "Test payoff / period", "ci": "± (95%)",
        "vs_baseline": f"vs {label}", "config": "Configuration"}).round(3), hide_index=True, width="stretch")
    cur = out["curve"]
    f = go.Figure()
    for k, (fam, d) in enumerate(cur.groupby("family", sort=False)):
        if fams[fam].space:
            f.add_trace(go.Scatter(x=d["budget"], y=d["validation"], mode="lines+markers", name=fams[fam].name,
                                   line=dict(color=CAT[k % len(CAT)], width=2),
                                   customdata=d[["train_runs"]].to_numpy(),
                                   hovertemplate="budget %{x} configurations (%{customdata[0]} training runs)<br>"
                                                 "validation payoff %{y:.3f}<extra></extra>"))
    f.update_xaxes(title="Tuning budget (configurations tried)", type="log", dtick=np.log10(2))
    f.update_yaxes(title="Validation payoff of the chosen configuration")
    st.plotly_chart(style(f, 340, "Tuning performance as a function of evaluation budget"))
    st.caption("Random search: at budget b the configuration with the best training payoff among the first b sampled "
               "is scored on the validation environments (mean over repetitions). Training runs = b × training "
               "environments.")
    tests = pd.concat([r.test for r in out["results"].values()], ignore_index=True)
    set_engine(engine)
    download(rk, f"{prefix}_ranking.csv", "Download ranking (CSV)")
    download(cur, f"{prefix}_tuning_curve.csv", "Download tuning curve (CSV)")
    download(tests, f"{prefix}_test_environments.csv", "Download per-environment test results (CSV)")
    st.download_button("Plan and environments (JSON)", json.dumps({"plan": out["plan"],
                                                                  "environments": out["environments"]},
                                                                 indent=1, default=float).encode(),
                       file_name=f"{prefix}_plan.json", mime="application/json", key=f"{prefix}_plan")


@st.cache_data(show_spinner=False, max_entries=4)
def cached_inventory(plan_js: str):
    d = json.loads(plan_js)
    return BI.run_inventory(BI.InventoryPlan(**{**d, "budgets": tuple(d["budgets"]), "families": tuple(d["families"])}))


@st.cache_data(show_spinner=False, max_entries=4)
def cached_bandit(plan_js: str):
    d = json.loads(plan_js)
    return BB.run_bandit(BB.BanditPlan(**{**d, "budgets": tuple(d["budgets"]), "families": tuple(d["families"])}))


st.header("2 · Comparisons", divider="gray")
tab_inv, tab_bandit, tab_refs = st.tabs(["A + B · Inventory: Bayes and robust optimization",
                                         "C · Nonstationary bandit", "Sources"])

with tab_inv:
    set_engine("bench_inventory")
    st.markdown(
        "**Task.** Mean demand starts at 100 and, with the environment's hazard, jumps by N(0, 30²), clipped to "
        "[30, 170]; demand is max(0, mean + σ·noise) and is observed with further noise τ. Leftover stock costs 1 and "
        "shortage 4 per unit, so the best order is the 80% quantile of demand. Orders use observations up to the "
        "previous period. **(A)** The correctly specified Bayesian filter knows this process and its parameters "
        "(benchmark); the change-point detector assumes a simpler, misspecified model and is tuned. **(B)** The "
        "robust order guards against every distribution within modified-χ² distance ρ of the recent empirical "
        "distribution; ρ = 0 is the empirical (SAA) order, which is a separate competitor.")
    st.dataframe(families_table(BI.FAMILIES), hide_index=True, width="stretch")
    scale = st.radio("Scale", ["Quick check", "Full"], horizontal=True, key="bench_inv_scale",
                     help="Quick: 400 periods, 3/3/5 training/validation/test environments, budgets up to 4 (about "
                          "10 seconds). Full: 1,000 periods, 10/10/20 environments, budgets up to 16 (several "
                          "minutes).")
    plan = BI.InventoryPlan(**(BI.QUICK_INVENTORY if scale == "Quick check" else {}))
    if st.button("Run inventory comparison", type="primary", key="bench_inv_run"):
        st.session_state["bench_inv_key"] = plan
    if st.session_state.get("bench_inv_key") == plan:
        with st.spinner("Tuning, selecting and testing every agent…"):
            out = cached_inventory(json.dumps(plan.to_dict()))
        show_results(out, BI.FAMILIES, "inventory", "bench_inventory", "default rule")

with tab_bandit:
    set_engine("bench_bandit")
    st.markdown(
        "**Task.** Two options; one pays 1 more on average and which one is better swaps without warning (the "
        "environment's hazard). In the Generalization page's version the agent sees both payoffs every period. Here "
        "it sees **only the payoff of the option it chose**, with observation error. Every bandit agent gets exactly "
        "that feedback and one choice per period. The full-feedback learner of the original task is shown beside the "
        "ranking to measure what full feedback is worth; it is not a bandit.")
    st.dataframe(families_table(BB.FAMILIES), hide_index=True, width="stretch")
    scale = st.radio("Scale", ["Quick check", "Full"], horizontal=True, key="bench_bandit_scale",
                     help="Quick: 400 periods, 3/3/5 environments, budgets up to 4. Full: 1,000 periods, 10/10/20 "
                          "environments, budgets up to 32.")
    plan_b = BB.BanditPlan(**(BB.QUICK_BANDIT if scale == "Quick check" else {}))
    if st.button("Run bandit comparison", type="primary", key="bench_bandit_run"):
        st.session_state["bench_bandit_key"] = plan_b
    if st.session_state.get("bench_bandit_key") == plan_b:
        with st.spinner("Tuning, selecting and testing every agent…"):
            out_b = cached_bandit(json.dumps(plan_b.to_dict()))
        show_results(out_b, BB.FAMILIES, "bandit", "bench_bandit", "UCB")

with tab_refs:
    keys = ("arrow1951b", "adams2007", "fearnhead2007", "bental2013", "garivier2011", "auer2002", "sutton2018",
            "behrens2007", "muth1960")
    st.markdown("\n".join(f"* {REFERENCES[k].apa}" for k in keys))
    st.caption("Adams & MacKay (2007) is a preprint; Fearnhead & Liu (2007) give the same exact recursion in a journal.")
