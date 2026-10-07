from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm import nk
from ui.common import CAT, download, evidence_note, set_engine, style

st.title("NK landscapes: interaction complexity, reliability learning and adaptation")
evidence_note("simulation")
set_engine("nk")
st.markdown(
    "An NK landscape (Kauffman & Levin 1987; Kauffman 1993; Levinthal 1997; Rivkin 2000) represents **complexity**: N "
    "binary decisions whose payoff contributions each depend on K_NK other decisions. Raising K_NK makes the landscape "
    "more rugged (more local peaks), so improving one decision at a time can strand a searcher. Complexity alone is not "
    "uncertainty: two **separate uncertainty mechanisms** are added and can be varied independently, **observation "
    "noise** (an evaluation can show an improvement that is not there) and **environmental change** (parts of the "
    "landscape are redrawn without announcement). K_NK is named to avoid confusion with the error-to-signal ratio K.")
st.info("The searchers are established ideas (hill climbing, simulated annealing, satisficing, imitation); the "
        "reliability gates are the laboratory's own **proposed extension**. The default settings (N = 8, "
        "K_NK = 0, 2, 4, 7) are pilot settings, not optimal design values.", icon="🧩")

with st.expander("Design and outcome measures", expanded=False):
    doc = nk.__doc__
    st.markdown("```\n" + doc[doc.index("Landscape (established model)"):doc.index("Uncertainty\n")] + "```")

st.subheader("Settings")
c1, c2, c3, c4 = st.columns(4)
N = c1.number_input("N (components)", 4, 20, 8, key="nk_N")
Ks = c2.multiselect("K_NK values", list(range(int(N))), [k for k in (0, 2, 4, 7) if k < N], key="nk_K")
topology = c3.selectbox("Interaction topology", nk.TOPOLOGIES, key="nk_top")
observe = c4.slider("Imitation: share of the leader observable", 0.0, 1.0, 0.5, 0.125, key="nk_obs")
c1, c2, c3, c4 = st.columns(4)
noise_txt = c1.text_input("Observation noise levels (s.d.)", "0, 0.02, 0.05", key="nk_noise")
hazard_txt = c2.text_input("Change hazards (per period)", "0, 0.005, 0.02", key="nk_hazard")
change_frac = c3.slider("Share of components redrawn at a change", 0.05, 1.0, 0.25, 0.05, key="nk_cf")
cost = c4.number_input("Switching cost per component", 0.0, 0.05, 0.002, 0.001, format="%.3f", key="nk_cost")
c1, c2, c3, c4 = st.columns(4)
budget = c1.number_input("Evaluation budget", 10, 2000, 200, 10, key="nk_budget")
periods = c2.number_input("Periods", 20, 3000, 300, 20, key="nk_T")
n_land = c3.number_input("Landscapes per condition", 2, 100, 6, key="nk_L")
n_starts = c4.number_input("Starting configurations per landscape", 1, 20, 2, key="nk_S")


def _levels(txt):
    try:
        return tuple(sorted({float(v) for v in txt.split(",") if v.strip()}))
    except ValueError:
        return ()


noise_levels, hazard_levels = _levels(noise_txt), _levels(hazard_txt)
if not Ks or not noise_levels or not hazard_levels:
    st.warning("Choose at least one K_NK value and enter numeric noise and hazard levels.")
    st.stop()
base_noise = st.select_slider("Noise level held fixed while hazard varies", noise_levels,
                              noise_levels[len(noise_levels) // 2], key="nk_bn")
base_hazard = st.select_slider("Hazard held fixed while noise varies", hazard_levels,
                               hazard_levels[len(hazard_levels) // 2], key="nk_bh")
plan = nk.NKPlan(N=int(N), K_values=tuple(sorted(Ks)), noise_values=noise_levels, hazard_values=hazard_levels,
                 base_noise=base_noise, base_hazard=base_hazard, topology=topology, change_frac=change_frac,
                 cost=float(cost), budget=int(budget), periods=int(periods), observe=observe, n_landscapes=int(n_land),
                 n_starts=int(n_starts), n_boot=500)
if N > nk.ENUM_MAX:
    st.caption(f"N > {nk.ENUM_MAX}: the landscape is not enumerated; regret is measured against the **best-known** "
               "payoff (multi-start full-information hill climbing and every configuration an agent held), not a "
               "proven optimum.")
st.caption(f"{len(nk.conditions(plan))} conditions × {plan.n_landscapes} landscapes × {plan.n_starts} starts × "
           f"{len(plan.searchers)} searchers.")


@st.cache_data(show_spinner=False, max_entries=3)
def cached_runs(plan_dict: dict) -> pd.DataFrame:
    p = dict(plan_dict)
    p["params"] = nk.SearchParams(**p["params"])
    return nk.run_nk(nk.NKPlan(**p))


if st.button("Run", type="primary", key="nk_run"):
    st.session_state["nk_key"] = asdict(plan)
if st.session_state.get("nk_key") != asdict(plan):
    st.info("Press **Run**.")
    st.stop()
with st.spinner("Searching landscapes…"):
    runs = cached_runs(asdict(plan))

st.header("Results", divider="gray")
st.caption("Means with 95% intervals from a cluster bootstrap over landscapes: runs on the same landscape (different "
           "starts, same tables and changes) are dependent. Benchmark: "
           + ", ".join(sorted(set(runs["benchmark"]))) + ".")
metric = st.selectbox("Outcome", list(nk.METRICS), format_func=nk.METRICS.get, key="nk_metric")
shown = st.multiselect("Searchers", list(nk.SEARCHERS), list(nk.SEARCHERS), format_func=nk.LABELS.get, key="nk_show")


def line_chart(df: pd.DataFrame, x: str, title: str, xlabel: str):
    s = nk.summarize(df[df["searcher"].isin(shown)], metric, [x, "searcher"], plan.n_boot)
    f = go.Figure()
    for k, sr in enumerate(nk.SEARCHERS):
        d = s[s["searcher"] == sr]
        if d.empty:
            continue
        f.add_trace(go.Scatter(x=d[x], y=d["mean"], mode="lines+markers", name=nk.LABELS[sr],
                               line=dict(color=CAT[k % len(CAT)]),
                               error_y=dict(type="data", symmetric=False, array=d["hi"] - d["mean"],
                                            arrayminus=d["mean"] - d["lo"], thickness=1)))
    f.update_xaxes(title=xlabel)
    f.update_yaxes(title=nk.METRICS[metric])
    st.plotly_chart(style(f, 380, title=title))
    return s


noise_runs = runs[runs["sweep"] == "noise"]
hazard_runs = runs[runs["sweep"] == "hazard"]
st.subheader("Performance versus interdependence")
line_chart(noise_runs[noise_runs["obs_noise"] == base_noise], "K_NK",
           f"Noise {base_noise:g}, hazard {base_hazard:g}", "K_NK (interacting components per contribution)")
k_sel = st.select_slider("K_NK for the noise and change panels", plan.K_values, key="nk_ksel")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Versus observation noise")
    line_chart(noise_runs[noise_runs["K_NK"] == k_sel], "obs_noise", f"K_NK = {k_sel}, hazard {base_hazard:g}",
               "Observation noise (s.d.)")
with c2:
    st.subheader("Versus environmental change")
    line_chart(hazard_runs[hazard_runs["K_NK"] == k_sel], "hazard", f"K_NK = {k_sel}, noise {base_noise:g}",
               "Change hazard per period")

st.subheader("Gates on identical proposals and budgets (paired differences)")
st.caption("Each difference compares two gated searchers on the same landscape, start, proposals, noise draws and "
           "evaluation budget; they diverge only after their first different decision.")
pairs = [("gate_gain", "gate_none"), ("gate_lcb", "gate_none"), ("gate_lcb", "gate_gain")]
tabs = []
for a, b in pairs:
    d = nk.paired(noise_runs, a, b, metric, ["K_NK", "obs_noise"], plan.n_boot)
    d.insert(0, "comparison", f"{nk.LABELS[a]} − {nk.LABELS[b]}")
    tabs.append(d)
gt = pd.concat(tabs, ignore_index=True)
st.dataframe(gt.round(4), hide_index=True, width="stretch")
st.caption("Small differences, or differences in average rank, are not evidence that one approach is generally "
           "superior: read the intervals, and note that these are pilot settings.")

table = nk.summarize(runs, metric, ["sweep", "K_NK", "obs_noise", "hazard", "searcher"], plan.n_boot)
download(table, "nk_summary.csv", "Download summary (CSV)")
download(gt, "nk_gate_differences.csv", "Download gate differences (CSV)")
download(runs, "nk_runs.csv", "Download per-run outcomes (CSV)")
