import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.gate_study import ORACLE_SEED_BASE, compare_gates, gate_scenario
from heiner_abm.gates import GATE_LABELS, GATES
from heiner_abm.information import AGENT_BY_KEY, adaptive_key
from ui.common import CAT, base_scenario, download, evidence_note, from_json, set_engine, style, to_json

st.title("Reliability gates under uncertainty")
evidence_note("simulation")
set_engine("market", [AGENT_BY_KEY[("market", adaptive_key(g))] for g in GATES])
st.warning(
    "**Proposed extension.** Heiner (1983) states when restricting flexibility pays (the reliability condition); he "
    "did not propose confidence bounds, minimum-evidence rules or exploration. The confidence-sensitive and "
    "exploration-enabled gates are this laboratory's operationalizations. Results test these implementations in this "
    "market, not Heiner's theory.", icon="🧩")
st.markdown(
    "Every Adaptive firm uses the same production rule, so it receives the same recommendation x\\* in every gate. "
    "The gates differ only in **whether to adopt it** instead of the default x_B (rule B: keep last output). The "
    "quantity each gate estimates is the **advantage**\n\n"
    "A = E[ payoff over the next W periods with x\\* − the same with x_B | information at the time of the decision ]"
    "\n\npooled by size of the recommended change (five bins) and kept apart from the **adjustment cost c** paid for "
    "each adaptation.")
st.dataframe(pd.DataFrame([
    {"Gate": GATE_LABELS["gain"], "Evidence": "Estimated counterfactual: holding x* vs x_B with rivals' observed output "
     "and its believed demand curve, released W periods after each decision",
     "Adopts when": "estimated advantage ≥ c (no evidence: adopts, the existing optimistic start)"},
    {"Gate": GATE_LABELS["lcb"], "Evidence": "The same feedback",
     "Adopts when": "effective evidence ≥ minimum and lower confidence bound > c; otherwise keeps x_B"},
    {"Gate": GATE_LABELS["explore"], "Evidence": "Only its own realized payoffs after randomized trials (no "
     "counterfactual feedback)", "Adopts when": "lower bound > c; keeps if upper bound ≤ c; when uncertain, a "
     "randomized trial with the exploration rate, else keeps x_B"},
    {"Gate": GATE_LABELS["oracle_table"], "Evidence": "ORACLE: true mean advantage per bin from independent runs "
     "(researcher only)", "Adopts when": "true bin advantage > c; does not learn"},
]), hide_index=True, width="stretch")
with st.expander("Uncertainty estimator and its assumptions"):
    st.markdown(
        "Feedback values are weighted by the memory λ (weight λ^age). The estimate is the weighted mean; its "
        "standard error uses Kish's effective sample size n_eff = (Σw)²/Σw² and the unbiased weighted variance; the "
        "lower bound is mean − t(level; n_eff − 1)·s.e. The exploration gate compares its two arms with a "
        "Welch–Satterthwaite interval. A bin without evidence has **no** estimate (it is never set to zero).\n\n"
        "The bound would have its nominal coverage for independent feedback with a common mean. **That is not "
        "claimed here:** consecutive judgement windows overlap (dependence), costs drift and demand shifts "
        "(nonstationarity), the exploration gate stops randomizing once it is confident (adaptive sampling), and "
        "estimated feedback uses a possibly misspecified demand curve. Coverage is therefore measured below, "
        "against the ORACLE benchmark, not assumed.\n\n"
        "**ORACLE benchmark.** The true mean advantage per firm and bin is estimated by the researcher's forked "
        f"counterfactuals in independent markets (seeds from {ORACLE_SEED_BASE:,} on; the evaluation seeds are "
        "below). Pass 1 lets every firm adopt every recommendation; pass 2 re-estimates under the benchmark gate "
        "that uses the pass-1 table. Bins never seen in those runs are treated as 'keep'. The table reaches only the "
        "benchmark gate. Because it decides per bin, it is a benchmark, **not** an upper bound: a gate can beat it.")

# ------------------------------------------------------------------------------------------------ settings
base = base_scenario()
with st.form("gates_form"):
    st.subheader("Settings")
    st.caption("Market, firms, information specification and demand shifts come from the sidebar; every firm is "
               "switched to the Adaptive rule with the gate under test.")
    c = st.columns(4)
    periods = c[0].number_input("Periods", 200, 5000, min(base.periods, 600), step=100)
    reps = c[1].number_input("Markets (seeds)", 2, 60, 8, help="Common random numbers: every gate runs on the same "
                                                              "seeds.")
    W = c[2].number_input("Judgement window W", 1, 50, 5, help="Periods over which the advantage of a decision is "
                                                                "judged; also the researcher's measurement horizon.")
    memory = c[3].slider("Memory λ", 0.8, 1.0, 0.95, 0.005, help="1 = no forgetting.")
    c = st.columns(4)
    cost = c[0].number_input("Adjustment cost c", 0.0, 5000.0, 20.0, step=5.0)
    conf = c[1].slider("Confidence level", 0.5, 0.99, 0.9, 0.01, help="One-sided; 0.5 = the point estimate.")
    nmin = c[2].number_input("Minimum evidence", 2.0, 100.0, 5.0, step=1.0)
    rate = c[3].slider("Exploration rate", 0.0, 1.0, 0.3, 0.05)
    c = st.columns(4)
    on_change = c[0].selectbox("Response to an observed change", ["forget", "reset"],
                               format_func=lambda x: {"forget": "Memory only", "reset": "Discard evidence"}[x])
    oracle_runs = c[1].number_input("ORACLE runs", 2, 40, 8)
    seed0 = c[2].number_input("First seed", 1, ORACLE_SEED_BASE - 100, int(base.seed))
    chosen = c[3].multiselect("Gates", list(GATES), default=list(GATES), format_func=lambda g: GATE_LABELS[g])
    run = st.form_submit_button("Run comparison", type="primary")

settings = dict(window=int(W), memory=float(memory), adjust_cost=float(cost), confidence=float(conf),
                min_evidence=float(nmin), explore_rate=float(rate), on_change=on_change)
scn = base.copy(periods=int(periods))
probe = gate_scenario(scn, "explore", int(seed0), **settings)
errs = [e for e in probe.validate() if e not in probe.info_errors()]
for e in errs:
    st.error(e, icon="⛔")
if errs or not chosen:
    st.stop()


@st.cache_data(show_spinner=False, max_entries=8)
def cached_study(js: str, gates: tuple, reps: int, seed0: int, oracle_runs: int, settings_js: str):
    return compare_gates(from_json(js), gates, reps=reps, seed0=seed0, oracle_runs=oracle_runs,
                         **json.loads(settings_js))


key = (to_json(scn), tuple(chosen), int(reps), int(seed0), int(oracle_runs), json.dumps(settings, sort_keys=True))
if run:
    st.session_state["gates_key"] = key
if st.session_state.get("gates_key") != key:
    st.info("Choose settings and press **Run comparison**.")
    st.stop()
with st.spinner("Running the ORACLE benchmark and every gate on the same markets…"):
    study = cached_study(*key)

for g, why in study.not_run.items():
    st.warning(f"**{GATE_LABELS[g]}: not run.** {why}", icon="⛔")
if study.summary.empty:
    st.stop()
if scn.info.feedback == "oracle":
    st.warning("The sidebar sets ORACLE feedback: the estimated-gain and confidence-sensitive gates learn from the "
               "researcher's counterfactual. Label these results as an oracle diagnostic.", icon="⚠️")

S = study.summary
lab = {g: GATE_LABELS[g].split(" (")[0] for g in GATES}
color = {g: CAT[k] for k, g in enumerate(GATES)}

st.header("Results", divider="gray")
st.caption(f"{len(study.settings['seeds'])} markets (seeds {study.settings['seeds'][0]}–{study.settings['seeds'][-1]}), "
           f"recorded periods after burn-in, Adaptive firms; means over markets with 95% intervals. Realized "
           f"advantages are the researcher's forked counterfactuals over H = {study.settings['horizon']} periods "
           "(the firm then holds its output, rivals follow their rules); agents never see them.")
show = pd.DataFrame({
    "Gate": [lab[g] for g in S["gate"]],
    "Net payoff / period": S["net_payoff"].round(1),
    "± (95%)": S["net_payoff_ci"].round(1),
    "vs ORACLE": S.get("net_payoff_vs_oracle", pd.Series(np.nan, index=S.index)).round(1),
    "Adoption rate": S["adoption_rate"].round(3),
    "Trial share": S["trial_share"].round(3),
    "False adaptations (realized)": S["false_adapt_realized"].round(3),
    "False adaptations (ORACLE)": S["false_adapt_oracle"].round(3),
    "Missed (realized)": S["missed_realized"].round(3),
    "Missed (ORACLE)": S["missed_oracle"].round(3),
    "Learning delay (periods)": S["learning_delay"].round(0),
    "Evidence n_eff": S["evidence"].round(1),
})
st.dataframe(show, hide_index=True, width="stretch")
st.caption("**False adaptations**: share of adoptions whose advantage did not exceed c, judged ex post by the "
           "realized counterfactual (includes bad luck) or ex ante by the ORACLE bin value. **Missed**: share of "
           "opportunities worth more than c that were not adopted. **Learning delay**: period at which the gate "
           "first agrees with the ORACLE decision on 80% of its last 50 opportunities. **vs ORACLE**: paired "
           "difference in net payoff (net payoff = market profit − c × adoptions).")

c1, c2 = st.columns(2)
f = go.Figure()
for r in S.itertuples():
    f.add_trace(go.Bar(x=[lab[r.gate]], y=[r.net_payoff], marker_color=color[r.gate], name=lab[r.gate],
                       error_y=dict(type="data", array=[r.net_payoff_ci])))
f.update_yaxes(title="Net payoff per period")
c1.plotly_chart(style(f, 320, "Net payoff"))
f = go.Figure()
for k, (col, name) in enumerate((("false_adapt_oracle", "False adaptations"), ("missed_oracle", "Missed opportunities"))):
    f.add_trace(go.Bar(x=[lab[g] for g in S["gate"]], y=S[col], name=name, marker_color=CAT[k + 4]))
f.update_yaxes(title="Share (judged by the ORACLE table)", range=[0, 1])
c2.plotly_chart(style(f, 320, "Selection errors", barmode="group"))

st.subheader("Calibration")
st.caption("Mean realized advantage by decile of the gate's predicted advantage (the dashed line is perfect "
           "calibration). Lower-bound coverage is the share of predictions whose lower bound lies at or below the "
           "ORACLE value of its bin, measured here; it is not a guarantee. With W > 1 the exploration gate predicts "
           "payoffs while it keeps applying its own rules, whereas the realized advantage holds output after the "
           "decision, so its calibration is not like-for-like (the two coincide for W = 1).")
c1, c2 = st.columns([3, 2])
f = go.Figure()
cal = study.calibration
for g, d in cal.groupby("gate", sort=False):
    f.add_trace(go.Scatter(x=d["pred"], y=d["realized"], mode="lines+markers", name=lab[g],
                           line=dict(color=color[g], width=2)))
if not cal.empty:
    lo, hi = float(min(cal["pred"].min(), cal["realized"].min())), float(max(cal["pred"].max(), cal["realized"].max()))
    f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect", line=dict(dash="dash", color="gray")))
f.update_xaxes(title="Predicted advantage"); f.update_yaxes(title="Realized advantage")
c1.plotly_chart(style(f, 360, "Predicted vs realized advantage"))
c2.dataframe(pd.DataFrame({"Gate": [lab[g] for g in S["gate"]], "Bias (pred − realized)": S["bias"].round(1),
                           "Calibration slope": S["calibration_slope"].round(2),
                           "Lower-bound coverage": S["coverage_lower"].round(3),
                           "Share with a prediction": S["share_predicted"].round(3)}), hide_index=True,
             width="stretch")

st.subheader("After regime changes")
if study.after_shift.empty:
    st.info("No demand shift with a complete window before and after it. Enable unannounced structural change in "
            "the sidebar (and enough periods) to evaluate performance after regime changes.")
else:
    f = go.Figure()
    for g, d in study.after_shift.groupby("gate", sort=False):
        f.add_trace(go.Scatter(x=d["k"], y=d["net_payoff_vs_pre"], mode="lines", name=lab[g],
                               line=dict(color=color[g], width=2)))
    f.add_vline(x=0, line=dict(color="gray", dash="dot"))
    f.update_xaxes(title="Periods relative to the shift"); f.update_yaxes(title="Net payoff vs pre-shift mean")
    st.plotly_chart(style(f, 340, f"Net payoff around shifts ({int(study.after_shift['n'].iloc[0])} shift windows "
                              "per gate)"))
    st.dataframe(pd.DataFrame({"Gate": [lab[g] for g in S["gate"]],
                               "Net payoff change after shift": S.get("post_shift_change").round(1),
                               "False adaptations after shift (realized)": S.get("post_shift_false_adapt").round(3)}),
                 hide_index=True, width="stretch")

with st.expander("ORACLE table (researcher only)"):
    edges = (0,) + tuple(base.adaptive.bin_edges)
    cols = [f"|Δq| ≥ {e:g}" for e in edges]
    st.dataframe(pd.DataFrame(study.oracle["table"], columns=cols, index=[f"Firm {i + 1}" for i in
                                                                          range(len(study.oracle['table']))]).round(1))
    st.caption(f"Opportunities per cell in the independent runs (seeds {study.oracle['seeds'][0]:,}–"
               f"{study.oracle['seeds'][-1]:,}); −inf marks bins never observed (the benchmark keeps there).")
    st.dataframe(pd.DataFrame(study.oracle["counts"], columns=cols,
                              index=[f"Firm {i + 1}" for i in range(len(study.oracle['counts']))]))

download(S, "reliability_gates_summary.csv", "Download summary (CSV)")
download(study.markets, "reliability_gates_markets.csv", "Download per-market results (CSV)")
