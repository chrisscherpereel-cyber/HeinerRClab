import numpy as np
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.experiments import PARAMS, slope_summary
from ui.common import CAT, base_scenario, behaviour_horizon, download, measurement, reps, run_sweep_ui, show_errors, slope_chart, style

st.title("Experiment designer")
st.caption("Sweep one or two parameters from the base scenario and plot any outcome. Every condition uses the same "
           "cost shocks per replication. Download the firm-level and market-level data for your own analysis.")

base = base_scenario()
if not show_errors(base):
    st.stop()
H, cont = measurement()
keys = list(PARAMS)

c = st.columns(4)
p1 = c[0].selectbox("Parameter 1 (x-axis)", keys, index=keys.index("delta"), format_func=lambda k: PARAMS[k].label,
                    help="Main parameter to sweep; it forms the chart's horizontal axis.")
P1 = PARAMS[p1]
r1 = c[1].slider("Range", float(P1.lo), float(P1.hi), (float(P1.lo), float(P1.hi)), key=f"d_r1_{p1}",
                  help=f"Lowest and highest {P1.label} to sweep. {P1.help}".strip())
n1 = c[2].slider("Steps", 2, 25, 8, key="d_n1", help="Number of evenly spaced values of parameter 1.")
use2 = c[3].toggle("Add parameter 2 (lines)",
                   help="Cross parameter 1 with a second parameter; each value of parameter 2 becomes one line.")
p2 = vals2 = None
if use2:
    c = st.columns(4)
    p2 = c[0].selectbox("Parameter 2", [k for k in keys if k != p1], format_func=lambda k: PARAMS[k].label,
                        help="Second parameter to sweep; one line per value.")
    P2 = PARAMS[p2]
    r2 = c[1].slider("Range 2", float(P2.lo), float(P2.hi), (float(P2.lo), float(P2.hi)), key=f"d_r2_{p2}",
                      help=f"Lowest and highest {P2.label} to sweep. {P2.help}".strip())
    n2 = c[2].slider("Steps 2", 2, 6, 3, key="d_n2", help="Number of evenly spaced values of parameter 2 (lines).")
    vals2 = list(np.round(np.linspace(*r2, n2), 3))
vals1 = list(np.round(np.linspace(*r1, n1), 3))
if "n_firms" in (p1, p2):
    vals1 = sorted(set(int(round(x)) for x in vals1)) if p1 == "n_firms" else vals1
    vals2 = sorted(set(int(round(x)) for x in vals2)) if p2 == "n_firms" else vals2

OUTCOMES = {
    "slope": "Payoff to flexibility: profit slope on φ (95% CI)",
    "avg_firm_profit": "Average firm profit", "avg_margin": "Average margin P − c",
    "spearman": "Rank correlation φ ~ profit", "flex_best": "Share: most flexible firm best",
    "rigid_best": "Share: most rigid firm best", "pi": "π (preferred exceptions)", "r": "r", "w": "w",
    "G": "G", "D": "D", "ratio": "Median reliability ratio r/w", "tolerance": "Median tolerance limit",
    "log_rc_margin": "Median RC margin ln(ratio ÷ tolerance)", "rc_share": "Share of firms satisfying the RC",
    "cd_gap": "Measured CD-gap", "sc_cost": "Serial corr. of cost", "sc_price": "Serial corr. of price",
}
outcome = st.selectbox("Outcome", list(OUTCOMES), format_func=OUTCOMES.get,
                       help="Statistic plotted on the vertical axis, averaged over replications. Reliability outcomes "
                            "(π, r, w, G, D, ratio, tolerance, RC) need the counterfactual horizon H and run slower.")
RC_OUTCOMES = {"pi", "r", "w", "G", "D", "ratio", "tolerance", "log_rc_margin", "rc_share"}
if outcome not in RC_OUTCOMES:
    H = behaviour_horizon(base)      # profit outcomes don't need counterfactual forks
n_runs = len(vals1) * (len(vals2) if vals2 else 1) * reps()
st.caption(f"{n_runs:,} markets × {base.periods:,} periods, H = {H}.")
key = (p1, tuple(vals1), p2, tuple(vals2) if vals2 else None, reps(), H, cont, str(base))
if st.button("Run sweep", type="primary"):
    st.session_state["des_key"] = key
if st.session_state.get("des_key") != key:
    st.stop()

firms, mk = run_sweep_ui(base, p1, vals1, reps(), p2, vals2, horizon=H, continuation=cont)
summ = slope_summary(firms, mk, [p1] + ([p2] if p2 else []))
if outcome == "slope":
    fig = slope_chart(summ, p1, P1.label, group=p2, group_title=PARAMS[p2].label if p2 else None)
else:
    fig = go.Figure()
    groups = [(None, summ)] if not p2 else list(summ.groupby(p2))
    for k, (gv, g) in enumerate(groups):
        g = g.sort_values(p1)
        fig.add_trace(go.Scatter(x=g[p1], y=g[outcome], mode="lines+markers", line=dict(color=CAT[k % len(CAT)], width=2),
                                 name=OUTCOMES[outcome] if gv is None else f"{PARAMS[p2].label} = {gv:g}"))
    fig.update_xaxes(title=P1.label); fig.update_yaxes(title=OUTCOMES[outcome])
    fig = style(fig)
st.plotly_chart(fig)
st.dataframe(summ, hide_index=True, width="stretch")
c1, c2, c3 = st.columns(3)
with c1:
    download(summ, "sweep_summary.csv", "Download condition summary")
with c2:
    download(firms, "sweep_firms.csv", "Download firm-level data")
with c3:
    download(mk, "sweep_markets.csv", "Download market-level data")
