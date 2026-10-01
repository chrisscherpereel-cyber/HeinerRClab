"""Shared Streamlit pieces: base-scenario sidebar, presets, cached runners, chart helpers."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.analysis import market_table, reliability_table
from heiner_abm.engine import run_batch
from heiner_abm.experiments import (EnvRanges, adjustment_bound_experiment, encompassing_test, evolution_runs,
                                   horse_race, random_environments, rc_validation, regime_event_study, run_sweep,
                                   uncertainty_comparison)
from heiner_abm.params import (AdaptiveParams, EvolutionParams, FirmSpec, GlobalFirmParams, MarketParams,
                               SELECTION_HELP, SELECTION_RULES, Scenario, StructuralParams, linear_flex_firms)
from heiner_abm.literature import Hypothesis
from heiner_abm.theories import run_tournament

# ------------------------------------------------------------------------------------------------
# Colours (reference palette): firms are ordered by flexibility -> one-hue sequential ramp
# ------------------------------------------------------------------------------------------------
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
             "#104281", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PRICE_C, COST_C = "#2a78d6", "#eb6834"
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]   # red (bad) - gray - blue (good)
SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]


def firm_colors(n: int) -> List[str]:
    """Rigid (light) -> flexible (dark) along the blue ramp."""
    if n == 1:
        return [BLUE_RAMP[5]]
    idx = np.linspace(0, len(BLUE_RAMP) - 1, n).round().astype(int)
    return [BLUE_RAMP[i] for i in idx]


def style(fig: go.Figure, height: int = 380, title: Optional[str] = None, **kw) -> go.Figure:
    # legend below the plot area so multi-row legends never collide with the title or the toolbar
    fig.update_layout(height=height + 40, margin=dict(l=10, r=10, t=50 if title else 30, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0), **kw)
    if title:
        fig.update_layout(title=dict(text=title, y=0.98, yanchor="top"))
    if fig.layout.hovermode is None:
        fig.update_layout(hovermode="closest")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.15)",
                     zeroline=True, zerolinecolor="rgba(128,128,128,0.45)")
    return fig


# ------------------------------------------------------------------------------------------------
# Base scenario configuration (sidebar)
# ------------------------------------------------------------------------------------------------
DEFAULTS: Dict[str, object] = dict(
    p_max=100.0, p_min=10.0, q_range=1500.0, c0=45.0, c_max=80.0, delta=10.0,
    n_firms=4, rule="Bertrand", flex_intercept=0.0, flex_slope=0.25, selection="Always", threshold=25.0,
    desired_margin=5.0, foresight=0.0, noise=0.0, q0=200.0, q_min=15.0,
    flex_cost_slope=0.0, fixed_cost=0.0, margin_includes_fixed=False,
    periods=1000, burn_in=25, reps=20, seed=1, horizon=20, continuation="default", memory=0.97,
    discount=1.0, oos_split=0.5,
    struct_on=False, hazard=0.02, intercept_sd=15.0, slope_sd=0.4, belief_lag=20,
)

PRESETS: Dict[str, Dict[str, object]] = {
    "Baseline (Bertrand, 4 firms)": {},
    "Cournot competition": dict(rule="Cournot"),
    "Low-profit Bertrand (fierce competition, m* = 0.5)": dict(desired_margin=0.5),
    "High-profit Bertrand (gentlemanly, m* = 14)": dict(desired_margin=14.0),
    "Volatile raw materials (Δ = 25)": dict(delta=25.0),
    "Selection rules: SR2 'Large' (θ = 25)": dict(selection="Large", threshold=25.0),
    "Selection rules: SR1 'Small' (θ = 10)": dict(selection="Small", threshold=10.0),
    "Reliability-learning (Adaptive) agents, H = 25": dict(selection="Adaptive", horizon=25),
    "Costly flexibility (a = 500, b = 100, in margin)": dict(flex_cost_slope=500.0, fixed_cost=100.0,
                                                             margin_includes_fixed=True),
    "Ten firms (larger market)": dict(n_firms=10, flex_slope=0.1, q0=80.0),
    "Knightian uncertainty: demand regime shifts (Cournot, φ ≤ 0.4)": dict(rule="Cournot", flex_slope=0.1,
                                                                              struct_on=True, delta=5.0),
    "Knightian uncertainty: demand regime shifts (Bertrand)": dict(struct_on=True, delta=5.0),
}

K = "cfg_"


def _init_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(K + k, v)


def apply_preset():
    name = st.session_state.get("preset_choice")
    values = dict(DEFAULTS)
    values.update(PRESETS.get(name, {}))
    for k, v in values.items():
        st.session_state[K + k] = v


def reset_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[K + k] = v


def v(key):
    return st.session_state.get(K + key, DEFAULTS[key])


def render_sidebar():
    _init_state()
    sb = st.sidebar
    sb.markdown("### Base scenario")
    sb.caption("Every page starts from this market. Experiments override only the parameters they sweep.")
    sb.selectbox("Preset", list(PRESETS), key="preset_choice",
                 help="Ready-made scenarios. Pick one, then press **Apply preset** to load its values into every sidebar "
                      "control; **Reset** restores the baseline calibration.")
    c1, c2 = sb.columns(2)
    c1.button("Apply preset", on_click=apply_preset, width="stretch")
    c2.button("Reset", on_click=reset_defaults, width="stretch")

    with sb.expander("Market", expanded=False):
        st.number_input("Max price (demand intercept)", 20.0, 1000.0, key=K + "p_max", step=5.0,
                        help="Price at zero quantity on the linear demand curve P = max price − slope·Q. "
                             "Never reached in practice because every firm produces at least the min production.")
        st.number_input("Min price (floor & lower cost bound)", 0.0, 500.0, key=K + "p_min", step=1.0,
                        help="Price floor of the hockey-stick demand curve, P = max(min price, …). Also the lower "
                             "reflecting bound for the raw-material cost.")
        st.number_input("Quantity range (sets demand slope)", 100.0, 20000.0, key=K + "q_range", step=100.0,
                        help="Slope = (max price − min price) / quantity range. Steep demand destabilises the cobweb.")
        st.number_input("Initial raw-material cost c₀", 0.0, 500.0, key=K + "c0", step=1.0,
                        help="Raw-material cost in period 0; the cost random walk starts here. Must lie between "
                             "min price and max raw-material cost.")
        st.number_input("Max raw-material cost", 1.0, 500.0, key=K + "c_max", step=1.0,
                        help="Upper reflecting bound; lowering it raises industry profitability.")
        st.slider("Cost volatility Δ (max change per period)", 0.0, 60.0, key=K + "delta", step=0.5,
                  help="Difficulty of the environment. c[t] = c[t−1] + Δ·U(−1,1), reflected at the bounds.")

    with sb.expander("Firms", expanded=True):
        st.slider("Number of firms", 2, 12, key=K + "n_firms",
                  help="Firms in the market. Firm i gets flexibility φᵢ = intercept + slope·i, so firms are "
                       "ordered from most rigid (firm 1) to most flexible.")
        st.radio("Production rule", ["Bertrand", "Cournot"], key=K + "rule", horizontal=True,
                 help="How a firm computes its recommended output q*. **Cournot** (model-based): move a fraction φ "
                      "of the way toward the best reply on the believed demand curve (φ ≤ 1). **Bertrand** "
                      "(model-free): q* = q + φ·(observed margin − m*), using only the last price.")
        c1, c2 = st.columns(2)
        c1.number_input("Flex intercept", 0.0, 5.0, key=K + "flex_intercept", step=0.05,
                         help="Constant part of each firm's flexibility: φᵢ = intercept + slope·i.")
        c2.number_input("Flex slope", 0.0, 5.0, key=K + "flex_slope", step=0.05,
                         help="Increase in flexibility from one firm to the next: φᵢ = intercept + slope·i. "
                              "0 makes all firms equally flexible.")
        n = int(v("n_firms"))
        flexes = [v("flex_slope") * i + v("flex_intercept") for i in range(1, n + 1)]
        st.caption("φᵢ = intercept + slope·i → " + ", ".join(f"{x:.2f}" for x in flexes))
        if v("rule") == "Cournot" and max(flexes) > 1:
            st.warning("Cournot flexibility must be ≤ 1; values will be capped at 1.")
        st.selectbox("Selection rule (deviate from 'keep q'?)", SELECTION_RULES, key=K + "selection",
                     help="\n\n".join(f"**{k}** – {t}" for k, t in SELECTION_HELP.items()))
        if v("selection") in ("Small", "Large"):
            st.number_input("Threshold θ", 0.0, 500.0, key=K + "threshold", step=1.0,
                            help="Cut-off on the recommended change |q* − q|. Small (SR1) deviates only below θ; "
                                 "Large (SR2) deviates only above θ.")
        if v("rule") == "Bertrand":
            st.slider("Desired margin m*", -5.0, 30.0, key=K + "desired_margin", step=0.5,
                      help="Intensity of competition. Small = fierce, large = gentlemanly.")
        st.slider("Competence κ (cost foresight)", 0.0, 1.0, key=K + "foresight", step=0.05,
                  help="Share of the coming cost change the firm anticipates. Baseline firms have κ = 0.")
        st.slider("Perception noise σ", 0.0, 20.0, key=K + "noise", step=0.5,
                  help="Standard deviation of random error in each firm's estimate of the coming cost. "
                       "Widens the CD-gap (lower competence).")
        c1, c2 = st.columns(2)
        c1.number_input("Initial production", 1.0, 5000.0, key=K + "q0", step=10.0,
                         help="Output q of every firm in period 0.")
        c2.number_input("Min production", 1.0, 500.0, key=K + "q_min", step=1.0,
                         help="Lowest output a firm can choose; recommendations below it are raised to it.")

    with sb.expander("Cost of flexibility", expanded=False):
        st.number_input("Flexibility cost slope a", 0.0, 10000.0, key=K + "flex_cost_slope", step=50.0,
                        help="Per-period cost Fᵢ = a·φᵢ + b. The decisive test of free flexibility sets a = b = 0.")
        st.number_input("Fixed cost per period b", 0.0, 10000.0, key=K + "fixed_cost", step=50.0,
                        help="Intercept of the per-period cost Fᵢ = a·φᵢ + b, paid by every firm regardless of "
                             "output. Only affects decisions if the margin includes F/q.")
        st.checkbox("Bertrand margin includes F/q", key=K + "margin_includes_fixed",
                    help="Off = baseline (margin = P − c). On = margin = P − c − F/q, so fixed costs "
                         "change decisions, not just profit levels.")

    with sb.expander("Simulation & measurement", expanded=False):
        st.number_input("Periods per run", 100, 20000, key=K + "periods", step=100,
                        help="Length of each simulated market. Longer runs give more precise estimates but take longer.")
        st.number_input("Burn-in (discarded periods)", 0, 2000, key=K + "burn_in", step=5,
                        help="Initial periods excluded from all statistics so the market can settle away from its "
                             "starting values. The baseline discards 25 periods.")
        st.number_input("Replications per condition", 1, 200, key=K + "reps", step=1,
                        help="Independent markets (different cost shocks) simulated for each experimental condition. "
                             "More replications give narrower confidence intervals.")
        st.number_input("Random seed", 0, 10_000_000, key=K + "seed", step=1,
                        help="Starting point of the random-number streams. The same seed reproduces the same cost "
                             "shocks and results exactly.")
        st.slider("Counterfactual horizon H", 1, 100, key=K + "horizon",
                  help="Periods over which a deviation from rule B is evaluated when measuring π, r, w, G, D. "
                       "H = 1 is Heiner's one-shot comparison, which ignores that production changes persist; "
                       "H ≈ 20+ captures their consequences. Cost grows with H. Profit-only experiments skip it "
                       "unless Adaptive agents (who learn from it) are present.")
        st.radio("After the decision, in the counterfactual the firm…",
                 ["default", "rules"], key=K + "continuation",
                 format_func=lambda x: {"default": "returns to rule B (Heiner)",
                                        "rules": "keeps using its own rules"}[x],
                 help="How the firm behaves in the counterfactual branch after the evaluated decision, for H > 1. "
                      "**Rule B**: it then keeps its output fixed, isolating the value of this one decision "
                      "(Heiner). **Own rules**: it continues with its production and selection rules.")
        st.slider("Discount γ within the horizon", 0.80, 1.0, key=K + "discount", step=0.01,
                  help="Weight γʰ on the h-th period after a decision when valuing it (dynamic RC).")
        st.slider("Estimation window (share of recorded periods)", 0.2, 0.8, key=K + "oos_split", step=0.05,
                  help="Out-of-sample tests estimate the RC in this first part of each run and predict "
                       "performance in the remaining part.")
        st.slider("Adaptive agents' memory λ", 0.5, 0.999, key=K + "memory", step=0.005,
                  help="Forgetting factor of Adaptive firms' learned payoff to deviating: new estimate = "
                       "λ·old + (1 − λ)·latest gain. Closer to 1 = longer memory, slower learning.")

    with sb.expander("Structural uncertainty (regime shifts)", expanded=bool(v("struct_on"))):
        st.toggle("Unannounced demand-regime shifts", key=K + "struct_on",
                  help="Knightian uncertainty: the demand curve occasionally jumps to a new regime that "
                       "model-based (Cournot) firms don't know about until they update their model.")
        if v("struct_on"):
            st.slider("Hazard λ (shift probability per period)", 0.0, 0.2, key=K + "hazard", step=0.005,
                      help="Probability that the demand curve jumps to a new regime in any period. "
                           "Expected regime length is 1/λ periods.")
            st.slider("Shift size: demand intercept s.d.", 0.0, 40.0, key=K + "intercept_sd", step=1.0,
                      help="At a shift, max price = baseline + s.d.·z with z ~ N(0,1) (kept at least 10 above "
                           "min price).")
            st.slider("Shift size: log demand-slope s.d.", 0.0, 1.0, key=K + "slope_sd", step=0.05,
                      help="At a shift, slope = baseline slope × e^(s.d.·z), z ~ N(0,1), clipped to 0.25–4× baseline.")
            st.number_input("Model-updating lag L (periods, −1 = never)", -1, 500, key=K + "belief_lag", step=1,
                            help="How many periods model-based (Cournot) firms keep using the old demand curve "
                                 "after a shift. 0 = they learn it one period later; −1 = they never update.")
    sb.markdown("---")
    sb.caption("Grounded in Heiner (1983, 1989) and rival theories of flexibility under uncertainty. "
               "See *Research & contribution*.")


def base_scenario() -> Scenario:
    n = int(v("n_firms"))
    rule = v("rule")
    firms = linear_flex_firms(n=n, slope=float(v("flex_slope")), intercept=float(v("flex_intercept")), rule=rule,
                              selection=v("selection"), threshold=float(v("threshold")),
                              desired_margin=float(v("desired_margin")), foresight=float(v("foresight")),
                              noise=float(v("noise")))
    if rule == "Cournot":
        for f in firms:
            f.flex = min(f.flex, 1.0)
    return Scenario(
        market=MarketParams(p_max=float(v("p_max")), p_min=float(v("p_min")), q_range=float(v("q_range")),
                            c0=float(v("c0")), c_max=float(v("c_max")), delta=float(v("delta"))),
        firms=firms,
        firm_globals=GlobalFirmParams(q0=float(v("q0")), q_min=float(v("q_min")),
                                      flex_cost_slope=float(v("flex_cost_slope")),
                                      fixed_cost=float(v("fixed_cost")),
                                      margin_includes_fixed=bool(v("margin_includes_fixed"))),
        adaptive=AdaptiveParams(memory=float(v("memory"))),
        structural=StructuralParams(enabled=bool(v("struct_on")), hazard=float(v("hazard")),
                                    intercept_sd=float(v("intercept_sd")), slope_sd=float(v("slope_sd")),
                                    belief_lag=int(v("belief_lag"))),
        periods=int(v("periods")), burn_in=int(v("burn_in")), seed=int(v("seed")))


def measurement():
    return int(v("horizon")), str(v("continuation"))


def measure_opts():
    """Discount and estimation-window split used by the dynamic and out-of-sample RC measures."""
    return float(v("discount")), float(v("oos_split"))


def behaviour_horizon(scn: Scenario) -> int:
    """Horizon needed when only profits are reported: H changes behaviour only through Adaptive agents'
    learning, so skip the (costly) counterfactual forks otherwise."""
    return int(v("horizon")) if any(f.selection == "Adaptive" for f in scn.firms) else 1


def reps() -> int:
    return int(v("reps"))


def show_errors(scn: Scenario) -> bool:
    errs = scn.validate()
    for e in errs:
        st.error(e, icon="⛔")
    return not errs


# ------------------------------------------------------------------------------------------------
# (De)serialisation for caching
# ------------------------------------------------------------------------------------------------
def to_json(scn: Scenario) -> str:
    return json.dumps(asdict(scn), sort_keys=True)


def from_json(js: str) -> Scenario:
    d = json.loads(js)
    return Scenario(market=MarketParams(**d["market"]), firms=[FirmSpec(**f) for f in d["firms"]],
                    firm_globals=GlobalFirmParams(**d["firm_globals"]),
                    adaptive=AdaptiveParams(**{**d["adaptive"], "bin_edges": tuple(d["adaptive"]["bin_edges"])}),
                    evolution=EvolutionParams(**d["evolution"]),
                    structural=StructuralParams(**d.get("structural", {})),
                    periods=d["periods"], burn_in=d["burn_in"], seed=d["seed"])


@st.cache_data(show_spinner=False, max_entries=64)
def cached_single(js: str, horizon: int, continuation: str, discount: float = 1.0, oos_split: float = 0.5):
    scn = from_json(js)
    res = run_batch([scn], record_firm_history=True, horizon=horizon, continuation=continuation,
                    discount=discount, oos_split=oos_split)
    tables = {(m, w): reliability_table(res, m, w) for m in ("static", "persist", "full")
              for w in ("all", "est", "eval")}
    return dict(firms=reliability_table(res), tables=tables, market=market_table(res), price=res.price[0],
                cost=res.cost[0], quantity=res.quantity[0], hist={k: v[0] for k, v in res.firm_hist.items()},
                flex_path=res.flex_path[0], shifts=res.shifts[0], p_max_path=res.p_max_path[0],
                slope_path=res.slope_path[0], split_t=res.meta["split_t"])


@st.cache_data(show_spinner=False, max_entries=64)
def cached_sweep(js: str, p1: str, values1: tuple, n_reps: int, p2: Optional[str], values2: Optional[tuple],
                 horizon: int, continuation: str, discount: float = 1.0, oos_split: float = 0.5):
    return run_sweep(from_json(js), p1, list(values1), n_reps, p2, list(values2) if values2 else None,
                     horizon=horizon, continuation=continuation, discount=discount, oos_split=oos_split)


def run_sweep_ui(scn: Scenario, p1, values1, n_reps, p2=None, values2=None, horizon=1, continuation="default",
                 discount=1.0, oos_split=0.5):
    n_runs = len(values1) * (len(values2) if values2 else 1) * n_reps
    with st.spinner(f"Simulating {n_runs:,} markets × {scn.periods:,} periods (H = {horizon})…"):
        return cached_sweep(to_json(scn), p1, tuple(float(x) for x in values1), int(n_reps), p2,
                            tuple(float(x) for x in values2) if values2 else None, int(horizon), continuation,
                            float(discount), float(oos_split))


@st.cache_data(show_spinner=False, max_entries=16)
def cached_rc_validation(js: str, n_env: int, n_reps: int, ranges_js: str, horizon: int, continuation: str,
                         env_seed: int, discount: float = 1.0, oos_split: float = 0.5):
    r = json.loads(ranges_js)
    ranges = EnvRanges(**{k: tuple(val) for k, val in r.items()})
    envs = random_environments(from_json(js), n_env, ranges, seed=env_seed)
    return rc_validation(envs, n_reps, horizon=horizon, continuation=continuation, discount=discount,
                         oos_split=oos_split)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_tournament(js: str, n_reps: int, horizon: int):
    return run_tournament(from_json(js), n_reps, horizon=horizon)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_horse_race(js: str, n_env: int, n_reps: int, ranges_js: str, horizon: int, continuation: str,
                      env_seed: int, discount: float, oos_split: float, measure: str):
    df = cached_rc_validation(js, n_env, n_reps, ranges_js, horizon, continuation, env_seed, discount, oos_split)
    enc, gain = encompassing_test(df, measure)
    return horse_race(df, measure), enc, gain, float((df["dyn_adv_eval"] > 0).mean()), len(df)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_bound(js: str, phis: tuple, difficulty: str, levels: tuple, n_reps: int):
    return adjustment_bound_experiment(from_json(js), list(phis), difficulty, list(levels), n_reps)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_uncertainty(js: str, deltas: tuple, hazards: tuple, n_reps: int, horizon: int, continuation: str,
                       discount: float):
    return uncertainty_comparison(from_json(js), list(deltas), list(hazards), n_reps, horizon=horizon,
                                  continuation=continuation, discount=discount)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_event(js: str, n_reps: int, horizon: int, continuation: str, pre: int, post: int):
    return regime_event_study(from_json(js), n_reps, horizon=horizon, continuation=continuation, pre=pre, post=post)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_evolution(js: str, deltas: tuple, n_reps: int, horizon: int, continuation: str):
    return evolution_runs(from_json(js), list(deltas), n_reps, horizon=horizon, continuation=continuation)


# ------------------------------------------------------------------------------------------------
# Reusable display blocks
# ------------------------------------------------------------------------------------------------
def _literature():
    """The literature registry as currently loaded. Looked up at call time, not bound at import, because Streamlit
    reloads changed modules only: a name imported here would keep pointing at an outdated registry after an update
    that changed the registry but not this file."""
    import heiner_abm.literature as lit
    return lit


def hypothesis_card(hid: str, statement: str, title: Optional[str] = None, rc: Optional[str] = None,
                    trad: Optional[str] = None):
    """Hypothesis, the competing predictions, the research behind each, and the simulation's contribution.
    Title and predictions come from the literature registry unless overridden."""
    h = _literature().HYPOTHESIS_BY_ID.get(hid)
    if h is None:      # never take a page down for a missing registry entry
        st.markdown(f"#### {title or hid}")
        st.markdown(statement)
        st.caption(f"Research basis for {hid} is unavailable. If the app was just updated, reboot it so that every "
                   "module is reloaded.")
        return
    rc, trad = rc or h.rc_prediction, trad or h.alt_prediction
    title = title or (f"{hid}: {h.title}" if hid.startswith("H") else h.title)
    st.markdown(f"#### {title}")
    st.markdown(statement)
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Heiner / reliability condition predicts**")
        st.markdown(rc)
    with c2.container(border=True):
        st.markdown(f"**Alternative ({h.alt_label.lower()}) predicts**")
        st.markdown(trad)
    research_panel(h)


def _evidence_md(items) -> str:
    refs = _literature().REFERENCES
    return "\n".join(f"* **{refs[k].cite}**: {note}" for k, note in items)


def research_panel(h: Hypothesis, nested: bool = False):
    """Supporting research for the RC prediction and for the alternative, plus the simulation's contribution.
    nested=True draws a bordered box instead of an expander (Streamlit does not allow nested expanders)."""
    label = (f"📚 Research basis · {len(h.support)} studies support the RC prediction, "
             f"{len(h.alternative)} support the alternative")
    box = st.container(border=True) if nested else st.expander(label)
    with box:
        if nested:
            st.markdown(f"**{label}**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Supporting research: reliability condition**")
            st.markdown(_evidence_md(h.support))
        with c2:
            st.markdown(f"**Supporting research: alternative ({h.alt_label.lower()})**")
            st.markdown(_evidence_md(h.alternative))
        st.markdown("**References**")
        keys = list(dict.fromkeys(k for k, _ in h.support + h.alternative))
        st.markdown("\n".join(f"* {r.apa}" for r in _literature().bibliography(keys)))
    st.caption(f"**Contribution:** {h.contribution}")


def verdict(kind: str, text: str):
    {"support": st.success, "reject": st.error, "neutral": st.info, "warn": st.warning}[kind](
        text, icon={"support": "✅", "reject": "❌", "neutral": "➖", "warn": "⚠️"}[kind])


def slope_chart(summary: pd.DataFrame, x: str, x_title: str, group: Optional[str] = None,
                group_title: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    """Mean within-market profit~flexibility slope with 95% CI against x (one line per group)."""
    fig = go.Figure()
    groups = [(None, summary)] if group is None else list(summary.groupby(group))
    for k, (gv, g) in enumerate(groups):
        g = g.sort_values(x)
        color = CAT[k % len(CAT)]
        name = "Mean slope" if gv is None else (f"{group_title or group} = {gv:g}" if isinstance(gv, (int, float))
                                                 else str(gv))
        fig.add_trace(go.Scatter(
            x=g[x], y=g["slope"], mode="lines+markers", name=name, line=dict(color=color, width=2),
            marker=dict(size=8), error_y=dict(type="data", symmetric=False, array=g["hi"] - g["slope"],
                                               arrayminus=g["slope"] - g["lo"], thickness=1.2, width=4),
            customdata=np.stack([g["lo"], g["hi"], g["flex_best"], g["rigid_best"]], axis=1),
            hovertemplate=(f"{x_title}: %{{x:.3g}}<br>slope: %{{y:.1f}} [%{{customdata[0]:.1f}}, "
                           "%{customdata[1]:.1f}]<br>most flexible firm best: %{customdata[2]:.0%}"
                           "<br>most rigid firm best: %{customdata[3]:.0%}<extra>" + name + "</extra>")))
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"))
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="Δ avg profit per unit of flexibility φ")
    return style(fig, title=title)


def firm_profit_bars(firms: pd.DataFrame, by: str, by_title: str, fmt=lambda x: f"{x:g}") -> go.Figure:
    """Average profit by firm (ordered by flexibility) within each condition, with 95% CI across reps."""
    fig = go.Figure()
    fl = sorted(firms["flex"].round(4).unique())
    cols = firm_colors(len(fl))
    agg = firms.assign(flexr=firms["flex"].round(4)).groupby([by, "flexr"])["avg_profit"].agg(["mean", "std", "count"])
    agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
    agg = agg.reset_index()
    for j, f in enumerate(fl):
        d = agg[agg["flexr"] == f]
        fig.add_trace(go.Bar(x=[fmt(x) for x in d[by]], y=d["mean"], name=f"φ = {f:.2f}", marker_color=cols[j],
                             error_y=dict(type="data", array=d["ci"], thickness=1, width=3),
                             hovertemplate=f"{by_title} %{{x}}<br>φ = {f:.2f}<br>avg profit %{{y:,.0f}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_xaxes(title=by_title, type="category")
    fig.update_yaxes(title="Average profit per period")
    return style(fig)


def download(df: pd.DataFrame, name: str, label: str = "Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv")


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"
