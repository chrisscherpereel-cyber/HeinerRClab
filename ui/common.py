"""Shared Streamlit pieces: base-scenario sidebar, presets, cached runners, chart helpers."""
from __future__ import annotations

import functools
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
from heiner_abm.gates import GATE_LABELS
from heiner_abm.information import DEMAND_LABELS, FEEDBACK_LABELS, InfoSpec, UnsupportedInformation
from heiner_abm.params import (LEGACY_FEEDBACK, AdaptiveParams, EvolutionParams, FirmSpec, GlobalFirmParams,
                               MarketParams, SELECTION_HELP, SELECTION_RULES, Scenario, StructuralParams,
                               linear_flex_firms)
from heiner_abm.literature import Hypothesis
from heiner_abm.theories import run_tournament
from heiner_abm.terminology import rule_option

# ------------------------------------------------------------------------------------------------
# Colors (reference palette): firms are ordered by flexibility -> one-hue sequential ramp
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


from heiner_abm.scenarios import DEFAULTS, PRESETS, PRESET_INFO, build_scenario  # noqa: E402

K = "cfg_"


def _init_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(K + k, v)
    if st.session_state.get("focal_theory") not in _focal().FOCAL_KEYS:
        st.session_state["focal_theory"] = _focal().DEFAULT_FOCAL


def _focal():
    """The focal-theory module, looked up at call time (see _literature)."""
    import heiner_abm.focal as focal
    return focal


def focal_theory() -> str:
    """Key of the theory the user has chosen to test and highlight (default: Heiner's reliability condition)."""
    key = st.session_state.get("focal_theory")
    return key if key in _focal().FOCAL_KEYS else _focal().DEFAULT_FOCAL


def focal_title() -> str:
    return _focal().title(focal_theory())


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


SECTION_NOTES = {
    "Start": "Choose a question: guided questions, example studies and saved experiments.",
    "Experiment": "Configure and run a comparison in one workspace; registered study pages follow.",
    "Results": "Interpret a completed experiment; specialized mechanism analyses follow.",
    "Validation": "Analytical benchmarks, empirical data and human experiments: what the findings can support.",
    "Reference": "Theory, equations, agent specifications and bibliography.",
}


# Simulation pages by fairness (keyed by page url path, i.e. the file name without .py). General simulations compare
# the theories on equal terms: every theory takes part as agents (or as a stated prediction for every experiment) with
# the same information, random draws and tuning budget. Special simulations are built around Heiner's framework
# (rule B, the market model's flexibility φ, the CD-gap and the reliability-condition bookkeeping).
SIM_GROUPS = {
    "theories": "general", "arena": "general", "rule_choice": "general", "special_tests": "general",
    "market_lab": "special", "designer": "special", "evolution": "special", "generalisation": "special",
    "learnability": "special", "nk": "special", "mechanisms": "special", "reliability_gates": "special",
    "dynamic_rc": "special", "cd_gap": "special", "rc_validation": "special", "tracking": "special",
}
SIM_GROUP_LABELS = {"general": ("General simulations", "fair comparisons: every theory as agents, on equal terms"),
                    "special": ("Special simulations", "built around Heiner's framework")}


def _grouped_links(pages) -> None:
    """Page links, with simulation pages under General and Special simulations headings and the rest after them."""
    rest = [p for p in pages if p.url_path not in SIM_GROUPS]
    for g, (label, note) in SIM_GROUP_LABELS.items():
        mine = [p for p in pages if SIM_GROUPS.get(p.url_path) == g]
        if mine:
            st.markdown(f"**{label}**")
            st.caption(note)
            for page in mine:
                st.page_link(page, label=page.title, icon=page.icon)
    if rest and len(rest) < len(pages):
        st.markdown("**Other pages**")
    for page in rest:
        st.page_link(page, label=page.title, icon=page.icon)


def _arena():
    """The agent definitions, looked up at call time (see _literature)."""
    import heiner_abm.arena as arena
    return arena


def render_agents() -> None:
    """Sidebar: every theory's agents, with the decision rule each one implements (read from the code)."""
    ar = _arena()
    sb = st.sidebar
    sb.markdown("### Agents")
    sb.caption("Every theory is implemented as agents that compete in the same market with the same information. "
               "Open a theory to see its agents' decision rules.")
    for theory, keys in ar.THEORY_DESIGNS.items():
        with sb.expander(ar.THEORY_NAMES[theory]):
            for k in keys:
                d = ar.DESIGNS[k]
                params = ", ".join(d.SPACE) or "none"
                st.markdown(f"**{d.name}** · `{k}`  \n{d.rule}  \n*Tuned parameters:* {params}")


def render_navigation(sections, current) -> None:
    """Sidebar navigation: the interface mode, the five destinations, then the pages within the current section and
    every other page in collapsed groups (simulation pages split into General and Special simulations), then every
    theory's agents."""
    sb = st.sidebar
    sb.markdown("## ⚖️ Decision under uncertainty lab")
    sb.radio("Mode", ["explore", "research"], key="wb_mode", horizontal=True,
             format_func={"explore": "Explore", "research": "Research"}.get,
             help="Explore: teaching and first investigation (simplified uncertainty presets, quick previews). "
                  "Research: controlled experiments (every setting, validity checks, background runs). Both use the "
                  "same simulation engines.")
    here = next((n for n, pages in sections.items() if any(p.url_path == current.url_path for p in pages)), None)
    for name, pages in sections.items():
        sb.page_link(pages[0], label=name, icon=pages[0].icon)
    if here and len(sections[here]) > 1:
        with sb.expander(f"More in {here}", expanded=True):
            st.caption(SECTION_NOTES.get(here, ""))
            _grouped_links(sections[here][1:])
    with sb.expander("All pages", expanded=False):
        for name, pages in sections.items():
            if name == here:
                continue
            st.markdown(f"#### {name}")
            _grouped_links(pages[1:])
    with sb.expander("Theory under test", expanded=False):
        st.selectbox("Highlighted theory", _focal().FOCAL_KEYS, format_func=_focal().title, key="focal_theory",
                     help="The theory whose predictions are highlighted on every hypothesis card, in the overview "
                          "and research tables and in the tournament reviews. The others are shown as competitors. "
                          "Heiner's reliability condition is one choice among nine.")
    render_agents()


def render_sidebar(sections=None, current=None):
    _init_state()
    sb = st.sidebar
    if sections is not None and current is not None:
        render_navigation(sections, current)
    else:
        sb.markdown("### Theory under test")
        sb.selectbox("Highlighted theory", _focal().FOCAL_KEYS, format_func=_focal().title, key="focal_theory",
                     help="The theory whose predictions are highlighted on every hypothesis card.")
    sb.markdown("### Base market scenario")
    sb.caption("Used by the specialized market pages (not by the Experiment workspace, which keeps its own "
               "specification). Experiments override only the parameters they sweep.")
    sb.selectbox("Preset", list(PRESETS), key="preset_choice",
                 help="Ready-made scenarios. Pick one, then press **Apply preset** to load its values into every sidebar "
                      "control; **Reset** restores the baseline calibration. Every preset is described on the "
                      "*Model & methods* page.")
    info = PRESET_INFO.get(st.session_state.get("preset_choice", next(iter(PRESETS))))
    if info:
        with sb.popover("About this preset", width="stretch"):
            st.markdown(f"**Setup.** {info['setup']}\n\n**Why run it.** {info['why']}\n\n"
                        f"**Used for.** {info['hypotheses']}\n\n**Typical result.** {info['typical']}")
            chosen = PRESETS.get(st.session_state.get("preset_choice"), {})
            replication_notice("preset_adaptive" if chosen.get("selection") == "Adaptive" else "presets")
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
                        help="Slope = (max price − min price) / quantity range. Steep demand destabilizes the cobweb.")
        st.number_input("Initial raw-material cost c₀", 0.0, 500.0, key=K + "c0", step=1.0,
                        help="Raw-material cost in period 0; the cost random walk starts here. Must lie between "
                             "min price and max raw-material cost.")
        st.number_input("Max raw-material cost", 1.0, 500.0, key=K + "c_max", step=1.0,
                        help="Upper reflecting bound; lowering it raises industry profitability.")
        st.slider("Cost volatility Δ (max change per period)", 0.0, 60.0, key=K + "delta", step=0.5,
                  help="Difficulty of the environment. c[t] = c[t−1] + Δ·U(−1,1), reflected at the bounds.")

    with sb.expander("Firms", expanded=False):
        st.slider("Number of firms", 2, 12, key=K + "n_firms",
                  help="Firms in the market. Firm i gets flexibility φᵢ = intercept + slope·i, so firms are "
                       "ordered from most rigid (firm 1) to most flexible.")
        st.radio("Production rule", ["Bertrand", "Cournot"], key=K + "rule", horizontal=True, format_func=rule_option,
                 help="How a firm computes its recommended output q*. **Cournot best reply** (model-based): move a "
                      "fraction φ of the way toward the best reply on the believed demand curve (φ ≤ 1). "
                      "**Margin-feedback quantity rule** (model-free): q* = q + φ·(observed margin − m*), using only "
                      "the last price. Despite its configuration name 'Bertrand', it is not price-setting Bertrand "
                      "competition: firms choose quantities and the market clears one price.")
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
        st.checkbox("Margin-feedback rule: margin includes F/q", key=K + "margin_includes_fixed",
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
                       "H ≈ 20+ captures their consequences. Cost grows with H. This is a researcher's measurement: "
                       "agents never see it, unless the researcher-only *oracle* feedback below is chosen.")
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
        st.slider("Adaptive agents' judgement window", 1, 100, key=K + "adaptive_window",
                  help="An Adaptive firm judges each past deviation once this many periods have passed: holding the "
                       "recommended output versus holding its old output (rule B), with rivals' actual output, "
                       "realized costs and prices on its own believed demand curve. All of it has been observed by "
                       "then, so the firm uses no future information.")
        st.radio("Adaptive agents' selection gate", ["gain", "lcb", "explore"], key=K + "gate",
                 format_func=lambda x: GATE_LABELS[x],
                 help="How an Adaptive firm turns its evidence into a decision (the same recommendations in every "
                      "case). **Estimated gain** (existing): adopt if the estimated advantage is at least the "
                      "adjustment cost. **Confidence-sensitive**: adopt only with enough evidence and a lower "
                      "confidence bound above the adjustment cost. **Exploration**: learns from its own payoffs in "
                      "randomized trials, without counterfactual feedback. The last two are proposed extensions; see "
                      "the *Reliability gates* page.")
        if v("gate") != "gain" or float(v("adjust_cost")) > 0:
            st.number_input("Adjustment cost c (per adaptation)", 0.0, 10000.0, key=K + "adjust_cost", step=5.0,
                            help="Compared with the estimated advantage of adopting; charged in the net-payoff "
                                 "accounting, not in market prices. Kept separate from estimation uncertainty.")
        if v("gate") in ("lcb", "explore"):
            st.slider("Confidence level of the lower bound", 0.5, 0.99, key=K + "confidence", step=0.01,
                      help="One-sided level of the Student-t bound on the advantage. 0.5 uses the point estimate. "
                           "Nominal only for independent, identically distributed feedback, which the market does "
                           "not guarantee.")
            st.number_input("Minimum evidence (effective feedback items)", 2.0, 200.0, key=K + "min_evidence",
                            step=1.0, help="Effective number of feedback items (per arm for exploration) a size bin "
                                           "needs before the gate acts on it; with less it keeps the default.")
            st.selectbox("Response to an observed environmental change", ["forget", "reset"], key=K + "on_change",
                         format_func=lambda x: {"forget": "Exponential memory only",
                                                "reset": "Discard evidence when a change is seen"}[x],
                         help="A change is seen when a shift is announced or the firms' demand curve is updated.")
        if v("gate") == "explore":
            st.slider("Exploration rate", 0.0, 1.0, key=K + "explore_rate", step=0.05,
                      help="Probability that an uncertain decision becomes a randomized trial (adopt or keep, each "
                           "with probability 1/2).")

    with sb.expander("Information & feedback", expanded=False):
        st.caption("What firms may observe and what feedback they receive. Unsupported combinations are reported, "
                   "never run with extra information. Every agent's access is on the *Information & feedback* page.")
        st.radio("Feedback treatment", ["chosen", "estimated", "oracle", "full"], key=K + "info_feedback",
                 format_func=lambda x: FEEDBACK_LABELS[x],
                 help="**Chosen-action**: only the payoff of the firm's own action. **Estimated** (default): the firm "
                      "estimates the payoff of the alternative from its observations and its own believed demand "
                      "curve. **Oracle**: the researcher's counterfactual (true demand curve, rivals' simulated "
                      "reactions), a labeled diagnostic benchmark. **Full**: true payoffs of alternatives, available "
                      "only where the design shows them; the cobweb market does not, so it is rejected here. Adaptive "
                      "firms need estimated or oracle feedback, except with the exploration gate, which learns from "
                      "chosen-action feedback.")
        st.slider("Observation noise (s.d.)", 0.0, 20.0, key=K + "info_noise", step=0.5,
                  help="Noise added to every observed market price and market output (separate random stream).")
        st.number_input("Observation delay (periods)", 0, 20, key=K + "info_delay", step=1,
                        help="Periods before a period's market outcomes (price, output, cost, payoffs) are observed. "
                             "Feedback is released that many periods after it matures.")
        st.radio("Demand knowledge", ["believed", "true", "none"], key=K + "info_demand",
                 format_func=lambda x: DEMAND_LABELS[x],
                 help="**Believed**: the baseline curve, updated L periods after a shift. **True**: the true current "
                      "curve (ORACLE information). **None**: no demand model; the Cournot rule and estimated feedback "
                      "need one and are then rejected.")
        st.checkbox("Regime shifts are announced (with the new demand curve)", key=K + "info_announced")
        st.checkbox("Rivals' individual outputs and payoffs are visible", key=K + "info_rivals",
                    help="Needed by evolution of flexibility (imitating the most profitable firm).")
        if v("info_feedback") == "oracle" or v("info_demand") == "true":
            st.warning("ORACLE information: firms receive researcher-only quantities. Label any result as an oracle "
                       "benchmark, not as behavior of an agent with realistic information.", icon="⚠️")

    with sb.expander("Structural uncertainty (regime shifts)", expanded=False):
        st.toggle("Unannounced demand-regime shifts", key=K + "struct_on",
                  help="Unannounced structural change: the demand curve occasionally jumps to a new regime that "
                       "model-based (Cournot) firms don't know about until they update their model. The simulator "
                       "specifies the shift process; the firms are not told it (see *Model & methods*, kinds of "
                       "uncertainty).")
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
    sb.caption("See the *Theories* overview and *Research & contribution* for the research behind every test.")


def base_scenario() -> Scenario:
    return build_scenario({k: v(k) for k in DEFAULTS})


def measurement():
    return int(v("horizon")), str(v("continuation"))


def measure_opts():
    """Discount and estimation-window split used by the dynamic and out-of-sample RC measures."""
    return float(v("discount")), float(v("oos_split"))


def behaviour_horizon(scn: Scenario) -> int:
    """Horizon needed when only profits are reported. H is a researcher's measurement; it changes behavior only when
    Adaptive agents are given the researcher-only oracle feedback, so skip the (costly) forks otherwise."""
    oracle = scn.info.feedback == "oracle" and any(f.selection == "Adaptive" for f in scn.firms)
    return int(v("horizon")) if oracle else 1


def reps() -> int:
    return int(v("reps"))


def show_errors(scn: Scenario) -> bool:
    errs = scn.validate()
    for e in errs:
        st.error(e, icon="⛔")
    return not errs


# ------------------------------------------------------------------------------------------------
# (De)serialization for caching
# ------------------------------------------------------------------------------------------------
def to_json(scn: Scenario) -> str:
    return json.dumps(asdict(scn), sort_keys=True)


def from_json(js: str) -> Scenario:
    d = json.loads(js)
    return Scenario(market=MarketParams(**d["market"]), firms=[FirmSpec(**f) for f in d["firms"]],
                    firm_globals=GlobalFirmParams(**d["firm_globals"]),
                    adaptive=AdaptiveParams(**{k: val for k, val in {**d["adaptive"],
                                               "bin_edges": tuple(d["adaptive"]["bin_edges"]),
                                               "oracle_table": tuple(tuple(r) for r in
                                                                     d["adaptive"].get("oracle_table", ()))}.items()
                                               if k != "feedback"}),
                    info=InfoSpec(**d["info"]) if "info" in d else
                    InfoSpec(feedback=LEGACY_FEEDBACK.get(d["adaptive"].get("feedback", "observable"), "estimated")),
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
def cached_agent_track(js: str, n_reps: int, periods: int):
    from heiner_abm.agent_track import agent_track
    return agent_track(from_json(js), n_reps, periods=periods)


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


_SOURCE_NOTE = {"tournament": "Stated in the directional tournament (Competing theories page).",
                "theory": "Derived from the theory's core claim.",
                "general": "This theory makes no specific prediction here; its general stance is shown."}


def hypothesis_card(hid: str, statement: str, title: Optional[str] = None, rc: Optional[str] = None,
                    trad: Optional[str] = None, notes: Optional[Dict[str, str]] = None):
    """Hypothesis, the prediction of the theory under test (chosen in the sidebar), the competing predictions, the
    research behind them, and the simulation's contribution. The statement describes the test neutrally; theory-
    specific reasoning goes in `notes` ({theory key: text}) and is shown under that theory's prediction. Title and
    predictions come from the literature registry unless overridden (rc: the reliability-condition prediction; trad:
    the registered alternative)."""
    notes = notes or {}
    h = _literature().HYPOTHESIS_BY_ID.get(hid)
    if h is None:      # never take a page down for a missing registry entry
        st.markdown(f"#### {title or hid}")
        st.markdown(statement)
        st.caption(f"Research basis for {hid} is unavailable. If the app was just updated, reboot it so that every "
                   "module is reloaded.")
        return
    F = _focal()
    focal = focal_theory()
    rc, trad = rc or h.rc_prediction, trad or h.alt_prediction
    owners = F.alternative_owners(h.alt_label)
    if focal == "heiner":
        pred, src = rc, "registry"
    elif focal in owners:
        pred, src = trad, "registry"
    else:
        pred, src = F.prediction(hid, focal)
    title = title or (f"{hid}: {h.title}" if hid.startswith("H") else h.title)
    st.markdown(f"#### {title}")
    st.markdown(statement)
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown(f"**⭐ {F.title(focal)} predicts** (theory under test)")
        st.markdown(pred)
        if notes.get(focal):
            st.markdown(f"*Reasoning:* {notes[focal]}")
        if src in _SOURCE_NOTE:
            st.caption(_SOURCE_NOTE[src])
    with c2.container(border=True):
        st.markdown("**Competing predictions**")
        lines = []
        if focal != "heiner":
            lines.append(f"* **{F.title('heiner')}:** {rc}" + (f" *Reasoning:* {notes['heiner']}"
                                                                if notes.get("heiner") else ""))
        if focal not in owners:
            extra = " ".join(notes[k] for k in owners if notes.get(k))
            lines.append(f"* **Alternative ({h.alt_label.lower()}):** {trad}" + (f" *Reasoning:* {extra}" if extra
                                                                                  else ""))
        st.markdown("\n".join(lines))
        with st.popover("All nine theories", width="stretch"):
            st.dataframe(pd.DataFrame([dict(Theory=("⭐ " if k == focal else "") + F.title(k),
                                            Prediction=F.prediction(hid, k)[0], Basis=F.prediction(hid, k)[1])
                                       for k in F.FOCAL_KEYS]), hide_index=True, width="stretch")
            st.caption("Basis: registry = the hypothesis's registered predictions; tournament = the directional "
                       "tournament; theory = derived from the theory's core claim; general = no specific prediction.")
    research_panel(h)


def _evidence_md(items) -> str:
    refs = _literature().REFERENCES
    return "\n".join(f"* **{refs[k].cite}**: {note}" for k, note in items)


def research_panel(h: Hypothesis, nested: bool = False):
    """Research supporting each side of the hypothesis, ordered so that the theory under test comes first, plus the
    simulation's contribution. nested=True draws a bordered box instead of an expander (Streamlit does not allow
    nested expanders)."""
    F = _focal()
    focal = focal_theory()
    rc_side = (f"Supporting research: {F.title('heiner')}", h.support)
    alt_side = (f"Supporting research: alternative ({h.alt_label.lower()})", h.alternative)
    if focal != "heiner" and focal in F.alternative_owners(h.alt_label):
        first, second = ((f"Supporting research: ⭐ {F.title(focal)} (the alternative)", h.alternative), rc_side)
    elif focal == "heiner":
        first, second = ((f"Supporting research: ⭐ {F.title('heiner')}", h.support), alt_side)
    else:
        first, second = rc_side, alt_side
    label = (f"📚 Research basis · {len(first[1])} studies support the first position, {len(second[1])} the "
             "second")
    box = st.container(border=True) if nested else st.expander(label)
    with box:
        if nested:
            st.markdown(f"**{label}**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{first[0]}**")
            st.markdown(_evidence_md(first[1]))
        with c2:
            st.markdown(f"**{second[0]}**")
            st.markdown(_evidence_md(second[1]))
        if focal != "heiner" and focal not in F.alternative_owners(h.alt_label):
            st.caption(f"Neither list was compiled for {F.title(focal)}; its own sources are on its theory page.")
        st.markdown("**References**")
        keys = list(dict.fromkeys(k for k, _ in h.support + h.alternative))
        st.markdown("\n".join(f"* {r.apa}" for r in _literature().bibliography(keys)))
    st.caption(f"**Contribution:** {h.contribution}")


PREREG_TEXT = (
    "**What it is.** Before a confirmatory study is run, its plan is fixed in writing: the hypotheses, the exact test "
    "and decision rule for each (what result counts as support and what does not), the environments and sample sizes, "
    "the tuning budget every agent receives, and the random seeds. The app turns the plan, together with the source "
    "code of the agents and the analysis, into a short fingerprint (a hash, such as `110b3146bb072c2c`).\n\n"
    "**Why it matters.**\n"
    "* *No moving the goalposts.* Tests, settings and thresholds cannot be chosen after seeing which ones favor a "
    "theory (the 'garden of forking paths', p-hacking, or hypothesizing after the results are known).\n"
    "* *Fairness between theories.* Every theory's agents get the same budget, environments and yardsticks, fixed "
    "before any of them is run.\n"
    "* *Verifiability.* The same hash means the same plan and the same code, so a reported result can be traced to "
    "exactly what produced it. The plan can be downloaded as a file and deposited, with a time stamp, in a registry "
    "such as OSF or AsPredicted before the study is run.\n\n"
    "**Registered versus exploratory.** Changing any setting (or the code) changes the hash, and the app then labels "
    "the run *exploratory*. Exploratory runs are useful for learning and for generating new hypotheses, but they are "
    "not confirmatory evidence. Results reported in the README and on the theory pages are tied to the registered "
    "hashes, and the test suite fails if the code changes them without the results being rerun (Nosek et al., 2018).\n\n"
    "**What the hash does not show.** In this laboratory a *pre-registered plan* means a plan frozen in the code and "
    "identified by its hash. The repository contains no record of any plan being deposited with an external, "
    "time-stamped registry. The hash shows which plan and which code produced a result; it does not show that the plan "
    "was fixed before the results were seen. Plans were revised during development before they were frozen (see the "
    "README). Treat registered results accordingly, and deposit a plan externally before a confirmatory run if that "
    "guarantee is needed.")


def prereg_explainer():
    """An expander explaining frozen plans, for every page that runs one."""
    from heiner_abm.terminology import FROZEN_PLAN_NOTE
    with st.expander("ℹ️ What is a frozen ('pre-registered') plan, and what is it not?"):
        st.markdown(PREREG_TEXT + "\n\n**Frozen hashed specification versus external preregistration.** "
                    + FROZEN_PLAN_NOTE)


def evidence_note(*keys: str, scope: bool = True):
    """Caption stating what kind of evidence a page's results are (heiner_abm.terminology.EVIDENCE_TYPES) and, for
    comparisons, that conclusions are limited to the tested implementations and environment."""
    from heiner_abm.terminology import EVIDENCE_TYPES, SCOPE_NOTE
    info = {k: (label, does, doesnt) for k, label, does, doesnt in EVIDENCE_TYPES}
    text = " ".join(f"**Evidence type: {info[k][0]}.** {info[k][1]} *Limits:* {info[k][2]}" for k in keys)
    if scope and any(k in ("simulation", "behavioral") for k in keys):
        text += " " + SCOPE_NOTE
    st.caption(text)


def replication_notice(*keys: str, where=None):
    """Flag reported findings whose code changed after they were reported (heiner_abm.registered). The reported
    numbers stay as they were; the notice says they require replication until rerun."""
    from heiner_abm import registered
    where = where or st
    for key in keys:
        note = registered.replication_note(key)
        if note:
            where.warning(f"**Requires replication: {registered.FINDING_SOURCES[key][0]}.** {note} The numbers shown "
                          "are those originally reported; rerun the study before relying on them.", icon="⚠️")


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


def _explain_unsupported(fn):
    """Wrap a cached runner: if the information specification does not support an agent the experiment needs, say
    so and stop the page instead of running it (never with extra information)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except UnsupportedInformation as err:
            st.warning(f"**Not run: the information specification does not support this experiment.** {err} Change "
                       "it in the sidebar's *Information & feedback* section; the *Information & feedback* page shows "
                       "what every agent needs.", icon="⛔")
            st.stop()
    return wrapper


for _name in [n for n in list(globals()) if n.startswith("cached_")]:
    globals()[_name] = _explain_unsupported(globals()[_name])


def adaptive_supported(scn: Scenario) -> bool:
    """Whether Adaptive firms could run under the scenario's information specification."""
    from heiner_abm.information import AGENT_BY_KEY, adaptive_key, compatibility
    return compatibility(AGENT_BY_KEY[("market", adaptive_key(scn.adaptive.gate))], scn.info,
                         "market").status == "supported"


def set_engine(engine: str, agents=None):
    """Declare which simulation engine (and agents) produced this page's results; app.py resets it to the market
    engine on every run. The information specification of that engine is exported with every download."""
    st.session_state["_info_engine"] = (engine, agents)


def info_record() -> dict:
    """The information-and-feedback specification behind the current page's results."""
    from heiner_abm.information import export_record, market_agents
    engine, agents = st.session_state.get("_info_engine", ("market", None))
    if engine == "none":
        return {"engine": "none", "note": "Rules fitted to recorded human choices; no simulated agents, so no "
                                          "information treatment applies. The data's own information conditions are "
                                          "those of the original experiments."}
    if engine == "market":
        scn = base_scenario()
        return export_record("market", scn.info, market_agents(scn) if agents is None else agents)
    return export_record(engine, agents=agents)


def download(df: pd.DataFrame, name: str, label: str = "Download CSV"):
    """CSV of the results, with the information-and-feedback specification that produced them beside it."""
    c1, c2 = st.columns(2)
    c1.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv", width="stretch")
    c2.download_button("Information spec (JSON)", json.dumps(info_record(), indent=1).encode(),
                       file_name=name.rsplit(".", 1)[0] + "_information_spec.json", mime="application/json",
                       width="stretch", key=f"_spec_{name}_{id(df)}",
                       help="What the agents could observe and what feedback they received when these results were "
                            "produced (engine, specification, and each agent's access).")


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"
