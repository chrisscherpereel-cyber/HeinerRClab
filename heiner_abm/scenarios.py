"""Market scenarios: the baseline calibration, the preset scenarios and the scenario builder.

Shared by the app's sidebar (ui.common) and the command-line research runner (heiner_abm.research), so both build
exactly the same Scenario from the same settings. No Streamlit dependency.

The typical result of each preset is not stored here: it is a claim rendered from a recorded run of the `presets`
study (heiner_abm.research.claims), so the numbers shown always identify the run that produced them.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from .information import InfoSpec
from .params import (AdaptiveParams, GlobalFirmParams, MarketParams, Scenario, StructuralParams,
                     linear_flex_firms)

DEFAULTS: Dict[str, object] = dict(
    p_max=100.0, p_min=10.0, q_range=1500.0, c0=45.0, c_max=80.0, delta=10.0,
    n_firms=4, rule="Bertrand", flex_intercept=0.0, flex_slope=0.25, selection="Always", threshold=25.0,
    desired_margin=5.0, foresight=0.0, noise=0.0, q0=200.0, q_min=15.0,
    flex_cost_slope=0.0, fixed_cost=0.0, margin_includes_fixed=False,
    periods=1000, burn_in=25, reps=20, seed=1, horizon=20, continuation="default", memory=0.97,
    adaptive_window=20, gate="gain", adjust_cost=0.0, confidence=0.9, min_evidence=5.0, explore_rate=0.2,
    on_change="forget", info_feedback="estimated", info_noise=0.0, info_delay=0, info_demand="believed",
    info_announced=False, info_rivals=True,
    discount=1.0, oos_split=0.5,
    struct_on=False, hazard=0.02, intercept_sd=15.0, slope_sd=0.4, belief_lag=20,
)

PRESETS: Dict[str, Dict[str, object]] = {
    "Baseline (margin-feedback rule, 4 firms)": {},
    "Cournot best-reply rule": dict(rule="Cournot"),
    "Low-profit margin-feedback market (fierce competition, m* = 0.5)": dict(desired_margin=0.5),
    "High-profit margin-feedback market (gentlemanly, m* = 14)": dict(desired_margin=14.0),
    "Volatile raw materials (Δ = 25)": dict(delta=25.0),
    "Selection rules: SR2 'Large' (θ = 25)": dict(selection="Large", threshold=25.0),
    "Selection rules: SR1 'Small' (θ = 10)": dict(selection="Small", threshold=10.0),
    "Reliability-learning (Adaptive) agents, judged over 25 periods": dict(selection="Adaptive", horizon=25,
                                                                           adaptive_window=25),
    "Costly flexibility (a = 500, b = 100, in margin)": dict(flex_cost_slope=500.0, fixed_cost=100.0,
                                                             margin_includes_fixed=True),
    "Ten firms (larger market)": dict(n_firms=10, flex_slope=0.1, q0=80.0),
    "Unannounced demand shifts (Cournot best reply, φ ≤ 0.4)": dict(rule="Cournot", flex_slope=0.1,
                                                                              struct_on=True, delta=5.0),
    "Unannounced demand shifts (margin-feedback rule)": dict(struct_on=True, delta=5.0),
}

# What each preset changes, why one would run it, and what a typical run shows (6 replications of 1,000 periods,
# checked against the simulation; "slope" is the change in average profit from the most rigid to the most flexible
# firm, per firm step).
PRESET_INFO: Dict[str, Dict[str, str]] = {
    "Baseline (margin-feedback rule, 4 firms)": dict(
        setup="Four firms with flexibility φ = 0.25, 0.50, 0.75 and 1.00. Each adjusts output to the observed margin "
              "(price − cost estimate) relative to a desired margin m* = 5, without a model of demand (the margin-feedback quantity rule). "
              "Raw-material cost moves by up to Δ = 10 per period; no perception noise, no demand shifts, flexibility "
              "is free.",
        why="The reference point. Every other preset changes one or two settings relative to it, so differences can "
            "be attributed to those settings.",
        hypotheses="H1, H2, H3 (as the comparison case)",
        typical="Average price ≈ 48.6 against cost ≈ 43.7; prices are smoother than costs; more flexible firms earn "
                "somewhat more (slope ≈ +23)."),
    "Cournot best-reply rule": dict(
        setup="As the baseline, but each firm moves a share φ of the way toward its best reply on the believed demand "
              "curve (the Cournot best-reply quantity rule, model-based).",
        why="Checks whether conclusions depend on how firms decide: model-based (Cournot best reply) versus "
            "model-free (margin feedback). Model-based decisions interact through rivals' reactions and can overshoot.",
        hypotheses="H6, H11, the Competing theories tournament (run under both rules)",
        typical="Prices overshoot (oscillating regime), margins ≈ 11; more flexible firms earn less (slope ≈ −22)."),
    "Low-profit margin-feedback market (fierce competition, m* = 0.5)": dict(
        setup="Baseline with a desired margin of only 0.5, so firms compete fiercely and margins are close to zero.",
        why="When margins are thin, a wrong move is costly relative to what a right move gains. The reliability "
            "condition predicts that flexibility then hurts even though it is free; optimization predicts it cannot.",
        hypotheses="H1 (free flexibility), H2 (profitability switch)",
        typical="Margin ≈ 0.4 and profit ≈ 70 per period; more flexible firms earn less (slope ≈ −6)."),
    "High-profit margin-feedback market (gentlemanly, m* = 14)": dict(
        setup="Baseline with a desired margin of 14: firms compete softly and margins are wide.",
        why="The mirror image of the low-profit case: mistakes are cheap relative to the gains from adjusting, so "
            "flexibility should pay. Together they locate the profitability at which the effect of flexibility "
            "switches sign.",
        hypotheses="H2 (profitability switch), H5 (competition intensity)",
        typical="Margin ≈ 14; flexibility pays strongly (slope ≈ +107)."),
    "Volatile raw materials (Δ = 25)": dict(
        setup="Baseline with cost volatility Δ = 25 instead of 10: the raw-material cost can move by up to 25 per "
              "period.",
        why="A harder, more volatile environment (risk). Theories disagree on its effect: real options and "
            "optimization say volatility raises the value of flexibility, the reliability condition and bias–variance "
            "reasoning say it lowers it.",
        hypotheses="H3 (volatility), H10 (predictability), E2 in Competing theories",
        typical="Flexibility's payoff turns negative (slope ≈ −14)."),
    "Selection rules: SR2 'Large' (θ = 25)": dict(
        setup="Firms keep last period's output unless the recommended change exceeds θ = 25: they act only on large, "
              "clear signals and ignore small ones.",
        why="Tests whether acting only on clear signals beats always adjusting: the selection rule, not the target, "
            "is what differs.",
        hypotheses="H9 (selection rules)",
        typical="Profit rises with flexibility (slope ≈ +270): restraint on small signals makes flexibility pay."),
    "Selection rules: SR1 'Small' (θ = 10)": dict(
        setup="The opposite rule: firms adjust only when the recommended change is below θ = 10 and ignore large "
              "recommended changes.",
        why="A contrast to SR2: if the gain comes from ignoring noisy small signals, ignoring large signals instead "
            "should hurt.",
        hypotheses="H9 (selection rules)",
        typical="Profit falls with flexibility (slope ≈ −171)."),
    "Reliability-learning (Adaptive) agents, judged over 25 periods": dict(
        setup="Every firm learns, for each size of recommended change, whether deviating from keeping its output has "
              "paid off, and deviates only where it has. It judges each deviation once the 25 periods it covers have "
              "passed, from what it has observed: rivals' actual output, realized costs and prices on its own "
              "believed demand curve (no future information). The researcher's measurement horizon H is also 25. "
              "Nothing is imposed about how often to deviate.",
        why="Lets restraint emerge from experience rather than imposing it. Needed for the predictability hypothesis "
            "and for the reliability-condition validation, which measure how often firms choose to deviate.",
        hypotheses="H10 (predictability), RC validation, CD-gap explorer",
        typical="Average price ≈ 48.2 (baseline 48.6); firms adopt about 74% of the recommended changes, and "
                "flexibility pays more than in the baseline (slope ≈ +58 against +23)."),
    "Costly flexibility (a = 500, b = 100, in margin)": dict(
        setup="Flexibility now costs F = 500·φ + 100 per period, and the fixed cost is included in the margin firms "
              "react to (margin = P − c − F/q).",
        why="Separates the traditional explanation (flexibility is costly) from the reliability explanation "
            "(flexibility is free but error-prone), and tests whether higher fixed costs raise the profitability at "
            "which flexibility starts to pay.",
        hypotheses="H4 (fixed costs), E5 in Competing theories",
        typical="Flexibility is strongly unprofitable (slope ≈ −875), mostly through its direct cost."),
    "Ten firms (larger market)": dict(
        setup="Ten firms with φ = 0.1, 0.2, …, 1.0 and a lower starting output (80 each) so total output starts near "
              "the baseline level.",
        why="More rivals make each firm's target move more with the others' choices (strategic difficulty). Heiner "
            "(1989) and cobweb theory both predict the reliable adjustment speed falls with the number of firms.",
        hypotheses="H11 (number of rivals), E6 in Competing theories",
        typical="Lower profit per firm (≈ 410) with a mildly positive effect of flexibility (slope ≈ +15) under "
                "the margin-feedback rule; rerun under Cournot to see the strategic effect."),
    "Unannounced demand shifts (Cournot best reply, φ ≤ 0.4)": dict(
        setup="Cournot firms (φ = 0.1–0.4, low enough to keep the market stable) face unannounced shifts of the "
              "demand curve (hazard 0.02 per period, intercept s.d. 15, log-slope s.d. 0.4); their demand model is "
              "updated only 20 periods after a shift. Cost volatility is lowered to Δ = 5 so that the shifts dominate.",
        why="Structural change and model misspecification rather than risk: firms that rely on a demand model act on "
            "an outdated model after each shift. Compare with the volatile-costs preset at matched unpredictability.",
        hypotheses="KNIGHT (risk vs unannounced structural change), H12 (model-updating lag), PUNCT (punctuated adjustment)",
        typical="About 19 shifts per run; prices follow costs between shifts; flexibility hurts (slope ≈ −9)."),
    "Unannounced demand shifts (margin-feedback rule)": dict(
        setup="The same demand shifts with margin-feedback (model-free) firms, which react to the observed margin and "
              "never "
              "use a demand model.",
        why="The control for the Cournot regime-shift preset: model-free firms are not misled by an outdated model, "
            "so the theories that point to misspecification predict that flexibility still pays here.",
        hypotheses="KNIGHT (risk vs unannounced structural change)",
        typical="About 19 shifts per run; flexibility pays (slope ≈ +244)."),
}



def build_scenario(values: Mapping[str, Any]) -> Scenario:
    """The Scenario for a full set of settings (DEFAULTS keys); missing keys take their DEFAULTS value."""
    v = {**DEFAULTS, **dict(values)}
    n = int(v["n_firms"])
    rule = v["rule"]
    firms = linear_flex_firms(n=n, slope=float(v["flex_slope"]), intercept=float(v["flex_intercept"]), rule=rule,
                              selection=v["selection"], threshold=float(v["threshold"]),
                              desired_margin=float(v["desired_margin"]), foresight=float(v["foresight"]),
                              noise=float(v["noise"]))
    if rule == "Cournot":
        for f in firms:
            f.flex = min(f.flex, 1.0)
    return Scenario(
        market=MarketParams(p_max=float(v["p_max"]), p_min=float(v["p_min"]), q_range=float(v["q_range"]),
                            c0=float(v["c0"]), c_max=float(v["c_max"]), delta=float(v["delta"])),
        firms=firms,
        firm_globals=GlobalFirmParams(q0=float(v["q0"]), q_min=float(v["q_min"]),
                                      flex_cost_slope=float(v["flex_cost_slope"]),
                                      fixed_cost=float(v["fixed_cost"]),
                                      margin_includes_fixed=bool(v["margin_includes_fixed"])),
        adaptive=AdaptiveParams(memory=float(v["memory"]), window=int(v["adaptive_window"]), gate=str(v["gate"]),
                                adjust_cost=float(v["adjust_cost"]), confidence=float(v["confidence"]),
                                min_evidence=float(v["min_evidence"]), explore_rate=float(v["explore_rate"]),
                                on_change=str(v["on_change"])),
        info=InfoSpec(feedback=str(v["info_feedback"]), obs_noise=float(v["info_noise"]),
                      obs_delay=int(v["info_delay"]), demand_knowledge=str(v["info_demand"]),
                      regime_announced=bool(v["info_announced"]), rivals_visible=bool(v["info_rivals"])),
        structural=StructuralParams(enabled=bool(v["struct_on"]), hazard=float(v["hazard"]),
                                    intercept_sd=float(v["intercept_sd"]), slope_sd=float(v["slope_sd"]),
                                    belief_lag=int(v["belief_lag"])),
        periods=int(v["periods"]), burn_in=int(v["burn_in"]), seed=int(v["seed"]))


def preset_values(name: str) -> Dict[str, Any]:
    return {**DEFAULTS, **PRESETS[name]}
