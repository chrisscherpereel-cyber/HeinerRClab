"""Theory tournament: Heiner's reliability condition against rival theories of flexibility and uncertainty.

Each rival theory is reduced to directional predictions about experiments the cobweb model can run. Every
experiment reports one statistic (usually the regression coefficient of the within-market profit~flexibility
slope on the manipulated variable) whose sign is read at the 5% level as '+', '-' or '0'. A theory's
prediction is scored as a match, a contradiction or inconclusive.

The predictions are stylized readings of each literature, not quotations. Where a theory is silent, or its
prediction depends on assumptions the model does not pin down, it makes no prediction (None).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .analysis import flex_profit_by_market, instability_index, ols, summarize_slopes
from .experiments import apply_param, evolution_runs, run_sweep
from .information import UnsupportedInformation
from .literature import THEORY_SOURCES, cite
from .params import Scenario


@dataclass(frozen=True)
class Theory:
    key: str
    name: str
    sources: str
    claim: str


THEORIES: List[Theory] = [
    Theory("heiner", "Heiner: reliability condition", cite(*THEORY_SOURCES["heiner"]),
           "Flexibility pays only if the agent's reliability r/w exceeds the tolerance limit (D/G)(1−π)/π. "
           "When the gap between difficulty and competence (the CD-gap) widens, rule-governed behavior wins, "
           "even when flexibility is free."),
    Theory("neo", "Neoclassical optimization", cite(*THEORY_SOURCES["neo"]),
           "Relaxing a constraint cannot lower an optimizer's payoff, so free flexibility never hurts. Errors "
           "are unsystematic, and better information makes flexibility more valuable."),
    Theory("options", "Real options / value of flexibility", cite(*THEORY_SOURCES["options"]),
           "Flexibility is an option whose value rises with uncertainty. With adjustment costs the zone of "
           "inaction widens as uncertainty grows (hysteresis)."),
    Theory("cobweb", "Cobweb stability theory", cite(*THEORY_SOURCES["cobweb"]),
           "Fast adjustment hurts only by destabilizing the market. Stability depends on adjustment speed, "
           "demand slope and the number of firms, not on additive shocks such as cost volatility or noise."),
    Theory("biasvar", "Bias–variance / ecological rationality", cite(*THEORY_SOURCES["biasvar"]),
           "Simple rules beat flexible ones when estimation noise is high relative to the signal. Only "
           "accuracy matters; the payoff asymmetry (stakes) does not enter."),
    Theory("satisficing", "Satisficing / aspiration-level search", cite(*THEORY_SOURCES["satisficing"]),
           "Firms change behavior when performance falls below aspiration. Worse or more volatile "
           "environments trigger more search and change."),
    Theory("ecology", "Structural inertia (organizational ecology)", cite(*THEORY_SOURCES["ecology"]),
           "Selection favors reliable, inert organizations in any environment, so rigidity is selected "
           "regardless of how volatile the environment is; change itself is hazardous."),
]
THEORY_NAMES = {t.key: t.name for t in THEORIES}


@dataclass
class Outcome:
    stat: float                 # the experiment's test statistic
    p: float                    # two-sided p-value of stat = 0
    observed: str               # '+', '-' or '0' at the 5% level
    summary: str                # one-line description of the result
    detail: pd.DataFrame        # per-condition table for display


@dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    manipulation: str
    statistic: str
    predictions: Dict[str, Optional[str]]   # theory key -> '+', '-', '0', '>=0' or None
    runner: Callable[..., Outcome]
    why: Dict[str, str]
    hid: str = ""                           # hypothesis in the literature registry that this experiment tests


def _sign(coef: float, p: float, alpha: float = 0.05) -> str:
    if p is None or np.isnan(p) or p >= alpha:
        return "0"
    return "+" if coef > 0 else "-"


def _slope_trend(firms: pd.DataFrame, x: str) -> Outcome:
    """Regress each market's profit~flexibility slope on the manipulated variable x."""
    per = pd.concat([flex_profit_by_market(g).assign(**{x: v}) for v, g in firms.groupby(x)]).dropna(subset=["slope"])
    tab, _ = ols(per["slope"].to_numpy(float), [per[x].to_numpy(float)], [x])
    coef, p = float(tab.loc[1, "coef"]), float(tab.loc[1, "p"])
    detail = pd.DataFrame([{x: v, **summarize_slopes(flex_profit_by_market(g))} for v, g in firms.groupby(x)])
    detail = detail[[x, "slope", "lo", "hi", "n"]]
    return Outcome(coef, p, _sign(coef, p), f"slope changes by {coef:+.3g} per unit (p = {p:.3g})", detail)


def _free(base: Scenario) -> Scenario:
    s = base.copy()
    s.firm_globals.flex_cost_slope = 0.0
    s.firm_globals.fixed_cost = 0.0
    return s


# ---------------------------------------------------------------------------------------------- runners
def run_free_harm(base: Scenario, reps: int, **_) -> Outcome:
    """Payoff to free flexibility in a low-profit market that cobweb theory calls stable."""
    s = _free(base)
    lever, level = ("desired_margin", 0.5) if s.firms[0].rule == "Bertrand" else ("c_max", 95.0)
    firms, _ = run_sweep(s, lever, [level], reps)
    sm = summarize_slopes(flex_profit_by_market(firms))
    rho = instability_index(apply_param(s.copy(), lever, level))
    obs = _sign(sm["slope"], sm["p"])
    detail = pd.DataFrame([dict(condition=f"{lever} = {level:g}, a = b = 0", slope=sm["slope"], lo=sm["lo"],
                                hi=sm["hi"], n=sm["n"], stability_radius=rho)])
    stable = "stable" if rho < 1 else "UNSTABLE"
    return Outcome(sm["slope"], sm["p"], obs, f"mean slope {sm['slope']:+.3g} (95% CI {sm['lo']:.3g} to "
                   f"{sm['hi']:.3g}); cobweb spectral radius {rho:.2f} ({stable})", detail)


def run_volatility(base: Scenario, reps: int, **_) -> Outcome:
    firms, _ = run_sweep(base, "delta", [2, 8, 16, 24, 32], reps)
    return _slope_trend(firms, "delta")


def run_competence(base: Scenario, reps: int, **_) -> Outcome:
    firms, _ = run_sweep(base, "foresight", [0, 0.25, 0.5, 0.75, 1.0], reps)
    return _slope_trend(firms, "foresight")


def run_noise(base: Scenario, reps: int, **_) -> Outcome:
    firms, _ = run_sweep(base, "noise", [0, 5, 10, 15, 20], reps)
    return _slope_trend(firms, "noise")


def run_stakes(base: Scenario, reps: int, **_) -> Outcome:
    s = base.copy()
    for f in s.firms:
        f.rule = "Bertrand"
    s.firm_globals.margin_includes_fixed = True
    s.firm_globals.flex_cost_slope = 0.0
    firms, _ = run_sweep(s, "fixed_cost", [0, 500, 1000, 1500], reps)
    return _slope_trend(firms, "fixed_cost")


def run_rivals(base: Scenario, reps: int, **_) -> Outcome:
    firms, _ = run_sweep(base, "n_firms", [2, 4, 6, 8, 12], reps)
    return _slope_trend(firms, "n_firms")


def run_predictability(base: Scenario, reps: int, horizon: int = 20, **_) -> Outcome:
    """Reliability-learning (Adaptive) agents: does their deviation rate fall as volatility rises?"""
    s = base.copy()
    for f in s.firms:
        f.selection = "Adaptive"
    firms, _ = run_sweep(s, "delta", [2, 8, 16, 24, 32], reps, horizon=max(2, int(horizon)))
    firms = firms[firms["opportunities"] > 0].copy()
    firms["dev_rate"] = firms["deviations"] / firms["opportunities"]
    per = firms.groupby(["market", "delta"])["dev_rate"].mean().reset_index()
    tab, _ = ols(per["dev_rate"].to_numpy(float), [per["delta"].to_numpy(float)], ["delta"])
    coef, p = float(tab.loc[1, "coef"]), float(tab.loc[1, "p"])
    detail = per.groupby("delta")["dev_rate"].agg(["mean", "std", "count"]).reset_index()
    return Outcome(coef, p, _sign(coef, p), f"deviation rate changes by {coef:+.4f} per unit of Δ (p = {p:.3g})",
                   detail.rename(columns={"mean": "deviation_rate", "std": "sd", "count": "n"}))


def run_partial_adjustment(base: Scenario, reps: int, **_) -> Outcome:
    """Symmetric Cournot industry: does the profit-maximizing adjustment speed fall as errors grow?"""
    s = base.copy()
    for f in s.firms:
        f.rule, f.selection = "Cournot", "Always"
    phis = list(np.round(np.linspace(0.05, 1.0, 12), 3))
    levels = [0, 5, 10, 20]
    firms, _ = run_sweep(s, "noise", levels, reps, "common_flex", phis)
    ind = firms.groupby(["noise", "rep", "common_flex"])["avg_profit"].mean().reset_index()
    best = ind.loc[ind.groupby(["noise", "rep"])["avg_profit"].idxmax()]
    tab, _ = ols(best["common_flex"].to_numpy(float), [best["noise"].to_numpy(float)], ["noise"])
    coef, p = float(tab.loc[1, "coef"]), float(tab.loc[1, "p"])
    detail = best.groupby("noise")["common_flex"].agg(["mean", "std", "count"]).reset_index()
    return Outcome(coef, p, _sign(coef, p), f"best common φ changes by {coef:+.4f} per unit of σ (p = {p:.3g})",
                   detail.rename(columns={"mean": "best_phi", "std": "sd", "count": "n"}))


def run_evolution(base: Scenario, reps: int, **_) -> Outcome:
    s = base.copy(periods=3000)
    s.evolution.flex_max = 2.0 if s.firms[0].rule == "Bertrand" else 1.0
    _, summ = evolution_runs(s, [3, 10, 20, 30], max(4, reps // 2))
    tab, _ = ols(summ["flex_end"].to_numpy(float), [summ["delta"].to_numpy(float)], ["delta"])
    coef, p = float(tab.loc[1, "coef"]), float(tab.loc[1, "p"])
    detail = summ.groupby("delta")[["flex_start", "flex_end"]].mean().reset_index()
    return Outcome(coef, p, _sign(coef, p), f"evolved φ changes by {coef:+.4f} per unit of Δ (p = {p:.3g})", detail)


EXPERIMENTS: List[Experiment] = [
    Experiment("free_harm", "E1 · Free flexibility in a stable, low-profit market",
               "Flexibility costs a = b = 0; low profitability (margin-feedback rule with m* = 0.5, or Cournot max cost 95).",
               "Mean within-market slope of profit on φ",
               dict(heiner="-", neo=">=0", options=">=0", cobweb=">=0", biasvar="-", satisficing=None, ecology="-"),
               run_free_harm,
               dict(heiner="Errors dominate when mistakes are costly: rigid firms win.",
                    neo="Relaxing a constraint cannot hurt.", options="An option is never worth less than zero.",
                    cobweb="Flexibility can only hurt in an unstable market; this one is stable.",
                    biasvar="Firms decide on last period's cost, so estimates are noisy: simple rules win.",
                    ecology="Inert organizations are selected because they are reliable."), hid="H1"),
    Experiment("volatility", "E2 · Cost volatility Δ rises",
               "Δ = 2, 8, 16, 24, 32 (sidebar market otherwise).", "Coefficient of the slope on Δ",
               dict(heiner="-", neo="+", options="+", cobweb="0", biasvar="-", satisficing=None, ecology="0"),
               run_volatility,
               dict(heiner="Difficulty rises, the CD-gap widens and reliability falls.",
                    neo="More change to respond to raises the value of responding.",
                    options="Option value rises with uncertainty.",
                    cobweb="Additive shocks do not change the stability of a linear cobweb.",
                    biasvar="Noisier environment: the variance of flexible rules grows.",
                    ecology="Inertia is favored in every environment; volatility does not change the ranking."), hid="H3"),
    Experiment("competence", "E3 · Competence κ (cost foresight) rises",
               "κ = 0, 0.25, 0.5, 0.75, 1.", "Coefficient of the slope on κ",
               dict(heiner="+", neo="+", options="-", cobweb="0", biasvar="+", satisficing=None, ecology="0"),
               run_competence,
               dict(heiner="Competence closes the CD-gap: r rises, w falls.",
                    neo="Better information complements flexibility.",
                    options="Less uncertainty about the future lowers the option value of flexibility.",
                    cobweb="Foresight shifts perceived cost but not the adjustment dynamics.",
                    biasvar="Lower estimation error favors the flexible rule.",
                    ecology="Selection acts on organizational inertia, not on individual competence."), hid="H7"),
    Experiment("noise", "E4 · Perception noise σ rises",
               "σ = 0, 5, 10, 15, 20.", "Coefficient of the slope on σ",
               dict(heiner="-", neo=None, options="+", cobweb="0", biasvar="-", satisficing=None, ecology="0"),
               run_noise,
               dict(heiner="Noise lowers competence: the CD-gap widens.",
                    options="More uncertainty raises option value.",
                    cobweb="Additive noise does not change stability.",
                    biasvar="Estimation variance punishes flexible rules.",
                    ecology="Inertia is favored whatever the noise; no gradient."), hid="H8"),
    Experiment("stakes", "E5 · Fixed costs raise the stakes (margin-feedback rule, margin includes F/q)",
               "b = 0, 500, 1000, 1500; marginal costs unchanged.", "Coefficient of the slope on b",
               dict(heiner="-", neo="0", options=None, cobweb=None, biasvar="0", satisficing=None, ecology="0"),
               run_stakes,
               dict(heiner="Gains shrink and losses grow, so the tolerance limit rises.",
                    neo="Decisions are made at the margin; fixed costs are sunk.",
                    biasvar="Accuracy is unchanged, so rules keep their ranking.",
                    ecology="Inertia is favored at every level of fixed cost."), hid="H4"),
    Experiment("rivals", "E6 · More rivals (strategic difficulty)",
               "n = 2, 4, 6, 8, 12 firms, φ spread over the same range.", "Coefficient of the slope on n",
               dict(heiner="-", neo=None, options=None, cobweb="-", biasvar=None, satisficing=None, ecology="0"),
               run_rivals,
               dict(heiner="The target moves more (f′ = −(n−1)/2), so the reliable speed β₀ falls.",
                    cobweb="More firms shrink the stability region (φ < 4/(n+1)).",
                    ecology="Inertia is favored whatever the number of competitors."), hid="H11"),
    Experiment("predictability", "E7 · Do reliability-learning agents become more predictable as Δ rises?",
               "Every firm uses the Adaptive selection rule; Δ = 2, 8, 16, 24, 32.",
               "Coefficient of the deviation rate on Δ",
               dict(heiner="-", neo="+", options="-", cobweb=None, biasvar=None, satisficing="+", ecology="-"),
               run_predictability,
               dict(heiner="Heiner's core 1983 claim: greater uncertainty makes behavior more rule-governed.",
                    neo="Bigger shocks make re-optimizing worthwhile more often.",
                    options="The zone of inaction widens with uncertainty (Dixit 1989).",
                    satisficing="Volatility pushes results below aspiration more often, triggering change.",
                    ecology="Change is hazardous, and more so in volatile environments: organizations become more inert."), hid="H10"),
    Experiment("partial", "E8 · Best adjustment speed as errors grow (symmetric Cournot)",
               "Every firm uses the same φ from 0.05 to 1; perception noise σ = 0, 5, 10, 20.",
               "Coefficient of the profit-maximizing φ on σ",
               dict(heiner="-", neo="-", options="+", cobweb="0", biasvar="-", satisficing=None, ecology="0"),
               run_partial_adjustment,
               dict(heiner="Heiner (1989) Theorem 2: β₀ = 1/((1+K)(1−f′)) falls as K rises.",
                    neo="An optimizer that accounts for noise attenuates its response (optimal filtering; "
                        "Muth 1960, Kalman 1960, Brainard 1967). This prediction is shared with Heiner.",
                    options="More uncertainty favors keeping and using flexibility.",
                    cobweb="The best speed is set by the stability limit, which noise does not move.",
                    biasvar="Shrink estimates toward the default as noise rises.",
                    ecology="Inertia is favored at every noise level: no gradient."), hid="BOUND"),
    Experiment("evolution", "E9 · Which flexibility do industries evolve at different volatilities?",
               "Firms imitate the most profitable rival's φ (plus mutation) for 3,000 periods; Δ = 3, 10, 20, 30.",
               "Coefficient of evolved φ on Δ",
               dict(heiner="-", neo="+", options="+", cobweb="0", biasvar=None, satisficing=None, ecology="0"),
               run_evolution,
               dict(heiner="Volatile industries evolve toward rigidity.",
                    neo="Volatility raises the value of flexibility, so it is selected.",
                    options="Option value rises with volatility.",
                    cobweb="Selection on stability does not depend on additive shocks.",
                    ecology="Inertia is selected everywhere; no volatility gradient."), hid="EVO"),
]
EXPERIMENT_BY_KEY = {e.key: e for e in EXPERIMENTS}


def score(prediction: Optional[str], observed: Optional[str]) -> Optional[str]:
    """'match', 'contradicted', 'inconclusive' (directional prediction, no significant effect) or None (no prediction,
    or the experiment was not run because the information specification does not support it)."""
    if prediction is None or observed is None:
        return None
    if prediction == "0":
        return "match" if observed == "0" else "contradicted"
    if prediction == ">=0":
        return "contradicted" if observed == "-" else "match"
    if observed == "0":
        return "inconclusive"
    return "match" if observed == prediction else "contradicted"


def scoreboard(outcomes: Dict[str, Outcome]) -> pd.DataFrame:
    """One row per theory: matches, contradictions, inconclusive tests and the share of its predictions that
    the simulation bears out."""
    rows = []
    for t in THEORIES:
        res = [score(EXPERIMENT_BY_KEY[k].predictions.get(t.key), o.observed) for k, o in outcomes.items()]
        made = [r for r in res if r is not None]
        m, c, i = made.count("match"), made.count("contradicted"), made.count("inconclusive")
        rows.append(dict(theory=t.name, predictions=len(made), matches=m, contradicted=c, inconclusive=i,
                         net=m - c, hit_rate=m / len(made) if made else np.nan))
    # ties on net score go to the theory that made, and passed, more predictions (it risked more)
    return pd.DataFrame(rows).sort_values(["net", "matches", "hit_rate"], ascending=False, ignore_index=True)


def run_tournament(base: Scenario, reps: int, horizon: int = 20,
                   progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Outcome]:
    out = {}
    for k, e in enumerate(EXPERIMENTS):
        if progress:
            progress(k / len(EXPERIMENTS), e.title)
        try:
            out[e.key] = e.runner(base, reps, horizon=horizon)
        except UnsupportedInformation as err:      # disable the comparison; never run it with extra information
            out[e.key] = Outcome(float("nan"), float("nan"), None, f"not run: {err}", pd.DataFrame())
    if progress:
        progress(1.0, "done")
    return out
