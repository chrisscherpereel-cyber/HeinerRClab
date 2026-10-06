"""Model specification: every agent's objective, information, actions, feedback, assumptions and limits.

One row per agent the laboratory runs. Composite tournament designs (target x selection rule) are assembled from the
specification of their target and of their selection rule, so the table follows the code. tests/test_model_spec.py
checks that every production rule, selection rule, tournament design and task layer has a row.

"Objective" is what the rule is built to pursue; none of these agents solves an intertemporal optimization problem.
Timing follows heiner_abm.agents.DECISION_SCHEDULE: decisions use information up to the previous period, and learning
uses only feedback whose periods have occurred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .terminology import INACTION_BAND_LABEL

COLUMNS = ("Group", "Agent", "Key", "Objective", "Information", "Actions", "Feedback", "Assumptions", "Limits")

# What every simulated firm can see when it decides in period t.
_MARKET_INFO = ("last period's price and market output, its own output and profit, its cost estimate "
                "ĉ = c[t−1] + κ(c[t] − c[t−1]) + σε (κ: cost foresight, a competence parameter)")


@dataclass(frozen=True)
class Spec:
    group: str
    agent: str
    key: str
    objective: str
    information: str
    actions: str
    feedback: str
    assumptions: str
    limits: str

    def row(self) -> Dict[str, str]:
        return dict(zip(COLUMNS, (self.group, self.agent, self.key, self.objective, self.information, self.actions,
                                  self.feedback, self.assumptions, self.limits)))


# ------------------------------------------------------------------------------------------------ market lab
MARKET_LAB: List[Spec] = [
    Spec("Market lab · production rule", "Margin-feedback quantity rule", "Bertrand",
         "Keep the realized margin near a desired margin m*",
         _MARKET_INFO + "; no demand model",
         "Recommends output q* = q + φ(P − ĉ − m*), rounded, at least q_min",
         "None of its own (the recommendation has no memory)",
         "Firms set quantities; one market-clearing price; m* fixed",
         "Not price-setting Bertrand competition; ignores rivals' reactions and the demand slope"),
    Spec("Market lab · production rule", "Cournot best-reply quantity rule", "Cournot",
         "Move toward the static profit-maximizing reply to rivals' last output",
         _MARKET_INFO + "; its believed demand curve (lags the true curve by L periods after a shift)",
         "Recommends q* = q + φ(best reply − q), rounded, at least q_min",
         "None of its own",
         "Rivals' output stays at last period's level; linear demand",
         "Myopic: no anticipation of rivals' adjustment; misspecified after unannounced shifts"),
    Spec("Market lab · selection rule", "Always", "Always",
         "Act on every recommendation (maximal flexibility)", "The recommendation", "Adopt q*", "None",
         "—", "Inherits every error of the production rule"),
    Spec("Market lab · selection rule", "Never (rule B)", "Never",
         "Hold output fixed (maximal rigidity)", "None", "Keep q", "None", "—",
         "Cannot respond to any change; used as the rigid twin"),
    Spec("Market lab · selection rule", "Small (SR1)", "Small",
         "Act only on small recommended changes", "Size of the recommended change", "Adopt q* if |q* − q| < θ",
         "None", "Large changes are risky", "Fixed threshold θ; no learning"),
    Spec("Market lab · selection rule", "Large (SR2)", "Large",
         "Act only on large, clear signals", "Size of the recommended change", "Adopt q* if |q* − q| > θ",
         "None", "Small changes are mostly noise", "Fixed threshold θ; no learning"),
    Spec("Market lab · selection rule", "Adaptive (learned reliability)", "Adaptive",
         "Deviate from rule B only for sizes of change where deviating has paid",
         "Its learned table of gains per size bin (5 bins)",
         "Adopt q* if the learned gain for that size is ≥ the adjustment cost c (0 by default; estimated-gain gate)",
         "Estimated counterfactual (default): released W periods after each decision (plus any observation delay); "
         "the gain of holding q* vs q, with rivals' observed output, realized costs and its believed demand curve. "
         "Oracle treatment (researcher-only): the forked counterfactual, released after H periods. Not run under "
         "chosen-action feedback (it needs a counterfactual)",
         "Exponential memory λ; judgement window W",
         "Judges with a possibly misspecified demand model; learning is noisy when the environment shifts"),
    Spec("Market lab · selection gate (proposed extension)", "Adaptive · confidence-sensitive gate", "Adaptive:lcb",
         "Adopt q* only where the advantage over rule B reliably exceeds the adjustment cost c",
         "Weighted mean, standard error and effective number of feedback items per size bin",
         "Adopt q* if n_eff ≥ minimum evidence and mean − t·s.e. > c; otherwise keep q",
         "As the Adaptive rule: estimated counterfactual released W periods after each decision (oracle treatment "
         "only if chosen); not run under chosen-action feedback",
         "Student-t bound with Kish effective sample size; nominal only for independent feedback with a common mean",
         "Overlapping judgement windows, drift and shifts make the bound optimistic; coverage is measured, not assumed"),
    Spec("Market lab · selection gate (proposed extension)", "Adaptive · exploration-enabled gate", "Adaptive:explore",
         "Learn whether adopting pays from its own payoffs, without counterfactual feedback",
         "Its own payoffs after randomized trials per size bin (arm adopt / arm keep)",
         "Adopt if the lower bound > c, keep if the upper bound ≤ c; otherwise a randomized trial with probability "
         "explore_rate (adopt or keep, 1/2 each), else keep q",
         "Chosen-action: own realized payoff over W periods minus its pre-decision payoff at the realized cost, "
         "released W periods later (plus any delay); only randomized trials are evidence",
         "Randomized trials (separate random stream); Welch–Satterthwaite interval for the difference of arms",
         "Trials cost payoff; stops randomizing once confident (optional stopping); with W > 1 its target includes its "
         "own later decisions"),
    Spec("Market lab · selection gate (benchmark)", "Adaptive · ORACLE benchmark gate", "Adaptive:oracle_table",
         "Adopt q* where the true bin advantage exceeds c", "ORACLE: true mean advantage per firm and size bin, "
         "estimated by the researcher's counterfactuals in independent runs", "Adopt q* if the table value > c",
         "None (does not learn)", "ORACLE table accepted by no other gate",
         "Decides per bin, so it is a benchmark, not an upper bound"),
    Spec("Market lab · evolution", "Imitation of flexibility", "evolution",
         "Copy the flexibility φ of the most profitable firm", "Every firm's realized profit over the last window",
         "Set φ (with probability p) plus mutation", "Realized window profits",
         "Profits are observable across firms", "Selects on realized profit, which is noisy"),
]

# ------------------------------------------------------------------------------------------------ tournament
_TOURNAMENT_INFO = _MARKET_INFO + "; believed demand curve (lag L after a shift)"
TARGETS: Dict[str, Spec] = {
    "model": Spec("", "", "", "Best reply on the believed demand curve",
                  _TOURNAMENT_INFO + "; filtered forecasts of cost and rivals' output",
                  "Move a share φ toward the target", "", "Linear believed demand; rivals as forecast",
                  "Misspecified after unannounced shifts until the model updates"),
    "price": Spec("", "", "", "Output that is best at an adaptive expectation of the price",
                  _TOURNAMENT_INFO + "; uses the believed slope only, no demand level",
                  "Move a share φ toward the target", "", "Adaptive price expectation (gain λ)",
                  "Ignores how its own output moves the price except through the slope"),
}
SELECTIONS: Dict[str, Spec] = {
    "always": Spec("", "", "", "", "", "every period", "None", "", ""),
    "band": Spec("", "", "", "", "Volatility of recent target changes",
                 "only when the gap exceeds k standard deviations", "None (the band follows measured volatility)",
                 INACTION_BAND_LABEL + "; no adjustment cost or option valuation", "Band width k is tuned, not derived"),
    "rc": Spec("", "", "", "", "Its learned gains per size bin",
               "only where the learned gain for that size of change is ≥ 0",
               "Gain of holding the candidate vs the old output over its h periods, released h periods after the "
               "decision, priced on its believed demand curve",
               "Exponential memory; h-period judgement", "Estimates are noisy and, after shifts, biased by the model"),
    "aspiration": Spec("", "", "", "", "Its own last profit and an adaptive aspiration",
                       "only when profit falls below the aspiration", "Own realized profit",
                       "Aspiration adapts at rate α", "Search is triggered by outcomes, not by expected gains"),
}
VARIANT_NOTES = {
    "true": ("Researcher variant: judges past decisions with the TRUE demand curve (removes model bias); mechanism "
             "study only, never in tournament lineups"),
    "oracle": ("ORACLE variant: does not learn; deviates by the true expected gain per size bin, estimated from long "
               "independent runs of the same environment; mechanism study only, never in tournament lineups"),
}

HAND_WRITTEN: Dict[str, Spec] = {
    "opt_nash": Spec("", "", "", "Static Cournot–Nash output for its cost estimate",
                     _TOURNAMENT_INFO + "; number of firms", "Move a share φ toward the Nash output", "None",
                     "Every rival is rational and plays Nash", "Wrong when rivals are not rational"),
    "heur_wsls": Spec("", "", "", "Keep moving in a direction while profit rises", "Own last profit change",
                      "Change output by a fixed step, reversing after a fall", "Own realized profit", "Fixed step size",
                      "Ignores costs and demand; oscillates around a target"),
    "heur_markup": Spec("", "", "", "Keep the observed margin near a target margin m", _MARKET_INFO,
                        "Raise output if P − ĉ > m, lower it otherwise (step φ)", "None", "Target margin m is tuned",
                        "Margin feedback with a tuned target; no demand model"),
    "rl_softmax": Spec("", "", "", "Choose the move with the highest learned value", "Own last profit change",
                       "Five moves (−2 … +2 steps), chosen by softmax", "Profit change after each move",
                       "Learning rate η; temperature", "Learns values of moves, not of states"),
    "rl_erevroth": Spec("", "", "", "Choose moves in proportion to propensities", "Own last profit change",
                        "Five moves (−2 … +2 steps)", "Profit change after each move; forgetting",
                        "Erev–Roth reinforcement", "Slow in changing environments"),
    "imit_best": Spec("", "", "", "Do what the most profitable firm did", "Every firm's last output and profit",
                      "Copy the best firm's output with probability p, plus noise", "Rivals' realized profits",
                      "Rivals' outputs and profits are observable", "Can over-produce relative to Cournot–Nash"),
    "imit_avg": Spec("", "", "", "Conform to the average firm", "Every firm's last output",
                     "Move to the others' average output with probability p, plus noise", "None",
                     "Rivals' outputs are observable", "Ignores profitability"),
    "ruleb": Spec("", "", "", "Hold output fixed", "None", "Keep last period's output", "None", "—",
                  "Benchmark only"),
}


def tournament_specs() -> List[Spec]:
    from .arena import ALL_DESIGNS, Composite
    out = []
    for key, d in ALL_DESIGNS.items():
        if issubclass(d, Composite):
            tg, se = TARGETS[d.TARGET], SELECTIONS[d.SELECT]
            variant = VARIANT_NOTES["oracle"] if d.ORACLE else VARIANT_NOTES["true"] if d.GAIN == "true" else ""
            feedback = "Oracle gains (fixed parameters)" if d.ORACLE else se.feedback
            if d.GAIN == "true":
                feedback = feedback.replace("its believed demand curve", "the TRUE demand curve")
            out.append(Spec("Agent tournament" + (" · researcher variant" if variant else ""), d.name, key,
                            tg.objective, tg.information + ("; " + se.information if se.information else ""),
                            f"{tg.actions}, {se.actions}", feedback,
                            "; ".join(x for x in (tg.assumptions, se.assumptions) if x),
                            "; ".join(x for x in (tg.limits, se.limits, variant) if x)))
        else:
            h = HAND_WRITTEN[key]
            out.append(Spec("Agent tournament", d.name, key, h.objective, h.information, h.actions, h.feedback,
                            h.assumptions, h.limits))
    return out


# ------------------------------------------------------------------------------------------------ other studies
OTHER: List[Spec] = [
    Spec("Rule choice", "Rule-choosing firm", "rulechoice",
         "Use the rule that has recently earned most", "Recent average profit of each rule's current users",
         "Switch rule each period by logit choice (intensity β); the chosen rule then sets output",
         "Realized profits of each rule's users, after clearing", "Rule performance is observable across firms",
         "Rules are the tuned tournament designs; choice ignores why a rule did well"),
    Spec("Generalization tasks", "Always deviate (flexible rule)", "always",
         "Act on the fast estimate", "Observed outcomes so far", "Take the flexible action", "None", "—",
         "Inherits the fast estimate's errors"),
    Spec("Generalization tasks", "Inaction band", "band",
         "Act only on strong signals", "Signal strength (gap relative to dispersion)",
         "Deviate when the signal exceeds the band", "None", INACTION_BAND_LABEL, "Band width tuned"),
    Spec("Generalization tasks", "RC, learned", "rc_learned",
         "Deviate where deviating has paid", "Running mean gain per signal bin",
         "Deviate if the bin's mean gain > 0 (always for the first five)",
         "Observed gain of the flexible over the default action in the decision period, after its outcome",
         "No cost of deviating", "Short histories make bins noisy"),
    Spec("Generalization tasks", "RC, oracle", "rc_oracle",
         "Deviate where deviating truly pays", "True mean gain per bin from a long independent run",
         "Deviate if the true gain > 0", "None (oracle values)", "ORACLE: researcher-only knowledge",
         "Benchmark of the principle, not a feasible agent"),
    Spec("Generalization tasks", "Rule B (default)", "ruleb",
         "Follow the slow, long-memory estimate", "Observed outcomes so far", "Take the default action", "None",
         "—", "Benchmark only"),
    Spec("Solvable benchmark", "Partial adjustment / Kalman filter and restricted variants", "tracking",
         "Track a random walk observed with noise (squared loss, possibly lopsided)",
         "The current noisy observation and the filter's estimate",
         "Move a share of the gap, possibly only beyond thresholds or at direction-specific speeds", "None (rules "
         "are tuned on training paths)", "Known q and r for the Kalman gain", "Single agent; no strategic interaction"),
    Spec("Human experiment", "Participant", "human",
         "Earn as much as possible in the experiment's market", "The screen: last price, market output, own "
         "output and profit, cost estimate", "Choose output each period", "Own realized outcomes",
         "Three counterbalanced uncertainty blocks", "No data collected yet; protocol only"),
    Spec("Calibration and empirical validation", "Rules fitted to human data", "calibration",
         "Predict a person's next forecast, quantity or order", "That person's own past choices and the market "
         "information of the experiment", "One prediction per period", "Parameters fitted on the first half only",
         "Grid-search fit per person", "Predictive fit, not a causal model of the person"),
]


_BENCH_ACTIONS = {"inventory": "Order quantity S each period (0.8 critical fractile of its demand forecast)",
                  "bandit": "Choose one of two options each period"}
_BENCH_OBJECTIVE = {"competitor": "Maximize payoff with the design's information",
                    "benchmark": "Bayes-optimal payoff under the true model (researcher benchmark, not ranked)",
                    "oracle": "ORACLE: best action given the true state (not ranked)",
                    "reference": "Full-feedback reference for the bandit (not ranked; different feedback)"}


def benchmark_specs() -> List[Spec]:
    """Rows for the decision benchmarks, generated from their family registries."""
    from . import bench_bandit, bench_inventory
    rows = []
    for name, mod in (("inventory", bench_inventory), ("bandit", bench_bandit)):
        for k, f in mod.FAMILIES.items():
            tuned = ", ".join(f.space) or "none"
            rows.append(Spec(f"Decision benchmark · {name}", f.name, mod.REGISTRY_KEY[k], _BENCH_OBJECTIVE[f.role],
                             f.feedback, _BENCH_ACTIONS[name],
                             {"inventory": "Observed demand after each period (uncensored)",
                              "bandit": "Payoff of the chosen option only"}[name] if f.role == "competitor" else
                             f.feedback, f.assumptions + f" Hyperparameters tuned on training environments: {tuned}.",
                             f"Compute: {f.compute}" + (" ORACLE information." if f.role in ("benchmark", "oracle")
                                                         else "")))
    return rows


def all_specs() -> List[Spec]:
    return MARKET_LAB + tournament_specs() + OTHER + benchmark_specs()


def table():
    import pandas as pd
    return pd.DataFrame([s.row() for s in all_specs()], columns=list(COLUMNS))
