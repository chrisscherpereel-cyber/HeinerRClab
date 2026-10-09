"""Environment adapters: one interface over the simulation engines the laboratory already has.

Each adapter states its controls (grouped by the uncertainty panel), its candidate-generation rules, its policies
(selection policies that act on a common candidate, complete policies with their own assumptions, and researcher
benchmarks that are never ranked), its outcomes, and how to run one cell of an experiment and record a decision trace.
The adapters call the existing engines; they do not reimplement environments:
    market       heiner_abm.engine.run_batch (the shared market; every firm uses the compared selection policy)
    inventory    heiner_abm.learnability (factor-controlled newsvendor with feedback availability and default quality)
    learning,
    investment   heiner_abm.tasks (generalization tasks)
    nk           heiner_abm.nk (NK landscapes)
The one exception is the decision trace of the generalization tasks' learned reliability condition: tasks._layer
returns only summaries, so the trace recomputes its documented recursion; a test checks that the recomputed decisions
equal tasks._layer's.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Cancel = Callable[[], bool]


@dataclass(frozen=True)
class Control:
    key: str
    label: str
    group: str                    # an uncertainty group, "agents" (policy settings) or "environment"
    kind: str                     # "float", "int", "choice", "bool"
    default: Any
    lo: Any = None
    hi: Any = None
    step: Any = None
    options: Tuple[Any, ...] = ()
    help: str = ""
    advanced: bool = False
    treatable: bool = True        # can be a manipulated treatment


@dataclass(frozen=True)
class PolicyInfo:
    key: str
    label: str
    kind: str                     # "selection" (acts on the common candidate), "complete", "benchmark"
    description: str
    uses: str                     # what it reads (information)
    assumptions: str = ""
    extension: bool = False       # a proposed extension of this laboratory


@dataclass(frozen=True)
class Outcome:
    key: str
    label: str
    higher_better: Optional[bool]  # None: descriptive
    tab: str                       # "performance", "behavior", "mechanisms"


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    description: str
    params: Tuple[Control, ...] = ()


class Environment:
    key = ""
    label = ""
    description = ""
    engine = ""                    # heiner_abm.information.ENGINES key
    candidates: Tuple[Candidate, ...] = ()
    controls: Tuple[Control, ...] = ()
    not_applicable: Dict[str, str] = {}
    policies: Tuple[PolicyInfo, ...] = ()
    outcomes: Tuple[Outcome, ...] = ()
    primary: str = ""
    seconds_per_unit = 1e-4        # rough wall time per policy x replication x period (workload estimate)
    tuning_note = "No tuning: the documented fixed parameters are used."
    cluster_note = "Replications are independent environments; intervals resample them."

    # ---------------------------------------------------------------------------------------- helpers
    def control(self, key: str) -> Control:
        for c in self.all_controls():
            if c.key == key:
                return c
        raise KeyError(key)

    def all_controls(self) -> Tuple[Control, ...]:
        return self.controls + tuple(p for c in self.candidates for p in c.params)

    def policy(self, key: str) -> PolicyInfo:
        return next(p for p in self.policies if p.key == key)

    def defaults(self) -> Dict[str, Any]:
        return {c.key: c.default for c in self.controls}

    def params(self, spec, level: Any = None) -> Dict[str, Any]:
        p = {**self.defaults(), **{k: v for k, v in spec.env_params.items() if k in self.defaults()}}
        cand = self.candidate_for(spec)
        p.update({c.key: c.default for c in cand.params})
        p.update({k: v for k, v in spec.candidate_params.items() if k in {c.key for c in cand.params}})
        p.update(spec.policy_params)
        if spec.design.treatment.control and level is not None:
            p[spec.design.treatment.control] = level
        return p

    def candidate_for(self, spec) -> Candidate:
        return next((c for c in self.candidates if c.key == spec.candidate), self.candidates[0])

    def outcome(self, key: str) -> Outcome:
        return next(o for o in self.outcomes if o.key == key)

    # ---------------------------------------------------------------------------------------- interface
    def validate(self, spec) -> List[Tuple[str, str, str]]:
        """Environment-specific issues: (step, level, message)."""
        return []

    def information(self, spec) -> Dict[str, Any]:
        from ..information import ENGINES
        return ENGINES[self.engine].native.to_dict()

    def tune(self, spec, params, seeds: Sequence[int], cancel: Cancel) -> Dict[str, Any]:
        return {}

    def run_cell(self, spec, params, seeds: Sequence[int], hp: Dict[str, Any], cancel: Cancel,
                 trace: bool) -> Tuple[List[Dict[str, Any]], Dict[str, pd.DataFrame]]:
        raise NotImplementedError


def _num(c: Control, v):
    return int(v) if c.kind == "int" else float(v) if c.kind == "float" else v


# ================================================================================================ inventory
class Inventory(Environment):
    key = "inventory"
    label = "Inventory (newsvendor with shifting demand)"
    description = ("Each period the agent orders before demand is known (shortage costs 4, leftover stock 1 per unit). "
                   "Mean demand changes without announcement; demand may be observed with error or not at all. "
                   "A default order is compared with a forecast-based target.")
    engine = "learnability"
    candidates = (Candidate("fast_forecast", "Forecast-based target",
                            "Order the 0.8 critical fractile of a fast exponentially weighted demand forecast.",
                            (Control("gain", "Forecast gain g (reaction to each observation)", "agents", "float", 0.1,
                                     0.01, 0.8, 0.01, advanced=True),)),)
    controls = (
        Control("sigma", "Demand volatility (s.d. around the mean)", "outcome", "float", 20.0, 0.0, 60.0, 1.0),
        Control("tau", "Observation error (s.d.)", "observation", "float", 10.0, 0.0, 40.0, 1.0),
        Control("avail", "Share of periods whose demand is observed", "observation", "float", 1.0, 0.05, 1.0, 0.05,
                help="Unobserved periods give no feedback (missing observations)."),
        Control("hazard", "Change frequency (per period)", "change", "float", 0.01, 0.0, 0.1, 0.001),
        Control("family", "Change process", "change", "choice", "jump", options=("jump", "switch", "drift"),
                help="jump: mean jumps by N(0, 30²); switch: alternates between 70 and 130; drift: random walk.",
                advanced=True),
        Control("default", "Default order", "agents", "choice", "fixed_initial",
                options=("slow", "fixed_initial", "biased", "dominated"),
                help="slow: slow forecast; fixed_initial: best order for the first regime, deteriorates after a "
                     "change; biased: 40 units low; dominated: orders nothing."),
        Control("cost", "Adaptation cost (per period that departs from the default)", "agents", "float", 1.0, 0.0,
                20.0, 0.5),
        Control("memory", "Gates' memory λ", "agents", "float", 0.97, 0.8, 0.999, 0.005, advanced=True),
        Control("band_b", "Inaction band width b (in s.d.)", "agents", "float", 0.5, 0.0, 4.0, 0.25, advanced=True),
        Control("gate_conf", "Confidence gate: confidence", "agents", "float", 0.9, 0.5, 0.995, 0.005, advanced=True,
                help="One-sided confidence of the lower bound the confidence-sensitive gate requires."),
        Control("gate_nmin", "Confidence gate: minimum evidence", "agents", "float", 5.0, 2.0, 50.0, 1.0,
                advanced=True, help="Effective feedback items a size bin needs before the gate will adapt."),
    )
    not_applicable = {"model": "No demand model: every policy forecasts from observed demand only.",
                      "complexity": "One order per period; no interactions or search budget."}
    policies = (
        PolicyInfo("always", "Always adapt", "selection", "Order the forecast-based target every period.",
                   "observed demands"),
        PolicyInfo("default", "Retain the default", "selection", "Order the default every period.", "its default"),
        PolicyInfo("band", "Inaction band", "selection", "Adapt only if the target differs from the default by more "
                   "than b estimated s.d.", "observed demands"),
        PolicyInfo("gate_gain", "Estimated-gain gate", "selection", "Adapt where the learned mean gain of adapting "
                   "(per size bin) covers the cost.", "observed gains per size bin", extension=False),
        PolicyInfo("gate_lcb", "Confidence-sensitive gate", "selection", "Adapt only with enough evidence and a lower "
                   "confidence bound on the gain above the cost.", "observed gains, their uncertainty and count",
                   extension=True),
        PolicyInfo("bocpd", "Bayesian change detection", "complete", "Orders from a Bayesian online change-point "
                   "posterior predictive (Adams & MacKay 2007).", "observed demands",
                   "Gaussian regimes with a constant hazard (misspecified for switch and drift families); "
                   "hazard 0.005, prior s.d. 30, observation s.d. 25."),
        PolicyInfo("dro", "Distributionally robust order", "complete", "Worst-case order over a modified-χ² ball "
                   "around the last N observed demands (Ben-Tal et al. 2013).", "the last N observed demands",
                   "Window N = 20, radius ρ = 0.01; treats the window as exchangeable."),
        PolicyInfo("oracle", "ORACLE (perfect information)", "benchmark", "Orders the true critical fractile.",
                   "the true mean demand (researcher-only)"),
    )
    outcomes = (
        Outcome("net_payoff", "Net payoff per period", True, "performance"),
        Outcome("regret", "Regret against perfect information", False, "performance"),
        Outcome("cvar5", "Downside loss (mean of worst 5% of periods)", True, "performance"),
        Outcome("compute_ms", "Compute time per replication (ms)", None, "performance"),
        Outcome("adaptation_rate", "Adjustment frequency", None, "behavior"),
        Outcome("change_size", "Adjustment magnitude (mean |ΔS|)", None, "behavior"),
        Outcome("recovery_delay", "Recovery delay after a change (periods)", False, "behavior"),
        Outcome("missed", "Missed opportunities (share)", False, "mechanisms"),
        Outcome("calib_bias", "Calibration bias of predicted gain", None, "mechanisms"),
        Outcome("calib_slope", "Calibration slope of predicted gain", None, "mechanisms"),
    )
    primary = "net_payoff"
    seconds_per_unit = 4e-5
    tuning_note = ("Grid tuning on training seeds, four candidates each: the forecast gain (scored on always "
                   "adapting), the band width, and the confidence-sensitive gate's confidence and minimum evidence. "
                   "The estimated-gain gate has no free hyperparameters; change detection and the robust order keep "
                   "their documented parameters.")

    def _cfg(self, params):
        from .. import learnability as L
        return L.Config(family=params["family"], sigma=float(params["sigma"]), tau=float(params["tau"]),
                        hazard=float(params["hazard"]), avail=float(params["avail"]), memory=float(params["memory"]),
                        default=params["default"], cost=float(params["cost"]))

    def _hp(self, params, hp):
        """Hyperparameters for learnability.policy_orders. The "gate" entry must be present: without it the
        confidence-sensitive gate silently fell back to learnability.GATE_HP0 and neither the control nor any tuned
        value reached it (fixed 8 October 2026)."""
        return {"gain": float(hp.get("gain", params["gain"])),
                "band": {"b": float(hp.get("band_b", params["band_b"]))},
                "gate": {"conf": float(hp.get("gate_conf", params["gate_conf"])),
                         "nmin": float(hp.get("gate_nmin", params["gate_nmin"]))},
                "bocpd": {"hazard": 0.005, "prior_sd": 30.0, "obs_sd": 25.0}, "dro": {"window": 20, "rho": 0.01}}

    def tune(self, spec, params, seeds, cancel):
        from .. import learnability as L
        cfg = self._cfg(params)
        T, burn = spec.design.periods, spec.design.burn_in

        def score(policy, hp):
            vals = []
            for s in seeds:
                p = L.paths(cfg, T, s)
                c = L.candidate_orders(cfg, p, hp["gain"])
                S, pr, go = L.policy_orders(policy, cfg, p, c, hp)
                vals.append(L.outcomes(policy, cfg, p, c, S, pr, go, burn)["net_payoff"])
            return float(np.mean(vals))

        base = self._hp(params, {})
        gain = max((0.05, 0.1, 0.2, 0.4), key=lambda g: score("always", {**base, "gain": g}))
        if cancel():
            return {}
        b = max((0.25, 0.5, 1.0, 2.0), key=lambda b: score("band", {**base, "gain": gain, "band": {"b": b}}))
        if cancel():
            return {"gain": gain, "band_b": b}
        # Same number of candidates as the two policies above, so the gate is not compared on a smaller budget.
        gate_grid = ((0.9, 5.0), (0.8, 3.0), (0.9, 12.0), (0.95, 25.0))
        conf, nmin = max(gate_grid, key=lambda v: score(
            "gate_lcb", {**base, "gain": gain, "gate": {"conf": v[0], "nmin": v[1]}}))
        return {"gain": gain, "band_b": b, "gate_conf": conf, "gate_nmin": nmin}

    def run_cell(self, spec, params, seeds, hp, cancel, trace):
        from .. import learnability as L
        cfg = self._cfg(params)
        h = self._hp(params, hp)
        T, burn = spec.design.periods, spec.design.burn_in
        rows, traces = [], {}
        for r, s in enumerate(seeds):
            if cancel():
                break
            p = L.paths(cfg, T, s)
            c = L.candidate_orders(cfg, p, h["gain"])
            for pol in spec.policies:
                t0 = time.perf_counter()
                S, pred, gobs = L.policy_orders(pol, cfg, p, c, h)
                out = L.outcomes(pol, cfg, p, c, S, pred, gobs, burn)
                out["change_size"] = float(np.abs(np.diff(S[burn:])).mean())
                out["compute_ms"] = 1000 * (time.perf_counter() - t0)
                rows.append(dict(policy=pol, replication=r, seed=s, **out))
                if trace and r == 0:
                    traces[pol] = self._trace(pol, cfg, p, c, S, pred, gobs, h)
        return rows, traces

    def _trace(self, pol, cfg, p, c, S, pred, gobs, h):
        T = len(S)
        adapt = np.abs(S - c["S_D"]) > 1e-9
        reason = {
            "always": lambda t: "always takes the target",
            "default": lambda t: "always keeps the default",
            "band": lambda t: (f"|target − default| = {abs(c['S_F'][t] - c['S_D'][t]):.1f} "
                               f"{'>' if adapt[t] else '≤'} b·s.d. = {h['band']['b'] * c['sd'][t]:.1f}"),
            "gate_gain": lambda t: ("no evidence in this bin: adapts (optimistic start)" if np.isnan(pred[t]) else
                                    f"learned gain {pred[t]:.2f} {'≥' if adapt[t] else '<'} cost {cfg.cost:g}"),
            "gate_lcb": lambda t: ("not enough evidence or lower bound ≤ cost: keeps the default" if not adapt[t] else
                                   "lower confidence bound above the cost"),
            "bocpd": lambda t: "posterior predictive critical fractile (complete policy)",
            "dro": lambda t: "worst-case order over recent observations (complete policy)",
            "oracle": lambda t: "true critical fractile (researcher benchmark)",
        }[pol]
        return pd.DataFrame(dict(
            period=np.arange(T),
            observed=np.where(p["seen"], p["y"], np.nan),
            belief_sd=c["sd"], default_order=c["S_D"], proposed=c["S_F"], chosen=S, acted=adapt,
            predicted_gain=pred if pred is not None else np.nan,
            feedback=gobs if gobs is not None else np.nan,
            feedback_available=np.where(p["seen"], np.arange(T), -1),
            reason=[reason(t) for t in range(T)],
            researcher_demand=p["d"], researcher_mean=p["mu"]))


# ================================================================================================ market
class Market(Environment):
    key = "market"
    label = "Market (quantity competition with cost shocks)"
    description = ("Firms choose output each period; price clears a linear demand curve and raw-material cost follows "
                   "a reflecting random walk. Every firm uses the compared selection policy on a shared production rule "
                   "(identical cost paths across policies).")
    engine = "market"
    candidates = (
        Candidate("Bertrand", "Margin-feedback quantity rule (model-free)",
                  "q* = q + φ·(observed margin − m*): uses only the last price and its cost estimate.",
                  (Control("flex", "Responsiveness φ", "agents", "float", 0.25, 0.0, 2.0, 0.05),
                   Control("desired_margin", "Desired margin m*", "agents", "float", 5.0, -5.0, 30.0, 0.5,
                           advanced=True))),
        Candidate("Cournot", "Cournot best reply (model-based)",
                  "Move a share φ toward the best reply on the believed demand curve.",
                  (Control("flex", "Adjustment share φ (≤ 1)", "agents", "float", 0.5, 0.0, 1.0, 0.05),)),
    )
    controls = (
        Control("delta", "Cost volatility Δ (max change per period)", "outcome", "float", 10.0, 0.0, 40.0, 0.5),
        Control("noise", "Cost-perception noise σ", "observation", "float", 5.0, 0.0, 20.0, 0.5),
        Control("obs_noise", "Noise on observed price and market output", "observation", "float", 0.0, 0.0, 10.0, 0.5,
                advanced=True),
        Control("obs_delay", "Observation delay (periods)", "observation", "int", 0, 0, 10, 1),
        Control("hazard", "Demand-regime shift frequency (per period)", "change", "float", 0.0, 0.0, 0.05, 0.002),
        Control("intercept_sd", "Shift magnitude (s.d. of the demand intercept)", "change", "float", 10.0, 0.0, 30.0,
                1.0, advanced=True),
        Control("announced", "Shifts announced", "change", "bool", False),
        Control("demand_knowledge", "Demand model", "model", "choice", "believed", options=("none", "believed"),
                help="none: no demand model (the Cournot rule needs one); believed: the believed curve, updated a lag "
                     "after a shift."),
        Control("belief_lag", "Belief lag after a shift (periods)", "model", "int", 20, -1, 100, 1, advanced=True),
        Control("n_firms", "Number of firms", "complexity", "int", 4, 2, 8, 1),
        Control("adjust_cost", "Adjustment cost per adaptation", "agents", "float", 20.0, 0.0, 200.0, 5.0),
        Control("threshold", "Inaction band threshold θ (units of output)", "agents", "float", 25.0, 0.0, 200.0, 5.0),
        Control("memory", "Gates' memory λ", "agents", "float", 0.97, 0.8, 0.999, 0.005, advanced=True),
        Control("window", "Judgement window (periods)", "agents", "int", 5, 1, 50, 1, advanced=True),
        Control("confidence", "Confidence level of the lower bound", "agents", "float", 0.9, 0.5, 0.99, 0.01,
                advanced=True),
    )
    not_applicable = {}
    policies = (
        PolicyInfo("always", "Always act", "selection", "Adopt every recommendation (selection rule Always).",
                   "observed price, cost estimate, own output"),
        PolicyInfo("default", "Retain the default", "selection", "Keep the current output (rule B).", "own output"),
        PolicyInfo("band", "Inaction band", "selection", "Adopt only changes larger than θ (selection rule Large).",
                   "recommended change"),
        PolicyInfo("gate_gain", "Reliability gate: estimated gain", "selection", "Adopt where the learned advantage "
                   "per size bin covers the adjustment cost (Adaptive rule).", "estimated counterfactual feedback"),
        PolicyInfo("gate_lcb", "Reliability gate: confidence-sensitive", "selection", "Adopt only with enough evidence "
                   "and a lower confidence bound above the cost.", "estimated feedback, its uncertainty and count",
                   extension=True),
        PolicyInfo("gate_explore", "Reliability gate: exploration", "selection", "Randomized trials; learns from its "
                   "own payoffs only.", "own realized payoffs", extension=True),
    )
    outcomes = (
        Outcome("net_profit", "Net profit per firm-period", True, "performance"),
        Outcome("profit", "Gross profit per firm-period", True, "performance"),
        Outcome("compute_ms", "Compute time per replication (ms)", None, "performance"),
        Outcome("adaptation_rate", "Adjustment frequency", None, "behavior"),
        Outcome("change_size", "Adjustment magnitude (mean |Δq|)", None, "behavior"),
        Outcome("false_share", "False adaptations (share of adaptations that lost)", False, "mechanisms"),
        Outcome("missed", "Missed opportunities (share of profitable changes not made)", False, "mechanisms"),
        Outcome("perception_error", "Cost-perception error (RMS)", None, "mechanisms"),
    )
    primary = "net_profit"
    seconds_per_unit = 1.5e-4
    tuning_note = "Grid tuning on training seeds: the inaction band threshold θ. Other policies keep their settings."

    def information(self, spec):
        p = self.params(spec)
        return dict(feedback=spec.information.get("feedback", "estimated"), obs_noise=float(p["obs_noise"]),
                    obs_delay=int(p["obs_delay"]), demand_knowledge=p["demand_knowledge"],
                    regime_announced=bool(p["announced"]), rivals_visible=bool(spec.information.get("rivals_visible",
                                                                                                     True)),
                    unmet_demand_observed=True)

    def validate(self, spec):
        out = []
        p = self.params(spec)
        if spec.candidate == "Cournot" and p["demand_knowledge"] == "none":
            out.append(("environment", "error", "The Cournot best reply needs a demand model; choose 'believed' or "
                                                "the margin-feedback rule."))
        if spec.candidate == "Cournot" and float(p["flex"]) > 1:
            out.append(("agents", "error", "The Cournot adjustment share must be at most 1."))
        if "gate_explore" in spec.policies and spec.information.get("feedback", "estimated") not in ("chosen",
                                                                                                      "estimated"):
            out.append(("information", "warning", "The exploration gate learns from its own payoffs; other feedback "
                                                  "treatments do not change it."))
        return out

    def _scenario(self, spec, params, policy, seed, hp):
        from ..information import InfoSpec
        from ..params import FirmSpec, GlobalFirmParams, MarketParams, Scenario, StructuralParams
        sel = {"always": "Always", "default": "Never", "band": "Large"}.get(policy, "Adaptive")
        th = float(hp.get("threshold", params["threshold"]))
        firms = [FirmSpec(rule=spec.candidate or "Bertrand", flex=float(params["flex"]), selection=sel, threshold=th,
                          desired_margin=float(params.get("desired_margin", 5.0)), noise=float(params["noise"]))
                 for _ in range(int(params["n_firms"]))]
        s = Scenario(market=MarketParams(delta=float(params["delta"])), firms=firms,
                     firm_globals=GlobalFirmParams(),
                     structural=StructuralParams(float(params["hazard"]) > 0, max(float(params["hazard"]), 1e-9),
                                                 float(params["intercept_sd"]), 0.3, int(params["belief_lag"])),
                     periods=int(spec.design.periods), burn_in=int(spec.design.burn_in), seed=int(seed))
        s.info = InfoSpec(**self.information(spec))
        ad = s.adaptive
        ad.memory, ad.window = float(params["memory"]), int(params["window"])
        ad.adjust_cost, ad.confidence = float(params["adjust_cost"]), float(params["confidence"])
        ad.gate = {"gate_lcb": "lcb", "gate_explore": "explore"}.get(policy, "gain")
        return s

    def _net(self, res, params):
        a = res.acc
        net = (a["sum_profit"] - float(params["adjust_cost"]) * a["n_dev"]) / a["n_rec"]
        return net

    def tune(self, spec, params, seeds, cancel):
        from ..engine import run_batch
        scores = {}
        for th in (10.0, 25.0, 50.0, 100.0):
            if cancel():
                return {}
            res = run_batch([self._scenario(spec, params, "band", s, {"threshold": th}) for s in seeds])
            scores[th] = float(self._net(res, params).mean())
        return {"threshold": max(scores, key=scores.get)}

    def run_cell(self, spec, params, seeds, hp, cancel, trace):
        from ..engine import run_batch
        rows, traces = [], {}
        for pol in spec.policies:
            if cancel():
                break
            t0 = time.perf_counter()
            scns = [self._scenario(spec, params, pol, s, hp) for s in seeds]
            res = run_batch(scns)
            ms = 1000 * (time.perf_counter() - t0) / len(seeds)
            a = res.acc
            net = self._net(res, params)
            for r, s in enumerate(seeds):
                n_dev, n_rec = a["n_dev"][r], a["n_rec"][r]
                rows.append(dict(
                    policy=pol, replication=r, seed=s, net_profit=float(net[r].mean()),
                    profit=float((a["sum_profit"][r] / n_rec).mean()), compute_ms=ms,
                    adaptation_rate=float((n_dev / n_rec).mean()),
                    change_size=float((a["sum_absdq"][r] / n_rec).mean()),
                    false_share=float(np.nansum(a["n_dev_npe"][r]) / max(np.nansum(n_dev), 1.0)),
                    missed=float(np.nansum(a["n_pe"][r] - a["n_dev_pe"][r]) / max(np.nansum(a["n_pe"][r]), 1.0)),
                    perception_error=float(np.sqrt((a["sum_cerr2"][r] / n_rec).mean()))))
            if trace and len(seeds):
                one = run_batch([scns[0]], record_firm_history=True)
                traces[pol] = self._trace(one, pol, params, hp)
        return rows, traces

    def _trace(self, res, pol, params, hp):
        h = {k: v[0, :, 0] for k, v in res.firm_hist.items() if getattr(v, "ndim", 0) == 3}
        T = len(h["q"])
        q_prev = np.concatenate([[np.nan], h["q"][:-1]])
        th = float(hp.get("threshold", params["threshold"]))
        reasons = []
        for t in range(T):
            ch = abs(h["rec"][t] - q_prev[t]) if t else np.nan
            if not h["opportunity"][t]:
                reasons.append("recommendation equals current output")
            elif pol == "always":
                reasons.append("always adopts")
            elif pol == "default":
                reasons.append("always keeps output (rule B)")
            elif pol == "band":
                reasons.append(f"|Δ| = {ch:.0f} {'>' if h['deviate'][t] else '≤'} θ = {th:g}")
            else:
                pr, bd = h.get("adv_pred", np.full(T, np.nan))[t], h.get("adv_bound", np.full(T, np.nan))[t]
                reasons.append(f"predicted advantage {pr:.1f}, lower bound {bd:.1f}; "
                               f"{'adopts' if h['deviate'][t] else 'keeps'}")
        rel = h.get("fb_release", np.full(T, -1))
        return pd.DataFrame(dict(
            period=np.arange(T), observed_price=np.concatenate([[np.nan], res.price[0, :-1]]),
            belief_cost=h.get("c_hat", np.full(T, np.nan)), belief_best_reply=h["best_reply"],
            current_output=q_prev, proposed=h["rec"], acted=h["deviate"], opportunity=h["opportunity"],
            predicted_gain=h.get("adv_pred", np.full(T, np.nan)), lower_bound=h.get("adv_bound", np.full(T, np.nan)),
            feedback=h.get("fb_value", np.full(T, np.nan)), feedback_available=rel, reason=reasons,
            researcher_cost=res.cost[0], researcher_gain=h["gain_full"]))


# ================================================================================================ generalization tasks
class _Task(Environment):
    task = ""
    engine = "tasks"
    seconds_per_unit = 1.5e-5
    not_applicable = {"model": "No model: both rules smooth observed outcomes.",
                      "complexity": "One binary or continuous choice per period; no search budget."}
    policies = (
        PolicyInfo("always", "Always deviate", "selection", "Take the flexible action every period.", "observed outcomes"),
        PolicyInfo("ruleb", "Retain the default (rule B)", "selection", "Take the default action.", "observed outcomes"),
        PolicyInfo("band", "Inaction band", "selection", "Deviate only when the signal exceeds b times the agent's "
                   "noise estimate.", "signal strength"),
        PolicyInfo("rc_learned", "Reliability condition, learned", "selection", "Per signal bin, deviate where the "
                   "learned mean gain of deviating is positive.", "observed gains of both actions per bin"),
        PolicyInfo("rc_oracle", "Reliability condition, ORACLE", "benchmark", "Uses the true mean gain per bin from a "
                   "long independent run.", "true bin gains (researcher-only)"),
    )
    outcomes = (
        Outcome("payoff", "Payoff per period", True, "performance"),
        Outcome("compute_ms", "Compute time per replication (ms)", None, "performance"),
        Outcome("deviation_rate", "Adjustment frequency", None, "behavior"),
        Outcome("K", "Error-to-signal ratio K of the flexible rule", None, "mechanisms"),
    )
    primary = "payoff"

    def __init__(self):
        from ..tasks import RANGES
        r = RANGES[self.task]
        mid = lambda k: round((r[k][0] + r[k][1]) / 2, 3)
        self.controls = (
            Control("noise", "Outcome noise (s.d.)", "outcome", "float", mid("noise"), 0.0, r["noise"][1] * 1.5),
            Control("obs_noise", "Observation error (s.d.)", "observation", "float", mid("obs_noise"), 0.0,
                    r["obs_noise"][1] * 1.5),
            Control("hazard", "Shift frequency (per period)", "change", "float", 0.01, 0.0, 0.1, 0.001),
            Control("band_b", "Inaction band width b", "agents", "float", 1.0, 0.0, 4.0, 0.25, advanced=True),
            Control("rc_memory", "Learned RC memory", "agents", "float", 0.995, 0.9, 0.9999, 0.0005, advanced=True),
        )
        self.candidates = (Candidate("fast_estimate", "Fast re-estimate",
                                     "The flexible action from a fast exponentially weighted estimate of the outcomes.",
                                     (Control("gain", "Gain of the fast estimate", "agents", "float", 0.3, 0.02, 0.9,
                                              0.01),)),)

    def _env(self, params, seed):
        from ..tasks import TaskEnv
        return TaskEnv(self.task, float(params["noise"]), float(params["hazard"]), float(params["obs_noise"]),
                       float(params["gain"]), int(seed))

    def run_cell(self, spec, params, seeds, hp, cancel, trace):
        from .. import tasks as TK
        T, burn = spec.design.periods, spec.design.burn_in
        rows, traces = [], {}

        class _P:
            oracle_periods = max(4 * T, 4000)

        for r, s in enumerate(seeds):
            if cancel():
                break
            env = self._env(params, s)
            dec = TK._decisions(env, T, s)
            table = TK.oracle_table(env, _P()) if "rc_oracle" in spec.policies else None
            for pol in spec.policies:
                t0 = time.perf_counter()
                out = TK._layer(dec, pol, burn, band=float(params["band_b"]), table=table,
                                rc_memory=float(params["rc_memory"]))
                rows.append(dict(policy=pol, replication=r, seed=s, payoff=out["payoff"],
                                 deviation_rate=out["deviation_rate"], K=out["K"],
                                 compute_ms=1000 * (time.perf_counter() - t0)))
                if trace and r == 0:
                    traces[pol] = task_trace(dec, pol, float(params["band_b"]), table, float(params["rc_memory"]))
        return rows, traces


def task_trace(dec, layer, band, table, rc_memory) -> pd.DataFrame:
    """Per-period decisions of a generalization-task layer, recomputing tasks._layer's documented rules (checked
    against it in tests)."""
    from ..tasks import N_BINS
    T = len(dec["d"])
    dev = np.zeros(T, bool)
    learned = np.full(T, np.nan)
    reason = []
    lg, lc = np.zeros(N_BINS), np.zeros(N_BINS)
    for t in range(T):
        b, diff = int(dec["bin"][t]), bool(dec["differ"][t])
        if not diff:
            reason.append("flexible and default actions coincide")
            continue
        if layer == "always":
            dev[t] = True
            reason.append("always deviates")
        elif layer == "ruleb":
            reason.append("never deviates")
        elif layer == "band":
            dev[t] = dec["sig"][t] > band
            reason.append(f"signal {dec['sig'][t]:.2f} {'>' if dev[t] else '≤'} b = {band:g}")
        elif layer == "rc_oracle":
            dev[t] = table[b] > 0
            reason.append(f"true mean gain in bin {b}: {table[b]:.3f}")
        else:
            learned[t] = lg[b] if lc[b] else np.nan
            dev[t] = lc[b] < 5 or lg[b] > 0
            reason.append(f"bin {b}: {int(lc[b])} observations, learned gain "
                          + (f"{lg[b]:.3f}" if lc[b] else "none") + (" (fewer than 5: deviates)" if lc[b] < 5 else ""))
            lc[b] += 1
            lg[b] += (dec["obs_gain"][t] - lg[b]) * max(1 - rc_memory, 1 / lc[b])
    return pd.DataFrame(dict(period=np.arange(T), signal=dec["sig"], bin=dec["bin"], default_action=dec["d"],
                             proposed=dec["x"], acted=dev & dec["differ"], predicted_gain=learned,
                             feedback=dec["obs_gain"], feedback_available=np.arange(T), reason=reason,
                             researcher_gain=dec["true_gain"], researcher_optimum=dec["opt"]))


class Learning(_Task):
    key = "learning"
    task = "learning"
    label = "Learning task (two options whose payoffs swap)"
    description = ("Two options whose mean payoffs swap without warning; both actions' payoffs are observed (full "
                   "feedback). The default follows a slow estimate, the flexible action a fast one.")


class Investment(_Task):
    key = "investment"
    task = "investment"
    label = "Irreversible investment (booms and busts)"
    description = ("One project arrives each period; investing commits the firm and bad projects cost a write-off. The "
                   "default invests on a slow estimate of project quality, the flexible action on a fast one.")


# ================================================================================================ NK landscapes
class NKLandscape(Environment):
    key = "nk"
    label = "NK landscape (interacting decisions)"
    description = ("N binary decisions whose payoff contributions each depend on K_NK others (complexity). Observation "
                   "noise and unannounced landscape changes are separate uncertainty mechanisms.")
    engine = "nk"
    candidates = (Candidate("stochastic", "Local and distant search proposals",
                            "Flip 1 + Binomial(N − 1, p) components of the current configuration.",
                            (Control("p_jump", "Probability of each extra flip p", "agents", "float", 0.25, 0.0, 1.0,
                                     0.05, advanced=True),)),)
    controls = (
        Control("obs_noise", "Observation noise (s.d.)", "observation", "float", 0.02, 0.0, 0.2, 0.005),
        Control("hazard", "Landscape change frequency (per period)", "change", "float", 0.005, 0.0, 0.05, 0.001),
        Control("change_frac", "Share of components redrawn at a change", "change", "float", 0.25, 0.05, 1.0, 0.05,
                advanced=True),
        Control("N", "Number of decisions N", "complexity", "int", 8, 4, 16, 1, treatable=True),
        Control("K_NK", "Interactions per decision K_NK", "complexity", "int", 2, 0, 15, 1),
        Control("topology", "Interaction topology", "complexity", "choice", "adjacent",
                options=("adjacent", "random", "block"), advanced=True),
        Control("budget", "Evaluation budget", "complexity", "int", 200, 10, 2000, 10),
        Control("cost", "Switching cost per changed component", "agents", "float", 0.002, 0.0, 0.05, 0.001),
        Control("observe", "Imitation: share of the leader observable", "agents", "float", 0.5, 0.0, 1.0, 0.125,
                advanced=True),
    )
    not_applicable = {"outcome": "Payoffs are deterministic given the configuration; variation enters through "
                                 "observation noise and change.",
                      "model": "No model: agents see noisy payoffs only."}
    policies = (
        PolicyInfo("gate_none", "Act on apparent improvements", "selection", "Switch whenever a proposal looks better "
                   "(pays for verification it ignores: same budget as the gates).", "noisy payoffs"),
        PolicyInfo("gate_gain", "Reliability gate: estimated gain", "selection", "Switch where the verified gain of "
                   "apparent improvements of this strength has been non-negative.", "verified gains per bin",
                   extension=True),
        PolicyInfo("gate_lcb", "Reliability gate: confidence-sensitive", "selection", "Switch only with enough "
                   "evidence and a lower bound on the verified gain above zero.", "verified gains, uncertainty, count",
                   extension=True),
        PolicyInfo("hill", "Local hill climbing", "complete", "One-flip proposals; switch on apparent improvements.",
                   "noisy payoffs", "Own one-flip proposal generator; no verification."),
        PolicyInfo("stochastic", "Stochastic search (annealing)", "complete", "Multi-flip proposals; accepts some "
                   "worse candidates early.", "noisy payoffs", "Temperature falls with the budget used."),
        PolicyInfo("satisfice", "Satisficing", "complete", "Searches only while performance is below an adaptive "
                   "aspiration.", "noisy payoffs, aspiration", "Aspiration starts above the first observation."),
        PolicyInfo("imitate", "Imitation", "complete", "Copies observable components of a leader.",
                   "a share of the leader's configuration", "Leader is researcher-generated full-information search."),
    )
    outcomes = (
        Outcome("payoff", "Attained payoff per period", True, "performance"),
        Outcome("regret", "Regret per period (exact maximum for N ≤ 16)", False, "performance"),
        Outcome("evaluations", "Evaluations used (computational cost)", None, "performance"),
        Outcome("switches", "Switches", None, "behavior"),
        Outcome("escapes", "Escapes from local peaks", None, "behavior"),
        Outcome("recovery_delay", "Recovery delay after change (lower bound)", False, "behavior"),
        Outcome("false_share", "False improvements (share of switches)", False, "mechanisms"),
        Outcome("missed_share", "Missed improvements (share)", False, "mechanisms"),
        Outcome("stuck_share", "Share of periods at a non-global local peak", False, "mechanisms"),
    )
    primary = "payoff"
    seconds_per_unit = 3e-5
    cluster_note = "Each replication is a new landscape with its own starting configuration."

    def validate(self, spec):
        p = self.params(spec)
        out = []
        if not 0 <= int(p["K_NK"]) <= int(p["N"]) - 1:
            out.append(("environment", "error", f"K_NK must lie between 0 and N − 1 = {int(p['N']) - 1}."))
        lv = spec.design.treatment
        if lv.control == "K_NK" and any(int(k) > int(p["N"]) - 1 for k in lv.levels):
            out.append(("design", "error", "A K_NK treatment level exceeds N − 1."))
        return out

    def _env(self, spec, params, seed):
        from .. import nk
        return nk.NKEnv(N=int(params["N"]), K_NK=int(params["K_NK"]), topology=params["topology"],
                        obs_noise=float(params["obs_noise"]), hazard=float(params["hazard"]),
                        change_frac=float(params["change_frac"]), cost=float(params["cost"]),
                        budget=int(params["budget"]), periods=int(spec.design.periods), observe=float(params["observe"]),
                        seed=int(seed))

    def run_cell(self, spec, params, seeds, hp, cancel, trace):
        from .. import nk
        hpar = nk.SearchParams(p_jump=float(params["p_jump"]))
        rows, traces = [], {}
        for r, s in enumerate(seeds):
            if cancel():
                break
            env = self._env(spec, params, s)
            t0 = time.perf_counter()
            df = nk.run_landscape(env, [s + 1], hpar, tuple(spec.policies))
            ms = 1000 * (time.perf_counter() - t0) / max(len(spec.policies), 1)
            for _, row in df.iterrows():
                d = row.drop(["landscape", "run", "searcher", "benchmark"]).to_dict()
                rows.append(dict(policy=row["searcher"], replication=r, seed=s, compute_ms=ms, **d))
            if trace and r == 0:
                path = nk.landscape_path(env)
                st_ = nk.run_streams(env, path, s + 1)
                obs = nk.Observer(env, path, st_.noise, st_.leader, st_.visible)
                for pol in spec.policies:
                    xs, costs, log = nk.simulate(pol, env, obs, st_.start, st_.U, hpar)
                    traces[pol] = self._trace(env, path, xs, log)
        return rows, traces

    def _trace(self, env, path, xs, log):
        from .. import nk
        T = env.periods
        ep = np.zeros(T, dtype=int)
        for k, e in enumerate(path):
            ep[e.start:] = k
        by_t = {d["t"]: d for d in log}
        rec = []
        for t in range(T):
            d = by_t.get(t)
            true = nk.true_payoff(path[ep[t]], int(xs[t]))
            if d is None:
                rec.append(dict(period=t, configuration=format(int(xs[t]), f"0{env.N}b"), observed=np.nan,
                                proposed="", observed_candidate=np.nan, acted=False, predicted_gain=np.nan,
                                feedback=np.nan, feedback_available=-1, reason="no proposal (budget or aspiration met)",
                                researcher_payoff=true))
                continue
            g1 = d["o_y"] - d["o_x"] - d["c"]
            rec.append(dict(period=t, configuration=format(int(d["x"]), f"0{env.N}b"), observed=d["o_x"],
                            proposed=format(int(d["y"]), f"0{env.N}b"), observed_candidate=d["o_y"],
                            acted=bool(d["accept"] and d["y"] != d["x"]), predicted_gain=g1, feedback=d["g2"],
                            feedback_available=t if np.isfinite(d["g2"]) else -1,
                            reason=(f"apparent net gain {g1:+.4f} (strength bin {d['bin']}); "
                                    + ("accepted" if d["accept"] else "rejected")),
                            researcher_payoff=true))
        return pd.DataFrame(rec)


ENVIRONMENTS: Dict[str, Environment] = {e.key: e for e in (Inventory(), Market(), Learning(), Investment(),
                                                           NKLandscape())}
