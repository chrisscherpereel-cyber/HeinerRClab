"""Market replication of the learnability study (heiner_abm.learnability). PROPOSED EXTENSION.

The market engine (engine.run_batch) with every firm using the same selection policy on shared exogenous paths
(common random numbers; same seeds for every policy):
    always      adopt every recommendation (selection rule Always)
    default     keep last output (rule B, selection rule Never)
    band        inaction band: adopt only changes larger than a threshold (selection rule Large, threshold tuned)
    gate_gain   Adaptive rule, existing estimated-gain gate (heiner_abm.gates); no free hyperparameters
    gate_lcb    Adaptive rule, confidence-sensitive gate (confidence and minimum evidence tuned)
Equal tuning budget: `band` and `gate_lcb` are the only policies with free hyperparameters, and each is given the same
number of candidate settings (MarketPlan.band_grid, MarketPlan.gate_grid), scored on the same training configurations
and paths. Before 7 October 2026 only the band was tuned and the gate ran at the AdaptiveParams defaults; set
MarketPlan.tune_gate = False for that legacy behavior.
Bayesian change detection and the distributionally robust order are inventory-task policies with no counterpart in
the market's selection decision; they are not run here (reported as not applicable).

Manipulated: cost volatility (delta), perception error (each firm's cost-noise sigma), regime-change frequency
(structural hazard), feedback availability (observation delay, which postpones every observation and every release of
feedback), and the gates' memory. Default quality is not manipulated: rule B's quality is endogenous to the market.
Net payoff per period = market profit minus the adaptation cost c for every adopted recommendation (all policies).

Learnability ratio (independent pilot paths), as in the inventory study: per firm and demand regime, mu_r = the
researcher's counterfactual mean advantage of adopting over the judgement window (pilot only), s = pooled noise of the
gain gate's released feedback around mu_r, n_needed = (z_0.9 s / |mu_r|)^2, n_avail = released feedback values in the
regime; R = median of n_needed / max(n_avail, 1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .engine import run_batch
from .params import GlobalFirmParams, MarketParams, Scenario, StructuralParams, linear_flex_firms

Z90 = 1.2815515655446004
MPOLICIES = ("always", "default", "band", "gate_gain", "gate_lcb")
NOT_APPLICABLE = {"bocpd": "Bayesian change detection orders a newsvendor quantity; the market's decision is whether to "
                           "adopt the production rule's recommendation, which it does not model.",
                  "dro": "The robust newsvendor order has no counterpart in the market's selection decision."}


@dataclass(frozen=True)
class MarketConfig:
    delta: float = 10.0
    noise: float = 5.0
    hazard: float = 0.01
    delay: int = 0
    memory: float = 0.97
    cost: float = 20.0


@dataclass(frozen=True)
class MarketPlan:
    periods: int = 1000
    burn_in: int = 50
    paths: int = 6
    pilot_paths: int = 4
    n_train_configs: int = 4
    n_test_configs: int = 16
    seed: int = 20261008
    window: int = 5
    band_grid: Tuple[float, ...] = (10.0, 25.0, 50.0)
    gate_grid: Tuple[Tuple[float, float], ...] = ((0.9, 5.0), (0.8, 3.0), (0.9, 12.0))   # (confidence, min evidence)
    tune_gate: bool = True                 # False = legacy: the gate is not tuned and uses gate_grid[0]
    lhs_ranges: Tuple[Tuple[str, Tuple[float, float]], ...] = (
        ("delta", (2.0, 30.0)), ("noise", (0.0, 15.0)), ("hazard", (0.002, 0.05)), ("delay", (0.0, 10.0)),
        ("memory", (0.9, 0.995)))
    n_boot: int = 2000


QUICK_MARKET = dict(periods=400, paths=3, pilot_paths=2, n_train_configs=2, n_test_configs=5, n_boot=300)


def scenario(cfg: MarketConfig, policy: str, seed: int, periods: int, burn: int, window: int, theta: float,
             gate_hp: Tuple[float, float] = (0.9, 5.0)) -> Scenario:
    """One market where every firm runs `policy`. theta is the tuned inaction-band threshold (used by `band` only);
    gate_hp is the tuned (confidence, minimum evidence) of the confidence-sensitive gate (used by `gate_lcb` only)."""
    sel = {"always": "Always", "default": "Never", "band": "Large"}.get(policy, "Adaptive")
    s = Scenario(market=MarketParams(100.0, 10.0, 1500.0, 45.0, 80.0, float(cfg.delta)),
                 firms=linear_flex_firms(4, 0.25, 0.0, "Bertrand", sel, float(theta), 5.0, 0.0, float(cfg.noise)),
                 firm_globals=GlobalFirmParams(200.0, 15.0, 0.0, 0.0, False),
                 structural=StructuralParams(True, float(cfg.hazard), 15.0, 0.4, 20), periods=periods, burn_in=burn,
                 seed=int(seed))
    s.info.obs_delay = int(round(cfg.delay))
    ad = s.adaptive
    ad.memory, ad.window, ad.adjust_cost = float(cfg.memory), int(window), float(cfg.cost)
    ad.gate = "lcb" if policy == "gate_lcb" else "gain"
    ad.confidence, ad.min_evidence = float(gate_hp[0]), float(gate_hp[1])
    return s


def net_payoffs(cfg: MarketConfig, policy: str, seeds: Sequence[int], plan: MarketPlan, theta: float,
                gate_hp: Tuple[float, float] = (0.9, 5.0)) -> np.ndarray:
    """Mean net payoff per firm and period, one value per path (market)."""
    scns = [scenario(cfg, policy, sd, plan.periods, plan.burn_in, plan.window, theta, gate_hp) for sd in seeds]
    res = run_batch(scns, horizon=1)
    a = res.acc
    net = (a["sum_profit"] - cfg.cost * a["n_dev"]) / a["n_rec"]
    return net.mean(axis=1)


def ratio(cfg: MarketConfig, seeds: Sequence[int], plan: MarketPlan, theta: float) -> Dict[str, float]:
    """Learnability ratio from pilot paths: per firm and demand regime, the researcher's counterfactual gives the
    regime's mean advantage of adopting (over the judgement window), the gain gate's released feedback gives the noise
    of what the firm learns from and the number of informative observations."""
    scns = [scenario(cfg, "gate_gain", sd, plan.periods, plan.burn_in, plan.window, theta) for sd in seeds]
    res = run_batch(scns, record_firm_history=True, horizon=plan.window)
    h = res.firm_hist
    regimes, resid = [], []
    for b in range(len(scns)):
        cps = sorted(set([plan.burn_in] + [t for t in np.flatnonzero(res.shifts[b]) if t > plan.burn_in]
                         + [plan.periods]))
        for lo, hi in zip(cps[:-1], cps[1:]):
            for i in range(h["q"].shape[2]):
                opp = h["opportunity"][b, lo:hi, i]
                if not opp.any():
                    continue
                mu_r = float(h["gain_full"][b, lo:hi, i][opp].mean())
                g = h["fb_value"][b, lo:hi, i]
                g = g[np.isfinite(g)]
                resid.extend(g - mu_r)
                regimes.append((mu_r, len(g)))
    s = float(np.std(resid, ddof=1)) if len(resid) > 1 else np.nan
    need = np.array([np.clip((Z90 * s / max(abs(m), 1e-9)) ** 2, 1.0, 1e6) for m, _ in regimes])
    avail = np.array([n for _, n in regimes], float)
    R = float(np.median(need / np.maximum(avail, 1.0)))
    return dict(n_needed=float(np.median(need)), n_avail=float(np.median(avail)), ratio=R, log10_ratio=float(np.log10(R)))


def _lhs(plan: MarketPlan, n: int, seed: int) -> List[MarketConfig]:
    rng = np.random.default_rng(seed)
    cols = {}
    for k, (lo, hi) in plan.lhs_ranges:
        u = (rng.permutation(n) + rng.random(n)) / n
        cols[k] = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if k == "hazard" else lo + u * (hi - lo)
    return [MarketConfig(**{k: float(v[j]) for k, v in cols.items()}) for j in range(n)]


def _seeds(plan: MarketPlan, tag: int, j: int, n: int) -> List[int]:
    return [plan.seed * 1000 + tag * 100_000_000 + j * 1000 + r for r in range(n)]


def _boot(x, n_boot, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def run_market(plan: MarketPlan, progress=None) -> Dict:
    """Tuning on training configurations only, with the same budget for the two policies that have hyperparameters:
    len(plan.band_grid) candidates for the inaction band and len(plan.gate_grid) for the confidence-sensitive gate,
    each scored on the same training configurations and the same two paths per configuration."""
    train = _lhs(plan, plan.n_train_configs, plan.seed + 1)
    tune_on = lambda pol, th, gh: float(np.mean([net_payoffs(c, pol, _seeds(plan, 1, j, 2), plan, th, gh).mean()
                                                 for j, c in enumerate(train)]))
    cands = [(float(a), float(b)) for a, b in plan.gate_grid]      # a JSON round trip turns the tuples into lists
    scores = {float(th): tune_on("band", float(th), cands[0]) for th in plan.band_grid}
    theta = max(scores, key=scores.get)
    if plan.tune_gate:
        gate_scores = {g: tune_on("gate_lcb", theta, g) for g in cands}
        gate_hp = max(gate_scores, key=gate_scores.get)
    else:
        gate_scores, gate_hp = {}, cands[0]
    rows = []
    for j, cfg in enumerate(_lhs(plan, plan.n_test_configs, plan.seed + 3)):
        if progress:
            progress(j / plan.n_test_configs, f"Market configuration {j + 1}")
        pilot = {p: net_payoffs(cfg, p, _seeds(plan, 2, j, plan.pilot_paths), plan, theta, gate_hp).mean()
                 for p in ("always", "default")}
        best = max(pilot, key=pilot.get)
        r = ratio(cfg, _seeds(plan, 2, j, plan.pilot_paths), plan, theta)
        test = {p: net_payoffs(cfg, p, _seeds(plan, 3, j, plan.paths), plan, theta, gate_hp) for p in MPOLICIES}
        row = {**asdict(cfg), **r, "best_fixed": best, "theta": theta,
               "gate_confidence": gate_hp[0], "gate_min_evidence": gate_hp[1]}
        for p in MPOLICIES:
            row[f"np_{p}"] = float(test[p].mean())
        for p in ("gate_lcb", "gate_gain", "band"):
            m, lo, hi = _boot(test[p] - test[best], plan.n_boot)
            row[f"adv_{p}"], row[f"adv_{p}_lo"], row[f"adv_{p}_hi"] = m, lo, hi
        rows.append(row)
    return dict(tests=pd.DataFrame(rows), theta=theta, band_scores=scores, gate_hp=gate_hp,
                gate_scores=gate_scores, not_applicable=NOT_APPLICABLE, plan=asdict(plan))
