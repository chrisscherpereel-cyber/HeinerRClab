"""Study: when can reliability be learned before the environment changes? Inventory task (primary).

PROPOSED EXTENSION. The question, the ratio construct and the policies' operationalizations are the laboratory's own.
The specification (LearnPlan) is frozen in the repository with a hash of the plan and the code (registered.LEARN_PLAN);
it has not been preregistered with any external registry.

Task
    A newsvendor (Arrow, Harris & Marschak 1951): each period the agent orders S before demand d is known; leftover
    stock costs H_OVER = 1 and shortage P_SHORT = 4 per unit, so with a known demand distribution the best order is
    its 0.8-quantile. The mean demand mu_t changes over time (process families below); d_t = max(0, mu_t + sigma e_t);
    the agent observes y_t = d_t + tau u_t, but only with probability `avail` (otherwise the period's demand is never
    observed). Orders use observations up to t - 1.

Manipulated factors (Config)
    sigma     outcome volatility (s.d. of demand around its mean)
    tau       observation error
    hazard    regime-change frequency
    avail     feedback availability (probability that a period's demand is observed)
    memory    the reliability gates' memory lambda (their effective window is about (1 + lambda) / (1 - lambda))
    default   default quality: 'slow' (slow forecast, gain 0.01), 'fixed_initial' (the best order for the first
              regime, which deteriorates after the first change), 'biased' (40 units too low), 'dominated' (orders 0)
    cost      default-departure overhead: charged in every period whose order differs from the default's
    switch_cost     fixed switching cost: charged in every period whose order differs from the PREVIOUS period's
    magnitude_cost  charged per unit of |S_t - S_{t-1}|
    cost_model      "components" (all three above) or "legacy_departure_only" (see below)

Three distinct costs, deliberately not interchangeable
    Departing from a default, changing your mind, and changing by a lot are different economic frictions and are
    modeled as three separate charges. They are *not* three parameterizations of one construct, and nothing here
    assumes they stand for the same thing:
        departure  c_dep * 1[S_t != S_D,t]        an overhead for not being on the default (for example the cost of
                                                  running a process that is not the standard one)
        switching  c_sw  * 1[S_t != S_{t-1}]      a fixed cost of changing the action at all (a setup or changeover)
        magnitude  c_mag * |S_t - S_{t-1}|        a cost proportional to how far the action moved
    The distinction matters: an agent that holds a constant non-default order pays the departure overhead every
    period and nothing for switching or magnitude, while an agent that tracks a moving default pays no departure
    overhead but pays for every change. Before 9 October 2026 only the departure overhead existed and the same
    indicator also *defined* the reported adaptation rate, so "how often the policy departed from the default" and
    "how often the policy actually changed its order" could not be told apart. They are now reported separately as
    `departure_rate` and `adjustment_rate`. `Config.cost_model = "legacy_departure_only"` restores the old
    calculation exactly (the two new charges are ignored, whatever they are set to).

Conventions fixed once, so the components are comparable
    tolerance     two orders count as different when they differ by more than ACTION_TOL (1e-9); the same tolerance
                  defines departure, switching and the "missed opportunity" accounting
    first period  period 0 has no previous action, so it is charged no switching and no magnitude cost; it is
                  charged the departure overhead if its order differs from that period's default
    burn-in       costs are charged in every period, and the reported means cover the recorded periods t >= burn_in.
                  The switching and magnitude charges at t = burn_in compare with the order in period burn_in - 1,
                  which is a real decision, so no recorded period is charged against an undefined predecessor.
    objective     tuning scores exactly the evaluation objective, `net_payoff` = gross payoff minus all charges that
                  the configuration switches on, so a policy is never tuned against a different economic objective
                  from the one it is judged by.

Process families (Config.family): every family starts at mu = 100 and keeps mu in [30, 170]
    jump      with probability `hazard` the mean jumps by N(0, 30^2) (tuning and the main test)
    switch    with probability `hazard` the mean switches between 70 and 130 (new family, test only)
    drift     the mean follows a random walk with per-period s.d. 30 sqrt(hazard), the same variance per expected regime
              length as `jump`, without discrete regimes (new family, test only)

Policies (all see the same exogenous paths: common random numbers)
    always      always adapt: order the fast forecast's 0.8-quantile, S_F = forecast + z * sd (gain tuned)
    default     retain the default S_D
    band        inaction band: S_F if |S_F - S_D| > b * sd, else S_D (b tuned)
    gate_gain   existing estimated-gain gate (heiner_abm.gates): per size bin of |S_F - S_D| / sd, adapt if the
                learned mean of g = payoff(S_F, y) - payoff(S_D, y) is at least c
    gate_lcb    confidence-sensitive gate: adapt if the bin has at least `nmin` effective observations and the
                one-sided lower bound at confidence `conf` exceeds c; (conf, nmin) are tuned on the training
                configurations with the same budget as every other tunable policy (both gates learn from every
                observed period, whatever they chose: the observed demand values both orders; feedback is released
                at the end of the period)
    bocpd       Bayesian change detection (Adams & MacKay 2007), misspecified Gaussian reset model, tuned; orders its
                predictive 0.8-quantile; it does not use the default
    dro         distributionally robust newsvendor on the last N observed demands (modified chi-squared ball,
                Ben-Tal et al. 2013), N and rho tuned
    oracle      PERFECT INFORMATION (knows mu_t): order max(0, mu_t + z sigma); a bound, not ranked
Every policy pays c whenever its order differs from the default's.

Equal tuning budget
    Every policy with free hyperparameters gets the same number of candidate settings (len of its entry in
    LearnPlan.tuning_grid, four by default), each scored on the same training configurations and paths, and the best
    is carried into the pilot and test runs: `always` (forecast gain), `band` (b), `gate_lcb` (conf, nmin), `bocpd`
    (hazard, prior sd, obs sd) and `dro` (window, rho). `default` and `gate_gain` have no free hyperparameters, so
    they receive none; `oracle` is a bound, not a competitor. tune() returns a log with one row per policy giving the
    number of candidates evaluated, so the equality is checkable rather than asserted. Before 7 October 2026 the two
    gates received no tuning at all while their four comparators received four candidates each, and LearnPlan's
    `gate_confidence` / `gate_min_evidence` were never read: see LearnPlan.tune_gate for the legacy setting.

Outcomes (per path, recorded periods t >= burn_in)
    primary     net payoff per period = gross payoff - departure - switching - magnitude charges, where the gross
                payoff is -(H_OVER (S - d)+ + P_SHORT (d - S)+). Reported alongside it: gross_payoff, the three
                charges separately (cost_departure, cost_switching, cost_magnitude, cost_total), the share of
                periods departing from the default (departure_rate), the share actually changing the order
                (adjustment_rate) and the mean |S_t - S_{t-1}| (adjustment_magnitude)
    secondary   regret (oracle's gross payoff minus net payoff); downside loss (mean of the worst 5% of periods' net
                payoff, CVaR 5%); calibration of the gates' predicted advantage (slope and bias against the realized
                observed gain); adaptation rate (share of periods with S != S_D); missed opportunities (share of
                periods whose true gain of S_F over S_D exceeds c in which the policy did not order S_F; policies that
                choose between S_D and S_F only); recovery delay (periods after a change until the policy's 20-period
                rolling excess loss over the oracle falls to its own median outside the 50 periods after changes;
                censored at the next change)

Proposed construct: the learnability ratio R (measured on independent pilot paths, never on test paths)
    For each regime of a pilot path in which the two candidate orders differ (for `drift`, consecutive blocks of the
    nominal length 1 / hazard):
        mu_r     the regime's true mean advantage of S_F over S_D (from the pilot's true demands: researcher data)
        s        the noise of what the learner observes: s.d. of observed gains around their regime's mu_r, pooled
        n_needed = (z_0.9 s / |mu_r|)^2   observations to determine the sign of the regime's advantage with 90%
                                           one-sided confidence (the specified precision), clipped to [1, 1e6]
        n_avail  = informative observations in the regime (demand observed and S_F != S_D)
    R = median over regimes of n_needed / max(n_avail, 1). R < 1: the sign of the advantage of adapting can typically
    be learned within a regime; R > 1: it cannot. R is a proposed construct: whether it explains performance beyond
    volatility and observation noise is a hypothesis of the study (H2), not an assumption.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import gates
from .bench_inventory import (BOCPD, FRACTILE, H_OVER, P_SHORT, Z_FRAC, dro_orders_window, gaussian_mixture_quantile,
                              newsvendor_loss)

MU0, MU_LO, MU_HI, JUMP_SD, SWITCH_LEVELS = 100.0, 30.0, 170.0, 30.0, (70.0, 130.0)
SLOW_GAIN = 0.01
EDGES = np.array([0.5, 1.0, 2.0, 4.0])                 # size bins of |S_F - S_D| / sd for the gates
FAMILIES = ("jump", "switch", "drift")
DEFAULTS = ("slow", "fixed_initial", "biased", "dominated")
POLICIES = ("always", "default", "band", "gate_gain", "gate_lcb", "bocpd", "dro", "oracle")
POLICY_LABELS = {"always": "Always adapt", "default": "Retain the default", "band": "Inaction band",
                 "gate_gain": "Estimated-gain gate", "gate_lcb": "Confidence-sensitive gate",
                 "bocpd": "Bayesian change detection", "dro": "Distributionally robust", "oracle": "ORACLE (perfect information)"}
CHOOSERS = ("always", "default", "band", "gate_gain", "gate_lcb")   # choose between S_D and S_F
Z90 = 1.2815515655446004
# Candidate 0 of the lcb gate: the settings used before the gate was given a tuning budget (7 October 2026).
GATE_HP0 = {"conf": 0.9, "nmin": 5.0}
# NC3 noninferiority margin (net payoff per period). The control asks whether the confidence-sensitive gate is
# *no worse than* the better fixed rule by more than this in an easy environment; see negative_controls().
NC3_MARGIN = 0.25
# Two orders count as different when they differ by more than this. One tolerance for every comparison:
# departure from the default, switching from the previous order, and the missed-opportunity accounting.
ACTION_TOL = 1e-9
COST_MODELS = ("components", "legacy_departure_only")
# Policies with free hyperparameters, and the key of their entry in LearnPlan.tuning_grid. Every one of them receives
# the same number of candidate settings; the rest (default, gate_gain, oracle) have no hyperparameters to tune.
TUNABLE = {"always": "gain", "band": "band_b", "gate_lcb": "gate", "bocpd": "bocpd", "dro": "dro"}


@dataclass(frozen=True)
class Config:
    family: str = "jump"
    sigma: float = 20.0
    tau: float = 10.0
    hazard: float = 0.01
    avail: float = 1.0
    memory: float = 0.97
    default: str = "fixed_initial"
    cost: float = 1.0              # default-departure overhead, charged when S_t != S_D,t
    switch_cost: float = 0.0       # fixed switching cost, charged when S_t != S_{t-1}
    magnitude_cost: float = 0.0    # charged per unit of |S_t - S_{t-1}|
    cost_model: str = "components"  # or "legacy_departure_only": the departure overhead alone, as before 9 Oct 2026
    perfect: bool = False          # negative control: the flexible order is the oracle's order

    def __post_init__(self):
        if self.cost_model not in COST_MODELS:
            raise ValueError(f"Unknown cost model {self.cost_model!r} (one of {', '.join(COST_MODELS)}).")

    @property
    def zero_cost(self) -> bool:
        """True when no charge can apply, whatever the cost model: the zero-cost control."""
        return not (self.cost or (self.cost_model == "components" and (self.switch_cost or self.magnitude_cost)))


# ================================================================================================ environment
def paths(cfg: Config, T: int, seed: int) -> Dict[str, np.ndarray]:
    """One exogenous path: mean demand, demand, observation (NaN if unobserved) and change indicators."""
    ss = np.random.SeedSequence(seed).spawn(4)
    r_reg, r_dem, r_obs, r_av = (np.random.default_rng(s) for s in ss)
    u, z = r_reg.random(T), r_reg.standard_normal(T)
    mu, change = np.empty(T), np.zeros(T, bool)
    m = MU0
    lvl = 0
    for t in range(T):
        if cfg.family == "jump":
            if u[t] < cfg.hazard:
                m, change[t] = min(MU_HI, max(MU_LO, m + JUMP_SD * z[t])), True
        elif cfg.family == "switch":
            if t == 0:
                m = MU0
            if u[t] < cfg.hazard:
                lvl = 1 - lvl if t > 0 and m != MU0 else int(z[t] > 0)
                m, change[t] = SWITCH_LEVELS[lvl], True
        elif cfg.family == "drift":
            m = min(MU_HI, max(MU_LO, m + JUMP_SD * np.sqrt(cfg.hazard) * z[t]))
        else:
            raise ValueError(cfg.family)
        mu[t] = m
    d = np.maximum(0.0, mu + cfg.sigma * r_dem.standard_normal(T))
    y = d + cfg.tau * r_obs.standard_normal(T)
    seen = r_av.random(T) < cfg.avail
    return dict(mu=mu, d=d, y=np.where(seen, y, np.nan), seen=seen, change=change)


def payoff(S, d):
    """Gross payoff: the newsvendor payoff before any adaptation charge."""
    return -newsvendor_loss(S, d)


def cost_components(cfg: Config, S: np.ndarray, S_D: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-period charges for a path of orders, as three separate economic frictions.

    departure  cfg.cost           each period the order differs from that period's default
    switching  cfg.switch_cost    each period the order differs from the previous period's order
    magnitude  cfg.magnitude_cost per unit of |S_t - S_{t-1}|

    Period 0 has no predecessor, so it carries no switching and no magnitude charge; it does carry the departure
    overhead if it departs. Under cost_model "legacy_departure_only" only the departure overhead is charged and the
    other two are ignored however they are set, reproducing the calculation used before 9 October 2026.

    Also returns the two behavioral indicators that used to be the same array: `departs` (differs from the default)
    and `changes` (differs from the previous order). Conflating them is what made "adaptation rate" ambiguous.
    """
    S, S_D = np.asarray(S, float), np.asarray(S_D, float)
    departs = np.abs(S - S_D) > ACTION_TOL
    step = np.zeros_like(S)
    step[1:] = np.abs(S[1:] - S[:-1])                    # period 0 has no previous action
    changes = step > ACTION_TOL
    legacy = cfg.cost_model == "legacy_departure_only"
    dep = cfg.cost * departs
    sw = np.zeros_like(S) if legacy else cfg.switch_cost * changes
    mag = np.zeros_like(S) if legacy else cfg.magnitude_cost * step
    return dict(departs=departs, changes=changes, step=step,
                departure=dep, switching=sw, magnitude=mag, total=dep + sw + mag)


# ================================================================================================ candidate orders
def candidate_orders(cfg: Config, p: Dict[str, np.ndarray], gain: float) -> Dict[str, np.ndarray]:
    """Default order S_D, flexible order S_F and the agent's demand s.d. estimate, from observations up to t - 1."""
    y, T = p["y"], len(p["y"])
    slow, fast, sd = np.empty(T), np.empty(T), np.empty(T)
    ms, mf, v = MU0, MU0, cfg.sigma ** 2 + cfg.tau ** 2 + 1.0     # starting variance guess
    for t in range(T):
        slow[t], fast[t], sd[t] = ms, mf, np.sqrt(max(v, 1.0))
        o = y[t]
        if not np.isnan(o):
            v += SLOW_GAIN * ((o - ms) ** 2 - v)
            ms += SLOW_GAIN * (o - ms)
            mf += gain * (o - mf)
    if cfg.perfect:
        S_F = np.maximum(0.0, p["mu"] + Z_FRAC * cfg.sigma)
    else:
        S_F = np.maximum(0.0, fast + Z_FRAC * sd)
    S_D = {"slow": np.maximum(0.0, slow + Z_FRAC * sd),
           "fixed_initial": np.full(T, MU0 + Z_FRAC * cfg.sigma),
           "biased": np.full(T, MU0 - 40.0 + Z_FRAC * cfg.sigma),
           "dominated": np.zeros(T)}[cfg.default]
    return dict(S_F=S_F, S_D=S_D, sd=sd)


# ================================================================================================ policies
def gated(cfg: Config, p, c: Dict[str, np.ndarray], gate: str, conf: float = GATE_HP0["conf"],
          nmin: float = GATE_HP0["nmin"]):
    """Orders of a reliability gate (heiner_abm.gates) and its prediction per period (NaN without evidence).

    conf and nmin are the confidence and the minimum effective evidence of the `lcb` gate; the `gain` gate ignores
    them. Callers pass the tuned values (hyperparameters["gate"]); the defaults are the registered candidate 0."""
    S_F, S_D, sd, y = c["S_F"], c["S_D"], c["sd"], p["y"]
    T = len(y)
    nb = len(EDGES) + 1
    learned, S, XS = np.zeros(nb), np.zeros((gates.N_STATS, nb)), np.zeros((2, gates.N_STATS, nb))
    lam = cfg.memory
    adapt = np.zeros(T, bool)
    pred = np.full(T, np.nan)
    gain_obs = np.full(T, np.nan)
    b_all = np.searchsorted(EDGES, np.abs(S_F - S_D) / sd, side="right")
    for t in range(T):
        differ = S_F[t] != S_D[t]
        if differ:
            # The gate weighs the learned gain against the DEPARTURE overhead only: it chooses between the
            # default and the flexible order for this period and does not look ahead to switching or
            # magnitude charges, which depend on the order it happens to be holding. A stated limitation,
            # not an equivalence between the three charges.
            tab = gates.gate_table(gate, learned, S, XS, lam=lam, cost=cfg.cost, conf=conf, nmin=nmin)
            b = b_all[t]
            adapt[t] = tab["mode"][b] == gates.ADAPT
            pred[t] = tab["pred"][b]
            if not np.isnan(y[t]):                         # released at the end of period t
                g = payoff(S_F[t], y[t]) - payoff(S_D[t], y[t])
                gain_obs[t] = g
                learned[b] = lam * learned[b] + (1 - lam) * g
                S[:, b] = gates.update(S[:, b], g, lam)
    return np.where(adapt, S_F, S_D), pred, gain_obs


def bocpd_orders(y: np.ndarray, hazard: float, s0: float, sm: float, rmax: int = 150) -> np.ndarray:
    """Orders from the Bayesian change-point posterior predictive, one per period.

    Every period advances calendar time by exactly one hazard transition, whether or not its demand was observed:
    an observed period conditions on the demand (update), an unobserved one marginalizes over it (advance). Before
    8 October 2026 unobserved periods were skipped entirely, so a run of missing demands left the run-length
    posterior frozen and the model behaved as if no time, and so no regime risk, had passed."""
    model = BOCPD(hazard, MU0, s0, sm, rmax)
    out = np.empty(len(y))
    for t in range(len(y)):
        out[t] = max(0.0, gaussian_mixture_quantile(*model.predictive()))
        if np.isnan(y[t]):
            model.advance()
        else:
            model.update(float(y[t]))
    return out


def dro_orders(y: np.ndarray, N: int, rho: float, fallback: np.ndarray) -> np.ndarray:
    """Robust order from the last N observed demands (fewer at the start; the default's order before two exist)."""
    T = len(y)
    obs_idx = np.flatnonzero(~np.isnan(y))
    yo = y[obs_idx]
    k = np.searchsorted(obs_idx, np.arange(T))                   # observations before period t
    S = np.array(fallback, float)
    for t in np.flatnonzero((k >= 2) & (k < N)):
        S[t] = dro_orders_window(yo[None, :k[t]], rho)[0]
    full = np.flatnonzero(k >= N)
    if len(full):
        from numpy.lib.stride_tricks import sliding_window_view
        W = sliding_window_view(yo, N)[k[full] - N]
        S[full] = dro_orders_window(W, rho)
    return S


def policy_orders(policy: str, cfg: Config, p, c, hp: Dict[str, Dict[str, float]]):
    """Orders of one policy; returns (orders, gate prediction or None, observed gain or None)."""
    if policy == "always":
        return c["S_F"], None, None
    if policy == "default":
        return c["S_D"], None, None
    if policy == "band":
        dev = np.abs(c["S_F"] - c["S_D"]) > hp["band"]["b"] * c["sd"]
        return np.where(dev, c["S_F"], c["S_D"]), None, None
    if policy in ("gate_gain", "gate_lcb"):
        g = hp.get("gate", GATE_HP0)
        return gated(cfg, p, c, policy[5:], conf=float(g["conf"]), nmin=float(g["nmin"]))
    if policy == "bocpd":
        h = hp["bocpd"]
        return bocpd_orders(p["y"], h["hazard"], h["prior_sd"], h["obs_sd"]), None, None
    if policy == "dro":
        return dro_orders(p["y"], int(hp["dro"]["window"]), hp["dro"]["rho"], c["S_D"]), None, None
    if policy == "oracle":
        return np.maximum(0.0, p["mu"] + Z_FRAC * cfg.sigma), None, None
    raise ValueError(policy)


# ================================================================================================ outcomes
def _recovery_delay(excess: np.ndarray, change: np.ndarray, burn: int, window: int = 20) -> float:
    T = len(excess)
    if T < window + burn:
        return np.nan
    roll = np.convolve(excess, np.ones(window) / window, mode="valid")          # roll[t] = mean excess[t:t+window]
    cps = [t for t in np.flatnonzero(change) if t >= burn and t + window < T]
    if not cps:
        return np.nan
    near = np.zeros(len(roll), bool)
    for t0 in np.flatnonzero(change):
        near[max(0, t0):t0 + 50] = True
    base = np.median(roll[~near]) if (~near).any() else np.median(roll)
    delays = []
    for j, t0 in enumerate(cps):
        nxt = next((t for t in np.flatnonzero(change) if t > t0), T)
        seg = roll[t0:min(nxt, len(roll))]
        hit = np.flatnonzero(seg <= base)
        delays.append(float(hit[0]) if len(hit) else float(len(seg)))       # censored at the next change
    return float(np.mean(delays))


def outcomes(policy: str, cfg: Config, p, c, S, pred, gain_obs, burn: int) -> Dict[str, float]:
    """Per-path outcomes over the recorded periods t >= burn.

    Reports the gross payoff, each cost component, and the net payoff separately, and keeps the two behavioral
    rates apart: `departure_rate` (the order differs from the default) and `adjustment_rate` (the order differs
    from the previous order). `net_payoff` is the primary outcome and is exactly what tuning scores.
    """
    d, S_D, S_F = p["d"], c["S_D"], c["S_F"]
    oracle = np.maximum(0.0, p["mu"] + Z_FRAC * cfg.sigma)
    k = cost_components(cfg, S, S_D)
    gross = payoff(S, d)
    net = gross - k["total"]
    w = slice(burn, None)
    regret = payoff(oracle, d) - net
    worst = np.sort(net[w])[:max(1, int(0.05 * len(net[w])))]
    out = dict(net_payoff=float(net[w].mean()), gross_payoff=float(gross[w].mean()),
               cost_departure=float(k["departure"][w].mean()), cost_switching=float(k["switching"][w].mean()),
               cost_magnitude=float(k["magnitude"][w].mean()), cost_total=float(k["total"][w].mean()),
               departure_rate=float(k["departs"][w].mean()), adjustment_rate=float(k["changes"][w].mean()),
               adjustment_magnitude=float(k["step"][w].mean()),
               regret=float(regret[w].mean()), cvar5=float(worst.mean()))
    if policy in CHOOSERS:
        # A missed opportunity is one where adopting the flexible order would have been worth its charges. Both
        # branches are priced against the order actually held in the previous period, so the switching and
        # magnitude charges of adopting and of keeping are compared like for like.
        prev = np.concatenate(([S[0]], S[:-1]))
        legacy = cfg.cost_model == "legacy_departure_only"
        sw, mag = (0.0, 0.0) if legacy else (cfg.switch_cost, cfg.magnitude_cost)
        def _charge(x, departs_from_default):
            step = np.abs(x - prev)
            step[0] = 0.0                                  # the first period has no predecessor
            return (cfg.cost * departs_from_default + sw * (step > ACTION_TOL) + mag * step)
        net_adopt = payoff(S_F, d) - _charge(S_F, np.abs(S_F - S_D) > ACTION_TOL)
        net_keep = payoff(S_D, d) - _charge(S_D, np.zeros_like(S_D, dtype=bool))
        opp = (net_adopt > net_keep)[w]
        took = (np.abs(S - S_F) < ACTION_TOL)[w]
        out["missed"] = float((~took[opp]).mean()) if opp.any() else np.nan
    else:
        out["missed"] = np.nan
    if pred is not None:
        m = ~np.isnan(pred[w]) & ~np.isnan(gain_obs[w])
        x, yv = pred[w][m], gain_obs[w][m]
        out["calib_bias"] = float((x - yv).mean()) if m.sum() else np.nan
        out["calib_slope"] = float(np.polyfit(x, yv, 1)[0]) if m.sum() > 2 and np.ptp(x) > 0 else np.nan
    else:
        out["calib_bias"] = out["calib_slope"] = np.nan
    excess = (payoff(oracle, d) - net)
    out["recovery_delay"] = _recovery_delay(excess, p["change"], burn) if cfg.family != "drift" else np.nan
    return out


# ================================================================================================ ratio construct
def learnability_ratio(cfg: Config, gain: float, T: int, seeds: Sequence[int], burn: int) -> Dict[str, float]:
    """The proposed learnability ratio R, from independent pilot paths (see the module docstring)."""
    regimes, resid = [], []
    for sd_ in seeds:
        p = paths(cfg, T, sd_)
        c = candidate_orders(cfg, p, gain)
        differ = c["S_F"] != c["S_D"]
        g_true = payoff(c["S_F"], p["d"]) - payoff(c["S_D"], p["d"])           # researcher's view (pilot only)
        g_obs = payoff(c["S_F"], p["y"]) - payoff(c["S_D"], p["y"])            # what the learner sees (NaN if unseen)
        info = differ & ~np.isnan(g_obs)
        if cfg.family == "drift":
            L = max(20, int(round(1 / max(cfg.hazard, 1e-6))))
            bounds = list(range(burn, T, L)) + [T]
        else:
            bounds = sorted(set([burn] + [t for t in np.flatnonzero(p["change"]) if t > burn] + [T]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            d = differ[a:b]
            if not d.any():
                continue                                                       # nothing to decide in this regime
            mu_r = float(g_true[a:b][d].mean())
            go = g_obs[a:b][info[a:b]]
            resid.extend(go - mu_r)
            regimes.append((mu_r, len(go)))
    if not regimes:
        return dict(n_needed=np.nan, n_avail=np.nan, ratio=np.nan, log10_ratio=np.nan, noise_sd=np.nan, regimes=0)
    s = float(np.std(resid, ddof=1)) if len(resid) > 1 else np.nan
    need = np.array([np.clip((Z90 * s / max(abs(m), 1e-9)) ** 2, 1.0, 1e6) for m, _ in regimes])
    avail = np.array([n for _, n in regimes], float)
    R = float(np.median(need / np.maximum(avail, 1.0)))
    return dict(n_needed=float(np.median(need)), n_avail=float(np.median(avail)), ratio=R,
                log10_ratio=float(np.log10(R)), noise_sd=s, regimes=len(regimes))


# ================================================================================================ plan
HYPOTHESES = (
    ("H1", "Learnability: the confidence-sensitive gate's advantage over the better fixed rule (always adapt or retain "
           "the default, whichever was better on pilot paths) falls as the learnability ratio R rises.",
     "Across the jump-family test configurations, OLS slope of the configuration-mean advantage (net payoff per "
     "period) on log10 R; 95% bootstrap CI over configurations. Supported if the CI lies below 0."),
    ("H2", "Incremental validity of R: log10 R explains the gate's advantage beyond outcome volatility and observation "
           "noise.",
     "Leave-one-configuration-out cross-validated R² of advantage ~ log sigma + log(1 + tau) + log10 R minus that of "
     "advantage ~ log sigma + log(1 + tau); 95% bootstrap CI over configurations. Supported if the CI lies above 0."),
    ("H3", "Transfer to new process families: the H1 relation holds when R is measured and the gate is tested on the "
           "switching and drifting families, which were not used for tuning.",
     "OLS slope of the advantage on log10 R pooled over the two new families, 95% bootstrap CI over configurations. "
     "Supported if the CI lies below 0."),
    ("H4", "Replication in the market: the same relation holds for the market's confidence-sensitive gate "
           "(heiner_abm.learnability_market).",
     "As H1, in the market configurations. Supported if the 95% bootstrap CI of the slope lies below 0."),
)
NEGATIVE_CONTROLS = (
    ("NC1", "Perfect information and reversible, costless adaptation",
     "The flexible order is the oracle's order and c = 0. Expected: no policy beats always adapting. Passes if no "
     "policy's paired net-payoff difference over always adapting has a 95% CI lower bound above 0.25 per period."),
    ("NC2", "Dominated default",
     "The default orders nothing. Expected: the gates learn to adapt. Passes if both gates adapt in at least 90% of "
     "periods and lose no more than 5% of always adapting's net payoff."),
    ("NC3", "Stable environment with abundant feedback",
     "No regime changes, every demand observed, little noise, 3,000 periods. Expected: R < 1 and the "
     "confidence-sensitive gate is at least as good as the better fixed rule. Estimand: the gate's mean paired "
     "advantage over that rule, in net payoff per period. Noninferiority test at a prespecified margin of 0.25 per "
     "period: passes if R < 1 and the *lower* limit of the 95% bootstrap CI of the advantage exceeds −0.25, so "
     "that the gate being worse by more than the margin is ruled out. Until 8 October 2026 the rule compared the "
     "*upper* limit with −0.25, which only fails when the gate is confidently worse and so established nothing."),
)


@dataclass(frozen=True)
class LearnPlan:
    version: str = "1.0"
    periods: int = 1500
    burn_in: int = 100
    paths: int = 10                          # independent paths per test configuration
    pilot_paths: int = 6                     # independent pilot paths per configuration (ratio and best fixed rule)
    n_train_configs: int = 8                 # training configurations (jump family) for tuning
    train_paths: int = 3
    n_test_configs: int = 30                 # Latin-hypercube test configurations
    seed: int = 20261007
    baseline: Tuple[Tuple[str, object], ...] = tuple(asdict(Config()).items())
    sweeps: Tuple[Tuple[str, Tuple], ...] = (
        ("sigma", (5.0, 20.0, 40.0)), ("tau", (0.0, 10.0, 30.0)), ("hazard", (0.002, 0.01, 0.05)),
        ("avail", (1.0, 0.5, 0.2)), ("memory", (0.9, 0.97, 0.995)),
        ("default", ("slow", "fixed_initial", "biased")))
    lhs_ranges: Tuple[Tuple[str, Tuple[float, float]], ...] = (
        ("sigma", (5.0, 40.0)), ("tau", (0.0, 30.0)), ("hazard", (0.002, 0.05)), ("avail", (0.2, 1.0)),
        ("memory", (0.9, 0.995)))
    new_families: Tuple[str, ...] = ("switch", "drift")
    n_new_family_configs: int = 15
    tuning_grid: Tuple[Tuple[str, Tuple], ...] = (
        ("gain", (0.05, 0.1, 0.2, 0.4)), ("band_b", (0.25, 0.5, 1.0, 2.0)),
        ("gate", ((0.9, 5.0), (0.8, 3.0), (0.9, 12.0), (0.95, 25.0))),
        ("bocpd", ((0.005, 30.0, 25.0), (0.01, 40.0, 25.0), (0.02, 40.0, 20.0), (0.005, 60.0, 35.0))),
        ("dro", ((20, 0.01), (40, 0.01), (40, 0.1), (80, 0.05))))
    tune_gate: bool = True                   # False = legacy: the gates are not tuned and use the two fields below
    gate_confidence: float = 0.9             # confidence of the lcb gate when tune_gate is False (and candidate 0)
    gate_min_evidence: float = 5.0           # minimum effective evidence of the lcb gate when tune_gate is False
    n_boot: int = 2000
    hypotheses: Tuple = HYPOTHESES
    negative_controls: Tuple = NEGATIVE_CONTROLS

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "code": code_digest()}, sort_keys=True, indent=1, default=str)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def code_digest() -> str:
    """Hash of this study's code and of the gate and benchmark code it uses; changing any of it changes the plan hash."""
    from . import bench_inventory, engine, learnability_market, params
    mods = [sys.modules[__name__], gates, learnability_market, engine, params]
    src = "".join(inspect.getsource(m) for m in mods)
    src += "".join(inspect.getsource(f) for f in (bench_inventory.BOCPD, bench_inventory.gaussian_mixture_quantile,
                                                   bench_inventory.chi2_worst_case, bench_inventory.dro_objective,
                                                   bench_inventory.dro_orders_window, bench_inventory.newsvendor_loss))
    return hashlib.sha256(src.encode().replace(b"\r\n", b"\n")).hexdigest()


QUICK_LEARN = dict(periods=500, burn_in=50, paths=3, pilot_paths=2, n_train_configs=3, train_paths=2,
                   n_test_configs=6, n_new_family_configs=3, n_boot=300)


# ================================================================================================ running
def _lhs_configs(plan: LearnPlan, n: int, seed: int, family: str = "jump") -> List[Config]:
    rng = np.random.default_rng(seed)
    base = Config(**dict(plan.baseline))
    cols = {}
    for k, (lo, hi) in plan.lhs_ranges:
        u = (rng.permutation(n) + rng.random(n)) / n
        cols[k] = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if k == "hazard" else lo + u * (hi - lo)
    return [replace(base, family=family, **{k: float(v[j]) for k, v in cols.items()}) for j in range(n)]


def _seeds(plan: LearnPlan, tag: int, j: int, n: int) -> List[int]:
    """Disjoint seed blocks: tag 1 training, 2 pilot, 3 test, 4 negative controls."""
    return [plan.seed * 1000 + tag * 100_000_000 + j * 1000 + r for r in range(n)]


def run_config(cfg: Config, hp, seeds: Sequence[int], T: int, burn: int,
               policies: Sequence[str] = POLICIES) -> pd.DataFrame:
    rows = []
    for sd_ in seeds:
        p = paths(cfg, T, sd_)
        c = candidate_orders(cfg, p, hp["gain"])
        for pol in policies:
            S, pred, gobs = policy_orders(pol, cfg, p, c, hp)
            rows.append(dict(seed=sd_, policy=pol, **outcomes(pol, cfg, p, c, S, pred, gobs, burn)))
    return pd.DataFrame(rows)


def tune(plan: LearnPlan, progress=None) -> Tuple[Dict, pd.DataFrame]:
    """Hyperparameters chosen on training configurations only (jump family, seeds disjoint from pilot and test).

    Every tunable policy (TUNABLE) gets the same number of candidate settings from LearnPlan.tuning_grid, scored on
    the same training configurations and paths; candidate 0 is the registered starting value. The returned log has one
    row per policy with the number of candidates evaluated and the chosen setting, so the budgets can be compared.
    Policies are tuned in order: the forecast gain first (it defines the candidate orders every chooser uses), then
    the band, the confidence-sensitive gate, change detection and the robust order.

    With plan.tune_gate = False the gate is not tuned (legacy behavior up to 6 October 2026); it then uses
    plan.gate_confidence and plan.gate_min_evidence, and its budget is logged as 0."""
    cfgs = _lhs_configs(plan, plan.n_train_configs, plan.seed + 1)
    grid = dict(plan.tuning_grid)
    T, burn = plan.periods, plan.burn_in
    log = []

    def score(policy, hp):
        tot = []
        for j, cfg in enumerate(cfgs):
            df = run_config(cfg, hp, _seeds(plan, 1, j, plan.train_paths), T, burn, (policy,))
            tot.append(df["net_payoff"].mean())
        return float(np.mean(tot))

    def pick(policy, key, make, note):
        """Score every candidate of `policy` and keep the best; `make` turns a grid entry into its hyperparameters."""
        cands = list(grid[key])
        scores = [score(policy, {**hp, **make(v)}) for v in cands]
        b = int(np.argmax(scores))
        log.append(dict(policy=policy, parameters=key, n_candidates=len(cands), chosen=str(cands[b]),
                        candidate0_score=float(scores[0]), chosen_score=float(scores[b]),
                        gain_over_candidate0=float(scores[b] - scores[0])))
        hp.update(make(cands[b]))
        if progress:
            progress(note[0], note[1])

    hp = {"gain": grid["gain"][0], "band": {"b": grid["band_b"][0]},
          "gate": dict(zip(("conf", "nmin"), grid["gate"][0])),
          "bocpd": dict(zip(("hazard", "prior_sd", "obs_sd"), grid["bocpd"][0])),
          "dro": dict(zip(("window", "rho"), grid["dro"][0]))}
    pick("always", "gain", lambda g: {"gain": g}, (0.2, "Tuned the flexible forecast"))
    pick("band", "band_b", lambda b: {"band": {"b": b}}, (0.4, "Tuned the inaction band"))
    if plan.tune_gate:
        pick("gate_lcb", "gate", lambda v: {"gate": dict(zip(("conf", "nmin"), v))},
             (0.6, "Tuned the confidence-sensitive gate"))
    else:
        hp["gate"] = {"conf": float(plan.gate_confidence), "nmin": float(plan.gate_min_evidence)}
        log.append(dict(policy="gate_lcb", parameters="gate", n_candidates=0, chosen=str(tuple(hp["gate"].values())),
                        candidate0_score=np.nan, chosen_score=np.nan, gain_over_candidate0=np.nan))
    pick("bocpd", "bocpd", lambda v: {"bocpd": dict(zip(("hazard", "prior_sd", "obs_sd"), v))},
         (0.8, "Tuned change detection"))
    pick("dro", "dro", lambda v: {"dro": dict(zip(("window", "rho"), v))}, (1.0, "Tuned the robust order"))
    return hp, pd.DataFrame(log)


def _boot_ci(x: np.ndarray, n_boot: int, seed: int = 0) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired(df: pd.DataFrame, a: str, b: str, col: str = "net_payoff", n_boot: int = 2000):
    """Mean paired difference a − b over shared paths with a 95% bootstrap CI."""
    w = df.pivot_table(index="seed", columns="policy", values=col)
    return _boot_ci((w[a] - w[b]).to_numpy(), n_boot)


def evaluate_config(cfg: Config, hp, plan: LearnPlan, j: int, tag: int = 3) -> Dict:
    """Test paths, independent pilot (ratio, better fixed rule), effect sizes."""
    T, burn = plan.periods, plan.burn_in
    pilot = run_config(cfg, hp, _seeds(plan, 2, j, plan.pilot_paths), T, burn, ("always", "default"))
    pm = pilot.groupby("policy")["net_payoff"].mean()
    best_fixed = "always" if pm["always"] >= pm["default"] else "default"
    ratio = learnability_ratio(cfg, hp["gain"], T, _seeds(plan, 2, j, plan.pilot_paths), burn)
    test = run_config(cfg, hp, _seeds(plan, tag, j, plan.paths), T, burn)
    eff = {}
    for pol in ("gate_lcb", "gate_gain", "band", "bocpd", "dro"):
        eff[f"adv_{pol}"] = paired(test, pol, best_fixed, n_boot=plan.n_boot)
    return dict(config=asdict(cfg), best_fixed=best_fixed, ratio=ratio, test=test, effects=eff)


def _slope_ci(x, y, n_boot, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    b = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0:
            bs.append(np.polyfit(x[i], y[i], 1)[0])
    return b, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _cv_r2(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> float:
    """Leave-one-group-out cross-validated R² of an OLS fit with intercept (groups: original configuration ids, so a
    bootstrap duplicate is never in both the fitting and the held-out set)."""
    n = len(y)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    pred = np.empty(n)
    A = np.column_stack([np.ones(n), X])
    for g in np.unique(groups):
        m = groups != g
        beta = np.linalg.lstsq(A[m], y[m], rcond=None)[0]
        pred[~m] = A[~m] @ beta
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def incremental_validity(df: pd.DataFrame, target: str, n_boot: int, seed: int = 0) -> Tuple[float, float, float]:
    """Cross-validated R² gain from adding log10 R to log sigma + log(1 + tau), with a bootstrap CI over
    configurations (grouped cross-validation inside every bootstrap sample)."""
    base = np.column_stack([np.log(df["sigma"]), np.log1p(df["tau"])])
    full = np.column_stack([base, df["log10_ratio"]])
    y = df[target].to_numpy(float)
    d0 = _cv_r2(full, y) - _cv_r2(base, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(i)) > 4:                                  # enough distinct configurations to fit 4 terms
            bs.append(_cv_r2(full[i], y[i], i) - _cv_r2(base[i], y[i], i))
    if len(bs) < 20:
        return float(d0), np.nan, np.nan
    return float(d0), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def config_table(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        means = r["test"].groupby("policy").mean(numeric_only=True)
        row = {**r["config"], **{k: r["ratio"][k] for k in ("n_needed", "n_avail", "ratio", "log10_ratio")},
               "best_fixed": r["best_fixed"]}
        for k, (m, lo, hi) in r["effects"].items():
            row[k], row[k + "_lo"], row[k + "_hi"] = m, lo, hi
        for pol in means.index:
            row[f"np_{pol}"] = means.loc[pol, "net_payoff"]
        rows.append(row)
    return pd.DataFrame(rows)


def negative_controls(plan: LearnPlan, hp) -> pd.DataFrame:
    base = Config(**dict(plan.baseline))
    T, burn = plan.periods, plan.burn_in
    out = []
    # NC1: perfect information, costless adaptation
    c1 = replace(base, perfect=True, cost=0.0, switch_cost=0.0, magnitude_cost=0.0)
    assert c1.zero_cost, "NC1 must charge nothing at all"
    t1 = run_config(c1, hp, _seeds(plan, 4, 1, plan.paths), T, burn)
    diffs = {pol: paired(t1, pol, "always", n_boot=plan.n_boot) for pol in POLICIES if pol not in ("always", "oracle")}
    ok1 = all(not (lo > 0.25) for _, lo, _ in diffs.values())
    out.append(("NC1", ok1, "; ".join(f"{POLICY_LABELS[k]} − always {m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                                       for k, (m, lo, hi) in diffs.items())))
    # NC2: dominated default
    c2 = replace(base, default="dominated")
    t2 = run_config(c2, hp, _seeds(plan, 4, 2, plan.paths), T, burn, ("always", "gate_gain", "gate_lcb"))
    m2 = t2.groupby("policy").mean(numeric_only=True)
    ok2 = all(m2.loc[g, "departure_rate"] >= 0.9 and m2.loc[g, "net_payoff"] >= m2.loc["always", "net_payoff"]
              - 0.05 * abs(m2.loc["always", "net_payoff"]) for g in ("gate_gain", "gate_lcb"))
    out.append(("NC2", ok2, "; ".join(f"{POLICY_LABELS[g]}: departure {m2.loc[g, 'departure_rate']:.3f}, net payoff "
                                      f"{m2.loc[g, 'net_payoff']:.2f} (always {m2.loc['always', 'net_payoff']:.2f})"
                                      for g in ("gate_gain", "gate_lcb"))))
    # NC3: stable and abundant
    c3 = replace(base, hazard=0.0, avail=1.0, tau=0.0, sigma=5.0, default="slow")
    T3 = max(T, 3000)
    r3 = learnability_ratio(c3, hp["gain"], T3, _seeds(plan, 4, 30, plan.pilot_paths), burn)
    pil = run_config(c3, hp, _seeds(plan, 4, 31, plan.pilot_paths), T3, burn, ("always", "default"))
    pm = pil.groupby("policy")["net_payoff"].mean()
    bf = "always" if pm["always"] >= pm["default"] else "default"
    t3 = run_config(c3, hp, _seeds(plan, 4, 3, plan.paths), T3, burn, ("always", "default", "gate_lcb"))
    m, lo, hi = paired(t3, "gate_lcb", bf, n_boot=plan.n_boot)
    # Noninferiority: reject "the gate is worse by more than the margin" when the lower confidence limit clears it.
    ok3 = bool(r3["ratio"] < 1 and lo > -NC3_MARGIN)
    out.append(("NC3", ok3, f"R = {r3['ratio']:.3g}; confidence gate − {POLICY_LABELS[bf].lower()} {m:+.2f} "
                            f"[{lo:+.2f}, {hi:+.2f}]; noninferiority margin −{NC3_MARGIN:g} "
                            f"({'lower limit clears it' if lo > -NC3_MARGIN else 'lower limit does not clear it'})"))
    return pd.DataFrame(out, columns=["id", "passed", "result"])


@dataclass
class LearnResult:
    plan_hash: str
    hyperparameters: Dict
    sweeps: pd.DataFrame          # one row per (factor, level) with paired effects
    tests: pd.DataFrame           # one row per test configuration (jump family)
    new_families: pd.DataFrame    # one row per configuration of the new families
    controls: pd.DataFrame
    verdicts: pd.DataFrame
    market: Optional[Dict] = None
    paths: pd.DataFrame = field(default_factory=pd.DataFrame)    # per-path outcomes of every evaluated configuration
    tuning: pd.DataFrame = field(default_factory=pd.DataFrame)   # one row per policy: candidates evaluated and choice


def run_study(plan: LearnPlan, progress: Optional[Callable[[float, str], None]] = None, market: bool = True,
              market_plan=None) -> LearnResult:
    note = progress or (lambda f, m: None)
    note(0.0, "Tuning on training configurations")
    hp, tuning_log = tune(plan, lambda f, m: note(0.15 * f, m))
    base = Config(**dict(plan.baseline))
    all_paths = []
    # one-at-a-time sweeps
    sweep_rows, j = [], 0
    n_sweep = sum(len(v) for _, v in plan.sweeps)
    for factor, levels in plan.sweeps:
        for lv in levels:
            j += 1
            note(0.15 + 0.3 * j / n_sweep, f"Sweep: {factor} = {lv}")
            r = evaluate_config(replace(base, **{factor: lv}), hp, plan, 1000 + j)
            row = config_table([r]).iloc[0].to_dict()
            row.update(factor=factor, level=str(lv))
            for pol in POLICIES:
                if pol != "always":
                    m, lo, hi = paired(r["test"], pol, "always", n_boot=plan.n_boot)
                    row[f"vs_always_{pol}"], row[f"vs_always_{pol}_lo"], row[f"vs_always_{pol}_hi"] = m, lo, hi
            sweep_rows.append(row)
            all_paths.append(r["test"].assign(part="sweep", factor=factor, level=str(lv)))
    # Latin-hypercube test configurations (jump) and new families
    tests = []
    for i, cfg in enumerate(_lhs_configs(plan, plan.n_test_configs, plan.seed + 3)):
        note(0.45 + 0.3 * i / plan.n_test_configs, f"Test configuration {i + 1}")
        r = evaluate_config(cfg, hp, plan, 2000 + i)
        tests.append(r)
        all_paths.append(r["test"].assign(part="test", config=i))
    newf = []
    for fi, fam in enumerate(plan.new_families):
        for i, cfg in enumerate(_lhs_configs(plan, plan.n_new_family_configs, plan.seed + 4 + fi, fam)):
            note(0.75 + 0.15 * (fi + i / plan.n_new_family_configs) / len(plan.new_families), f"{fam}: {i + 1}")
            r = evaluate_config(cfg, hp, plan, 3000 + 100 * fi + i)
            newf.append(r)
            all_paths.append(r["test"].assign(part=f"family_{fam}", config=i))
    note(0.9, "Negative controls")
    controls = negative_controls(plan, hp)
    T1, NF = config_table(tests), config_table(newf)
    mres = None
    if market:
        from . import learnability_market as LM
        note(0.92, "Market replication")
        mres = LM.run_market(market_plan or LM.MarketPlan())
    verdicts = evaluate(T1, NF, mres, plan)
    note(1.0, "Done")
    return LearnResult(plan.digest, hp, pd.DataFrame(sweep_rows), T1, NF, controls, verdicts, mres,
                       pd.concat(all_paths, ignore_index=True), tuning_log)


def evaluate(tests: pd.DataFrame, newf: pd.DataFrame, market: Optional[Dict], plan: LearnPlan) -> pd.DataFrame:
    rows = []
    b, lo, hi = _slope_ci(tests["log10_ratio"], tests["adv_gate_lcb"], plan.n_boot)
    rows.append(("H1", "supported" if hi < 0 else "not supported",
                 f"slope {b:+.2f} per unit of log10 R [{lo:+.2f}, {hi:+.2f}] ({len(tests)} configurations)"))
    d, dlo, dhi = incremental_validity(tests, "adv_gate_lcb", plan.n_boot)
    rows.append(("H2", "supported" if np.isfinite(dlo) and dlo > 0 else "not supported",
                 f"cross-validated R² gain {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"))
    if len(newf):
        b3, lo3, hi3 = _slope_ci(newf["log10_ratio"], newf["adv_gate_lcb"], plan.n_boot)
        rows.append(("H3", "supported" if hi3 < 0 else "not supported",
                     f"slope {b3:+.2f} [{lo3:+.2f}, {hi3:+.2f}] ({len(newf)} configurations, "
                     f"{', '.join(plan.new_families)})"))
    if market is not None:
        mt = market["tests"]
        b4, lo4, hi4 = _slope_ci(mt["log10_ratio"], mt["adv_gate_lcb"], plan.n_boot)
        rows.append(("H4", "supported" if hi4 < 0 else "not supported",
                     f"slope {b4:+.1f} [{lo4:+.1f}, {hi4:+.1f}] ({len(mt)} market configurations)"))
    return pd.DataFrame([dict(id=h[0], hypothesis=h[1], decision_rule=h[2], verdict=v[1], result=v[2])
                         for h in plan.hypotheses for v in rows if v[0] == h[0]])


SECONDARY = ("gross_payoff", "cost_departure", "cost_switching", "cost_magnitude", "cost_total",
             "regret", "cvar5", "departure_rate", "adjustment_rate", "adjustment_magnitude", "missed",
             "calib_bias", "calib_slope", "recovery_delay")


def secondary_table(paths_df: pd.DataFrame, part: str = "test", n_boot: int = 2000) -> pd.DataFrame:
    """Secondary outcomes per policy on the untouched test paths of one part of the study: mean and 95% bootstrap CI
    over paths (descriptive; no decision rule)."""
    d = paths_df[paths_df["part"] == part]
    rows = []
    for pol, g in d.groupby("policy", sort=False):
        r = {"policy": pol, "label": POLICY_LABELS[pol]}
        for k in SECONDARY:
            m, lo, hi = _boot_ci(g[k].to_numpy(float), n_boot)
            r[k], r[k + "_lo"], r[k + "_hi"] = m, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


# ================================================================================================ registration
def registration_document(plan: LearnPlan, registered_hash: Optional[str] = None) -> str:
    """Markdown registration document of the frozen specification (exportable)."""
    status = ("This specification is frozen in the repository: its hash covers the plan and the study code, and a test "
              "fails if either changes without a new registration entry. It has **not** been preregistered with an "
              "external registry (for example OSF or AsPredicted). Any external registration must be done separately; "
              "until then, do not describe the study as externally preregistered.")
    cur = plan.digest
    lines = [f"# Registration: when can reliability be learned before the environment changes?", "",
             f"* Plan hash: `{cur}`" + (f" (registered in the repository: `{registered_hash}`"
                                        f"{'' if registered_hash == cur else ' — DIFFERENT: this plan is exploratory'})"
                                        if registered_hash else ""),
             f"* Status: {status}", "",
             "## Question", "When does an agent have enough informative feedback within a regime to learn whether "
             "adapting is reliable, before the environment changes? A proposed construct, the learnability ratio R "
             "(observations needed to determine the sign of the advantage of adapting with 90% one-sided confidence, "
             "divided by the informative observations available within a regime), is tested for whether it explains "
             "policy performance beyond volatility and observation noise.", "",
             "## Design", __doc__.split("Task\n", 1)[1].split("Proposed construct")[0].strip(), "",
             "## Proposed construct", __doc__.split("Proposed construct:", 1)[1].strip(), "",
             "## Primary outcome", "Net payoff per period (newsvendor payoff minus the adaptation cost c whenever the "
             "order differs from the default's). Effects are paired differences over shared exogenous paths with 95% "
             "bootstrap intervals; rankings are not used to declare any theory superior.", "",
             "## Hypotheses and decision rules"]
    lines += [f"* **{h[0]}.** {h[1]} *Decision rule:* {h[2]}" for h in plan.hypotheses]
    lines += ["", "## Negative controls"] + [f"* **{c[0]} · {c[1]}.** {c[2]}" for c in plan.negative_controls]
    lines += ["", "## Sets", "Training configurations (jump family) for tuning; independent pilot paths for R and for "
              "choosing the better fixed rule; untouched test paths (new seeds) for every reported effect; new process "
              "families (switching, drifting) for transfer. Seed blocks are disjoint by construction.", "",
              "## Plan", "```json", plan.to_json(), "```"]
    return "\n".join(lines)


LEARN_PREREG = LearnPlan()
