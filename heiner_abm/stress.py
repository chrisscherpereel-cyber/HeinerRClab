"""A registered stress-test suite: five process families that break the assumptions the benchmarks rest on.

PROPOSED EXTENSION. The decision benchmarks in `bench_inventory` are compared on one process: a
piecewise-constant mean with Gaussian demand around it. On that process one model is *correctly specified* and the
rest are not, and the ranking largely reproduces that fact. This suite asks the more useful question - what
happens when the process is not the one anybody's model assumes - by running the same agents, under the same
information restrictions and the same cost definitions, on five deliberately hostile processes.

Nothing here is a registered finding of the laboratory. It is its own frozen plan.

================================================================================ the processes

Throughout, `mu_t` is the latent mean demand, `d_t >= 0` the demand that occurs, `y_t` what the agent observes,
`S_t` the order it placed before seeing `d_t`, and

    payoff_t = -( H_OVER * max(S_t - d_t, 0) + P_SHORT * max(d_t - S_t, 0) ) - adjustment_t
    adjustment_t = switch_cost * 1{ |S_t - S_{t-1}| > ACTION_TOL }

with H_OVER = 1, P_SHORT = 4, so the newsvendor fractile is 0.8. Every agent chooses S_t from y_0..y_{t-1} only.
`sigma` is outcome noise, `tau` observation noise, `hazard` the per-period change probability, `magnitude` the
shock size, and `switch_cost` the adjustment cost - the five factors of the design.

CONTROL  `base` - the registered process, included so the stressed results have a reference point.
    jump_t ~ Bernoulli(hazard);  mu_t = clip(mu_{t-1} + jump_t * N(0, magnitude), 30, 170)
    d_t = max(0, mu_t + sigma * N(0,1));   y_t = d_t + tau * N(0,1)

A  `reversal` - long stability, then an unannounced reversal of the preferred action.
    mu_t = mu_0                      for t < c
    mu_t = mu_0 - direction * magnitude   for t >= c,   c ~ Uniform{stable_lo, stable_hi}
    Further reversals occur at rate `hazard` afterwards, so later changes can interrupt a recovery that is still
    in progress - which is why recovery below is right-censored rather than averaged over the episodes that
    happened to finish.
    The point is that nothing announces c, and the preferred order before and after are on opposite sides of
    everything the agent has ever seen.

B  `heavy` - heavy-tailed outcomes and rare contamination shocks.
    mu_t as in `base`;   d_t = max(0, mu_t + sigma * e_t)
    e_t = N(0,1)                       with probability 1 - HEAVY_RATE  (= 0.95)
          c * T_3                      with probability HEAVY_RATE,  c = HEAVY_SCALE * magnitude / 30
    so the contaminating draw has a standard deviation of between 5.3 and 14.7 times sigma across the design.
    The contamination is applied at the scale of the outcome noise, not of the observation noise: a Student-t
    *observation* error is absorbed by the payoff noise and leaves the realized kurtosis near Gaussian, which is
    how a heavy-tailed condition ends up not being heavy. Variant `student_t` instead draws every innovation from
    a standardized T_3, which has no contamination component at all.

C  `outside_support` - outcomes outside the historical observed support.
    For t < history_len (45% of the horizon):  d_t = mu_0 + clip(sigma * N(0,1), -w, +w),  w = 0.5 sigma
    For t >= history_len:                      d_t = mu_0 + sigma * N(0,1), plus an excursion of size
                                               `magnitude` with probability `hazard`, above the historical
                                               maximum (variant `upper`) or beyond either extreme (`both_tails`)
    The narrow history is the point. The newsvendor fractile is 0.8, so the optimal order is about mu + 0.84
    sigma - which is OUTSIDE the band every observation so far has fallen in. A model whose support is the
    observed data cannot order it at any confidence level, however much data it has and however certain it is;
    the moment model's worst case is a two-point distribution that need not sit on observed values, so it can.
    The excursions then test the same limitation further out in the tail.

D  `multimodal` - a mixture whose mode probabilities move.
    modes m_1 = mu_0 - magnitude/2, m_2 = mu_0 + magnitude/2   (variant `three_mode` adds m_3 = mu_0)
    logit(w_t) = logit(w_{t-1}) + N(0, MODE_WALK) * 1{Bernoulli(hazard)}
    k_t ~ Categorical(w_t);   d_t = max(0, m_{k_t} + sigma * N(0,1));   y_t = d_t + tau * N(0,1)
    Every unimodal model tracks the mixture *mean*, which for well-separated modes is a demand level that almost
    never occurs.

E  `decision_dependent` - the agent's own action moves the process.
    variant `level`  (goodwill): a goodwill STOCK that decays back, so the effect is persistent but not absorbing
        g_t = (1 - RECOVER) * g_{t-1} + KAPPA * lost_{t-1},   lost_{t-1} = max(0, d_{t-1} - S_{t-1})
        mu_t = clip(b_t - g_t, 30, 170),   b_t the exogenous level, jumping as in `base`
        Customers turned away stay away for a while. Without the recovery term every agent, the oracle
        included, drives the mean to the lower clip and the family measures the clip instead of the mechanism.
    variant `regime` (instability): hazard_t = hazard * (1 + LAMBDA * 1{stocked out at t-1})
        the agent's own stockouts make the environment change more often.
    Every model in the suite assumes the process is exogenous, so every model is misspecified here, including the
    one that is correct everywhere else. That is the point of including it.

================================================================================ regret, and what it means here

Regret needs a benchmark that is stated rather than implied. Two are defined:

    oracle_path   knows mu_t ON THE PATH THE AGENT ACTUALLY GENERATED, and orders mu_t + Z_FRAC * sigma. It does
                  not know the noise draw, so it is the best *feasible* order given the state, not hindsight.
                  Pathwise, so per-period regret is well defined for every family INCLUDING E.
    oracle_free   plays its own trajectory from the start. Under A-D the path is exogenous and this coincides
                  with oracle_path. Under E it does not: by never stocking out it faces a different, easier
                  process.

`regret` below is always against oracle_path. For E the gap between the two oracles is reported separately as
`endogenous_gap`, because it measures something the agent could have avoided rather than something it got wrong,
and adding the two together would double-count.

================================================================================ academic connections

Checked against the primary publications, not against secondary descriptions. Where this suite departs from a
cited method, the departure is stated.

    Page (1954)              CUSUM, the original sequential change detector. Used here only as the reference for
                             what "detecting a change quickly" means; no CUSUM agent is run in this suite.
    Genalti et al. (2025)    change-point detection for heavy-tailed, piecewise-stationary bandits, assuming only
                             a bounded moment of order 1+eps. THIS SUITE DOES NOT IMPLEMENT THEIR METHOD: their
                             detector is Catoni-style and the robust estimator already in this repository
                             (`bench_bandit.median_of_means`) is not. It is cited as the setting family B belongs
                             to, and as the reason a finite-variance contamination model is the weaker of the two
                             heavy-tailed variants here - their assumption admits infinite variance, ours does not.
    Scarf (1958)             the distribution-free newsvendor: the moment model used against family C.
    Gallego & Moon (1993)    its review and extensions.
    Luo & Mehrotra (2020)    distributionally robust optimization with decision-dependent ambiguity sets, with a
                             newsvendor illustration. Family E is the simulation analogue of their setting; no
                             decision-dependent ambiguity set is solved here, so this is a connection, not an
                             implementation.
    Yu & Basciftci (2026)    multimodal decision-dependent ambiguity sets, where the decision shifts the
                             likelihood of each mode. Families D and E separate the two halves of that setting -
                             moving mode probabilities, and decision-dependence - which this suite deliberately
                             does NOT combine, since no agent here could represent the combination.

The suite measures how existing agents fail on these processes. It does not claim to implement, reproduce or test
any of the methods above.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from heiner_abm import bench_inventory as B

# ================================================================================================ constants
H_OVER, P_SHORT = B.H_OVER, B.P_SHORT
FRACTILE = B.FRACTILE                       # 0.8
Z_FRAC = B.Z_FRAC
MU0, MU_LO, MU_HI = 100.0, 30.0, 170.0
ACTION_TOL = 1e-9

HEAVY_RATE = 0.05                           # share of contaminated outcomes in family B
HEAVY_DF = 3.0                              # Student-t degrees of freedom (finite variance, infinite kurtosis)
HEAVY_SCALE = 8.0                           # contamination s.d. IN UNITS OF SIGMA at the reference magnitude.
MAGNITUDE_REF = 30.0                        # Scaling the contamination by magnitude/sigma instead would make
                                            # the family inert wherever the shock is no larger than the noise -
                                            # which is half of its own factorial - so the multiplier is in units
                                            # of sigma and `magnitude` modulates it around this reference.
MODE_WALK = 1.2                             # s.d. of the logit step of the mode probability in family D
KAPPA = 0.35                                # goodwill lost per unit of unmet demand, family E variant `level`
RECOVER = 0.10                              # per-period share of lost goodwill that returns
LAMBDA = 6.0                                # multiplier on the hazard after a stockout, family E variant `regime`
HISTORY_FRACTION = 0.45                     # share of the horizon that is "history" in family C
HISTORY_BAND = 0.5                          # family C: the history is this many sigma wide, so that the
                                            # optimal order itself lies OUTSIDE everything ever observed
STABLE_LO, STABLE_HI = 0.35, 0.60           # the reversal in family A lands in this fraction of the horizon

# Seed streams, disjoint from each other and from the registered studies.
STREAM = {"process": 818_181, "outcome": 828_282, "obs": 838_383, "mode": 848_484, "contam": 858_585}


# ================================================================================================ the environment
@dataclass(frozen=True)
class StressEnv:
    """One cell of the design: a process family, a variant of it, and the five factors."""
    family: str
    variant: str
    sigma: float = 18.0                     # outcome noise
    hazard: float = 0.01                    # change frequency
    magnitude: float = 30.0                 # shock magnitude
    tau: float = 8.0                        # observation noise (lower = more precise)
    switch_cost: float = 0.0                # adjustment cost
    periods: int = 600
    burn_in: int = 60

    def __post_init__(self):
        if self.family not in PROCESSES:
            raise ValueError(f"unknown process family {self.family!r}")
        if self.variant not in PROCESSES[self.family].variants:
            raise ValueError(f"{self.family!r} has no variant {self.variant!r}")
        for name in ("sigma", "magnitude", "tau", "switch_cost"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.hazard <= 1.0:
            raise ValueError("hazard must be a probability")
        if self.burn_in >= self.periods:
            raise ValueError("burn_in must leave periods to evaluate")


def _tag(text: str) -> int:
    """A STABLE integer for a name.

    Python randomizes the hash of a string per process, so `hash(family)` in a seed would give a different path
    on every run and the frozen plan would not be reproducible at all - the kind of defect that is invisible
    until someone tries to reproduce a number months later. A digest is deterministic across processes and
    machines, and a test runs a subprocess to confirm it.
    """
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _rng(env: StressEnv, seed: int, stream: str) -> np.random.Generator:
    return np.random.default_rng([seed, STREAM[stream], _tag(env.family), _tag(env.variant)])


# ================================================================================================ the processes
@dataclass(frozen=True)
class Process:
    """A process family: its equations, what the agent may see, and which models it breaks."""
    key: str
    name: str
    kind: str                               # "control" or "stress"
    variants: Tuple[str, ...]
    equations: str
    information: str
    breaks: str                             # which assumption this family violates, in one line
    sources: Tuple[str, ...] = ()
    decision_dependent: bool = False


PROCESSES: Dict[str, Process] = {p.key: p for p in [
    Process("base", "Control: piecewise-constant mean", "control", ("standard",),
            "jump_t ~ Bern(hazard); mu_t = clip(mu_{t-1} + jump_t N(0, magnitude), 30, 170); "
            "d_t = max(0, mu_t + sigma N(0,1)); y_t = d_t + tau N(0,1)",
            "y_0..y_{t-1}", "Nothing: this is the process the correctly specified model assumes.",
            ("arrow1951b", "fearnhead2007")),
    Process("reversal", "A. Long stability, unannounced reversal", "stress", ("single", "double"),
            "mu_t = mu_0 for t < c; mu_0 - direction * magnitude for t >= c; c ~ U{0.35T, 0.60T}; "
            "further reversals at rate hazard",
            "y_0..y_{t-1}; the change time c is researcher-only",
            "Constant hazard: the change time is not geometric, and a long quiet spell is not evidence of "
            "stability.", ("page1954",)),
    Process("heavy", "B. Heavy tails and rare contamination", "stress", ("contaminated", "student_t"),
            "d_t = max(0, mu_t + sigma e_t); e_t ~ N(0,1) w.p. 1-0.05, (magnitude/sigma) T_3 w.p. 0.05 "
            "[variant student_t: e_t ~ standardized T_3 every period]",
            "y_0..y_{t-1}", "Gaussian outcomes: the likelihood every filter uses, and the sample mean the "
            "window models use, are both non-robust.", ("genalti2025",)),
    Process("outside_support", "C. Outcomes outside the observed support", "stress", ("upper", "both_tails"),
            "d_t truncated to mu_0 +- 2 sigma for t < 0.45T; thereafter an excursion of size magnitude beyond "
            "the historical extreme with probability hazard",
            "y_0..y_{t-1}", "Empirical support: a model that can only reweight observed values cannot price an "
            "outcome it has never seen.", ("scarf1958", "gallego1993")),
    Process("multimodal", "D. Multimodal demand, moving mode probabilities", "stress",
            ("two_mode", "three_mode"),
            "modes at mu_0 -+ magnitude/2 (three_mode adds mu_0); logit(w_t) walks by N(0, 1.2) when "
            "Bern(hazard); k_t ~ Cat(w_t); d_t = max(0, m_{k_t} + sigma N(0,1))",
            "y_0..y_{t-1}; the active mode k_t is researcher-only",
            "Unimodality: the predictive mean sits between the modes, at a demand that almost never occurs.",
            ("yu2026",)),
    Process("decision_dependent", "E. Decision-dependent uncertainty", "stress", ("level", "regime"),
            "level: g_t = 0.9 g_{t-1} + 0.35 lost_{t-1}; mu_t = clip(b_t - g_t, 30, 170); "
            "regime: hazard_t = hazard (1 + 6 * 1{stocked out at t-1})",
            "y_0..y_{t-1}; mu_t and the realized hazard are researcher-only",
            "Exogeneity: every model here treats the demand process as unaffected by its own orders.",
            ("luo2020", "yu2026"), decision_dependent=True),
]}

TUNING_VARIANTS = {"base": ("standard",), "reversal": ("single",), "heavy": ("contaminated",),
                   "outside_support": ("upper",), "multimodal": ("two_mode",),
                   "decision_dependent": ("level",)}
EVALUATION_VARIANTS = {"base": ("standard",), "reversal": ("double",), "heavy": ("student_t",),
                       "outside_support": ("both_tails",), "multimodal": ("three_mode",),
                       "decision_dependent": ("regime",)}
# `base` is the control and its single variant is used in both roles deliberately: it is the reference point, not
# a test. Every STRESS family reserves a variant that tuning never touches, and a test asserts that.


# ================================================================================================ generating a path
def _standardized_t(rng: np.random.Generator, n: int, df: float = HEAVY_DF) -> np.ndarray:
    """Student-t scaled to unit variance, so that `sigma` keeps its meaning across variants."""
    return rng.standard_t(df, n) / np.sqrt(df / (df - 2.0))


def _heavy_scale(env: StressEnv) -> float:
    """Contamination s.d. in units of sigma."""
    return HEAVY_SCALE * env.magnitude / MAGNITUDE_REF


def _innovations(env: StressEnv, seed: int) -> np.ndarray:
    """The outcome innovation e_t, in units of sigma."""
    T = env.periods
    rng = _rng(env, seed, "outcome")
    e = rng.standard_normal(T)
    if env.family != "heavy":
        return e
    if env.variant == "student_t":
        return _standardized_t(rng, T)
    crng = _rng(env, seed, "contam")
    hit = crng.random(T) < HEAVY_RATE
    return np.where(hit, _heavy_scale(env) * _standardized_t(crng, T), e)


def _mean_path(env: StressEnv, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """The latent mean and the change indicator, for the EXOGENOUS families (everything but E)."""
    T = env.periods
    rng = _rng(env, seed, "process")
    mu = np.full(T, MU0)
    change = np.zeros(T, bool)

    if env.family in ("base", "heavy"):
        jumps = rng.random(T) < env.hazard
        m = MU0
        for t in range(T):
            if jumps[t]:
                m = min(MU_HI, max(MU_LO, m + rng.normal(0.0, env.magnitude)))
                change[t] = True
            mu[t] = m

    elif env.family == "reversal":
        c = int(rng.integers(int(STABLE_LO * T), int(STABLE_HI * T)))
        direction = 1.0 if rng.random() < 0.5 else -1.0
        level = MU0
        after = min(MU_HI, max(MU_LO, MU0 - direction * env.magnitude))
        extra = rng.random(T) < env.hazard
        for t in range(T):
            if t == c:
                level, change[t] = after, True
            elif t > c and extra[t] and env.variant == "double":
                level = min(MU_HI, max(MU_LO, MU0 if abs(level - after) < 1e-9 else after))
                change[t] = True
            mu[t] = level

    elif env.family == "outside_support":
        mu[:] = MU0                                    # the level is quiet; the SUPPORT is the stress
        # The one event in this family is the moment the support widens. Marking it as a change point is what
        # lets the recovery estimator answer the question the family is about: how long does an agent go on
        # ordering inside a band that is no longer where demand lives?
        change[int(HISTORY_FRACTION * T)] = True

    elif env.family == "multimodal":
        mu[:] = MU0                                    # mu is the mixture mean; modes are applied in _outcomes
        walk = rng.random(T) < env.hazard
        change[:] = walk

    return mu, change


def _mode_draws(env: StressEnv, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Family D: the active mode each period and the mode probabilities that generated it."""
    T = env.periods
    rng = _rng(env, seed, "mode")
    modes = (np.array([-0.5, 0.5]) if env.variant == "two_mode" else np.array([-0.5, 0.0, 0.5])) * env.magnitude
    k = len(modes)
    logits = np.zeros((T, k))
    cur = np.zeros(k)
    step = rng.normal(0.0, MODE_WALK, (T, k))
    walk = rng.random(T) < env.hazard
    for t in range(T):
        if walk[t]:
            cur = cur + step[t]
        logits[t] = cur
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    draw = rng.random(T)
    active = (w.cumsum(axis=1) < draw[:, None]).sum(axis=1).clip(0, k - 1)
    return MU0 + modes[active], w


def _truncated_history(env: StressEnv, seed: int, mu: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Family C: a narrow bounded history, then outcomes - and an optimal order - outside it."""
    T = env.periods
    h = int(HISTORY_FRACTION * T)
    w = HISTORY_BAND * env.sigma
    d = mu + env.sigma * e
    d[:h] = mu[:h] + np.clip(env.sigma * e[:h], -w, w)
    rng = _rng(env, seed, "contam")
    hit = rng.random(T) < env.hazard
    hit[:h] = False
    sign = np.where(rng.random(T) < 0.5, -1.0, 1.0) if env.variant == "both_tails" else np.ones(T)
    return np.where(hit, mu + sign * (w + env.magnitude), d)


def exogenous_path(env: StressEnv, seed: int) -> Dict[str, np.ndarray]:
    """Latent mean, demand, observation and the researcher-only change indicator, for families base/A/B/C/D."""
    if PROCESSES[env.family].decision_dependent:
        raise ValueError(f"{env.family!r} is decision-dependent and has no path independent of the agent")
    mu, change = _mean_path(env, seed)
    e = _innovations(env, seed)
    if env.family == "multimodal":
        level, _ = _mode_draws(env, seed)
        d = np.maximum(0.0, level + env.sigma * e)
        mu = level                                      # the mean of the ACTIVE mode: what an oracle would track
    elif env.family == "outside_support":
        d = np.maximum(0.0, _truncated_history(env, seed, mu, e))
    else:
        d = np.maximum(0.0, mu + env.sigma * e)
    y = d + env.tau * _rng(env, seed, "obs").standard_normal(env.periods)
    return {"mu": mu, "demand": d, "obs": y, "change": change}


# ================================================================================================ the agents
# Every agent is ONLINE: it chooses S_t from its state, then is told y_t. The vectorized forms in bench_inventory
# are the same computations arranged for speed on an exogenous path; they cannot be used for family E, where the
# path does not exist until the agent has acted, so the sequential form is the one that keeps all six families
# under identical information restrictions.
SLOW_GAIN = 0.01


class Policy:
    """Base class. `order()` must not consult anything the agent has not been told through `update()`."""
    key = ""

    def __init__(self, env: StressEnv, cfg: Dict[str, float]):
        self.env, self.cfg = env, dict(cfg)

    def order(self) -> float:
        raise NotImplementedError

    def update(self, y: float) -> None:
        raise NotImplementedError


class _Ewma(Policy):
    """Exponentially weighted mean and variance, plus a fractile margin from its own spread estimate."""

    def __init__(self, env, cfg, gain=None):
        super().__init__(env, cfg)
        self.gain = float(SLOW_GAIN if gain is None else gain)
        self.m, self.v = MU0, (env.sigma if env.sigma > 0 else 1.0) ** 2

    def order(self) -> float:
        return float(max(0.0, self.m + Z_FRAC * np.sqrt(max(self.v, 1.0))))

    def update(self, y: float) -> None:
        self.v += self.gain * ((y - self.m) ** 2 - self.v)
        self.m += self.gain * (y - self.m)


class RuleB(_Ewma):
    key = "rule_b"


class Ewma(_Ewma):
    key = "ewma"

    def __init__(self, env, cfg):
        super().__init__(env, cfg, gain=cfg["gain"])


class _Window(Policy):
    """Keeps the last N observations and re-solves each period. The fallback before N observations exist is the
    default rule, so that no window model gets a head start the others do not have."""

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.N = int(cfg["window"])
        self.hist: List[float] = []
        self.fallback = RuleB(env, {})

    def solve(self, w: np.ndarray) -> float:
        raise NotImplementedError

    def order(self) -> float:
        if len(self.hist) < self.N:
            return self.fallback.order()
        return float(self.solve(np.asarray(self.hist[-self.N:], float)))

    def update(self, y: float) -> None:
        self.hist.append(float(y))
        self.fallback.update(y)


class Saa(_Window):
    key = "saa"

    def solve(self, w):
        return B.saa_order_window(w[None, :])[0]


class DroEmpirical(_Window):
    """Modified chi-squared ball around the EMPIRICAL distribution: it reweights observed values and cannot place
    mass anywhere else. Kept under a precise name because that limitation is the hypothesis family C tests."""
    key = "dro_emp"

    def solve(self, w):
        return B.dro_orders_window(w[None, :], float(self.cfg["rho"]))[0]


class DroMoment(_Window):
    """Moment ambiguity (Scarf): mean and s.d. from the window, shape and SUPPORT otherwise free, so outcomes
    never observed are priced."""
    key = "dro_moment"

    def solve(self, w):
        return B.moment_order_window(w, float(self.cfg["sd_scale"]))


class GridFilter(Policy):
    """The state-space filter of bench_inventory, run one period at a time.

    Latent variation and measurement error stay apart: the likelihood uses both, the order uses sigma alone.
    """
    key = "bayes_est"

    def __init__(self, env, cfg, true_params=False):
        super().__init__(env, cfg)
        self.grid = np.arange(MU_LO, MU_HI + 1e-9, 1.0)
        if true_params:
            self.hazard, self.sigma, self.tau = env.hazard, env.sigma, env.tau
            self.jump_sd = env.magnitude
        else:
            self.hazard, self.sigma, self.tau = float(cfg["hazard"]), float(cfg["sigma"]), float(cfg["tau"])
            self.jump_sd = float(cfg.get("jump_sd", B.JUMP_SD))
        self.P = B.grid_transition(self.grid, max(self.hazard, 1e-9), max(self.jump_sd, 1e-6))
        self.w = np.zeros(len(self.grid))
        self.w[np.argmin(np.abs(self.grid - MU0))] = 1.0
        self.pred = self.w

    def order(self) -> float:
        self.pred = self.w @ self.P
        return float(B.censored_mixture_quantile(self.pred, self.grid, max(self.sigma, 1e-6)))

    def update(self, y: float) -> None:
        if not np.isfinite(y):
            self.w = self.pred
            return
        post = self.pred * B.obs_likelihood(float(y), self.grid, max(self.sigma, 1e-6), max(self.tau, 0.0))
        tot = post.sum()
        self.w = post / tot if tot > 0 else self.pred


class BayesTrue(GridFilter):
    """The same filter, given the generating parameters by the researcher. Correctly specified ONLY on the
    control family; on A-E the parameters are true but the MODEL is not (see SPECIFICATION)."""
    key = "bayes_true"

    def __init__(self, env, cfg):
        super().__init__(env, cfg, true_params=True)


POLICIES: Dict[str, type] = {c.key: c for c in
                             (RuleB, Ewma, Saa, DroEmpirical, DroMoment, GridFilter, BayesTrue)}

SPACES: Dict[str, Dict[str, Tuple[float, float, bool]]] = {
    # name -> (low, high, log). Integer windows are rounded by the policy.
    "rule_b": {},
    "ewma": {"gain": (0.02, 0.8, True)},
    "saa": {"window": (5, 150, True)},
    "dro_emp": {"window": (5, 150, True), "rho": (0.001, 5.0, True)},
    "dro_moment": {"window": (5, 150, True), "sd_scale": (0.2, 2.0, True)},
    "bayes_est": {"hazard": (0.001, 0.2, True), "sigma": (2.0, 60.0, True), "tau": (0.1, 60.0, True)},
    "bayes_true": {},
}
ROLE = {"rule_b": "default", "ewma": "competitor", "saa": "competitor", "dro_emp": "competitor",
        "dro_moment": "competitor", "bayes_est": "competitor", "bayes_true": "reference"}


# ================================================================================================ specification
# Which model is right about which process. "correct" means the model of the demand process IS the generating
# process (parameters aside); "estimated" means the form is right and the parameters are not given; "agnostic"
# means the model makes no distributional assumption that this family violates - which is not the same as being
# right, since every window model still assumes the window is i.i.d.; "misspecified" names the assumption that
# fails.
SPECIFICATION: Dict[str, Dict[str, Tuple[str, str]]] = {
    "base": {
        "bayes_true": ("correct", "the generating model with its generating parameters"),
        "bayes_est": ("estimated", "the generating model; hazard, sigma and tau tuned, never given"),
        "saa": ("misspecified", "treats the window as i.i.d. draws of the coming demand and ignores tau"),
        "dro_emp": ("misspecified", "as SAA, with a ball around the empirical distribution"),
        "dro_moment": ("misspecified", "keeps only the first two moments of the window"),
        "ewma": ("misspecified", "no model of change; one gain for every regime"),
        "rule_b": ("misspecified", "as EWMA with a fixed slow gain"),
    },
    "reversal": {
        "bayes_true": ("misspecified", "assumes geometric change times and Gaussian jump sizes; here the change "
                                       "time is a single draw in a known window and the jump is +-magnitude"),
        "bayes_est": ("misspecified", "the same model form, parameters tuned"),
        "saa": ("misspecified", "i.i.d. window; a long quiet history makes the window confident just before it "
                                "becomes wrong"),
        "dro_emp": ("misspecified", "the ball is around a history containing none of the post-reversal level"),
        "dro_moment": ("misspecified", "moments of a window straddling the reversal describe neither side"),
        "ewma": ("misspecified", "no model of change"),
        "rule_b": ("misspecified", "as EWMA, and slower"),
    },
    "heavy": {
        "bayes_true": ("misspecified", "Gaussian likelihood and Gaussian predictive; the true innovation has "
                                       "infinite kurtosis, and sigma understates the realized spread"),
        "bayes_est": ("misspecified", "as above; tuning sigma upward trades bias for a permanently wide margin"),
        "saa": ("agnostic", "an empirical quantile assumes no shape, so heavy tails cost it sample efficiency "
                            "rather than correctness - but it still assumes the window is i.i.d."),
        "dro_emp": ("agnostic", "shape-free within the observed support"),
        "dro_moment": ("misspecified", "the ambiguity set is shape-free, but the mean and s.d. it is built from "
                                       "are the two least robust statistics available"),
        "ewma": ("misspecified", "a squared-error update is driven by the contaminated draws"),
        "rule_b": ("misspecified", "as EWMA, and slower to forget a contaminated draw"),
    },
    "outside_support": {
        "bayes_true": ("misspecified", "Gaussian demand around the mean; the history is truncated and the "
                                       "excursions are not Gaussian"),
        "bayes_est": ("misspecified", "as above"),
        "saa": ("misspecified", "an empirical quantile cannot exceed the largest value ever observed"),
        "dro_emp": ("misspecified", "BY CONSTRUCTION: the ball reweights observed values and can place no mass "
                                    "on an outcome that has never occurred"),
        "dro_moment": ("agnostic", "the worst case is a two-point distribution that need not sit on observed "
                                   "values, so unseen outcomes are priced"),
        "ewma": ("misspecified", "its margin comes from its own observed variance"),
        "rule_b": ("misspecified", "as EWMA"),
    },
    "multimodal": {
        "bayes_true": ("misspecified", "unimodal predictive; it tracks the mixture mean, a level that almost "
                                       "never occurs"),
        "bayes_est": ("misspecified", "as above"),
        "saa": ("agnostic", "an empirical quantile represents a mixture directly, given a window long enough to "
                            "contain both modes and short enough to be current"),
        "dro_emp": ("agnostic", "as SAA, with a ball around it"),
        "dro_moment": ("misspecified", "a mean and an s.d. cannot tell a wide unimodal demand from a narrow "
                                       "bimodal one, and the two have different optimal orders"),
        "ewma": ("misspecified", "tracks the mixture mean"),
        "rule_b": ("misspecified", "as EWMA"),
    },
    "decision_dependent": {
        k: ("misspecified", "assumes the demand process is unaffected by its own orders; here it is not")
        for k in POLICIES
    },
}


# ================================================================================================ the benchmark
def _t3_quantile(frac: float) -> float:
    from scipy.stats import t
    return float(t.ppf(frac, HEAVY_DF) / np.sqrt(HEAVY_DF / (HEAVY_DF - 2.0)))


def _bisect_quantile(cdf: Callable[[float], float], lo: float, hi: float, frac: float, iters: int = 80) -> float:
    if cdf(lo) >= frac:
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if cdf(mid) < frac else (lo, mid)
    return hi


def oracle_order(env: StressEnv, state: Dict[str, Any]) -> float:
    """The explicitly defined regret benchmark: the 0.8 quantile of the TRUE conditional demand distribution
    given the current state.

    This is the best FEASIBLE order, not hindsight - the oracle knows the state and the law of the noise, never
    the draw. It is defined per family because the conditional law differs; using mu + Z sigma everywhere would
    quietly hand the benchmark a Gaussian assumption that four of these families violate, and would understate
    every agent of ordering too little exactly where the process is hardest.
    """
    from scipy.stats import t as student
    mu = float(state["mu"])
    sd = max(env.sigma, 1e-9)

    if env.family in ("base", "reversal", "decision_dependent"):
        return float(max(0.0, mu + Z_FRAC * sd))

    if env.family == "heavy":
        if env.variant == "student_t":
            return float(max(0.0, mu + sd * _t3_quantile(FRACTILE)))
        scale = _heavy_scale(env)
        nu = np.sqrt(HEAVY_DF / (HEAVY_DF - 2.0))

        def cdf(S):
            z = (S - mu) / sd
            return float((1 - HEAVY_RATE) * ndtr(z) + HEAVY_RATE * student.cdf(z / scale * nu, HEAVY_DF))
        return float(max(0.0, _bisect_quantile(cdf, mu - 40 * sd, mu + 80 * sd, FRACTILE)))

    if env.family == "outside_support":
        w = HISTORY_BAND * sd
        if state.get("phase") == "history":                 # demand is clipped into the band: so is the order
            return float(max(0.0, min(mu + w, mu + Z_FRAC * sd)))
        p = float(state.get("excursion_rate", 0.0))
        up, dn = mu + w + env.magnitude, mu - w - env.magnitude
        p_up = p * (0.5 if env.variant == "both_tails" else 1.0)
        p_dn = p - p_up

        def cdf(S):
            return float((1 - p) * float(ndtr((S - mu) / sd)) + p_up * (S >= up) + p_dn * (S >= dn))
        return float(max(0.0, _bisect_quantile(cdf, dn - 1.0, up + 1.0, FRACTILE)))

    if env.family == "multimodal":
        wts = np.asarray(state["mode_w"], float)
        m = np.asarray(state["mode_levels"], float)

        def cdf(S):
            return float(np.dot(wts, ndtr((S - m) / sd)))
        return float(max(0.0, _bisect_quantile(cdf, 0.0, float(m.max() + 10 * sd), FRACTILE)))

    raise ValueError(env.family)


def payoff(S, d):
    return -(H_OVER * np.maximum(S - d, 0.0) + P_SHORT * np.maximum(d - S, 0.0))


# ================================================================================================ the engine
def _endogenous_run(env: StressEnv, policy: Optional[Policy], seed: int) -> Dict[str, np.ndarray]:
    """Family E: the path does not exist until the agent has acted.

    All shocks are drawn in advance from streams that do not depend on the action - a uniform for the change
    decision, a step size, an outcome innovation, an observation error - so that two policies compared on the
    same seed face the same luck and differ only through what their own orders did to the process. With the
    hazard itself decision-dependent (variant `regime`), common random numbers mean comparing u_t against a
    moving threshold rather than redrawing the change.

    `policy=None` runs the free oracle, which acts on its own trajectory.
    """
    T = env.periods
    prng = _rng(env, seed, "process")
    u = prng.random(T)
    steps = prng.normal(0.0, max(env.magnitude, 1e-9), T)
    e = _innovations(env, seed)
    obs = _rng(env, seed, "obs").standard_normal(T)

    mu = np.empty(T)
    d = np.empty(T)
    y = np.empty(T)
    S = np.empty(T)
    change = np.zeros(T, bool)
    oracle_S = np.empty(T)
    base = MU0
    goodwill = 0.0
    lost_prev, out_prev = 0.0, False

    for t in range(T):
        h = env.hazard if env.variant == "level" else min(1.0, env.hazard * (1.0 + LAMBDA * float(out_prev)))
        if u[t] < h:
            base = min(MU_HI, max(MU_LO, base + steps[t]))
            change[t] = True
        if env.variant == "level":
            goodwill = (1.0 - RECOVER) * goodwill + KAPPA * lost_prev
        m = min(MU_HI, max(MU_LO, base - goodwill))
        mu[t] = m

        state = {"mu": m}
        oracle_S[t] = oracle_order(env, state)
        S[t] = oracle_S[t] if policy is None else max(0.0, float(policy.order()))

        d[t] = max(0.0, m + env.sigma * e[t])
        y[t] = d[t] + env.tau * obs[t]
        if policy is not None:
            policy.update(y[t])
        lost_prev = max(0.0, d[t] - S[t])
        out_prev = bool(S[t] <= d[t] + ACTION_TOL)

    return {"mu": mu, "demand": d, "obs": y, "change": change, "order": S, "oracle_order": oracle_S}


def simulate(env: StressEnv, policy_key: str, cfg: Optional[Dict[str, float]] = None, seed: int = 0,
             free_oracle: bool = False) -> Dict[str, Any]:
    """One agent on one path. Returns per-period arrays plus the summary metrics.

    `free_oracle` is only meaningful for family E: it lets the oracle act, so it faces the easier process its own
    behaviour creates. For A-D the path is exogenous and the two oracles coincide, which a test asserts.
    """
    cfg = dict(cfg or {})
    dd = PROCESSES[env.family].decision_dependent
    policy = None if free_oracle else POLICIES[policy_key](env, cfg)

    if dd:
        run = _endogenous_run(env, policy, seed)
        mu, d, y, change = run["mu"], run["demand"], run["obs"], run["change"]
        S, oracle_S = run["order"], run["oracle_order"]
    else:
        path = exogenous_path(env, seed)
        mu, d, y, change = path["mu"], path["demand"], path["obs"], path["change"]
        T = env.periods
        S = np.empty(T)
        oracle_S = np.empty(T)
        mode_w = _mode_draws(env, seed)[1] if env.family == "multimodal" else None
        levels = None
        if env.family == "multimodal":
            offs = (np.array([-0.5, 0.5]) if env.variant == "two_mode" else np.array([-0.5, 0.0, 0.5]))
            levels = MU0 + offs * env.magnitude
        # The oracle solves a quantile of the true conditional law, which for the heavy and excursion families
        # means a bisection. That law depends only on the state, and the state repeats for long stretches, so it
        # is solved once per distinct state rather than once per period.
        memo: Dict[Any, float] = {}
        for t in range(T):
            state: Dict[str, Any] = {"mu": float(mu[t])}
            key: Any = round(float(mu[t]), 6)
            if env.family == "multimodal":
                state.update(mu=MU0, mode_w=mode_w[t], mode_levels=levels)
                key = tuple(np.round(mode_w[t], 4))
            elif env.family == "outside_support":
                live = t >= int(HISTORY_FRACTION * T)
                state.update(mu=MU0, excursion_rate=(env.hazard if live else 0.0),
                             phase=("live" if live else "history"))
                key = ("c", live)
            if key not in memo:
                memo[key] = oracle_order(env, state)
            oracle_S[t] = memo[key]
            S[t] = oracle_S[t] if free_oracle else max(0.0, float(policy.order()))
            if not free_oracle:
                policy.update(y[t])

    gross = payoff(S, d)
    adj = np.zeros(len(S))
    adj[1:] = env.switch_cost * (np.abs(np.diff(S)) > ACTION_TOL)
    net = gross - adj
    o_gross = payoff(oracle_S, d)
    o_adj = np.zeros(len(S))
    o_adj[1:] = env.switch_cost * (np.abs(np.diff(oracle_S)) > ACTION_TOL)
    o_net = o_gross - o_adj
    regret_t = o_net - net

    rec = slice(env.burn_in, env.periods)
    frame = pd.DataFrame({"period": np.arange(env.periods), "mu": mu, "demand": d, "obs": y, "order": S,
                          "oracle_order": oracle_S, "gross": gross, "adjustment": adj, "net": net,
                          "oracle_net": o_net, "regret": regret_t, "change": change})
    out = {"family": env.family, "variant": env.variant, "policy": policy_key, "seed": seed, "path": frame,
           **metrics(env, frame)}
    return out


# ================================================================================================ metrics
def metrics(env: StressEnv, frame: pd.DataFrame) -> Dict[str, float]:
    """Cumulative net payoff, regret against the defined benchmark, tail loss, calibration."""
    r = frame.iloc[env.burn_in:]
    net = r["net"].to_numpy(float)
    k = max(1, int(np.ceil(0.05 * len(net))))
    worst = np.sort(net)[:k]
    served = (r["demand"].to_numpy(float) <= r["order"].to_numpy(float) + ACTION_TOL)
    return {
        "cum_net_payoff": float(net.sum()),
        "mean_net_payoff": float(net.mean()),
        "regret": float(r["regret"].mean()),
        "cum_regret": float(r["regret"].sum()),
        "tail_loss_cvar5": float(worst.mean()),
        "worst_period": float(net.min()),
        "service_level": float(served.mean()),
        "calibration_gap": float(served.mean() - FRACTILE),
        "stockout_rate": float(1.0 - served.mean()),
        "mean_order": float(r["order"].mean()),
        "adjustment_cost": float(r["adjustment"].mean()),
        "switches": float((r["adjustment"].to_numpy(float) > 0).mean()),
    }


# ================================================================================================ recovery
# A recovery time is the answer to: after a change, how long until the agent is doing as well as it was before
# it? Three things make that harder than it sounds, and each is handled explicitly rather than averaged away.
#
#   ALIGNMENT   the comparison window must lie entirely on one side of the change. A trailing window that
#               straddles the change mixes the two regimes and reports recovery that has not happened. So the
#               baseline is taken over [c - W, c) and the earliest possible recovery is at c + W - 1.
#   SUBSEQUENT  a later change can arrive while the agent is still recovering from this one. That episode has
#   CHANGES     not "failed to recover" and must not be scored as a long recovery either: it is cut short.
#   CENSORING   likewise an episode still running when the horizon ends. Both are RIGHT-CENSORED observations.
#               Averaging only the episodes that finished would report the fast ones and silently drop the slow,
#               which biases every comparison towards whichever agent fails most often. The estimator below is
#               Kaplan-Meier, and the censoring rate is always reported beside it.
def recovery_episodes(regret: np.ndarray, change_idx: Sequence[int], window: int, tol: float,
                      start: int, end: int) -> Tuple[List[Tuple[int, int]], int]:
    """(duration, event) per measurable change point - event = 1 recovered, 0 right-censored - and the number of
    change points that were NOT measurable.

    Both windows have to be aligned, and the baseline has to be a SETTLED level. The window after the change must
    lie entirely after it, or it mixes the two regimes and reports a recovery that has not happened. The baseline
    window [c - W, c) must lie entirely inside a regime the agent had already recovered in - otherwise the level
    it measures is the agent still failing from the previous change, and "returning to it" is satisfied by
    never recovering at all. An agent that is permanently behind would then post the fastest recovery times in
    the suite, which is exactly backwards.

    An episode whose baseline is not settled has no definable target. It is excluded and counted, not censored:
    censoring asserts a lower bound on a duration, and here there is no duration to bound.
    """
    out: List[Tuple[int, int]] = []
    changes = [int(c) for c in change_idx if start <= c < end]
    unmeasurable = 0
    settled_from = start                     # the earliest period at which the current level counts as settled
    for i, c in enumerate(changes):
        nxt = changes[i + 1] if i + 1 < len(changes) else end
        if c - window < settled_from:
            unmeasurable += 1
            settled_from = end               # an unmeasured episode leaves the next baseline unsettled too
            continue
        baseline = float(regret[c - window:c].mean())
        threshold = baseline + tol
        found = None
        for tau in range(c + window - 1, nxt):
            if float(regret[tau - window + 1:tau + 1].mean()) <= threshold:
                found = tau - c + 1
                break
        if found is not None:
            out.append((int(found), 1))
            settled_from = c + found         # settled again once it has recovered
        else:
            out.append((int(max(nxt - c, 1)), 0))
            settled_from = end               # never recovered: nothing after this is a settled baseline
    return out, unmeasurable


def kaplan_meier(times: Sequence[int], events: Sequence[int], unmeasurable: int = 0) -> Dict[str, float]:
    """Kaplan-Meier survival of the recovery time, and its median.

    Median = the first time at which the estimated survival drops to 0.5 or below. If survival never reaches 0.5
    - too many episodes cut short - the median is NOT estimable and is reported as NaN rather than as the largest
    observed duration, which would be a guess dressed as a number.
    """
    t = np.asarray(times, int)
    e = np.asarray(events, int)
    if len(t) == 0:
        return {"median_recovery": float("nan"), "episodes": 0, "censoring_rate": float("nan"),
                "recovered_rate": float("nan"), "unmeasurable": int(unmeasurable)}
    order = np.argsort(t)
    t, e = t[order], e[order]
    surv, at_risk, median = 1.0, len(t), float("nan")
    for time in np.unique(t):
        n_at = int((t >= time).sum())
        n_ev = int(((t == time) & (e == 1)).sum())
        if n_at == 0:
            break
        if n_ev:
            surv *= (1.0 - n_ev / n_at)
        if np.isnan(median) and surv <= 0.5:
            median = float(time)
    return {"median_recovery": median, "episodes": int(len(t)),
            "censoring_rate": float((e == 0).mean()), "recovered_rate": float((e == 1).mean()),
            "unmeasurable": int(unmeasurable)}


def recovery(env: StressEnv, frame: pd.DataFrame, window: int, tol_fraction: float) -> Dict[str, float]:
    """Recovery of one run. The tolerance is a fraction of the run's own pre-change regret scale, so that a noisy
    environment is not required to return to a quiet environment's absolute level."""
    regret = frame["regret"].to_numpy(float)
    changes = np.flatnonzero(frame["change"].to_numpy(bool))
    scale = float(np.abs(regret[env.burn_in:]).mean())
    eps, unmeasurable = recovery_episodes(regret, changes, window, tol_fraction * scale,
                                          env.burn_in, env.periods)
    return kaplan_meier([d for d, _ in eps], [e for _, e in eps], unmeasurable)


# ================================================================================================ the design
# Five factors at two levels each, crossed: a full 2^5 factorial, 32 cells per process variant. Two levels
# rather than three because the question is which factor moves which failure, not the shape of the response, and
# 32 cells per variant is already 192 cells per arm. Every policy sees exactly the same cells and the same seeds.
FACTORS: Dict[str, Tuple[float, float]] = {
    "sigma": (10.0, 28.0),          # outcome noise
    "hazard": (0.004, 0.030),       # change frequency (in family C, the excursion rate)
    "magnitude": (20.0, 55.0),      # shock magnitude
    "tau": (2.0, 20.0),             # observation noise: LOW tau is high precision
    "switch_cost": (0.0, 6.0),      # adjustment cost
}
FACTOR_NAMES = tuple(FACTORS)


def factorial(family: str, variant: str, periods: int, burn_in: int) -> List[Tuple[str, StressEnv]]:
    """The 2^5 cells, each labelled by its corner, e.g. `-+--+`."""
    cells = []
    for i in range(1 << len(FACTOR_NAMES)):
        bits = [(i >> j) & 1 for j in range(len(FACTOR_NAMES))]
        kw = {name: FACTORS[name][b] for name, b in zip(FACTOR_NAMES, bits)}
        label = "".join("+" if b else "-" for b in bits)
        cells.append((label, StressEnv(family=family, variant=variant, periods=periods, burn_in=burn_in, **kw)))
    return cells


def matched_pairs(cells: Sequence[Tuple[str, StressEnv]], factor: str) -> List[Tuple[str, str]]:
    """The 16 pairs of cells identical in every factor but `factor`: the matched contrast that isolates it."""
    j = FACTOR_NAMES.index(factor)
    by_label = {lab: lab for lab, _ in cells}
    out = []
    for lab in by_label:
        if lab[j] == "-":
            partner = lab[:j] + "+" + lab[j + 1:]
            if partner in by_label:
                out.append((lab, partner))
    return sorted(out)


# ================================================================================================ the plan
def stress_code_digest() -> str:
    """A digest of the code that decides what happens, not just of the numbers in the plan.

    A plan hash over parameters alone is not enough. Changing how a seed is derived, or how a path is generated,
    moves every stored result while leaving the parameters identical - so the hash would still match results it
    no longer describes. Including the source of the path-generating and scoring functions means any such change
    invalidates the plan, which is the behaviour this repository requires of a registered run.
    """
    import inspect
    mod = sys.modules[__name__]
    funcs = (mod._tag, mod._rng, mod._heavy_scale, mod._innovations, mod._mean_path, mod._mode_draws,
             mod._truncated_history, mod.exogenous_path, mod._endogenous_run, mod.simulate, mod.oracle_order,
             mod.payoff, mod.metrics, mod.recovery_episodes, mod.kaplan_meier, mod._path_seed,
             mod.sample_configs, mod.factorial)
    src = "".join(inspect.getsource(f) for f in funcs)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class StressPlan:
    """Frozen specification. The hash covers the factor levels, the process equations and the specification map,
    so none of them can be edited without the hash changing - which a test asserts."""
    version: str = "1.0"
    seed: int = 20261013
    families: Tuple[str, ...] = ("base", "reversal", "heavy", "outside_support", "multimodal",
                                 "decision_dependent")
    periods: int = 600
    burn_in: int = 60
    paths: int = 4
    # tuning: a strict subset of cells, a shorter horizon, and only the TUNING variants
    tuning_configs: int = 16
    tuning_cells: int = 8
    tuning_paths: int = 2
    tuning_periods: int = 400
    eval_cells: int = 0                     # 0 = the whole factorial; a positive value subsamples it (tests)
    # recovery
    recovery_window: int = 20
    recovery_tolerance: float = 0.25        # fraction of the run's own regret scale
    seed_block_tuning: int = 1
    seed_block_evaluation: int = 2

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "code": stress_code_digest(),
                           "factors": FACTORS, "specification": SPECIFICATION,
                           "equations": {k: p.equations for k, p in PROCESSES.items()},
                           "tuning_variants": TUNING_VARIANTS, "evaluation_variants": EVALUATION_VARIANTS,
                           "constants": {"HEAVY_RATE": HEAVY_RATE, "HEAVY_DF": HEAVY_DF, "HEAVY_SCALE": HEAVY_SCALE, "KAPPA": KAPPA,
                                         "RECOVER": RECOVER, "LAMBDA": LAMBDA, "MODE_WALK": MODE_WALK,
                                         "HISTORY_FRACTION": HISTORY_FRACTION, "HISTORY_BAND": HISTORY_BAND}},
                          sort_keys=True, indent=1, default=str)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


QUICK = dict(periods=250, burn_in=40, paths=1, tuning_configs=4, tuning_cells=2, tuning_paths=1,
             tuning_periods=200, eval_cells=8)
TINY = dict(periods=120, burn_in=20, paths=1, tuning_configs=2, tuning_cells=1, tuning_paths=1,
            tuning_periods=100, eval_cells=2, recovery_window=8)


def _path_seed(plan: StressPlan, block: int, family: str, cell: str, k: int) -> int:
    """Seeds are namespaced by block (tuning vs evaluation), family, cell and path, so that no evaluation path
    can coincide with a path any tuning run has already been scored on."""
    h = hashlib.sha256(f"{block}|{family}|{cell}|{k}".encode()).hexdigest()[:8]
    return (plan.seed + block * 1_000_000 + int(h, 16)) % (2 ** 31)


# ================================================================================================ tuning
def sample_configs(policy: str, n: int, seed: int) -> List[Dict[str, float]]:
    """Log-uniform draws from the policy's own space. An empty space yields exactly one configuration: a model
    with no parameters cannot spend a tuning budget, and pretending otherwise by giving it duplicate draws would
    make the budget table dishonest rather than equal."""
    space = SPACES[policy]
    if not space:
        return [{}]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        cfg = {}
        for name, (lo, hi, log) in sorted(space.items()):
            u = rng.random()
            v = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))) if log else float(lo + u * (hi - lo))
            cfg[name] = max(1, int(round(v))) if name == "window" else v
        out.append(cfg)
    return out


def tune(plan: StressPlan, progress: Optional[Callable[[float, str], None]] = None
         ) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """One configuration per policy, chosen on the TUNING variants only.

    Equivalent opportunity: every policy with parameters draws the same number of configurations and scores each
    on the same cells, the same paths and the same horizon. The score is mean regret against the defined
    benchmark, pooled over families - a policy is tuned once for the whole suite, not once per family, because
    tuning per family would let it know which stress it was about to face.
    """
    note = progress or (lambda f, m: None)
    rows = []
    chosen: Dict[str, Dict[str, float]] = {}
    pol_list = list(POLICIES)
    for i, policy in enumerate(pol_list):
        configs = sample_configs(policy, plan.tuning_configs, plan.seed + 7 * i)
        best, best_score = configs[0], np.inf
        for c_i, cfg in enumerate(configs):
            scores = []
            for family in plan.families:
                for variant in TUNING_VARIANTS[family]:
                    cells = factorial(family, variant, plan.tuning_periods, plan.burn_in)
                    step = max(1, len(cells) // plan.tuning_cells)
                    for label, env in cells[::step][:plan.tuning_cells]:
                        for k in range(plan.tuning_paths):
                            seed = _path_seed(plan, plan.seed_block_tuning, family, label, k)
                            scores.append(simulate(env, policy, cfg, seed)["regret"])
            score = float(np.mean(scores))
            rows.append({"policy": policy, "config_index": c_i, "regret": score, **cfg})
            if score < best_score:
                best, best_score = cfg, score
            note((i + (c_i + 1) / len(configs)) / len(pol_list), f"tuning {policy} {c_i + 1}/{len(configs)}")
        chosen[policy] = best
        rows.append({"policy": policy, "config_index": -1, "regret": best_score, "selected": True, **best})
    return chosen, pd.DataFrame(rows)


def budget_table(plan: StressPlan) -> pd.DataFrame:
    """The computational budget each policy was given, stated rather than implied."""
    per_config = (sum(len(TUNING_VARIANTS[f]) for f in plan.families) * plan.tuning_cells * plan.tuning_paths)
    rows = []
    for policy in POLICIES:
        n = len(sample_configs(policy, plan.tuning_configs, 0))
        rows.append({"policy": policy, "role": ROLE[policy], "parameters": len(SPACES[policy]),
                     "configurations": n, "runs_per_configuration": per_config,
                     "tuning_runs": n * per_config, "tuning_periods": plan.tuning_periods,
                     "note": "no parameters to tune" if not SPACES[policy] else ""})
    return pd.DataFrame(rows)


def boundary_report(tuned: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Which selected parameters sit at the edge of the grid they were searched in - the usual sign that the
    range, not the data, chose the value."""
    rows = []
    for policy, cfg in tuned.items():
        for name, value in sorted(cfg.items()):
            lo, hi, log = SPACES[policy][name]
            span = (np.log(hi) - np.log(lo)) if log else (hi - lo)
            pos = ((np.log(value) - np.log(lo)) if log else (value - lo)) / span if span > 0 else 0.5
            rows.append({"policy": policy, "parameter": name, "value": float(value), "low": lo, "high": hi,
                         "position": float(pos),
                         "at_boundary": bool(pos <= 0.05 or pos >= 0.95)})
    return pd.DataFrame(rows)


# ================================================================================================ evaluation
def run_evaluation(plan: StressPlan, tuned: Dict[str, Dict[str, float]],
                   progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, pd.DataFrame]:
    """The reserved arm: tuned configurations frozen, EVALUATION variants, a disjoint seed block.

    Recovery episodes are pooled across the paths of a cell before the Kaplan-Meier estimate, because a single
    600-period path of a slow-changing process contributes very few episodes and a median taken from two of them
    is not a median.
    """
    note = progress or (lambda f, m: None)
    rows: List[Dict[str, Any]] = []
    rec_rows: List[Dict[str, Any]] = []
    jobs = [(f, v) for f in plan.families for v in EVALUATION_VARIANTS[f]]

    def design(family, variant):
        cells = factorial(family, variant, plan.periods, plan.burn_in)
        if plan.eval_cells and plan.eval_cells < len(cells):
            step = max(1, len(cells) // plan.eval_cells)
            cells = cells[::step][:plan.eval_cells]
        return cells

    total = max(1, sum(len(design(f, v)) for f, v in jobs))
    done = 0

    for family, variant in jobs:
        dd = PROCESSES[family].decision_dependent
        for label, env in design(family, variant):
            free_net = {}
            if dd:
                for k in range(plan.paths):
                    seed = _path_seed(plan, plan.seed_block_evaluation, family, label, k)
                    free_net[k] = simulate(env, "rule_b", {}, seed, free_oracle=True)["mean_net_payoff"]
            for policy in POLICIES:
                cfg = tuned.get(policy, {})
                episodes: List[Tuple[int, int]] = []
                unmeasurable = 0
                for k in range(plan.paths):
                    seed = _path_seed(plan, plan.seed_block_evaluation, family, label, k)
                    r = simulate(env, policy, cfg, seed)
                    frame = r["path"]
                    regret = frame["regret"].to_numpy(float)
                    scale = float(np.abs(regret[env.burn_in:]).mean())
                    eps, un = recovery_episodes(regret, np.flatnonzero(frame["change"].to_numpy(bool)),
                                                plan.recovery_window, plan.recovery_tolerance * scale,
                                                env.burn_in, env.periods)
                    episodes += eps
                    unmeasurable += un
                    row = {"family": family, "variant": variant, "cell": label, "policy": policy,
                           "role": ROLE[policy], "path": k, "seed": seed,
                           "specification": SPECIFICATION[family][policy][0],
                           **{f: getattr(env, f) for f in FACTOR_NAMES},
                           **{k2: v for k2, v in r.items() if isinstance(v, float)}}
                    if dd:
                        row["endogenous_gap"] = float(free_net[k] - float(frame["oracle_net"]
                                                                          .iloc[env.burn_in:].mean()))
                    rows.append(row)
                km = kaplan_meier([d for d, _ in episodes], [e for _, e in episodes], unmeasurable)
                rec_rows.append({"family": family, "variant": variant, "cell": label, "policy": policy,
                                 "role": ROLE[policy], "specification": SPECIFICATION[family][policy][0],
                                 **{f: getattr(env, f) for f in FACTOR_NAMES}, **km})
            done += 1
            note(done / total, f"{family}/{variant} cell {label}")
    note(1.0, "done")
    cells = pd.DataFrame(rows)
    return {"cells": cells, "recovery": pd.DataFrame(rec_rows),
            "summary": summary_table(cells), "factors": factor_effects(cells),
            "specification": specification_table()}


def summary_table(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per family and policy, with the paired standard error over cells and paths."""
    g = cells.groupby(["family", "variant", "policy", "role", "specification"], as_index=False).agg(
        regret=("regret", "mean"), regret_sd=("regret", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
        cum_net_payoff=("cum_net_payoff", "mean"),
        tail_loss_cvar5=("tail_loss_cvar5", "mean"),
        calibration_gap=("calibration_gap", "mean"),
        service_level=("service_level", "mean"),
        adjustment_cost=("adjustment_cost", "mean"),
        runs=("regret", "size"))
    g["regret_se"] = g["regret_sd"] / np.sqrt(g["runs"].clip(lower=1))
    return g.drop(columns=["regret_sd"]).sort_values(["family", "regret"]).reset_index(drop=True)


def factor_effects(cells: pd.DataFrame) -> pd.DataFrame:
    """The matched contrast for each factor: the mean change in regret when that factor alone moves from its low
    to its high level, every other factor held at the same corner. This is what the factorial design is for -
    each effect is a paired difference over 16 matched cells, not a regression coefficient."""
    out = []
    for (family, variant, policy), g in cells.groupby(["family", "variant", "policy"]):
        wide = g.groupby("cell")["regret"].mean()
        for j, factor in enumerate(FACTOR_NAMES):
            diffs = []
            for lab in wide.index:
                if lab[j] == "-":
                    hi = lab[:j] + "+" + lab[j + 1:]
                    if hi in wide.index:
                        diffs.append(wide[hi] - wide[lab])
            if diffs:
                d = np.asarray(diffs, float)
                # a single pair has a difference but no spread: report the effect and leave the error undefined
                se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
                out.append({"family": family, "variant": variant, "policy": policy, "factor": factor,
                            "low": FACTORS[factor][0], "high": FACTORS[factor][1],
                            "effect_on_regret": float(d.mean()), "standard_error": se, "pairs": len(d)})
    return pd.DataFrame(out)


def specification_table() -> pd.DataFrame:
    rows = []
    for family, per in SPECIFICATION.items():
        for policy, (status, why) in per.items():
            rows.append({"family": family, "family_name": PROCESSES[family].name, "policy": policy,
                         "role": ROLE[policy], "status": status, "reason": why})
    return pd.DataFrame(rows).sort_values(["family", "policy"]).reset_index(drop=True)


def process_table() -> pd.DataFrame:
    return pd.DataFrame([{"family": p.key, "name": p.name, "kind": p.kind,
                          "variants": ", ".join(p.variants),
                          "tuning_variant": ", ".join(TUNING_VARIANTS[p.key]),
                          "reserved_variant": ", ".join(EVALUATION_VARIANTS[p.key]),
                          "decision_dependent": p.decision_dependent,
                          "equations": p.equations, "information": p.information, "breaks": p.breaks,
                          "sources": ", ".join(p.sources)} for p in PROCESSES.values()])


def run_suite(plan: Optional[StressPlan] = None,
              progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, pd.DataFrame]:
    """Tune, then evaluate. The two halves never share a variant of a stress family or a seed block."""
    plan = plan or StressPlan()
    note = progress or (lambda f, m: None)
    tuned, tuning_rows = tune(plan, lambda f, m: note(0.35 * f, m))
    frames = run_evaluation(plan, tuned, lambda f, m: note(0.35 + 0.65 * f, m))
    frames["tuning"] = tuning_rows
    frames["budget"] = budget_table(plan)
    frames["boundary"] = boundary_report(tuned)
    frames["processes"] = process_table()
    frames["tuned"] = pd.DataFrame([{"policy": k, **v} for k, v in tuned.items()])
    return frames


EXPORTS = {
    "cells": "one row per family, variant, design cell, policy and path: every metric",
    "summary": "one row per family and policy: regret with its standard error, payoff, tail loss, calibration",
    "recovery": "Kaplan-Meier recovery per cell and policy, with the censoring rate beside it",
    "factors": "matched contrast for each of the five factors, per family and policy",
    "specification": "which model is correctly specified, estimated, agnostic or misspecified, and why",
    "processes": "the five stress families and the control: equations, information, reserved variant",
    "tuning": "every configuration scored during tuning, and the one selected",
    "budget": "the computational budget each policy was given",
    "boundary": "selected parameters sitting at the edge of their search range",
    "tuned": "the frozen configuration each policy carried into evaluation",
}
