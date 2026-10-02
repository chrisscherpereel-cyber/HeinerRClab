"""Agent tournament: rival theories of flexibility compete as agents in the same market.

Each theory is implemented as decision rules (agent designs). All agents face the same cobweb oligopoly as the main
model: the same demand curve and unannounced demand-regime shifts, the same reflecting raw-material cost process,
the same information (last price, last market quantity, own output, a cost estimate of the same quality) and the same
random streams. They differ only in how they turn that information into output.

Many designs are compositions of a *target* (where a flexible firm would move) and a *selection rule* (when it moves):
    targets:          model-based (best reply on the believed demand curve, with filtered forecasts)
                      price-based (adaptive expectation of the price; no demand model needed)
    selection rules:  always · inaction band (real options) · reliability condition (Heiner) · aspiration (satisficing)
so Heiner's claim, that a reliability-based selection rule improves on a flexible rule, can be tested on the same
target. Other designs (Cournot-Nash play, a markup rule, win-stay/lose-shift, two learners, two imitators) stand alone.

Fairness protocol (see PREREG):
    1. every theory has two designs, and every design is tuned with the same budget of evaluations (Latin-hypercube
       search) on training environments only, in two rounds; each theory enters the tournament with whichever of its
       two designs scored higher on the training environments;
    2. environments are drawn at random from pre-registered ranges; training and test environments are disjoint;
    3. the plan (ranges, budgets, hypotheses, decision rules) and the agent and analysis code are hashed;
    4. results use several yardsticks: head-to-head profit, downside risk, survival, volatility, regret and worst case
       in mixed markets; a target x selection-rule experiment; invasion tests; global sensitivity analysis; and
       replication of the whole protocol with fresh seeds.

Everything is vectorized over markets: a batch holds B markets, and every agent parameter is an array over markets,
so a whole tuning grid runs as one batch.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .agents import belief_index, cost_path, demand_path, make_streams
from .analysis import ols
from .params import FirmSpec, GlobalFirmParams, MarketParams, Scenario, StructuralParams

P_MAX, P_MIN, C0, Q_MIN = 100.0, 10.0, 45.0, 5.0
N_BINS = 5
CAPITAL_PERIODS = 20      # survival buffer: a firm is ruined if cumulative profit falls below -20 periods of Nash profit


# ================================================================================================ environments
@dataclass(frozen=True)
class Env:
    delta: float          # cost volatility
    c_max: float          # upper cost bound (profitability)
    q_range: float        # demand quantity range (slope = (P_MAX - P_MIN) / q_range)
    noise: float          # perception noise sigma (same for every agent)
    foresight: float      # cost foresight kappa (same for every agent)
    hazard: float         # demand-regime shift probability (0 = none)
    belief_lag: int       # periods before the demand model is updated after a shift
    seed: int


def sample_envs(n: int, seed: int, ranges: Dict[str, Tuple[float, float]], seed_base: int) -> List[Env]:
    """Draw environments uniformly from the pre-registered ranges. A share of them have demand-regime shifts."""
    rng = np.random.default_rng(seed)
    u = lambda k: float(rng.uniform(*ranges[k]))
    envs = []
    for e in range(n):
        shifts = rng.random() < ranges["share_with_shifts"][0]
        envs.append(Env(delta=u("delta"), c_max=u("c_max"), q_range=u("q_range"), noise=u("noise"),
                        foresight=u("foresight"), hazard=u("hazard") if shifts else 0.0,
                        belief_lag=int(round(u("belief_lag"))), seed=seed_base + e))
    return envs


def _scenario(env: Env, n: int, periods: int) -> Scenario:
    return Scenario(market=MarketParams(p_max=P_MAX, p_min=P_MIN, q_range=env.q_range, c0=C0, c_max=env.c_max,
                                        delta=min(env.delta, env.c_max - P_MIN)),
                    firms=[FirmSpec() for _ in range(n)], firm_globals=GlobalFirmParams(q_min=Q_MIN),
                    structural=StructuralParams(enabled=env.hazard > 0, hazard=env.hazard, belief_lag=env.belief_lag),
                    periods=periods, seed=env.seed)


def nash_output(q_range: float, n: int) -> float:
    """Static Cournot-Nash output per firm at the initial cost: every market starts at equilibrium."""
    s = (P_MAX - P_MIN) / q_range
    return max(Q_MIN, round((P_MAX - C0) / (s * (n + 1))))


def nash_profit(q_range: float, n: int) -> float:
    s = (P_MAX - P_MIN) / q_range
    q = nash_output(q_range, n)
    return max(1.0, (P_MAX - s * n * q - C0) * q)


# ================================================================================================ market state
class Market:
    """Shared state of a batch of markets, read by every agent."""

    def __init__(self, envs: Sequence[Env], n: int, periods: int):
        self.B, self.N, self.T = len(envs), n, periods
        scns = [_scenario(e, n, periods) for e in envs]
        streams = [make_streams(s.seed, n, periods) for s in scns]
        self.cost = np.stack([cost_path(s, st[0]) for s, st in zip(scns, streams)])          # (B, T)
        self.eps = np.stack([st[1] for st in streams])                                       # (B, T, N)
        dp = [demand_path(s) for s in scns]
        self.PM, self.SL = np.stack([d[0] for d in dp]), np.stack([d[1] for d in dp])
        base_pm = np.full((self.B, 1), P_MAX)
        base_sl = np.array([s.market.slope for s in scns])[:, None]
        bi = np.stack([belief_index(s) for s in scns])
        self.PMB = np.where(bi < 0, base_pm, np.take_along_axis(self.PM, np.maximum(bi, 0), axis=1))
        self.SLB = np.where(bi < 0, base_sl, np.take_along_axis(self.SL, np.maximum(bi, 0), axis=1))
        # agents' own random draws (exploration, imitation), independent of the market's streams
        rngs = [np.random.default_rng([e.seed, 7]) for e in envs]
        self.UR = np.stack([r.random((periods, n)) for r in rngs])
        self.ZR = np.stack([r.standard_normal((periods, n)) for r in rngs])
        self.kappa = np.array([e.foresight for e in envs])
        self.sigma = np.array([e.noise for e in envs])
        self.R_hist = np.zeros((self.B, periods, n))     # rivals' actual output faced by each firm
        self.q_hist = np.zeros((self.B, periods, n))     # each firm's actual output
        self.t = 0
        self.P_prev = self.Q_prev = None
        self.q_prev = self.profit_prev = self.c_hat = None

    def br(self, idx, c_hat, rivals):
        """Best reply on the believed demand curve (the model-based target)."""
        pm, sl = self.PMB[idx, self.t], self.SLB[idx, self.t]
        return (pm - sl * rivals - c_hat) / (2 * sl)


# ================================================================================================ agents
@dataclass(frozen=True)
class Spec:
    default: float
    lo: float
    hi: float
    scale: str = "lin"        # "lin", "log" or "int"
    help: str = ""


class Agent:
    """One agent design on one slot of a subset `idx` of the batch's markets. Parameters are arrays over idx."""
    key = ""
    name = ""
    theory = ""
    rule = ""
    sources: Tuple[str, ...] = ()
    SPACE: Dict[str, Spec] = {}

    def __init__(self, idx: np.ndarray, slot: int, p: Dict[str, np.ndarray], mk: Market):
        self.idx, self.i, self.p, self.mk = idx, slot, p, mk
        self.n = len(idx)

    @property
    def q(self):
        return self.mk.q_prev[self.idx, self.i]

    @property
    def c_hat(self):
        return self.mk.c_hat[self.idx, self.i]

    @property
    def rivals(self):
        return self.mk.Q_prev[self.idx] - self.q

    @property
    def profit(self):
        return self.mk.profit_prev[self.idx, self.i]

    def act(self) -> np.ndarray:
        raise NotImplementedError

    def update(self):
        pass


# ------------------------------------------------------------------------------------------------ composite designs
TARGET_SPACE = {
    "model": {"a_cost": Spec(1.0, 0.05, 1.0, "log", "Filter gain on the cost estimate"),
              "a_rival": Spec(1.0, 0.05, 1.0, "log", "Filter gain on rivals' output"),
              "phi": Spec(1.0, 0.05, 1.0, help="Share of the way toward the target per move")},
    "price": {"lam": Spec(0.5, 0.05, 1.0, "log", "Adaptive-expectations gain on the price"),
              "phi": Spec(0.5, 0.05, 1.0, help="Share of the way toward the target per move")},
}
SELECT_SPACE = {
    "always": {},
    "band": {"k": Spec(1.0, 0.0, 3.0, help="Band width in standard deviations of target changes")},
    "rc": {"theta": Spec(10.0, 1.0, 60.0, "log", "Scale of the size-of-change bins"),
           "memory": Spec(0.97, 0.8, 0.995, help="Exponential memory of the learned gains"),
           "horizon": Spec(10, 1, 20, "int", "Periods over which a deviation is judged")},
    "aspiration": {"alpha": Spec(0.1, 0.01, 0.5, "log", "Speed at which aspirations adapt")},
}
TARGET_TEXT = {"model": "best reply on the believed demand curve, with filtered cost and rival forecasts",
               "price": "output that is best at an adaptive price expectation (no demand model)"}
SELECT_TEXT = {"always": "moves a share φ toward it every period",
               "band": "moves only when the gap exceeds a band that widens with the volatility of the target",
               "rc": ("keeps rule B unless its learned reliability condition holds for changes of that size "
                      "(π·G − (1−π)·D ≥ 0, judged ex post over h periods)"),
               "aspiration": "keeps its output while profit meets an adaptive aspiration; below it, moves"}


class Composite(Agent):
    """Target x selection rule.

    Reliability-condition variants (SELECT = "rc"):
        GAIN = "believed"  judges past decisions with the agent's own (possibly outdated) demand model (default);
        GAIN = "true"      judges them with the true demand curve (removes model bias from the estimates);
        ORACLE = True      does not learn: deviates by fixed expected gains per size bin, passed as parameters
                           e0..e4 (the true reliability of its rule in this environment, estimated elsewhere).
    Every composite records its target's error against the ex-post best reply (Heiner's 1989 error-to-signal
    ratio K), and RC variants record the gain of every opportunity by bin (used to estimate oracle values)."""
    TARGET = "model"
    SELECT = "always"
    FIXED: Dict[str, float] = {}
    GAIN = "believed"
    ORACLE = False

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.p = {**{k: np.full(self.n, v) for k, v in self.FIXED.items()}, **self.p}
        self.cf = self.rf = self.pe = None
        self.v, self.last = np.zeros(self.n), None
        self.A = None
        self.tgt = np.zeros((mk.T, self.n))
        self.err2 = np.zeros(self.n)
        self.sig2 = np.zeros(self.n)
        self.gsum = np.zeros((self.n, N_BINS))
        self.gcnt = np.zeros((self.n, N_BINS))
        if self.SELECT == "rc":
            self.E = np.zeros((self.n, N_BINS))
            if self.ORACLE:
                self.E = np.column_stack([self.p[f"e{b}"] for b in range(N_BINS)])
            self.qs = np.zeros((mk.T, self.n))
            self.qb = np.zeros((mk.T, self.n))
            self.bin = np.zeros((mk.T, self.n), dtype=int)
            self.opp = np.zeros((mk.T, self.n), dtype=bool)
            self.h = self.p["horizon"].astype(int)
            self.edges = self.p["theta"][:, None] * np.array([0.5, 1.0, 2.0, 4.0])[None, :]

    def target(self):
        mk = self.mk
        if self.TARGET == "model":
            if self.cf is None:
                self.cf, self.rf = self.c_hat.copy(), self.rivals.copy()
            else:
                self.cf = self.cf + self.p["a_cost"] * (self.c_hat - self.cf)
                self.rf = self.rf + self.p["a_rival"] * (self.rivals - self.rf)
            return mk.br(self.idx, self.cf, self.rf)
        P = mk.P_prev[self.idx]
        self.pe = P.copy() if self.pe is None else self.pe + self.p["lam"] * (P - self.pe)
        s = mk.SLB[self.idx, mk.t]
        return (self.pe + s * self.q - self.c_hat) / (2 * s)

    def act(self):
        q, tgt = self.q, self.target()
        self.tgt[self.mk.t] = tgt
        cand = np.maximum(Q_MIN, np.rint(q + self.p["phi"] * (tgt - q)))
        if self.SELECT == "always":
            dev = np.ones(self.n, bool)
        elif self.SELECT == "band":
            if self.last is not None:
                self.v = 0.9 * self.v + 0.1 * (tgt - self.last) ** 2
            self.last = tgt
            dev = np.abs(cand - q) > self.p["k"] * np.sqrt(self.v)
        elif self.SELECT == "aspiration":
            pr = self.profit
            if self.A is None:
                self.A = pr.copy()
            dev = pr < self.A
            self.A = self.A + self.p["alpha"] * (pr - self.A)
        else:  # rc
            t = self.mk.t
            sig = np.abs(cand - q)
            b = (sig[:, None] > self.edges).sum(1)
            self.qs[t], self.qb[t], self.bin[t], self.opp[t] = cand, q, b, sig > 0
            dev = (sig > 0) & (self.E[np.arange(self.n), b] >= 0)
        return np.where(dev, cand, q)

    def update(self):
        """Record the target's error against the ex-post best reply. Reliability-condition learning: judge the decision
        made h periods ago by holding the candidate vs holding the old level (rule B) for h periods, rivals on their
        actual path, prices from the believed (or true) demand curve and realized costs."""
        mk, t = self.mk, self.mk.t
        R, i = mk.R_hist[self.idx, t, self.i], self.idx
        br_true = (mk.PM[i, t] - mk.SL[i, t] * R - mk.cost[i, t]) / (2 * mk.SL[i, t])
        self.err2 += (self.tgt[t] - br_true) ** 2
        self.sig2 += (br_true - self.q_before(t)) ** 2
        if self.SELECT != "rc":
            return
        demand = (mk.PMB, mk.SLB) if self.GAIN == "believed" else (mk.PM, mk.SL)
        for h in np.unique(self.h):
            d = t - h + 1
            if d < 1:
                continue
            rows = np.flatnonzero((self.h == h) & self.opp[d])
            if not len(rows):
                continue
            mi = self.idx[rows]
            R = mk.R_hist[mi, d:t + 1, self.i]
            pm, sl, c = demand[0][mi, d:t + 1], demand[1][mi, d:t + 1], mk.cost[mi, d:t + 1]
            x1, x0 = self.qs[d, rows][:, None], self.qb[d, rows][:, None]
            gain = (((np.maximum(P_MIN, pm - sl * (R + x1)) - c) * x1)
                    - ((np.maximum(P_MIN, pm - sl * (R + x0)) - c) * x0)).sum(1)
            b = self.bin[d, rows]
            self.gsum[rows, b] += gain
            self.gcnt[rows, b] += 1
            if not self.ORACLE:
                lam = self.p["memory"][rows]
                self.E[rows, b] = lam * self.E[rows, b] + (1 - lam) * gain

    def q_before(self, t):
        """Own output before the period-t decision (rule B's level)."""
        return self.mk.q_hist[self.idx, t - 1, self.i]


def composite(key, name, theory, target, select, sources, fixed=None, gain="believed", oracle=False):
    space = {**TARGET_SPACE[target], **SELECT_SPACE[select]}
    for k in (fixed or {}):
        space.pop(k, None)
    rule = f"Target: {TARGET_TEXT[target]}. Selection: {SELECT_TEXT[select]}."
    return type(key, (Composite,), dict(key=key, name=name, theory=theory, TARGET=target, SELECT=select,
                                        FIXED=dict(fixed or {}), SPACE=space, rule=rule, sources=tuple(sources),
                                        GAIN=gain, ORACLE=oracle))


# ------------------------------------------------------------------------------------------------ stand-alone designs
class RuleB(Agent):
    key, name, theory = "ruleb", "Rule B (rigid)", "Benchmark"
    rule = "Always keeps last period's output."
    sources = ("heiner1983",)

    def act(self):
        return self.q


class NashRE(Agent):
    key, name, theory = "opt_nash", "Rational expectations (Cournot–Nash)", "Neoclassical optimization"
    rule = ("Expects every rival to be rational and moves toward the static Cournot–Nash output for its filtered cost "
            "estimate on the believed demand curve.")
    sources = ("muth1961", "theocharis1960")
    SPACE = {"a_cost": Spec(1.0, 0.05, 1.0, "log", "Filter gain on the cost estimate"),
             "phi": Spec(1.0, 0.05, 1.0, help="Share of the way toward the Nash output per period")}

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.cf = None

    def act(self):
        mk = self.mk
        self.cf = self.c_hat.copy() if self.cf is None else self.cf + self.p["a_cost"] * (self.c_hat - self.cf)
        tgt = (mk.PMB[self.idx, mk.t] - self.cf) / (mk.SLB[self.idx, mk.t] * (mk.N + 1))
        return self.q + self.p["phi"] * (tgt - self.q)


class Heuristic(Agent):
    key, name, theory = "heur_wsls", "Win-stay, lose-shift", "Bias–variance / ecological rationality"
    rule = ("Ignores the demand model: keeps changing output in the same direction while profit rises, reverses when it "
            "falls (fixed step).")
    sources = ("gigerenzer2009", "nowak1993")
    SPACE = {"step": Spec(5.0, 1.0, 40.0, "log", "Size of each output change")}

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.dir = np.ones(self.n)
        self.last = None

    def act(self):
        pr = self.profit
        if self.last is not None:
            self.dir = np.where(pr >= self.last, self.dir, -self.dir)
        self.last = pr.copy()
        return self.q + self.dir * self.p["step"]


class Markup(Agent):
    key, name, theory = "heur_markup", "Target-margin rule", "Bias–variance / ecological rationality"
    rule = "Raises output when the observed margin P − ĉ exceeds a target margin m, lowers it otherwise (step φ)."
    sources = ("hall1939", "gigerenzer2009")
    SPACE = {"phi": Spec(0.5, 0.05, 3.0, "log", "Output change per unit of margin gap"),
             "m": Spec(5.0, 0.0, 30.0, help="Target margin")}

    def act(self):
        P = self.mk.P_prev[self.idx]
        return self.q + self.p["phi"] * (P - self.c_hat - self.p["m"])


class Reinforcement(Agent):
    key, name, theory = "rl_softmax", "Softmax value learner", "Reinforcement learning"
    rule = ("Chooses among five moves (−2, −1, 0, +1, +2 steps) by softmax over learned values; a move's value is the "
            "profit change that followed it.")
    sources = ("sutton2018",)
    SPACE = {"step": Spec(5.0, 1.0, 30.0, "log", "Size of one step"),
             "eta": Spec(0.1, 0.01, 0.5, "log", "Learning rate"),
             "temp": Spec(0.2, 0.01, 2.0, "log", "Exploration temperature (relative to typical profit changes)")}
    MOVES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.V = np.zeros((self.n, 5))
        self.a = None
        self.last = None
        self.scale = np.ones(self.n)

    def probs(self):
        z = self.V / (self.p["temp"] * self.scale + 1e-9)[:, None]
        w = np.exp(z - z.max(1, keepdims=True))
        return w / w.sum(1, keepdims=True)

    def learn(self, r):
        rows = np.arange(self.n)
        self.V[rows, self.a] += self.p["eta"] * (r - self.V[rows, self.a])

    def act(self):
        pr = self.profit
        if self.a is not None:
            r = pr - self.last
            self.scale = 0.95 * self.scale + 0.05 * np.abs(r)
            self.learn(r)
        cdf = np.cumsum(self.probs(), axis=1)
        u = self.mk.UR[self.idx, self.mk.t, self.i]
        self.a = np.minimum((cdf < u[:, None]).sum(1), 4)
        self.last = pr.copy()
        return self.q + self.MOVES[self.a] * self.p["step"]


class ErevRoth(Reinforcement):
    key, name = "rl_erevroth", "Erev–Roth propensity learner"
    rule = ("Chooses among five moves with probability proportional to propensities; a move's propensity grows with "
            "the profit gain that followed it, and all propensities decay (forgetting).")
    sources = ("erev1998", "roth1995")
    SPACE = {"step": Spec(5.0, 1.0, 30.0, "log", "Size of one step"),
             "forget": Spec(0.05, 0.001, 0.3, "log", "Forgetting rate of propensities"),
             "gain": Spec(1.0, 0.1, 10.0, "log", "Reinforcement per typical profit change")}

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.V = np.ones((self.n, 5))

    def probs(self):
        return self.V / self.V.sum(1, keepdims=True)

    def learn(self, r):
        rows = np.arange(self.n)
        self.V *= (1 - self.p["forget"])[:, None]
        self.V[rows, self.a] += self.p["gain"] * np.maximum(r, 0) / (self.scale + 1e-9)
        self.V = np.maximum(self.V, 1e-3)


class Imitator(Agent):
    key, name, theory = "imit_best", "Imitate the best", "Evolutionary selection / imitation"
    rule = ("With probability p copies last period's output of the most profitable firm in the market, plus small "
            "random experimentation.")
    sources = ("vegaredondo1997", "alchian1950")
    SPACE = {"p": Spec(0.5, 0.05, 1.0, help="Probability of imitating each period"),
             "noise": Spec(2.0, 0.0, 20.0, help="Standard deviation of experimentation")}

    def model(self):
        mk, idx = self.mk, self.idx
        return mk.q_prev[idx, mk.profit_prev[idx].argmax(1)]

    def act(self):
        mk = self.mk
        u, z = mk.UR[self.idx, mk.t, self.i], mk.ZR[self.idx, mk.t, self.i]
        return np.where(u < self.p["p"], self.model(), self.q) + self.p["noise"] * z


class Conformist(Imitator):
    key, name = "imit_avg", "Imitate the average"
    rule = ("With probability p moves to the average output of the other firms (conformist transmission), plus small "
            "random experimentation.")
    sources = ("boyd1985", "alchian1950")

    def model(self):
        return self.rivals / (self.mk.N - 1)


DESIGNS: Dict[str, type] = {d.key: d for d in (
    composite("heiner_m", "Reliability condition · model-based target", "Heiner: reliability condition", "model", "rc",
              ("heiner1983", "heiner1989")),
    composite("heiner_p", "Reliability condition · price-based target", "Heiner: reliability condition", "price", "rc",
              ("heiner1983", "heiner1989")),
    composite("opt_br", "Filtered best reply", "Neoclassical optimization", "model", "always",
              ("muth1960", "kalman1960")),
    NashRE,
    composite("options_m", "Inaction band · model-based target", "Real options / value of flexibility", "model",
              "band", ("dixit1989", "dixit1994")),
    composite("options_p", "Inaction band · price-based target", "Real options / value of flexibility", "price", "band",
              ("dixit1989", "dixit1994")),
    composite("cobweb_p", "Adaptive price expectations", "Cobweb stability theory", "price", "always",
              ("nerlove1958", "ezekiel1938")),
    composite("cobweb_q", "Adaptive quantity adjustment", "Cobweb stability theory", "model", "always",
              ("theocharis1960", "nerlove1958"), fixed={"a_cost": 1.0}),
    Heuristic, Markup,
    composite("satis_m", "Aspiration search · model-based target", "Satisficing / aspiration-level search", "model",
              "aspiration", ("simon1955", "cyert1963")),
    composite("satis_p", "Aspiration search · price-based target", "Satisficing / aspiration-level search", "price",
              "aspiration", ("simon1955", "cyert1963")),
    Reinforcement, ErevRoth, Imitator, Conformist, RuleB)}

THEORY_DESIGNS: Dict[str, Tuple[str, ...]] = {
    "heiner": ("heiner_m", "heiner_p"), "optimiser": ("opt_br", "opt_nash"), "options": ("options_m", "options_p"),
    "cobweb": ("cobweb_p", "cobweb_q"), "heuristic": ("heur_wsls", "heur_markup"),
    "satisficing": ("satis_m", "satis_p"), "rl": ("rl_softmax", "rl_erevroth"), "imitation": ("imit_best", "imit_avg"),
    "ruleb": ("ruleb",)}
THEORY_NAMES = {"heiner": "Heiner: reliability condition", "optimiser": "Neoclassical optimization",
                "options": "Real options", "cobweb": "Cobweb / adaptive expectations",
                "heuristic": "Simple heuristics (bias–variance)", "satisficing": "Satisficing",
                "rl": "Reinforcement learning", "imitation": "Imitation / evolutionary selection",
                "ruleb": "Rule B (benchmark)"}
KEYS = list(THEORY_DESIGNS)                       # theories, in tournament slot order
TUNED = [k for k in KEYS if k != "ruleb"]
# the target x selection-rule experiment: same targets, four selection rules
FACTORIAL = ("opt_br", "options_m", "heiner_m", "satis_m", "cobweb_p", "options_p", "heiner_p", "satis_p")
AGENTS = DESIGNS                                   # alias

# Reliability-condition variants used by the mechanism study (not tournament entries): the same agent judging its
# past decisions with the true demand curve, and an oracle that knows the true expected gain of each kind of deviation.
VARIANTS: Dict[str, type] = {d.key: d for d in (
    composite("heiner_m_true", "Reliability condition · model-based · judged with the true model",
              "Heiner: reliability condition", "model", "rc", ("heiner1983",), gain="true"),
    composite("heiner_p_true", "Reliability condition · price-based · judged with the true model",
              "Heiner: reliability condition", "price", "rc", ("heiner1983",), gain="true"),
    composite("heiner_m_oracle", "Reliability condition · model-based · oracle", "Heiner: reliability condition",
              "model", "rc", ("heiner1983",), oracle=True),
    composite("heiner_p_oracle", "Reliability condition · price-based · oracle", "Heiner: reliability condition",
              "price", "rc", ("heiner1983",), oracle=True),
)}
ALL_DESIGNS: Dict[str, type] = {**DESIGNS, **VARIANTS}


def default_params(design: str) -> Dict[str, float]:
    return {n: s.default for n, s in DESIGNS[design].SPACE.items()}


# ================================================================================================ simulation
def simulate(envs: Sequence[Env], lineup: np.ndarray, params: Dict[str, Dict[str, np.ndarray]], periods: int,
             burn_in: int, keep_path: bool = False) -> Dict[str, np.ndarray]:
    """Run B markets. lineup (B, N) holds design keys per market and slot; params[design][name] is an array over the
    B markets (or a scalar). Returns average profit per period after burn-in (B, N), the share of periods in which
    each firm changed its output, and with keep_path the per-period profits after burn-in (B, T - burn_in, N)."""
    lineup = np.asarray(lineup, dtype=object)
    B, N = lineup.shape
    mk = Market(envs, N, periods)
    q0 = np.array([nash_output(e.q_range, N) for e in envs], float)
    q = np.repeat(q0[:, None], N, axis=1)
    Q = q.sum(1)
    P = np.maximum(P_MIN, mk.PM[:, 0] - mk.SL[:, 0] * Q)
    mk.P_prev, mk.Q_prev, mk.q_prev = P, Q, q
    mk.profit_prev = (P - mk.cost[:, 0])[:, None] * q
    mk.R_hist[:, 0] = Q[:, None] - q
    mk.q_hist[:, 0] = q
    agents: List[Agent] = []
    for i in range(N):
        for key in np.unique(lineup[:, i]):
            idx = np.flatnonzero(lineup[:, i] == key)
            cls = ALL_DESIGNS[key]
            given = params.get(key, {})
            names = list(cls.SPACE) + [n for n in given if n not in cls.SPACE]     # extra per-market inputs (oracle)
            p = {n: np.broadcast_to(np.asarray(given.get(n, cls.SPACE[n].default if n in cls.SPACE else 0.0), float),
                                    (B,))[idx].copy() for n in names}
            agents.append(cls(idx, i, p, mk))
    sum_profit = np.zeros((B, N))
    changes = np.zeros((B, N))
    path = np.zeros((B, periods - burn_in, N)) if keep_path else None
    for t in range(1, periods):
        mk.t = t
        c_prev, c_now = mk.cost[:, t - 1], mk.cost[:, t]
        mk.c_hat = (c_prev + mk.kappa * (c_now - c_prev))[:, None] + mk.sigma[:, None] * mk.eps[:, t, :]
        new = np.empty((B, N))
        for a in agents:
            new[a.idx, a.i] = a.act()
        new = np.maximum(Q_MIN, np.rint(np.nan_to_num(new, nan=Q_MIN, posinf=Q_MIN, neginf=Q_MIN)))
        Q = new.sum(1)
        P = np.maximum(P_MIN, mk.PM[:, t] - mk.SL[:, t] * Q)
        profit = (P - c_now)[:, None] * new
        mk.R_hist[:, t] = Q[:, None] - new
        mk.q_hist[:, t] = new
        if t >= burn_in:
            sum_profit += profit
            changes += new != mk.q_prev
            if keep_path:
                path[:, t - burn_in] = profit
        mk.P_prev, mk.Q_prev, mk.q_prev, mk.profit_prev = P, Q, new, profit
        for a in agents:
            a.update()
    n_rec = periods - burn_in
    K = np.full((B, N), np.nan)
    gsum, gcnt = np.zeros((B, N, N_BINS)), np.zeros((B, N, N_BINS))
    for a in agents:
        if isinstance(a, Composite):
            K[a.idx, a.i] = np.sqrt(a.err2 / np.maximum(a.sig2, 1e-12))
            gsum[a.idx, a.i], gcnt[a.idx, a.i] = a.gsum, a.gcnt
    return dict(profit=sum_profit / n_rec, change_rate=changes / n_rec, path=path, K=K, gain_sum=gsum, gain_count=gcnt)


# ================================================================================================ pre-registration
ENV_RANGES = {"delta": (2.0, 30.0), "c_max": (60.0, 95.0), "q_range": (800.0, 3000.0), "noise": (0.0, 10.0),
              "foresight": (0.0, 0.5), "hazard": (0.005, 0.05), "belief_lag": (0.0, 40.0),
              "share_with_shifts": (0.5, 0.5)}

HYPOTHESES = (
    ("PR1", "Head-to-head: in held-out mixed markets the reliability-condition agent earns more than each rival.",
     "Paired profit difference per market; 95% cluster-bootstrap CI over environments, Holm-adjusted across rivals. "
     "Supported if the reliability-condition agent is significantly better than every rival."),
    ("PR2", "Conditional advantage: the reliability-condition agent's advantage over the optimizer grows with "
            "environmental difficulty.",
     "OLS of the paired difference on a difficulty index (mean of standardized Δ, σ and hazard). Supported if the "
     "slope is positive with p < α."),
    ("PR3", "Evolutionary stability: a population of reliability-condition agents cannot be invaded by any rival.",
     "For each rival mutant, the mutant's profit minus the residents' mean profit. Supported if no mutant is "
     "significantly better (Holm-adjusted)."),
    ("PR4", "Invasion: the reliability-condition agent can invade every rival population.",
     "Mutant reliability-condition agent's profit minus the residents' mean profit. Supported if it is significantly "
     "positive (Holm-adjusted) in every rival population."),
    ("PR5", "Selection rule: reliability-condition selection improves on always adjusting toward the same target.",
     "Target × selection-rule market. Paired profit difference reliability condition − always, for the model-based "
     "and the price-based target, Holm-adjusted. Supported if significantly positive for both targets."),
    ("PR6", "Downside risk: the reliability-condition agent has less downside risk than always adjusting toward the "
            "same target.",
     "Same market. Paired difference in CVaR 5% of per-period profit (mean of the worst 5% of periods), reliability "
     "condition − always, Holm-adjusted over the two targets. Supported if significantly positive for both."),
)


@dataclass(frozen=True)
class Prereg:
    """The frozen analysis plan. Its hash identifies the plan and the code; any change makes a run exploratory."""
    version: str = "2.0"
    env_ranges: Tuple[Tuple[str, Tuple[float, float]], ...] = tuple(ENV_RANGES.items())
    n_train: int = 24
    n_test: int = 40
    reps_test: int = 2
    periods: int = 800
    burn_in: int = 50
    budget: int = 24
    rounds: int = 2
    train_seed: int = 1001
    test_seed: int = 2002
    residents: int = 4
    alpha: float = 0.05
    n_boot: int = 1000
    hypotheses: Tuple[Tuple[str, str, str], ...] = HYPOTHESES

    @property
    def ranges(self) -> Dict[str, Tuple[float, float]]:
        return dict(self.env_ranges)

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "code_sha256": code_digest()}, sort_keys=True, indent=1)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def code_digest() -> str:
    """Hash of the agents, the simulator and the analysis code: changing any of them changes the plan's hash."""
    hand_written = [d for d in DESIGNS.values() if not issubclass(d, Composite)]
    objs = (*hand_written, Agent, Composite, Market,
            simulate, tune, head_to_head, factorial, invasion, criteria, evaluate, sample_envs, composite)
    src = "".join(sorted(inspect.getsource(o) for o in objs)) + json.dumps(
        {k: [d.TARGET, d.SELECT, d.FIXED, d.GAIN, d.ORACLE] for k, d in ALL_DESIGNS.items() if issubclass(d, Composite)},
        sort_keys=True) + json.dumps(THEORY_DESIGNS, sort_keys=True)
    return hashlib.sha256(src.encode()).hexdigest()


def train_envs(pr: Prereg) -> List[Env]:
    return sample_envs(pr.n_train, pr.train_seed, pr.ranges, seed_base=100_000)


def test_envs(pr: Prereg) -> List[Env]:
    base = sample_envs(pr.n_test, pr.test_seed, pr.ranges, seed_base=200_000)
    out = []
    for r in range(pr.reps_test):
        out += [Env(**{**asdict(e), "seed": e.seed + 50_000 * r}) for e in base]
    return out


# ================================================================================================ tuning
def _lhs(space: Dict[str, Spec], k: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Latin-hypercube sample of k parameter vectors; candidate 0 is the default."""
    out = {}
    for name, s in space.items():
        u = (rng.permutation(k) + rng.random(k)) / k
        if s.scale == "log":
            v = np.exp(np.log(s.lo) + u * (np.log(s.hi) - np.log(s.lo)))
        else:
            v = s.lo + u * (s.hi - s.lo)
        if s.scale == "int":
            v = np.rint(v)
        v[0] = s.default
        out[name] = v
    return out


@dataclass
class Tuned:
    """Selected design per theory, and tuned parameters for every design."""
    design: Dict[str, str]                          # theory -> design entering the tournament
    params: Dict[str, Dict[str, float]]             # design -> parameters

    def lineup(self, theories: Sequence[str] = KEYS) -> List[str]:
        return [self.design[t] for t in theories]


def tune(pr: Prereg, progress: Optional[Callable[[float, str], None]] = None) -> Tuple[Tuned, pd.DataFrame]:
    """Equal-budget tuning and design selection on the training environments.

    Every design gets `budget` candidate parameter vectors (default included) per round, evaluated in a mixed market
    with one firm per theory, where the other theories use their current design and parameters (round 1: first
    design, defaults). Each theory then enters with whichever of its two designs scored higher. Every theory has two
    designs, so every theory gets the same total budget."""
    envs = train_envs(pr)
    E, K = len(envs), pr.budget
    tuned = Tuned(design={t: d[0] for t, d in THEORY_DESIGNS.items()},
                  params={d: default_params(d) for d in DESIGNS})
    log = []
    steps, done = pr.rounds * sum(len(THEORY_DESIGNS[t]) for t in TUNED), 0
    for rnd in range(1, pr.rounds + 1):
        new_params = {d: dict(v) for d, v in tuned.params.items()}
        new_design = dict(tuned.design)
        for theory in TUNED:
            slot = KEYS.index(theory)
            best_score, best_design = -np.inf, None
            for design in THEORY_DESIGNS[theory]:
                if progress:
                    progress(done / steps, f"Round {rnd}: tuning {DESIGNS[design].name}")
                rng = np.random.default_rng([pr.train_seed, rnd, list(DESIGNS).index(design)])
                cand = _lhs(DESIGNS[design].SPACE, K, rng)
                lineup = np.tile(np.array(tuned.lineup(), dtype=object), (K * E, 1))
                lineup[:, slot] = design
                params = {d: dict(v) for d, v in tuned.params.items()}
                params[design] = {n: np.repeat(v, E) for n, v in cand.items()}
                out = simulate(envs * K, lineup, params, pr.periods, pr.burn_in)
                score = out["profit"][:, slot].reshape(K, E).mean(1)
                b = int(np.argmax(score))
                new_params[design] = {n: float(v[b]) for n, v in cand.items()}
                log.append(dict(round=rnd, theory=theory, design=design, design_name=DESIGNS[design].name,
                                default_score=float(score[0]), best_score=float(score[b]),
                                gain=float(score[b] - score[0]),
                                **{f"param:{n}": float(v[b]) for n, v in cand.items()}))
                if score[b] > best_score:
                    best_score, best_design = float(score[b]), design
                done += 1
            new_design[theory] = best_design
        tuned = Tuned(design=new_design, params=new_params)
    if progress:
        progress(1.0, "Tuning done")
    log = pd.DataFrame(log)
    last = log["round"] == log["round"].max()
    log["selected"] = last & log.apply(lambda r: tuned.design[r["theory"]] == r["design"], axis=1)
    return tuned, log


# ================================================================================================ yardsticks
def _market_rows(envs, keys, names, out, label) -> pd.DataFrame:
    """Per (market, agent) outcomes on several criteria from simulate(..., keep_path=True)."""
    rows = []
    path = out["path"]
    for b, e in enumerate(envs):
        prof = out["profit"][b]
        rank = (-prof).argsort().argsort() + 1
        cap = CAPITAL_PERIODS * nash_profit(e.q_range, len(keys))
        cum = np.cumsum(path[b], axis=0)
        k5 = max(1, int(round(0.05 * path.shape[1])))
        worst = np.sort(path[b], axis=0)[:k5].mean(0)
        for i, key in enumerate(keys):
            rows.append(dict(market=b, env=e.seed % 50_000, **{label: key}, name=names[key], profit=prof[i],
                             rank=int(rank[i]), rel_profit=prof[i] - prof.mean(), regret=prof.max() - prof[i],
                             sd=float(path[b, :, i].std()), cvar5=float(worst[i]),
                             survived=float(cum[:, i].min() > -cap), change_rate=out["change_rate"][b, i],
                             **asdict(e)))
    return pd.DataFrame(rows)


def head_to_head(pr: Prereg, tuned: Tuned) -> pd.DataFrame:
    """One firm per theory (its selected design) per market, held-out environments. One row per (market, theory)."""
    envs = test_envs(pr)
    lineup = np.tile(np.array(tuned.lineup(), dtype=object), (len(envs), 1))
    out = simulate(envs, lineup, tuned.params, pr.periods, pr.burn_in, keep_path=True)
    df = _market_rows(envs, KEYS, {t: THEORY_NAMES[t] for t in KEYS}, out, "agent")
    df["design"] = df["agent"].map(tuned.design)
    return df


def factorial(pr: Prereg, tuned: Tuned) -> pd.DataFrame:
    """Target x selection-rule market: the eight composite designs (two targets x four selection rules), each with its
    own tuned parameters, compete in the same held-out markets. One row per (market, design)."""
    envs = test_envs(pr)
    lineup = np.tile(np.array(FACTORIAL, dtype=object), (len(envs), 1))
    out = simulate(envs, lineup, tuned.params, pr.periods, pr.burn_in, keep_path=True)
    df = _market_rows(envs, list(FACTORIAL), {d: DESIGNS[d].name for d in FACTORIAL}, out, "design")
    df["target"] = df["design"].map(lambda d: DESIGNS[d].TARGET)
    df["select"] = df["design"].map(lambda d: DESIGNS[d].SELECT)
    return df


def invasion(pr: Prereg, tuned: Tuned, progress: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
    """For every resident theory Y and mutant theory X != Y (selected designs): markets with `residents` firms of
    type Y and one of type X. Fitness = mutant profit - mean resident profit. One row per (market, resident, mutant)."""
    envs = test_envs(pr)
    rows = []
    for r_i, res in enumerate(KEYS):
        if progress:
            progress(r_i / len(KEYS), f"Invasion of {THEORY_NAMES[res]} populations")
        muts = [k for k in KEYS if k != res]
        lineup = np.array([[tuned.design[res]] * pr.residents + [tuned.design[m]] for m in muts for _ in envs],
                          dtype=object)
        out = simulate(envs * len(muts), lineup, tuned.params, pr.periods, pr.burn_in)
        prof = out["profit"]
        fit = prof[:, -1] - prof[:, :-1].mean(1)
        scale = np.abs(prof[:, :-1].mean(1)) + 1e-9
        for j, (m, e) in enumerate([(m, e) for m in muts for e in envs]):
            rows.append(dict(resident=res, mutant=m, env=e.seed % 50_000, fitness=fit[j], rel_fitness=fit[j] / scale[j]))
    if progress:
        progress(1.0, "Invasion done")
    return pd.DataFrame(rows)


def cluster_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int, conf: float, seed: int = 0):
    """Mean with a cluster-bootstrap CI (resampling environments) and a two-sided bootstrap p-value for mean = 0."""
    df = pd.DataFrame(dict(v=values, c=clusters))
    g = df.groupby("c")["v"].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), (n_boot, len(g)))
    boots = s[pick].sum(1) / n[pick].sum(1)
    tail = 50 * (1 - conf)
    lo, hi = np.percentile(boots, [tail, 100 - tail])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(s.sum() / n.sum()), float(lo), float(hi), float(max(p, 1 / n_boot))


def holm(pvals: Sequence[float], alpha: float) -> List[bool]:
    """Holm (1979) step-down: which hypotheses are rejected at family-wise level alpha."""
    order = np.argsort(pvals)
    m, rej, stop = len(pvals), [False] * len(pvals), False
    for k, j in enumerate(order):
        if not stop and pvals[j] <= alpha / (m - k):
            rej[j] = True
        else:
            stop = True
    return rej


CRITERIA = (("rel_profit", "Mean profit (vs market mean)", 1), ("cvar5", "Downside: CVaR 5%", 1),
            ("survived", "Survival", 1), ("sd", "Volatility (s.d.)", -1), ("regret", "Regret vs market's best", -1),
            ("worst10", "Worst case: 10th percentile", 1))


def criteria(df: pd.DataFrame, label: str, pr: Prereg) -> pd.DataFrame:
    """Several performance criteria per agent, their ranks, an aggregate (mean) rank and Pareto efficiency."""
    rows = []
    for key, d in df.groupby(label, sort=False):
        m, lo, hi, _ = cluster_ci(d["rel_profit"].to_numpy(), d["env"].to_numpy(), pr.n_boot, 1 - pr.alpha)
        rows.append(dict(key=key, agent=d["name"].iloc[0], mean_rank=d["rank"].mean(), win_share=(d["rank"] == 1).mean(),
                         rel_profit=m, lo=lo, hi=hi, profit=d["profit"].mean(), cvar5=d["cvar5"].mean(),
                         survived=d["survived"].mean(), sd=d["sd"].mean(), regret=d["regret"].mean(),
                         worst10=float(np.percentile(d["rel_profit"], 10)), change_rate=d["change_rate"].mean()))
    out = pd.DataFrame(rows)
    for col, _, sign in CRITERIA:
        out[f"rank_{col}"] = (-sign * out[col]).rank(method="min")
    out["aggregate_rank"] = out[[f"rank_{c}" for c, _, _ in CRITERIA]].mean(1)
    vals = np.column_stack([sign * out[c].to_numpy(float) for c, _, sign in CRITERIA])
    out["pareto"] = [not any(np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]) for j in range(len(vals)) if j != i)
                     for i in range(len(vals))]
    return out.sort_values("mean_rank", ignore_index=True)


def ranking(h2h: pd.DataFrame, pr: Prereg) -> pd.DataFrame:
    out = criteria(h2h, "agent", pr)
    out["design"] = out["key"].map(dict(zip(h2h["agent"], h2h["design"])))
    out["design_name"] = out["design"].map(lambda d: DESIGNS[d].name)
    return out


def paired(df: pd.DataFrame, label: str, a: str, b: str, col: str, pr: Prereg):
    w = df.pivot_table(index=["market", "env"], columns=label, values=col).reset_index()
    return cluster_ci((w[a] - w[b]).to_numpy(), w["env"].to_numpy(), pr.n_boot, 1 - pr.alpha)


def pairwise(h2h: pd.DataFrame, focal: str, pr: Prereg) -> pd.DataFrame:
    """Paired profit difference focal - rival per market, Holm-adjusted across rivals."""
    rows = []
    for key in KEYS:
        if key == focal:
            continue
        m, lo, hi, p = paired(h2h, "agent", focal, key, "profit", pr)
        rows.append(dict(rival=THEORY_NAMES[key], key=key, diff=m, lo=lo, hi=hi, p=p))
    df = pd.DataFrame(rows)
    df["holm_reject"] = holm(df["p"].to_list(), pr.alpha)
    return df


def selection_effects(fac: pd.DataFrame, pr: Prereg) -> pd.DataFrame:
    """For each target: every selection rule against 'always', on profit and on downside risk (CVaR 5%)."""
    rows = []
    for target, always in (("model", "opt_br"), ("price", "cobweb_p")):
        for sel in ("band", "rc", "aspiration"):
            d = next(k for k in FACTORIAL if DESIGNS[k].TARGET == target and DESIGNS[k].SELECT == sel)
            for col in ("profit", "cvar5"):
                m, lo, hi, p = paired(fac, "design", d, always, col, pr)
                rows.append(dict(target=target, select=sel, design=DESIGNS[d].name, criterion=col, diff=m, lo=lo,
                                 hi=hi, p=p))
    return pd.DataFrame(rows)


def difficulty_index(df: pd.DataFrame) -> np.ndarray:
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    return ((z(df["delta"]) + z(df["noise"]) + z(df["hazard"])) / 3).to_numpy()


FEATURES = ["delta", "c_max", "q_range", "noise", "foresight", "hazard", "belief_lag"]


def robustness(h2h: pd.DataFrame) -> pd.DataFrame:
    """Standardized regression coefficients of each agent's relative profit on the environment parameters (a global
    sensitivity analysis over the sampled environment space)."""
    rows = []
    for key in KEYS:
        d = h2h[h2h["agent"] == key]
        y = d["rel_profit"].to_numpy(float)
        X = [((d[f] - d[f].mean()) / (d[f].std() + 1e-12)).to_numpy(float) for f in FEATURES]
        tab, r2 = ols((y - y.mean()) / (y.std() + 1e-12), X, FEATURES)
        rows.append(dict(agent=THEORY_NAMES[key], key=key, r2=r2,
                         **{f: float(tab.loc[tab["term"] == f, "coef"].iloc[0]) for f in FEATURES}))
    return pd.DataFrame(rows)


def invasion_matrix(inv: pd.DataFrame, pr: Prereg) -> pd.DataFrame:
    rows = []
    for (res, mut), d in inv.groupby(["resident", "mutant"]):
        m, lo, hi, p = cluster_ci(d["rel_fitness"].to_numpy(), d["env"].to_numpy(), pr.n_boot, 1 - pr.alpha)
        rows.append(dict(resident=res, mutant=mut, rel_fitness=m, lo=lo, hi=hi, p=p,
                         invades_share=(d["fitness"] > 0).mean()))
    df = pd.DataFrame(rows)
    df["holm_reject"] = False
    for res, g in df.groupby("resident"):
        df.loc[g.index, "holm_reject"] = holm(g["p"].to_list(), pr.alpha)
    df["verdict"] = np.where(df["holm_reject"] & (df["rel_fitness"] > 0), "invades",
                             np.where(df["holm_reject"] & (df["rel_fitness"] < 0), "repelled", "neutral"))
    return df


def evaluate(pr: Prereg, h2h: pd.DataFrame, inv: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-registered decision rules."""
    pw = pairwise(h2h, "heiner", pr)
    won = pw[pw["holm_reject"] & (pw["diff"] > 0)]["rival"].tolist()
    lost = pw[pw["holm_reject"] & (pw["diff"] < 0)]["rival"].tolist()
    pr1 = ("supported" if len(won) == len(pw) else "not supported",
           f"Better than {len(won)} of {len(pw)} rivals" + (f"; worse than {', '.join(lost)}" if lost else ""))
    w = h2h.pivot_table(index=["market", "env"], columns="agent", values="profit").reset_index()
    envs = h2h.drop_duplicates("market").set_index("market").loc[w["market"]]
    diff = (w["heiner"] - w["optimiser"]).to_numpy(float)
    tab, _ = ols(diff, [difficulty_index(envs.reset_index())], ["difficulty"])
    slope, p2 = float(tab.loc[1, "coef"]), float(tab.loc[1, "p"])
    pr2 = ("supported" if slope > 0 and p2 < pr.alpha else "not supported",
           f"slope {slope:+.3g} per SD of difficulty (p = {p2:.3g})")
    im = invasion_matrix(inv, pr)
    res_h = im[im["resident"] == "heiner"]
    invaders = [THEORY_NAMES[k] for k in res_h[res_h["verdict"] == "invades"]["mutant"]]
    pr3 = ("supported" if not invaders else "not supported",
           "No rival invades" if not invaders else "Invaded by " + ", ".join(invaders))
    mut_h = im[im["mutant"] == "heiner"]
    beaten = [THEORY_NAMES[k] for k in mut_h[mut_h["verdict"] == "invades"]["resident"]]
    pr4 = ("supported" if len(beaten) == len(mut_h) else "not supported",
           f"Invades {len(beaten)} of {len(mut_h)} rival populations")
    se = selection_effects(fac, pr)
    out = []
    for col, label in (("profit", "profit"), ("cvar5", "CVaR 5%")):
        d = se[(se["select"] == "rc") & (se["criterion"] == col)].reset_index(drop=True)
        rej = holm(d["p"].to_list(), pr.alpha)
        ok = [r and x > 0 for r, x in zip(rej, d["diff"])]
        out.append(("supported" if all(ok) else "not supported",
                    "; ".join(f"{t} target {x:+.0f} [{lo:.0f}, {hi:.0f}]{' *' if r else ''}"
                              for t, x, lo, hi, r in zip(d["target"], d["diff"], d["lo"], d["hi"], rej))
                    + f" ({label}, reliability condition − always; * = Holm-significant)"))
    return pd.DataFrame([dict(id=h[0], hypothesis=h[1], decision_rule=h[2], verdict=v[0], result=v[1])
                         for h, v in zip(pr.hypotheses, (pr1, pr2, pr3, pr4, *out))])


@dataclass
class TournamentResult:
    prereg_hash: str
    exploratory: bool
    tuned: Tuned
    tuning_log: pd.DataFrame
    h2h: pd.DataFrame
    fac: pd.DataFrame
    inv: pd.DataFrame
    ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairwise: pd.DataFrame = field(default_factory=pd.DataFrame)
    factorial: pd.DataFrame = field(default_factory=pd.DataFrame)
    selection: pd.DataFrame = field(default_factory=pd.DataFrame)
    invasion: pd.DataFrame = field(default_factory=pd.DataFrame)
    robustness: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdicts: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_protocol(pr: Prereg, progress: Optional[Callable[[float, str], None]] = None) -> TournamentResult:
    stage = lambda lo, hi: (lambda f, msg: progress(lo + (hi - lo) * f, msg)) if progress else None
    tuned, log = tune(pr, stage(0.0, 0.7))
    if progress:
        progress(0.7, "Head-to-head and selection-rule markets on held-out environments")
    h2h = head_to_head(pr, tuned)
    fac = factorial(pr, tuned)
    inv = invasion(pr, tuned, stage(0.75, 1.0))
    return TournamentResult(prereg_hash=pr.digest, exploratory=pr.digest != PREREG.digest, tuned=tuned,
                            tuning_log=log, h2h=h2h, fac=fac, inv=inv, ranking=ranking(h2h, pr),
                            pairwise=pairwise(h2h, "heiner", pr), factorial=criteria(fac, "design", pr),
                            selection=selection_effects(fac, pr), invasion=invasion_matrix(inv, pr),
                            robustness=robustness(h2h), verdicts=evaluate(pr, h2h, inv, fac))


def replicate(pr: Prereg, n: int, progress: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
    """Re-run the whole protocol (tuning and design selection included) with n independent pairs of training and test
    seeds, to show which conclusions survive a new draw of environments. One row per (replication, agent); the
    selection-rule effects of every replication are in .attrs["selection"]."""
    rows, sel = [], []
    for r in range(n):
        p = Prereg(**{**asdict(pr), "train_seed": pr.train_seed + 100 * (r + 1), "test_seed": pr.test_seed + 100 * (r + 1)})
        if progress:
            progress(r / n, f"Replication {r + 1} of {n}")
        res = run_protocol(p)
        sel.append(res.selection.assign(replication=r + 1))
        ver = "".join("✓" if v == "supported" else "✗" for v in res.verdicts["verdict"])
        for _, x in res.ranking.iterrows():
            rows.append(dict(replication=r + 1, train_seed=p.train_seed, agent=x["agent"], key=x["key"],
                             design=x["design_name"], mean_rank=x["mean_rank"], aggregate_rank=x["aggregate_rank"],
                             rel_profit=x["rel_profit"], cvar5=x["cvar5"], survived=x["survived"], verdicts=ver))
    if progress:
        progress(1.0, "Replications done")
    out = pd.DataFrame(rows)
    out.attrs["selection"] = pd.concat(sel, ignore_index=True)    # selection-rule effects per replication
    return out


def prereg_from_dict(d: dict) -> Prereg:
    d = {k: v for k, v in d.items() if k in Prereg.__dataclass_fields__}
    d["env_ranges"] = tuple((k, tuple(v)) for k, v in d["env_ranges"])
    d["hypotheses"] = tuple(tuple(h) for h in d["hypotheses"])
    return Prereg(**d)


PREREG = Prereg()
QUICK = Prereg(n_train=6, n_test=8, reps_test=1, periods=250, burn_in=30, budget=6, rounds=1, n_boot=200)
