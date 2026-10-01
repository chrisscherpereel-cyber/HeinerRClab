"""Agent tournament: rival theories of flexibility compete as agents in the same market.

Each theory is implemented as a decision rule (an agent type). All agents face the same cobweb oligopoly as the main
model: the same demand curve and unannounced demand-regime shifts, the same reflecting raw-material cost process,
the same information (last price, last market quantity, own output, a cost estimate of the same quality) and the same
random streams. They differ only in how they turn that information into output.

Fairness protocol (see PREREG):
    1. every agent type has its free parameters tuned with the same budget of evaluations (Latin-hypercube search),
       on training environments only, in two rounds (round 2 tunes against the round-1 tuned rivals);
    2. environments are drawn at random from pre-registered ranges; training and test environments are disjoint;
    3. the plan (ranges, budgets, metrics, hypotheses, decision rules) is frozen and hashed before running;
    4. results use several yardsticks: head-to-head profit in mixed markets, invasion tests (can one mutant of type X
       out-earn residents of type Y?), and robustness across environments (standardised regression coefficients).

Everything is vectorised over markets: a batch holds B markets, and every agent parameter is an array over markets,
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
    """Draw environments uniformly from the pre-registered ranges. Half of them have demand-regime shifts."""
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
        # per-period information (filled by simulate)
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
    """One agent type on one slot of a subset `idx` of the batch's markets. Parameters are arrays over idx."""
    key = ""
    name = ""
    theory = ""
    rule = ""
    sources: Tuple[str, ...] = ()
    SPACE: Dict[str, Spec] = {}

    def __init__(self, idx: np.ndarray, slot: int, p: Dict[str, np.ndarray], mk: Market):
        self.idx, self.i, self.p, self.mk = idx, slot, p, mk
        self.n = len(idx)

    # helpers ---------------------------------------------------------------------------------
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


class RuleB(Agent):
    key, name, theory = "ruleb", "Rule B (rigid)", "Benchmark: rule-governed default"
    rule = "Always keeps last period's output."
    sources = ("heiner1983",)

    def act(self):
        return self.q


class Optimiser(Agent):
    key, name, theory = "optimiser", "Filtered best reply", "Neoclassical optimisation"
    rule = ("Always plays the best reply to filtered forecasts: exponential (Kalman-type) filters on its cost estimate "
            "and on rivals' output.")
    sources = ("muth1960", "kalman1960", "muth1961")
    SPACE = {"a_cost": Spec(1.0, 0.05, 1.0, "log", "Filter gain on the cost estimate"),
             "a_rival": Spec(1.0, 0.05, 1.0, "log", "Filter gain on rivals' output")}

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.cf = self.rf = None

    def target(self, a_c, a_r):
        if self.cf is None:
            self.cf, self.rf = self.c_hat.copy(), self.rivals.copy()
        else:
            self.cf = self.cf + a_c * (self.c_hat - self.cf)
            self.rf = self.rf + a_r * (self.rivals - self.rf)
        return self.mk.br(self.idx, self.cf, self.rf)

    def act(self):
        return self.target(self.p["a_cost"], self.p["a_rival"])


class RealOptions(Optimiser):
    key, name, theory = "options", "Inaction band", "Real options / value of flexibility"
    rule = ("Same filtered target as the optimiser, but moves to it only when the gap exceeds a band that widens with "
            "the measured volatility of the target (k standard deviations).")
    sources = ("dixit1989", "dixit1994")
    SPACE = {"a_cost": Spec(1.0, 0.05, 1.0, "log", "Filter gain on the cost estimate"),
             "a_rival": Spec(1.0, 0.05, 1.0, "log", "Filter gain on rivals' output"),
             "k": Spec(1.0, 0.0, 3.0, help="Band width in standard deviations of target changes")}

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.v = np.zeros(self.n)
        self.last = None

    def act(self):
        tgt = self.target(self.p["a_cost"], self.p["a_rival"])
        if self.last is not None:
            self.v = 0.9 * self.v + 0.1 * (tgt - self.last) ** 2
        self.last = tgt
        q = self.q
        return np.where(np.abs(tgt - q) > self.p["k"] * np.sqrt(self.v), tgt, q)


class Heiner(Optimiser):
    key, name, theory = "heiner", "Reliability-condition agent", "Heiner: reliability condition"
    rule = ("Same filtered target as the optimiser, but rule B (keep output) is the default. The agent deviates to "
            "the target only when its learned reliability condition holds for changes of that size: "
            "π·G − (1−π)·D ≥ 0, judged ex post over an h-period horizon with exponential memory.")
    sources = ("heiner1983", "heiner1989")
    SPACE = {"a_cost": Spec(1.0, 0.05, 1.0, "log", "Filter gain on the cost estimate"),
             "a_rival": Spec(1.0, 0.05, 1.0, "log", "Filter gain on rivals' output"),
             "theta": Spec(10.0, 1.0, 60.0, "log", "Scale of the size-of-change bins"),
             "memory": Spec(0.97, 0.8, 0.995, help="Exponential memory of the learned gains"),
             "horizon": Spec(10, 1, 20, "int", "Periods over which a deviation is judged")}

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.E = np.zeros((self.n, N_BINS))
        self.qs = np.zeros((mk.T, self.n))
        self.qb = np.zeros((mk.T, self.n))
        self.bin = np.zeros((mk.T, self.n), dtype=int)
        self.opp = np.zeros((mk.T, self.n), dtype=bool)
        self.h = self.p["horizon"].astype(int)
        self.edges = self.p["theta"][:, None] * np.array([0.5, 1.0, 2.0, 4.0])[None, :]

    def act(self):
        t, q = self.mk.t, self.q
        cand = np.maximum(Q_MIN, np.rint(self.target(self.p["a_cost"], self.p["a_rival"])))
        sig = np.abs(cand - q)
        b = (sig[:, None] > self.edges).sum(1)
        self.qs[t], self.qb[t], self.bin[t], self.opp[t] = cand, q, b, sig > 0
        dev = (sig > 0) & (self.E[np.arange(self.n), b] >= 0)
        return np.where(dev, cand, q)

    def update(self):
        """Judge the decision made h periods ago: hold the candidate vs hold the old level (rule B) for h periods,
        rivals on their actual path, prices from the believed demand curve, realised costs."""
        mk, t = self.mk, self.mk.t
        for h in np.unique(self.h):
            d = t - h + 1
            if d < 1:
                continue
            rows = np.flatnonzero((self.h == h) & self.opp[d])
            if not len(rows):
                continue
            mi = self.idx[rows]
            R = mk.R_hist[mi, d:t + 1, self.i]
            pm, sl, c = mk.PMB[mi, d:t + 1], mk.SLB[mi, d:t + 1], mk.cost[mi, d:t + 1]
            x1, x0 = self.qs[d, rows][:, None], self.qb[d, rows][:, None]
            gain = (((np.maximum(P_MIN, pm - sl * (R + x1)) - c) * x1)
                    - ((np.maximum(P_MIN, pm - sl * (R + x0)) - c) * x0)).sum(1)
            b = self.bin[d, rows]
            lam = self.p["memory"][rows]
            self.E[rows, b] = lam * self.E[rows, b] + (1 - lam) * gain


class Cobweb(Agent):
    key, name, theory = "cobweb", "Adaptive expectations", "Cobweb stability theory"
    rule = ("Forms an adaptive price expectation (gain λ) and moves a share φ toward the output that is optimal at "
            "that expected price (partial adjustment).")
    sources = ("nerlove1958", "ezekiel1938")
    SPACE = {"lam": Spec(0.5, 0.05, 1.0, "log", "Adaptive-expectations gain"),
             "phi": Spec(0.5, 0.05, 1.0, help="Partial-adjustment speed")}

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.pe = None

    def act(self):
        P = self.mk.P_prev[self.idx]
        self.pe = P.copy() if self.pe is None else self.pe + self.p["lam"] * (P - self.pe)
        s, q = self.mk.SLB[self.idx, self.mk.t], self.q
        tgt = (self.pe + s * q - self.c_hat) / (2 * s)
        return q + self.p["phi"] * (tgt - q)


class Heuristic(Agent):
    key, name, theory = "heuristic", "Win-stay, lose-shift", "Bias–variance / ecological rationality"
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


class Satisficer(Agent):
    key, name, theory = "satisficing", "Aspiration-level search", "Satisficing / aspiration-level search"
    rule = ("Keeps output while profit meets an adaptive aspiration level; below it, searches by moving a share φ "
            "toward the best reply.")
    sources = ("simon1955", "cyert1963")
    SPACE = {"alpha": Spec(0.1, 0.01, 0.5, "log", "Speed at which aspirations adapt"),
             "phi": Spec(0.5, 0.05, 1.0, help="Search step toward the best reply")}

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.A = None

    def act(self):
        pr, q = self.profit, self.q
        if self.A is None:
            self.A = pr.copy()
        tgt = self.mk.br(self.idx, self.c_hat, self.rivals)
        out = np.where(pr >= self.A, q, q + self.p["phi"] * (tgt - q))
        self.A = self.A + self.p["alpha"] * (pr - self.A)
        return out


class Reinforcement(Agent):
    key, name, theory = "rl", "Reinforcement learner", "Reinforcement learning"
    rule = ("Chooses among five moves (−2, −1, 0, +1, +2 steps) by softmax over learned values; a move's value is the "
            "profit change that followed it.")
    sources = ("erev1998", "sutton2018")
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

    def act(self):
        pr, rows = self.profit, np.arange(self.n)
        if self.a is not None:
            r = pr - self.last
            self.scale = 0.95 * self.scale + 0.05 * np.abs(r)
            eta = self.p["eta"]
            self.V[rows, self.a] += eta * (r - self.V[rows, self.a])
        z = self.V / (self.p["temp"] * self.scale + 1e-9)[:, None]
        w = np.exp(z - z.max(1, keepdims=True))
        cdf = np.cumsum(w / w.sum(1, keepdims=True), axis=1)
        u = self.mk.UR[self.idx, self.mk.t, self.i]
        self.a = np.minimum((cdf < u[:, None]).sum(1), 4)
        self.last = pr.copy()
        return self.q + self.MOVES[self.a] * self.p["step"]


class Imitator(Agent):
    key, name, theory = "imitation", "Imitate the best", "Structural inertia / evolutionary selection"
    rule = ("With probability p copies last period's output of the most profitable firm in the market, plus small "
            "random experimentation.")
    sources = ("vegaredondo1997", "alchian1950")
    SPACE = {"p": Spec(0.5, 0.05, 1.0, help="Probability of imitating each period"),
             "noise": Spec(2.0, 0.0, 20.0, help="Standard deviation of experimentation")}

    def act(self):
        mk, idx = self.mk, self.idx
        best = mk.profit_prev[idx].argmax(1)
        q_best = mk.q_prev[idx, best]
        u, z = mk.UR[idx, mk.t, self.i], mk.ZR[idx, mk.t, self.i]
        return np.where(u < self.p["p"], q_best, self.q) + self.p["noise"] * z


AGENTS: Dict[str, type] = {a.key: a for a in
                           (Heiner, Optimiser, RealOptions, Cobweb, Heuristic, Satisficer, Reinforcement, Imitator,
                            RuleB)}
KEYS = list(AGENTS)
TUNED = [k for k in KEYS if AGENTS[k].SPACE]


def default_params(key: str) -> Dict[str, float]:
    return {n: s.default for n, s in AGENTS[key].SPACE.items()}


# ================================================================================================ simulation
def simulate(envs: Sequence[Env], lineup: np.ndarray, params: Dict[str, Dict[str, np.ndarray]], periods: int,
             burn_in: int) -> Dict[str, np.ndarray]:
    """Run B markets. lineup (B, N) holds agent keys per market and slot; params[key][name] is an array over the B
    markets (or a scalar). Returns average profit per period after burn-in (B, N) and the share of periods in which
    each firm changed its output."""
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
    agents: List[Agent] = []
    for i in range(N):
        for key in np.unique(lineup[:, i]):
            idx = np.flatnonzero(lineup[:, i] == key)
            p = {n: np.broadcast_to(np.asarray(params.get(key, {}).get(n, s.default), float), (B,))[idx].copy()
                 for n, s in AGENTS[key].SPACE.items()}
            agents.append(AGENTS[key](idx, i, p, mk))
    sum_profit = np.zeros((B, N))
    changes = np.zeros((B, N))
    for t in range(1, periods):
        mk.t = t
        c_prev, c_now = mk.cost[:, t - 1], mk.cost[:, t]
        mk.c_hat = (c_prev + mk.kappa * (c_now - c_prev))[:, None] + mk.sigma[:, None] * mk.eps[:, t, :]
        new = np.empty((B, N))
        for a in agents:
            new[a.idx, a.i] = a.act()
        new = np.maximum(Q_MIN, np.rint(np.nan_to_num(new, nan=Q_MIN)))
        Q = new.sum(1)
        P = np.maximum(P_MIN, mk.PM[:, t] - mk.SL[:, t] * Q)
        profit = (P - c_now)[:, None] * new
        mk.R_hist[:, t] = Q[:, None] - new
        if t >= burn_in:
            sum_profit += profit
            changes += new != mk.q_prev
        mk.P_prev, mk.Q_prev, mk.q_prev, mk.profit_prev = P, Q, new, profit
        for a in agents:
            a.update()
    n_rec = periods - burn_in
    return dict(profit=sum_profit / n_rec, change_rate=changes / n_rec)


# ================================================================================================ pre-registration
ENV_RANGES = {"delta": (2.0, 30.0), "c_max": (60.0, 95.0), "q_range": (800.0, 3000.0), "noise": (0.0, 10.0),
              "foresight": (0.0, 0.5), "hazard": (0.005, 0.05), "belief_lag": (0.0, 40.0),
              "share_with_shifts": (0.5, 0.5)}

HYPOTHESES = (
    ("PR1", "Head-to-head: in held-out mixed markets the reliability-condition agent earns more than each rival.",
     "Paired profit difference per market; 95% cluster-bootstrap CI over environments, Holm-adjusted across rivals. "
     "Supported against a rival if the adjusted CI lies above 0."),
    ("PR2", "Conditional advantage: the reliability-condition agent's advantage over the filtered best reply grows "
            "with environmental difficulty.",
     "OLS of the paired difference on a difficulty index (mean of standardised Δ, σ and hazard). Supported if the "
     "slope is positive with p < α."),
    ("PR3", "Evolutionary stability: a population of reliability-condition agents cannot be invaded by any rival.",
     "For each rival mutant, the mutant's profit minus the residents' mean profit. Supported if no mutant's 95% CI "
     "lies above 0 (Holm-adjusted)."),
    ("PR4", "Invasion: the reliability-condition agent can invade every rival population.",
     "Mutant reliability-condition agent's profit minus the residents' mean profit. Supported against a rival if the "
     "Holm-adjusted 95% CI lies above 0."),
)


@dataclass(frozen=True)
class Prereg:
    """The frozen analysis plan. Its hash identifies the plan; any change makes a run exploratory."""
    version: str = "1.0"
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
    src = "".join(inspect.getsource(o) for o in (*AGENTS.values(), Agent, Market, simulate, tune, head_to_head,
                                                  invasion, evaluate, sample_envs))
    return hashlib.sha256(src.encode()).hexdigest()


PREREG = Prereg()
QUICK = Prereg(n_train=6, n_test=8, reps_test=1, periods=250, burn_in=30, budget=6, rounds=1, n_boot=200)


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


def tune(pr: Prereg, progress: Optional[Callable[[float, str], None]] = None
         ) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """Equal-budget tuning. Every agent type gets `budget` candidate parameter vectors (default included), each
    evaluated on the same training environments in a mixed market with one firm of every type. Round r tunes each
    type against the others' parameters from round r - 1 (round 1: defaults). Returns tuned parameters and a log."""
    envs = train_envs(pr)
    E, K = len(envs), pr.budget
    current = {k: default_params(k) for k in KEYS}
    log = []
    steps, done = pr.rounds * len(TUNED), 0
    for rnd in range(1, pr.rounds + 1):
        new = {k: dict(v) for k, v in current.items()}
        for key in TUNED:
            if progress:
                progress(done / steps, f"Round {rnd}: tuning {AGENTS[key].name}")
            rng = np.random.default_rng([pr.train_seed, rnd, KEYS.index(key)])
            cand = _lhs(AGENTS[key].SPACE, K, rng)
            lineup = np.tile(np.array(KEYS, dtype=object), (K * E, 1))
            params = {k: dict(v) for k, v in current.items()}
            params[key] = {n: np.repeat(v, E) for n, v in cand.items()}
            out = simulate(envs * K, lineup, params, pr.periods, pr.burn_in)
            score = out["profit"][:, KEYS.index(key)].reshape(K, E).mean(1)
            best = int(np.argmax(score))
            new[key] = {n: float(v[best]) for n, v in cand.items()}
            log.append(dict(round=rnd, agent=AGENTS[key].name, key=key, default_score=float(score[0]),
                            best_score=float(score[best]), gain=float(score[best] - score[0]),
                            **{f"param:{n}": float(v[best]) for n, v in cand.items()}))
            done += 1
        current = new
    if progress:
        progress(1.0, "Tuning done")
    return current, pd.DataFrame(log)


# ================================================================================================ yardsticks
def head_to_head(pr: Prereg, tuned: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """One firm of every type per market, held-out environments. One row per (market, agent)."""
    envs = test_envs(pr)
    lineup = np.tile(np.array(KEYS, dtype=object), (len(envs), 1))
    out = simulate(envs, lineup, tuned, pr.periods, pr.burn_in)
    rows = []
    for b, e in enumerate(envs):
        prof = out["profit"][b]
        rank = (-prof).argsort().argsort() + 1
        for i, key in enumerate(KEYS):
            rows.append(dict(market=b, env=e.seed % 50_000, agent=key, profit=prof[i], rank=int(rank[i]),
                             rel_profit=prof[i] - prof.mean(), change_rate=out["change_rate"][b, i], **asdict(e)))
    return pd.DataFrame(rows)


def invasion(pr: Prereg, tuned: Dict[str, Dict[str, float]],
             progress: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
    """For every resident type Y and mutant type X != Y: markets with `residents` firms of type Y and one of type X.
    Fitness = mutant profit - mean resident profit. One row per (market, resident, mutant)."""
    envs = test_envs(pr)
    rows = []
    for r_i, res in enumerate(KEYS):
        if progress:
            progress(r_i / len(KEYS), f"Invasion of {AGENTS[res].name} populations")
        muts = [k for k in KEYS if k != res]
        lineup = np.array([[res] * pr.residents + [m] for m in muts for _ in envs], dtype=object)
        out = simulate(envs * len(muts), lineup, tuned, pr.periods, pr.burn_in)
        prof = out["profit"]
        fit = prof[:, -1] - prof[:, :-1].mean(1)
        scale = np.abs(prof[:, :-1].mean(1)) + 1e-9
        for j, (m, e) in enumerate([(m, e) for m in muts for e in envs]):
            rows.append(dict(resident=res, mutant=m, env=e.seed % 50_000, fitness=fit[j], rel_fitness=fit[j] / scale[j]))
    if progress:
        progress(1.0, "Invasion done")
    return pd.DataFrame(rows)


def cluster_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int, conf: float, seed: int = 0):
    """Mean with a cluster-bootstrap CI (resampling environments)."""
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


def ranking(h2h: pd.DataFrame, pr: Prereg) -> pd.DataFrame:
    rows = []
    for key in KEYS:
        d = h2h[h2h["agent"] == key]
        m, lo, hi, _ = cluster_ci(d["rel_profit"].to_numpy(), d["env"].to_numpy(), pr.n_boot, 1 - pr.alpha)
        rows.append(dict(agent=AGENTS[key].name, key=key, theory=AGENTS[key].theory, mean_rank=d["rank"].mean(),
                         win_share=(d["rank"] == 1).mean(), rel_profit=m, lo=lo, hi=hi, profit=d["profit"].mean(),
                         change_rate=d["change_rate"].mean()))
    return pd.DataFrame(rows).sort_values("mean_rank", ignore_index=True)


def pairwise(h2h: pd.DataFrame, focal: str, pr: Prereg) -> pd.DataFrame:
    """Paired profit difference focal - rival per market, Holm-adjusted across rivals."""
    w = h2h.pivot_table(index=["market", "env"], columns="agent", values="profit").reset_index()
    rows = []
    for key in KEYS:
        if key == focal:
            continue
        diff = (w[focal] - w[key]).to_numpy()
        m, lo, hi, p = cluster_ci(diff, w["env"].to_numpy(), pr.n_boot, 1 - pr.alpha)
        rows.append(dict(rival=AGENTS[key].name, key=key, diff=m, lo=lo, hi=hi, p=p))
    df = pd.DataFrame(rows)
    df["holm_reject"] = holm(df["p"].to_list(), pr.alpha)
    return df


def difficulty_index(df: pd.DataFrame) -> np.ndarray:
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    return ((z(df["delta"]) + z(df["noise"]) + z(df["hazard"])) / 3).to_numpy()


def robustness(h2h: pd.DataFrame) -> pd.DataFrame:
    """Standardised regression coefficients of each agent's relative profit on the environment parameters (a global
    sensitivity analysis over the sampled environment space)."""
    feats = ["delta", "c_max", "q_range", "noise", "foresight", "hazard", "belief_lag"]
    rows = []
    for key in KEYS:
        d = h2h[h2h["agent"] == key]
        y = d["rel_profit"].to_numpy(float)
        X = [((d[f] - d[f].mean()) / (d[f].std() + 1e-12)).to_numpy(float) for f in feats]
        tab, r2 = ols((y - y.mean()) / (y.std() + 1e-12), X, feats)
        rows.append(dict(agent=AGENTS[key].name, key=key, r2=r2,
                         **{f: float(tab.loc[tab["term"] == f, "coef"].iloc[0]) for f in feats}))
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


def evaluate(pr: Prereg, h2h: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
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
    invaders = [AGENTS[k].name for k in res_h[res_h["verdict"] == "invades"]["mutant"]]
    pr3 = ("supported" if not invaders else "not supported",
           "No rival invades" if not invaders else "Invaded by " + ", ".join(invaders))
    mut_h = im[im["mutant"] == "heiner"]
    beaten = [AGENTS[k].name for k in mut_h[mut_h["verdict"] == "invades"]["resident"]]
    pr4 = ("supported" if len(beaten) == len(mut_h) else "not supported",
           f"Invades {len(beaten)} of {len(mut_h)} rival populations")
    return pd.DataFrame([dict(id=h[0], hypothesis=h[1], decision_rule=h[2], verdict=v[0], result=v[1])
                         for h, v in zip(pr.hypotheses, (pr1, pr2, pr3, pr4))])


@dataclass
class TournamentResult:
    prereg_hash: str
    exploratory: bool
    tuned: Dict[str, Dict[str, float]]
    tuning_log: pd.DataFrame
    h2h: pd.DataFrame
    inv: pd.DataFrame
    ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairwise: pd.DataFrame = field(default_factory=pd.DataFrame)
    invasion: pd.DataFrame = field(default_factory=pd.DataFrame)
    robustness: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdicts: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_protocol(pr: Prereg, progress: Optional[Callable[[float, str], None]] = None) -> TournamentResult:
    stage = lambda lo, hi: (lambda f, msg: progress(lo + (hi - lo) * f, msg)) if progress else None
    tuned, log = tune(pr, stage(0.0, 0.6))
    if progress:
        progress(0.6, "Head-to-head on held-out environments")
    h2h = head_to_head(pr, tuned)
    inv = invasion(pr, tuned, stage(0.65, 1.0))
    return TournamentResult(prereg_hash=pr.digest, exploratory=pr.digest != PREREG.digest, tuned=tuned,
                            tuning_log=log, h2h=h2h, inv=inv, ranking=ranking(h2h, pr),
                            pairwise=pairwise(h2h, "heiner", pr), invasion=invasion_matrix(inv, pr),
                            robustness=robustness(h2h), verdicts=evaluate(pr, h2h, inv))


def replicate(pr: Prereg, n: int, progress: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
    """Re-run the whole protocol (tuning included) with n independent pairs of training and test seeds, to show
    which conclusions survive a new draw of environments. One row per (replication, agent)."""
    rows = []
    for r in range(n):
        p = Prereg(**{**asdict(pr), "train_seed": pr.train_seed + 100 * (r + 1), "test_seed": pr.test_seed + 100 * (r + 1)})
        if progress:
            progress(r / n, f"Replication {r + 1} of {n}")
        res = run_protocol(p)
        ver = "".join("✓" if v == "supported" else "✗" for v in res.verdicts["verdict"])
        for _, x in res.ranking.iterrows():
            rows.append(dict(replication=r + 1, train_seed=p.train_seed, agent=x["agent"], key=x["key"],
                             mean_rank=x["mean_rank"], rel_profit=x["rel_profit"], verdicts=ver))
    if progress:
        progress(1.0, "Replications done")
    return pd.DataFrame(rows)


def prereg_from_dict(d: dict) -> Prereg:
    d = {k: v for k, v in d.items() if k in Prereg.__dataclass_fields__}
    d["env_ranges"] = tuple((k, tuple(v)) for k, v in d["env_ranges"])
    d["hypotheses"] = tuple(tuple(h) for h in d["hypotheses"])
    return Prereg(**d)
