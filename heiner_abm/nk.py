"""NK-landscape decision environment: interaction complexity, reliability learning and adaptation.

What the environment represents
    The NK model (Kauffman & Levin 1987; Kauffman 1993), as used in organization science (Levinthal 1997; Rivkin 2000),
    represents COMPLEXITY: how strongly the payoff of one decision depends on other decisions. It is not itself a model
    of uncertainty. This environment adds two separately controlled uncertainty mechanisms:
      * observation noise (`obs_noise`): every evaluation returns the true payoff plus N(0, obs_noise^2) error, so an
        apparent improvement can be false;
      * environmental change (`hazard`, `change_frac`): with probability `hazard` per period, the contribution tables of
        a share `change_frac` of the components are redrawn without announcement, so a configuration that was good can
        stop being good.
    Interdependence (K_NK), noise and change can therefore be varied one at a time. K_NK is named to avoid confusion
    with the market's error-to-signal ratio K.

Task interface (as in heiner_abm.tasks)
    A default (keep the current configuration), a flexible alternative (a candidate configuration proposed by the
    search rule), and every period a decision whether to switch. Environments are frozen dataclasses (NKEnv); the
    exogenous parts (landscape path, observation noise draws, proposal draws) are generated from seeds before any agent
    acts and are shared by every searcher (common random numbers).

Landscape (established model)
    N binary components x_1..x_N. Component i's contribution C_i depends on x_i and on the K_NK components it
    interacts with (0 <= K_NK <= N - 1); C_i is a table of 2^(K_NK + 1) values drawn i.i.d. Uniform(0, 1). Payoff
    F(x) = (1/N) * sum_i C_i(x_i, x_neighbors(i)). Interaction topology:
        adjacent  the K_NK next components in cyclic order (Kauffman's adjacent neighbourhood)
        random    K_NK distinct other components drawn at random per component (seeded)
        block     consecutive blocks of K_NK + 1 components interact fully; a block that runs past N wraps to the
                  start so that every component has exactly K_NK partners
        explicit  a tuple of N tuples: row i lists the K_NK partners of component i
    For N <= ENUM_MAX the landscape is enumerated: the global maximum and every local peak are EXACT. For larger N the
    benchmark is the BEST-KNOWN value (the best of multi-start full-information hill climbs and of every configuration
    any agent held), labelled as such; optimality is not claimed.

Timing and information
    In period t the agent operates its configuration x_t and observes its payoff once (free: it is operating it). It
    may then evaluate one candidate y (one unit of the evaluation budget) and observe F_t(y) + noise. It decides at the
    end of period t; a switch takes effect in period t + 1 and costs `cost` per changed component (charged in t + 1).
    Agents see only their own noisy observations, their own costs and, for imitation, the observable part of a
    leader's configuration. The true payoffs, the landscape tables, the global maximum and the change times are
    researcher-only quantities used for the outcome measures; agents receive an Observer that exposes nothing else.

Search rules (established ideas; parameters are this laboratory's choices)
    hill        Local hill climbing: propose one component flip; switch if the observed gain exceeds the switching cost.
    stochastic  Broader stochastic search: propose 1 + Binomial(N - 1, p_jump) flips; accept an apparent improvement,
                or a worse candidate with probability exp(net gain / temperature), the temperature falling linearly with
                the budget used (simulated annealing; Kirkpatrick et al. 1983).
    satisfice   Satisficing (Simon 1955; Cyert & March 1963): search (one flip) only while the observed payoff is below
                an aspiration level; accept a candidate whose observed payoff net of the switching cost meets the
                aspiration; the aspiration starts `aspiration_start` above the first observed payoff and adapts toward
                observed payoffs, so search stops once performance is satisfactory (even below a peak).
    imitate     Imitation with explicit observability (Rivkin 2000): a leader firm (researcher-generated: full-
                information steepest-ascent hill climbing from its own start, re-climbing after changes) is observed on
                a fixed set of round(observe * N) components only, as of the end of the previous period. The imitator proposes
                to copy one observable component on which it differs (one flip if none) and accepts apparent
                improvements. It never sees the leader's other components or payoffs.
    gate_none, gate_gain, gate_lcb
                Reliability-gated search (PROPOSED EXTENSION, the laboratory's operationalization). All three use the
                same proposals (the stochastic generator) and the same evaluation budget: every apparent improvement
                (observed gain above the switching cost) is evaluated a second time, independently (verification, one
                more budget unit), whatever is decided. The verified net gain is released after the decision and
                updates per-bin evidence (heiner_abm.gates statistics), the bin being the apparent gain's strength
                relative to the agent's own noise estimate. gate_none accepts every apparent improvement (it pays for
                verification but ignores it: the reference with identical proposals and budget); gate_gain accepts if
                the bin's learned mean verified gain is at least 0 (the existing estimated-gain gate, optimistic
                without evidence); gate_lcb accepts only with at least `min_evidence` effective observations and a
                lower confidence bound above 0 (the confidence-sensitive gate). The evidence estimates the reliability
                of acting on an apparent improvement of a given strength (selected apparent gains overstate true gains).

Outcomes (researcher-only, per run)
    payoff               mean per period of F_t(x_t) minus switching costs
    regret               mean per period of max F_t - F_t(x_t) plus switching costs (exact or best-known maximum)
    evaluations          budget units used
    false_improvements   switches whose true net gain was <= 0 (count; share of switches in false_share)
    missed_improvements  evaluated candidates with true net gain > 0 that were not adopted (count; share in
                         missed_share)
    escapes              switches from a true local peak that is not the global (best-known) maximum to a configuration
                         with higher true payoff; stuck_share = share of periods at such a peak
    recovery_delay       per landscape change: periods until the normalized payoff F/max F regains its level just
                         before the change (censored at the next change or the horizon; censored delays enter at their
                         censoring length, so the mean is a lower bound); recovered_share = share not censored

Uncertainty
    Runs on the same landscape share its contribution tables and change path, so landscapes are the unit of
    resampling: intervals are cluster bootstraps over landscapes (starting configurations nested within them).
    The pilot settings (N = 8, K_NK in {0, 2, 4, 7}) are configurable starting points, not optimal design values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import gates
from .arena import cluster_ci

ENUM_MAX = 16                      # enumerate every configuration (exact benchmark) up to this N
EDGES = np.array([0.5, 1.0, 2.0, 4.0])
N_BINS = len(EDGES) + 1
TOPOLOGIES = ("adjacent", "random", "block")
SEARCHERS = ("hill", "stochastic", "satisfice", "imitate", "gate_none", "gate_gain", "gate_lcb")
GATED = ("gate_none", "gate_gain", "gate_lcb")
LABELS = {"hill": "Local hill climbing", "stochastic": "Broader stochastic search", "satisfice": "Satisficing",
          "imitate": "Imitation (partial observability)", "gate_none": "Ungated (same proposals and budget)",
          "gate_gain": "Reliability-gated: estimated gain", "gate_lcb": "Reliability-gated: confidence-sensitive"}
METRICS = {"payoff": "Attained payoff per period", "regret": "Regret per period", "evaluations": "Evaluations used",
           "false_improvements": "False improvements (count)", "false_share": "False improvements (share of switches)",
           "missed_improvements": "Missed improvements (count)",
           "missed_share": "Missed improvements (share of true improvements evaluated)",
           "escapes": "Escapes from local peaks (count)", "stuck_share": "Share of periods at a non-global local peak",
           "recovery_delay": "Recovery delay after change (periods, lower bound)",
           "recovered_share": "Share of changes recovered from"}
Topology = Union[str, Tuple[Tuple[int, ...], ...]]


@dataclass(frozen=True)
class NKEnv:
    task: str = "nk"
    N: int = 8
    K_NK: int = 2
    topology: Topology = "adjacent"
    obs_noise: float = 0.02            # observation noise s.d. (payoffs lie in [0, 1])
    hazard: float = 0.0                # probability per period of an unannounced landscape change
    change_frac: float = 0.25          # share of components whose contribution tables are redrawn at a change
    cost: float = 0.002                # switching cost per changed component
    budget: int = 200                  # evaluation budget (candidate evaluations, verifications included)
    periods: int = 300
    observe: float = 0.5               # imitation: share of the leader's components that are observable
    seed: int = 0                      # landscape seed (tables, topology, change path)


@dataclass(frozen=True)
class SearchParams:
    p_jump: float = 0.25               # stochastic proposals: extra flips ~ Binomial(N - 1, p_jump)
    temp0: float = 0.01                # stochastic search: initial temperature
    aspiration_rate: float = 0.1       # satisficing: aspiration adaptation
    aspiration_start: float = 0.05     # satisficing: initial aspiration above the first observed payoff
    memory: float = 0.97               # gates: forgetting per update
    confidence: float = 0.9            # gate_lcb: one-sided confidence
    min_evidence: float = 5.0          # gate_lcb: minimum effective observations
    noise_rate: float = 0.05           # agents' running estimate of their observation noise
    noise_prior: float = 0.01          # its starting value (s.d.)


# ================================================================================================ landscape
def interactions(N: int, K_NK: int, topology: Topology, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """(N, K_NK + 1) array: row i = (i, its K_NK partners)."""
    if not 0 <= K_NK <= N - 1:
        raise ValueError(f"K_NK must lie in [0, N - 1] = [0, {N - 1}], got {K_NK}")
    if isinstance(topology, (tuple, list)):
        rows = [tuple(int(j) for j in r) for r in topology]
        if len(rows) != N or any(len(r) != K_NK or len(set(r)) != K_NK or i in r or not all(0 <= j < N for j in r)
                                 for i, r in enumerate(rows)):
            raise ValueError("explicit topology: N rows, each with K_NK distinct partners other than the component")
        return np.array([[i, *r] for i, r in enumerate(rows)], dtype=np.int64)
    if topology == "adjacent":
        return np.array([[(i + j) % N for j in range(K_NK + 1)] for i in range(N)], dtype=np.int64)
    if topology == "random":
        rng = rng if rng is not None else np.random.default_rng(0)
        out = []
        for i in range(N):
            others = np.array([j for j in range(N) if j != i])
            out.append([i, *sorted(rng.choice(others, K_NK, replace=False).tolist())])
        return np.array(out, dtype=np.int64).reshape(N, K_NK + 1)
    if topology == "block":
        b = K_NK + 1
        out = []
        for i in range(N):
            start = (i // b) * b
            members = [(start + j) % N for j in range(b)]
            out.append([i, *[m for m in members if m != i]])
        return np.array(out, dtype=np.int64)
    raise ValueError(f"Unknown topology {topology!r}")


@dataclass
class Landscape:
    nbr: np.ndarray                    # (N, K_NK + 1)
    tables: np.ndarray                 # (N, 2^(K_NK + 1))

    @property
    def N(self) -> int:
        return self.nbr.shape[0]

    def index(self, X: np.ndarray) -> np.ndarray:
        """Row index into each component's table for configurations X (integers, bit i = component i)."""
        X = np.asarray(X, dtype=np.int64).reshape(-1)
        bits = (X[:, None] >> np.arange(self.N)) & 1
        idx = np.zeros((len(X), self.N), dtype=np.int64)
        for j in range(self.nbr.shape[1]):
            idx |= bits[:, self.nbr[:, j]] << j
        return idx

    def fitness(self, X) -> np.ndarray:
        idx = self.index(X)
        return self.tables[np.arange(self.N), idx].mean(axis=1)

    def contributions(self, X) -> np.ndarray:
        return self.tables[np.arange(self.N), self.index(X)]


def make_landscape(N: int, K_NK: int, topology: Topology, rng: np.random.Generator) -> Landscape:
    nbr = interactions(N, K_NK, topology, rng)
    return Landscape(nbr, rng.random((N, 2 ** (K_NK + 1))))


@dataclass
class Epoch:
    start: int
    land: Landscape
    F: Optional[np.ndarray]            # all payoffs if enumerated
    fmax: float
    exact: bool
    peak: Optional[np.ndarray] = None  # local peak indicator per configuration if enumerated


def _enumerate(land: Landscape) -> Tuple[np.ndarray, np.ndarray]:
    F = land.fitness(np.arange(2 ** land.N))
    X = np.arange(2 ** land.N)
    nb = np.stack([F[X ^ (1 << i)] for i in range(land.N)], 1)
    return F, F >= nb.max(1)


def landscape_path(env: NKEnv) -> List[Epoch]:
    """The landscape and its unannounced changes over the horizon (seeded by env.seed only)."""
    ss = np.random.SeedSequence([env.seed, 1])
    r_land, r_change = (np.random.default_rng(s) for s in ss.spawn(2))
    land = make_landscape(env.N, env.K_NK, env.topology, r_land)
    hits = r_change.random(env.periods) < env.hazard
    # Components redrawn at a change. change_frac = 0 means no component changes, so the landscape never changes
    # and no further epoch is created: it is the negative control for environmental change. (Until 8 October 2026
    # this was max(1, ...), so change_frac = 0 still redrew one component at every hit and the "no change" control
    # silently changed the landscape.)
    m = int(round(env.change_frac * env.N))
    epochs = [land]
    starts = [0]
    for t in (np.flatnonzero(hits) if m > 0 else ()):
        if t == 0:
            continue
        comps = r_change.choice(env.N, m, replace=False)
        tab = epochs[-1].tables.copy()
        tab[comps] = r_change.random((m, tab.shape[1]))
        epochs.append(Landscape(land.nbr, tab))
        starts.append(int(t))
    out = []
    for s, L in zip(starts, epochs):
        if env.N <= ENUM_MAX:
            F, peak = _enumerate(L)
            out.append(Epoch(s, L, F, float(F.max()), True, peak))
        else:
            out.append(Epoch(s, L, None, _best_known(L, np.random.default_rng([env.seed, s, 7])), False))
    return out


def _climb(land: Landscape, x: int, F=None) -> int:
    """Full-information steepest ascent by single flips to a local peak."""
    N = land.N
    while True:
        nb = np.array([x ^ (1 << i) for i in range(N)])
        fn = F[nb] if F is not None else land.fitness(nb)
        fx = F[x] if F is not None else land.fitness([x])[0]
        j = int(np.argmax(fn))
        if fn[j] <= fx:
            return x
        x = int(nb[j])


def _best_known(land: Landscape, rng: np.random.Generator, starts: int = 50) -> float:
    xs = rng.integers(0, 2 ** land.N, starts)
    return float(max(land.fitness([_climb(land, int(x))])[0] for x in xs))


# ================================================================================================ information
class Observer:
    """The only access agents have to the world: noisy payoff observations (and the leader's observable components).

    Slots: 0 operating observation of the incumbent, 1 candidate evaluation, 2 verification of the candidate,
    3 re-observation of the incumbent for verification. Noise draws are fixed per (period, slot), shared by searchers."""

    def __init__(self, env: NKEnv, path: List[Epoch], noise: np.ndarray, leader_seen: Optional[np.ndarray] = None,
                 visible: Optional[np.ndarray] = None):
        self._env, self._path, self._noise = env, path, noise
        self._ep = np.zeros(env.periods, dtype=np.int64)
        for k, e in enumerate(path):
            self._ep[e.start:] = k
        self._leader, self._visible = leader_seen, visible

    def observe(self, x: int, t: int, slot: int) -> float:
        return true_payoff(self._path[self._ep[t]], x) + self._env.obs_noise * self._noise[t, slot]

    def leader(self, t: int) -> Dict[int, int]:
        """Observable components of the leader's configuration at the end of period t: {component: value}."""
        if self._leader is None:
            return {}
        x = int(self._leader[t])
        return {int(i): (x >> int(i)) & 1 for i in self._visible}


def true_payoff(ep: Epoch, x: int) -> float:
    return float(ep.F[x]) if ep.F is not None else float(ep.land.fitness([x])[0])


def _is_peak(ep: Epoch, x: int) -> bool:
    if ep.peak is not None:
        return bool(ep.peak[x])
    nb = np.array([x ^ (1 << i) for i in range(ep.land.N)])
    return bool(ep.land.fitness(nb).max() <= ep.land.fitness([x])[0])


# ================================================================================================ runs
@dataclass
class RunStreams:
    start: int
    noise: np.ndarray                  # (T, 4) standard normal
    U: np.ndarray                      # (T, 2N + 2) uniforms for proposals and acceptance
    leader: np.ndarray                 # leader's configuration at the end of each period
    visible: np.ndarray                # leader components observable to the imitator


def run_streams(env: NKEnv, path: List[Epoch], run_seed: int) -> RunStreams:
    """Starting configuration, noise and proposal draws for one run, plus the leader's path (seeded by run_seed)."""
    ss = np.random.SeedSequence([env.seed, run_seed, 2])
    r_start, r_noise, r_prop, r_lead = (np.random.default_rng(s) for s in ss.spawn(4))
    T, N = env.periods, env.N
    start = int(r_start.integers(0, 2 ** N))
    noise = r_noise.standard_normal((T, 4))
    U = r_prop.random((T, 2 * N + 2))
    ep = np.zeros(T, dtype=np.int64)
    for k, e in enumerate(path):
        ep[e.start:] = k
    x = int(r_lead.integers(0, 2 ** N))
    lead = np.empty(T, dtype=np.int64)
    for t in range(T):                                          # one steepest-ascent step per period
        e = path[ep[t]]
        nb = np.array([x ^ (1 << i) for i in range(N)])
        fn = e.F[nb] if e.F is not None else e.land.fitness(nb)
        j = int(np.argmax(fn))
        if fn[j] > true_payoff(e, x):
            x = int(nb[j])
        lead[t] = x
    visible = np.sort(r_lead.permutation(N)[:int(round(env.observe * N))])
    return RunStreams(start, noise, U, lead, visible)


def _flip_set(U_row: np.ndarray, N: int, d: int) -> List[int]:
    return np.argsort(U_row[:N], kind="stable")[:d].tolist()


def _propose(searcher: str, x: int, t: int, U: np.ndarray, N: int, hp: SearchParams, obs: Observer) -> int:
    row = U[t]
    if searcher in ("stochastic",) + GATED:
        d = 1 + int((row[N + 2:2 * N + 1] < hp.p_jump).sum())
        bits = _flip_set(row, N, d)
    elif searcher == "imitate":
        seen = obs.leader(t - 1) if t > 0 else {}
        diff = [i for i, v in seen.items() if ((x >> i) & 1) != v]
        bits = [min(diff, key=lambda i: row[i])] if diff else _flip_set(row, N, 1)
    else:
        bits = _flip_set(row, N, 1)
    y = x
    for i in bits:
        y ^= 1 << i
    return y


def simulate(searcher: str, env: NKEnv, obs: Observer, start: int, U: np.ndarray, hp: SearchParams,
             record: Optional[Dict[str, list]] = None) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Agent side: the searcher's configurations and switching costs per period, from observations only.

    Returns (configuration held in each period, switching cost charged in each period, decision log). The decision log
    holds what the agent saw and did (researcher accounting adds true values later)."""
    T, N, budget = env.periods, env.N, env.budget
    xs = np.empty(T, dtype=np.int64)
    costs = np.zeros(T)
    log = []
    x, used = start, 0
    pending_cost = 0.0
    A = None                                                   # satisficing aspiration
    v = hp.noise_prior ** 2                                    # agent's running estimate of observation variance
    prev_o, prev_x = None, None
    S = np.zeros((gates.N_STATS, N_BINS))
    for t in range(T):
        xs[t], costs[t] = x, pending_cost
        pending_cost = 0.0
        o_x = obs.observe(x, t, 0)
        if prev_o is not None and prev_x == x:
            v += hp.noise_rate * ((o_x - prev_o) ** 2 / 2.0 - v)
        prev_o, prev_x = o_x, x
        if searcher == "satisfice":
            A = o_x + hp.aspiration_start if A is None else A
            want = o_x < A
        else:
            want = True
        width = 2 if searcher in GATED else 1
        if want and used + 1 <= budget and (searcher not in GATED or used + width <= budget):
            y = _propose(searcher, x, t, U, N, hp, obs)
            o_y = obs.observe(y, t, 1)
            used += 1
            c_move = env.cost * bin(x ^ y).count("1")
            g1 = o_y - o_x - c_move
            sig = g1 / max(np.sqrt(2.0 * v), 1e-9)
            b = int((sig > EDGES).sum())
            if searcher == "hill" or searcher == "imitate":
                accept = g1 > 0
            elif searcher == "stochastic":
                temp = hp.temp0 * max(1.0 - used / budget, 0.0)
                accept = g1 > 0 or (temp > 0 and U[t, N] < np.exp(g1 / temp))
            elif searcher == "satisfice":
                accept = o_y - c_move >= A
            else:
                apparent = g1 > 0
                if not apparent:
                    accept = False
                elif searcher == "gate_none":
                    accept = True
                else:
                    lam = hp.memory
                    tab = gates.gate_table("gain" if searcher == "gate_gain" else "lcb", (1 - lam) * S[1], S, None,
                                           lam, 0.0, hp.confidence, hp.min_evidence)
                    accept = bool(tab["mode"][b] == gates.ADAPT)
            entry = dict(t=t, x=x, y=y, o_x=o_x, o_y=o_y, c=c_move, bin=b, accept=bool(accept), g2=np.nan)
            if searcher in GATED and g1 > 0:                   # verification, released after the decision
                g2 = obs.observe(y, t, 2) - obs.observe(x, t, 3) - c_move
                used += 1
                S[:, b] = gates.update(S[:, b], g2, hp.memory)
                entry["g2"] = g2
            log.append(entry)
            if accept and y != x:
                x, pending_cost = y, c_move
        if searcher == "satisfice":
            A += hp.aspiration_rate * (o_x - A)
    if record is not None:
        record["used"] = used
    return xs, costs, log


def evaluate_run(env: NKEnv, path: List[Epoch], xs: np.ndarray, costs: np.ndarray, log: List[dict], used: int,
                 fmax: Sequence[float]) -> Dict[str, float]:
    """Researcher accounting: true payoffs, regret against the epoch maximum (exact or best-known), false and missed
    improvements, escapes from local peaks and recovery after changes."""
    T = env.periods
    ep = np.zeros(T, dtype=np.int64)
    for k, e in enumerate(path):
        ep[e.start:] = k
    f = np.array([true_payoff(path[ep[t]], int(xs[t])) for t in range(T)])
    top = np.array([fmax[k] for k in ep])
    net = f - costs
    false = missed = true_imp = adopted = 0
    for d in log:
        e = path[ep[d["t"]]]
        tg = true_payoff(e, d["y"]) - true_payoff(e, d["x"]) - d["c"]
        if d["accept"] and d["y"] != d["x"]:
            adopted += 1
            false += tg <= 0
        if tg > 0:
            true_imp += 1
            missed += not d["accept"]
    esc, stuck = 0, 0
    for t in range(T):
        e = path[ep[t]]
        at_peak = _is_peak(e, int(xs[t])) and f[t] < fmax[ep[t]] - 1e-12
        stuck += at_peak
        if t + 1 < T and at_peak and xs[t + 1] != xs[t] and ep[t + 1] == ep[t] and f[t + 1] > f[t]:
            esc += 1
    delays, recovered = [], []
    norm = f / top
    for k in range(1, len(path)):
        t0 = path[k].start
        t1 = path[k + 1].start if k + 1 < len(path) else T
        level = norm[t0 - 1]
        hit = np.flatnonzero(norm[t0:t1] >= level - 1e-12)
        delays.append(float(hit[0]) if len(hit) else float(t1 - t0))
        recovered.append(bool(len(hit)))
    return dict(payoff=float(net.mean()), regret=float((top - f + costs).mean()), evaluations=float(used),
                switches=float(adopted), false_improvements=float(false),
                false_share=float(false / adopted) if adopted else np.nan, missed_improvements=float(missed),
                missed_share=float(missed / true_imp) if true_imp else np.nan, escapes=float(esc),
                stuck_share=float(stuck / T), recovery_delay=float(np.mean(delays)) if delays else np.nan,
                recovered_share=float(np.mean(recovered)) if recovered else np.nan, changes=float(len(path) - 1))


def run_landscape(env: NKEnv, run_seeds: Sequence[int], hp: SearchParams = SearchParams(),
                  searchers: Sequence[str] = SEARCHERS) -> pd.DataFrame:
    """Every searcher from every starting configuration on one landscape path (common random numbers)."""
    path = landscape_path(env)
    sims = []
    for rs in run_seeds:
        st = run_streams(env, path, rs)
        obs = Observer(env, path, st.noise, st.leader, st.visible)
        for s in searchers:
            rec = {}
            xs, costs, log = simulate(s, env, obs, st.start, st.U, hp, rec)
            sims.append((rs, s, xs, costs, log, rec["used"]))
    fmax = [e.fmax for e in path]
    if not path[0].exact:                                       # best-known: also every configuration agents held
        ep = np.zeros(env.periods, dtype=np.int64)
        for k, e in enumerate(path):
            ep[e.start:] = k
        for _, _, xs, _, _, _ in sims:
            for k, e in enumerate(path):
                held = np.unique(xs[ep == k])
                if len(held):
                    fmax[k] = max(fmax[k], float(e.land.fitness(held).max()))
    rows = []
    for rs, s, xs, costs, log, used in sims:
        rows.append(dict(landscape=env.seed, run=rs, searcher=s, benchmark="exact" if path[0].exact else "best-known",
                         **evaluate_run(env, path, xs, costs, log, used, fmax)))
    return pd.DataFrame(rows)


# ================================================================================================ study
@dataclass(frozen=True)
class NKPlan:
    N: int = 8
    K_values: Tuple[int, ...] = (0, 2, 4, 7)                    # pilot settings, not claimed optimal design values
    noise_values: Tuple[float, ...] = (0.0, 0.02, 0.05)
    hazard_values: Tuple[float, ...] = (0.0, 0.005, 0.02)
    base_noise: float = 0.02
    base_hazard: float = 0.005
    topology: Topology = "adjacent"
    change_frac: float = 0.25
    cost: float = 0.002
    budget: int = 200
    periods: int = 300
    observe: float = 0.5
    n_landscapes: int = 20
    n_starts: int = 4
    seed: int = 20261008
    n_boot: int = 1000
    searchers: Tuple[str, ...] = SEARCHERS
    params: SearchParams = field(default_factory=SearchParams)


QUICK_NK = dict(n_landscapes=6, n_starts=2, n_boot=300)


def conditions(plan: NKPlan) -> List[Tuple[str, NKEnv]]:
    """One-factor sweeps around the base noise and hazard, at every K_NK."""
    base = NKEnv(N=plan.N, topology=plan.topology, change_frac=plan.change_frac, cost=plan.cost, budget=plan.budget,
                 periods=plan.periods, observe=plan.observe)
    out = []
    for K in plan.K_values:
        for nz in plan.noise_values:
            out.append(("noise", replace(base, K_NK=K, obs_noise=nz, hazard=plan.base_hazard)))
        for hz in plan.hazard_values:
            out.append(("hazard", replace(base, K_NK=K, obs_noise=plan.base_noise, hazard=hz)))
    return out


def run_nk(plan: NKPlan, progress: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
    conds = conditions(plan)
    frames = []
    for c, (sweep, env) in enumerate(conds):
        if progress:
            progress(c / len(conds), f"K_NK = {env.K_NK}, noise {env.obs_noise:g}, hazard {env.hazard:g}")
        for li in range(plan.n_landscapes):
            # landscapes are seeded by (plan seed, K_NK, index): the same landscape across noise and hazard levels
            e = replace(env, seed=plan.seed * 100 + env.K_NK * 1000 + li)
            df = run_landscape(e, [plan.seed + 7919 * r for r in range(plan.n_starts)], plan.params, plan.searchers)
            df.insert(0, "sweep", sweep)
            df.insert(1, "K_NK", env.K_NK)
            df.insert(2, "obs_noise", env.obs_noise)
            df.insert(3, "hazard", env.hazard)
            frames.append(df)
    if progress:
        progress(1.0, "Done")
    return pd.concat(frames, ignore_index=True)


def summarize(runs: pd.DataFrame, metric: str, by: Sequence[str], n_boot: int = 1000) -> pd.DataFrame:
    """Mean and 95% cluster-bootstrap interval over landscapes (runs on a landscape are dependent)."""
    rows = []
    for key, g in runs.groupby(list(by), sort=True):
        g = g.dropna(subset=[metric])
        if g.empty:
            continue
        m, lo, hi, _ = cluster_ci(g[metric].to_numpy(float), g["landscape"].to_numpy(), n_boot, 0.95)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "mean": m, "lo": lo, "hi": hi,
                     "landscapes": g["landscape"].nunique(), "runs": len(g)})
    return pd.DataFrame(rows)


def paired(runs: pd.DataFrame, a: str, b: str, metric: str, by: Sequence[str], n_boot: int = 1000) -> pd.DataFrame:
    """Paired difference a - b per run (same landscape, start and draws), clustered by landscape."""
    keys = list(by) + ["landscape", "run"]
    w = runs[runs["searcher"].isin([a, b])].pivot_table(index=keys, columns="searcher", values=metric).reset_index()
    w["diff"] = w[a] - w[b]
    w["searcher"] = f"{a} - {b}"
    return summarize(w, "diff", by, n_boot)
