"""Competing theories of decision making under uncertainty, each implemented as an agent.

Every agent faces the same market (the cobweb market of `agents.py`: linear demand with optional
unannounced regime shifts, reflecting random-walk raw-material cost, production decided before the
new cost is known) and receives the same information each period:

    last price P[t-1], last market quantity Q[t-1], every firm's last output and profit,
    last raw-material costs c[t-1], c[t-2], its believed demand curve (outdated for L periods after a
    regime shift) and, if it has competence kappa / noise sigma, a perception of the coming cost.

Agents differ only in their decision architecture, which is the theory being tested:

    paper        Scherpereel & Summers rules: Cournot partial adjustment / Bertrand margin feedback with
                 selection rules (Always, Never, SR1 Small, SR2 Large)
    neoclassical static optimisation with rational expectations: play the Cournot-Nash quantity
    real_options flexibility is an option whose value rises with volatility: adjust faster when volatile
    bayesian     optimal signal extraction (Muth 1960; Kalman filter): partially adjust with the
                 statistically optimal gain implied by the observed signal-to-noise ratio
    heiner       reliability: adjustment speed capped by Heiner's (1989) bound from measured decision
                 errors, deviations from rule B only where learned reliability says they pay (Heiner 1983)
    satisficing  Simon (1955), Cyert & March (1963): keep the routine while profit meets an adaptive
                 aspiration; otherwise local, problemistic search
    biases       heuristics & biases (Tversky & Kahneman): trend extrapolation, overconfident full
                 adjustment and loss aversion
    imitation    fast-and-frugal social heuristic (Gigerenzer & Todd): imitate the most profitable firm
    ecology      organisational ecology (Hannan & Freeman): structurally inert firms with a founding
                 output; adaptation happens through selection (exit and entry), not within the firm

Each agent object handles one firm slot in B parallel markets (state arrays have shape (B,)), so
replications run vectorised while every theory stays a self-contained, readable class.

Market-level selection (organisational ecology's mechanism) is available for any population: a firm
whose capital (initial endowment + cumulative profit) turns negative exits, and an entrant takes its
place: either the same theory, or a theory drawn in proportion to the surviving firms (selection of
organisational forms).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .agents import belief_index, demand_path, make_streams
from .params import Scenario


# =====================================================================================================
# Registry: descriptions, literature, default parameters
# =====================================================================================================

@dataclass(frozen=True)
class TheoryInfo:
    key: str
    label: str
    short: str
    rule: str          # one-line decision rule
    claim: str         # the theory's prediction tested in the scorecard
    literature: str
    defaults: Dict[str, float]
    param_help: Dict[str, str]


THEORIES: Dict[str, TheoryInfo] = {t.key: t for t in [
    TheoryInfo(
        "neoclassical", "Neoclassical optimizer",
        "Static optimisation with rational expectations: every period it produces the Cournot–Nash quantity "
        "for the expected cost, assuming all rivals do the same. Fully flexible and equilibrium-seeking.",
        "q = (a − ĉ) / (b (n + 1)), with a, b its demand model and ĉ = last cost",
        "Optimising firms earn the most in every environment, and markets of optimisers sit at the Nash equilibrium.",
        "Cournot (1838); Muth (1961); textbook IO",
        {}, {}),
    TheoryInfo(
        "real_options", "Real-options flexibility seeker",
        "Treats flexibility as an option whose value grows with uncertainty. It partially adjusts toward its "
        "best reply, with an adjustment speed φ that rises with the volatility of costs it has observed.",
        "q ← q + φₜ (BR − q),  φₜ = clip(σ̂ / σ_ref, φ_min, 1), σ̂ = smoothed |Δc|",
        "The value of flexibility rises with volatility, so this agent gains ground as volatility rises.",
        "Dixit & Pindyck (1994); Trigeorgis (1996)",
        {"sigma_ref": 10.0, "phi_min": 0.1, "memory": 0.95},
        {"sigma_ref": "Volatility at which it reaches full flexibility (σ̂ is the smoothed |Δc|)",
         "phi_min": "Minimum adjustment speed", "memory": "Smoothing of the volatility estimate"}),
    TheoryInfo(
        "bayesian", "Bayesian (Kalman) learner",
        "Optimal statistical updating: treats its perceived best reply as a noisy signal of a drifting target "
        "and moves toward it with the Kalman gain implied by the measured signal-to-noise ratio (Muth's optimal "
        "adaptive expectations). The optimising explanation of partial adjustment.",
        "x̂ ← x̂ + g (BR − x̂),  g = steady-state Kalman gain from the variance and autocovariance of ΔBR",
        "Statistically optimal updating beats ad-hoc adjustment speeds under risk.",
        "Muth (1960); Kalman (1960); Brainard (1967)",
        {"memory": 0.98, "g_init": 0.5},
        {"memory": "Smoothing of the variance estimates", "g_init": "Gain used until estimates are available"}),
    TheoryInfo(
        "heiner", "Heiner reliability agent",
        "Constrains itself where its decisions are unreliable. Its adjustment speed is capped by Heiner's (1989) "
        "bound β₀ = 1/((1+K)(1−f′)), using its own measured error-to-signal ratio K. It deviates from rule B only "
        "for move sizes whose learned net gain is positive (the reliability condition). Gains are evaluated over "
        "H periods from what it observed afterwards.",
        "rec = q + min(β_max, β₀(K̂)) (BR − q);  adopt rec only if the learned gain for |rec − q| ≥ 0",
        "Constrained, reliability-based behaviour gains relative to flexible agents as uncertainty (the CD-gap) "
        "grows.",
        "Heiner (1983, 1989)",
        {"beta_max": 1.0, "horizon": 10, "memory": 0.97},
        {"beta_max": "Upper limit on the adjustment speed", "horizon": "Periods over which it judges a past deviation",
         "memory": "Forgetting factor for K̂ and learned gains"}),
    TheoryInfo(
        "satisficing", "Satisficer",
        "Bounded rationality as satisficing. It keeps producing the same quantity while profit meets an adaptive "
        "aspiration level. When profit falls short, it searches locally: it steps output in the direction that "
        "last improved profit, and reverses if things got worse.",
        "if π ≥ A: keep q; else q ← q (1 ± s);  A ← (1−α) A + α π",
        "Satisficers survive at least as well as optimisers while not maximising profit.",
        "Simon (1955); Cyert & March (1963)",
        {"step": 0.05, "alpha": 0.1},
        {"step": "Search step as a share of output", "alpha": "Speed at which aspirations adapt to experience"}),
    TheoryInfo(
        "biases", "Heuristics & biases agent",
        "A best-replying agent with systematic biases. Representativeness leads it to extrapolate recent cost "
        "trends; overconfidence leads it to adjust fully to its forecast; loss aversion makes it over-react to "
        "losses by cutting output more than the forecast implies.",
        "ĉ = c₋₁ + τ (c₋₁ − c₋₂); q ← BR(ĉ); after a loss, q ← q + λ (BR − q) for cuts",
        "Biases are costly: the biased agent earns less than its debiased twin, more so when volatile.",
        "Tversky & Kahneman (1974); Kahneman & Tversky (1979); Tversky & Kahneman (1992)",
        {"trend": 0.5, "loss_aversion": 2.25},
        {"trend": "Trend extrapolation τ (0 = none)", "loss_aversion": "Over-reaction λ to losses (1 = none)"}),
    TheoryInfo(
        "imitation", "Fast-and-frugal imitator",
        "A frugal social heuristic: ignore costs and demand and copy last period's output of the most profitable "
        "firm in the market (keep its own output if it was the best). One cue, no model.",
        "q ← q of argmaxⱼ π_j (last period)",
        "Less is more: a one-cue heuristic does at least as well as model-based optimisation under uncertainty.",
        "Gigerenzer, Todd & ABC Research Group (1999); Gigerenzer & Brighton (2009)",
        {}, {}),
    TheoryInfo(
        "ecology", "Organizational-ecology inert firm",
        "Structurally inert: it keeps its founding output forever. Adaptation happens at the population level: "
        "with market selection switched on, firms that run out of capital exit and entrants with new founding "
        "outputs or forms take their place.",
        "q = q_founding (never revised); founding q = q₀ (1 + d z)",
        "Inert, reliable forms are favoured by selection: they fail less and gain population share when the "
        "environment is uncertain.",
        "Hannan & Freeman (1977, 1984)",
        {"dispersion": 0.15},
        {"dispersion": "Spread of founding outputs around the initial output"}),
    TheoryInfo(
        "paper", "Paper's Cournot/Bertrand rule",
        "The decision rules of Scherpereel & Summers. Cournot: move a fraction φ toward the best reply. Bertrand: "
        "adjust by φ × (margin − m*). A selection rule (Always / Never / SR1 Small / SR2 Large) decides when to "
        "deviate from rule B.",
        "Cournot: q ← φ BR + (1−φ) q;  Bertrand: q ← q + φ (P − ĉ − m*); selection rule on |Δq| vs θ",
        "(Reference model of the paper, tested in the Special Simulations.)",
        "Scherpereel & Summers (2011)",
        {"rule": 0.0, "flex": 0.5, "selection": 0.0, "threshold": 25.0, "desired_margin": 5.0},
        {"rule": "0 = Cournot, 1 = Bertrand", "flex": "Flexibility φ",
         "selection": "0 Always, 1 Never, 2 Small (SR1), 3 Large (SR2)", "threshold": "θ for SR1/SR2",
         "desired_margin": "m* (Bertrand)"}),
]}

GENERAL_THEORIES = ["neoclassical", "real_options", "bayesian", "heiner", "satisficing", "biases", "imitation", "ecology"]
SEL_NAMES = ["Always", "Never", "Small", "Large"]


@dataclass
class AgentSpec:
    theory: str
    params: Dict[str, float] = field(default_factory=dict)
    label: Optional[str] = None
    foresight: float = 0.0       # competence (cost foresight), same meaning as in the paper model
    noise: float = 0.0           # perception noise

    def name(self) -> str:
        return self.label or THEORIES[self.theory].label


@dataclass
class SelectionParams:
    enabled: bool = False
    capital0: float = 20000.0          # initial capital of every firm (and entrant)
    mode: str = "same"                 # "same" theory re-enters, or "survivors": drawn from surviving forms
    mutation: float = 0.05             # probability an entrant's form is drawn uniformly from all candidates


# =====================================================================================================
# Shared context and observations
# =====================================================================================================

@dataclass
class Ctx:
    B: int
    N: int
    T: int
    p_min: np.ndarray
    q_min: np.ndarray
    q0: np.ndarray
    rngs: List[np.random.Generator]      # per-market streams for founding draws and entry


@dataclass
class Obs:
    t: int
    P_prev: np.ndarray       # (B,)
    Q_prev: np.ndarray       # (B,)
    q_prev: np.ndarray       # (B, N)
    profit_prev: np.ndarray  # (B, N)
    c_prev: np.ndarray       # (B,)
    c_prev2: np.ndarray      # (B,)
    c_now: np.ndarray        # (B,) realised later this period (only used through foresight)
    pmb: np.ndarray          # (B,) believed demand intercept
    slb: np.ndarray          # (B,) believed demand slope
    eps: np.ndarray          # (B, N)


@dataclass
class Outcome:
    t: int
    q: np.ndarray            # (B, N) chosen outputs
    q_prev: np.ndarray       # (B, N) outputs before the decision
    Q: np.ndarray            # (B,)
    P: np.ndarray            # (B,)
    c: np.ndarray            # (B,) realised cost
    profit: np.ndarray       # (B, N)
    pmb: np.ndarray          # (B,) believed demand at t
    slb: np.ndarray


# =====================================================================================================
# Agents
# =====================================================================================================

class Agent:
    key = "base"

    def __init__(self, spec: AgentSpec, ctx: Ctx, i: int):
        self.spec = spec
        self.p = {**THEORIES[spec.theory].defaults, **spec.params}
        self.ctx = ctx
        self.i = i
        self.flex = np.zeros(ctx.B)          # adjustment-capacity measure (used for flexibility costs)
        self.reset(np.ones(ctx.B, dtype=bool))

    def reset(self, mask: np.ndarray):
        """(Re)initialise internal state for markets in mask (entry of a new firm)."""

    def cost_estimate(self, o: Obs) -> np.ndarray:
        return o.c_prev + self.spec.foresight * (o.c_now - o.c_prev) + self.spec.noise * o.eps[:, self.i]

    def best_reply(self, o: Obs, c_hat: np.ndarray) -> np.ndarray:
        q_other = o.Q_prev - o.q_prev[:, self.i]
        return (o.pmb - o.slb * q_other - c_hat) / (2 * o.slb)

    def decide(self, o: Obs) -> np.ndarray:
        raise NotImplementedError

    def observe(self, out: Outcome):
        """Learning after the market clears (default: none)."""


class PaperAgent(Agent):
    """Scherpereel & Summers rules, identical to engine.run_batch for Always/Never/Small/Large."""
    key = "paper"

    def decide(self, o):
        p, i = self.p, self.i
        q = o.q_prev[:, i]
        c_hat = self.cost_estimate(o)
        phi = float(p["flex"])
        if int(p["rule"]) == 0:
            rec = phi * self.best_reply(o, c_hat) + (1 - phi) * q
        else:
            rec = q + phi * ((o.P_prev - c_hat) - p["desired_margin"])
        rec = np.maximum(self.ctx.q_min, np.rint(rec))
        change = np.abs(rec - q)
        sel = int(p["selection"])
        dev = [np.ones_like(change, bool), np.zeros_like(change, bool), change < p["threshold"],
               change > p["threshold"]][sel]
        self.flex[:] = phi
        return np.where(dev, rec, q)


class NeoclassicalAgent(Agent):
    key = "neoclassical"

    def decide(self, o):
        c_hat = self.cost_estimate(o)
        return (o.pmb - c_hat) / (o.slb * (self.ctx.N + 1))


class RealOptionsAgent(Agent):
    key = "real_options"

    def reset(self, mask):
        if not hasattr(self, "sig"):
            self.sig = np.zeros(self.ctx.B)
        self.sig[mask] = 0.0

    def decide(self, o):
        p = self.p
        lam = p["memory"]
        self.sig = lam * self.sig + (1 - lam) * np.abs(o.c_prev - o.c_prev2)
        phi = np.clip(self.sig / max(p["sigma_ref"], 1e-9), p["phi_min"], 1.0)
        self.flex[:] = phi
        q = o.q_prev[:, self.i]
        return q + phi * (self.best_reply(o, self.cost_estimate(o)) - q)


class BayesianAgent(Agent):
    key = "bayesian"

    def reset(self, mask):
        B = self.ctx.B
        for name, val in (("xhat", np.nan), ("y_prev", np.nan), ("d_prev", 0.0), ("g0", 0.0), ("g1", 0.0),
                          ("n_obs", 0.0)):
            if not hasattr(self, name):
                setattr(self, name, np.full(B, val))
            getattr(self, name)[mask] = val

    def gain(self):
        r = np.maximum(-self.g1, 1e-9)
        qv = np.maximum(self.g0 - 2 * r, 1e-9)
        rho = qv / r
        g = (-rho + np.sqrt(rho * rho + 4 * rho)) / 2
        return np.where(self.n_obs > 10, np.clip(g, 0.01, 1.0), self.p["g_init"])

    def decide(self, o):
        q = o.q_prev[:, self.i]
        y = self.best_reply(o, self.cost_estimate(o))
        lam = self.p["memory"]
        first = np.isnan(self.y_prev)
        d = np.where(first, 0.0, y - np.nan_to_num(self.y_prev))
        self.g0 = np.where(first, self.g0, lam * self.g0 + (1 - lam) * d * d)
        self.g1 = np.where(first, self.g1, lam * self.g1 + (1 - lam) * d * self.d_prev)
        self.n_obs = self.n_obs + (~first)
        self.d_prev, self.y_prev = d, y
        g = self.gain()
        self.xhat = np.where(np.isnan(self.xhat), q, self.xhat)
        self.xhat = self.xhat + g * (y - self.xhat)
        self.flex[:] = g
        return self.xhat


class HeinerAgent(Agent):
    key = "heiner"
    EDGES = np.array([2.0, 5.0, 10.0, 20.0])

    def reset(self, mask):
        B = self.ctx.B
        Hh = max(1, int(self.p["horizon"]))
        if not hasattr(self, "learned"):
            self.learned = np.zeros((B, len(self.EDGES) + 1))
            self.exi2 = np.zeros(B); self.esig2 = np.zeros(B)
            self.pend_rec = np.zeros((B, Hh)); self.pend_qb = np.zeros((B, Hh))
            self.pend_bin = np.zeros((B, Hh), dtype=int); self.pend_acc = np.zeros((B, Hh))
            self.pend_age = np.zeros((B, Hh), dtype=int); self.pend_valid = np.zeros((B, Hh), dtype=bool)
            self._new = None
        self.learned[mask] = 0.0
        self.exi2[mask] = 0.0; self.esig2[mask] = 0.0
        self.pend_valid[mask] = False
        self.br_perc = np.zeros(B)

    def K(self):
        return np.where(self.esig2 > 0, np.sqrt(self.exi2 / np.maximum(self.esig2, 1e-12)), 1.0)

    def beta(self):
        fprime = -(self.ctx.N - 1) / 2.0
        b0 = 1.0 / ((1.0 + self.K()) * (1.0 - fprime))
        return np.minimum(self.p["beta_max"], b0)

    def decide(self, o):
        q = o.q_prev[:, self.i]
        br = self.best_reply(o, self.cost_estimate(o))
        self.br_perc = br
        beta = self.beta()
        self.flex[:] = beta
        rec = np.maximum(self.ctx.q_min, np.rint(q + beta * (br - q)))
        change = np.abs(rec - q)
        bins = np.searchsorted(self.EDGES, change, side="right")
        ok = self.learned[np.arange(self.ctx.B), bins] >= 0.0
        dev = (change > 0) & ok
        self._new = (rec, q.copy(), bins, change > 0)
        return np.where(dev, rec, q)

    def observe(self, out):
        i, lam = self.i, self.p["memory"]
        # Heiner (1989) error-to-signal ratio from the ex-post best reply (believed demand, realised cost)
        q_other = out.Q - out.q[:, i]
        br_ex = (out.pmb - out.slb * q_other - out.c) / (2 * out.slb)
        xi = self.br_perc - br_ex
        sig = br_ex - out.q_prev[:, i]
        self.exi2 = lam * self.exi2 + (1 - lam) * xi * xi
        self.esig2 = lam * self.esig2 + (1 - lam) * sig * sig
        # delayed persistence counterfactual: adopt rec and hold it vs keep q, rivals on their observed path
        Hh = self.pend_rec.shape[1]
        slot = out.t % Hh
        rec, qb, bins, opp = self._new
        self.pend_rec[:, slot], self.pend_qb[:, slot], self.pend_bin[:, slot] = rec, qb, bins
        self.pend_acc[:, slot], self.pend_age[:, slot], self.pend_valid[:, slot] = 0.0, 0, opp
        pmin = self.ctx.p_min[:, None]
        pa = np.maximum(pmin, out.pmb[:, None] - out.slb[:, None] * (q_other[:, None] + self.pend_rec))
        pb = np.maximum(pmin, out.pmb[:, None] - out.slb[:, None] * (q_other[:, None] + self.pend_qb))
        c = out.c[:, None]
        self.pend_acc += np.where(self.pend_valid, (pa - c) * self.pend_rec - (pb - c) * self.pend_qb, 0.0)
        self.pend_age += self.pend_valid
        done = self.pend_valid & (self.pend_age >= Hh)
        if done.any():
            bi, ki = np.nonzero(done)
            mem = self.p["memory"]
            for b, k in zip(bi, ki):
                j = self.pend_bin[b, k]
                self.learned[b, j] = mem * self.learned[b, j] + (1 - mem) * self.pend_acc[b, k]
            self.pend_valid[done] = False


class SatisficingAgent(Agent):
    key = "satisficing"

    def reset(self, mask):
        B = self.ctx.B
        for name, val in (("A", np.nan), ("pp", np.nan), ("dir", 1.0)):
            if not hasattr(self, name):
                setattr(self, name, np.full(B, val))
            getattr(self, name)[mask] = val

    def decide(self, o):
        p = self.p
        q = o.q_prev[:, self.i]
        pi = o.profit_prev[:, self.i]
        self.A = np.where(np.isnan(self.A), pi, self.A)
        dissat = pi < self.A
        improved = np.where(np.isnan(self.pp), True, pi >= self.pp)
        new_dir = np.where(improved, self.dir, -self.dir)
        step = np.maximum(1.0, p["step"] * q)
        x = np.where(dissat, q + new_dir * step, q)
        self.dir = np.where(dissat, new_dir, self.dir)
        self.pp = pi
        self.A = (1 - p["alpha"]) * self.A + p["alpha"] * pi
        self.flex[:] = dissat
        return x


class BiasedAgent(Agent):
    key = "biases"

    def decide(self, o):
        p = self.p
        q = o.q_prev[:, self.i]
        c_hat = self.cost_estimate(o) + p["trend"] * (o.c_prev - o.c_prev2)
        target = self.best_reply(o, c_hat)
        loss = o.profit_prev[:, self.i] < 0
        return np.where(loss & (target < q), q + p["loss_aversion"] * (target - q), target)


class ImitationAgent(Agent):
    key = "imitation"

    def decide(self, o):
        best = np.argmax(o.profit_prev, axis=1)
        return o.q_prev[np.arange(self.ctx.B), best]


class EcologyAgent(Agent):
    key = "ecology"

    def reset(self, mask):
        if not hasattr(self, "qf"):
            self.qf = np.zeros(self.ctx.B)
        for b in np.flatnonzero(mask):
            z = self.ctx.rngs[b].standard_normal()
            self.qf[b] = max(float(self.ctx.q_min[b]), self.ctx.q0[b] * (1 + self.p["dispersion"] * z))

    def decide(self, o):
        return self.qf.copy()


AGENT_CLASSES = {c.key: c for c in (PaperAgent, NeoclassicalAgent, RealOptionsAgent, BayesianAgent, HeinerAgent,
                                     SatisficingAgent, BiasedAgent, ImitationAgent, EcologyAgent)}


# =====================================================================================================
# The market: runs B parallel markets populated by theory agents
# =====================================================================================================

@dataclass
class TheoryResult:
    specs: List[AgentSpec]                 # initial slot specs
    templates: Dict[str, AgentSpec]        # one spec per candidate theory (used for entrants)
    price: np.ndarray                      # (B, T)
    cost: np.ndarray
    quantity: np.ndarray
    nash_price: np.ndarray                 # (B, T) Cournot-Nash price for the realised cost and true demand
    types: np.ndarray                      # (B, T, N) theory index of each slot (into `theory_keys`)
    theory_keys: List[str]
    acc: Dict[str, np.ndarray]             # (B, n_theories) accumulators after burn-in
    exits: np.ndarray                      # (B, n_theories)
    shifts: np.ndarray                     # (B, T)
    burn_in: int
    hist: Optional[Dict[str, np.ndarray]] = None


def run_theories(scenarios: Sequence[Scenario], specs: Sequence[AgentSpec],
                 selection: Optional[SelectionParams] = None, record: bool = False,
                 candidates: Optional[Sequence[AgentSpec]] = None) -> TheoryResult:
    """Simulate B markets (one per scenario), each with the same slot line-up of theory agents.

    The scenarios supply market parameters, cost volatility, regime shifts, the believed-demand lag,
    initial output, fixed and flexibility costs, periods, burn-in and seed (common random numbers).
    `candidates` lists the forms entrants can take under survivor-weighted selection (default: the
    theories present at the start)."""
    scns = list(scenarios)
    sel = selection or SelectionParams()
    B, N, T = len(scns), len(specs), scns[0].periods
    burn = scns[0].burn_in
    if any(s.periods != T or s.burn_in != burn for s in scns):
        raise ValueError("All scenarios need the same periods and burn-in.")
    templates: Dict[str, AgentSpec] = {}
    for sp in list(specs) + list(candidates or []):
        templates.setdefault(sp.theory, sp)
    keys = list(templates)
    tindex = {k: j for j, k in enumerate(keys)}

    streams = [make_streams(s.seed, N, T) for s in scns]
    U = np.stack([st[0] for st in streams]); EPS = np.stack([st[1] for st in streams])
    rngs = [np.random.default_rng(np.random.SeedSequence(s.seed).spawn(5)[4]) for s in scns]
    dp = [demand_path(s) for s in scns]
    PM = np.stack([d[0] for d in dp]); SL = np.stack([d[1] for d in dp]); SHIFT = np.stack([d[2] for d in dp])
    base_pm = np.array([s.market.p_max for s in scns])[:, None]
    base_sl = np.array([s.market.slope for s in scns])[:, None]
    BI = np.stack([belief_index(s) for s in scns])
    PMB = np.where(BI < 0, base_pm, np.take_along_axis(PM, np.maximum(BI, 0), axis=1))
    SLB = np.where(BI < 0, base_sl, np.take_along_axis(SL, np.maximum(BI, 0), axis=1))
    vec = lambda f: np.array([f(s) for s in scns], dtype=float)
    p_min, c_max, delta = vec(lambda s: s.market.p_min), vec(lambda s: s.market.c_max), vec(lambda s: s.market.delta)
    q_min, q0 = vec(lambda s: s.firm_globals.q_min), vec(lambda s: s.firm_globals.q0)
    fixed, fc_slope = vec(lambda s: s.firm_globals.fixed_cost), vec(lambda s: s.firm_globals.flex_cost_slope)

    cost = np.empty((B, T)); cost[:, 0] = vec(lambda s: s.market.c0)
    for t in range(1, T):
        v = cost[:, t - 1] + delta * U[:, t]
        v = np.where(v <= p_min, 2 * p_min - v, v)
        v = np.where(v >= c_max, 2 * c_max - v, v)
        cost[:, t] = v

    ctx = Ctx(B=B, N=N, T=T, p_min=p_min, q_min=q_min, q0=q0, rngs=rngs)
    # every slot holds one agent per theory it may take; `typ` says which one is active in each market
    agents: List[Dict[str, Agent]] = []
    typ = np.zeros((B, N), dtype=int)
    slot_theories = keys if (sel.enabled and sel.mode == "survivors") else None
    for i, sp in enumerate(specs):
        d = {}
        for k in (slot_theories or [sp.theory]):
            src = sp if k == sp.theory else templates[k]
            d[k] = AGENT_CLASSES[k](AgentSpec(k, dict(src.params), src.label, src.foresight, src.noise), ctx, i)
        agents.append(d)
        typ[:, i] = tindex[sp.theory]

    def founding_q(i):
        q = q0.copy()
        for k, ag in agents[i].items():
            if isinstance(ag, EcologyAgent):
                m = typ[:, i] == tindex[k]
                q = np.where(m, ag.qf, q)
        return q

    q = np.stack([founding_q(i) for i in range(N)], axis=1)
    q = np.maximum(q_min[:, None], np.rint(q))
    Q = q.sum(1)
    P = np.maximum(p_min, PM[:, 0] - SL[:, 0] * Q)
    F = fixed[:, None] + np.zeros((B, N))
    profit = (P - cost[:, 0])[:, None] * q - F
    capital = np.full((B, N), sel.capital0) + profit

    nT = len(keys)
    acc = {k: np.zeros((B, nT)) for k in ("profit", "periods", "absdq", "q", "flex")}
    exits = np.zeros((B, nT))
    price = np.empty((B, T)); quantity = np.empty((B, T)); price[:, 0], quantity[:, 0] = P, Q
    types = np.zeros((B, T, N), dtype=np.int16); types[:, 0] = typ
    hist = None
    if record:
        hist = {k: np.zeros((B, T, N)) for k in ("q", "profit", "flex", "capital")}
        hist["q"][:, 0], hist["profit"][:, 0], hist["capital"][:, 0] = q, profit, capital
    b_idx = np.arange(B)

    for t in range(1, T):
        o = Obs(t=t, P_prev=P, Q_prev=Q, q_prev=q, profit_prev=profit, c_prev=cost[:, t - 1],
                c_prev2=cost[:, max(t - 2, 0)], c_now=cost[:, t], pmb=PMB[:, t], slb=SLB[:, t], eps=EPS[:, t, :])
        new_q = np.empty((B, N)); flexv = np.zeros((B, N))
        for i in range(N):
            for k, ag in agents[i].items():
                m = typ[:, i] == tindex[k]
                x = ag.decide(o)          # every agent decides (keeps its learning consistent); only active ones act
                new_q[m, i] = x[m]
                flexv[m, i] = ag.flex[m]
        new_q = np.maximum(q_min[:, None], np.rint(np.nan_to_num(new_q, nan=0.0)))
        Qn = new_q.sum(1)
        Pn = np.maximum(p_min, PM[:, t] - SL[:, t] * Qn)
        Fv = fixed[:, None] + np.where(np.isin(typ, [tindex.get("paper", -9), tindex.get("real_options", -9)]),
                                       fc_slope[:, None] * flexv, 0.0)
        prof = (Pn - cost[:, t])[:, None] * new_q - Fv
        out = Outcome(t=t, q=new_q, q_prev=q, Q=Qn, P=Pn, c=cost[:, t], profit=prof, pmb=PMB[:, t], slb=SLB[:, t])
        for i in range(N):
            for ag in agents[i].values():
                ag.observe(out)
        if t >= burn:
            for j in range(nT):
                m = (typ == j)
                acc["profit"][:, j] += np.where(m, prof, 0).sum(1)
                acc["periods"][:, j] += m.sum(1)
                acc["absdq"][:, j] += np.where(m, np.abs(new_q - q), 0).sum(1)
                acc["q"][:, j] += np.where(m, new_q, 0).sum(1)
                acc["flex"][:, j] += np.where(m, flexv, 0).sum(1)
        capital = capital + prof
        q, Q, P, profit = new_q, Qn, Pn, prof
        price[:, t], quantity[:, t] = P, Q
        # ---- market selection: exit when capital is exhausted, an entrant takes the slot
        if sel.enabled:
            dead = capital < 0
            if dead.any():
                for b, i in zip(*np.nonzero(dead)):
                    if t >= burn:
                        exits[b, typ[b, i]] += 1
                    if sel.mode == "survivors":
                        alive = [typ[b, j] for j in range(N) if j != i and not dead[b, j]]
                        rng = rngs[b]
                        if not alive or rng.random() < sel.mutation:
                            new_t = int(rng.integers(0, nT))
                        else:
                            new_t = int(alive[int(rng.integers(0, len(alive)))])
                    else:
                        new_t = typ[b, i]
                    typ[b, i] = new_t
                    mask = np.zeros(B, dtype=bool); mask[b] = True
                    ag = agents[i][keys[new_t]]
                    ag.reset(mask)
                    others = [j for j in range(N) if j != i]
                    entry_q = ag.qf[b] if isinstance(ag, EcologyAgent) else (np.mean(q[b, others]) if others else q0[b])
                    q[b, i] = max(q_min[b], np.rint(entry_q))
                    profit[b, i] = 0.0
                    capital[b, i] = sel.capital0
                Q = q.sum(1)
        types[:, t] = typ
        if record:
            hist["q"][:, t], hist["profit"][:, t], hist["flex"][:, t], hist["capital"][:, t] = q, profit, flexv, capital

    q_n = (PM - cost) / (SL * (N + 1))
    nash_price = np.maximum(p_min[:, None], PM - SL * N * q_n)
    return TheoryResult(specs=list(specs), templates=templates, price=price, cost=cost, quantity=quantity,
                        nash_price=nash_price, types=types, theory_keys=keys, acc=acc, exits=exits, shifts=SHIFT,
                        burn_in=burn, hist=hist)


def theory_table(res: TheoryResult) -> pd.DataFrame:
    """One row per (market, theory): average profit per firm-period, relative profit, activity, exits."""
    a = res.acc
    burn = res.burn_in
    rows = []
    mkt_mean = a["profit"].sum(1) / np.maximum(a["periods"].sum(1), 1)
    for b in range(res.price.shape[0]):
        for j, k in enumerate(res.theory_keys):
            n = a["periods"][b, j]
            if n == 0 and res.exits[b, j] == 0:
                continue
            rows.append(dict(market=b, theory=k, label=res.templates[k].name(),
                             firm_periods=n, avg_profit=a["profit"][b, j] / n if n else np.nan,
                             rel_profit=(a["profit"][b, j] / n - mkt_mean[b]) if n else np.nan,
                             avg_absdq=a["absdq"][b, j] / n if n else np.nan,
                             avg_q=a["q"][b, j] / n if n else np.nan,
                             avg_flex=a["flex"][b, j] / n if n else np.nan,
                             exits=res.exits[b, j],
                             share=n / max(a["periods"][b].sum(), 1)))
    return pd.DataFrame(rows)


def industry_table(res: TheoryResult) -> pd.DataFrame:
    burn = res.burn_in
    P, C, NP = res.price[:, burn:], res.cost[:, burn:], res.nash_price[:, burn:]
    a = res.acc
    return pd.DataFrame(dict(
        market=np.arange(P.shape[0]), avg_price=P.mean(1), avg_margin=(P - C).mean(1),
        price_change_rms=np.sqrt((np.diff(P, axis=1) ** 2).mean(1)),
        nash_gap=np.abs(P - NP).mean(1), price_minus_nash=(P - NP).mean(1),
        industry_profit=a["profit"].sum(1) / (P.shape[1]),
        exits=res.exits.sum(1), n_shifts=res.shifts[:, burn:].sum(1)))
