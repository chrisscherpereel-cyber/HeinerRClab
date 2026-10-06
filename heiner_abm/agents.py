"""Readable agent-based reference implementation.

Each Firm is an agent with its own decision rule, selection rule, competence and
memory. The Market is a single homogeneous-product, market-clearing cobweb market
with a linear hockey-stick demand curve and a reflecting random-walk raw-material
cost. The Industry runs the decision schedule below (DECISION_SCHEDULE), which the
vectorized engine (heiner_abm.engine) follows exactly; tests/test_equivalence.py
checks that the two agree and tests/test_information.py checks the timing.

`heiner_abm.engine` is a vectorized twin of this module used for Monte-Carlo
experiments.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .information import Observation
from .params import Scenario, FirmSpec

DECISION_SCHEDULE = """
Period t (t = 1, ..., T - 1):

 1. Decide. Each firm receives an Observation (heiner_abm.information) built from what it has observed under the
    scenario's information specification: the market outcomes of period s = t - 1 - delay (price and market output,
    with observation noise; the cost c[s]; its own output and payoff; rivals' individual outputs and payoffs if
    visible), its current output, the demand curve it may use (believed, true or none; updated at once if shifts
    are announced) and its cost estimate c_hat[t] = c[s] + kappa * (c[t] - c[s]) + sigma * eps[t]. Cost foresight
    kappa is a competence parameter: a firm with kappa > 0 anticipates that share of the coming cost change (kappa = 0
    in the baseline). Its selection rule uses its learned table as it stood at the end of period t - 1.
 2. Clear. The market clears on the true demand curve, P[t] = max(Pmin, Pmax[t] - s[t] * Q[t]); the cost c[t] is
    realized and every firm books profit (P[t] - c[t]) * q - F.
 3. Researcher evaluation (never visible to agents). The market is forked at t and each firm's decision is valued
    against rule B over the measurement horizon H (Heiner's pi, r, w, G, D). The forks use periods t .. t + H - 1,
    the true demand curve and rivals' true rules; inside the forks, firms decide from observations built the same
    way (from the actual history before t and the fork's own history from t on).
 4. Feedback enters the pending queue. Every decision with an opportunity (q* != q) is queued with its maturity:
      estimated (default): matures at t + W - 1 (W = AdaptiveParams.window); its gain accumulates period by period,
                           up to maturity, from what the firm observes (rivals' observed output, realized cost,
                           believed demand);
      oracle (researcher-only diagnostic): the researcher's H-period counterfactual of step 3, matures at t + H - 1.
    Each pending estimated item that has not yet matured then adds period t's observed term.
 5. Release. Every pending item whose release period, maturity + observation delay, is t is released to its firm, in
    order of decision time, and updates the firm's learned table. Released feedback is first used in the decisions
    of period t + 1. Items that would be released after the last period T - 1 are never released (no partial
    feedback). Chosen-action and full feedback carry no counterfactual for the Adaptive rule, which the information
    specification therefore rejects under those treatments.
 6. Evolution (if enabled): firms revise flexibility by imitating the most profitable rival over the window just
    ended (realized profits only).

With W = 1 or H = 1 the feedback about the decision of period t is released at the end of period t, as before.
"""


def make_streams(seed: int, n_firms: int, periods: int):
    """Independent random streams so the cost path depends only on the seed (common random numbers)."""
    cost_ss, noise_ss, evo_ss = np.random.SeedSequence(seed).spawn(3)
    u = np.random.default_rng(cost_ss).uniform(-1.0, 1.0, size=periods)
    eps = np.random.default_rng(noise_ss).standard_normal((periods, n_firms))
    return u, eps, np.random.default_rng(evo_ss)


def regime_draws(seed: int, periods: int) -> np.ndarray:
    """Separate stream for structural shocks (4th child), so enabling them leaves other streams intact."""
    ss = np.random.SeedSequence(seed).spawn(4)[3]
    rng = np.random.default_rng(ss)
    return np.stack([rng.random(periods), rng.standard_normal(periods), rng.standard_normal(periods)])


def observation_noise(seed: int, periods: int, n_firms: int) -> np.ndarray:
    """Separate stream (5th child) for observation noise on (price, market quantity) per period and firm, shape
    (2, periods, n_firms); enabling it leaves every other stream intact."""
    ss = np.random.SeedSequence(seed).spawn(5)[4]
    return np.random.default_rng(ss).standard_normal((2, periods, n_firms))


def demand_path(scn: Scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True demand parameters per period (p_max[t], slope[t]) and regime-shift indicator."""
    m, sp, T = scn.market, scn.structural, scn.periods
    pmax = np.full(T, m.p_max)
    slope = np.full(T, m.slope)
    shift = np.zeros(T, dtype=bool)
    if sp.enabled and sp.hazard > 0:
        u, z1, z2 = regime_draws(scn.seed, T)
        for t in range(1, T):
            if u[t] < sp.hazard:
                shift[t] = True
                pmax[t] = max(m.p_min + 10.0, m.p_max + sp.intercept_sd * z1[t])
                slope[t] = m.slope * float(np.clip(np.exp(sp.slope_sd * z2[t]), 0.25, 4.0))
            else:
                pmax[t], slope[t] = pmax[t - 1], slope[t - 1]
    return pmax, slope, shift


def belief_index(scn: Scenario) -> np.ndarray:
    """Index of the regime a firm's demand model reflects when deciding in period t (-1 = baseline)."""
    T, sp = scn.periods, scn.structural
    t = np.arange(T)
    if not sp.enabled or sp.belief_lag < 0:
        return np.full(T, -1)
    return np.maximum(t - 1 - sp.belief_lag, 0)


def reflect(value: float, lo: float, hi: float) -> float:
    """Reflecting boundary: overshoot beyond a bound is mirrored back inside."""
    if value <= lo:
        return lo + (lo - value)
    if value >= hi:
        return hi - (value - hi)
    return value


def cost_path(scn: Scenario, u: np.ndarray) -> np.ndarray:
    m = scn.market
    c = np.empty(scn.periods)
    c[0] = m.c0
    for t in range(1, scn.periods):
        c[t] = reflect(c[t - 1] + m.delta * u[t], m.p_min, m.c_max)
    return c


def belief_curves(scn: Scenario, p_max: np.ndarray, slope: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The demand curve firms decide with in each period. 'believed': the baseline curve, updated L periods after a
    shift, or at once if shifts are announced; 'true': the true current curve (ORACLE information); with 'none' the
    curve is kept only for the researcher's diagnostics and is not given to the firms."""
    info = scn.info
    if info.demand_knowledge == "true" or info.regime_announced:
        return p_max.copy(), slope.copy()
    bi = belief_index(scn)
    return (np.where(bi < 0, scn.market.p_max, p_max[np.maximum(bi, 0)]),
            np.where(bi < 0, scn.market.slope, slope[np.maximum(bi, 0)]))


class Market:
    def __init__(self, scn: Scenario, u: np.ndarray):
        self.p = scn.market
        self.costs = cost_path(scn, u)
        self.p_max, self.slope, self.shift = demand_path(scn)
        self.belief_p_max, self.belief_slope = belief_curves(scn, self.p_max, self.slope)
        self.price = 0.0
        self.quantity = 0.0

    def clearing_price(self, quantity: float, t: int) -> float:
        return max(self.p.p_min, self.p_max[t] - self.slope[t] * quantity)

    def clear(self, quantity: float, t: int) -> float:
        self.quantity = quantity
        self.price = self.clearing_price(quantity, t)
        return self.price


@dataclass
class Decision:
    recommended: float
    current: float
    deviate: bool
    chosen: float
    c_hat: float
    best_reply: float     # Cournot perceived best reply (Heiner's perceived target), before partial adjustment


class Firm:
    def __init__(self, idx: int, spec: FirmSpec, scn: Scenario):
        self.idx = idx
        self.spec = spec
        self.g = scn.firm_globals
        self.flex = spec.flex
        self.fixed = scn.flex_cost(spec.flex)
        self.q = scn.firm_globals.q0
        self.bin_edges = np.asarray(scn.adaptive.bin_edges, dtype=float)
        self.memory = scn.adaptive.memory
        self.learned_gain = np.zeros(len(self.bin_edges) + 1)
        self.last: Decision | None = None

    def set_flex(self, flex: float, scn: Scenario):
        self.flex = flex
        self.fixed = scn.flex_cost(flex)

    # --- perception (competence) ---------------------------------------------
    def perceive_cost(self, c_prev: float, c_next: float, eps: float) -> float:
        s = self.spec
        return c_prev + s.foresight * (c_next - c_prev) + s.noise * eps

    # --- production rule: recommendation q* ------------------------------------
    def recommend(self, obs: Observation) -> Tuple[float, float]:
        s, q, c_hat = self.spec, self.q, obs.cost_estimate
        if obs.demand_model is not None:
            pm, sl = obs.demand_model
            best = (pm - sl * obs.rivals_output - c_hat) / (2.0 * sl)
        else:
            best = float("nan")          # no demand model: no perceived best reply (the validation rejects Cournot)
        if s.rule == "Cournot":
            rec = self.flex * best + (1.0 - self.flex) * q
        else:  # margin-feedback quantity rule, config "Bertrand" (model-free: uses the observed price only)
            margin = obs.price - c_hat - (self.fixed / q if self.g.margin_includes_fixed else 0.0)
            rec = q + self.flex * (margin - s.desired_margin)
        return max(self.g.q_min, float(np.rint(rec))), best

    # --- selection rule: deviate from default rule B? -------------------------
    def _bin(self, change: float) -> int:
        return int(np.searchsorted(self.bin_edges, change, side="right"))

    def select(self, rec: float) -> bool:
        change = abs(rec - self.q)
        rule = self.spec.selection
        if rule == "Always":
            return True
        if rule == "Never":
            return False
        if rule == "Small":
            return change < self.spec.threshold
        if rule == "Large":
            return change > self.spec.threshold
        if rule == "Adaptive":
            return change > 0 and self.learned_gain[self._bin(change)] >= 0.0
        raise ValueError(rule)

    def decide(self, obs: Observation) -> Decision:
        """Decide from an Observation only (see heiner_abm.information)."""
        rec, best = self.recommend(obs)
        dev = self.select(rec)
        self.last = Decision(rec, self.q, dev, rec if dev else self.q, obs.cost_estimate, best)
        return self.last

    # --- learning: only from released (matured) feedback, see DECISION_SCHEDULE --------------------------------
    def learn_bin(self, b: int, gain_from_rule: float):
        self.learned_gain[b] = self.memory * self.learned_gain[b] + (1 - self.memory) * gain_from_rule


class _History:
    """Market outcomes by period: the recorded arrays, with optional overrides (a forked market's own periods)."""

    def __init__(self, price, quantity, q, profit):
        self.price, self.quantity, self.q, self.profit = price, quantity, q, profit
        self.over = {}

    def set(self, s, P, Q, q, profit):
        self.over[s] = (P, Q, np.asarray(q, float), np.asarray(profit, float))

    def at(self, s):
        if s in self.over:
            return self.over[s]
        return self.price[s], self.quantity[s], self.q[s], self.profit[s]


@dataclass
class PendingFeedback:
    """Feedback about one firm's decision, held until every period it covers has occurred."""
    firm: int
    decided: int          # period of the decision
    matures: int          # last period the feedback covers; released at the end of this period
    bin: int              # size-of-change bin of the decision
    x1: float             # recommended output (the deviation)
    x0: float             # previous output (rule B)
    kind: str             # "estimated" or "oracle"
    gain: float = 0.0     # estimated: accumulated period by period up to maturity; oracle: the researcher's counterfactual
    release: int = 0      # maturity + observation delay: the period at whose end it is released


class Industry:
    """Runs one market populated by heterogeneous firm agents and records everything.

    feedback_log lists every released item as (released_at, firm, decided, matures, bin, gain, kind);
    learned_path[t] is every firm's learned table at the end of period t (after release)."""

    def __init__(self, scn: Scenario, horizon: int = 1, continuation: str = "default", discount: float = 1.0):
        scn.check_runnable()
        self.scn = scn
        self.horizon = max(1, int(horizon))
        self.continuation = continuation
        self.discount = float(discount)
        n, T = scn.n_firms, scn.periods
        self.u, self.eps, self.evo_rng = make_streams(scn.seed, n, T)
        self.market = Market(scn, self.u)
        self.firms = [Firm(i, f, scn) for i, f in enumerate(scn.firms)]
        shape = (T, n)
        self.h = {k: np.zeros(shape) for k in
                  ("q", "rec", "profit", "profit_rule", "profit_default", "c_hat", "flex", "best_reply")}
        self.h["deviate"] = np.zeros(shape, dtype=bool)
        self.h["opportunity"] = np.zeros(shape, dtype=bool)
        self.price = np.zeros(T)
        self.quantity = np.zeros(T)
        self.window_profit = np.zeros(n)
        info = scn.info
        self.delay = int(info.obs_delay)
        self.obs_nz = info.obs_noise * observation_noise(scn.seed, T, n) if info.obs_noise > 0 else None
        self.pending: deque = deque()
        self.feedback_log: List[Tuple[int, int, int, int, int, float, str]] = []
        self.learned_path = np.zeros((T, n, len(scn.adaptive.bin_edges) + 1))
        self._learning = any(f.selection == "Adaptive" for f in scn.firms)

    def run(self) -> "Industry":
        scn, mk, firms = self.scn, self.market, self.firms
        costs = mk.costs
        # period 0: initial production and cost are given
        Q = sum(f.q for f in firms)
        P = mk.clear(Q, 0)
        self.price[0], self.quantity[0] = P, Q
        for i, f in enumerate(firms):
            self.h["q"][0, i] = f.q
            self.h["profit"][0, i] = (P - costs[0]) * f.q - f.fixed
            self.h["flex"][0, i] = f.flex

        actual = _History(self.price, self.quantity, self.h["q"], self.h["profit"])
        for t in range(1, scn.periods):
            decisions = [f.decide(self.observe(i, t, actual, f)) for i, f in enumerate(firms)]
            for f, d in zip(firms, decisions):
                f.q = d.chosen
            Q = sum(f.q for f in firms)
            P = mk.clear(Q, t)
            c = costs[t]
            self.price[t], self.quantity[t] = P, Q
            chosen = [d.chosen for d in decisions]
            hold = self.continuation == "default"
            if self.horizon > 1 and not hold:
                actual_cum = self._branch(t, chosen)
            # 1) evaluate every firm's counterfactual before anyone learns (forks see the same memories)
            evaluated = []
            for i, (f, d) in enumerate(zip(firms, decisions)):
                profit = (P - c) * d.chosen - f.fixed
                alt = d.current if d.deviate else d.recommended
                if self.horizon == 1:
                    p_alt = mk.clearing_price(Q - d.chosen + alt, t)
                    own, profit_alt = profit, (p_alt - c) * alt - f.fixed
                elif hold:   # firm i follows rule B after this decision in both branches
                    own = self._branch(t, chosen, hold_firm=i)[i]
                    profit_alt = self._branch(t, chosen[:i] + [alt] + chosen[i + 1:], hold_firm=i)[i]
                else:        # firm i keeps following its own rules in both branches
                    own = actual_cum[i]
                    profit_alt = self._branch(t, chosen[:i] + [alt] + chosen[i + 1:])[i]
                p_rule, p_def = (own, profit_alt) if d.deviate else (profit_alt, own)
                evaluated.append((profit, p_rule, p_def))
            # 2) record (researcher evaluation of step 3 is stored, never shown to the agents)
            for i, (f, d) in enumerate(zip(firms, decisions)):
                profit, p_rule, p_def = evaluated[i]
                h = self.h
                h["q"][t, i], h["rec"][t, i], h["c_hat"][t, i] = d.chosen, d.recommended, d.c_hat
                h["profit"][t, i], h["profit_rule"][t, i], h["profit_default"][t, i] = profit, p_rule, p_def
                h["deviate"][t, i] = d.deviate
                h["opportunity"][t, i] = d.recommended != d.current
                h["flex"][t, i] = f.flex
                h["best_reply"][t, i] = d.best_reply
                self.window_profit[i] += profit
            # 3) feedback: queue this period's decisions, accumulate, release what has matured
            if self._learning:
                self._queue_feedback(t, decisions, evaluated)
                self._accumulate_observable(t)
                self._release(t)
                self.learned_path[t] = [f.learned_gain for f in firms]
            ev = scn.evolution
            if ev.enabled and t % ev.every == 0:
                self._evolve()
        return self

    # --- observation -------------------------------------------------------------------------------------------
    def observe(self, i: int, t: int, hist: "_History", firm: Optional[Firm] = None) -> Observation:
        """Firm i's Observation for its decision in period t, built from `hist` (actual or forked history) up to the
        latest observed period s = t - 1 - delay. Nothing from after s enters, except the firm's own current output
        and its cost estimate (which includes cost foresight kappa, a competence parameter)."""
        info, mk = self.scn.info, self.market
        firm = firm or self.firms[i]
        s = max(t - 1 - self.delay, 0)
        P, Q, qs, prof = hist.at(s)
        if self.obs_nz is not None:
            P, Q = P + self.obs_nz[0, s, i], Q + self.obs_nz[1, s, i]
        c_hat = firm.perceive_cost(mk.costs[s], mk.costs[t], self.eps[t, i])
        dm = None if info.demand_knowledge == "none" else (mk.belief_p_max[t], mk.belief_slope[t])
        rivals = (tuple((float(qs[j]), float(prof[j])) for j in range(len(qs)) if j != i)
                  if info.rivals_visible else None)
        return Observation(period=t, observed_period=s, own_output=float(firm.q), own_output_then=float(qs[i]),
                           own_payoff=float(prof[i]), price=float(P), market_quantity=float(Q),
                           observed_cost=float(mk.costs[s]), cost_estimate=float(c_hat), demand_model=dm,
                           regime_announced=bool(mk.shift[t]) if info.regime_announced else None, rivals=rivals)

    # --- feedback -----------------------------------------------------------------------------------------------
    def _queue_feedback(self, t: int, decisions: List[Decision], evaluated):
        kind = self.scn.info.feedback
        for i, (f, d) in enumerate(zip(self.firms, decisions)):
            if d.recommended == d.current:
                continue
            item = PendingFeedback(i, t, 0, f._bin(abs(d.recommended - d.current)), float(d.recommended),
                                   float(d.current), kind)
            if kind == "oracle":
                _, p_rule, p_def = evaluated[i]
                item.matures, item.gain = t + self.horizon - 1, p_rule - p_def
            else:
                item.matures = t + int(self.scn.adaptive.window) - 1
            item.release = item.matures + self.delay
            self.pending.append(item)

    def _accumulate_observable(self, t: int):
        """Add period t's observed term to every pending estimated item that has not yet matured (rivals' observed
        output, realized cost, prices on the firm's believed demand curve). Mirrored exactly by engine._observed_term."""
        mk, lo = self.market, self.scn.market.p_min
        pm, sl, c = mk.belief_p_max[t], mk.belief_slope[t], mk.costs[t]
        for item in self.pending:
            if item.kind != "estimated" or item.matures < t:
                continue
            Q = self.quantity[t] + (self.obs_nz[1, t, item.firm] if self.obs_nz is not None else 0.0)
            R = Q - self.h["q"][t, item.firm]
            x1, x0 = item.x1, item.x0
            item.gain = item.gain + ((max(lo, pm - sl * (R + x1)) - c) * x1 - (max(lo, pm - sl * (R + x0)) - c) * x0)

    def _release(self, t: int):
        """Release, in order of decision time, every item whose release period (maturity + delay) is t."""
        keep = deque()
        while self.pending:
            item = self.pending.popleft()
            if item.release == t:
                self.firms[item.firm].learn_bin(item.bin, item.gain)
                self.feedback_log.append((t, item.firm, item.decided, item.matures, item.bin, item.gain, item.kind))
            else:
                keep.append(item)
        self.pending = keep

    def _branch(self, t: int, first_choices: List[float], hold_firm: Optional[int] = None) -> List[float]:
        """Fork the market at period t with the given production choices and let every agent follow
        its rules for the rest of the horizon (learning and evolution frozen); `hold_firm`, if given,
        follows the default rule B instead. Returns each firm's discounted cumulative profit."""
        mk, costs, T = self.market, self.market.costs, self.scn.periods
        firms = copy.deepcopy(self.firms)
        if hold_firm is not None:
            firms[hold_firm].spec.selection = "Never"
        for f, qc in zip(firms, first_choices):
            f.q = qc
        Q = sum(f.q for f in firms)
        P = mk.clearing_price(Q, t)
        hist = _History(self.price, self.quantity, self.h["q"], self.h["profit"])     # actual before t, fork from t
        hist.set(t, P, Q, [f.q for f in firms], [(P - costs[t]) * f.q - f.fixed for f in firms])
        cum = [(P - costs[t]) * f.q - f.fixed for f in firms]
        w = 1.0
        for tt in range(t + 1, min(t + self.horizon, T)):
            w *= self.discount
            ds = [f.decide(self.observe(i, tt, hist, f)) for i, f in enumerate(firms)]
            for f, d in zip(firms, ds):
                f.q = d.chosen
            Q = sum(f.q for f in firms)
            P = mk.clearing_price(Q, tt)
            hist.set(tt, P, Q, [f.q for f in firms], [(P - costs[tt]) * f.q - f.fixed for f in firms])
            for i, f in enumerate(firms):
                cum[i] += w * ((P - costs[tt]) * f.q - f.fixed)
        return cum

    def _evolve(self):
        """Social learning of flexibility: imitate the most profitable rival, then experiment."""
        ev, rng = self.scn.evolution, self.evo_rng
        n = len(self.firms)
        best = int(np.argmax(self.window_profit))
        imitate = rng.random(n) < ev.imitation_prob
        mutation = rng.standard_normal(n) * ev.mutation_sd
        best_flex = self.firms[best].flex
        for i, f in enumerate(self.firms):
            new = best_flex if (imitate[i] and i != best) else f.flex
            hi = min(ev.flex_max, 1.0) if f.spec.rule == "Cournot" else ev.flex_max
            f.set_flex(float(np.clip(new + mutation[i], ev.flex_min, hi)), self.scn)
        self.window_profit[:] = 0.0
