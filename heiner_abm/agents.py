"""Readable agent-based reference implementation.

Each Firm is an agent with its own decision rule, selection rule, competence and
memory. The Market is a single homogeneous-product, market-clearing cobweb market
with a linear hockey-stick demand curve and a reflecting random-walk raw-material
cost. The Industry schedules one round:

    1. each firm observes last period's price P[t-1], market quantity Q[t-1] and
       (its perception of) the raw-material cost;
    2. its production rule (Cournot / Bertrand) recommends q*;
    3. its selection rule decides whether to deviate from the default rule B
       ("keep producing q") and adopt q*;
    4. the market clears at P[t] = max(Pmin, Pmax - s*Q[t]) and the new cost c[t]
       is realised (firms decided before seeing it - this is the CD-gap);
    5. every firm books profit (P[t]-c[t])*q - F and evaluates the counterfactual:
       what it would have earned over the next H periods had it made the other
       choice (the market is forked and every agent keeps following its rules).
       With H = 1 this is the paper's one-period comparison with rivals held fixed.
       This is how Heiner's quantities pi, r, w, G and D are measured.

`heiner_abm.engine` is a vectorised twin of this module used for Monte-Carlo
experiments; tests/test_equivalence.py checks that the two agree exactly.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .params import Scenario, FirmSpec


def make_streams(seed: int, n_firms: int, periods: int):
    """Independent random streams so the cost path depends only on the seed (common random numbers)."""
    cost_ss, noise_ss, evo_ss = np.random.SeedSequence(seed).spawn(3)
    u = np.random.default_rng(cost_ss).uniform(-1.0, 1.0, size=periods)
    eps = np.random.default_rng(noise_ss).standard_normal((periods, n_firms))
    return u, eps, np.random.default_rng(evo_ss)


def reflect(value: float, lo: float, hi: float) -> float:
    """VBA 'reflexive boundary': overshoot beyond a bound is mirrored back inside."""
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


class Market:
    def __init__(self, scn: Scenario, u: np.ndarray):
        self.p = scn.market
        self.slope = scn.market.slope
        self.costs = cost_path(scn, u)
        self.price = 0.0
        self.quantity = 0.0

    def clearing_price(self, quantity: float) -> float:
        return max(self.p.p_min, self.p.p_max - self.slope * quantity)

    def clear(self, quantity: float) -> float:
        self.quantity = quantity
        self.price = self.clearing_price(quantity)
        return self.price


@dataclass
class Decision:
    recommended: float
    current: float
    deviate: bool
    chosen: float
    c_hat: float


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
    def recommend(self, market: Market, prev_price: float, prev_quantity: float, c_hat: float) -> float:
        s, q = self.spec, self.q
        if s.rule == "Cournot":
            best = (market.p.p_max - market.slope * (prev_quantity - q) - c_hat) / (2.0 * market.slope)
            rec = self.flex * best + (1.0 - self.flex) * q
        else:  # Bertrand margin feedback
            margin = prev_price - c_hat - (self.fixed / q if self.g.margin_includes_fixed else 0.0)
            rec = q + self.flex * (margin - s.desired_margin)
        return max(self.g.q_min, float(np.rint(rec)))

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

    def decide(self, market: Market, prev_price, prev_quantity, c_prev, c_next, eps) -> Decision:
        c_hat = self.perceive_cost(c_prev, c_next, eps)
        rec = self.recommend(market, prev_price, prev_quantity, c_hat)
        dev = self.select(rec)
        self.last = Decision(rec, self.q, dev, rec if dev else self.q, c_hat)
        return self.last

    # --- learning from counterfactual payoffs ---------------------------------
    def learn(self, gain_from_rule: float):
        d = self.last
        change = abs(d.recommended - d.current)
        if change > 0:
            b = self._bin(change)
            self.learned_gain[b] = self.memory * self.learned_gain[b] + (1 - self.memory) * gain_from_rule


class Industry:
    """Runs one market populated by heterogeneous firm agents and records everything."""

    def __init__(self, scn: Scenario, horizon: int = 1, continuation: str = "default"):
        errs = scn.validate()
        if errs:
            raise ValueError("; ".join(errs))
        self.scn = scn
        self.horizon = max(1, int(horizon))
        self.continuation = continuation
        n, T = scn.n_firms, scn.periods
        self.u, self.eps, self.evo_rng = make_streams(scn.seed, n, T)
        self.market = Market(scn, self.u)
        self.firms = [Firm(i, f, scn) for i, f in enumerate(scn.firms)]
        shape = (T, n)
        self.h = {k: np.zeros(shape) for k in
                  ("q", "rec", "profit", "profit_rule", "profit_default", "c_hat", "flex")}
        self.h["deviate"] = np.zeros(shape, dtype=bool)
        self.h["opportunity"] = np.zeros(shape, dtype=bool)
        self.price = np.zeros(T)
        self.quantity = np.zeros(T)
        self.window_profit = np.zeros(n)

    def run(self) -> "Industry":
        scn, mk, firms = self.scn, self.market, self.firms
        costs = mk.costs
        # period 0: initial production and cost are given
        Q = sum(f.q for f in firms)
        P = mk.clear(Q)
        self.price[0], self.quantity[0] = P, Q
        for i, f in enumerate(firms):
            self.h["q"][0, i] = f.q
            self.h["profit"][0, i] = (P - costs[0]) * f.q - f.fixed
            self.h["flex"][0, i] = f.flex

        for t in range(1, scn.periods):
            decisions = [f.decide(mk, P, Q, costs[t - 1], costs[t], self.eps[t, i])
                         for i, f in enumerate(firms)]
            for f, d in zip(firms, decisions):
                f.q = d.chosen
            Q = sum(f.q for f in firms)
            P = mk.clear(Q)
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
                    p_alt = mk.clearing_price(Q - d.chosen + alt)
                    own, profit_alt = profit, (p_alt - c) * alt - f.fixed
                elif hold:   # firm i follows rule B after this decision in both branches
                    own = self._branch(t, chosen, hold_firm=i)[i]
                    profit_alt = self._branch(t, chosen[:i] + [alt] + chosen[i + 1:], hold_firm=i)[i]
                else:        # firm i keeps following its own rules in both branches
                    own = actual_cum[i]
                    profit_alt = self._branch(t, chosen[:i] + [alt] + chosen[i + 1:])[i]
                p_rule, p_def = (own, profit_alt) if d.deviate else (profit_alt, own)
                evaluated.append((profit, p_rule, p_def))
            # 2) learn and record
            for i, (f, d) in enumerate(zip(firms, decisions)):
                profit, p_rule, p_def = evaluated[i]
                f.learn(p_rule - p_def)
                h = self.h
                h["q"][t, i], h["rec"][t, i], h["c_hat"][t, i] = d.chosen, d.recommended, d.c_hat
                h["profit"][t, i], h["profit_rule"][t, i], h["profit_default"][t, i] = profit, p_rule, p_def
                h["deviate"][t, i] = d.deviate
                h["opportunity"][t, i] = d.recommended != d.current
                h["flex"][t, i] = f.flex
                self.window_profit[i] += profit
            ev = scn.evolution
            if ev.enabled and t % ev.every == 0:
                self._evolve()
        return self

    def _branch(self, t: int, first_choices: List[float], hold_firm: Optional[int] = None) -> List[float]:
        """Fork the market at period t with the given production choices and let every agent follow
        its rules for the rest of the horizon (learning and evolution frozen); `hold_firm`, if given,
        follows the default rule B instead. Returns each firm's cumulative profit over the horizon."""
        mk, costs, T = self.market, self.market.costs, self.scn.periods
        firms = copy.deepcopy(self.firms)
        if hold_firm is not None:
            firms[hold_firm].spec.selection = "Never"
        for f, qc in zip(firms, first_choices):
            f.q = qc
        Q = sum(f.q for f in firms)
        P = mk.clearing_price(Q)
        cum = [(P - costs[t]) * f.q - f.fixed for f in firms]
        for tt in range(t + 1, min(t + self.horizon, T)):
            ds = [f.decide(mk, P, Q, costs[tt - 1], costs[tt], self.eps[tt, i]) for i, f in enumerate(firms)]
            for f, d in zip(firms, ds):
                f.q = d.chosen
            Q = sum(f.q for f in firms)
            P = mk.clearing_price(Q)
            for i, f in enumerate(firms):
                cum[i] += (P - costs[tt]) * f.q - f.fixed
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
