"""A market that advances one period at a time, so that a person can play one of the firms.

Slot 0 is the human participant; the other slots are agent rivals. Alongside the participant, every candidate design
runs in *shadow* mode on slot 0: each sees exactly what the participant sees (including the participant's actual past
outputs) and records the output it would have chosen. Comparing the participant's choices with these shadow
predictions classifies the participant by the design that predicts them best.

The period logic mirrors arena.simulate; tests/test_validation.py checks that a participant who always plays a given
design's proposal reproduces simulate exactly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .arena import ALL_DESIGNS, P_MIN, Q_MIN, Env, Market, nash_output


class InteractiveMarket:
    def __init__(self, env: Env, rivals: Sequence[str], params: Dict[str, Dict[str, float]],
                 shadows: Sequence[str], periods: int, label: str = ""):
        self.env, self.label, self.periods = env, label, periods
        self.N = 1 + len(rivals)
        mk = self.mk = Market([env], self.N, periods)
        q = np.full((1, self.N), float(nash_output(env.q_range, self.N)))
        Q = q.sum(1)
        P = np.maximum(P_MIN, mk.PM[:, 0] - mk.SL[:, 0] * Q)
        mk.P_prev, mk.Q_prev, mk.q_prev = P, Q, q
        mk.profit_prev = (P - mk.cost[:, 0])[:, None] * q
        mk.R_hist[:, 0] = Q[:, None] - q
        mk.q_hist[:, 0] = q
        idx = np.array([0])

        def make(key, slot):
            cls = ALL_DESIGNS[key]
            return cls(idx, slot, {n: np.array([float(params.get(key, {}).get(n, s.default))])
                                   for n, s in cls.SPACE.items()}, mk)
        self.rivals = [make(k, i + 1) for i, k in enumerate(rivals)]
        self.shadow_keys = list(shadows)
        self.shadows = [make(k, 0) for k in self.shadow_keys]
        self.t = 0
        self.pending: Optional[Dict] = None
        self.log: List[Dict] = []

    @property
    def done(self) -> bool:
        return self.t >= self.periods - 1

    def info(self) -> Dict:
        """What the participant sees before deciding in the next period."""
        if self.pending is None:
            self._prepare()
        mk = self.mk
        return dict(period=self.t + 1, last_price=float(mk.P_prev[0]), last_market_q=float(mk.Q_prev[0]),
                    own_q=float(mk.q_prev[0, 0]), own_profit=float(mk.profit_prev[0, 0]),
                    cost_estimate=float(mk.c_hat[0, 0]), last_cost=float(mk.cost[0, self.t]))

    def _prepare(self):
        mk = self.mk
        t = self.t + 1
        mk.t = t
        c_prev, c_now = mk.cost[:, t - 1], mk.cost[:, t]
        mk.c_hat = (c_prev + mk.kappa * (c_now - c_prev))[:, None] + mk.sigma[:, None] * mk.eps[:, t, :]
        self.pending = dict(rivals=[float(a.act()[0]) for a in self.rivals],
                            shadows={k: float(a.act()[0]) for k, a in zip(self.shadow_keys, self.shadows)})

    def step(self, q_human: float) -> Dict:
        if self.pending is None:
            self._prepare()
        mk, t = self.mk, self.mk.t
        q_prev = float(mk.q_prev[0, 0])
        new = np.array([[q_human] + self.pending["rivals"]], float)
        new = np.maximum(Q_MIN, np.rint(np.nan_to_num(new, nan=Q_MIN, posinf=Q_MIN, neginf=Q_MIN)))
        Q = new.sum(1)
        P = np.maximum(P_MIN, mk.PM[:, t] - mk.SL[:, t] * Q)
        profit = (P - mk.cost[:, t])[:, None] * new
        mk.R_hist[:, t] = Q[:, None] - new
        mk.q_hist[:, t] = new
        row = dict(block=self.label, period=t, delta=self.env.delta, noise=self.env.noise, hazard=self.env.hazard,
                   q_prev=q_prev, q=float(new[0, 0]), deviated=bool(new[0, 0] != q_prev),
                   change=float(new[0, 0] - q_prev), price=float(P[0]), cost=float(mk.cost[0, t]),
                   cost_estimate=float(mk.c_hat[0, 0]), profit=float(profit[0, 0]),
                   rivals_mean_profit=float(profit[0, 1:].mean()),
                   **{f"shadow:{k}": v for k, v in self.pending["shadows"].items()})
        mk.P_prev, mk.Q_prev, mk.q_prev, mk.profit_prev = P, Q, new, profit
        for a in self.rivals + self.shadows:
            a.update()
        self.t = t
        self.pending = None
        self.log.append(row)
        return row

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)
