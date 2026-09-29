"""Vectorised batch engine: simulates B independent markets (scenarios) at once.

Same model and same random streams as `agents.Industry`; used for Monte-Carlo
replications and parameter sweeps. All scenarios in a batch must share the number
of firms, periods, burn-in and counterfactual horizon.

Heiner's reliability bookkeeping (per firm, over recorded periods):
    opportunity  : the production rule recommends a change (q* != q)
    preferred exc: adopting q* pays more than the default rule B ("keep q"), judged by
                   the firm's cumulative profit over a horizon of H periods. The market is
                   forked: one branch with the firm's rule choice, one with the default,
                   every agent following its own rules afterwards, same random shocks.
                   H = 1 is the paper's one-period comparison (rivals' choices fixed).
    pi = P(preferred exception | opportunity)
    r  = P(deviate | preferred exception)          (1 - P(type I error))
    w  = P(deviate | no preferred exception)       (P(type II error))
    G  = mean gain when correctly deviating,  D = mean loss when wrongly deviating
    Reliability condition:  r/w  >  (D/G) * (1-pi)/pi
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .agents import make_streams
from .params import Scenario

SEL_CODES = {"Always": 0, "Never": 1, "Small": 2, "Large": 3, "Adaptive": 4}

ACC_KEYS = ("n_opp", "n_pe", "n_dev", "n_dev_pe", "n_dev_npe", "sum_gain_dev_pe", "sum_loss_dev_npe",
            "sum_gain_pe", "sum_loss_npe", "sum_profit", "sum_q", "sum_absdq", "sum_realized_adv",
            "n_rec", "sum_cerr2")


@dataclass
class BatchResult:
    scenarios: List[Scenario]
    price: np.ndarray            # (B, T)
    cost: np.ndarray             # (B, T)
    quantity: np.ndarray         # (B, T)
    acc: Dict[str, np.ndarray]   # (B, N) accumulators over recorded periods
    flex_final: np.ndarray       # (B, N)
    flex_path: np.ndarray        # (B, T, N) if evolution or histories requested, else (B, 0, N)
    firm_hist: Optional[Dict[str, np.ndarray]] = None  # (B, T, N) arrays when requested
    burn_in: int = 25
    horizon: int = 1
    meta: Dict = field(default_factory=dict)

    @property
    def B(self):
        return self.price.shape[0]

    @property
    def N(self):
        return self.flex_final.shape[1]


class _Kernel:
    """The agents' decision rules and the market-clearing rule, written once for any array layout.

    Market-level arrays have shape L (e.g. (B,) or (B, K)); firm-level arrays have shape L + (N,).
    Parameters are pre-reshaped so they broadcast against those layouts.
    """

    def __init__(self, a: dict, edges: np.ndarray, extra_dims: int):
        m_shape = (-1,) + (1,) * extra_dims            # market params -> broadcast with Q
        f_shape = (a["flex"].shape[0],) + (1,) * extra_dims + (a["flex"].shape[1],)
        self.m = {k: a[k].reshape(m_shape) for k in ("p_max", "p_min", "slope", "q_min", "incl", "memory")}
        self.f = {k: a[k].reshape(f_shape) for k in ("threshold", "margin", "kappa", "sigma", "cournot", "sel")}
        self.edges = edges

    def clearing(self, Q):
        return np.maximum(self.m["p_min"], self.m["p_max"] - self.m["slope"] * Q)

    def clearing_firm(self, Qf):
        """Price for firm-level hypothetical market quantities (shape L + (N,))."""
        return np.maximum(self.m["p_min"][..., None], self.m["p_max"][..., None] - self.m["slope"][..., None] * Qf)

    def decide(self, q, Q, P, F, flex, learned, c_prev, c_now, eps, sel=None):
        m, f = self.m, self.f
        c_hat = c_prev[..., None] + f["kappa"] * (c_now - c_prev)[..., None] + f["sigma"] * eps
        slope = m["slope"][..., None]
        best = (m["p_max"][..., None] - slope * (Q[..., None] - q) - c_hat) / (2 * slope)
        rec_c = flex * best + (1 - flex) * q
        margin = P[..., None] - c_hat - np.where(m["incl"][..., None] > 0, F / q, 0.0)
        rec_b = q + flex * (margin - f["margin"])
        rec = np.maximum(m["q_min"][..., None], np.rint(np.where(f["cournot"], rec_c, rec_b)))
        change = np.abs(rec - q)
        opp = rec != q
        bins = np.searchsorted(self.edges, change, side="right")
        lg = np.take_along_axis(learned, bins[..., None], axis=-1)[..., 0]
        sel = f["sel"] if sel is None else sel
        thr = f["threshold"]
        dev = np.select([sel == 0, sel == 1, sel == 2, sel == 3, sel == 4],
                        [True, False, change < thr, change > thr, opp & (lg >= 0.0)])
        return rec, dev.astype(bool), opp, bins, c_hat


def _stack(scns: Sequence[Scenario]):
    f = lambda getter: np.array([[getter(s, fs) for fs in s.firms] for s in scns], dtype=float)
    m = lambda getter: np.array([getter(s) for s in scns], dtype=float)
    return dict(
        p_max=m(lambda s: s.market.p_max), p_min=m(lambda s: s.market.p_min),
        slope=m(lambda s: s.market.slope), c0=m(lambda s: s.market.c0),
        c_max=m(lambda s: s.market.c_max), delta=m(lambda s: s.market.delta),
        q0=m(lambda s: s.firm_globals.q0), q_min=m(lambda s: s.firm_globals.q_min),
        fc_slope=m(lambda s: s.firm_globals.flex_cost_slope), fc_fixed=m(lambda s: s.firm_globals.fixed_cost),
        incl=m(lambda s: float(s.firm_globals.margin_includes_fixed)),
        memory=m(lambda s: s.adaptive.memory),
        flex=f(lambda s, fs: fs.flex), threshold=f(lambda s, fs: fs.threshold),
        margin=f(lambda s, fs: fs.desired_margin), kappa=f(lambda s, fs: fs.foresight),
        sigma=f(lambda s, fs: fs.noise),
        cournot=f(lambda s, fs: fs.rule == "Cournot").astype(bool),
        sel=f(lambda s, fs: SEL_CODES[fs.selection]).astype(int),
    )


CONTINUATIONS = ("default", "rules")


def run_batch(scenarios: Sequence[Scenario], record_firm_history: bool = False, horizon: int = 1,
              continuation: str = "default") -> BatchResult:
    """Simulate a batch of markets.

    horizon      : periods over which a deviation is evaluated against the default rule B (H >= 1).
    continuation : what the evaluated firm does after the decision inside the counterfactual forks.
        "default" - it returns to rule B (keeps its new level) while rivals follow their rules.
                    This is Heiner's comparison: deviate from B at this one instance, otherwise
                    follow B. Summed over instances it approximates flexible-vs-rigid performance.
        "rules"   - it keeps applying its own production and selection rules (a timing comparison).
    Irrelevant when H = 1.
    """
    if continuation not in CONTINUATIONS:
        raise ValueError(continuation)
    scns = list(scenarios)
    if not scns:
        raise ValueError("empty batch")
    H = max(1, int(horizon))
    N, T, burn = scns[0].n_firms, scns[0].periods, scns[0].burn_in
    for s in scns:
        errs = s.validate()
        if errs:
            raise ValueError("; ".join(errs))
        if s.n_firms != N or s.periods != T or s.burn_in != burn:
            raise ValueError("All scenarios in a batch need the same number of firms, periods and burn-in.")
    edges = np.asarray(scns[0].adaptive.bin_edges, dtype=float)
    if any(tuple(s.adaptive.bin_edges) != tuple(edges) for s in scns):
        raise ValueError("All scenarios in a batch need the same adaptive bin edges.")

    B = len(scns)
    a = _stack(scns)
    col = lambda v: v[:, None]
    K1 = _Kernel(a, edges, 0)          # layout (B,) / (B, N)
    KF = _Kernel(a, edges, 1)          # fork layout (B, K) / (B, K, N)
    hold = continuation == "default"
    NF = 2 * N if hold else N + 1      # number of forks per market
    fork_sel = None
    if hold:
        # forks 2i / 2i+1: firm i makes its actual / alternative choice at t, then follows rule B
        fork_sel = np.repeat(a["sel"][:, None, :], NF, axis=1)
        for i in range(N):
            fork_sel[:, 2 * i:2 * i + 2, i] = SEL_CODES["Never"]

    streams = [make_streams(s.seed, N, T) for s in scns]
    U = np.stack([st[0] for st in streams])                   # (B, T)
    EPS = np.stack([st[1] for st in streams])                 # (B, T, N)
    evo_rngs = [st[2] for st in streams]

    # raw-material cost: random walk with reflecting bounds
    cost = np.empty((B, T))
    cost[:, 0] = a["c0"]
    lo, hi = a["p_min"], a["c_max"]
    for t in range(1, T):
        v = cost[:, t - 1] + a["delta"] * U[:, t]
        v = np.where(v <= lo, 2 * lo - v, v)
        v = np.where(v >= hi, 2 * hi - v, v)
        cost[:, t] = v

    flex = a["flex"].copy()
    evolve = [s.evolution for s in scns]
    evo_on = any(e.enabled for e in evolve)
    F = col(a["fc_slope"]) * flex + col(a["fc_fixed"])
    q = np.repeat(col(a["q0"]), N, axis=1).astype(float)
    Q = q.sum(1)
    P = K1.clearing(Q)
    price = np.empty((B, T)); quantity = np.empty((B, T))
    price[:, 0], quantity[:, 0] = P, Q

    acc = {k: np.zeros((B, N)) for k in ACC_KEYS}
    learned = np.zeros((B, N, len(edges) + 1))
    window = np.zeros((B, N))
    keep_path = evo_on or record_firm_history
    flex_path = np.zeros((B, T if keep_path else 0, N))
    if keep_path:
        flex_path[:, 0] = flex
    hist = None
    if record_firm_history:
        hist = {k: np.zeros((B, T, N)) for k in ("q", "rec", "profit", "profit_rule", "profit_default", "c_hat")}
        hist["deviate"] = np.zeros((B, T, N), dtype=bool)
        hist["opportunity"] = np.zeros((B, T, N), dtype=bool)
        hist["q"][:, 0] = q
        hist["profit"][:, 0] = col(P - cost[:, 0]) * q - F

    b_idx = np.arange(B)[:, None]; n_idx = np.arange(N)[None, :]
    mem = col(a["memory"])
    any_adaptive = bool((a["sel"] == SEL_CODES["Adaptive"]).any())
    eye = np.eye(N, dtype=bool)

    for t in range(1, T):
        c_prev, c_now = cost[:, t - 1], cost[:, t]
        rec, dev, opp, bins, c_hat = K1.decide(q, Q, P, F, flex, learned, c_prev, c_now, EPS[:, t, :])
        new_q = np.where(dev, rec, q)
        alt = np.where(dev, q, rec)
        Qn = new_q.sum(1)
        Pn = K1.clearing(Qn)
        profit = col(Pn - c_now) * new_q - F

        need_cf = t >= burn or any_adaptive or record_firm_history
        if H == 1 or not need_cf:
            # one-period counterfactual: rivals' period-t choices held fixed
            p_alt = K1.clearing_firm(col(Qn) - new_q + alt)
            cum_actual = profit
            cum_alt = (p_alt - col(c_now)) * alt - F
        else:
            # fork the market. hold : forks (2i, 2i+1) = firm i actual / alternative choice at t, then B
            #                  rules: fork 0 = actual choices, fork i+1 = firm i alternative choice at t
            fq = np.repeat(new_q[:, None, :], NF, axis=1)
            if hold:
                fq[:, 1::2, :] = np.where(eye[None], alt[:, None, :], fq[:, 1::2, :])
            else:
                fq[:, 1:, :] = np.where(eye[None], alt[:, None, :], fq[:, 1:, :])
            fF = np.broadcast_to(F[:, None, :], fq.shape)
            fflex = np.broadcast_to(flex[:, None, :], fq.shape)
            flearn = np.broadcast_to(learned[:, None], (B, NF) + learned.shape[1:])
            fQ = fq.sum(-1)
            fP = KF.clearing(fQ)
            cum = (fP - col(c_now))[..., None] * fq - fF
            for h in range(1, H):
                tt = t + h
                if tt >= T:
                    break
                cp, cn = col(cost[:, tt - 1]), col(cost[:, tt])
                r2, d2, *_ = KF.decide(fq, fQ, fP, fF, fflex, flearn, cp, cn, EPS[:, tt, None, :], sel=fork_sel)
                fq = np.where(d2, r2, fq)
                fQ = fq.sum(-1)
                fP = KF.clearing(fQ)
                cum = cum + (fP - cn)[..., None] * fq - fF
            if hold:
                cum_actual = cum[:, 0::2, :][:, eye]
                cum_alt = cum[:, 1::2, :][:, eye]
            else:
                cum_actual = cum[:, 0, :]
                cum_alt = cum[:, 1:, :][:, eye]      # firm i's own profit in its own alternative branch
        p_rule = np.where(dev, cum_actual, cum_alt)
        p_def = np.where(dev, cum_alt, cum_actual)
        gain = p_rule - p_def

        # adaptive learning (all periods, only where a change was recommended)
        cur = learned[b_idx, n_idx, bins]
        learned[b_idx, n_idx, bins] = np.where(opp, mem * cur + (1 - mem) * gain, cur)

        if t >= burn:
            pe = opp & (gain > 0)
            npe = opp & ~pe
            acc["n_rec"] += 1
            acc["n_opp"] += opp
            acc["n_pe"] += pe
            acc["n_dev"] += dev & opp
            acc["n_dev_pe"] += dev & pe
            acc["n_dev_npe"] += dev & npe
            acc["sum_gain_dev_pe"] += np.where(dev & pe, gain, 0.0)
            acc["sum_loss_dev_npe"] += np.where(dev & npe, -gain, 0.0)
            acc["sum_gain_pe"] += np.where(pe, gain, 0.0)
            acc["sum_loss_npe"] += np.where(npe, -gain, 0.0)
            acc["sum_profit"] += profit
            acc["sum_q"] += new_q
            acc["sum_absdq"] += np.abs(new_q - q)
            acc["sum_realized_adv"] += np.where(dev, gain, 0.0)
            acc["sum_cerr2"] += (c_hat - col(c_now)) ** 2      # perception error -> empirical CD-gap
        if record_firm_history:
            hist["q"][:, t], hist["rec"][:, t], hist["c_hat"][:, t] = new_q, rec, c_hat
            hist["profit"][:, t], hist["profit_rule"][:, t], hist["profit_default"][:, t] = profit, p_rule, p_def
            hist["deviate"][:, t], hist["opportunity"][:, t] = dev, opp

        q, Q, P = new_q, Qn, Pn
        price[:, t], quantity[:, t] = P, Q
        window += profit
        if keep_path:
            flex_path[:, t] = flex
        if evo_on:
            changed = False
            for b, ev in enumerate(evolve):
                if ev.enabled and t % ev.every == 0:
                    rng = evo_rngs[b]
                    best_i = int(np.argmax(window[b]))
                    imitate = rng.random(N) < ev.imitation_prob
                    mutation = rng.standard_normal(N) * ev.mutation_sd
                    new = np.where(imitate & (np.arange(N) != best_i), flex[b, best_i], flex[b]) + mutation
                    hi_b = np.where(a["cournot"][b], min(ev.flex_max, 1.0), ev.flex_max)
                    flex[b] = np.clip(new, ev.flex_min, hi_b)
                    window[b] = 0.0
                    changed = True
            if changed:
                F = col(a["fc_slope"]) * flex + col(a["fc_fixed"])

    return BatchResult(scenarios=scns, price=price, cost=cost, quantity=quantity, acc=acc,
                       flex_final=flex.copy(), flex_path=flex_path, firm_hist=hist, burn_in=burn, horizon=H,
                       meta=dict(continuation=continuation))


def run_one(scn: Scenario, horizon: int = 1, continuation: str = "default") -> BatchResult:
    return run_batch([scn], record_firm_history=True, horizon=horizon, continuation=continuation)
