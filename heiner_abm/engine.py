"""Vectorized batch engine: simulates B independent markets (scenarios) at once.

Same model and same random streams as `agents.Industry`; used for Monte-Carlo
replications and parameter sweeps. All scenarios in a batch must share the number
of firms, periods and burn-in.

Heiner's reliability bookkeeping (per firm):
    opportunity  : the production rule recommends a change (q* != q)
    preferred exc: adopting q* pays more than the default rule B ("keep q")
    pi = P(preferred exception | opportunity)
    r  = P(deviate | preferred exception)          (1 - P(type I error))
    w  = P(deviate | no preferred exception)       (P(type II error))
    G  = mean gain when correctly deviating,  D = mean loss when wrongly deviating
    Reliability condition:  r/w  >  (D/G) * (1-pi)/pi

Three *measures* of the gain from following the rule instead of B (the dynamic RC):
    "static"  : one period, rivals' choices held fixed (Heiner's one-shot comparison; H = 1)
    "persist" : H periods, the firm keeps the chosen level (rule B afterwards), rivals follow
                their *actual* path (no reaction to the deviation) -> adds persistence
    "full"    : H periods in a forked market where rivals react (continuation "default"), or
                where the firm also keeps applying its own rules (continuation "rules")
    strategic feedback = full - persist,  persistence effect = persist - static.
Three *windows*: "all" recorded periods, "est" (first part) and "eval" (second part), so that
the RC can be estimated in one window and used to predict performance in the other. For the measures that look H
periods ahead ("persist" and "full"), the estimation window ends H - 1 periods before the evaluation window starts:
a decision's measured gain covers periods t .. t + H - 1, so only decisions whose horizon ends before the split are
used to estimate. Estimates therefore use nothing from the evaluation window. The "static" measure (one period) uses
the whole first part.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agents import belief_curves, demand_path, make_streams, observation_noise
from .params import Scenario

SEL_CODES = {"Always": 0, "Never": 1, "Small": 2, "Large": 3, "Adaptive": 4}
MEASURES = ("static", "persist", "full")
WINDOWS = ("all", "est", "eval")
CONTINUATIONS = ("default", "rules")

ACC_KEYS = ("n_opp", "n_pe", "n_dev", "n_dev_pe", "n_dev_npe", "sum_gain_dev_pe", "sum_loss_dev_npe",
            "sum_gain_pe", "sum_loss_npe", "sum_profit", "sum_q", "sum_absdq", "sum_realized_adv",
            "n_rec", "sum_cerr2", "sum_brerr2", "sum_brsig2", "n_br")


@dataclass
class BatchResult:
    scenarios: List[Scenario]
    price: np.ndarray            # (B, T)
    cost: np.ndarray             # (B, T)
    quantity: np.ndarray         # (B, T)
    acc: Dict[str, np.ndarray]   # (B, N) accumulators: measure "full", window "all"
    flex_final: np.ndarray       # (B, N)
    flex_path: np.ndarray        # (B, T, N) if evolution or histories requested, else (B, 0, N)
    firm_hist: Optional[Dict[str, np.ndarray]] = None  # (B, T, N) arrays when requested
    burn_in: int = 25
    horizon: int = 1
    meta: Dict = field(default_factory=dict)
    accs: Dict[Tuple[str, str], Dict[str, np.ndarray]] = field(default_factory=dict)
    p_max_path: Optional[np.ndarray] = None   # (B, T) true demand intercept
    slope_path: Optional[np.ndarray] = None   # (B, T) true demand slope
    shifts: Optional[np.ndarray] = None       # (B, T) regime-shift indicator
    steps: Optional[Dict[str, np.ndarray]] = None  # per-step arrays when keep_steps=True

    @property
    def B(self):
        return self.price.shape[0]

    @property
    def N(self):
        return self.flex_final.shape[1]


class _Kernel:
    """The agents' decision rules and the market-clearing rule, written once for any array layout.

    Market-level arrays have shape L (e.g. (B,) or (B, K)); firm-level arrays have shape L + (N,).
    Demand parameters (true for clearing, believed for Cournot best replies) are passed per call
    because they change with regime shifts.
    """

    def __init__(self, a: dict, edges: np.ndarray, extra_dims: int):
        m_shape = (-1,) + (1,) * extra_dims
        f_shape = (a["flex"].shape[0],) + (1,) * extra_dims + (a["flex"].shape[1],)
        self.m = {k: a[k].reshape(m_shape) for k in ("p_min", "q_min", "incl", "no_dm")}
        self.f = {k: a[k].reshape(f_shape) for k in ("threshold", "margin", "kappa", "sigma", "cournot", "sel")}
        self.edges = edges

    def clearing(self, Q, pm, sl):
        return np.maximum(self.m["p_min"], pm - sl * Q)

    def clearing_firm(self, Qf, pm, sl):
        return np.maximum(self.m["p_min"][..., None], pm[..., None] - sl[..., None] * Qf)

    def decide(self, q, R, P, F, flex, learned, c_obs, c_now, eps, pmb, slb, sel=None):
        """Decisions from observations (agents.Industry.observe): q own current output, R rivals' observed output and
        P observed price per firm, c_obs the observed cost per market, (pmb, slb) the demand curve firms may use."""
        m, f = self.m, self.f
        c_hat = c_obs[..., None] + f["kappa"] * (c_now - c_obs)[..., None] + f["sigma"] * eps
        s = slb[..., None]
        best = (pmb[..., None] - s * R - c_hat) / (2 * s)
        best = np.where(self.m["no_dm"][..., None], np.nan, best)
        rec_c = flex * best + (1 - flex) * q
        margin = P - c_hat - np.where(m["incl"][..., None] > 0, F / q, 0.0)
        rec_b = q + flex * (margin - f["margin"])
        with np.errstate(invalid="ignore"):
            rec = np.maximum(m["q_min"][..., None], np.rint(np.where(f["cournot"], rec_c, rec_b)))
        change = np.abs(rec - q)
        opp = rec != q
        bins = np.searchsorted(self.edges, change, side="right")
        lg = np.take_along_axis(learned, bins[..., None], axis=-1)[..., 0]
        sel = f["sel"] if sel is None else sel
        thr = f["threshold"]
        dev = np.select([sel == 0, sel == 1, sel == 2, sel == 3, sel == 4],
                        [True, False, change < thr, change > thr, opp & (lg >= 0.0)])
        return rec, dev.astype(bool), opp, bins, c_hat, best


def _stack(scns: Sequence[Scenario]):
    f = lambda getter: np.array([[getter(s, fs) for fs in s.firms] for s in scns], dtype=float)
    m = lambda getter: np.array([getter(s) for s in scns], dtype=float)
    return dict(
        p_min=m(lambda s: s.market.p_min), c0=m(lambda s: s.market.c0),
        c_max=m(lambda s: s.market.c_max), delta=m(lambda s: s.market.delta),
        q0=m(lambda s: s.firm_globals.q0), q_min=m(lambda s: s.firm_globals.q_min),
        fc_slope=m(lambda s: s.firm_globals.flex_cost_slope), fc_fixed=m(lambda s: s.firm_globals.fixed_cost),
        incl=m(lambda s: float(s.firm_globals.margin_includes_fixed)),
        memory=m(lambda s: s.adaptive.memory),
        oracle_fb=m(lambda s: float(s.info.feedback == "oracle")).astype(bool),
        delay=m(lambda s: int(s.info.obs_delay)).astype(int), obs_noise=m(lambda s: s.info.obs_noise),
        no_dm=m(lambda s: float(s.info.demand_knowledge == "none")).astype(bool),
        window=m(lambda s: int(s.adaptive.window)).astype(int),
        flex=f(lambda s, fs: fs.flex), threshold=f(lambda s, fs: fs.threshold),
        margin=f(lambda s, fs: fs.desired_margin), kappa=f(lambda s, fs: fs.foresight),
        sigma=f(lambda s, fs: fs.noise),
        cournot=f(lambda s, fs: fs.rule == "Cournot").astype(bool),
        sel=f(lambda s, fs: SEL_CODES[fs.selection]).astype(int),
    )


def _accumulate(st: Dict[str, np.ndarray], gain: np.ndarray, mask_t: np.ndarray) -> Dict[str, np.ndarray]:
    """Sum Heiner's bookkeeping over the periods selected by mask_t (T,), for one gain measure."""
    w = mask_t[None, :, None]
    opp = st["opp"] & w
    dev = st["dev"]
    pe = opp & (gain > 0)
    npe = opp & ~pe
    g = gain.astype(float)
    S = lambda x: x.sum(axis=1, dtype=float)
    return {
        "n_rec": np.broadcast_to(float(mask_t.sum()), opp.shape[::2]).copy(),
        "n_opp": S(opp), "n_pe": S(pe), "n_dev": S(dev & opp), "n_dev_pe": S(dev & pe),
        "n_dev_npe": S(dev & npe),
        "sum_gain_dev_pe": S(np.where(dev & pe, g, 0.0)), "sum_loss_dev_npe": S(np.where(dev & npe, -g, 0.0)),
        "sum_gain_pe": S(np.where(pe, g, 0.0)), "sum_loss_npe": S(np.where(npe, -g, 0.0)),
        "sum_realized_adv": S(np.where(dev & w, g, 0.0)),
        "sum_profit": S(np.where(w, st["profit"], 0.0)), "sum_q": S(np.where(w, st["q"], 0.0)),
        "sum_absdq": S(np.where(w, np.abs(st["q"] - st["q_prev"]), 0.0)),
        "sum_cerr2": S(np.where(w, st["cerr2"], 0.0)),
        # Heiner (1989): decision error xi = perceived target - true target; signal = true target - q_prev
        "sum_brerr2": S(np.where(w, (st["br_perc"] - st["br_true"]) ** 2, 0.0)),
        "sum_brsig2": S(np.where(w, (st["br_true"] - st["q_prev"]) ** 2, 0.0)),
        "n_br": S(np.broadcast_to(w, opp.shape)),
    }


def run_batch(scenarios: Sequence[Scenario], record_firm_history: bool = False, horizon: int = 1,
              continuation: str = "default", discount: float = 1.0, oos_split: float = 0.5,
              keep_steps: bool = False) -> BatchResult:
    """Simulate a batch of markets.

    horizon      : periods over which a deviation is evaluated against the default rule B (H >= 1).
    continuation : what the evaluated firm does after the decision inside the "full" forks.
        "default" - it returns to rule B (keeps its new level) while rivals follow their rules.
                    Heiner's comparison: deviate from B at this one instance, otherwise follow B.
        "rules"   - it keeps applying its own production and selection rules (a timing comparison).
    discount     : weight gamma^h on the h-th period of the horizon.
    oos_split    : share of the recorded periods in the estimation window (rest = evaluation window).
    keep_steps   : return per-step arrays (for event studies and signal-detection analysis).
    """
    if continuation not in CONTINUATIONS:
        raise ValueError(continuation)
    scns = list(scenarios)
    if not scns:
        raise ValueError("empty batch")
    H = max(1, int(horizon))
    gam = float(discount)
    N, T, burn = scns[0].n_firms, scns[0].periods, scns[0].burn_in
    for s in scns:
        s.check_runnable()
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

    # demand regimes (true) and the firms' believed demand model
    dp = [demand_path(s) for s in scns]
    PM = np.stack([d[0] for d in dp]); SL = np.stack([d[1] for d in dp]); SHIFT = np.stack([d[2] for d in dp])
    bc = [belief_curves(s, PM[b], SL[b]) for b, s in enumerate(scns)]     # the curve firms may use (information spec)
    PMB = np.stack([x[0] for x in bc]); SLB = np.stack([x[1] for x in bc])
    # observations: delay d per market, noise on observed price and market quantity (separate stream)
    D = a["delay"]
    NZ = (np.stack([a["obs_noise"][b] * observation_noise(s.seed, T, N) for b, s in enumerate(scns)])
          if (a["obs_noise"] > 0).any() else None)                                       # (B, 2, T, N)
    bB = np.arange(B)

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
    P = K1.clearing(Q, PM[:, 0], SL[:, 0])
    price = np.empty((B, T)); quantity = np.empty((B, T))
    price[:, 0], quantity[:, 0] = P, Q

    # per-step arrays (float32 to save memory)
    f32 = lambda: np.zeros((B, T, N), dtype=np.float32)   # integers / diagnostics: float32 is exact enough
    f64 = lambda: np.zeros((B, T, N))                      # payoffs: full precision
    st = {k: f32() for k in ("q", "q_prev", "rec", "cerr2", "br_perc", "br_true")}
    st.update({k: f64() for k in ("profit", "g_static", "g_full")})
    st["dev"] = np.zeros((B, T, N), dtype=bool)
    st["opp"] = np.zeros((B, T, N), dtype=bool)
    st["q"][:, 0] = q
    st["q_prev"][:, 0] = q
    st["profit"][:, 0] = col(P - cost[:, 0]) * q - F

    learned = np.zeros((B, N, len(edges) + 1))
    window = np.zeros((B, N))
    keep_path = evo_on or record_firm_history
    flex_path = np.zeros((B, T if keep_path else 0, N))
    if keep_path:
        flex_path[:, 0] = flex
    b_idx = np.arange(B)[:, None]; n_idx = np.arange(N)[None, :]
    mem = col(a["memory"])
    # Adaptive feedback (agents.DECISION_SCHEDULE): queued at decision time, released only at maturity
    is_adaptive = (a["sel"] == SEL_CODES["Adaptive"]).any(axis=1)                # (B,)
    oracle_rows = np.flatnonzero(is_adaptive & a["oracle_fb"])                   # researcher-only treatment
    # groups of markets that share a judgement window (estimated) or measurement horizon (oracle) and a delay
    est_groups = {(int(w), int(d)): np.flatnonzero(is_adaptive & ~a["oracle_fb"] & (a["window"] == w) & (D == d))
                  for w, d in {(a["window"][b], D[b]) for b in np.flatnonzero(is_adaptive & ~a["oracle_fb"])}}
    oracle_groups = {int(d): np.flatnonzero(is_adaptive & a["oracle_fb"] & (D == d))
                     for d in {D[b] for b in oracle_rows}}
    pending: deque = deque()
    pmin_b = a["p_min"]
    eye = np.eye(N, dtype=bool)
    hist_full = {}

    for t in range(1, T):
        c_now = cost[:, t]
        pm, sl = PM[:, t], SL[:, t]
        so = np.maximum(t - 1 - D, 0)                                   # latest observed period per market
        P_obs = np.repeat(price[bB, so][:, None], N, axis=1)
        Q_obs = np.repeat(quantity[bB, so][:, None], N, axis=1)
        if NZ is not None:
            P_obs, Q_obs = P_obs + NZ[bB, 0, so], Q_obs + NZ[bB, 1, so]
        R_obs = Q_obs - st["q"][bB, so].astype(float)
        rec, dev, opp, bins, c_hat, best = K1.decide(q, R_obs, P_obs, F, flex, learned, cost[bB, so], c_now,
                                                     EPS[:, t, :], PMB[:, t], SLB[:, t])
        new_q = np.where(dev, rec, q)
        alt = np.where(dev, q, rec)
        Qn = new_q.sum(1)
        Pn = K1.clearing(Qn, pm, sl)
        profit = col(Pn - c_now) * new_q - F
        # one-period counterfactual: rivals' period-t choices held fixed
        p_alt = K1.clearing_firm(col(Qn) - new_q + alt, pm, sl)
        prof_alt1 = (p_alt - col(c_now)) * alt - F
        g_static = np.where(dev, profit - prof_alt1, prof_alt1 - profit)

        need_cf = t >= burn or len(oracle_rows) > 0 or record_firm_history
        if H == 1 or not need_cf:
            g_full = g_static
            cum_actual, cum_alt = profit, prof_alt1
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
            fP = KF.clearing(fQ, col(pm), col(sl))
            cum = (fP - col(c_now))[..., None] * fq - fF
            # fork history (periods t, t+1, ...) for delayed observations inside the forks
            hP, hQ, hq = np.empty((B, NF, H)), np.empty((B, NF, H)), np.empty((B, NF, H, N))
            hP[:, :, 0], hQ[:, :, 0], hq[:, :, 0] = fP, fQ, fq
            wgt = 1.0
            for h in range(1, H):
                tt = t + h
                if tt >= T:
                    break
                wgt *= gam
                cn = col(cost[:, tt])
                fo = np.maximum(tt - 1 - D, 0)                          # observed period per market
                k = fo - t                                              # >= 0: inside the fork
                kk = np.clip(k, 0, H - 1)
                inside = (k >= 0)[:, None]
                oP = np.where(inside, hP[bB, :, kk], price[bB, fo][:, None])
                oQ = np.where(inside, hQ[bB, :, kk], quantity[bB, fo][:, None])
                oq = np.where(inside[..., None], hq[bB, :, kk], st["q"][bB, fo][:, None, :].astype(float))
                oP = np.repeat(oP[..., None], N, axis=-1)
                oQ = np.repeat(oQ[..., None], N, axis=-1)
                if NZ is not None:
                    oP, oQ = oP + NZ[bB, 0, fo][:, None, :], oQ + NZ[bB, 1, fo][:, None, :]
                r2, d2, *_ = KF.decide(fq, oQ - oq, oP, fF, fflex, flearn, col(cost[bB, fo]), cn,
                                       EPS[:, tt, None, :], col(PMB[:, tt]), col(SLB[:, tt]), sel=fork_sel)
                fq = np.where(d2, r2, fq)
                fQ = fq.sum(-1)
                fP = KF.clearing(fQ, col(PM[:, tt]), col(SL[:, tt]))
                hP[:, :, h], hQ[:, :, h], hq[:, :, h] = fP, fQ, fq
                cum = cum + wgt * ((fP - cn)[..., None] * fq - fF)
            if hold:
                cum_actual = cum[:, 0::2, :][:, eye]
                cum_alt = cum[:, 1::2, :][:, eye]
            else:
                cum_actual = cum[:, 0, :]
                cum_alt = cum[:, 1:, :][:, eye]      # firm i's own profit in its own alternative branch
            g_full = np.where(dev, cum_actual - cum_alt, cum_alt - cum_actual)

        # Heiner (1989) true target: ex-post best reply to rivals' actual output, realized cost, true demand
        br_true = (col(pm) - col(sl) * (col(Qn) - new_q) - col(c_now)) / (2 * col(sl))
        st["q"][:, t], st["q_prev"][:, t], st["rec"][:, t], st["profit"][:, t] = new_q, q, rec, profit
        st["g_static"][:, t], st["g_full"][:, t] = g_static, g_full
        st["dev"][:, t], st["opp"][:, t] = dev, opp
        st["cerr2"][:, t] = (c_hat - col(c_now)) ** 2
        st["br_perc"][:, t], st["br_true"][:, t] = best, br_true
        if record_firm_history:
            p_rule = np.where(dev, cum_actual, cum_alt)
            p_def = np.where(dev, cum_alt, cum_actual)
            for k, val in (("profit_rule", p_rule), ("profit_default", p_def), ("c_hat", c_hat)):
                hist_full.setdefault(k, np.zeros((B, T, N)))[:, t] = val

        # feedback: queue this period's decisions, accumulate observed terms, release what matures at t
        for (w, d), rows in est_groups.items():
            pending.append(dict(kind="estimated", rows=rows, decided=t, matures=t + w - 1, release=t + w - 1 + d,
                                bins=bins[rows], opp=opp[rows], x1=rec[rows], x0=q[rows],
                                gain=np.zeros((len(rows), N))))
        for d, rows in oracle_groups.items():
            pending.append(dict(kind="oracle", rows=rows, decided=t, matures=t + H - 1, release=t + H - 1 + d,
                                bins=bins[rows], opp=opp[rows], gain=g_full[rows]))
        for item in pending:
            if item["kind"] == "estimated" and item["matures"] >= t:
                r = item["rows"]
                Qo = Qn[r, None] + (NZ[r, 1, t] if NZ is not None else 0.0)     # observed market output
                item["gain"] = item["gain"] + _observed_term(item["x1"], item["x0"], Qo - new_q[r],
                                                             PMB[r, t, None], SLB[r, t, None], cost[r, t, None],
                                                             pmin_b[r, None])
        due = [item for item in pending if item["release"] == t]     # in order of decision time
        pending = deque(item for item in pending if item["release"] != t)
        for item in due:
            r = item["rows"]
            cur = learned[r[:, None], n_idx, item["bins"]]
            m_r = mem[r]
            learned[r[:, None], n_idx, item["bins"]] = np.where(item["opp"], m_r * cur + (1 - m_r) * item["gain"], cur)

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

    # ---- persistence measure: firm holds the chosen level for H periods, rivals follow their actual path
    g_persist = st["g_static"] if H == 1 else _persistence_gain(st, quantity, cost, PM, SL, a, H, gam)

    # ---- accumulate every measure in every window
    t_idx = np.arange(T)
    split_t = burn + int(round((T - burn) * float(oos_split)))
    est_end = {m: split_t - (0 if m == "static" else H - 1) for m in MEASURES}     # gap of H - 1 periods
    masks = {(m, "all"): t_idx >= burn for m in MEASURES}
    masks.update({(m, "est"): (t_idx >= burn) & (t_idx < est_end[m]) for m in MEASURES})
    masks.update({(m, "eval"): t_idx >= split_t for m in MEASURES})
    gains = {"static": st["g_static"], "persist": g_persist, "full": st["g_full"]}
    accs = {(m, w): _accumulate(st, gains[m], masks[(m, w)]) for m in MEASURES for w in WINDOWS}

    hist = None
    if record_firm_history:
        hist = {"q": st["q"].astype(float), "rec": st["rec"].astype(float), "profit": st["profit"].astype(float),
                "deviate": st["dev"].copy(), "opportunity": st["opp"].copy(),
                "gain_static": st["g_static"].astype(float), "gain_persist": np.asarray(g_persist, float),
                "gain_full": st["g_full"].astype(float), "best_reply": st["br_perc"].astype(float),
                "true_best_reply": st["br_true"].astype(float), **hist_full}
    steps = None
    if keep_steps:
        steps = {**st, "g_persist": np.asarray(g_persist, dtype=np.float32)}

    return BatchResult(scenarios=scns, price=price, cost=cost, quantity=quantity, acc=accs[("full", "all")],
                       flex_final=flex.copy(), flex_path=flex_path, firm_hist=hist, burn_in=burn, horizon=H,
                       meta=dict(continuation=continuation, discount=gam, split_t=split_t, est_end=est_end,
                                 info=[s.info.to_dict() for s in scns]),
                       accs=accs, p_max_path=PM, slope_path=SL, shifts=SHIFT, steps=steps)


def _observed_term(x1, x0, R, pm, sl, c, lo):
    """One period of what an Adaptive agent observes about a pending decision: profit from holding the recommended
    level x1 minus profit from holding its old level x0 (rule B), with rivals' actual output R, the realized cost c
    and prices on its believed demand curve (pm, sl). Same arithmetic as agents.Industry._accumulate_observable."""
    pa = np.maximum(lo, pm - sl * (R + x1))
    pb = np.maximum(lo, pm - sl * (R + x0))
    return (pa - c) * x1 - (pb - c) * x0


def _persistence_gain(st, quantity, cost, PM, SL, a, H, gam):
    """Gain of adopting q* at t and then holding it (rule B) versus holding q_prev, for H periods,
    with rivals following their actual path and no reaction to the deviation."""
    B, T, N = st["q"].shape
    qa = st["rec"].astype(float)       # branch 'rule'
    qb = st["q_prev"].astype(float)    # branch 'default'
    qact = st["q"].astype(float)
    pmin = a["p_min"][:, None, None]
    out = np.zeros((B, T, N))
    for h in range(H):
        n = T - h
        if n <= 1:
            break
        riv = quantity[:, h:, None] - qact[:, h:]          # rivals' actual output at t+h, (B, n, N)
        pm = PM[:, h:, None]; sl = SL[:, h:, None]; c = cost[:, h:, None]
        pa = np.maximum(pmin, pm - sl * (riv + qa[:, :n]))
        pb = np.maximum(pmin, pm - sl * (riv + qb[:, :n]))
        out[:, :n] += (gam ** h) * ((pa - c) * qa[:, :n] - (pb - c) * qb[:, :n])
    out[:, 0] = 0.0
    return out


def run_one(scn: Scenario, horizon: int = 1, continuation: str = "default", discount: float = 1.0,
            oos_split: float = 0.5) -> BatchResult:
    return run_batch([scn], record_firm_history=True, horizon=horizon, continuation=continuation,
                     discount=discount, oos_split=oos_split, keep_steps=True)
