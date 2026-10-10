"""Decision-level traces of tournament agents: what each agent observed, predicted and did, and when feedback arrived.

trace_markets follows arena.simulate step for step (the same Market, agents and random streams) and records one row per
market, period and firm. tests/test_research.py checks that its profits and change rates equal simulate's exactly, so
the trace describes the computation that produced the reported results.

Columns
    obs_*          information available to the agent at its decision in period t (from period t − 1, plus the cost
                   estimate for period t of the quality its environment gives every agent)
    pred_target    the output the agent's target rule aims at (composite designs; empty for hand-written rules)
    action_output  the output chosen (after rounding and the production floor)
    feedback_*     own profit of period t, which the agent first sees at its decision in period t + 1; for
                   reliability-condition agents, the h-period judgement of this decision, which is complete after
                   period t + h − 1 and first used at the decision in period t + h
    researcher_*   quantities never shown to any agent (true cost, true demand curve, ex-post best reply)
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .. import arena as A

FEEDBACK = {
    "own_profit": "own profit of period t, observed at the decision in period t + 1",
    "market": "price and market quantity of period t, observed at the decision in period t + 1",
    "rivals": "imitators also observe each rival's output and profit of period t at the decision in period t + 1",
    "rc_judgement": ("reliability-condition agents judge the decision of period d after periods d … d + h − 1 have "
                     "been played (released at the end of period d + h − 1)"),
}


def trace_markets(envs: Sequence[A.Env], lineup: np.ndarray, params: Dict[str, Dict], periods: int,
                  burn_in: int) -> Dict[str, object]:
    """Run the markets exactly as arena.simulate and return its outputs plus a decision trace (key 'trace')."""
    lineup = np.asarray(lineup, dtype=object)
    B, N = lineup.shape
    mk = A.Market(envs, N, periods)
    q0 = np.array([A.nash_output(e.q_range, N) for e in envs], float)
    q = np.repeat(q0[:, None], N, axis=1)
    Q = q.sum(1)
    P = np.maximum(A.P_MIN, mk.PM[:, 0] - mk.SL[:, 0] * Q)
    mk.P_prev, mk.Q_prev, mk.q_prev = P, Q, q
    mk.profit_prev = (P - mk.cost[:, 0])[:, None] * q
    mk.R_hist[:, 0] = Q[:, None] - q
    mk.q_hist[:, 0] = q
    agents: List[A.Agent] = []
    for i in range(N):
        for key in np.unique(lineup[:, i]):
            idx = np.flatnonzero(lineup[:, i] == key)
            cls = A.ALL_DESIGNS[key]
            given = params.get(key, {})
            names = list(cls.SPACE) + [n for n in given if n not in cls.SPACE]
            p = {n: np.broadcast_to(np.asarray(given.get(n, cls.SPACE[n].default if n in cls.SPACE else 0.0), float),
                                    (B,))[idx].copy() for n in names}
            agents.append(cls(idx, i, p, mk))
    horizon = np.full((B, N), np.nan)
    for a in agents:
        if isinstance(a, A.Composite) and a.SELECT == "rc":
            horizon[a.idx, a.i] = a.h
    sum_profit = np.zeros((B, N))
    changes = np.zeros((B, N))
    rows = []
    for t in range(1, periods):
        mk.t = t
        c_prev, c_now = mk.cost[:, t - 1], mk.cost[:, t]
        mk.c_hat = (c_prev + mk.kappa * (c_now - c_prev))[:, None] + mk.sigma[:, None] * mk.eps[:, t, :]
        obs = dict(P=mk.P_prev.copy(), Q=mk.Q_prev.copy(), q=mk.q_prev.copy(), pi=mk.profit_prev.copy(),
                   c_hat=mk.c_hat.copy())
        new = np.empty((B, N))
        for a in agents:
            new[a.idx, a.i] = a.act()
        new = np.maximum(A.Q_MIN, np.rint(np.nan_to_num(new, nan=A.Q_MIN, posinf=A.Q_MIN, neginf=A.Q_MIN)))
        target = np.full((B, N), np.nan)
        for a in agents:
            if isinstance(a, A.Composite):
                target[a.idx, a.i] = a.tgt[t]
        Q = new.sum(1)
        P = np.maximum(A.P_MIN, mk.PM[:, t] - mk.SL[:, t] * Q)
        profit = (P - c_now)[:, None] * new
        mk.R_hist[:, t] = Q[:, None] - new
        mk.q_hist[:, t] = new
        if t >= burn_in:
            sum_profit += profit
            changes += new != mk.q_prev
        R = Q[:, None] - new
        br_true = (mk.PM[:, t, None] - mk.SL[:, t, None] * R - c_now[:, None]) / (2 * mk.SL[:, t, None])
        for b in range(B):
            for i in range(N):
                h = horizon[b, i]
                rows.append((b, t, i, lineup[b, i], obs["P"][b], obs["Q"][b], obs["q"][b, i], obs["pi"][b, i],
                             obs["c_hat"][b, i], mk.PMB[b, t], mk.SLB[b, t], target[b, i], new[b, i],
                             bool(new[b, i] != obs["q"][b, i]), profit[b, i], t + 1,
                             t + h if np.isfinite(h) else np.nan, c_now[b], mk.PM[b, t], mk.SL[b, t], br_true[b, i],
                             t >= burn_in))
        mk.P_prev, mk.Q_prev, mk.q_prev, mk.profit_prev = P, Q, new, profit
        for a in agents:
            a.update()
    cols = ["market", "period", "slot", "design", "obs_price_prev", "obs_market_output_prev", "obs_own_output_prev",
            "obs_own_profit_prev", "obs_cost_estimate", "obs_believed_intercept", "obs_believed_slope", "pred_target",
            "action_output", "changed", "feedback_profit", "feedback_available_period", "rc_judgement_used_period",
            "researcher_cost", "researcher_true_intercept", "researcher_true_slope", "researcher_best_reply",
            "recorded"]
    trace = pd.DataFrame(rows, columns=cols)
    n_rec = periods - burn_in
    return dict(profit=sum_profit / n_rec, change_rate=changes / n_rec, trace=trace)


def agent_spec(designs: Sequence[str]) -> List[Dict[str, object]]:
    """Information and feedback specification of each design, read from the agent definitions."""
    out = []
    for k in designs:
        c = A.ALL_DESIGNS[k]
        comp = issubclass(c, A.Composite)
        info = ["last price", "last market output", "own last output and profit",
                "cost estimate of the quality set by the environment (foresight κ, noise σ)"]
        if comp and c.TARGET == "model" or k == "opt_nash":
            info.append("believed demand curve (lags unannounced shifts by the model-updating lag)")
        if k.startswith("imit"):
            info.append("each rival's last output and profit")
        fb = [FEEDBACK["own_profit"], FEEDBACK["market"]]
        if k.startswith("imit"):
            fb.append(FEEDBACK["rivals"])
        if comp and c.SELECT == "rc":
            fb.append(FEEDBACK["rc_judgement"])
        out.append(dict(design=k, name=c.name, theory=c.theory, rule=c.rule, sources=list(c.sources),
                        parameters={n: dict(default=s.default, lo=s.lo, hi=s.hi, scale=s.scale, help=s.help)
                                    for n, s in c.SPACE.items()},
                        target=getattr(c, "TARGET", None) if comp else None,
                        selection=getattr(c, "SELECT", None) if comp else None,
                        judged_with=getattr(c, "GAIN", None) if comp else None,
                        oracle=bool(getattr(c, "ORACLE", False)) if comp else False,
                        information=info, feedback=fb))
    return out


def trace_choice(envs, params, beta, n_firms: int, periods: int, burn_in: int, revision: float, memory: float,
                 rule_costs, rules, record: Sequence[int], rng_seed=None) -> Dict[str, object]:
    """rulechoice.simulate_choice step for step (same streams: the choice RNG is seeded with the first environment's
    seed and the batch size, so the whole batch is replayed), recording firm decisions in the markets `record`.
    Returns restricted_share and change_rate (equal to simulate_choice's; tested) and the trace.

    Columns: chosen_rule (the rule the firm used), proposal:<rule> (each carried rule's proposed output: the
    predictions), action_output, feedback_profit (seen at the next decision), fitness:<rule> (exponentially weighted
    profit of the rule's current users, the information the choice is based on, updated after clearing), revised
    (whether the firm drew a new rule after this period) and researcher_* quantities."""
    from .. import rulechoice as R
    B, N, K = len(envs), n_firms, len(rules)
    beta = np.broadcast_to(np.asarray(beta, float), (B,))
    cost = np.array([(rule_costs or {}).get(k, 0.0) for k in rules])
    mk = A.Market(envs, N, periods)
    q0 = np.array([A.nash_output(e.q_range, N) for e in envs], float)
    q = np.repeat(q0[:, None], N, axis=1)
    Q = q.sum(1)
    P = np.maximum(A.P_MIN, mk.PM[:, 0] - mk.SL[:, 0] * Q)
    mk.P_prev, mk.Q_prev, mk.q_prev = P, Q, q
    mk.profit_prev = (P - mk.cost[:, 0])[:, None] * q
    mk.R_hist[:, 0] = Q[:, None] - q
    mk.q_hist[:, 0] = q
    idx = np.arange(B)
    agents = [[A.ALL_DESIGNS[k](idx, i, {n: np.full(B, float(params.get(k, {}).get(n, s.default)))
                                         for n, s in A.ALL_DESIGNS[k].SPACE.items()}, mk) for i in range(N)]
              for k in rules]
    rng = np.random.default_rng([int(envs[0].seed) if rng_seed is None else rng_seed, 99, B])
    choice = rng.integers(0, K, (B, N))
    U = np.zeros((B, K))
    scale = np.maximum(np.abs(mk.profit_prev).mean(1), 1.0)
    restricted = np.isin(np.array(rules), R.RESTRICTED)
    rule_share = np.zeros((B, K))
    changes, n_rec = np.zeros(B), 0
    rows = []
    for t in range(1, periods):
        mk.t = t
        c_prev, c_now = mk.cost[:, t - 1], mk.cost[:, t]
        mk.c_hat = (c_prev + mk.kappa * (c_now - c_prev))[:, None] + mk.sigma[:, None] * mk.eps[:, t, :]
        obs_P = mk.P_prev.copy()
        props = np.empty((K, B, N))
        for k in range(K):
            for i in range(N):
                props[k, :, i] = agents[k][i].act()
        new = np.take_along_axis(props, choice[None], axis=0)[0]
        new = np.maximum(A.Q_MIN, np.rint(np.nan_to_num(new, nan=A.Q_MIN, posinf=A.Q_MIN, neginf=A.Q_MIN)))
        Q = new.sum(1)
        P = np.maximum(A.P_MIN, mk.PM[:, t] - mk.SL[:, t] * Q)
        profit = (P - c_now)[:, None] * new
        mk.R_hist[:, t] = Q[:, None] - new
        mk.q_hist[:, t] = new
        changed = (new != mk.q_prev)
        prev_q, chosen = mk.q_prev.copy(), choice.copy()
        mk.P_prev, mk.Q_prev, mk.q_prev, mk.profit_prev = P, Q, new, profit
        for k in range(K):
            for i in range(N):
                agents[k][i].update()
        users = choice[:, :, None] == np.arange(K)[None, None, :]
        cnt = users.sum(1)
        mean_k = np.where(cnt > 0, (profit[:, :, None] * users).sum(1) / np.maximum(cnt, 1), 0.0) - cost
        U = np.where(cnt > 0, memory * U + (1 - memory) * mean_k, U)
        scale = 0.99 * scale + 0.01 * np.maximum(np.abs(profit).mean(1), 1.0)
        rev = rng.random((B, N)) < revision
        if rev.any():
            z = beta[:, None] * U / scale[:, None]
            pz = np.exp(z - z.max(1, keepdims=True))
            pz /= pz.sum(1, keepdims=True)
            draw = (np.cumsum(pz, 1)[:, None, :] < rng.random((B, N, 1))).sum(2)
            choice = np.where(rev, np.minimum(draw, K - 1), choice)
        for b in record:
            for i in range(N):
                rows.append(dict(market=b, period=t, firm=i, beta=beta[b], chosen_rule=rules[chosen[b, i]],
                                 obs_price_prev=obs_P[b],
                                 obs_own_output_prev=prev_q[b, i], obs_cost_estimate=mk.c_hat[b, i],
                                 **{f"proposal:{k}": props[j, b, i] for j, k in enumerate(rules)},
                                 action_output=new[b, i], changed=bool(changed[b, i]), feedback_profit=profit[b, i],
                                 feedback_available_period=t + 1,
                                 **{f"fitness:{k}": U[b, j] for j, k in enumerate(rules)},
                                 revised=bool(rev[b, i]), next_rule=rules[choice[b, i]],
                                 researcher_cost=c_now[b], researcher_price=P[b], recorded=t >= burn_in))
        if t >= periods - (periods - burn_in) // 3:
            rule_share += users.mean(1)
            changes += changed.mean(1)
            n_rec += 1
    return dict(restricted_share=(rule_share / n_rec)[:, restricted].sum(1), change_rate=changes / n_rec,
                trace=pd.DataFrame(rows))


ENGINE_COLUMNS = {
    "c_hat": "obs_cost_estimate", "best_reply": "pred_best_reply_believed", "rec": "pred_recommended_output",
    "adv_pred": "pred_advantage", "adv_se": "pred_advantage_se", "adv_bound": "pred_advantage_bound",
    "adv_neff": "pred_evidence", "deviate": "action_deviate", "opportunity": "opportunity", "q": "action_output",
    "explored": "action_explored", "profit": "feedback_profit", "fb_value": "feedback_judgement",
    "fb_release": "feedback_judgement_release_period", "true_best_reply": "researcher_best_reply",
    "gain_static": "researcher_gain_one_period", "gain_persist": "researcher_gain_persistent",
    "gain_full": "researcher_gain_H_periods", "profit_rule": "researcher_profit_own_rules",
    "profit_default": "researcher_profit_rule_b",
}


def trace_engine(scn, horizon: int = 1, continuation: str = "default") -> pd.DataFrame:
    """Decision trace of one market-lab market (heiner_abm.engine) from its recorded firm history. Timing follows
    agents.DECISION_SCHEDULE: decisions at t use information up to t − 1 plus the period-t cost estimate; profit of t
    is observed at t + 1; Adaptive judgements are released at the recorded release period."""
    from ..engine import run_batch
    res = run_batch([scn], record_firm_history=True, horizon=horizon, continuation=continuation)
    h = res.firm_hist
    T, N = h["q"].shape[1], h["q"].shape[2]
    t = np.repeat(np.arange(T), N)
    f = np.tile(np.arange(N), T)
    price = res.price[0]
    df = dict(period=t, firm=f, flex=np.tile([x.flex for x in scn.firms], T),
              obs_price_prev=np.repeat(np.concatenate([[np.nan], price[:-1]]), N),
              obs_own_output_prev=np.concatenate([np.full(N, np.nan), h["q"][0, :-1].reshape(-1)]))
    for k, col in ENGINE_COLUMNS.items():
        if k in h and getattr(h[k], "ndim", 0) == 3:
            df[col] = h[k][0].reshape(-1)
    df["feedback_profit_available_period"] = t + 1
    df["researcher_cost"] = np.repeat(res.cost[0], N)
    df["researcher_price"] = np.repeat(price, N)
    df["recorded"] = t >= res.burn_in
    return pd.DataFrame(df)
