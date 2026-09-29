"""Experiment designs: parameter sweeps, RC validation with paired counterfactual runs,
CD-gap grids and endogenous-flexibility (evolution) runs.

All experiments use common random numbers: replication r uses seed `base.seed + r` in every
condition, so conditions are compared on identical raw-material cost shocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .analysis import flex_profit_by_market, market_table, reliability_table, summarize_slopes
from .engine import run_batch
from .params import FirmSpec, Scenario, linear_flex_firms


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    target: str          # "market", "globals", "firms", "flexgrid", "special"
    attr: str
    lo: float
    hi: float
    step: float
    help: str = ""


PARAMS: Dict[str, Param] = {p.key: p for p in [
    Param("delta", "Cost volatility Δ (max change / period)", "market", "delta", 0.5, 40, 0.5,
          "Difficulty: size of the random raw-material cost shocks."),
    Param("c_max", "Max raw-material cost", "market", "c_max", 40, 99, 1,
          "Upper reflecting bound of cost; lowers/raises industry profitability."),
    Param("c0", "Initial raw-material cost", "market", "c0", 10, 99, 1),
    Param("q_range", "Demand quantity range (1/slope)", "market", "q_range", 300, 5000, 50,
          "Larger = flatter demand. Steep demand destabilises cobweb markets."),
    Param("desired_margin", "Desired margin m* (Bertrand)", "firms", "desired_margin", -5, 25, 0.5,
          "Intensity of competition: small m* = fierce, large m* = gentlemanly."),
    Param("foresight", "Competence κ (cost foresight)", "firms", "foresight", 0, 1, 0.05,
          "Share of the next cost change a firm anticipates. κ = 1 closes the cost CD-gap."),
    Param("noise", "Perception noise σ", "firms", "noise", 0, 20, 0.5,
          "Random misperception of cost: widens the CD-gap."),
    Param("threshold", "Selection threshold θ", "firms", "threshold", 0, 100, 1),
    Param("fixed_cost", "Fixed cost per period (b)", "globals", "fixed_cost", 0, 3000, 50),
    Param("flex_cost_slope", "Flexibility cost slope (a)", "globals", "flex_cost_slope", 0, 3000, 50),
    Param("flex_scale", "Flexibility scale (multiplies every φ)", "special", "flex_scale", 0.1, 4, 0.1,
          "Moves the whole industry between sluggish, tracking and oscillating regimes."),
    Param("n_firms", "Number of firms", "special", "n_firms", 2, 12, 1),
]}


def apply_param(scn: Scenario, key: str, value: float) -> Scenario:
    p = PARAMS[key]
    if p.target == "market":
        setattr(scn.market, p.attr, float(value))
    elif p.target == "globals":
        setattr(scn.firm_globals, p.attr, float(value))
    elif p.target == "firms":
        for f in scn.firms:
            setattr(f, p.attr, float(value))
    elif key == "flex_scale":
        for f in scn.firms:
            f.flex = f.flex * float(value)
            if f.rule == "Cournot":
                f.flex = min(f.flex, 1.0)
    elif key == "n_firms":
        n = int(value)
        tmpl = scn.firms[0]
        lo = min(f.flex for f in scn.firms)
        hi = max(f.flex for f in scn.firms)
        flexes = np.linspace(lo, hi, n) if n > 1 else [hi]
        scn.firms = [FirmSpec(rule=tmpl.rule, flex=float(x), selection=tmpl.selection, threshold=tmpl.threshold,
                              desired_margin=tmpl.desired_margin, foresight=tmpl.foresight, noise=tmpl.noise)
                     for x in flexes]
        # keep total initial capacity comparable to the 4-firm base
        scn.firm_globals.q0 = round(scn.firm_globals.q0 * 4 / n) if n != 4 else scn.firm_globals.q0
    return scn


def _conditions(base, p1, v1, p2=None, v2=None):
    conds = []
    for a in v1:
        if p2 is None:
            conds.append({p1: a})
        else:
            for b in v2:
                conds.append({p1: a, p2: b})
    return conds


def run_sweep(base: Scenario, p1: str, values1: Sequence[float], reps: int,
              p2: Optional[str] = None, values2: Optional[Sequence[float]] = None,
              progress: Optional[Callable[[float], None]] = None,
              horizon: int = 1, continuation: str = "default") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run every condition x replication. Returns (firm-level table, market-level table)."""
    conds = _conditions(base, p1, values1, p2, values2)
    # group by firm count so each batch has a constant N
    groups: Dict[int, List[Tuple[int, dict, int, Scenario]]] = {}
    for ci, cond in enumerate(conds):
        for r in range(reps):
            s = base.copy(seed=base.seed + r)
            for k, v in cond.items():
                apply_param(s, k, v)
            groups.setdefault(s.n_firms, []).append((ci, cond, r, s))
    firm_frames, mkt_frames = [], []
    done, total = 0, sum(len(g) for g in groups.values())
    for n, items in groups.items():
        chunk = 400
        for start in range(0, len(items), chunk):
            part = items[start:start + chunk]
            res = run_batch([it[3] for it in part], horizon=horizon, continuation=continuation)
            ft, mt = reliability_table(res), market_table(res)
            keys = pd.DataFrame([dict(condition=it[0], rep=it[2], **it[1]) for it in part])
            keys["market"] = np.arange(len(part))
            ft = ft.drop(columns=[c for c in keys.columns if c in ft.columns and c != "market"])
            mt = mt.drop(columns=[c for c in keys.columns if c in mt.columns and c != "market"])
            ft = ft.merge(keys, on="market")
            mt = mt.merge(keys, on="market")
            uid = f"{n}_{start}_"
            ft["market"] = uid + ft["market"].astype(str)
            mt["market"] = uid + mt["market"].astype(str)
            firm_frames.append(ft)
            mkt_frames.append(mt)
            done += len(part)
            if progress:
                progress(done / total)
    return pd.concat(firm_frames, ignore_index=True), pd.concat(mkt_frames, ignore_index=True)


def slope_summary(firms: pd.DataFrame, markets: pd.DataFrame, by: Sequence[str],
                  flex_col: str = "flex") -> pd.DataFrame:
    """Per condition: mean within-market profit~flexibility slope with 95% CI, plus market descriptors."""
    rows = []
    for key, g in firms.groupby(list(by)):
        sl = flex_profit_by_market(g, flex_col)
        s = summarize_slopes(sl)
        mk = markets[markets["market"].isin(g["market"].unique())]
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **s,
                     "avg_margin": mk["avg_margin"].mean(), "avg_firm_profit": g["avg_profit"].mean(),
                     "sc_cost": mk["sc_cost"].mean(), "sc_price": mk["sc_price"].mean(),
                     "pi": g["pi"].mean(), "r": g["r"].mean(), "w": g["w"].mean(),
                     "G": g["G"].mean(), "D": g["D"].mean(),
                     "rc_share": g["rc_holds"].mean(), "cd_gap": g["cd_gap"].mean(),
                     "ratio": g["reliability_ratio"].replace(np.inf, np.nan).median(),
                     "tolerance": g["tolerance_limit"].median(),
                     "log_rc_margin": g["log_rc_margin"].replace([np.inf, -np.inf], np.nan).median()})
    return pd.DataFrame(rows)


def switch_point(summary: pd.DataFrame, x: str) -> Optional[float]:
    """Linear interpolation of the x value where the mean slope crosses zero (first crossing)."""
    d = summary.sort_values(x)
    xs, ys = d[x].to_numpy(float), d["slope"].to_numpy(float)
    for i in range(len(xs) - 1):
        if np.sign(ys[i]) != np.sign(ys[i + 1]) and not np.isnan(ys[i]) and not np.isnan(ys[i + 1]):
            return float(xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i]))
    return None


# ---------------------------------------------------------------------------------------------
# RC validation: does the reliability condition predict whether a firm's flexibility pays off?
# ---------------------------------------------------------------------------------------------

@dataclass
class EnvRanges:
    delta: Tuple[float, float] = (2.0, 30.0)
    desired_margin: Tuple[float, float] = (0.0, 16.0)
    c_max: Tuple[float, float] = (60.0, 95.0)
    flex: Tuple[float, float] = (0.1, 1.0)
    threshold: Tuple[float, float] = (5.0, 40.0)
    foresight: Tuple[float, float] = (0.0, 0.0)
    selections: Tuple[str, ...] = ("Always", "Small", "Large", "Adaptive")
    rules: Tuple[str, ...] = ("Bertrand",)


def random_environments(base: Scenario, n_env: int, ranges: EnvRanges, seed: int = 12345) -> List[Scenario]:
    rng = np.random.default_rng(seed)
    envs = []
    u = lambda lohi: float(rng.uniform(*lohi)) if lohi[1] > lohi[0] else float(lohi[0])
    for e in range(n_env):
        s = base.copy(seed=base.seed + 1000 * e)
        s.market.delta = u(ranges.delta)
        s.market.c_max = max(u(ranges.c_max), s.market.c0 + 1)
        s.market.delta = min(s.market.delta, s.market.c_max - s.market.p_min)
        m = u(ranges.desired_margin)
        firms = []
        for f in s.firms:
            rule = str(rng.choice(ranges.rules))
            flex = u(ranges.flex)
            firms.append(FirmSpec(rule=rule, flex=min(flex, 1.0) if rule == "Cournot" else flex,
                                  selection=str(rng.choice(ranges.selections)), threshold=u(ranges.threshold),
                                  desired_margin=m, foresight=u(ranges.foresight), noise=f.noise))
        s.firms = firms
        envs.append(s)
    return envs


def rc_validation(envs: Sequence[Scenario], reps: int,
                  progress: Optional[Callable[[float], None]] = None, horizon: int = 1,
                  continuation: str = "default") -> pd.DataFrame:
    """For every environment, replication and firm i: run the market as specified and again with firm i
    switched to the rigid default rule ('Never'), same random numbers. The dynamic advantage of
    flexibility is profit(flexible) - profit(rigid twin). Compare with the RC computed in the run."""
    out = []
    jobs = []
    for e, env in enumerate(envs):
        for r in range(reps):
            s = env.copy(seed=env.seed + r)
            jobs.append((e, r, -1, s))
            for i in range(s.n_firms):
                t = s.copy()
                t.firms[i].selection = "Never"
                jobs.append((e, r, i, t))
    groups: Dict[int, list] = {}
    for j in jobs:
        groups.setdefault(j[3].n_firms, []).append(j)
    done, total = 0, len(jobs)
    for n, items in groups.items():
        for start in range(0, len(items), 500):
            part = items[start:start + 500]
            res = run_batch([j[3] for j in part], horizon=horizon, continuation=continuation)
            ft = reliability_table(res)
            mt = market_table(res)
            prof = res.acc["sum_profit"] / res.acc["n_rec"]
            idx = {(j[0], j[1], j[2]): k for k, j in enumerate(part)}
            for (e, r, i), k in idx.items():
                if i != -1:
                    continue
                for fi in range(n):
                    twin = idx.get((e, r, fi))
                    if twin is None:
                        continue
                    row = ft.iloc[k * n + fi].to_dict()     # rows are ordered by (market, firm)
                    row.update(env=e, rep=r, delta=envs[e].market.delta, c_max=envs[e].market.c_max,
                               avg_margin=mt.loc[k, "avg_margin"], sc_cost=mt.loc[k, "sc_cost"],
                               rigid_twin_profit=prof[twin, fi],
                               dynamic_adv=prof[k, fi] - prof[twin, fi])
                    out.append(row)
            done += len(part)
            if progress:
                progress(done / total)
    df = pd.DataFrame(out)
    return df[df["selection"] != "Never"].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Endogenous flexibility
# ---------------------------------------------------------------------------------------------

def evolution_runs(base: Scenario, deltas: Sequence[float], reps: int,
                   horizon: int = 1, continuation: str = "default") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Firms revise their flexibility by imitating the most profitable rival (plus experimentation).
    Returns (per-period path averaged over reps, per-run summary)."""
    scns, keys = [], []
    for d in deltas:
        for r in range(reps):
            s = base.copy(seed=base.seed + r)
            s.market.delta = float(d)
            s.evolution.enabled = True
            scns.append(s)
            keys.append((d, r))
    res = run_batch(scns, horizon=horizon, continuation=continuation)
    T = res.price.shape[1]
    margin = res.price - res.cost
    paths, summ = [], []
    ev = base.evolution
    for d in deltas:
        idx = [k for k, (dd, _) in enumerate(keys) if dd == d]
        fl = res.flex_path[idx].mean(axis=2)                    # (reps, T)
        mg = margin[idx]
        prof_t = None
        step = max(1, T // 200)
        for t in range(0, T, step):
            paths.append(dict(delta=d, period=t, flex_mean=fl[:, t].mean(), flex_lo=np.percentile(fl[:, t], 10),
                              flex_hi=np.percentile(fl[:, t], 90),
                              margin=mg[:, max(0, t - step):t + 1].mean()))
        q = T // 4
        for k in idx:
            summ.append(dict(delta=d, rep=keys[k][1], flex_start=res.flex_path[k, 0].mean(),
                             flex_end=res.flex_path[k, -1].mean(), margin_first_q=margin[k, 1:q].mean(),
                             margin_last_q=margin[k, -q:].mean(),
                             sc_cost=float(np.corrcoef(res.cost[k, 1:], res.cost[k, :-1])[0, 1])))
    return pd.DataFrame(paths), pd.DataFrame(summ)
