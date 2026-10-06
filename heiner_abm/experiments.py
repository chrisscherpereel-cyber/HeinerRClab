"""Experiment designs: parameter sweeps, RC validation with paired counterfactual runs (in-sample and
out-of-sample), CD-gap grids, Heiner (1989) partial-adjustment bounds, risk-versus-structural
uncertainty comparisons and endogenous-flexibility (evolution) runs.

All experiments use common random numbers: replication r uses seed `base.seed + r` in every
condition, so conditions are compared on identical raw-material cost shocks.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .analysis import (auc, auc_ci, event_study, flex_profit_by_market, instability_index, logistic_fit,
                       logistic_predict, market_table, reliability_table, summarize_slopes)
from .engine import MEASURES, run_batch
from .params import FirmSpec, Scenario, linear_flex_firms

CHUNK = 250          # markets per engine call (bounds memory use of the per-step arrays)
RC_COLS = ("pi", "r", "w", "G", "D", "rc_holds", "log_rc_margin", "realized_adv_per_period",
           "reliability_ratio", "tolerance_limit", "gain_per_opp")


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
    Param("c0", "Initial raw-material cost", "market", "c0", 10, 99, 1,
          "Raw-material cost in period 0; the random walk starts here."),
    Param("q_range", "Demand quantity range (1/slope)", "market", "q_range", 300, 5000, 50,
          "Larger = flatter demand. Steep demand destabilizes cobweb markets."),
    Param("desired_margin", "Desired margin m* (margin-feedback rule)", "firms", "desired_margin", -5, 25, 0.5,
          "Intensity of competition: small m* = fierce, large m* = gentlemanly."),
    Param("foresight", "Competence κ (cost foresight)", "firms", "foresight", 0, 1, 0.05,
          "Share of the next cost change a firm anticipates. κ = 1 closes the cost CD-gap."),
    Param("noise", "Perception noise σ", "firms", "noise", 0, 20, 0.5,
          "Random misperception of cost: widens the CD-gap."),
    Param("threshold", "Selection threshold θ", "firms", "threshold", 0, 100, 1,
          "Size of recommended change |q* − q| that Small (SR1) / Large (SR2) selection rules compare against."),
    Param("fixed_cost", "Fixed cost per period (b)", "globals", "fixed_cost", 0, 3000, 50,
          "Cost every firm pays each period regardless of output or flexibility."),
    Param("flex_cost_slope", "Flexibility cost slope (a)", "globals", "flex_cost_slope", 0, 3000, 50,
          "Per-period cost of flexibility: firm i pays a·φᵢ, so more flexible firms pay more."),
    Param("flex_scale", "Flexibility scale (multiplies every φ)", "special", "flex_scale", 0.1, 4, 0.1,
          "Moves the whole industry between sluggish, tracking and oscillating regimes."),
    Param("n_firms", "Number of firms", "special", "n_firms", 2, 12, 1,
          "Firms competing in the market; more firms make the cobweb less stable."),
    Param("common_flex", "Common flexibility φ (all firms)", "special", "common_flex", 0.02, 1.5, 0.02,
          "Symmetric industry: every firm uses the same φ (Heiner 1989 partial-adjustment test)."),
    Param("hazard", "Regime-shift hazard λ (per period)", "structural", "hazard", 0.0, 0.2, 0.005,
          "Unannounced structural change: probability of a demand-regime shift firms are not told about."),
    Param("intercept_sd", "Regime shift size: demand intercept s.d.", "structural", "intercept_sd", 0, 40, 1,
          "Standard deviation of the jump in max price (demand intercept) at each regime shift."),
    Param("slope_sd", "Regime shift size: log demand-slope s.d.", "structural", "slope_sd", 0, 1.0, 0.05,
          "Standard deviation of the log change in demand slope at each regime shift (slope × e^(s.d.·z))."),
    Param("belief_lag", "Model-updating lag L (periods)", "structural", "belief_lag", -1, 200, 1,
          "How long model-based firms take to learn a new demand regime (-1 = never)."),
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
    elif p.target == "structural":
        setattr(scn.structural, p.attr, int(value) if p.attr == "belief_lag" else float(value))
        scn.structural.enabled = True
    elif key == "common_flex":
        for f in scn.firms:
            f.flex = min(float(value), 1.0) if f.rule == "Cournot" else float(value)
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
              horizon: int = 1, continuation: str = "default", discount: float = 1.0,
              oos_split: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run every condition x replication. Returns (firm-level table, market-level table).

    Firm rows carry the "full" RC measure; the "static" and "persist" measures are added with suffixes
    (e.g. rc_holds_static, log_rc_margin_persist) so the dynamic decomposition is available."""
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
        chunk = CHUNK
        for start in range(0, len(items), chunk):
            part = items[start:start + chunk]
            res = run_batch([it[3] for it in part], horizon=horizon, continuation=continuation,
                            discount=discount, oos_split=oos_split)
            ft, mt = reliability_table(res), market_table(res)
            for m in ("static", "persist"):
                other = reliability_table(res, m, "all")
                for c in RC_COLS:
                    ft[f"{c}_{m}"] = other[c].to_numpy()
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
    with warnings.catch_warnings():     # medians of all-NaN columns (e.g. a collapsed market) are simply NaN
        warnings.simplefilter("ignore", RuntimeWarning)
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
                         "log_rc_margin": g["log_rc_margin"].replace([np.inf, -np.inf], np.nan).median(),
                         "K": g["K"].median(), "xi_rms": g["xi_rms"].median(),
                         "heiner_bound": g["heiner_bound"].median(),
                         "industry_profit": mk["industry_profit"].mean(),
                         "price_change_rms": mk["price_change_rms"].mean(),
                         **{f"rc_share_{m}": g[f"rc_holds_{m}"].mean() for m in ("static", "persist")
                            if f"rc_holds_{m}" in g},
                         **{f"log_rc_margin_{m}": g[f"log_rc_margin_{m}"].replace([np.inf, -np.inf], np.nan).median()
                            for m in ("static", "persist") if f"log_rc_margin_{m}" in g}})
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
    hazard: Tuple[float, float] = (0.0, 0.0)          # > 0 adds unannounced demand-regime shifts
    belief_lag: Tuple[float, float] = (20.0, 20.0)


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
        hz = u(ranges.hazard)
        if hz > 0:
            s.structural.enabled = True
            s.structural.hazard = hz
            s.structural.belief_lag = int(round(u(ranges.belief_lag)))
        envs.append(s)
    return envs


def rc_validation(envs: Sequence[Scenario], reps: int,
                  progress: Optional[Callable[[float], None]] = None, horizon: int = 1,
                  continuation: str = "default", discount: float = 1.0, oos_split: float = 0.5) -> pd.DataFrame:
    """For every environment, replication and firm i: run the market as specified and again with firm i
    switched to the rigid default rule ('Never'), same random numbers. The dynamic advantage of
    flexibility is profit(flexible) - profit(rigid twin).

    Each row carries, for every RC measure m in (static, persist, full):
        rc_all_m / margin_all_m : RC computed over all recorded periods (in-sample)
        rc_est_m / margin_est_m : RC computed in the estimation window only (for persist and full, decisions
                                  whose H-period horizon ends before the evaluation window starts)
    and the outcomes dyn_adv_all (all periods) and dyn_adv_eval (evaluation window only), so the RC can
    be tested out of sample: estimated on the first part of the run, predicting the second part."""
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
        for start in range(0, len(items), CHUNK):
            part = items[start:start + CHUNK]
            res = run_batch([j[3] for j in part], horizon=horizon, continuation=continuation,
                            discount=discount, oos_split=oos_split)
            base_t = reliability_table(res, "full", "all")
            tabs = {(m, w): reliability_table(res, m, w) for m in MEASURES for w in ("all", "est")}
            mt = market_table(res)
            # realized profits need no look-ahead gap: the one-period ("static") estimation window is the whole first part
            pa, pe, ps = res.accs[("full", "all")], res.accs[("full", "eval")], res.accs[("static", "est")]
            prof_all = pa["sum_profit"] / pa["n_rec"]
            prof_eval = pe["sum_profit"] / np.maximum(pe["n_rec"], 1)
            prof_est = ps["sum_profit"] / np.maximum(ps["n_rec"], 1)
            idx = {(j[0], j[1], j[2]): k for k, j in enumerate(part)}
            for (e, r, i), k in idx.items():
                if i != -1:
                    continue
                for fi in range(n):
                    twin = idx.get((e, r, fi))
                    if twin is None:
                        continue
                    row = base_t.iloc[k * n + fi].to_dict()     # rows are ordered by (market, firm)
                    for (m, w), tb in tabs.items():
                        tr = tb.iloc[k * n + fi]
                        row[f"rc_{w}_{m}"] = tr["rc_holds"]
                        row[f"margin_{w}_{m}"] = tr["log_rc_margin"]
                        row[f"inst_adv_{w}_{m}"] = tr["realized_adv_per_period"]
                        row[f"ratio_{w}_{m}"] = tr["reliability_ratio"]
                        row[f"tol_{w}_{m}"] = tr["tolerance_limit"]
                    row["K_est"] = tabs[("static", "est")].iloc[k * n + fi]["K"]          # no look-ahead in K
                    row["cd_gap_est"] = tabs[("static", "est")].iloc[k * n + fi]["cd_gap"]
                    row.update(env=e, rep=r, delta=envs[e].market.delta, c_max=envs[e].market.c_max,
                               instability=instability_index(envs[e]), n_firms=n,
                               hazard=envs[e].structural.hazard if envs[e].structural.enabled else 0.0,
                               n_shifts=mt.loc[k, "n_shifts"],
                               avg_margin=mt.loc[k, "avg_margin"], sc_cost=mt.loc[k, "sc_cost"],
                               rigid_twin_profit=prof_all[twin, fi],
                               dynamic_adv=prof_all[k, fi] - prof_all[twin, fi],
                               dyn_adv_all=prof_all[k, fi] - prof_all[twin, fi],
                               dyn_adv_est=prof_est[k, fi] - prof_est[twin, fi],
                               dyn_adv_eval=prof_eval[k, fi] - prof_eval[twin, fi])
                    out.append(row)
            done += len(part)
            if progress:
                progress(done / total)
    df = pd.DataFrame(out)
    return df[df["selection"] != "Never"].reset_index(drop=True)


MEASURE_LABELS = {"static": "RC, one-shot (one period, H = 1)",
                  "persist": "RC, persistence (H periods, rivals don't react)",
                  "full": "RC, full dynamic (H periods, rivals react)"}


def prediction_table(df: pd.DataFrame, n_boot: int = 300) -> pd.DataFrame:
    """AUC of each predictor for 'flexibility beat the rigid twin', in-sample and out-of-sample.

    Predictors: the RC margin ln(ratio / tolerance) under each measure, and naive baselines. AUC = 0.5 is
    uninformative; the traditional 'flexibility always pays' view is a constant prediction (AUC 0.5).
    CIs resample whole environments."""
    specs = []
    for m in MEASURES:
        specs.append((MEASURE_LABELS[m], f"margin_all_{m}", "dyn_adv_all", "In-sample (all periods)", 1))
        specs.append((MEASURE_LABELS[m], f"margin_est_{m}", "dyn_adv_eval", "Out-of-sample (estimate → evaluate)", 1))
    for lab, col, sign in (("Baseline: flexibility φ", "flex", 1), ("Baseline: low volatility (−Δ)", "delta", -1),
                           ("Baseline: industry margin", "avg_margin", 1)):
        specs.append((lab, col, "dyn_adv_eval", "Out-of-sample (estimate → evaluate)", sign))
    rows = []
    for lab, col, outcome, kind, sign in specs:
        if col not in df or outcome not in df:
            continue
        d = df[[col, outcome, "env"]].replace([np.inf, -np.inf], [50.0, -50.0]).dropna()
        if d.empty:
            continue
        a, lo, hi = auc_ci(sign * d[col].to_numpy(float), d[outcome].to_numpy(float) > 0, d["env"].to_numpy(),
                           n_boot=n_boot)
        rows.append(dict(predictor=lab, test=kind, auc=a, lo=lo, hi=hi, n=len(d)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Heiner (1989): partial adjustment toward an imperfectly perceived target
# ---------------------------------------------------------------------------------------------

def adjustment_bound_experiment(base: Scenario, phis: Sequence[float], difficulty: str,
                                levels: Sequence[float], reps: int, progress=None) -> pd.DataFrame:
    """Symmetric Cournot industry (every firm uses the same partial-adjustment weight phi), crossed with a
    difficulty parameter. Per cell: industry profit, measured error-to-signal ratio K and Heiner's
    stabilizing bound beta0 = 2 / ((1 + K)(n + 1)). Heiner's Theorem 2 predicts profit-maximizing
    flexibility at or below beta0, falling as difficulty (K) rises."""
    scn = base.copy()
    for f in scn.firms:
        f.rule = "Cournot"
        f.selection = "Always"
    firms, mk = run_sweep(scn, difficulty, levels, reps, "common_flex", phis, progress=progress)
    g = firms.groupby([difficulty, "common_flex"]).agg(avg_profit=("avg_profit", "mean"),
                                                       profit_sd=("avg_profit", "std"), K=("K", "median"),
                                                       xi_rms=("xi_rms", "median"),
                                                       bound=("heiner_bound", "median")).reset_index()
    m = mk.groupby([difficulty, "common_flex"]).agg(price_change_rms=("price_change_rms", "mean"),
                                                   sc_price=("sc_price", "mean")).reset_index()
    return g.merge(m, on=[difficulty, "common_flex"])


# ---------------------------------------------------------------------------------------------
# Risk versus unannounced structural change
# ---------------------------------------------------------------------------------------------

RISK, STRUCT = "Risk (cost volatility)", "Structural (regime shifts)"


def uncertainty_comparison(base: Scenario, risk_deltas: Sequence[float], hazards: Sequence[float], reps: int,
                           horizon: int = 1, continuation: str = "default", discount: float = 1.0,
                           progress=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Two families of environments with rising unpredictability:
        risk       - structural shifts off, cost volatility Delta rises (known, stationary distribution);
        structural - base cost volatility plus unannounced demand-regime shifts with rising hazard.
    Returns (firm-level table, market-level table) with 'family' and 'intensity' columns."""
    r = base.copy()
    r.structural.enabled = False
    fr, mr = run_sweep(r, "delta", risk_deltas, reps, horizon=horizon, continuation=continuation,
                       discount=discount)
    for d in (fr, mr):
        d["family"], d["intensity"] = RISK, d["delta"]
    fs, ms = run_sweep(base, "hazard", hazards, reps, horizon=horizon, continuation=continuation,
                       discount=discount)
    for d in (fs, ms):
        d["family"], d["intensity"] = STRUCT, d["hazard"]
    fs["market"] = "s" + fs["market"].astype(str)
    ms["market"] = "s" + ms["market"].astype(str)
    return pd.concat([fr, fs], ignore_index=True), pd.concat([mr, ms], ignore_index=True)


def regime_event_study(base: Scenario, reps: int, horizon: int = 1, continuation: str = "default",
                       pre: int = 10, post: int = 40, group: str = "selection") -> pd.DataFrame:
    """Event study of behavior around regime shifts (Heiner 1989, section 6: punctuated adjustment)."""
    s0 = base.copy()
    s0.structural.enabled = True
    scns = [s0.copy(seed=s0.seed + r) for r in range(reps)]
    res = run_batch(scns, horizon=horizon, continuation=continuation, keep_steps=True)
    return event_study(res, pre=pre, post=post, group=group)


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


# ---------------------------------------------------------------------------------------------
# Horse race: Heiner's RC against the predictors implied by rival theories (out of sample)
# ---------------------------------------------------------------------------------------------

def _log_clip(x):
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(np.log(np.clip(x, 1e-9, None)), -10.0, 10.0)


def theory_predictors(df: pd.DataFrame, measure: str = "full") -> Dict[str, Tuple[str, np.ndarray]]:
    """Each rival theory's forecast that a firm's flexibility will beat its rigid twin, as a score (higher =
    flexibility more likely to pay). Firm-specific scores use only the estimation window, so every score is
    known before the evaluation window it predicts. Returns {key: (theory, score)}."""
    ratio = _log_clip(df[f"ratio_est_{measure}"].replace(np.inf, 1e9))
    tol = _log_clip(df[f"tol_est_{measure}"])
    return {
        "rc": ("Heiner (1983): reliability condition, ln(r/w) − ln(tolerance)", ratio - tol),
        "K": ("Heiner (1989): low error-to-signal ratio, −K", -df["K_est"].to_numpy(float)),
        "accuracy": ("Bias–variance / ecological rationality: accuracy only, ln(r/w)", ratio),
        "stakes": ("Stakes only: low tolerance limit, −ln(tolerance)", -tol),
        "options": ("Real options: high volatility, +Δ", df["delta"].to_numpy(float)),
        "cobweb": ("Cobweb stability: stable market, −ρ(J)", -df["instability"].to_numpy(float)),
        "past": ("Reinforcement learning: flexibility paid in the estimation window",
                 df["dyn_adv_est"].to_numpy(float)),
    }


def horse_race(df: pd.DataFrame, measure: str = "full", n_boot: int = 300) -> pd.DataFrame:
    """Out-of-sample AUC of every theory's predictor for 'flexibility beat the rigid twin in the evaluation
    window', with environment-clustered bootstrap CIs. The neoclassical view (flexibility never hurts)
    makes the same forecast for every firm, so its AUC is 0.5 by construction."""
    y = df["dyn_adv_eval"].to_numpy(float) > 0
    env = df["env"].to_numpy()
    rows = [dict(key="neoclassical", theory="Neoclassical optimization: flexibility always pays (constant)",
                 auc=0.5, lo=np.nan, hi=np.nan, n=len(df))]
    for key, (lab, score) in theory_predictors(df, measure).items():
        ok = ~np.isnan(score)
        if ok.sum() < 10 or y[ok].all() or not y[ok].any():
            continue
        a, lo, hi = auc_ci(score[ok], y[ok], env[ok], n_boot=n_boot)
        rows.append(dict(key=key, theory=lab, auc=a, lo=lo, hi=hi, n=int(ok.sum())))
    return pd.DataFrame(rows).sort_values("auc", ascending=False, ignore_index=True)


RIVALS = ("K", "accuracy", "options", "cobweb", "past")


def _train_ranks(train: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Map x onto the training sample's empirical distribution (mid-rank / n, in [0, 1]). It depends only on the
    ordering of the values, so any monotone rescaling of a forecast gives the same result, and it removes the heavy
    tails and the mass at clipping bounds that otherwise dominate a linear logit. Only the training fold defines the
    mapping."""
    srt = np.sort(np.asarray(train, float))
    x = np.asarray(x, float)
    return (np.searchsorted(srt, x, "left") + np.searchsorted(srt, x, "right")) / (2.0 * len(srt))


def encompassing_test(df: pd.DataFrame, measure: str = "full", folds: int = 5, n_boot: int = 300,
                      seed: int = 0) -> Tuple[pd.DataFrame, Dict]:
    """Does the RC add out-of-sample information once every rival predictor is known?

    Logistic models of 'flexibility beat the rigid twin (evaluation window)' are fitted on estimation-window
    features and scored by environment-grouped K-fold cross-validation (no environment is in both the
    training and the test fold). Compares (a) all rival predictors, (b) rivals + the RC's extra ingredient,
    the tolerance limit, which turns ln(r/w) into the RC margin, and (c) the RC margin alone.

    Each feature is first mapped onto its training fold's empirical distribution (ranks), so that heavy tails and
    values piled at the clipping bounds cannot flatten a slope toward zero, and any monotone rescaling of a forecast
    gives the same result. A single-feature model is scored by the raw forecast in the direction its training-fold
    fit gives, so it scores exactly like the forecast itself.
    Returns (table of cross-validated AUCs, dict with the AUC gain of (b) over (a) and its bootstrap CI)."""
    pr = theory_predictors(df, measure)
    X_all = {k: v[1] for k, v in pr.items()}
    y = df["dyn_adv_eval"].to_numpy(float) > 0
    env = df["env"].to_numpy()
    cols = list(RIVALS) + ["stakes", "rc"]
    ok = np.all([~np.isnan(X_all[c]) for c in cols], axis=0)
    y, env = y[ok], env[ok]
    X = {c: X_all[c][ok] for c in cols}
    models = {"Rival theories combined": list(RIVALS), "Rivals + tolerance limit (= rivals + RC)":
              list(RIVALS) + ["stakes"], "RC margin alone": ["rc"]}
    uniq = np.unique(env)
    if len(uniq) < folds or y.all() or not y.any():
        return pd.DataFrame(), {}
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % folds))
    fid = np.array([fold_of[e] for e in env])
    oof = {name: np.full(len(y), np.nan) for name in models}
    for name, feats in models.items():
        M = np.column_stack([X[c] for c in feats])
        for f in range(folds):
            tr, te = fid != f, fid == f
            if y[tr].all() or not y[tr].any():
                continue
            R = np.column_stack([_train_ranks(M[tr, j], M[:, j]) for j in range(M.shape[1])])
            mu, sd = R[tr].mean(0), R[tr].std(0) + 1e-9
            beta = logistic_fit((R[tr] - mu) / sd, y[tr])
            if M.shape[1] == 1:      # one forecast: score it directly, in the direction learned on the training folds
                oof[name][te] = np.sign(beta[1]) * M[te, 0]
            else:
                oof[name][te] = logistic_predict((R[te] - mu) / sd, beta)
    def cv_auc(pred, idx):
        # AUC within each test fold, averaged (weighted by fold size). Pooling folds would mix in each
        # fold's intercept, which tracks the training base rate and so is biased against the test fold.
        vals, wts = [], []
        for f in range(folds):
            j = idx[fid[idx] == f]
            a_f = auc(pred[j], y[j])
            if not np.isnan(a_f):
                vals.append(a_f)
                wts.append(len(j))
        return float(np.average(vals, weights=wts)) if vals else np.nan

    all_idx = np.arange(len(y))
    rows = [dict(model=name, features=", ".join(models[name]), cv_auc=cv_auc(oof[name], all_idx)) for name in models]
    a, b = "Rival theories combined", "Rivals + tolerance limit (= rivals + RC)"
    groups = [np.flatnonzero(env == e) for e in uniq]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots.append(cv_auc(oof[b], idx) - cv_auc(oof[a], idx))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return pd.DataFrame(rows), dict(gain=cv_auc(oof[b], all_idx) - cv_auc(oof[a], all_idx), lo=lo, hi=hi,
                                    n=int(len(y)))
