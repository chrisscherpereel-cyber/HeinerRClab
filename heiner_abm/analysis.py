"""Turn raw simulation output into Heiner reliability metrics, market statistics and tests."""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .engine import BatchResult


def _div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b != 0, a / np.where(b == 0, 1, b), np.nan)
    return out


def serial_corr(x: np.ndarray) -> np.ndarray:
    """Lag-1 serial correlation along the last axis (the VBA SerialCorrel)."""
    x = np.asarray(x, dtype=float)
    a, b = x[..., :-1], x[..., 1:]
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return _div((a * b).sum(-1), den)


def reliability_table(res: BatchResult) -> pd.DataFrame:
    """One row per (market, firm): Heiner's pi, r, w, G, D, reliability ratio and tolerance limit."""
    a = res.acc
    n_opp, n_pe = a["n_opp"], a["n_pe"]
    n_npe = n_opp - n_pe
    pi = _div(n_pe, n_opp)
    r = _div(a["n_dev_pe"], n_pe)
    w = _div(a["n_dev_npe"], n_npe)
    # Heiner: G = average gain when *correctly* deviating, D = average loss when *mistakenly* deviating.
    # Fall back to the environment's potential gain/loss when the firm never deviated in that state.
    G = np.where(a["n_dev_pe"] > 0, _div(a["sum_gain_dev_pe"], a["n_dev_pe"]), _div(a["sum_gain_pe"], n_pe))
    D = np.where(a["n_dev_npe"] > 0, _div(a["sum_loss_dev_npe"], a["n_dev_npe"]), _div(a["sum_loss_npe"], n_npe))
    ratio = np.where(w > 0, _div(r, w), np.where(r > 0, np.inf, np.nan))
    tol = _div(D * (1 - pi), G * pi)
    rc_holds = np.where(np.isnan(ratio) | np.isnan(tol), np.nan, (ratio > tol).astype(float))
    exp_net = pi * r * G - (1 - pi) * w * D          # expected net gain per opportunity
    rows = []
    for b, scn in enumerate(res.scenarios):
        for i, f in enumerate(scn.firms):
            rows.append(dict(
                market=b, firm=i + 1, label=f.label or f"Firm {i + 1}", rule=f.rule, selection=f.selection,
                flex=f.flex, flex_final=res.flex_final[b, i], threshold=f.threshold,
                desired_margin=f.desired_margin, foresight=f.foresight, noise=f.noise,
                avg_profit=a["sum_profit"][b, i] / a["n_rec"][b, i],
                avg_q=a["sum_q"][b, i] / a["n_rec"][b, i],
                avg_abs_dq=a["sum_absdq"][b, i] / a["n_rec"][b, i],
                opportunities=int(n_opp[b, i]), deviations=int(a["n_dev"][b, i]),
                pi=pi[b, i], r=r[b, i], w=w[b, i], type1=1 - r[b, i] if not np.isnan(r[b, i]) else np.nan,
                type2=w[b, i], G=G[b, i], D=D[b, i], reliability_ratio=ratio[b, i], tolerance_limit=tol[b, i],
                rc_holds=rc_holds[b, i], expected_net_per_opp=exp_net[b, i],
                realized_adv_per_period=a["sum_realized_adv"][b, i] / a["n_rec"][b, i],
                cd_gap=float(np.sqrt(a["sum_cerr2"][b, i] / a["n_rec"][b, i])),
            ))
    df = pd.DataFrame(rows)
    df["log_rc_margin"] = np.log(df["reliability_ratio"].astype(float).clip(lower=1e-9)) - \
        np.log(df["tolerance_limit"].astype(float).clip(lower=1e-9))
    return df


def market_table(res: BatchResult) -> pd.DataFrame:
    burn = res.burn_in
    P, C, Q = res.price[:, burn:], res.cost[:, burn:], res.quantity[:, burn:]
    sc_p, sc_c = serial_corr(P), serial_corr(C)
    corr = np.array([np.corrcoef(p, c)[0, 1] if p.std() > 0 and c.std() > 0 else np.nan for p, c in zip(P, C)])
    regime = np.where(sc_p > sc_c + 0.02, "Sluggish (P smoother than cost)",
                      np.where(sc_p < sc_c - 0.02, "Oscillating (P overshoots)", "Tracking (P follows cost)"))
    margin = P - C
    return pd.DataFrame(dict(
        market=np.arange(res.B), avg_price=P.mean(1), sd_price=P.std(1), avg_cost=C.mean(1), sd_cost=C.std(1),
        avg_quantity=Q.mean(1), avg_margin=margin.mean(1), avg_abs_gap=np.abs(margin).mean(1),
        sc_price=sc_p, sc_cost=sc_c, corr_price_cost=corr, regime=regime,
        industry_profit=res.acc["sum_profit"].sum(1) / res.acc["n_rec"][:, 0],
        avg_flex_final=res.flex_final.mean(1),
    ))


def flex_profit_by_market(firms: pd.DataFrame, flex_col: str = "flex") -> pd.DataFrame:
    """Within each market: OLS slope of avg profit on flexibility and Spearman rank correlation."""
    out = []
    for m, g in firms.groupby("market"):
        x, y = g[flex_col].to_numpy(float), g["avg_profit"].to_numpy(float)
        if np.ptp(x) == 0 or len(x) < 2:
            slope, rho = np.nan, np.nan
        else:
            slope = np.polyfit(x, y, 1)[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = stats.spearmanr(x, y)[0] if len(x) > 2 else np.sign(slope)
        most, least = g.loc[g[flex_col].idxmax()], g.loc[g[flex_col].idxmin()]
        out.append(dict(market=m, slope=slope, spearman=rho,
                        flex_best=float(g["avg_profit"].idxmax() == g[flex_col].idxmax()),
                        rigid_best=float(g["avg_profit"].idxmax() == g[flex_col].idxmin()),
                        flex_minus_rigid=most["avg_profit"] - least["avg_profit"]))
    return pd.DataFrame(out)


def mean_ci(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = x.mean()
    if n < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def summarize_slopes(slopes: pd.DataFrame) -> Dict:
    m, lo, hi = mean_ci(slopes["slope"])
    s = slopes["slope"].dropna()
    t, p = stats.ttest_1samp(s, 0.0) if len(s) > 1 and s.std() > 0 else (np.nan, np.nan)
    return dict(slope=m, lo=lo, hi=hi, t=t, p=p, spearman=slopes["spearman"].mean(),
                flex_best=slopes["flex_best"].mean(), rigid_best=slopes["rigid_best"].mean(), n=len(s))


def verdict_from_ci(lo, hi, pos_label, neg_label, null_label="No significant relationship"):
    if np.isnan(lo) or np.isnan(hi):
        return null_label
    if lo > 0:
        return pos_label
    if hi < 0:
        return neg_label
    return null_label


def ols(y, X, names):
    """Small OLS helper returning coefficients, standard errors, t and p values."""
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean())) if y.std() > 0 else np.nan
    return pd.DataFrame(dict(term=["const"] + list(names), coef=beta, se=se, t=t, p=p)), r2


def confusion(pred: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    pred, actual = np.asarray(pred, bool), np.asarray(actual, bool)
    return pd.DataFrame(
        [[np.sum(pred & actual), np.sum(pred & ~actual)], [np.sum(~pred & actual), np.sum(~pred & ~actual)]],
        index=["RC satisfied", "RC violated"], columns=["Flexibility paid off", "Flexibility hurt"])
