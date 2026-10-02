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
    """Lag-1 serial correlation along the last axis (the predictability proxy)."""
    x = np.asarray(x, dtype=float)
    a, b = x[..., :-1], x[..., 1:]
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return _div((a * b).sum(-1), den)


def rc_arrays(a: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Heiner's quantities from one accumulator set (arrays of shape (B, N))."""
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
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.log(np.clip(np.where(np.isnan(ratio), np.nan, ratio), 1e-9, None)) -             np.log(np.clip(tol, 1e-9, None))
    gain_opp = _div(a["sum_gain_pe"] - a["sum_loss_npe"], n_opp)   # mean value of following the rule vs B
    return dict(pi=pi, r=r, w=w, G=G, D=D, ratio=ratio, tol=tol, rc_holds=rc_holds, gain_per_opp=gain_opp,
                exp_net=pi * r * G - (1 - pi) * w * D, log_margin=margin)


def heiner_bound(K, n_firms):
    """Heiner (1989, Theorem 2) maximal stabilizing partial-adjustment coefficient, symmetric Cournot:
    beta0 = 1 / ((1 + K)(1 - f')),  with f' = -(n - 1)/2 the slope of the best-reply target map."""
    fprime = -(np.asarray(n_firms, float) - 1) / 2.0
    return 1.0 / ((1.0 + np.asarray(K, float)) * (1.0 - fprime))


def reliability_table(res: BatchResult, measure: str = "full", window: str = "all") -> pd.DataFrame:
    """One row per (market, firm): Heiner's pi, r, w, G, D, reliability ratio and tolerance limit.

    measure: "static" (one period, Heiner's one-shot form), "persist" (H periods, no rival reaction) or "full"
             (H periods in a forked market). window: "all", "est" or "eval" periods.
    """
    a = res.accs.get((measure, window), res.acc) if res.accs else res.acc
    q = rc_arrays(a)
    # Heiner (1989): error-to-signal ratio K = RMS(decision error xi) / RMS(true optimal adjustment)
    K = np.sqrt(_div(a["sum_brerr2"], a["sum_brsig2"]))
    xi = np.sqrt(_div(a["sum_brerr2"], a["n_br"]))
    rows = []
    for b, scn in enumerate(res.scenarios):
        n = scn.n_firms
        for i, f in enumerate(scn.firms):
            nr = a["n_rec"][b, i]
            rows.append(dict(
                market=b, firm=i + 1, label=f.label or f"Firm {i + 1}", rule=f.rule, selection=f.selection,
                flex=f.flex, flex_final=res.flex_final[b, i], threshold=f.threshold,
                desired_margin=f.desired_margin, foresight=f.foresight, noise=f.noise,
                avg_profit=a["sum_profit"][b, i] / nr if nr else np.nan,
                avg_q=a["sum_q"][b, i] / nr if nr else np.nan,
                avg_abs_dq=a["sum_absdq"][b, i] / nr if nr else np.nan,
                opportunities=int(a["n_opp"][b, i]), deviations=int(a["n_dev"][b, i]),
                pi=q["pi"][b, i], r=q["r"][b, i], w=q["w"][b, i],
                type1=1 - q["r"][b, i] if not np.isnan(q["r"][b, i]) else np.nan,
                type2=q["w"][b, i], G=q["G"][b, i], D=q["D"][b, i], reliability_ratio=q["ratio"][b, i],
                tolerance_limit=q["tol"][b, i], rc_holds=q["rc_holds"][b, i], log_rc_margin=q["log_margin"][b, i],
                expected_net_per_opp=q["exp_net"][b, i], gain_per_opp=q["gain_per_opp"][b, i],
                realized_adv_per_period=a["sum_realized_adv"][b, i] / nr if nr else np.nan,
                cd_gap=float(np.sqrt(a["sum_cerr2"][b, i] / nr)) if nr else np.nan,
                xi_rms=xi[b, i], K=K[b, i], heiner_bound=float(heiner_bound(K[b, i], n)),
            ))
    return pd.DataFrame(rows)


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
        n_shifts=(res.shifts[:, burn:].sum(1) if res.shifts is not None else np.zeros(res.B)),
        xi_rms=np.sqrt(_div(res.acc["sum_brerr2"].sum(1), res.acc["n_br"].sum(1))),
        K=np.sqrt(_div(res.acc["sum_brerr2"].sum(1), res.acc["sum_brsig2"].sum(1))),
        price_change_rms=np.sqrt((np.diff(P, axis=1) ** 2).mean(1)),
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


# ------------------------------------------------------------------------------------------------
# Prediction quality: AUC (signal-detection area under the ROC curve) with clustered bootstrap
# ------------------------------------------------------------------------------------------------
def auc(score, label) -> float:
    """P(score of a random positive > score of a random negative), ties count 1/2 (Mann-Whitney)."""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    ok = ~np.isnan(score)
    score, label = score[ok], label[ok]
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    ranks = stats.rankdata(score)
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auc_ci(score, label, cluster, n_boot: int = 300, seed: int = 0, conf: float = 0.95):
    """AUC with a cluster bootstrap CI (resample whole environments, not single firms)."""
    df = pd.DataFrame(dict(s=np.asarray(score, float), y=np.asarray(label, bool), c=np.asarray(cluster)))
    point = auc(df["s"], df["y"])
    groups = [g for _, g in df.groupby("c")]
    if len(groups) < 3:
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        d = pd.concat([groups[i] for i in pick])
        boots.append(auc(d["s"], d["y"]))
    lo, hi = np.nanpercentile(boots, [50 * (1 - conf), 100 - 50 * (1 - conf)])
    return point, lo, hi


# ------------------------------------------------------------------------------------------------
# Signal-detection view of the RC (Heiner 1986): r = hit rate, w = false-alarm rate
# ------------------------------------------------------------------------------------------------
def sdt_roc(signal, gain, n_thresholds: int = 60) -> pd.DataFrame:
    """ROC of the decision 'deviate when the signal |q* - q| exceeds theta' (selection rule SR2).

    For every threshold: r(theta) = P(deviate | exception), w(theta) = P(deviate | no exception),
    the RC quantities, and the expected net gain per opportunity. The value-maximizing threshold is
    where the ROC's local slope equals the tolerance limit (the signal-detection optimal criterion)."""
    signal = np.asarray(signal, float)
    gain = np.asarray(gain, float)
    pe = gain > 0
    n, npe = len(gain), pe.sum()
    if n == 0 or npe == 0 or npe == n:
        return pd.DataFrame()
    pi = npe / n
    thetas = np.unique(np.quantile(signal, np.linspace(0, 1, n_thresholds)))
    thetas = np.r_[-np.inf, thetas]
    rows = []
    for th in thetas:
        d = signal > th
        r = (d & pe).sum() / npe
        w = (d & ~pe).sum() / (n - npe)
        G = gain[d & pe].mean() if (d & pe).any() else np.nan
        Dl = -gain[d & ~pe].mean() if (d & ~pe).any() else np.nan
        tol = (Dl / G) * (1 - pi) / pi if G and not np.isnan(G) and not np.isnan(Dl) else np.nan
        rows.append(dict(theta=th, r=r, w=w, G=G, D=Dl, ratio=r / w if w > 0 else np.inf, tolerance=tol,
                         net_per_opp=np.where(d, gain, 0.0).mean(), deviate_share=d.mean()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# Event study around regime shifts (Heiner 1989, section 6: punctuated adjustment)
# ------------------------------------------------------------------------------------------------
def event_study(res: BatchResult, pre: int = 10, post: int = 40, group: str = "selection") -> pd.DataFrame:
    """Average behavior of firms around demand-regime shifts, relative to the shift period (k = 0).

    Needs keep_steps=True. Reports deviation rate, |dq| relative to the firm's mean |dq|, the
    decision error |xi| (perceived vs true best reply), and profit relative to the firm's mean."""
    if res.steps is None or res.shifts is None:
        return pd.DataFrame()
    st = res.steps
    B, T, N = st["q"].shape
    burn = res.burn_in
    absdq = np.abs(st["q"] - st["q_prev"]).astype(float)
    xi = np.abs(st["br_perc"] - st["br_true"]).astype(float)
    base_dq = absdq[:, burn:].mean(1, keepdims=True) + 1e-9
    base_pr = np.abs(st["profit"][:, burn:]).mean(1, keepdims=True) + 1e-9
    base_xi = xi[:, burn:].mean(1, keepdims=True) + 1e-9
    rows = []
    for b in range(B):
        labels = [getattr(f, group) if group != "firm" else f"Firm {i + 1}" for i, f in enumerate(res.scenarios[b].firms)]
        for t0 in np.flatnonzero(res.shifts[b]):
            if t0 - pre < burn or t0 + post >= T:
                continue
            for k in range(-pre, post + 1):
                t = t0 + k
                for i in range(N):
                    rows.append((labels[i], k, float(st["dev"][b, t, i] & st["opp"][b, t, i]),
                                 absdq[b, t, i] / base_dq[b, 0, i], xi[b, t, i] / base_xi[b, 0, i],
                                 st["profit"][b, t, i] / base_pr[b, 0, i]))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[group, "k", "deviate", "rel_absdq", "rel_xi", "rel_profit"])
    return df.groupby([group, "k"]).mean().reset_index()


# ------------------------------------------------------------------------------------------------
# Rival-theory predictors: cobweb stability and a small logistic regression
# ------------------------------------------------------------------------------------------------
def instability_index(scn) -> float:
    """Cobweb stability theory (Ezekiel 1938; Nerlove 1958; Theocharis 1960): spectral radius of the
    linearized production dynamics q' = J q + const, ignoring noise, rounding and bounds.

    Cournot firm i: q_i' = (1 - phi_i/2) q_i - (phi_i/2) Q;  Bertrand firm i: q_i' = q_i - phi_i s Q.
    So J = diag(d) - u 1', and the market is stable when rho(J) < 1. Unit eigenvalues are neutral modes
    (with only Bertrand firms, how total output is split is indeterminate) and are left out."""
    s = scn.market.slope
    phi = np.array([f.flex for f in scn.firms], float)
    cour = np.array([f.rule == "Cournot" for f in scn.firms])
    d = np.where(cour, 1 - phi / 2, 1.0)
    u = np.where(cour, phi / 2, phi * s)
    ev = np.abs(np.linalg.eigvals(np.diag(d) - np.outer(u, np.ones(len(phi)))))
    ev = ev[np.abs(ev - 1.0) > 1e-9]
    return float(ev.max()) if len(ev) else 1.0


def logistic_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 50) -> np.ndarray:
    """Logistic regression (with intercept) by Newton-Raphson with a small ridge penalty for stability."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X1.shape[1])
    pen = np.full(X1.shape[1], ridge)
    pen[0] = 0.0
    y = np.asarray(y, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ beta, -30, 30)))
        g = X1.T @ (y - p) - pen * beta
        Hm = (X1 * (p * (1 - p))[:, None]).T @ X1 + np.diag(pen)
        step = np.linalg.solve(Hm + 1e-9 * np.eye(len(beta)), g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def logistic_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), X]) @ beta, -30, 30)))
