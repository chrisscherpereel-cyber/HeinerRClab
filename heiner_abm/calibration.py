"""Calibration to laboratory data: which theory's decision rule predicts human choices best, out of sample?

Two experimental formats are supported:

* Learning-to-forecast cobweb markets (the design of Hommes et al. 2007): subjects forecast next period's price;
  the market price follows from the forecasts. Columns: group, subject, period, price, forecast
  (forecast = the subject's forecast of the price in that period, made before the price was known).
* Quantity-setting Cournot markets (the design of Huck et al. 1999): subjects choose output. Columns: group, subject,
  period, quantity; market parameters are entered separately: each firm's inverse demand P_i = a − b(q_i + γ·Q_−i)
  (γ = 1 for a homogeneous product, γ < 1 for differentiated products) and unit cost c.

Every theory contributes decision rules of the same kind as its agent designs. Each rule's parameters are fitted per
subject on the first half of that subject's periods (grid search, squared error) and the rule is scored on the second
half, which it has not seen. Subjects are classified by the rule with the lowest out-of-sample error.

The datasets themselves are not bundled: the Cournot data of Huck, Normann & Oechssler (1999) are archived in
heiDATA (doi:10.11588/data/10012, access on request) and the cobweb data of Hommes et al. (2007) are available from the
authors. synthetic_forecast_data and synthetic_cournot_data generate data with known rule types, used to check that
the pipeline recovers the truth.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    key: str
    name: str
    theory: str
    grid: Dict[str, Sequence[float]]
    predict: Callable          # (subject frame, group frame, params, extra) -> predictions aligned with rows


# ================================================================================================ forecasting rules
def _fc_naive(d, g, p, x):
    return d["price_prev"].to_numpy()


def _adaptive_path(prices_prev, lam):
    f = np.empty(len(prices_prev))
    f[0] = prices_prev[0]
    for t in range(1, len(f)):
        f[t] = f[t - 1] + lam * (prices_prev[t] - f[t - 1])      # forecast of P_t uses P_{t-1}
    return f


def _fc_adaptive(d, g, p, x):
    return _adaptive_path(d["price_prev"].to_numpy(), p["lam"])


def _fc_rational(d, g, p, x):
    """Rational expectations: the equilibrium (fundamental) price if it is known, else the training-window mean."""
    if "fundamental" in d and d["fundamental"].notna().all():
        return d["fundamental"].to_numpy(float)
    fund = x.get("fundamental")
    return np.full(len(d), float(fund) if fund is not None else x["mean_price_train"])


def _fc_trend(d, g, p, x):
    return d["price_prev"].to_numpy() + p["gamma"] * (d["price_prev"] - d["price_prev2"]).to_numpy()


def _fc_anchor(d, g, p, x):
    pp = d["price_prev"].to_numpy()
    mean_so_far = np.cumsum(pp) / np.arange(1, len(pp) + 1)
    return 0.5 * (mean_so_far + pp) + p["gamma"] * (d["price_prev"] - d["price_prev2"]).to_numpy()


def _fc_keep(d, g, p, x):
    return d["own_prev"].to_numpy()


def _fc_band(d, g, p, x):
    """Real options: move the forecast to the adaptive target only when the gap exceeds k standard deviations of
    recent price changes; otherwise keep last period's own forecast."""
    tgt = _adaptive_path(d["price_prev"].to_numpy(), p["lam"])
    vol = d["price_prev"].diff().abs().expanding().mean().fillna(1.0).to_numpy()
    own = d["own_prev"].to_numpy()
    return np.where(np.abs(tgt - own) > p["k"] * vol, tgt, own)


def _fc_rc(d, g, p, x):
    """Reliability condition: keep last period's own forecast unless updating to the adaptive target has paid off,
    in past periods with a gap of similar size, in reduced squared error (learned with exponential memory)."""
    tgt = _adaptive_path(d["price_prev"].to_numpy(), p["lam"])
    own, price = d["own_prev"].to_numpy(), d["price"].to_numpy()
    edges = p["theta"] * np.array([0.5, 1.0, 2.0, 4.0])
    lag = int(x.get("horizon", 1)) - 1           # with two-period-ahead forecasts the price arrives a period later
    E = np.zeros(5)
    out = np.empty(len(d))
    pending = []
    for t in range(len(d)):
        b = int((abs(tgt[t] - own[t]) > edges).sum())
        out[t] = tgt[t] if E[b] >= 0 else own[t]
        pending.append((b, (price[t] - own[t]) ** 2 - (price[t] - tgt[t]) ** 2))   # gain, known once P_t is
        if len(pending) > lag:
            bb, gain = pending.pop(0)
            E[bb] = p["memory"] * E[bb] + (1 - p["memory"]) * gain
    return out


def _fc_satisfice(d, g, p, x):
    """Satisficing: update to the adaptive target only after a forecast error larger than an adaptive aspiration."""
    tgt = _adaptive_path(d["price_prev"].to_numpy(), p["lam"])
    own, err = d["own_prev"].to_numpy(), np.abs(d["own_err_prev"].to_numpy())
    A, out = err[0], np.empty(len(d))
    for t in range(len(d)):
        out[t] = tgt[t] if err[t] > A else own[t]
        A = A + p["alpha"] * (err[t] - A)
    return out


def _fc_imitate(d, g, p, x):
    """Imitation: move toward last period's forecast of the most accurate member of the group."""
    return d["own_prev"].to_numpy() + p["p"] * (d["best_prev"] - d["own_prev"]).to_numpy()


FORECAST_RULES: List[Rule] = [
    Rule("naive", "Naive expectations", "Cobweb theory", {}, _fc_naive),
    Rule("adaptive", "Adaptive expectations", "Cobweb theory", {"lam": np.linspace(0.05, 1.0, 20)}, _fc_adaptive),
    Rule("rational", "Rational expectations (constant equilibrium forecast)", "Neoclassical optimization", {},
         _fc_rational),
    Rule("trend", "Trend following", "Simple heuristics", {"gamma": np.linspace(-1.0, 1.5, 26)}, _fc_trend),
    Rule("anchor", "Anchoring and adjustment", "Simple heuristics", {"gamma": np.linspace(-1.0, 1.5, 26)}, _fc_anchor),
    Rule("keep", "Keep own previous forecast (rule B)", "Benchmark", {}, _fc_keep),
    Rule("band", "Inaction band around the adaptive target", "Real options",
         {"lam": np.linspace(0.1, 1.0, 10), "k": np.linspace(0.0, 3.0, 13)}, _fc_band),
    Rule("rc", "Reliability condition over the adaptive target", "Heiner: reliability condition",
         {"lam": np.linspace(0.1, 1.0, 10), "theta": (0.5, 1.0, 2.0, 4.0, 8.0), "memory": (0.8, 0.9, 0.97)}, _fc_rc),
    Rule("satisfice", "Update only after a large error (aspiration)", "Satisficing",
         {"lam": np.linspace(0.1, 1.0, 10), "alpha": (0.02, 0.05, 0.1, 0.2, 0.4)}, _fc_satisfice),
    Rule("imitate", "Imitate the most accurate group member", "Imitation", {"p": np.linspace(0.0, 1.0, 21)},
         _fc_imitate),
]


def prepare_forecasts(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add the lagged information each rule may use, respecting the experiment's information set.

    `forecast` is the subject's forecast of the price of that row's period. With horizon = 1 (cobweb markets) the
    latest price known when forecasting P_t is P_{t-1}; with horizon = 2 (asset-pricing experiments, where the
    forecast of P_{t+1} is made before P_t is known) it is P_{t-2}. Columns added: the latest two known prices, the
    subject's previous forecast, the error of its latest forecast whose price is known, and the forecast of the most
    accurate group member in the latest period whose price is known."""
    h = int(horizon)
    df = df.sort_values(["group", "subject", "period"]).copy()
    prices = df.groupby(["group", "period"])["price"].first()
    df["price_prev"] = [prices.get((g, t - h), np.nan) for g, t in zip(df["group"], df["period"])]
    df["price_prev2"] = [prices.get((g, t - h - 1), np.nan) for g, t in zip(df["group"], df["period"])]
    df["own_prev"] = df.groupby(["group", "subject"])["forecast"].shift(1)
    df["own_err_prev"] = (df.groupby(["group", "subject"])["price"].shift(h)
                          - df.groupby(["group", "subject"])["forecast"].shift(h))
    err = (df["forecast"] - df["price"]).abs()
    valid = df.assign(err=err).dropna(subset=["err"])
    best = valid.loc[valid.groupby(["group", "period"])["err"].idxmin()]
    best = best.set_index(["group", "period"])["forecast"]
    df["best_prev"] = [best.get((g, t - h), np.nan) for g, t in zip(df["group"], df["period"])]
    return df.dropna(subset=["price_prev", "price_prev2", "own_prev", "own_err_prev", "best_prev"]).reset_index(drop=True)


# ================================================================================================ quantity rules
def _br(R, m):
    """Best reply to rivals' total output R with inverse demand P_i = a − b(q_i + γR)."""
    return np.maximum(0.0, (m["a"] - m["c"] - m["b"] * m.get("gamma", 1.0) * R) / (2 * m["b"]))


def _q_keep(d, g, p, m):
    return d["own_prev"].to_numpy()


def _q_br(d, g, p, m):
    q = d["own_prev"].to_numpy()
    return q + p["phi"] * (_br(d["others_prev"].to_numpy(), m) - q)


def _q_nash(d, g, p, m):
    return np.full(len(d), (m["a"] - m["c"]) / (m["b"] * (2 + m.get("gamma", 1.0) * (m["n"] - 1))))


def _q_imit_best(d, g, p, m):
    q = d["own_prev"].to_numpy()
    return q + p["p"] * (d["best_q_prev"].to_numpy() - q)


def _q_imit_avg(d, g, p, m):
    q = d["own_prev"].to_numpy()
    return q + p["p"] * (d["others_prev"].to_numpy() / (m["n"] - 1) - q)


def _q_wsls(d, g, p, m):
    q, dq = d["own_prev"].to_numpy(), d["own_dq_prev"].to_numpy()
    up = d["profit_prev"].to_numpy() >= d["profit_prev2"].to_numpy()
    direction = np.where(dq == 0, 1.0, np.sign(dq)) * np.where(up, 1.0, -1.0)
    return q + p["step"] * direction


def _q_band(d, g, p, m):
    q = d["own_prev"].to_numpy()
    tgt = _br(d["others_prev"].to_numpy(), m)
    return np.where(np.abs(tgt - q) > p["k"], q + p["phi"] * (tgt - q), q)


def _q_rc(d, g, p, m):
    """Reliability condition over the partial best reply: deviate from keeping output only where moves of that size
    have paid off in the past (one-period counterfactual profit with rivals' realized output, learned)."""
    q, R = d["own_prev"].to_numpy(), d["others_prev"].to_numpy()
    R_now = d["others"].to_numpy()
    edges = p["theta"] * np.array([0.5, 1.0, 2.0, 4.0])
    E, out = np.zeros(5), np.empty(len(d))
    for t in range(len(d)):
        cand = q[t] + p["phi"] * (_br(R[t], m) - q[t])
        b = int((abs(cand - q[t]) > edges).sum())
        out[t] = cand if E[b] >= 0 else q[t]
        prof = lambda x: (max(0.0, m["a"] - m["b"] * (x + m.get("gamma", 1.0) * R_now[t])) - m["c"]) * x
        E[b] = p["memory"] * E[b] + (1 - p["memory"]) * (prof(cand) - prof(q[t]))
    return out


def _q_satisfice(d, g, p, m):
    q = d["own_prev"].to_numpy()
    tgt = _br(d["others_prev"].to_numpy(), m)
    pr = d["profit_prev"].to_numpy()
    A, out = pr[0], np.empty(len(d))
    for t in range(len(d)):
        out[t] = q[t] + p["phi"] * (tgt[t] - q[t]) if pr[t] < A else q[t]
        A = A + p["alpha"] * (pr[t] - A)
    return out


QUANTITY_RULES: List[Rule] = [
    Rule("keep", "Keep last output (rule B)", "Benchmark", {}, _q_keep),
    Rule("br", "Partial best reply", "Neoclassical optimization", {"phi": np.linspace(0.05, 1.0, 20)}, _q_br),
    Rule("nash", "Cournot–Nash output", "Neoclassical optimization", {}, _q_nash),
    Rule("imit_best", "Imitate the most profitable firm", "Imitation", {"p": np.linspace(0.0, 1.0, 21)}, _q_imit_best),
    Rule("imit_avg", "Imitate the average", "Imitation", {"p": np.linspace(0.0, 1.0, 21)}, _q_imit_avg),
    Rule("wsls", "Win-stay, lose-shift", "Simple heuristics", {"step": (1, 2, 3, 5, 8, 12, 20)}, _q_wsls),
    Rule("band", "Inaction band around the best reply", "Real options",
         {"phi": np.linspace(0.1, 1.0, 10), "k": (0, 1, 2, 4, 8, 16)}, _q_band),
    Rule("rc", "Reliability condition over the best reply", "Heiner: reliability condition",
         {"phi": np.linspace(0.1, 1.0, 10), "theta": (1, 2, 4, 8, 16), "memory": (0.8, 0.9, 0.97)}, _q_rc),
    Rule("satisfice", "Move toward the best reply only below aspiration", "Satisficing",
         {"phi": np.linspace(0.1, 1.0, 10), "alpha": (0.02, 0.05, 0.1, 0.2, 0.4)}, _q_satisfice),
]


def prepare_quantities(df: pd.DataFrame, a: float, b: float, c: float, gamma: float = 1.0) -> pd.DataFrame:
    df = df.sort_values(["group", "subject", "period"]).copy()
    tot = df.groupby(["group", "period"])["quantity"].transform("sum")
    df["others"] = tot - df["quantity"]
    df["price"] = np.maximum(0.0, a - b * (df["quantity"] + gamma * df["others"]))
    df["profit"] = (df["price"] - c) * df["quantity"]
    grp = df.groupby(["group", "subject"])
    df["own_prev"] = grp["quantity"].shift(1)
    df["own_dq_prev"] = df["own_prev"] - grp["quantity"].shift(2)
    df["others_prev"] = grp["others"].shift(1)
    df["profit_prev"] = grp["profit"].shift(1)
    df["profit_prev2"] = grp["profit"].shift(2)
    best = df.loc[df.groupby(["group", "period"])["profit"].idxmax()].set_index(["group", "period"])["quantity"]
    df["best_q_prev"] = [best.get((g, t - 1), np.nan) for g, t in zip(df["group"], df["period"])]
    return df.dropna(subset=["own_prev", "own_dq_prev", "others_prev", "profit_prev2", "best_q_prev"]
                     ).reset_index(drop=True)


# ================================================================================================ fitting
def fit_subjects(df: pd.DataFrame, rules: Sequence[Rule], target: str, extra: Dict) -> pd.DataFrame:
    """Per subject and rule: fit on the first half (grid search), score on the second half. One row per (subject,
    rule) with the fitted parameters and train/test root-mean-square error."""
    rows = []
    for (g, s), d in df.groupby(["group", "subject"]):
        d = d.reset_index(drop=True)
        n = len(d)
        if n < 6:
            continue
        cut = n // 2
        y = d[target].to_numpy(float)
        x = dict(extra, mean_price_train=float(d["price"].iloc[:cut].mean()) if "price" in d else 0.0)
        for r in rules:
            best = (np.inf, {})
            names = list(r.grid)
            for vals in (itertools.product(*[r.grid[k] for k in names]) if names else [()]):
                p = dict(zip(names, vals))
                pred = r.predict(d, None, p, x)
                err = np.mean((pred[:cut] - y[:cut]) ** 2)
                if err < best[0]:
                    best = (err, p)
            pred = r.predict(d, None, best[1], x)
            rows.append(dict(group=g, subject=s, rule=r.key, name=r.name, theory=r.theory,
                             train_rmse=float(np.sqrt(best[0])), test_rmse=float(np.sqrt(np.mean((pred[cut:] - y[cut:]) ** 2))),
                             n_test=n - cut, **{f"param:{k}": v for k, v in best[1].items()}))
    return pd.DataFrame(rows)


def summarise(fits: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rule ranking by out-of-sample error, and each subject's best-predicting rule."""
    best = fits.loc[fits.groupby(["group", "subject"])["test_rmse"].idxmin()]
    share = best["rule"].value_counts(normalize=True)
    rank = fits.groupby(["rule", "name", "theory"]).agg(test_rmse=("test_rmse", "mean"),
                                                        train_rmse=("train_rmse", "mean")).reset_index()
    rank["best_for_share"] = rank["rule"].map(share).fillna(0.0)
    return rank.sort_values("test_rmse", ignore_index=True), best[["group", "subject", "rule", "name", "theory",
                                                                   "test_rmse"]]


# ================================================================================================ synthetic data
def synthetic_forecast_data(n_groups: int = 4, group_size: int = 6, periods: int = 50, noise: float = 0.3,
                            seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Learning-to-forecast cobweb groups with known forecasting rules. Price: P_t = A − B·mean(forecasts) + shock
    with B = 0.95 (a stable cobweb; B > 1 would be unstable). Returns (data, true types)."""
    rng = np.random.default_rng(seed)
    kinds = ["naive", "adaptive", "trend", "keep", "rc", "anchor"]
    rows, types = [], []
    for g in range(n_groups):
        ks = [kinds[(g + i) % len(kinds)] for i in range(group_size)]
        lam = rng.uniform(0.3, 0.8, group_size)
        P = [5.5, 6.0]
        f_hist = {i: [6.0, 6.0] for i in range(group_size)}
        E = {i: np.zeros(5) for i in range(group_size)}
        for t in range(2, periods + 2):
            fs = []
            for i, k in enumerate(ks):
                pp, pp2, own = P[-1], P[-2], f_hist[i][-1]
                tgt = own + lam[i] * (pp - own)
                if k == "naive":
                    f = pp
                elif k == "adaptive":
                    f = tgt
                elif k == "trend":
                    f = pp + 0.8 * (pp - pp2)
                elif k == "anchor":
                    f = 0.5 * (np.mean(P) + pp) + 0.6 * (pp - pp2)
                elif k == "keep":
                    f = own
                else:
                    b = int((abs(tgt - own) > 1.0 * np.array([0.5, 1, 2, 4])).sum())
                    f = tgt if E[i][b] >= 0 else own
                fs.append(f + noise * rng.standard_normal())
            price = 13.0 - 0.95 * np.mean(fs) + 0.3 * rng.standard_normal()      # stable cobweb (slope < 1)
            for i, k in enumerate(ks):
                if k == "rc":
                    own = f_hist[i][-1]
                    tgt = own + lam[i] * (P[-1] - own)
                    b = int((abs(tgt - own) > 1.0 * np.array([0.5, 1, 2, 4])).sum())
                    E[i][b] = 0.9 * E[i][b] + 0.1 * ((price - own) ** 2 - (price - tgt) ** 2)
                f_hist[i].append(fs[i])
                rows.append(dict(group=g, subject=f"g{g}s{i}", period=t, price=price, forecast=fs[i]))
            P.append(price)
        types += [dict(group=g, subject=f"g{g}s{i}", true_rule=k) for i, k in enumerate(ks)]
    return pd.DataFrame(rows), pd.DataFrame(types)


def synthetic_cournot_data(n_groups: int = 6, group_size: int = 4, periods: int = 40, a: float = 100.0,
                           b: float = 1.0, c: float = 1.0, noise: float = 1.0, seed: int = 0
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cournot groups with known quantity rules (partial best reply, imitate the best, keep, win-stay/lose-shift)."""
    rng = np.random.default_rng(seed)
    kinds = ["br", "imit_best", "keep", "wsls"]
    rows, types = [], []
    m = dict(a=a, b=b, c=c, n=group_size)
    for g in range(n_groups):
        ks = [kinds[(g + i) % len(kinds)] for i in range(group_size)]
        q = rng.uniform(15, 30, group_size)
        last_prof = np.zeros(group_size)
        dirs = np.ones(group_size)
        prev_prof = None
        for t in range(periods):
            Q = q.sum()
            P = max(0.0, a - b * Q)
            prof = (P - c) * q
            for i in range(group_size):
                rows.append(dict(group=g, subject=f"g{g}s{i}", period=t, quantity=q[i]))
            new = q.copy()
            best = q[prof.argmax()]
            for i, k in enumerate(ks):
                R = Q - q[i]
                if k == "br":
                    new[i] = q[i] + 0.6 * (_br(R, m) - q[i])
                elif k == "imit_best":
                    new[i] = q[i] + 0.8 * (best - q[i])
                elif k == "wsls":
                    if prev_prof is not None and prof[i] < prev_prof[i]:
                        dirs[i] = -dirs[i]
                    new[i] = q[i] + 3.0 * dirs[i]
            prev_prof = prof
            q = np.maximum(0.0, new + noise * rng.standard_normal(group_size) * (np.array(ks) != "keep"))
        types += [dict(group=g, subject=f"g{g}s{i}", true_rule=k) for i, k in enumerate(ks)]
    return pd.DataFrame(rows), pd.DataFrame(types)
