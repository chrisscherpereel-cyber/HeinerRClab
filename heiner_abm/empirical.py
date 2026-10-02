"""Validation against public experimental datasets.

Five public sources can support validation (see DATASETS). None is bundled: each must be downloaded from its
repository under its own license and mapped to the columns the adapters expect. The protocol is fixed in advance and
hashed (EMPIRICAL_PLAN):
    * rules are fitted on each participant's earlier periods and evaluated on the later periods they have not seen;
    * each treatment is analyzed separately and only rules consistent with the treatment's information set are used
      (for example, no imitate-the-best rule where participants never saw rivals' individual outcomes);
    * uncertainty is reported at the participant level (participant-cluster bootstrap);
    * one primary Cournot validation, one independent forecasting validation, and the newsvendor data as an
      extension; the time-pressure data are supplementary and treated as a competence manipulation, not as a test of
      the simulation's uncertainty parameter.
Rules that predict human choices well do not show that people consciously apply Heiner's reliability condition; they
show which decision rules describe behavior, out of sample.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .analysis import ols
from .calibration import (FORECAST_RULES, QUANTITY_RULES, Rule, _br, fit_subjects, prepare_forecasts,
                          prepare_quantities, summarise)


# ================================================================================================ dataset registry
@dataclass(frozen=True)
class Dataset:
    key: str
    priority: int
    title: str
    ref: str                  # key in heiner_abm.literature.REFERENCES (the study that produced the data)
    url: str
    access: str
    licence: str
    files: str
    adapter: str              # 'cournot', 'forecast' or 'newsvendor'
    validates: str
    qualification: str
    recommended: str


DATASETS: List[Dataset] = [
    Dataset("cournot_gos", 1, "Firm-specific information and explicit collusion in experimental oligopolies",
            "gomezmartinez2016", "https://data.mendeley.com/datasets/bgpv5ynsxz/1",
            "Mendeley Data, 'Download All' (doi:10.17632/bgpv5ynsxz.1)", "CC BY-NC 3.0",
            "Session data of Cournot markets; treatments vary information about competitors and communication",
            "cournot",
            "Cournot quantity adjustment, imitation, learning from competitors' actions and outcomes, and how "
            "competitive markets become.",
            "Use the treatments without communication for the closest match; communication creates cartels, which "
            "none of the simulated rules models.",
            "Primary validation. Compare quantity-change frequency and size, responses to competitors' previous "
            "actions and outcomes, out-of-sample predictions of best-reply, imitation, inaction-band and "
            "learned-reliability rules, and aggregate output relative to the Cournot–Nash benchmark."),
    Dataset("ltf_egm", 2, "A Unified Model of Learning to Forecast (data and code)", "evans2025",
            "https://www.openicpsr.org/openicpsr/project/198204/version/V1/view",
            "openICPSR (free account login required)", "openICPSR terms of use",
            "Raw and cleaned data (MarketPrices.xls, combined_data.dta and further Stata files), programs, "
            "experimental source code", "forecast",
            "Forecast adjustment, adaptive learning and responses to structural changes in a demand-and-supply "
            "market with a production lag (one-period-ahead forecasts, as in a cobweb market).",
            "Structural changes were announced to participants, unlike the simulation's unannounced shifts: treat "
            "this as validation under announced changes, not as evidence about unexpected regime shifts.",
            "Independent forecasting validation. Test whether the forecasting rules predict adjustment speed, "
            "persistence and forecast errors before and after the changes (structural-break analysis)."),
    Dataset("ltf_ah", 3, "Evolutionary Selection of Individual Expectations and Aggregate Outcomes in Asset Pricing "
            "Experiments (replication package)", "anufriev2012",
            "https://www.openicpsr.org/openicpsr/project/114401/version/V1/view",
            "openICPSR (free account login required)", "openICPSR terms of use",
            "hstv05.mat and hstv08.mat: data from the asset-pricing experiments of Hommes et al. (2005, 2008), "
            "with model and fitting code", "forecast",
            "Heterogeneous forecasting rules, switching between rules, convergence and oscillations.",
            "These are asset-pricing experiments, not the 2007 cobweb study: participants forecast two periods "
            "ahead, and the positive (asset-market) feedback must be preserved. Treating the data as cobweb data "
            "would weaken the validation.",
            "Compare held-out forecasting accuracy across adaptive expectations, trend following, anchoring, "
            "imitation and restricted-adjustment rules, with the forecast horizon set to 2."),
    Dataset("newsvendor_bdp", 4, "Pull-to-center is not just for newsvendors (raw-data supplement S2)",
            "brokesova2022", "https://doi.org/10.1371/journal.pone.0264183.s002",
            "PLOS ONE supporting information (XLSX)", "As published with the article (PLOS ONE)",
            "Raw data: 105 participants, 100 decision periods each, newsvendor and mathematically equivalent pricing "
            "treatments, with feedback on realized demand (or threshold) and profit", "newsvendor",
            "Inventory decisions: order adjustment, demand chasing, deviations from the optimal quantity and "
            "performance under asymmetric overage and underage costs.",
            "Needs the inventory adapter, and the experiment's own demand distribution and costs must be entered "
            "rather than the simulation's shifting-demand environment.",
            "Extension. Use the newsvendor treatments to evaluate order adjustment, demand chasing, pull-to-center "
            "and held-out performance of the decision rules."),
    Dataset("ltf_time", 5, "Pushed to perform: Time pressure in long run Learning-to-Forecast experiments",
            "creed2021", "https://uvaauas.figshare.com/articles/dataset/Pushed_to_perform_Time_pressure_in_long_"
            "run_Learning-to-Forecast_experiments/13948409",
            "University of Amsterdam figshare, 'Download all' (4.7 MB)", "Restrictive license: check reuse terms",
            "Raw data for 13 treatments", "forecast",
            "Whether time pressure changes reliance on simple or restricted forecasting rules.",
            "Time pressure manipulates competence (time to think), not uncertainty; a time-pressure effect is not a "
            "direct test of the simulation's uncertainty parameter.",
            "Supplementary. Compare the share of participants best predicted by simple or restricted rules between "
            "high and low time pressure."),
]
DATASET_BY_KEY = {d.key: d for d in DATASETS}

RESTRICTED_FORECAST = ("keep", "band", "rc")
SIMPLE_FORECAST = ("naive", "adaptive", "keep", "band", "rc")

EMPIRICAL_HYPOTHESES = (
    ("V1", "Cournot (primary): the learned reliability condition predicts participants' later quantities better than "
           "the partial best reply it restricts.",
     "Per participant, test-window RMSE of the partial best reply minus that of the reliability-condition rule; "
     "participant-cluster bootstrap 95% CI. Supported if the CI lies above 0."),
    ("V2", "Cournot (primary), generative validity: markets simulated with each participant's fitted rule reproduce "
           "the human change frequency and output relative to Cournot–Nash.",
     "Simulated markets (same groups, parameters and starting quantities; 20 replications) give a 95% interval for "
     "each statistic. Supported if both human statistics lie inside their simulated intervals."),
    ("V3", "Forecasting (independent): the learned reliability condition predicts later forecasts better than the "
           "adaptive expectations it restricts.",
     "Per participant, test-window RMSE of adaptive expectations minus that of the reliability-condition rule; "
     "participant-cluster bootstrap 95% CI. Supported if the CI lies above 0."),
    ("V4", "Forecasting: expectations are heterogeneous: no single rule is best for a majority of participants.",
     "Share of participants whose best rule is the most common best rule; bootstrap 95% CI. Supported if the upper "
     "bound lies below 0.5."),
    ("V5", "Newsvendor (extension): orders show pull-to-center and demand chasing.",
     "Pull-to-center ratio (mean order − mean demand) / (optimal order − mean demand) per participant, and the "
     "demand-chasing slope of the order change on last period's demand minus order. Supported if the ratio's CI "
     "lies strictly between 0 and 1 and the slope's CI lies above 0."),
    ("V6", "Newsvendor (extension): the learned reliability condition predicts later orders better than the demand "
           "chasing it restricts.",
     "Per participant, test-window RMSE of demand chasing minus that of the reliability-condition rule; "
     "participant-cluster bootstrap 95% CI. Supported if the CI lies above 0."),
    ("V7", "Time pressure (supplementary, a competence manipulation): the share of participants best predicted by "
           "simple or restricted rules differs between high and low time pressure.",
     "Difference in that share (high − low), bootstrap 95% CI over participants. Supported if the CI excludes 0."),
)


@dataclass(frozen=True)
class EmpiricalPlan:
    version: str = "1.0"
    split: float = 0.5                 # share of each participant's periods used for fitting (earlier periods)
    min_periods: int = 10
    n_boot: int = 2000
    gen_reps: int = 20
    alpha: float = 0.05
    seed: int = 9009
    aggregation: str = ("Each dataset and treatment is tested separately; a hypothesis tested on several counts as "
                        "supported only if it is supported in every one.")
    hypotheses: Tuple[Tuple[str, str, str], ...] = EMPIRICAL_HYPOTHESES

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "datasets": [d.key for d in DATASETS]}, sort_keys=True, indent=1)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


EMPIRICAL_PLAN = EmpiricalPlan()


# ================================================================================================ reading files
def read_table(name: str, data: bytes) -> Dict[str, pd.DataFrame]:
    """Read an uploaded file into one or more tables: CSV, Excel (every sheet), Stata (.dta) or MATLAB (.mat, every
    numeric 2-D array, rows = observations)."""
    ext = name.lower().rsplit(".", 1)[-1]
    buf = io.BytesIO(data)
    if ext in ("csv", "txt"):
        return {name: pd.read_csv(buf, sep=None, engine="python")}
    if ext in ("xlsx", "xlsm", "xls"):
        return {f"{name} · {k}": v for k, v in pd.read_excel(buf, sheet_name=None).items()}
    if ext == "dta":
        return {name: pd.read_stata(buf)}
    if ext == "mat":
        from scipy.io import loadmat
        out = {}
        for k, v in loadmat(buf).items():
            if k.startswith("__") or not isinstance(v, np.ndarray) or v.dtype.kind not in "fiu" or v.ndim > 2:
                continue
            arr = v if v.ndim == 2 else v[:, None]
            out[f"{name} · {k}"] = pd.DataFrame(arr, columns=[f"{k}_{j}" for j in range(arr.shape[1])])
        return out
    raise ValueError(f"Unsupported file type: .{ext}")


def wide_to_long(df: pd.DataFrame, value: str, group: str = "group", period: str = "period") -> pd.DataFrame:
    """Reshape a wide table (one column per participant) into long form: group, subject, period, value."""
    ids = [c for c in (group, period) if c in df.columns]
    long = df.melt(id_vars=ids, var_name="subject", value_name=value)
    if group not in long:
        long[group] = 0
    return long


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, str], keep: Sequence[str] = ()) -> pd.DataFrame:
    """Rename the user's columns to the adapter's names and drop rows with missing required values."""
    cols = {v: k for k, v in mapping.items() if v}
    out = df.rename(columns=cols)[[k for k, v in mapping.items() if v] + [c for c in keep if c in df.columns]]
    return out.dropna(subset=[k for k, v in mapping.items() if v]).reset_index(drop=True)


# ================================================================================================ statistics
def cluster_boot(values: np.ndarray, n_boot: int, seed: int = 0) -> Tuple[float, float, float]:
    """Mean and 95% bootstrap CI, resampling participants (one value per participant)."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n_boot, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_rule_gain(fits: pd.DataFrame, flexible: str, restricted: str, n_boot: int) -> Dict:
    """Per participant: test RMSE of the flexible rule minus that of its restricted version (positive = the
    restricted rule predicts better), with a participant-cluster bootstrap CI."""
    w = fits.pivot_table(index=["group", "subject"], columns="rule", values="test_rmse")
    if flexible not in w or restricted not in w:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0, share=np.nan)
    d = (w[flexible] - w[restricted]).dropna().to_numpy()
    m, lo, hi = cluster_boot(d, n_boot)
    return dict(mean=m, lo=lo, hi=hi, n=len(d), share=float((d > 0).mean()) if len(d) else np.nan)


def modal_share(best: pd.DataFrame, n_boot: int, seed: int = 0) -> Dict:
    rules = best["rule"].to_numpy()
    if len(rules) == 0:
        return dict(rule=None, share=np.nan, lo=np.nan, hi=np.nan)
    vals, counts = np.unique(rules, return_counts=True)
    top = vals[np.argmax(counts)]
    m, lo, hi = cluster_boot((rules == top).astype(float), n_boot, seed)
    return dict(rule=str(top), share=m, lo=lo, hi=hi)


# ================================================================================================ Cournot
def cournot_rules(individual_info: bool) -> List[Rule]:
    """Rules consistent with the treatment's information set: imitating the most profitable firm needs individual
    information about rivals."""
    return [r for r in QUANTITY_RULES if individual_info or r.key != "imit_best"]


def nash_quantity(a: float, b: float, c: float, n: int, gamma: float = 1.0) -> float:
    return (a - c) / (b * (2 + gamma * (n - 1)))


def cournot_moments(df: pd.DataFrame, a: float, b: float, c: float, n: int, gamma: float = 1.0) -> Dict[str, float]:
    """Change frequency, mean absolute change, total output relative to Cournot–Nash, and responses to competitors
    (slopes of the quantity change on the gap to the best reply and to the most profitable firm's quantity)."""
    d = prepare_quantities(df, a, b, c, gamma)
    q_nash = nash_quantity(a, b, c, n, gamma)
    dq = d["quantity"] - d["own_prev"]
    tot = df.groupby(["group", "period"])["quantity"].sum()
    out = dict(change_rate=float((dq.abs() > 1e-9).mean()), mean_abs_change=float(dq.abs().mean()),
               output_vs_nash=float(tot.mean() / (n * q_nash)))
    m = dict(a=a, b=b, c=c, n=n, gamma=gamma)
    X1 = _br(d["others_prev"].to_numpy(), m) - d["own_prev"].to_numpy()
    X2 = (d["best_q_prev"] - d["own_prev"]).to_numpy()
    if len(d) > 5 and np.std(X1) > 0 and np.std(X2) > 0:
        tab, _ = ols(dq.to_numpy(float), [X1, X2], ["gap_best_reply", "gap_best_firm"])
        out.update(slope_best_reply=float(tab.loc[1, "coef"]), slope_imitation=float(tab.loc[2, "coef"]))
    return out


def _group_frame(Q: np.ndarray, a: float, b: float, c: float, subjects: Sequence, gamma: float = 1.0) -> pd.DataFrame:
    """The columns of prepare_quantities for one group, from a (periods × firms) quantity array (fast)."""
    T, n = Q.shape
    tot = Q.sum(1)
    P = np.maximum(0.0, a - b * (Q + gamma * (tot[:, None] - Q)))       # each firm's own price
    prof = (P - c) * Q
    best = Q[np.arange(T), prof.argmax(1)]
    rows = []
    for i, s in enumerate(subjects):
        for t in range(2, T):
            rows.append(dict(subject=s, period=t, quantity=Q[t, i], others=tot[t] - Q[t, i],
                             own_prev=Q[t - 1, i], own_dq_prev=Q[t - 1, i] - Q[t - 2, i],
                             others_prev=tot[t - 1] - Q[t - 1, i], profit_prev=prof[t - 1, i],
                             profit_prev2=prof[t - 2, i], best_q_prev=best[t - 1]))
    return pd.DataFrame(rows)


def quantity_step(df: pd.DataFrame) -> float:
    """The grid of the experiment's choices: 1 if every quantity is a whole number, else 0 (continuous)."""
    q = df["quantity"].to_numpy(float)
    return 1.0 if np.allclose(q, np.round(q)) else 0.0


def simulate_from_fits(df: pd.DataFrame, fits: pd.DataFrame, a: float, b: float, c: float, n: int, reps: int,
                       seed: int = 0, step: Optional[float] = None, gamma: float = 1.0) -> pd.DataFrame:
    """Generative check: every human group is re-run with each participant replaced by the rule that predicts it best
    (with its fitted parameters and its in-sample prediction error as decision noise), starting from the humans'
    first two periods. Simulated choices are rounded to the experiment's choice grid (`step`, inferred from the data
    by default). Returns the moments of each replication."""
    step = quantity_step(df) if step is None else step
    rules = {r.key: r for r in QUANTITY_RULES}
    best = fits.loc[fits.groupby(["group", "subject"])["test_rmse"].idxmin()].set_index(["group", "subject"])
    m = dict(a=a, b=b, c=c, n=n, gamma=gamma)
    rng = np.random.default_rng(seed)
    out = []
    for r in range(reps):
        frames = []
        for g, d in df.groupby("group"):
            subs = sorted(d["subject"].unique())
            if not all((g, s) in best.index for s in subs):
                continue
            T = int(d["period"].nunique())
            piv = d.pivot_table(index="period", columns="subject", values="quantity").sort_index()[subs].to_numpy()
            Q = np.full((T, len(subs)), np.nan)
            Q[:2] = piv[:2]
            for t in range(2, T):
                Q[t] = Q[t - 1]                      # placeholder for the current row (not used by any rule)
                fr = _group_frame(Q[: t + 1], a, b, c, subs, gamma)
                for i, s in enumerate(subs):
                    row = best.loc[(g, s)]
                    params = {k[6:]: row[k] for k in row.index if k.startswith("param:") and pd.notna(row[k])}
                    fs = fr[fr["subject"] == s].reset_index(drop=True)
                    pred = rules[row["rule"]].predict(fs, None, params, m)[-1]
                    x = max(0.0, pred + row["train_rmse"] * rng.standard_normal())
                    Q[t, i] = np.round(x / step) * step if step > 0 else x
            frames.append(pd.DataFrame([dict(group=g, subject=s, period=t, quantity=Q[t, i])
                                        for i, s in enumerate(subs) for t in range(T)]))
        if frames:
            out.append(dict(rep=r, **cournot_moments(pd.concat(frames, ignore_index=True), a, b, c, n, gamma)))
    return pd.DataFrame(out)


# ================================================================================================ newsvendor
@dataclass(frozen=True)
class NewsvendorSpec:
    """Demand distribution and costs of the experiment: overage cost per unit left over (co) and underage cost per
    unit of unmet demand (cu). With price p and unit cost w: co = w, cu = p − w."""
    dist: str = "uniform"          # 'uniform' (low, high) or 'normal' (mean, sd)
    low: float = 0.0
    high: float = 100.0
    mean: float = 50.0
    sd: float = 20.0
    co: float = 1.0
    cu: float = 3.0

    @property
    def fractile(self) -> float:
        return self.cu / (self.cu + self.co)

    @property
    def mu(self) -> float:
        return (self.low + self.high) / 2 if self.dist == "uniform" else self.mean

    @property
    def sigma(self) -> float:
        return (self.high - self.low) / np.sqrt(12) if self.dist == "uniform" else self.sd

    @property
    def optimal(self) -> float:
        if self.dist == "uniform":
            return self.low + self.fractile * (self.high - self.low)
        from scipy.stats import norm
        return float(self.mean + self.sd * norm.ppf(self.fractile))


def nv_cost(q, d, spec: NewsvendorSpec):
    return spec.co * np.maximum(q - d, 0.0) + spec.cu * np.maximum(d - q, 0.0)


def prepare_newsvendor(df: pd.DataFrame) -> pd.DataFrame:
    """Columns: group (optional), subject, period, order, demand. Adds last period's order and demand."""
    df = df.copy()
    if "group" not in df:
        df["group"] = 0
    df = df.sort_values(["group", "subject", "period"])
    g = df.groupby(["group", "subject"])
    df["own_prev"] = g["order"].shift(1)
    df["demand_prev"] = g["demand"].shift(1)
    df["price"] = df["demand"]          # fit_subjects needs a 'price' column for its training mean (unused here)
    return df.dropna(subset=["own_prev", "demand_prev"]).reset_index(drop=True)


def _nv_chase_target(d, p):
    q = d["own_prev"].to_numpy()
    return q + p["beta"] * (d["demand_prev"].to_numpy() - q)


def _nv_optimal(d, g, p, x):
    return np.full(len(d), x["spec"].optimal)


def _nv_mean(d, g, p, x):
    return np.full(len(d), x["spec"].mu)


def _nv_ptc(d, g, p, x):
    s = x["spec"]
    return np.full(len(d), s.optimal + p["alpha"] * (s.mu - s.optimal))


def _nv_chase(d, g, p, x):
    return _nv_chase_target(d, p)


def _nv_anchor(d, g, p, x):
    """Anchoring on mean demand and adjusting toward last period's demand (Schweitzer & Cachon 2000)."""
    s = x["spec"]
    return (1 - p["w"]) * (s.optimal + p["alpha"] * (s.mu - s.optimal)) + p["w"] * d["demand_prev"].to_numpy()


def _nv_keep(d, g, p, x):
    return d["own_prev"].to_numpy()


def _nv_band(d, g, p, x):
    q, tgt = d["own_prev"].to_numpy(), _nv_chase_target(d, p)
    return np.where(np.abs(d["demand_prev"].to_numpy() - q) > p["k"] * x["spec"].sigma, tgt, q)


def _nv_rc(d, g, p, x):
    """Reliability condition over demand chasing: keep last period's order unless moves of that size have paid off
    in the past (cost saved against the realized demand, learned with exponential memory)."""
    s = x["spec"]
    q, tgt, dem = d["own_prev"].to_numpy(), _nv_chase_target(d, p), d["demand"].to_numpy()
    edges = p["theta"] * s.sigma * np.array([0.5, 1.0, 2.0, 4.0]) / 4
    E, out = np.zeros(5), np.empty(len(d))
    for t in range(len(d)):
        b = int((abs(tgt[t] - q[t]) > edges).sum())
        out[t] = tgt[t] if E[b] >= 0 else q[t]
        gain = nv_cost(q[t], dem[t], s) - nv_cost(tgt[t], dem[t], s)       # known once demand is realized
        E[b] = p["memory"] * E[b] + (1 - p["memory"]) * gain
    return out


NEWSVENDOR_RULES: List[Rule] = [
    Rule("optimal", "Optimal (critical-fractile) order", "Neoclassical optimization", {}, _nv_optimal),
    Rule("mean", "Order mean demand", "Simple heuristics", {}, _nv_mean),
    Rule("ptc", "Pull-to-center (between optimal and mean)", "Behavioral bias", {"alpha": np.linspace(0, 1, 21)},
         _nv_ptc),
    Rule("chase", "Demand chasing (adjust toward last demand)", "Cobweb theory", {"beta": np.linspace(0.05, 1, 20)},
         _nv_chase),
    Rule("anchor", "Anchor on the pulled-to-center order, adjust toward last demand", "Simple heuristics",
         {"alpha": np.linspace(0, 1, 11), "w": np.linspace(0, 1, 11)}, _nv_anchor),
    Rule("keep", "Keep last order (rule B)", "Benchmark", {}, _nv_keep),
    Rule("band", "Inaction band around demand chasing", "Real options",
         {"beta": np.linspace(0.1, 1, 10), "k": np.linspace(0, 2, 9)}, _nv_band),
    Rule("rc", "Reliability condition over demand chasing", "Heiner: reliability condition",
         {"beta": np.linspace(0.1, 1, 10), "theta": (0.5, 1, 2, 4, 8), "memory": (0.8, 0.9, 0.97)}, _nv_rc),
]


def newsvendor_stats(df: pd.DataFrame, spec: NewsvendorSpec, n_boot: int) -> Dict:
    """Per-participant pull-to-center ratio and demand-chasing slope, with participant-bootstrap CIs."""
    d = prepare_newsvendor(df)
    ptc, chase, change = [], [], []
    for _, s in d.groupby(["group", "subject"]):
        if abs(spec.optimal - spec.mu) > 1e-9:
            ptc.append((s["order"].mean() - spec.mu) / (spec.optimal - spec.mu))
        x = (s["demand_prev"] - s["own_prev"]).to_numpy(float)
        y = (s["order"] - s["own_prev"]).to_numpy(float)
        if len(s) > 3 and np.std(x) > 0:
            chase.append(float(np.polyfit(x, y, 1)[0]))
        change.append(float((np.abs(y) > 1e-9).mean()))
    return dict(ptc=cluster_boot(ptc, n_boot), chase=cluster_boot(chase, n_boot), change=cluster_boot(change, n_boot),
                optimal=spec.optimal, mu=spec.mu, n=len(change))


# ================================================================================================ structural breaks
def break_analysis(df: pd.DataFrame, break_period: int, window: int = 5) -> pd.DataFrame:
    """Forecasting data around an (announced) structural change: per participant, mean absolute forecast error and
    the share of periods with a forecast change, in the windows before and after the change."""
    d = df.sort_values(["group", "subject", "period"]).copy()
    d["abs_err"] = (d["forecast"] - d["price"]).abs()
    d["changed"] = d.groupby(["group", "subject"])["forecast"].diff().abs() > 1e-9
    pre = d[(d["period"] >= break_period - window) & (d["period"] < break_period)]
    post = d[(d["period"] >= break_period) & (d["period"] < break_period + window)]
    late = d[(d["period"] >= break_period + window) & (d["period"] < break_period + 2 * window)]
    rows = []
    for lab, part in (("before", pre), ("first window after", post), ("second window after", late)):
        g = part.groupby(["group", "subject"]).agg(abs_err=("abs_err", "mean"), change_rate=("changed", "mean"))
        rows.append(dict(window=lab, mean_abs_error=float(g["abs_err"].mean()) if len(g) else np.nan,
                         change_rate=float(g["change_rate"].mean()) if len(g) else np.nan, participants=len(g)))
    return pd.DataFrame(rows)


# ================================================================================================ verdicts
def _v(ok: Optional[bool], text: str) -> Tuple[Optional[bool], str]:
    return ok, text


def _is_result(v) -> bool:
    return isinstance(v, dict) and ("fits" in v or "diff" in v)


def _named(v) -> Dict[str, Dict]:
    """A single analysis result, or a {label: result} dict of results (one per dataset or treatment)."""
    if not v:
        return {}
    return {"": v} if _is_result(v) else {k: x for k, x in v.items() if _is_result(x)}


def _cournot_checks(c: Dict) -> List[Tuple[Optional[bool], str]]:
    g = c["gain_rc"]
    out = [_v(bool(g["lo"] > 0), f"best reply − RC test RMSE {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}], RC "
              f"better for {g['share']:.0%} of {g['n']} participants")]
    sim, hum = c.get("sim"), c["moments"]
    if sim is not None and len(sim):
        parts, ok = [], True
        for k in ("change_rate", "output_vs_nash"):
            lo, hi = np.percentile(sim[k], [2.5, 97.5])
            ok &= bool(lo <= hum[k] <= hi)
            parts.append(f"{k.replace('_', ' ')}: human {hum[k]:.3f}, simulated [{lo:.3f}, {hi:.3f}]")
        out.append(_v(ok, "; ".join(parts)))
    else:
        out.append(_v(None, "generative check not run"))
    return out


def _forecast_checks(f: Dict) -> List[Tuple[Optional[bool], str]]:
    g, ms = f["gain_rc"], f["modal"]
    return [_v(bool(g["lo"] > 0), f"adaptive − RC test RMSE {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}], RC "
               f"better for {g['share']:.0%} of {g['n']} participants"),
            _v(bool(ms["hi"] < 0.5), f"most common best rule '{ms['rule']}' fits {ms['share']:.0%} "
               f"[{ms['lo']:.0%}, {ms['hi']:.0%}] of participants")]


def _newsvendor_checks(nv: Dict) -> List[Tuple[Optional[bool], str]]:
    s, g = nv["stats"], nv["gain_rc"]
    p, ch = s["ptc"], s["chase"]
    return [_v(bool(p[1] > 0 and p[2] < 1 and ch[1] > 0),
               f"pull-to-center ratio {p[0]:.2f} [{p[1]:.2f}, {p[2]:.2f}]; demand-chasing slope {ch[0]:+.2f} "
               f"[{ch[1]:+.2f}, {ch[2]:+.2f}]; n = {s['n']}"),
            _v(bool(g["lo"] > 0), f"chasing − RC test RMSE {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}], RC "
               f"better for {g['share']:.0%} of {g['n']} participants")]


def _time_checks(t: Dict) -> List[Tuple[Optional[bool], str]]:
    if t.get("diff") is None:
        return [_v(None, "conditions not found")]
    m, lo, hi = t["diff"]
    return [_v(bool(lo > 0 or hi < 0), f"share best predicted by simple/restricted rules, high − low time pressure "
               f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}] (high {t['high']:.0%}, low {t['low']:.0%}"
               f"{', paired by participant' if t.get('paired') else ''})")]


def evaluate_details(results: Dict, plan: EmpiricalPlan = EMPIRICAL_PLAN) -> pd.DataFrame:
    """One row per hypothesis and dataset/treatment. results maps 'cournot', 'forecast', 'newsvendor' and 'time' to
    one result or to {label: result} for several datasets or treatments."""
    spec = (("cournot", ("V1", "V2"), _cournot_checks), ("forecast", ("V3", "V4"), _forecast_checks),
            ("newsvendor", ("V5", "V6"), _newsvendor_checks), ("time", ("V7",), _time_checks))
    rows = []
    for kind, ids, fn in spec:
        for label, res in _named(results.get(kind)).items():
            for hid, (ok, text) in zip(ids, fn(res)):
                rows.append(dict(id=hid, data=label, ok=ok, result=text, synthetic=bool(res.get("synthetic"))))
    return pd.DataFrame(rows, columns=["id", "data", "ok", "result", "synthetic"])


def evaluate_empirical(results: Dict, plan: EmpiricalPlan = EMPIRICAL_PLAN) -> pd.DataFrame:
    """Verdict per hypothesis. A hypothesis tested on several datasets or treatments counts as supported only if it
    is supported in every one analyzed (plan.aggregation); without data it is 'not tested'."""
    det = evaluate_details(results, plan)
    out = []
    for hid, hyp, rule in plan.hypotheses:
        d = det[(det["id"] == hid) & det["ok"].notna()]
        if d.empty:
            verdict, text = "not tested", "no data analyzed"
        else:
            verdict = "supported" if bool(d["ok"].all()) else "not supported"
            text = "; ".join((f"{r.data}: " if r.data else "") + r.result for r in d.itertuples())
        out.append(dict(id=hid, hypothesis=hyp, decision_rule=rule, verdict=verdict, result=text))
    return pd.DataFrame(out)


# ================================================================================================ analyses
def run_cournot(df: pd.DataFrame, a: float, b: float, c: float, n: int, individual_info: bool,
                generative: bool = True, plan: EmpiricalPlan = EMPIRICAL_PLAN, gamma: float = 1.0) -> Dict:
    data = prepare_quantities(df, a, b, c, gamma)
    fits = fit_subjects(data, cournot_rules(individual_info), "quantity", dict(a=a, b=b, c=c, n=n, gamma=gamma))
    rank, best = summarise(fits)
    out = dict(fits=fits, rank=rank, best=best, moments=cournot_moments(df, a, b, c, n, gamma),
               gain_rc=paired_rule_gain(fits, "br", "rc", plan.n_boot),
               gain_band=paired_rule_gain(fits, "br", "band", plan.n_boot))
    if generative:
        out["sim"] = simulate_from_fits(df, fits, a, b, c, n, plan.gen_reps, plan.seed, gamma=gamma)
    return out


def run_forecast(df: pd.DataFrame, horizon: int, fundamental: Optional[float] = None,
                 plan: EmpiricalPlan = EMPIRICAL_PLAN) -> Dict:
    data = prepare_forecasts(df, horizon).dropna(subset=["forecast", "price"]).reset_index(drop=True)
    extra = dict(horizon=horizon, **({"fundamental": fundamental} if fundamental is not None else {}))
    fits = fit_subjects(data, FORECAST_RULES, "forecast", extra)
    rank, best = summarise(fits)
    return dict(fits=fits, rank=rank, best=best, gain_rc=paired_rule_gain(fits, "adaptive", "rc", plan.n_boot),
                gain_band=paired_rule_gain(fits, "adaptive", "band", plan.n_boot),
                modal=modal_share(best, plan.n_boot))


def run_newsvendor(df: pd.DataFrame, spec: NewsvendorSpec, plan: EmpiricalPlan = EMPIRICAL_PLAN) -> Dict:
    data = prepare_newsvendor(df)
    fits = fit_subjects(data, NEWSVENDOR_RULES, "order", dict(spec=spec))
    rank, best = summarise(fits)
    return dict(fits=fits, rank=rank, best=best, stats=newsvendor_stats(df, spec, plan.n_boot),
                gain_rc=paired_rule_gain(fits, "chase", "rc", plan.n_boot),
                gain_band=paired_rule_gain(fits, "chase", "band", plan.n_boot))


def run_time_pressure(df: pd.DataFrame, horizon: int, treatment: str, high: str, low: str,
                      plan: EmpiricalPlan = EMPIRICAL_PLAN) -> Dict:
    """Share of participants best predicted by simple or restricted rules, high minus low time pressure. When the same
    participants appear in both conditions (a within-subject design) the bootstrap is paired by participant."""
    best = {}
    for lab, val in (("high", high), ("low", low)):
        part = df[df[treatment].astype(str) == str(val)]
        if part.empty:
            return dict(diff=None)
        data = prepare_forecasts(part, horizon).dropna(subset=["forecast", "price"]).reset_index(drop=True)
        _, b = summarise(fit_subjects(data, FORECAST_RULES, "forecast", dict(horizon=horizon)))
        best[lab] = b.assign(simple=b["rule"].isin(SIMPLE_FORECAST).astype(float)).set_index("subject")["simple"]
    rng = np.random.default_rng(plan.seed)
    hi_, lo_ = best["high"], best["low"]
    both = hi_.index.intersection(lo_.index)
    if len(both) >= 5:
        d = (hi_.loc[both] - lo_.loc[both]).to_numpy()
        boots = rng.choice(d, (plan.n_boot, len(d))).mean(1)
        paired = True
    else:
        boots = np.array([rng.choice(hi_.to_numpy(), len(hi_)).mean() - rng.choice(lo_.to_numpy(), len(lo_)).mean()
                          for _ in range(plan.n_boot)])
        paired = False
    return dict(diff=(float(hi_.mean() - lo_.mean()), float(np.percentile(boots, 2.5)),
                      float(np.percentile(boots, 97.5))), high=float(hi_.mean()), low=float(lo_.mean()),
                paired=paired, n_high=len(hi_), n_low=len(lo_), n_both=len(both))


# ================================================================================================ synthetic data
def synthetic_newsvendor_data(spec: NewsvendorSpec = NewsvendorSpec(), n_subjects: int = 16, periods: int = 60,
                              noise: float = 2.0, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulated newsvendor participants with known rules (for checking the adapter; not evidence about people)."""
    rng = np.random.default_rng(seed)
    kinds = ["ptc", "chase", "keep", "optimal", "anchor", "rc"]
    rows, truth = [], []
    for i in range(n_subjects):
        k = kinds[i % len(kinds)]
        q = spec.mu
        dem = (rng.uniform(spec.low, spec.high, periods) if spec.dist == "uniform"
               else np.maximum(0, rng.normal(spec.mean, spec.sd, periods)))
        E = np.zeros(5)
        for t in range(periods):
            if t > 0:
                dp = dem[t - 1]
                if k == "ptc":
                    q = spec.optimal + 0.5 * (spec.mu - spec.optimal)
                elif k == "chase":
                    q = q + 0.5 * (dp - q)
                elif k == "optimal":
                    q = spec.optimal
                elif k == "anchor":
                    q = 0.6 * (spec.optimal + 0.4 * (spec.mu - spec.optimal)) + 0.4 * dp
                elif k == "rc":
                    tgt = q + 0.5 * (dp - q)
                    b = int((abs(tgt - q) > 2 * spec.sigma * np.array([0.5, 1, 2, 4]) / 4).sum())
                    new = tgt if E[b] >= 0 else q
                    E[b] = 0.9 * E[b] + 0.1 * (nv_cost(q, dem[t], spec) - nv_cost(tgt, dem[t], spec))
                    q = new
            order = q if k == "keep" else max(0.0, q + noise * rng.standard_normal())
            rows.append(dict(subject=f"nv{i:02d}", period=t, order=order, demand=dem[t]))
            if k != "keep":
                q = order if k in ("chase", "rc") else q
        truth.append(dict(group=0, subject=f"nv{i:02d}", true_rule=k))
    return pd.DataFrame(rows), pd.DataFrame(truth)
