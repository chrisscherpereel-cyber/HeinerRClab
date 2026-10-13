"""Incremental construct validity of the proposed learnability index R.

PROPOSED EXTENSION, and an analysis of the laboratory's own construct. The simulation is unchanged: this module
imports heiner_abm.learnability and does not alter the index, the task or the policies.

THE QUESTION
    heiner_abm.learnability proposes an index R and shows that a gate's advantage falls as R rises (H1), and that
    log10 R adds cross-validated R-squared beyond log sigma and log(1 + tau) (H2). Two covariates is a low bar. A
    negative slope on R is consistent with R being a distinct mechanism, and equally consistent with R being a
    convenient summary of quantities already known to matter. Distinguishing those is what this module is for.

WHAT IS PRESPECIFIED HERE, AND WHEN
    Everything below - the index, the comparison model, the interactions, the two held-out protocols, the matched
    designs, the robustness probes and the resampling scheme - is fixed in ValidityPlan and hashed before the
    confirmatory configurations are ever run. The index is imported unchanged from heiner_abm.learnability: it is
    retained as a NAMED PROPOSED INDEX and is not redefined here, before or after seeing any result.

    Exploratory and confirmatory data are separate by construction:
        exploratory   seed block 5, configurations drawn from EXPLORATORY_CONFIG_SEED. Used to build and debug
                      everything in this module. Results from it are exploratory and are labelled as such.
        confirmatory  seed block 6, configurations drawn from CONFIRMATORY_CONFIG_SEED. Run ONCE, after the plan
                      hash is fixed. No quantity in this module was chosen by looking at it.
    The study's own blocks (1 training, 2 pilot, 3 test, 4 negative controls) are untouched, so the registered
    learnability results and this analysis never share a path.

WHAT A POSITIVE RESULT WOULD AND WOULD NOT MEAN
    If R adds predictive accuracy beyond the comparison model on held-out configurations AND on held-out process
    families, R carries information those quantities do not. That is worth having, and it is the claim this module
    can support.
    It is NOT a demonstration that R names a distinct mechanism. An index can summarize several known quantities
    more usefully than they summarize themselves. The brief this module answers says so explicitly, and so does
    the verdict text: "useful summary index" and "distinct mechanism" are different conclusions, and only the
    first is within reach of a prediction contest.
    A negative slope of advantage on R is NOT by itself theoretical validation. It is the sign the construct
    predicts, and the construct was built to have it.

THE COMPARISON MODEL
    Prespecified covariates, each a quantity a reader could compute without the index:
        advantage-to-noise      |mean advantage| / s.d. of the observed gain          (pilot)
        regime duration         log10 of the expected regime length, 1 / hazard       (configuration)
        feedback availability   share of periods whose demand is observed             (configuration)
        feedback delay          periods between a demand and its observation          (configuration)
        learner memory          log10 of the effective window (1 + lambda)/(1 - lambda)  (configuration)
        effective evidence      log10 of informative observations per regime          (pilot)
        default quality         mean absolute gap between the default order and the oracle order  (pilot)
        adjustment costs        departure, switching and magnitude charges            (configuration)
    Prespecified interactions, chosen because each is a mechanism the index is supposed to summarize:
        advantage-to-noise x regime duration    evidence per regime against evidence needed
        availability x learner memory           how fast evidence arrives against how long it is kept
        departure cost x advantage-to-noise     how large an advantage must be before it is worth paying for
    No covariate or interaction is added or removed after a confirmatory result is seen.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import learnability as L
from .learnability import replace  # noqa: F401  (re-exported for robustness probes)

# ------------------------------------------------------------------------------------------------ the index
INDEX_KEY = "log10_ratio"
INDEX_NAME = "proposed learnability index R"
INDEX_NOTE = ("R = median over regimes of (observations needed to determine the sign of the advantage) / "
              "(informative observations available). Defined in heiner_abm.learnability.learnability_ratio and "
              "imported unchanged; this module never redefines it.")

# Seed blocks. The study uses 1 training, 2 pilot, 3 test, 4 negative controls; these two are new and disjoint.
EXPLORATORY_TAG = 5
CONFIRMATORY_TAG = 6
EXPLORATORY_CONFIG_SEED = 90_001
CONFIRMATORY_CONFIG_SEED = 90_002


def seeds(plan: "ValidityPlan", tag: int, j: int, n: int) -> List[int]:
    """Disjoint from heiner_abm.learnability._seeds by construction: it uses tags 1-4, this uses 5 and 6."""
    if tag not in (EXPLORATORY_TAG, CONFIRMATORY_TAG):
        raise ValueError(f"tag {tag} is not one of this module's blocks ({EXPLORATORY_TAG}, {CONFIRMATORY_TAG})")
    return [plan.seed * 1000 + tag * 100_000_000 + j * 1000 + r for r in range(n)]


# ------------------------------------------------------------------------------------------------ covariates
COVARIATES: Tuple[Tuple[str, str], ...] = (
    ("adv_noise", "|mean advantage| / s.d. of the observed gain, from pilot paths"),
    ("log_regime_len", "log10 of the expected regime length (1 / hazard)"),
    ("avail", "share of periods whose demand is observed"),
    ("delay", "periods between a demand occurring and its observation"),
    ("log_memory_window", "log10 of the learner's effective window (1 + lambda) / (1 - lambda)"),
    ("log_evidence", "log10 of informative observations per regime, from pilot paths"),
    ("default_gap", "mean |default order - oracle order|, from pilot paths"),
    ("cost", "default-departure overhead"),
    ("switch_cost", "fixed switching cost"),
    ("magnitude_cost", "magnitude cost per unit moved"),
)
INTERACTIONS: Tuple[Tuple[str, str], ...] = (
    ("adv_noise", "log_regime_len"),
    ("avail", "log_memory_window"),
    ("cost", "adv_noise"),
)
COVARIATE_KEYS = tuple(k for k, _ in COVARIATES)


def covariate_row(cfg: L.Config, pilot: Dict[str, float]) -> Dict[str, float]:
    """The comparison model's covariates for one configuration. Pilot quantities come from pilot paths only."""
    lam = float(cfg.memory)
    window = (1.0 + lam) / max(1.0 - lam, 1e-9)
    return {
        "adv_noise": float(pilot.get("adv_noise", np.nan)),
        "log_regime_len": float(np.log10(1.0 / max(cfg.hazard, 1e-9))),
        "avail": float(cfg.avail),
        "delay": float(cfg.obs_delay),
        "log_memory_window": float(np.log10(window)),
        "log_evidence": float(np.log10(max(pilot.get("n_avail", np.nan), 1.0)))
        if np.isfinite(pilot.get("n_avail", np.nan)) else np.nan,
        "default_gap": float(pilot.get("default_gap", np.nan)),
        "cost": float(cfg.cost),
        "switch_cost": float(cfg.switch_cost),
        "magnitude_cost": float(cfg.magnitude_cost),
    }


def design_matrix(df: pd.DataFrame, with_index: bool) -> Tuple[np.ndarray, List[str]]:
    """The comparison model, with or without the index added. Columns are fixed by the plan, in a fixed order.

    Constant columns are dropped (a factor held fixed in a design carries no information and would make the
    normal equations singular), and which were dropped is recoverable from the returned names.
    """
    names = list(COVARIATE_KEYS)
    cols = [df[k].to_numpy(float) for k in names]
    for a, b in INTERACTIONS:
        cols.append(df[a].to_numpy(float) * df[b].to_numpy(float))
        names.append(f"{a}_x_{b}")
    if with_index:
        cols.append(df[INDEX_KEY].to_numpy(float))
        names.append(INDEX_KEY)
    X = np.column_stack(cols)
    keep = [i for i in range(X.shape[1]) if np.nanstd(X[:, i]) > 1e-12]
    return X[:, keep], [names[i] for i in keep]


def _fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge-regularized least squares with an intercept, on standardized columns.

    Columns are centred and scaled by the TRAINING fold's own mean and s.d., so no information crosses from the
    held-out rows and the penalty means the same thing for every covariate whatever its units. The penalty is
    prespecified and identical for both models, so neither is advantaged by it; it exists because the comparison
    model has thirteen columns and these designs have tens of configurations.
    """
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    A = np.column_stack([np.ones(len(X_tr)), (X_tr - mu) / sd])
    B = np.column_stack([np.ones(len(X_te)), (X_te - mu) / sd])
    P = np.eye(A.shape[1]) * ridge
    P[0, 0] = 0.0                                             # never penalize the intercept
    beta = np.linalg.solve(A.T @ A + P, A.T @ y_tr)
    return B @ beta


def cv_r2(X: np.ndarray, y: np.ndarray, folds: Sequence[np.ndarray], ridge: float) -> float:
    """Cross-validated R-squared: out-of-fold predictions against the total sum of squares of y."""
    pred = np.full(len(y), np.nan)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(tr) < 3:
            continue
        pred[te] = _fit_predict(X[tr], y[tr], X[te], ridge)
    ok = np.isfinite(pred) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    ss_res = float(np.sum((y[ok] - pred[ok]) ** 2))
    ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def leave_one_out_folds(n: int) -> List[np.ndarray]:
    return [np.array([i]) for i in range(n)]


# ------------------------------------------------------------------------------------------------ the plan
@dataclass(frozen=True)
class ValidityPlan:
    """Frozen and hashed before the confirmatory configurations are run."""
    version: str = "1.0"
    seed: int = 20261012
    periods: int = 1200
    burn_in: int = 100
    pilot_paths: int = 8               # paths the index and the pilot covariates are estimated from
    eval_paths: int = 8                # paths the outcome is measured on
    n_configs: int = 60                # configurations PER FAMILY, drawn by Latin hypercube
    families: Tuple[str, ...] = ("jump", "switch", "drift")
    # chosen on the EXPLORATORY arm by cross-validated fit of the COMPARISON model alone, never on the
    # contrast with the index: selecting the penalty on the quantity under test would bias it
    ridge: float = 1.0
    n_boot: int = 1000
    inner_boot: int = 16               # pilot-path resamples per configuration; this is what propagates the
    #                                    uncertainty of estimating the index into the interval on the result
    matched_pairs: int = 20
    # the ranges the configurations are drawn from, wider than the study's so that delay, bias and correlation vary
    ranges: Tuple[Tuple[str, Tuple[float, float]], ...] = (
        ("sigma", (5.0, 40.0)), ("tau", (0.0, 30.0)), ("hazard", (0.002, 0.05)), ("avail", (0.2, 1.0)),
        ("memory", (0.9, 0.995)), ("obs_delay", (0.0, 6.0)), ("cost", (0.0, 4.0)))
    robustness: Tuple[Tuple[str, Tuple[float, ...]], ...] = (
        ("obs_corr", (0.0, 0.4, 0.8)),
        ("obs_bias", (0.0, 5.0, 15.0)),
        ("memory", (0.9, 0.97, 0.995)),
    )
    clip_variants: Tuple[Tuple[float, float], ...] = ((1.0, 1e6), (1.0, 1e3), (0.1, 1e9))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=1, default=str)

    @property
    def digest(self) -> str:
        return hashlib.sha256((self.to_json() + INDEX_NOTE).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ measurement
def pilot_summary(cfg: L.Config, gain: float, plan: ValidityPlan, pilot_seeds: Sequence[int]) -> Dict[str, float]:
    """Index and pilot covariates for one configuration, from pilot paths only (never from evaluation paths)."""
    r = L.learnability_ratio(cfg, gain, plan.periods, pilot_seeds, plan.burn_in)
    adv, gaps, noise = [], [], []
    for sd in pilot_seeds:
        p = L.paths(cfg, plan.periods, sd)
        c = L.candidate_orders(cfg, p, gain)
        differ = c["S_F"] != c["S_D"]
        g_true = L.payoff(c["S_F"], p["d"]) - L.payoff(c["S_D"], p["d"])
        if differ.any():
            adv.append(float(g_true[differ].mean()))
        oracle = np.maximum(p["mu"] + L.Z_FRAC * cfg.sigma, 0.0)
        gaps.append(float(np.mean(np.abs(c["S_D"] - oracle))))
        noise.append(r.get("noise_sd", np.nan))
    s = float(np.nanmean(noise)) if noise else np.nan
    mean_adv = float(np.mean(adv)) if adv else np.nan
    return {**r, "adv_noise": abs(mean_adv) / s if (s and np.isfinite(s) and s > 0) else np.nan,
            "mean_advantage": mean_adv, "default_gap": float(np.mean(gaps)) if gaps else np.nan}


def outcome(cfg: L.Config, hp: Dict[str, Any], plan: ValidityPlan, eval_seeds: Sequence[int]) -> float:
    """The outcome the index is supposed to predict: the confidence-sensitive gate's advantage per period over the
    better fixed rule, exactly as the study's H1 defines it, measured on evaluation paths."""
    got = {}
    for pol in ("gate_lcb", "always", "default"):
        vals = []
        for sd in eval_seeds:
            p = L.paths(cfg, plan.periods, sd)
            c = L.candidate_orders(cfg, p, hp["gain"])
            S, pr, go = L.policy_orders(pol, cfg, p, c, hp)
            vals.append(L.outcomes(pol, cfg, p, c, S, pr, go, plan.burn_in)["net_payoff"])
        got[pol] = float(np.mean(vals))
    return got["gate_lcb"] - max(got["always"], got["default"])


def _configs(plan: ValidityPlan, n: int, seed: int, family: str) -> List[L.Config]:
    """Latin-hypercube configurations over this plan's ranges."""
    rng = np.random.default_rng(seed)
    cols = {}
    for k, (lo, hi) in plan.ranges:
        u = (rng.permutation(n) + rng.random(n)) / n
        v = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if k == "hazard" else lo + u * (hi - lo)
        cols[k] = np.round(v) if k == "obs_delay" else v
    out = []
    for j in range(n):
        kw = {k: float(v[j]) for k, v in cols.items()}
        kw["obs_delay"] = int(kw["obs_delay"])
        out.append(L.Config(family=family, **kw))
    return out


def measure(plan: ValidityPlan, hp: Dict[str, Any], arm: str,
            progress: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
    """One row per configuration: the covariates, the index and the outcome.

    `arm` is "exploratory" or "confirmatory"; they use disjoint seed blocks and disjoint configuration draws.
    """
    if arm not in ("exploratory", "confirmatory"):
        raise ValueError(f"unknown arm {arm!r}")
    tag = EXPLORATORY_TAG if arm == "exploratory" else CONFIRMATORY_TAG
    cfg_seed = EXPLORATORY_CONFIG_SEED if arm == "exploratory" else CONFIRMATORY_CONFIG_SEED
    note = progress or (lambda f, m: None)
    rows = []
    total = max(1, len(plan.families) * plan.n_configs)
    done = 0
    for fi, fam in enumerate(plan.families):
        for j, cfg in enumerate(_configs(plan, plan.n_configs, cfg_seed + 17 * fi, fam)):
            note(done / total, f"{arm} {fam}: configuration {j + 1}")
            ps = seeds(plan, tag, 100 * fi + j, plan.pilot_paths)
            es = seeds(plan, tag, 50_000 + 100 * fi + j, plan.eval_paths)
            pilot = pilot_summary(cfg, hp["gain"], plan, ps)
            row = covariate_row(cfg, pilot)
            row.update(arm=arm, family=fam, config=j, **{INDEX_KEY: pilot.get(INDEX_KEY, np.nan)},
                       ratio=pilot.get("ratio", np.nan), advantage=outcome(cfg, hp, plan, es),
                       sigma=cfg.sigma, tau=cfg.tau, hazard=cfg.hazard, memory=cfg.memory)
            rows.append(row)
            done += 1
    note(1.0, "done")
    return pd.DataFrame(rows)


# ================================================================================================ the two tests
def incremental_validity(df: pd.DataFrame, plan: ValidityPlan, folds: Optional[Sequence[np.ndarray]] = None,
                         n_boot: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    """Does the index add predictive accuracy beyond the comparison model, on HELD-OUT CONFIGURATIONS?

    Leave-one-configuration-out cross-validated R-squared of (comparison + index) minus (comparison). The
    interval is a bootstrap over configurations, the unit that was sampled. Reported whatever its sign.
    """
    d = df.dropna(subset=[INDEX_KEY, "advantage"]).reset_index(drop=True)
    y = d["advantage"].to_numpy(float)
    n = len(d)
    if n < 8:
        return {"n": n, "note": "too few configurations to cross-validate"}
    f = list(folds) if folds is not None else leave_one_out_folds(n)
    X0, names0 = design_matrix(d, with_index=False)
    X1, _ = design_matrix(d, with_index=True)
    base, full = cv_r2(X0, y, f, plan.ridge), cv_r2(X1, y, f, plan.ridge)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_boot or plan.n_boot)):
        idx = rng.integers(0, n, n)
        bd = d.iloc[idx].reset_index(drop=True)
        by = bd["advantage"].to_numpy(float)
        bf = leave_one_out_folds(len(bd))
        b0, _ = design_matrix(bd, with_index=False)
        b1, _ = design_matrix(bd, with_index=True)
        a, b = cv_r2(b0, by, bf, plan.ridge), cv_r2(b1, by, bf, plan.ridge)
        if np.isfinite(a) and np.isfinite(b):
            draws.append(b - a)
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (np.nan, np.nan)
    return {"n": n, "protocol": "held-out configurations (leave one out)",
            "comparison_r2": base, "with_index_r2": full, "delta_r2": full - base,
            "ci_lo": lo, "ci_hi": hi, "supported": bool(lo > 0),
            "comparison_columns": names0, "index_column": INDEX_KEY,
            "claim": "a positive delta means the index carries information the comparison model does not; it does "
                     "NOT establish a distinct mechanism"}


def family_transfer(df: pd.DataFrame, plan: ValidityPlan, train_family: str = "jump",
                    n_boot: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    """Does the index still add accuracy on HELD-OUT PROCESS FAMILIES?

    Both models are fitted on one family and scored on the others, which were never seen. The harder protocol: a
    summary that works only within the process it was calibrated on fails here.
    """
    d = df.dropna(subset=[INDEX_KEY, "advantage"]).reset_index(drop=True)
    tr = d[d["family"] == train_family]
    te = d[d["family"] != train_family]
    if len(tr) < 8 or len(te) < 8:
        return {"n_train": len(tr), "n_test": len(te), "note": "too few configurations to transfer"}

    def r2(with_index: bool, a: pd.DataFrame, b: pd.DataFrame) -> float:
        Xa, na = design_matrix(a, with_index)
        Xb, nb = design_matrix(b, with_index)
        common = [c for c in na if c in nb]
        if len(common) < 2:
            return np.nan
        Xa = Xa[:, [na.index(c) for c in common]]
        Xb = Xb[:, [nb.index(c) for c in common]]
        pred = _fit_predict(Xa, a["advantage"].to_numpy(float), Xb, plan.ridge)
        yb = b["advantage"].to_numpy(float)
        ss_tot = float(np.sum((yb - yb.mean()) ** 2))
        return 1.0 - float(np.sum((yb - pred) ** 2)) / ss_tot if ss_tot > 0 else np.nan

    base, full = r2(False, tr, te), r2(True, tr, te)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_boot or plan.n_boot)):
        bt = tr.iloc[rng.integers(0, len(tr), len(tr))].reset_index(drop=True)
        bs = te.iloc[rng.integers(0, len(te), len(te))].reset_index(drop=True)
        a, b = r2(False, bt, bs), r2(True, bt, bs)
        if np.isfinite(a) and np.isfinite(b):
            draws.append(b - a)
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (np.nan, np.nan)
    return {"protocol": f"held-out process families (fitted on {train_family})",
            "n_train": len(tr), "n_test": len(te), "test_families": sorted(te["family"].unique()),
            "comparison_r2": base, "with_index_r2": full, "delta_r2": full - base,
            "ci_lo": lo, "ci_hi": hi, "supported": bool(lo > 0)}


# ================================================================================================ hierarchical
def hierarchical_incremental_validity(plan: ValidityPlan, hp: Dict[str, Any], df: pd.DataFrame, arm: str,
                                      inner: Optional[int] = None, seed: int = 0,
                                      progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
    """Incremental validity with the PILOT ESTIMATION UNCERTAINTY propagated into it.

    The index is not observed; it is estimated from finitely many pilot paths, and another draw of those paths
    would give another index and so another answer. A bootstrap over configurations alone treats the index as if
    it had been measured exactly. This resamples at both levels: configurations in the outer loop, and for each
    configuration its pilot paths in the inner loop, re-estimating the index from the resampled paths before any
    model is fitted. Without it the interval on the added R-squared is too narrow by however much the index
    wobbles.
    """
    note = progress or (lambda f, m: None)
    tag = EXPLORATORY_TAG if arm == "exploratory" else CONFIRMATORY_TAG
    inner = int(inner if inner is not None else plan.inner_boot)
    d = df.dropna(subset=[INDEX_KEY, "advantage"]).reset_index(drop=True)
    n = len(d)
    rng = np.random.default_rng(seed)
    pools: List[List[float]] = []
    for i in range(n):
        row = d.iloc[i]
        note(0.5 * i / max(n, 1), f"pilot resamples {i + 1}/{n}")
        fam_i = list(plan.families).index(str(row["family"]))
        base_seeds = seeds(plan, tag, 100 * fam_i + int(row["config"]), plan.pilot_paths)
        vals = [float(row[INDEX_KEY])]
        cfg = L.Config(family=str(row["family"]), sigma=float(row["sigma"]), tau=float(row["tau"]),
                       hazard=float(row["hazard"]), avail=float(row["avail"]), memory=float(row["memory"]),
                       cost=float(row["cost"]), obs_delay=int(row["delay"]))
        for _ in range(inner):
            take = [int(x) for x in rng.choice(base_seeds, size=len(base_seeds), replace=True)]
            r = L.learnability_ratio(cfg, hp["gain"], plan.periods, take, plan.burn_in)
            if np.isfinite(r.get(INDEX_KEY, np.nan)):
                vals.append(float(r[INDEX_KEY]))
        pools.append(vals)
    draws = []
    for b in range(int(plan.n_boot)):
        note(0.5 + 0.5 * b / max(plan.n_boot, 1), f"outer resample {b + 1}")
        idx = rng.integers(0, n, n)
        bd = d.iloc[idx].reset_index(drop=True)
        bd[INDEX_KEY] = [pools[i][int(rng.integers(len(pools[i])))] for i in idx]
        by = bd["advantage"].to_numpy(float)
        bf = leave_one_out_folds(len(bd))
        b0, _ = design_matrix(bd, with_index=False)
        b1, _ = design_matrix(bd, with_index=True)
        a, c = cv_r2(b0, by, bf, plan.ridge), cv_r2(b1, by, bf, plan.ridge)
        if np.isfinite(a) and np.isfinite(c):
            draws.append(c - a)
    note(1.0, "done")
    if not draws:
        return {"n": n, "note": "no usable resamples"}
    return {"n": n, "inner_resamples": inner, "outer_resamples": len(draws),
            "delta_r2_mean": float(np.mean(draws)),
            "ci_lo": float(np.percentile(draws, 2.5)), "ci_hi": float(np.percentile(draws, 97.5)),
            "supported": bool(np.percentile(draws, 2.5) > 0),
            "note": "configurations resampled in the outer loop, pilot paths in the inner loop, so the interval "
                    "carries the uncertainty of estimating the index itself"}


# ================================================================================================ matched designs
# Two designs that a regression cannot substitute for, because each holds something fixed BY CONSTRUCTION rather
# than by adjustment. Pairs are formed on the exploratory or confirmatory rows as given; the matching rule is
# prespecified and does not look at the outcome.
def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / (sd if sd > 1e-12 else 1.0)


def matched_simple_uncertainty(df: pd.DataFrame, plan: ValidityPlan) -> Dict[str, Any]:
    """Design 1: similar SIMPLE uncertainty measures, different proposed index.

    Pairs are chosen to be close in standardized (log sigma, log(1 + tau)) and far apart in the index. If the
    index carries information those two do not, the member with the lower index should do better within a pair.
    """
    d = df.dropna(subset=[INDEX_KEY, "advantage"]).reset_index(drop=True)
    if len(d) < 4:
        return {"pairs": 0, "note": "too few configurations to match"}
    simple = np.column_stack([_z(np.log(d["sigma"].to_numpy(float) + 1e-9)),
                              _z(np.log1p(d["tau"].to_numpy(float)))])
    idx = _z(d[INDEX_KEY].to_numpy(float))
    rows, used = [], set()
    # greedy: the closest pair on simple uncertainty with the largest index gap, then the next, without reuse
    cand = []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            close = float(np.hypot(*(simple[i] - simple[j])))
            gap = float(abs(idx[i] - idx[j]))
            if gap > 0:
                cand.append((close / gap, i, j, close, gap))
    for _, i, j, close, gap in sorted(cand)[:len(d) * len(d)]:
        if i in used or j in used or len(rows) >= plan.matched_pairs:
            continue
        used |= {i, j}
        lo_i, hi_i = (i, j) if idx[i] < idx[j] else (j, i)
        rows.append(dict(low_index=int(lo_i), high_index=int(hi_i), simple_distance=close, index_gap=gap,
                         advantage_low_index=float(d["advantage"][lo_i]),
                         advantage_high_index=float(d["advantage"][hi_i]),
                         difference=float(d["advantage"][lo_i] - d["advantage"][hi_i])))
    if not rows:
        return {"pairs": 0, "note": "no usable pairs"}
    diff = np.array([r["difference"] for r in rows], float)
    m, lo, hi = _boot_mean(diff, plan.n_boot)
    return {"design": "similar simple uncertainty, different index", "pairs": len(rows),
            "mean_difference": m, "ci_lo": lo, "ci_hi": hi, "supported": bool(lo > 0),
            "reading": "positive means the member of a pair with the LOWER index did better, which is the "
                       "direction the index predicts once volatility and observation noise are held close",
            "table": pd.DataFrame(rows)}


def matched_same_index(df: pd.DataFrame, plan: ValidityPlan) -> Dict[str, Any]:
    """Design 2: similar index, different underlying process.

    Pairs are chosen to be close in the index and drawn from different process families. If the index is a
    sufficient summary of learnability, the two members should perform alike and the mean difference should be
    indistinguishable from zero. A large difference says the index is not sufficient, whatever its slope.
    """
    d = df.dropna(subset=[INDEX_KEY, "advantage"]).reset_index(drop=True)
    if d["family"].nunique() < 2:
        return {"pairs": 0, "note": "only one process family present"}
    idx = d[INDEX_KEY].to_numpy(float)
    cand = []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if d["family"][i] != d["family"][j]:
                cand.append((abs(idx[i] - idx[j]), i, j))
    rows, used = [], set()
    for gap, i, j in sorted(cand):
        if i in used or j in used or len(rows) >= plan.matched_pairs:
            continue
        used |= {i, j}
        rows.append(dict(a=int(i), b=int(j), family_a=str(d["family"][i]), family_b=str(d["family"][j]),
                         index_gap=float(gap), advantage_a=float(d["advantage"][i]),
                         advantage_b=float(d["advantage"][j]),
                         abs_difference=float(abs(d["advantage"][i] - d["advantage"][j]))))
    if not rows:
        return {"pairs": 0, "note": "no cross-family pairs"}
    gaps = np.array([r["abs_difference"] for r in rows], float)
    m, lo, hi = _boot_mean(gaps, plan.n_boot)
    spread = float(np.nanstd(d["advantage"].to_numpy(float)))
    return {"design": "similar index, different process family", "pairs": len(rows),
            "mean_absolute_difference": m, "ci_lo": lo, "ci_hi": hi,
            "outcome_sd": spread, "relative_to_outcome_sd": m / spread if spread > 0 else np.nan,
            "reading": "the index is a SUFFICIENT summary only if matched pairs perform alike. A mean absolute "
                       "difference comparable to the spread of the outcome says it is not, however well it "
                       "predicts on average",
            "table": pd.DataFrame(rows)}


def _boot_mean(x: np.ndarray, n_boot: int, seed: int = 0) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (int(n_boot), len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ================================================================================================ robustness
def ratio_with_clip(cfg: L.Config, gain: float, T: int, pilot_seeds: Sequence[int], burn: int,
                    lo: float, hi: float) -> float:
    """The index recomputed with a different clip on the observations-needed term: a SENSITIVITY PROBE.

    heiner_abm.learnability clips n_needed to [1, 1e6]. Near-zero advantages drive that term towards infinity,
    so the upper clip is doing real work and the index partly reports where the clip was put. This recomputes
    the same quantity under other clips to show how much. It does NOT redefine the index: a test pins this
    function to learnability.learnability_ratio at the registered clip.
    """
    regimes, resid = [], []
    for sd_ in pilot_seeds:
        p = L.paths(cfg, T, sd_)
        c = L.candidate_orders(cfg, p, gain)
        differ = c["S_F"] != c["S_D"]
        g_true = L.payoff(c["S_F"], p["d"]) - L.payoff(c["S_D"], p["d"])
        g_obs = L.payoff(c["S_F"], p["y"]) - L.payoff(c["S_D"], p["y"])
        info = differ & ~np.isnan(g_obs)
        if cfg.family == "drift":
            step = max(20, int(round(1 / max(cfg.hazard, 1e-6))))
            bounds = list(range(burn, T, step)) + [T]
        else:
            bounds = sorted(set([burn] + [t for t in np.flatnonzero(p["change"]) if t > burn] + [T]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            dd = differ[a:b]
            if not dd.any():
                continue
            mu_r = float(g_true[a:b][dd].mean())
            go = g_obs[a:b][info[a:b]]
            resid.extend(go - mu_r)
            regimes.append((mu_r, len(go)))
    if not regimes:
        return float("nan")
    s = float(np.std(resid, ddof=1)) if len(resid) > 1 else float("nan")
    need = np.array([np.clip((L.Z90 * s / max(abs(m), 1e-9)) ** 2, lo, hi) for m, _ in regimes])
    avail = np.array([n for _, n in regimes], float)
    return float(np.log10(np.median(need / np.maximum(avail, 1.0))))


ROBUSTNESS_NOTES: Dict[str, str] = {
    "obs_corr": "Correlated feedback: the observation noise is AR(1) instead of independent. The index counts "
                "observations as if each were a fresh draw, so correlation should make the evidence worth less "
                "than the index assumes.",
    "obs_bias": "Biased observations: a systematic error in what the learner sees. The index is built from the "
                "spread of observed gains, not their location, so a bias is invisible to it by construction.",
    "memory": "Learner memory: how long evidence is kept. The index does not depend on it at all, so any effect "
              "on performance is something the index cannot see.",
    "near_zero": "Near-zero advantages: regimes where the two orders are almost equally good. The observations "
                 "needed to determine the sign explode, and the clip decides what the index reports.",
    "clip": "Clipping sensitivity: the same index under other clips on the observations-needed term.",
}


def robustness(plan: ValidityPlan, hp: Dict[str, Any], arm: str = "exploratory",
               base: Optional[L.Config] = None,
               progress: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
    """Vary one feedback or learner property at a time, and report what it does to the index and to performance.

    The point is not a ranking. It is whether the index moves when the thing it claims to summarize moves, and
    whether performance moves when the index does not.
    """
    note = progress or (lambda f, m: None)
    tag = EXPLORATORY_TAG if arm == "exploratory" else CONFIRMATORY_TAG
    base = base or L.Config(family="jump", sigma=20.0, tau=10.0, hazard=0.01, avail=0.6, memory=0.97, cost=1.0)
    rows = []
    items = [(k, v) for k, v in plan.robustness]
    total = sum(len(v) for _, v in items) + 2 * len(plan.clip_variants) + 3
    done = 0
    for factor, levels in items:
        for lv in levels:
            note(done / total, f"{factor} = {lv}")
            cfg = L.replace(base, **{factor: (float(lv) if factor != "obs_delay" else int(lv))})
            ps = seeds(plan, tag, 70_000 + done, plan.pilot_paths)
            es = seeds(plan, tag, 80_000 + done, plan.eval_paths)
            pil = pilot_summary(cfg, hp["gain"], plan, ps)
            rows.append(dict(probe=factor, level=float(lv), note=ROBUSTNESS_NOTES[factor],
                             index=pil.get(INDEX_KEY, np.nan), adv_noise=pil.get("adv_noise", np.nan),
                             mean_advantage=pil.get("mean_advantage", np.nan),
                             advantage=outcome(cfg, hp, plan, es)))
            done += 1
    # near-zero advantages: shrink the gap between the two candidate orders by making the default good
    for lv, default in enumerate(("slow", "fixed_initial", "dominated")):
        note(done / total, f"near_zero: default {default}")
        cfg = L.replace(base, default=default)
        ps = seeds(plan, tag, 70_000 + done, plan.pilot_paths)
        es = seeds(plan, tag, 80_000 + done, plan.eval_paths)
        pil = pilot_summary(cfg, hp["gain"], plan, ps)
        rows.append(dict(probe="near_zero", level=float(lv), note=ROBUSTNESS_NOTES["near_zero"],
                         default=default, index=pil.get(INDEX_KEY, np.nan),
                         adv_noise=pil.get("adv_noise", np.nan), mean_advantage=pil.get("mean_advantage", np.nan),
                         advantage=outcome(cfg, hp, plan, es)))
        done += 1
    # Clipping sensitivity. Run on the base configuration AND on one whose advantages are near zero, because
    # that is where the upper clip binds: on a configuration with a healthy advantage the clip never engages and
    # a sensitivity analysis there would report a reassuring zero for the wrong reason.
    for label, cfg in (("base", base), ("near_zero", L.replace(base, default="slow"))):
        ps = seeds(plan, tag, 70_000 + done, plan.pilot_paths)
        for lo, hi in plan.clip_variants:
            note(done / total, f"clip [{lo}, {hi}] on {label}")
            rows.append(dict(probe="clip", level=float(hi), note=ROBUSTNESS_NOTES["clip"], clip_on=label,
                             clip_lo=lo, clip_hi=hi,
                             index=ratio_with_clip(cfg, hp["gain"], plan.periods, ps, plan.burn_in, lo, hi)))
            done += 1
    note(1.0, "done")
    return pd.DataFrame(rows)


# ================================================================================================ the study
@dataclass
class ValidityResult:
    plan_hash: str
    arm: str
    measurements: pd.DataFrame
    incremental: Dict[str, Any]
    transfer: Dict[str, Any]
    hierarchical: Dict[str, Any]
    matched_simple: Dict[str, Any]
    matched_index: Dict[str, Any]
    robustness: pd.DataFrame
    verdicts: pd.DataFrame
    settings: Dict[str, Any] = field(default_factory=dict)


# What each test can and cannot conclude. Written before any confirmatory number exists, so that a result cannot
# be narrated into a stronger claim than its design supports.
CLAIMS: Tuple[Tuple[str, str, str], ...] = (
    ("V1", "The index adds predictive accuracy beyond the prespecified comparison model, on held-out "
           "configurations.",
     "Supported if the lower limit of the bootstrap interval on the added cross-validated R-squared exceeds 0. "
     "Supports: the index is a USEFUL SUMMARY INDEX carrying information the listed quantities do not. Does NOT "
     "support: a distinct mechanism."),
    ("V2", "The index still adds accuracy on process families it was not fitted on.",
     "As V1, fitting on the jump family and scoring on switch and drift. Failure here with V1 supported means "
     "the index summarizes within a process but does not transfer."),
    ("V3", "The added accuracy survives the uncertainty of estimating the index from pilot paths.",
     "As V1, with pilot paths resampled inside each configuration resample. A V1 supported and V3 not means the "
     "apparent gain is within the noise of measuring the index."),
    ("D1", "Holding simple uncertainty measures close, configurations with a lower index do better.",
     "Matched pairs close in standardized log sigma and log(1 + tau) and far apart in the index. Supported if "
     "the interval on the mean within-pair difference lies above 0."),
    ("D2", "Holding the index close, the underlying process does not matter.",
     "Matched cross-family pairs close in the index. The index is a SUFFICIENT summary only if the mean "
     "absolute within-pair difference is small relative to the spread of the outcome. This claim is reported, "
     "never asserted: a large difference is evidence against sufficiency, not against usefulness."),
)


def verdict_table(inc: Dict[str, Any], tr: Dict[str, Any], hier: Dict[str, Any],
                  d1: Dict[str, Any], d2: Dict[str, Any], arm: str) -> pd.DataFrame:
    """One row per prespecified claim, with the result stated in the language the design supports."""
    def fmt(d, key="delta_r2", lo="ci_lo", hi="ci_hi"):
        if key not in d or not np.isfinite(d.get(key, np.nan)):
            return d.get("note", "not computed")
        return f"{d[key]:+.4f} [{d.get(lo, float('nan')):+.4f}, {d.get(hi, float('nan')):+.4f}]"

    rows = [
        dict(id="V1", result=fmt(inc), supported=bool(inc.get("supported", False)),
             reading="useful summary index" if inc.get("supported") else "no added accuracy demonstrated"),
        dict(id="V2", result=fmt(tr), supported=bool(tr.get("supported", False)),
             reading="transfers to unseen process families" if tr.get("supported")
                     else "no transfer demonstrated"),
        dict(id="V3", result=fmt(hier, "delta_r2_mean"), supported=bool(hier.get("supported", False)),
             reading="survives pilot estimation uncertainty" if hier.get("supported")
                     else "within the noise of estimating the index"),
        dict(id="D1", result=fmt(d1, "mean_difference"), supported=bool(d1.get("supported", False)),
             reading="lower index did better at matched simple uncertainty" if d1.get("supported")
                     else "no matched-pair difference demonstrated"),
        dict(id="D2", result=(f"{d2.get('mean_absolute_difference', float('nan')):.3f} against an outcome s.d. of "
                              f"{d2.get('outcome_sd', float('nan')):.3f}"),
             supported=None,
             reading=("sufficiency not supported: matched pairs differ by about as much as the outcome varies"
                      if (d2.get("relative_to_outcome_sd") or 0) > 0.5
                      else "matched pairs perform alike, consistent with sufficiency")),
    ]
    text = {c[0]: c[1] for c in CLAIMS}
    test = {c[0]: c[2] for c in CLAIMS}
    out = pd.DataFrame(rows)
    out.insert(1, "claim", [text[i] for i in out["id"]])
    out["decision_rule"] = [test[i] for i in out["id"]]
    out["arm"] = arm
    out["status"] = "confirmatory" if arm == "confirmatory" else "EXPLORATORY (not a confirmatory result)"
    return out


def run_validity(plan: ValidityPlan, hp: Dict[str, Any], arm: str,
                 progress: Optional[Callable[[float, str], None]] = None) -> ValidityResult:
    """Run one arm end to end. The confirmatory arm is intended to be run ONCE, after the plan hash is fixed."""
    note = progress or (lambda f, m: None)
    note(0.0, f"{arm}: measuring configurations")
    df = measure(plan, hp, arm, progress=lambda f, m: note(0.45 * f, m))
    note(0.45, "held-out configurations")
    inc = incremental_validity(df, plan)
    note(0.55, "held-out process families")
    tr = family_transfer(df, plan)
    note(0.6, "pilot uncertainty")
    hier = hierarchical_incremental_validity(plan, hp, df, arm, progress=lambda f, m: note(0.6 + 0.2 * f, m))
    note(0.8, "matched designs")
    d1, d2 = matched_simple_uncertainty(df, plan), matched_same_index(df, plan)
    note(0.85, "robustness")
    rb = robustness(plan, hp, arm, progress=lambda f, m: note(0.85 + 0.14 * f, m))
    note(1.0, "done")
    return ValidityResult(
        plan_hash=plan.digest, arm=arm, measurements=df, incremental=inc, transfer=tr, hierarchical=hier,
        matched_simple=d1, matched_index=d2, robustness=rb,
        verdicts=verdict_table(inc, tr, hier, d1, d2, arm),
        settings={"plan": json.loads(plan.to_json()), "plan_hash": plan.digest, "index": INDEX_NAME,
                  "index_note": INDEX_NOTE, "covariates": dict(COVARIATES),
                  "interactions": [list(i) for i in INTERACTIONS],
                  "arm": arm, "seed_block": EXPLORATORY_TAG if arm == "exploratory" else CONFIRMATORY_TAG,
                  "claims": {c[0]: {"claim": c[1], "decision_rule": c[2]} for c in CLAIMS}})
