"""Mechanism study: when does restricting flexibility pay, and what does it cost to know when?

Two experiments on top of the agent tournament (heiner_abm.arena):

1. Principle versus implementation. On each target (model-based best reply, price-based expectation), a focal firm
   uses one selection rule while the other eight slots hold the tournament's tuned rivals. The focal variants are
       always            move toward the target every period (no restriction)
       band              inaction band (real options)
       rc_learned        reliability condition learned from experience, judged with the agent's own demand model
       rc_true_model     the same learner, judging its past decisions with the true demand curve
       rc_oracle         the reliability condition with the true expected gain of each kind of deviation
       rc_memory_m       the learner with memory m (how much experience it pools)
   The oracle's expected gains come from a long, independent run of the same environment (different random draws), so
   it knows the reliability of its rule but not the future. Gaps decompose the cost of applying the principle:
       oracle - learned = (oracle - true model) [estimation from limited, noisy experience]
                        + (true model - learned) [judging with a misspecified model].

2. Maps by type of uncertainty. Environments are drawn by Latin hypercube over separate sources of uncertainty (risk,
   perception error, model misspecification, competence, stakes). The gain from restriction is related to the
   measured error-to-signal ratio K of the flexible rule (the boundary test), to each source of uncertainty
   (standardized regression), and a quadratic ridge metamodel maps which theory's agent does best where.

The plan is frozen and hashed together with the code, like the tournament plan.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import arena
from .analysis import ols
from .arena import KEYS, N_BINS, SELECT_SPACE, TARGET_SPACE, Env, Tuned, cluster_ci, holm, simulate

MAP_RANGES = {"delta": (1.0, 35.0), "noise": (0.0, 15.0), "hazard": (0.0, 0.06), "belief_lag": (0.0, 60.0),
              "foresight": (0.0, 0.8), "c_max": (60.0, 95.0), "q_range": (800.0, 3000.0)}
AXES = {"delta": "Risk: cost volatility Δ", "noise": "Perception error σ", "hazard": "Misspecification: shift hazard λ",
        "belief_lag": "Misspecification: model-updating lag L", "foresight": "Competence: cost foresight κ",
        "c_max": "Stakes: maximum cost (lower = more profitable)", "q_range": "Demand quantity range"}
TARGETS = {"model": dict(always="opt_br", band="options_m", rc="heiner_m", true="heiner_m_true", oracle="heiner_m_oracle"),
           "price": dict(always="cobweb_p", band="options_p", rc="heiner_p", true="heiner_p_true", oracle="heiner_p_oracle")}
VARIANT_LABELS = {"always": "Always adjust", "band": "Inaction band", "rc_learned": "RC, learned",
                  "rc_true_model": "RC, learned with the true model", "rc_oracle": "RC, oracle"}

STUDY_HYPOTHESES = (
    ("M1", "Principle: with known reliability, reliability-based restriction beats always adjusting toward the "
           "model-based target.",
     "Paired profit difference rc_oracle − always on the model-based target, 95% cluster-bootstrap CI over "
     "environments. Supported if the CI lies above 0."),
    ("M2", "Cost of applying the principle: the oracle beats the learned reliability condition on both targets.",
     "Paired difference rc_oracle − rc_learned per target, Holm-adjusted over the two targets. Supported if both are "
     "significantly positive."),
    ("M3", "Model bias: judging past decisions with the true model improves the learned reliability condition on "
           "the model-based target.",
     "Paired difference rc_true_model − rc_learned on the model-based target. Supported if the CI lies above 0."),
    ("M4", "Boundary: the gain from reliability-based restriction rises with the flexible rule's error-to-signal "
           "ratio K.",
     "OLS of (rc_oracle − always) on ln K of the always-adjusting rule, with a target indicator, pooled over both "
     "targets. Supported if the ln K slope is positive with p < α."),
    ("M5", "Type of uncertainty: misspecification raises the gain from restriction more than risk does.",
     "Standardized regression of (rc_oracle − always) on the model-based target on all uncertainty axes. Supported "
     "if the bootstrap CI of (hazard coefficient − volatility coefficient) lies above 0."),
)


@dataclass(frozen=True)
class StudyPlan:
    version: str = "1.0"
    tournament_plan: str = ""          # hash of the tournament plan whose tuned rivals form the background
    ranges: Tuple[Tuple[str, Tuple[float, float]], ...] = tuple(MAP_RANGES.items())
    n_envs: int = 120
    periods: int = 800
    burn_in: int = 50
    seed: int = 3003
    oracle_periods: int = 3200
    memories: Tuple[float, ...] = (0.9, 0.97, 0.99, 0.999)
    alpha: float = 0.05
    n_boot: int = 1000
    hypotheses: Tuple[Tuple[str, str, str], ...] = STUDY_HYPOTHESES

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "code_sha256": study_code_digest()}, sort_keys=True, indent=1)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def study_code_digest() -> str:
    src = "".join(inspect.getsource(o) for o in (map_envs, run_study, effects, decomposition, boundary,
                                                  uncertainty_types, evaluate_study, fit_metamodel, _ridge, _design))
    return hashlib.sha256((src + arena.code_digest()).encode()).hexdigest()


def map_envs(plan: StudyPlan) -> List[Env]:
    """Latin-hypercube sample over the uncertainty axes; every environment has its own random streams."""
    rng = np.random.default_rng(plan.seed)
    r = dict(plan.ranges)
    n = plan.n_envs
    cols = {}
    for k, (lo, hi) in r.items():
        u = (rng.permutation(n) + rng.random(n)) / n
        cols[k] = lo + u * (hi - lo)
    return [Env(delta=float(cols["delta"][j]), c_max=float(cols["c_max"][j]), q_range=float(cols["q_range"][j]),
                noise=float(cols["noise"][j]), foresight=float(cols["foresight"][j]), hazard=float(cols["hazard"][j]),
                belief_lag=int(round(cols["belief_lag"][j])), seed=300_000 + j) for j in range(n)]


@dataclass
class StudyResult:
    plan_hash: str
    exploratory: bool
    focal: pd.DataFrame           # one row per (target, variant, environment)
    theories: pd.DataFrame        # one row per (environment, theory) in the full tournament lineup
    oracle_values: pd.DataFrame   # oracle expected gain per (target, environment, bin)
    effects: pd.DataFrame = field(default_factory=pd.DataFrame)
    decomposition: pd.DataFrame = field(default_factory=pd.DataFrame)
    boundary: Dict = field(default_factory=dict)
    types: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdicts: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_study(plan: StudyPlan, tuned: Tuned, progress: Optional[Callable[[float, str], None]] = None) -> StudyResult:
    envs = map_envs(plan)
    E, slot = len(envs), KEYS.index("heiner")
    base = np.array(tuned.lineup(), dtype=object)
    params0 = {k: dict(v) for k, v in tuned.params.items()}
    rows, orows = [], []
    steps = 2 * (6 + len(plan.memories)) + 1
    done = 0

    def tick(msg):
        nonlocal done
        if progress:
            progress(done / steps, msg)
        done += 1

    def run(design, prm, env_list, periods):
        lineup = np.tile(base, (len(env_list), 1))
        lineup[:, slot] = design
        params = {k: dict(v) for k, v in params0.items()}
        params[design] = prm
        return simulate(env_list, lineup, params, periods, plan.burn_in)

    for target, d in TARGETS.items():
        # every focal variant aims at the SAME target (the always-adjusting rule's tuned target settings) and differs
        # only in its selection rule, whose settings come from that rule's own tuned design
        tgt = {k: v for k, v in tuned.params[d["always"]].items() if k in TARGET_SPACE[target]}
        sel = lambda design, rule: {k: v for k, v in tuned.params[design].items() if k in SELECT_SPACE[rule]}
        rc = {**tgt, **sel(d["rc"], "rc")}
        band = {**tgt, **sel(d["band"], "band")}
        tick(f"{target} target: estimating the oracle's reliability values")
        long_envs = [replace(e, seed=e.seed + 900_000) for e in envs]
        out = run(d["true"], rc, long_envs, plan.oracle_periods)
        gs, gc = out["gain_sum"][:, slot], out["gain_count"][:, slot]
        e_or = np.where(gc > 0, gs / np.maximum(gc, 1), 0.0)
        for j in range(E):
            for b in range(N_BINS):
                orows.append(dict(target=target, env=j, bin=b, expected_gain=e_or[j, b], n=gc[j, b]))
        variants = [("always", d["always"], tgt), ("band", d["band"], band),
                    ("rc_learned", d["rc"], rc), ("rc_true_model", d["true"], rc),
                    ("rc_oracle", d["oracle"], {**rc, **{f"e{b}": e_or[:, b] for b in range(N_BINS)}})]
        variants += [(f"rc_memory_{m:g}", d["rc"], {**rc, "memory": m}) for m in plan.memories]
        for name, design, prm in variants:
            tick(f"{target} target: {VARIANT_LABELS.get(name, name)}")
            out = run(design, prm, envs, plan.periods)
            for j, e in enumerate(envs):
                rows.append(dict(target=target, variant=name, env=j, profit=out["profit"][j, slot],
                                 rel_profit=out["profit"][j, slot] - out["profit"][j].mean(), K=out["K"][j, slot],
                                 change_rate=out["change_rate"][j, slot], **{k: getattr(e, k) for k in MAP_RANGES}))
    tick("Full tournament lineup across the map")
    out = simulate(envs, np.tile(base, (E, 1)), params0, plan.periods, plan.burn_in)
    trows = [dict(env=j, theory=t, design=tuned.design[t], rel_profit=out["profit"][j, i] - out["profit"][j].mean(),
                  **{k: getattr(e, k) for k in MAP_RANGES}) for j, e in enumerate(envs) for i, t in enumerate(KEYS)]
    if progress:
        progress(1.0, "Done")
    focal, theories = pd.DataFrame(rows), pd.DataFrame(trows)
    res = StudyResult(plan_hash=plan.digest, exploratory=plan.digest != default_plan(plan.tournament_plan).digest,
                      focal=focal, theories=theories, oracle_values=pd.DataFrame(orows))
    res.effects = effects(focal, plan)
    res.decomposition = decomposition(focal, plan)
    res.boundary = boundary(focal, plan)
    res.types = uncertainty_types(focal, plan)
    res.verdicts = evaluate_study(res, plan)
    return res


# ================================================================================================ analysis
def _wide(focal: pd.DataFrame, target: str) -> pd.DataFrame:
    return focal[focal["target"] == target].pivot(index="env", columns="variant", values="profit")


def effects(focal: pd.DataFrame, plan: StudyPlan) -> pd.DataFrame:
    """Every variant against 'always' on the same target (paired by environment)."""
    rows = []
    for target in TARGETS:
        w = _wide(focal, target)
        for v in w.columns:
            if v == "always":
                continue
            m, lo, hi, p = cluster_ci((w[v] - w["always"]).to_numpy(), w.index.to_numpy(), plan.n_boot, 1 - plan.alpha)
            rows.append(dict(target=target, variant=v, label=VARIANT_LABELS.get(v, v.replace("rc_memory_", "RC, memory ")),
                             diff=m, lo=lo, hi=hi, p=p))
    return pd.DataFrame(rows)


def decomposition(focal: pd.DataFrame, plan: StudyPlan) -> pd.DataFrame:
    """Cost of applying the reliability condition: total = oracle - learned, split into estimation (oracle - true
    model) and model bias (true model - learned)."""
    rows = []
    for target in TARGETS:
        w = _wide(focal, target)
        for part, a, b in (("Total cost of learning", "rc_oracle", "rc_learned"),
                           ("Estimation from experience", "rc_oracle", "rc_true_model"),
                           ("Judging with a misspecified model", "rc_true_model", "rc_learned"),
                           ("Value of the principle (oracle − always)", "rc_oracle", "always")):
            m, lo, hi, p = cluster_ci((w[a] - w[b]).to_numpy(), w.index.to_numpy(), plan.n_boot, 1 - plan.alpha)
            rows.append(dict(target=target, component=part, diff=m, lo=lo, hi=hi, p=p))
    return pd.DataFrame(rows)


def boundary(focal: pd.DataFrame, plan: StudyPlan) -> Dict:
    """Gain from restriction against the always-adjusting rule's measured error-to-signal ratio K."""
    out, frames = {}, []
    for target in TARGETS:
        w = _wide(focal, target)
        k = focal[(focal["target"] == target) & (focal["variant"] == "always")].set_index("env")["K"]
        f = pd.DataFrame({"lnK": np.log(k.clip(lower=1e-6)), "target": target,
                          **{f"gain_{v}": w[v] - w["always"] for v in ("rc_oracle", "rc_learned", "band")}})
        frames.append(f.reset_index())
        for v in ("rc_oracle", "rc_learned", "band"):
            tab, r2 = ols(f[f"gain_{v}"].to_numpy(float), [f["lnK"].to_numpy(float)], ["lnK"])
            a, b = float(tab.loc[0, "coef"]), float(tab.loc[1, "coef"])
            out[(target, v)] = dict(intercept=a, slope=b, p=float(tab.loc[1, "p"]), r2=r2,
                                    K_star=float(np.exp(-a / b)) if b > 0 else np.nan)
    df = pd.concat(frames, ignore_index=True)
    tab, _ = ols(df["gain_rc_oracle"].to_numpy(float), [df["lnK"].to_numpy(float),
                                                        (df["target"] == "price").to_numpy(float)], ["lnK", "price"])
    out["pooled"] = dict(slope=float(tab.loc[1, "coef"]), p=float(tab.loc[1, "p"]))
    out["data"] = df
    return out


def _src(y: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    Z = (X - X.mean()) / (X.std() + 1e-12)
    tab, _ = ols((y - y.mean()) / (y.std() + 1e-12), [Z[c].to_numpy(float) for c in Z], list(Z))
    return tab["coef"].to_numpy()[1:]


def uncertainty_types(focal: pd.DataFrame, plan: StudyPlan) -> pd.DataFrame:
    """Standardized regression of each restriction gain on the uncertainty axes, with bootstrap CIs."""
    rows = []
    axes = list(MAP_RANGES)
    rng = np.random.default_rng(0)
    for target in TARGETS:
        w = _wide(focal, target)
        X = focal[(focal["target"] == target) & (focal["variant"] == "always")].set_index("env")[axes].loc[w.index]
        for v in ("rc_oracle", "rc_learned", "band"):
            y = (w[v] - w["always"]).to_numpy(float)
            coef = _src(y, X)
            boots = np.array([_src(y[i], X.iloc[i].reset_index(drop=True))
                              for i in (rng.integers(0, len(y), len(y)) for _ in range(min(plan.n_boot, 400)))])
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            dh = boots[:, axes.index("hazard")] - boots[:, axes.index("delta")]
            for j, a in enumerate(axes):
                rows.append(dict(target=target, variant=v, axis=a, label=AXES[a], coef=coef[j], lo=lo[j], hi=hi[j],
                                 hazard_minus_delta_lo=np.percentile(dh, 2.5),
                                 hazard_minus_delta_hi=np.percentile(dh, 97.5)))
    return pd.DataFrame(rows)


def evaluate_study(res: StudyResult, plan: StudyPlan) -> pd.DataFrame:
    dec = res.decomposition
    get = lambda t, c: dec[(dec["target"] == t) & (dec["component"] == c)].iloc[0]
    v1 = get("model", "Value of the principle (oracle − always)")
    m1 = ("supported" if v1["lo"] > 0 else "not supported", f"{v1['diff']:+.0f} [{v1['lo']:.0f}, {v1['hi']:.0f}]")
    tot = [get(t, "Total cost of learning") for t in TARGETS]
    rej = holm([r["p"] for r in tot], plan.alpha)
    m2 = ("supported" if all(r and x["diff"] > 0 for r, x in zip(rej, tot)) else "not supported",
          "; ".join(f"{t} {x['diff']:+.0f} [{x['lo']:.0f}, {x['hi']:.0f}]{' *' if r else ''}"
                    for t, x, r in zip(TARGETS, tot, rej)))
    v3 = get("model", "Judging with a misspecified model")
    m3 = ("supported" if v3["lo"] > 0 else "not supported", f"{v3['diff']:+.0f} [{v3['lo']:.0f}, {v3['hi']:.0f}]")
    pb = res.boundary["pooled"]
    m4 = ("supported" if pb["slope"] > 0 and pb["p"] < plan.alpha else "not supported",
          f"slope {pb['slope']:+.3g} per unit of ln K (p = {pb['p']:.3g})")
    ty = res.types
    r5 = ty[(ty["target"] == "model") & (ty["variant"] == "rc_oracle")].iloc[0]
    m5 = ("supported" if r5["hazard_minus_delta_lo"] > 0 else "not supported",
          f"hazard − volatility coefficient, 95% CI [{r5['hazard_minus_delta_lo']:+.2f}, "
          f"{r5['hazard_minus_delta_hi']:+.2f}]")
    return pd.DataFrame([dict(id=h[0], hypothesis=h[1], decision_rule=h[2], verdict=v[0], result=v[1])
                         for h, v in zip(plan.hypotheses, (m1, m2, m3, m4, m5))])


# ================================================================================================ metamodel maps
def _design(X: np.ndarray) -> np.ndarray:
    """Quadratic response surface: linear, squared and pairwise interaction terms."""
    k = X.shape[1]
    cols = [X, X ** 2] + [X[:, [i]] * X[:, [j]] for i in range(k) for j in range(i + 1, k)]
    return np.column_stack([np.ones(len(X))] + cols)


def _ridge(Xd, y, lam):
    pen = np.full(Xd.shape[1], lam)
    pen[0] = 0.0
    return np.linalg.solve(Xd.T @ Xd + np.diag(pen), Xd.T @ y)


def fit_metamodel(df: pd.DataFrame, y: str, folds: int = 5, seed: int = 0) -> Dict:
    """Quadratic ridge metamodel of y on the standardized uncertainty axes. The penalty is chosen by k-fold
    cross-validation, and the reported fit is the cross-validated R² (out-of-fold), not the in-sample R²."""
    axes = list(MAP_RANGES)
    mu, sd = df[axes].mean().to_numpy(), df[axes].std().to_numpy() + 1e-12
    Xd = _design((df[axes].to_numpy(float) - mu) / sd)
    yy = df[y].to_numpy(float)
    fold = np.random.default_rng(seed).permutation(len(yy)) % folds
    best = (-np.inf, None)
    for lam in (0.1, 1.0, 10.0, 100.0, 1000.0):
        pred = np.empty_like(yy)
        for f in range(folds):
            tr = fold != f
            pred[~tr] = Xd[~tr] @ _ridge(Xd[tr], yy[tr], lam)
        cv = 1 - ((yy - pred) ** 2).mean() / (yy.var() + 1e-12)
        if cv > best[0]:
            best = (cv, lam)
    return dict(beta=_ridge(Xd, yy, best[1]), mu=mu, sd=sd, axes=axes, r2=float(best[0]), ridge=best[1])


def predict_grid(model: Dict, x: str, y: str, base: Dict[str, float], n: int = 25):
    axes = model["axes"]
    gx = np.linspace(*MAP_RANGES[x], n)
    gy = np.linspace(*MAP_RANGES[y], n)
    X = np.array([[{**base, x: a, y: b}[k] for k in axes] for b in gy for a in gx], float)
    pred = _design((X - model["mu"]) / model["sd"]) @ model["beta"]
    return gx, gy, pred.reshape(n, n)


def theory_maps(theories: pd.DataFrame, x: str, y: str, n: int = 25):
    """Predicted profit relative to the market mean for every theory over an x-y grid (other axes at their medians),
    the predicted best theory in each cell, and each metamodel's cross-validated R²."""
    base = theories.drop_duplicates("env")[list(MAP_RANGES)].median().to_dict()
    preds, r2 = {}, {}
    for t in KEYS:
        m = fit_metamodel(theories[theories["theory"] == t], "rel_profit")
        gx, gy, preds[t] = predict_grid(m, x, y, base, n)
        r2[t] = m["r2"]
    stack = np.stack([preds[t] for t in KEYS])
    return gx, gy, preds, stack.argmax(0), r2


def gain_map(focal: pd.DataFrame, target: str, variant: str, x: str, y: str, n: int = 25):
    w = _wide(focal, target)
    env = focal[(focal["target"] == target) & (focal["variant"] == "always")].set_index("env")[list(MAP_RANGES)]
    df = env.loc[w.index].assign(gain=(w[variant] - w["always"]).to_numpy())
    m = fit_metamodel(df, "gain")
    gx, gy, z = predict_grid(m, x, y, env.median().to_dict(), n)
    return gx, gy, z, m["r2"]


def default_plan(tournament_plan: str) -> StudyPlan:
    return StudyPlan(tournament_plan=tournament_plan)


QUICK_STUDY = dict(n_envs=16, periods=250, burn_in=30, oracle_periods=600, memories=(0.9, 0.99), n_boot=200)
