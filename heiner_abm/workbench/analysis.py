"""Analysis service: summaries, the primary comparison and robustness checks from a run's trial data.

Every policy in a cell sees the same environments, so comparisons are paired by replication. Intervals are percentile
bootstraps over replications (each replication is one independent environment: a demand path, a market or a landscape).
Results lead with the size and uncertainty of the stated comparison, not with rankings.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .environments import ENVIRONMENTS


def _boot(x: np.ndarray, n_boot: int, seed: int = 0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(x) == 1:
        return float(x[0]), np.nan, np.nan, 1
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(x)


def summary(trials: pd.DataFrame, outcome: str, n_boot: int = 1000) -> pd.DataFrame:
    """Mean and 95% interval of an outcome per treatment level and policy."""
    rows = []
    if outcome not in trials:
        return pd.DataFrame()
    for (lvl, pol), g in trials.groupby(["level", "policy"], sort=False):
        m, lo, hi, n = _boot(g[outcome].to_numpy(), n_boot)
        rows.append(dict(level=lvl, policy=pol, mean=m, lo=lo, hi=hi, n=n))
    return pd.DataFrame(rows)


def paired(trials: pd.DataFrame, a: str, b: str, outcome: str, n_boot: int = 1000,
           rows: Optional[pd.Series] = None) -> pd.DataFrame:
    """Paired difference a − b per replication, by treatment level."""
    t = trials if rows is None else trials[rows]
    out = []
    for lvl, g in t.groupby("level", sort=False):
        w = g.pivot_table(index="replication", columns="policy", values=outcome)
        if a not in w or b not in w:
            continue
        d = (w[a] - w[b]).to_numpy()
        m, lo, hi, n = _boot(d, n_boot)
        out.append(dict(level=lvl, effect=m, lo=lo, hi=hi, n=n, mean_a=float(w[a].mean()), mean_b=float(w[b].mean()),
                        share_positive=float(np.mean(d[np.isfinite(d)] > 0)) if n else np.nan))
    return pd.DataFrame(out)


def interpret(effect: float, lo: float, hi: float, smallest: float, higher_better: Optional[bool]) -> str:
    """A plain statement of what the interval supports, relative to the smallest effect of practical interest."""
    if not np.isfinite(lo):
        return "Too few replications for an interval."
    sign = 1 if higher_better in (True, None) else -1
    lo_, hi_ = sorted((sign * lo, sign * hi))
    if lo_ > smallest:
        return (f"The interval lies entirely above the smallest effect of interest ({smallest:g}): the treatment policy "
                "does better here, by an amount the interval bounds.")
    if hi_ < -smallest:
        return (f"The interval lies entirely below −{smallest:g}: the treatment policy does worse here.")
    if lo_ > 0 or hi_ < 0:
        return ("The interval excludes zero but overlaps the range of effects smaller than the stated practical "
                "threshold: a difference is detected, its practical importance is not established.")
    return "The interval includes zero: these runs do not establish a difference in either direction."


def overview(run) -> Dict[str, Any]:
    spec = run.spec
    env = ENVIRONMENTS[spec.environment]
    a, b = spec.question.comparison
    oc = spec.question.primary_outcome
    o = env.outcome(oc)
    eff = paired(run.trials, a, b, oc, spec.design.n_boot)
    evidence = ("Held-out simulation environments: test seeds disjoint from the training seeds used for tuning."
                if spec.design.tuning == "grid" else
                "Simulation environments with documented fixed parameters (no tuning, so no held-out split is "
                "needed).")
    if run.kind == "preview":
        evidence = "PREVIEW (reduced replications and periods): for orientation only. " + evidence
    return dict(question=spec.question.text, comparison=f"{env.policy(a).label} versus {env.policy(b).label}",
                outcome=o.label, higher_better=o.higher_better, evidence=evidence, effects=eff,
                smallest=spec.question.smallest_effect)


def robustness(run, n_boot: int = 500) -> Dict[str, pd.DataFrame]:
    """The primary comparison in two disjoint halves of the replications, and its spread across replications."""
    spec = run.spec
    a, b = spec.question.comparison
    oc = spec.question.primary_outcome
    t = run.trials
    reps = sorted(t["replication"].unique())
    half = len(reps) // 2
    out = {}
    if half >= 2:
        first = t["replication"].isin(reps[:half])
        h1 = paired(t, a, b, oc, n_boot, first).assign(block="first half of replications")
        h2 = paired(t, a, b, oc, n_boot, ~first).assign(block="second half of replications")
        out["halves"] = pd.concat([h1, h2], ignore_index=True)
    rows = []
    for lvl, g in t.groupby("level", sort=False):
        w = g.pivot_table(index="replication", columns="policy", values=oc)
        if a in w and b in w:
            rows += [dict(level=lvl, replication=r, difference=float(v)) for r, v in (w[a] - w[b]).items()]
    out["per_replication"] = pd.DataFrame(rows)
    return out


def secondary(run, keys: Sequence[str], n_boot: int = 500) -> pd.DataFrame:
    frames = []
    for k in keys:
        s = summary(run.trials, k, n_boot)
        if len(s):
            frames.append(s.assign(outcome=k))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
