"""Tuning protocol shared by the decision benchmarks (heiner_abm.bench_inventory, heiner_abm.bench_bandit).

Environments come in three disjoint sets, drawn with different seeds:
    training     candidate hyperparameters are scored here (mean payoff over the training environments);
    validation   the configuration chosen at each tuning budget is scored here; the final configuration is the one
                 whose validation score is highest among the per-budget choices of every repetition;
    test         only the final configuration is run here; test results are the ones reported in rankings.

Tuning is random search. A budget b means the first b sampled configurations (of the same random sequence), so the
choice at budget 2b sees everything the choice at budget b saw. The search is repeated `reps` times with different
sampling seeds; the curve of validation payoff against budget is reported per family and repetition. Families without
hyperparameters are evaluated once and have a flat curve.

Roles
    competitor   an agent with the information of the design; ranked against the other competitors
    benchmark    a correctly specified model that knows the generating parameters (researcher information); reported
                 beside the ranking, never in it
    oracle       knows the true state; reported beside the ranking, never in it
    reference    a different design (for example full instead of chosen-action feedback); reported separately
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROLES = ("competitor", "benchmark", "oracle", "reference")
RANKED = ("competitor",)


@dataclass(frozen=True)
class Param:
    """A hyperparameter: continuous on [lo, hi] (log scale if log=True), rounded to an integer if integer=True."""
    lo: float
    hi: float
    log: bool = False
    integer: bool = False

    def sample(self, rng: np.random.Generator) -> float:
        u = rng.random()
        x = math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo))) if self.log else \
            self.lo + u * (self.hi - self.lo)
        return float(int(round(x))) if self.integer else float(x)


@dataclass(frozen=True)
class Family:
    key: str
    name: str
    role: str
    space: Dict[str, Param]
    assumptions: str
    compute: str
    sources: Tuple[str, ...] = ()
    feedback: str = ""           # what the agent observes, for the reference documentation


def sample_configs(space: Dict[str, Param], n: int, seed: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    if not space:
        return [{}]
    return [{k: p.sample(rng) for k, p in sorted(space.items())} for _ in range(n)]


@dataclass
class Tuned:
    family: str
    curve: pd.DataFrame          # rep, budget, config index, train score, validation score
    final: Dict[str, float]      # the selected configuration
    test: pd.DataFrame           # one row per test environment
    evaluations: int             # environment runs spent (training + validation + test)


def tune(family: Family, run: Callable[[object, Dict[str, float]], Dict[str, float]], train: Sequence,
         val: Sequence, test: Sequence, budgets: Sequence[int], reps: int, seed: int,
         progress: Optional[Callable[[str], None]] = None) -> Tuned:
    """Random-search tuning of one family (see the module docstring). `run(env, config)` returns a dict with at least
    'payoff' (higher is better)."""
    budgets = sorted(set(int(b) for b in budgets))
    score = lambda envs, cfg: float(np.mean([run(e, cfg)["payoff"] for e in envs]))
    n_runs, rows, val_cache = 0, [], {}
    reps_eff = reps if family.space else 1
    choices = []
    for rep in range(reps_eff):
        cfgs = sample_configs(family.space, max(budgets) if family.space else 1, seed + 7919 * rep)
        tr = []
        for cfg in cfgs:
            tr.append(score(train, cfg))
            n_runs += len(train)
        for b in budgets:
            k = int(np.argmax(tr[:b])) if family.space else 0
            key = tuple(sorted(cfgs[k].items()))
            if key not in val_cache:
                val_cache[key] = score(val, cfgs[k])
                n_runs += len(val)
            rows.append(dict(family=family.key, rep=rep, budget=b, train_runs=b * len(train) if family.space else
                             len(train), config=k, train=tr[k], validation=val_cache[key],
                             **{f"param_{p}": v for p, v in cfgs[k].items()}))
            choices.append((val_cache[key], key))
        if progress:
            progress(f"{family.name}: repetition {rep + 1}")
    best = max(choices, key=lambda x: x[0])[1]
    final = dict(best)
    test_rows = []
    for e in test:
        r = run(e, final)
        test_rows.append(dict(family=family.key, env=getattr(e, "seed", None), **r))
        n_runs += 1
    return Tuned(family.key, pd.DataFrame(rows), final, pd.DataFrame(test_rows), n_runs)


def ranking(results: Dict[str, Tuned], families: Dict[str, Family], baseline: Optional[str] = None) -> pd.DataFrame:
    """Test payoff per family with 95% intervals over test environments; competitors ranked, other roles listed
    separately (rank NaN). With `baseline`, also the paired difference to it."""
    rows = []
    base = results[baseline].test.set_index("env")["payoff"] if baseline in results else None
    for k, res in results.items():
        f = families[k]
        x = res.test["payoff"].to_numpy(float)
        r = dict(family=k, agent=f.name, role=f.role, mean_payoff=float(x.mean()),
                 ci=float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan, n_test=len(x),
                 config=", ".join(f"{p} = {v:.4g}" for p, v in res.final.items()) or "—", runs=res.evaluations)
        if base is not None:
            d = (res.test.set_index("env")["payoff"] - base).dropna().to_numpy(float)
            r["vs_baseline"] = float(d.mean())
            r["vs_baseline_ci"] = float(1.96 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    comp = df["role"].isin(RANKED)
    df["rank"] = np.nan
    df.loc[comp, "rank"] = df.loc[comp, "mean_payoff"].rank(ascending=False, method="min")
    df["_order"] = df["role"].map({r: i for i, r in enumerate(ROLES)})
    return df.sort_values(["_order", "mean_payoff"], ascending=[True, False]).drop(columns="_order")


def budget_curve(results: Dict[str, Tuned]) -> pd.DataFrame:
    """Mean validation payoff of the configuration chosen at each budget (configurations tried; train_runs =
    training-environment runs spent), over repetitions."""
    df = pd.concat([r.curve for r in results.values()], ignore_index=True)
    g = df.groupby(["family", "budget"]).agg(train_runs=("train_runs", "first"), validation=("validation", "mean"),
                                             sd=("validation", "std"), train=("train", "mean"),
                                             reps=("rep", "nunique")).reset_index()
    return g
