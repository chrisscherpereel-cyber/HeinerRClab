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

Equivalent training opportunities
    "The same budget" can mean two different things, and which one is in force is a choice, not a detail:
        equal_candidates   every family draws the same NUMBER of candidate configurations. Equal effort, but a
                           family with three parameters searches a three-dimensional space with the same number of
                           draws a one-parameter family uses on a line, so its search is sparser.
        per_parameter      the number of candidates is `base` per free parameter, so the density of the search is
                           equal and the effort is not. A family with no parameters still gets one evaluation.
    Neither is more correct; they answer different questions. design_budget() implements both, every run records
    which was used and what it cost (BudgetRecord), and the per-parameter density is reported under either rule so
    a reader can see the trade the other way.

Reading a tuning result
    boundary_report   whether a chosen value sits at the edge of its allowed range. A parameter pinned at an edge
                      usually means the range was too narrow, and the reported value is then a statement about the
                      grid rather than about the agent.
    sensitivity_report  how much the validation score moves across repetitions of the search, and how much was
                      gained between the smallest and the largest budget. A family whose score swings widely
                      across repetitions has not been tuned so much as sampled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROLES = ("competitor", "benchmark", "oracle", "reference")
RANKED = ("competitor",)

BUDGET_RULES = ("equal_candidates", "per_parameter")
BUDGET_RULE_NOTES = {
    "equal_candidates": "Every family draws the same number of candidates: equal effort, unequal search density.",
    "per_parameter": "Every family draws `base` candidates per free parameter: equal density, unequal effort.",
}
BOUNDARY_TOLERANCE = 0.02           # a value within 2% of a range end (on the sampling scale) counts as pinned


def design_budget(space: Dict[str, "Param"], rule: str, base: int) -> int:
    """Candidate configurations a family receives under the stated rule."""
    if rule not in BUDGET_RULES:
        raise ValueError(f"unknown budget rule {rule!r} (one of {', '.join(BUDGET_RULES)})")
    if not space:
        return 1                                            # nothing to search; one evaluation
    return int(base) if rule == "equal_candidates" else int(base) * len(space)


@dataclass(frozen=True)
class BudgetRecord:
    """What a family's tuning actually cost, recorded rather than inferred."""
    rule: str
    base: int
    n_parameters: int
    candidates: int
    candidates_per_parameter: float
    train_runs: int
    validation_runs: int
    test_runs: int

    @property
    def total_runs(self) -> int:
        return self.train_runs + self.validation_runs + self.test_runs

    def to_dict(self) -> Dict[str, object]:
        return {**{f.name: getattr(self, f.name) for f in fields(self)}, "total_runs": self.total_runs,
                "rule_note": BUDGET_RULE_NOTES[self.rule]}


def boundary_report(space: Dict[str, object], chosen: Dict[str, object],
                    tol: float = BOUNDARY_TOLERANCE) -> pd.DataFrame:
    """Where each chosen value sits in its allowed range, and whether it is pinned at an end.

    Position is measured on the scale the parameter is sampled on, so a log-scaled range is judged in logs. A
    pinned value is a statement about the range, not about the agent: it means the search wanted to go further.
    """
    rows = []
    for name, p in sorted(space.items()):
        if name not in chosen:
            continue
        if isinstance(p, Choice):
            # a declared list: pinned means the chosen value is the first or the last candidate offered
            vals = list(p.values)
            idx = vals.index(chosen[name]) if chosen[name] in vals else -1
            pos = (idx / (len(vals) - 1)) if len(vals) > 1 and idx >= 0 else 0.0
            rows.append(dict(parameter=name, chosen=chosen[name], low=vals[0], high=vals[-1], log=False,
                             position=float(pos), at_low=bool(idx == 0), at_high=bool(idx == len(vals) - 1),
                             pinned=bool(idx in (0, len(vals) - 1))))
            continue
        v = float(chosen[name])
        lo, hi = float(p.lo), float(p.hi)
        if p.log and lo > 0 and hi > 0 and v > 0:
            pos = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if hi > lo else 0.0
        else:
            pos = (v - lo) / (hi - lo) if hi > lo else 0.0
        rows.append(dict(parameter=name, chosen=v, low=lo, high=hi, log=bool(p.log),
                         position=float(min(max(pos, 0.0), 1.0)),
                         at_low=bool(pos <= tol), at_high=bool(pos >= 1.0 - tol),
                         pinned=bool(pos <= tol or pos >= 1.0 - tol)))
    return pd.DataFrame(rows)


def sensitivity_report(curve: pd.DataFrame) -> pd.DataFrame:
    """How stable the search was: the spread of validation scores across repetitions at each budget, and what the
    largest budget bought over the smallest."""
    if curve.empty or "validation" not in curve:
        return pd.DataFrame(columns=["family", "budget", "reps", "validation_mean", "validation_sd",
                                     "validation_range", "gain_over_smallest_budget"])
    rows = []
    for family, g in curve.groupby("family", sort=False):
        budgets = sorted(g["budget"].unique())
        smallest = float(g[g["budget"] == budgets[0]]["validation"].mean())
        for b in budgets:
            v = g[g["budget"] == b]["validation"].to_numpy(float)
            rows.append(dict(family=family, budget=int(b), reps=int(len(v)),
                             validation_mean=float(v.mean()),
                             validation_sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                             validation_range=float(v.max() - v.min()) if len(v) else 0.0,
                             gain_over_smallest_budget=float(v.mean() - smallest)))
    return pd.DataFrame(rows)


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
class Choice:
    """A hyperparameter searched over a declared list of candidate values rather than a range.

    The decision benchmarks sample from ranges; the workbench searches hand-picked values chosen to span the
    behaviour of a rule. Both are legitimate, and which is in force is a property of the space rather than of the
    protocol, so budgets, boundary reports and sensitivity are computed the same way for either.
    """
    values: Tuple
    help: str = ""

    def sample(self, rng: np.random.Generator):
        return self.values[int(rng.integers(len(self.values)))]

    @property
    def lo(self):
        return self.values[0]

    @property
    def hi(self):
        return self.values[-1]

    log: bool = False


def is_enumerated(space: Dict[str, object]) -> bool:
    """True when every entry is a declared list, so the search can enumerate instead of sample."""
    return bool(space) and all(isinstance(p, Choice) for p in space.values())


def candidate_configs(space: Dict[str, object], n: int, seed: int) -> List[Dict[str, object]]:
    """The first n candidates of a space: enumerated in declared order for lists, sampled for ranges.

    Enumerating in declared order means a budget of b sees exactly the first b candidates, so a larger budget is a
    superset of a smaller one - the same nesting the sampled search gets from reusing one random sequence.
    """
    if not space:
        return [{}]
    if is_enumerated(space):
        keys = sorted(space)
        out: List[Dict[str, object]] = [{}]
        for k in keys:
            out = [{**c, k: v} for c in out for v in space[k].values]
        return out[:max(int(n), 1)]
    return sample_configs(space, n, seed)


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
    budget: Optional[BudgetRecord] = None     # what the search was allowed and what it cost
    boundary: Optional[pd.DataFrame] = None   # where the selected values sit in their ranges
    sensitivity: Optional[pd.DataFrame] = None


def tune(family: Family, run: Callable[[object, Dict[str, float]], Dict[str, float]], train: Sequence,
         val: Sequence, test: Sequence, budgets: Sequence[int], reps: int, seed: int,
         progress: Optional[Callable[[str], None]] = None,
         budget_rule: str = "equal_candidates") -> Tuned:
    """Random-search tuning of one family (see the module docstring). `run(env, config)` returns a dict with at least
    'payoff' (higher is better).

    `budget_rule` decides what the largest budget means for this family: the same number of candidates as every
    other family, or the same number per free parameter. Either way the budget and what it cost are recorded on
    the result, together with the boundary and sensitivity reports.
    """
    if budget_rule not in BUDGET_RULES:
        raise ValueError(f"unknown budget rule {budget_rule!r} (one of {', '.join(BUDGET_RULES)})")
    base = max(int(b) for b in budgets)
    if budget_rule == "per_parameter" and family.space:
        budgets = sorted({design_budget(family.space, budget_rule, int(b)) for b in budgets})
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
    curve = pd.DataFrame(rows)
    n_params = len(family.space)
    candidates = design_budget(family.space, budget_rule, base)
    record = BudgetRecord(rule=budget_rule, base=base, n_parameters=n_params, candidates=candidates,
                          candidates_per_parameter=float(candidates) / n_params if n_params else float("nan"),
                          train_runs=int(candidates * len(train) * reps_eff) if family.space else len(train),
                          validation_runs=n_runs - (int(candidates * len(train) * reps_eff) if family.space
                                                    else len(train)) - len(test),
                          test_runs=len(test))
    return Tuned(family.key, curve, final, pd.DataFrame(test_rows), n_runs, budget=record,
                 boundary=boundary_report(family.space, final), sensitivity=sensitivity_report(curve))


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
