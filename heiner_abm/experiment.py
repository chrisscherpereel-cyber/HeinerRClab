"""Human experiment: participants play one firm in the same market, in blocks of low, medium and high uncertainty.

Protocol (frozen and hashed like the simulation plans):
    * three blocks, one per uncertainty condition, in an order counterbalanced across participants (Latin square);
    * in every block the participant faces the same three agent rivals (tuned designs from the registered tournament)
      and the same market seed as every other participant, so conditions are comparable across people;
    * each period the participant sees last period's price, market quantity, own output and profit, and an estimate of
      the coming raw-material cost, and chooses an output (keeping last period's output is the default);
    * every deterministic design runs in shadow mode on the participant's firm, recording what it would have chosen.

Analysis:
    * classification: each participant's best-predicting design (lowest RMSE of shadow predictions vs choices);
    * X1 predictability: deviation rate (share of periods in which output changed) high vs low uncertainty, paired
      within participants;
    * X2 does restraint pay: under high uncertainty, regression across participants of profit relative to rivals on
      the deviation rate.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .analysis import ols
from .arena import Env
from .registered import TUNED_PARAMS
from .stepper import InteractiveMarket

CONDITIONS = {"low": dict(delta=3.0, noise=0.0, hazard=0.0),
              "medium": dict(delta=12.0, noise=3.0, hazard=0.01),
              "high": dict(delta=25.0, noise=8.0, hazard=0.04)}
RIVALS = ("cobweb_p", "heur_markup", "options_p")
SHADOWS = ("ruleb", "opt_br", "opt_nash", "cobweb_p", "cobweb_q", "heur_markup", "heur_wsls", "options_p",
           "options_m", "heiner_p", "heiner_m", "satis_p", "satis_m", "imit_best", "imit_avg")

EXP_HYPOTHESES = (
    ("X1", "Predictability: participants change their output less often under high than under low uncertainty.",
     "Paired difference in deviation rate (high − low) across participants; 95% bootstrap CI. Supported if the CI "
     "lies below 0."),
    ("X2", "Restraint pays under uncertainty: under high uncertainty, participants who change output less often earn "
           "more relative to their rivals.",
     "OLS across participants of (own profit − rivals' mean profit) per period in the high-uncertainty block on the "
     "deviation rate in that block. Supported if the slope is negative with p < α."),
)


@dataclass(frozen=True)
class ExperimentPlan:
    version: str = "1.0"
    periods_per_block: int = 25
    conditions: Tuple[str, ...] = ("low", "medium", "high")
    rivals: Tuple[str, ...] = RIVALS
    seeds: Tuple[int, ...] = (6101, 6202, 6303)
    alpha: float = 0.05
    n_boot: int = 2000
    hypotheses: Tuple[Tuple[str, str, str], ...] = EXP_HYPOTHESES

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "condition_settings": CONDITIONS, "shadows": SHADOWS}, sort_keys=True,
                          indent=1)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


PLAN = ExperimentPlan()


def block_order(participant: str, plan: ExperimentPlan = PLAN) -> List[str]:
    """Counterbalanced order of conditions (Latin square over all permutations), from the participant ID."""
    orders = list(permutations(plan.conditions))
    return list(orders[zlib.crc32(participant.encode()) % len(orders)])


def block_market(condition: str, plan: ExperimentPlan = PLAN, params: Optional[Dict] = None) -> InteractiveMarket:
    seed = plan.seeds[plan.conditions.index(condition)]
    env = Env(c_max=80.0, q_range=1500.0, foresight=0.0, belief_lag=10, seed=seed, **CONDITIONS[condition])
    return InteractiveMarket(env, plan.rivals, params or TUNED_PARAMS, SHADOWS, plan.periods_per_block + 1,
                             label=condition)


# ================================================================================================ analysis
def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Per participant: RMSE of every design's shadow prediction against the participant's choices; best design."""
    cols = [c for c in df.columns if c.startswith("shadow:")]
    rows = []
    for pid, d in df.groupby("participant"):
        err = {c[7:]: float(np.sqrt(np.mean((np.maximum(5.0, np.rint(d[c])) - d["q"]) ** 2))) for c in cols}
        best = min(err, key=err.get)
        rows.append(dict(participant=pid, best_design=best, best_rmse=err[best], **{f"rmse:{k}": v for k, v in err.items()}))
    return pd.DataFrame(rows)


def by_condition(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["participant", "block"]).agg(deviation_rate=("deviated", "mean"), profit=("profit", "mean"),
                                                 rivals=("rivals_mean_profit", "mean")).reset_index()
    g["relative_profit"] = g["profit"] - g["rivals"]
    return g


def evaluate_experiment(df: pd.DataFrame, plan: ExperimentPlan = PLAN) -> pd.DataFrame:
    c = by_condition(df)
    w = c.pivot(index="participant", columns="block", values="deviation_rate").dropna()
    rows = []
    if {"high", "low"} <= set(w.columns) and len(w) >= 2:
        diff = (w["high"] - w["low"]).to_numpy()
        rng = np.random.default_rng(0)
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(plan.n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(("supported" if hi < 0 else "not supported",
                     f"high − low deviation rate {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], n = {len(diff)}"))
    else:
        rows.append(("not tested", "needs participants with both a low- and a high-uncertainty block"))
    h = c[c["block"] == "high"]
    if len(h) >= 3 and h["deviation_rate"].std() > 0:
        tab, _ = ols(h["relative_profit"].to_numpy(float), [h["deviation_rate"].to_numpy(float)], ["deviation_rate"])
        b, p = float(tab.loc[1, "coef"]), float(tab.loc[1, "p"])
        rows.append(("supported" if b < 0 and p < plan.alpha else "not supported",
                     f"slope {b:+.1f} profit per unit of deviation rate (p = {p:.3g}), n = {len(h)}"))
    else:
        rows.append(("not tested", "needs at least three participants with a high-uncertainty block"))
    return pd.DataFrame([dict(id=hh[0], hypothesis=hh[1], decision_rule=hh[2], verdict=v[0], result=v[1])
                         for hh, v in zip(plan.hypotheses, rows)])


def synthetic_participants(n: int = 12, noise: float = 4.0, seed: int = 0, plan: ExperimentPlan = PLAN
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulated 'participants' who follow a known design with noise, for demonstrating and testing the analysis.
    Never a substitute for human data."""
    rng = np.random.default_rng(seed)
    kinds = ["ruleb", "cobweb_p", "options_p", "heiner_p", "heur_markup", "opt_br"]
    frames, truth = [], []
    for j in range(n):
        pid, kind = f"synthetic-{j:02d}", kinds[j % len(kinds)]
        for cond in block_order(pid, plan):
            m = block_market(cond, plan)
            while not m.done:
                m.info()
                prop = m.pending["shadows"][kind]
                q = prop if kind == "ruleb" else prop + noise * rng.standard_normal()
                m.step(q)
            frames.append(m.frame().assign(participant=pid))
        truth.append(dict(participant=pid, true_design=kind))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(truth)
