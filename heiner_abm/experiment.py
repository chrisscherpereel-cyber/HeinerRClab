"""Human experiment, protocol 2.0: mechanisms of adaptation under uncertainty.

Version 1.0 (three blocks of mixed low, medium and high uncertainty; X1/X2) confounded the sources of uncertainty and
tested only the correlation between restraint and profit. No data were collected under it. Version 2.0 replaces it.

Design
    Within participants: four blocks, each differing from a common baseline in exactly one uncertainty mechanism
        B  baseline             cost volatility 8, observation noise 2, no demand shifts
        V  volatility           cost volatility 24
        N  observation noise    noise on the cost estimate 10
        R  regime change        unannounced demand-regime shifts, hazard 0.06 per period
      so each treatment effect is a contrast with B, not a mixture of mechanisms. Block order follows a balanced
      4 x 4 Williams square (each condition once in each position; each ordered pair of adjacent conditions once);
      the order and each block's position are recorded.
    Exogenous paths: four path seeds; a participant sees each path once, in a different condition, rotated by a
      path offset (path = (condition index + offset) mod 4), so every path appears equally often in every condition
      across participants and treatment effects are not confounded with particular paths. Within a path, conditions
      share the same underlying random draws (cost shocks, perception noise draws, regime draws), scaled by the
      condition's parameters.
    Between participants (randomized, balanced in permuted blocks):
        aid          decision aid shown (a reliability-gated recommendation: the output the 'Reliability condition,
                     model-based' design would choose in the participant's place, and whether its gate says change or
                     keep) or not shown (UNAIDED CONTROL: identical screens and information options, no recommendation;
                     the recommendation is still computed and recorded, never displayed)
        elicitation  belief elicitation on or off (measurement-effect treatment): if on, after every third decision a
                     separate, separately timed screen asks for a price forecast and the confidence that the chosen
                     output beats keeping last period's output
    Practice block: 8 periods of the baseline condition on a separate path; never analyzed.
    Comprehension checks: four multiple-choice questions (a fifth on the aid in the aid arm); all must be answered
      correctly, with at most three attempts; attempts are recorded and participants who fail are excluded.
    Rivals: three tuned agent designs from the registered tournament (they react to the participant's output, so a
      participant's market is interactive; participants never interact with each other).
    Recorded per decision: condition, block position, path, response time of the decision and of the belief screen
      (separately), information access (optional history panels opened), the aid's recommendation and whether it was
      shown, the chosen output, and the feedback shown afterwards (price, profit), plus every agent design's shadow
      choice.

Assignment: an allocation schedule (allocation_schedule) assigns participant slots to (Williams row, path offset,
aid, elicitation) in permuted blocks of 16: every 16 consecutive slots contain each (row, aid x elicitation)
combination once, with path offsets forming a Latin square over rows and arms, so every condition meets every path
equally often within each block. Experimenters assign slots in order.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .arena import Env
from .registered import TUNED_PARAMS
from .stepper import InteractiveMarket

BASELINE = dict(delta=8.0, noise=2.0, hazard=0.0)
CONDITIONS: Dict[str, Dict[str, float]] = {
    "B": dict(BASELINE),
    "V": {**BASELINE, "delta": 24.0},
    "N": {**BASELINE, "noise": 10.0},
    "R": {**BASELINE, "hazard": 0.06},
}
CONDITION_LABELS = {"B": "Baseline", "V": "Higher cost volatility", "N": "Noisier cost estimate",
                    "R": "Unannounced demand shifts"}
MECHANISM = {"V": "outcome volatility", "N": "observation noise", "R": "regime change"}
RIVALS = ("cobweb_p", "heur_markup", "options_p")
SHADOWS = ("ruleb", "opt_br", "opt_nash", "cobweb_p", "cobweb_q", "heur_markup", "heur_wsls", "options_p",
           "options_m", "heiner_p", "heiner_m", "satis_p", "satis_m", "imit_best", "imit_avg")
AID_DESIGN = "heiner_m"          # the decision aid's recommendation (reliability-gated, model-based)
SIGNAL_DESIGN = "opt_br"         # researcher's reference target for the mechanism models (never shown)
WILLIAMS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))     # balanced Latin square, order 4
ARMS = tuple(product((False, True), (False, True)))                      # (aid, elicitation)

COMPREHENSION = (
    ("profit", "How is your profit in a period computed?",
     ("(Price − unit cost) × your output", "Price × your output", "Your output − the price"), 0),
    ("price", "If the total output of all firms rises, the price…", ("falls", "rises", "does not change"), 0),
    ("estimate", "The unit-cost estimate you see before deciding is…",
     ("an estimate that may be off", "always exactly right", "last period's cost"), 0),
    ("keep", "Keeping last period's output…", ("is always allowed", "costs a penalty", "is not allowed"), 0),
)
AID_QUESTION = ("aid", "The recommendation shown on screen…",
                ("is advice you may follow or ignore", "must be followed", "is the output that maximizes profit"), 0)

EXP_HYPOTHESES = (
    ("E1", "Volatility: participants adjust output more often in V than in B.",
     "Adjustment rate per participant and block; participant fixed effects and path fixed effects; contrast V − B; "
     "95% CI clustered by participant (and session if recorded). Two-sided; supported if the CI excludes 0 (sign "
     "reported)."),
    ("E2", "Observation noise: adjustment frequency differs between N and B.",
     "As E1, contrast N − B."),
    ("E3", "Regime change: adjustment frequency differs between R and B.",
     "As E1, contrast R − B."),
    ("E4", "Decision aid (causal, randomized): the aid changes relative profit (own minus rivals' mean) per period.",
     "Participant means over the four blocks regressed on aid and elicitation indicators; HC1 (or session-clustered) "
     "95% CI; supported if the CI for aid excludes 0."),
    ("E5", "Measurement effect (causal, randomized): belief elicitation changes the adjustment rate.",
     "As E4, outcome the participant's mean adjustment rate, coefficient on elicitation."),
    ("M1", "Mechanism: the treatments act through adjustment probability and adjustment magnitude separately.",
     "Hurdle model of adjustment (probability: logit with lapse, inertia, condition shifts; magnitude: partial "
     "adjustment toward the reference target with rounding and lapse), partially pooled across participants; "
     "condition shifts on each part with participant-bootstrap CIs. Model variants compared on held-out blocks "
     "(log score, Brier score, RMSE). Secondary."),
    ("A1", "Associational only: across participants, adjustment frequency and relative profit.",
     "OLS slope with HC1 CI. Labeled associational: adjustment frequency is not randomized, so the slope is not a "
     "causal effect of restraint."),
)
EXCLUSIONS = ("failed the comprehension checks within three attempts",
              "fewer than 75% of the decisions of any main block completed",
              "median decision time below 1 second in any main block (inattentive clicking)")


@dataclass(frozen=True)
class ExperimentPlan:
    version: str = "2.0"
    periods_per_block: int = 30
    practice_periods: int = 8
    conditions: Tuple[str, ...] = ("B", "V", "N", "R")
    n_paths: int = 4
    path_seeds: Tuple[int, ...] = (7101, 7202, 7303, 7404)
    practice_seed: int = 7000
    rivals: Tuple[str, ...] = RIVALS
    aid_design: str = AID_DESIGN
    signal_design: str = SIGNAL_DESIGN
    elicit_every: int = 3
    max_attempts: int = 3
    allocation_seed: int = 20261008
    alpha: float = 0.05
    n_boot: int = 2000
    hypotheses: Tuple[Tuple[str, str, str], ...] = EXP_HYPOTHESES
    exclusions: Tuple[str, ...] = EXCLUSIONS

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "condition_settings": CONDITIONS, "shadows": SHADOWS,
                           "williams": WILLIAMS, "comprehension": COMPREHENSION + (AID_QUESTION,)},
                          sort_keys=True, indent=1)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


PLAN = ExperimentPlan()


# ================================================================================================ assignment
@dataclass(frozen=True)
class Assignment:
    slot: int
    williams_row: int
    path_offset: int
    aid: bool
    elicit: bool

    def order(self, plan: ExperimentPlan = PLAN) -> List[str]:
        return [plan.conditions[i] for i in WILLIAMS[self.williams_row]]

    def path(self, condition: str, plan: ExperimentPlan = PLAN) -> int:
        return (plan.conditions.index(condition) + self.path_offset) % plan.n_paths

    def to_dict(self, plan: ExperimentPlan = PLAN) -> Dict:
        return dict(slot=self.slot, williams_row=self.williams_row, path_offset=self.path_offset, aid=self.aid,
                    elicit=self.elicit, order="-".join(self.order(plan)))


def allocation_schedule(n: int, plan: ExperimentPlan = PLAN) -> List[Assignment]:
    """Slots 0..n-1 in permuted blocks of 16 (see the module docstring); reproducible from the plan's seed.

    Each block of 16 contains every (Williams row, arm) cell once, and the path offset is a Latin square over them,
    offset = (row + arm + base) mod 4, so within every 16 slots each offset occurs 4 times, once per row and once per
    arm; conditions are therefore never tied to particular paths. The base rotates across blocks."""
    rng = np.random.default_rng(plan.allocation_seed)
    out: List[Assignment] = []
    while len(out) < n:
        for base in rng.permutation(plan.n_paths):
            cells = [(r, a) for r in range(len(WILLIAMS)) for a in range(len(ARMS))]
            for j in rng.permutation(len(cells)):
                r, a = cells[j]
                aid, el = ARMS[a]
                out.append(Assignment(len(out), r, int((r + a + base) % plan.n_paths), bool(aid), bool(el)))
    return out[:n]


def assignment_for(slot: int, plan: ExperimentPlan = PLAN) -> Assignment:
    return allocation_schedule(slot + 1, plan)[slot]


def slot_from_id(participant: str, plan: ExperimentPlan = PLAN) -> int:
    """Fallback when no slot is assigned: a slot from the ID (balanced only in expectation)."""
    return zlib.crc32(participant.encode()) % 64


# ================================================================================================ markets
def block_market(condition: str, path: int, plan: ExperimentPlan = PLAN, params: Optional[Dict] = None,
                 shadows: Sequence[str] = SHADOWS, practice: bool = False) -> InteractiveMarket:
    seed = plan.practice_seed if practice else plan.path_seeds[path]
    env = Env(c_max=80.0, q_range=1500.0, foresight=0.0, belief_lag=10, seed=seed, **CONDITIONS[condition])
    periods = plan.practice_periods if practice else plan.periods_per_block
    return InteractiveMarket(env, plan.rivals, params or TUNED_PARAMS, shadows, periods + 1,
                             label="practice" if practice else condition)


def elicit_now(period: int, plan: ExperimentPlan = PLAN) -> bool:
    return period % plan.elicit_every == 0


def decision_row(m: InteractiveMarket, row: Dict, a: Assignment, block_index: int, condition: str, path: int,
                 extra: Dict, plan: ExperimentPlan = PLAN) -> Dict:
    """One recorded decision: the market's row plus design, timing, information access, aid and belief fields."""
    rec = row.get(f"shadow:{plan.aid_design}", np.nan)
    return {**row, "condition": condition, "block_index": block_index, "path": path,
            "path_seed": plan.path_seeds[path] if path >= 0 else plan.practice_seed,
            "practice": condition == "practice", "aid": a.aid, "elicit": a.elicit, "williams_row": a.williams_row,
            "path_offset": a.path_offset, "slot": a.slot, "order": "-".join(a.order(plan)),
            "aid_recommendation": float(np.maximum(5.0, np.rint(rec))) if np.isfinite(rec) else np.nan,
            "aid_shown": bool(a.aid), "aid_says_change": bool(np.isfinite(rec) and np.rint(rec) != row["q_prev"]),
            "feedback_price": row["price"], "feedback_profit": row["profit"], "plan": plan.digest, **extra}


# ================================================================================================ synthetic pilot
def synthetic_pilot(n: int, seed: int = 0, plan: ExperimentPlan = PLAN, model: str = "full",
                    population: Optional[Dict] = None, sessions: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """SYNTHETIC PILOT participants generated by a mechanism model (heiner_abm.human_models) in the real interactive
    market, following the allocation schedule. For design checks, recovery studies and power planning only: never
    evidence about people. Every row carries synthetic=True."""
    from . import human_models as HM
    rng = np.random.default_rng(seed)
    pop = {**HM.DEFAULT_POPULATION, **(population or {})}
    frames, truth = [], []
    for a in allocation_schedule(n, plan):
        pid = f"synthetic-{a.slot:03d}"
        theta = HM.draw_participant(model, pop, rng, aid=a.aid, elicit=a.elicit)
        truth.append(dict(participant=pid, model=model, **{f"true:{k}": v for k, v in theta.items()}))
        kept_last = 0.0
        for bi, cond in enumerate(a.order(plan)):
            path = a.path(cond, plan)
            m = block_market(cond, path, plan, shadows=(plan.signal_design, plan.aid_design))
            while not m.done:
                info = m.info()
                q_prev = info["own_q"]
                s = m.pending["shadows"][plan.signal_design] - q_prev
                rec = m.pending["shadows"][plan.aid_design]
                q, adj = HM.simulate_choice(model, theta, cond, s, q_prev, kept_last, rng,
                                            aid_rec=rec if a.aid else None)
                row = m.step(q)
                kept_last = 0.0 if row["deviated"] else 1.0
                el = a.elicit and elicit_now(row["period"], plan)
                extra = dict(participant=pid, session=f"S{a.slot // 8 % max(sessions, 1)}" if sessions else "",
                             rt_decision=float(rng.lognormal(1.3, 0.4)), info_history=bool(rng.random() < 0.3),
                             info_cost=bool(rng.random() < 0.2),
                             belief_price=float(row["price"] + rng.normal(0, 6)) if el else np.nan,
                             belief_confidence=float(rng.uniform(30, 90)) if el else np.nan,
                             rt_belief=float(rng.lognormal(1.6, 0.3)) if el else np.nan, elicited=el,
                             comprehension_passed=True, comprehension_attempts=1, synthetic=True)
                frames.append(decision_row(m, row, a, bi, cond, path, extra, plan))
    return pd.DataFrame(frames), pd.DataFrame(truth)


# ================================================================================================ descriptive
def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive: per participant, RMSE of every design's shadow prediction against the participant's choices."""
    cols = [c for c in df.columns if c.startswith("shadow:")]
    rows = []
    for pid, d in df[~df.get("practice", False)].groupby("participant"):
        err = {c[7:]: float(np.sqrt(np.mean((np.maximum(5.0, np.rint(d[c])) - d["q"]) ** 2))) for c in cols}
        best = min(err, key=err.get)
        rows.append(dict(participant=pid, best_design=best, best_rmse=err[best], **{f"rmse:{k}": v for k, v in err.items()}))
    return pd.DataFrame(rows)
