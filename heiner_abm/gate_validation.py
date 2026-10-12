"""Validation of the reliability gates: what each gate is estimating, and how well it estimates it.

PROPOSED EXTENSION, and a researcher-only one. Nothing here is visible to any agent (see "Leakage" below).

heiner_abm.gate_study compares the gates on outcomes. This module asks the prior question: the gates report a
predicted advantage and, for `lcb`, a lower bound on it - predictions of *what*, and are they any good? That needs
the target named precisely, because three different quantities are easy to conflate.

THREE QUANTITIES, NOT ONE
    1. Observed-feedback advantage  g_t
       What the gate actually learns from. For `gain` and `lcb` it is the estimated counterfactual the engine
       releases: the firm's profit over the judgement window minus the profit it computes for the action it did not
       take, using its BELIEVED demand curve and holding rivals at their observed path. For `explore` it is the
       firm's own realized payoff in a randomized trial, minus an action-independent baseline. It is an estimate
       made from inside the firm, with the firm's own misspecification in it.

    2. Realized advantage  A_t^realized
       The researcher's ex-post fork on the single shock path that actually occurred (engine.run_batch with
       horizon H; heiner_abm.gate_study uses this). It uses the TRUE demand curve and true costs, so it removes the
       firm's misspecification, but it is one draw: it still contains the continuation noise of that one path.

    3. Conditional expected true net advantage  A*_t = E[ net advantage | state_t ]     <- the estimand here
       The quantity a decision rule should want to know: the expectation of the discounted net advantage over the
       judgement window, conditional on the state at the moment of the decision, under a declared continuation
       policy, with the adaptation cost charged. Estimated by averaging `draws` INDEPENDENT continuations from the
       sampled state, so continuation noise is averaged out rather than carried.

    A gate predicting (1) while being judged against (2) looks noisier than it is; judged against (3) the
    comparison is between its prediction and the thing the prediction is for. Both comparisons are reported, and
    the gap between them is the firm's misspecification plus continuation noise, which `decompose()` separates.

THE CONTINUATION
    policy    "rivals_hold": from the sampled state the evaluated firm holds the action under test and every rival
              holds its period-t output, for H periods. This is deliberately the same assumption the firm's own
              estimated-counterfactual feedback makes (gates.py: it "holds rivals' output at its observed path"),
              so A* is the expectation of the thing the gate is trying to estimate, not of a different game. It is
              NOT the engine's "default" continuation, in which rivals re-decide each period; that difference is a
              property of the estimand and is stated, not hidden.
    regime    "frozen": the demand regime at t persists through the horizon. With regime shifts the advantage over
              the window mixes two regimes, which is a different question (gate_study measures behaviour after
              shifts directly).
    horizon   H periods, default the judgement window W (AdaptiveParams.window), discounted by `discount`.
    draws     independent continuations per state. Both arms of a state share their cost draws (common random
              numbers), so the difference has far less variance than either arm.
    noise     only the raw-material cost path is resampled; it is the exogenous process the firm faces. The
              reported Monte-Carlo standard error is of A* itself, so a prediction is never called biased on the
              strength of fewer draws than the claim needs.

WHAT IS NOT CLAIMED
    The `lcb` bound is NOT asserted to have nominal coverage; gates.py lists why (overlapping judgement windows,
    nonstationarity, optional stopping, misspecification). This module MEASURES coverage against A* and reports it
    with a Monte-Carlo interval of its own, so "90% bound covers 72% of the time" is a measurement, not a
    contradiction of a claim nobody made. coverage() refuses to call a bound calibrated; it reports the number.

LEAKAGE
    Continuations are researcher-only and cannot reach an agent:
      * they draw from CONTINUATION_SEED_BASE upward, disjoint from evaluation seeds and from
        gate_study.ORACLE_SEED_BASE;
      * they run in this module's own continuation function, never through an agent's feedback queue;
      * every function here is pure in its inputs: it copies what it reads and returns new frames;
      * researcher-only columns are prefixed `researcher_` and are rejected by the gate interface.
    tests/test_gate_validation.py checks each of these, including that running a validation leaves a rerun of the
    same market bit-identical.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .gate_study import ORACLE_SEED_BASE
from .params import Scenario

# Seeds for researcher-only continuations. Disjoint from evaluation seeds (small) and from the oracle benchmark's
# block, so a continuation can never replay an evaluation market.
CONTINUATION_SEED_BASE = 11_000_000
assert CONTINUATION_SEED_BASE > ORACLE_SEED_BASE

RESEARCHER_PREFIX = "researcher_"

ESTIMANDS: Dict[str, str] = {
    "observed_feedback": "g_t: the advantage the gate learns from. Estimated counterfactual on the firm's believed "
                         "demand curve with rivals held (gain, lcb), or own realized payoff in a randomized trial "
                         "minus an action-independent baseline (explore). Contains the firm's misspecification.",
    "realized_advantage": "A_t^realized: the researcher's ex-post fork on the single shock path that occurred, "
                          "true demand and true costs, horizon H. One draw, so it carries continuation noise.",
    "conditional_expected_net": "A*_t = E[sum_h discount^h (U_h(x*) - U_h(x_B)) | state_t] - adjustment cost, under "
                                "the declared continuation policy and horizon. Estimated from independent "
                                "continuations. This is the estimand a gate's prediction is judged against.",
}

CONTINUATION_POLICIES: Dict[str, str] = {
    "rivals_hold": "The evaluated firm holds the action under test; every rival holds its period-t output. Matches "
                   "the assumption the firm's own counterfactual feedback makes.",
}
REGIMES: Dict[str, str] = {
    "frozen": "The demand regime at the decision period persists through the horizon.",
}


@dataclass(frozen=True)
class ContinuationSpec:
    """How a conditional expectation is estimated. Frozen and hashed into the plan digest."""
    horizon: Optional[int] = None          # None = the judgement window W of the scenario
    draws: int = 64                        # independent continuations per state (shared across the two arms)
    policy: str = "rivals_hold"
    regime: str = "frozen"
    discount: float = 1.0
    seed_base: int = CONTINUATION_SEED_BASE

    def __post_init__(self):
        if self.policy not in CONTINUATION_POLICIES:
            raise ValueError(f"unknown continuation policy {self.policy!r}")
        if self.regime not in REGIMES:
            raise ValueError(f"unknown regime treatment {self.regime!r}")
        if self.draws < 2:
            raise ValueError("a conditional expectation needs at least two independent continuations")
        if self.seed_base <= ORACLE_SEED_BASE:
            raise ValueError("continuation seeds must not overlap the oracle benchmark's block")

    def resolved_horizon(self, scn: Scenario) -> int:
        return int(self.horizon or scn.adaptive.window)


# ================================================================================================ decision states
STATE_COLUMNS = ("market", "firm", "period", "own_output", "rivals_output", "recommended", "adopted", "bin",
                 "predicted_advantage", "lower_bound", "evidence", "researcher_cost", "researcher_true_intercept",
                 "researcher_true_slope", "researcher_fixed_cost", "researcher_realized_advantage")


def sample_decision_states(res, scns: Sequence[Scenario], n_states: int, seed: int = 0,
                           adaptive_only: bool = True) -> pd.DataFrame:
    """Sample decision states (opportunities) from a completed batch, with everything a continuation needs.

    An opportunity is a period in which the recommendation differs from the current output, after burn-in: the only
    periods in which a gate decides anything. Sampling is uniform over opportunities with a stated seed.
    """
    h = res.firm_hist
    if h is None:
        raise ValueError("sample_decision_states needs a batch run with record_firm_history=True")
    B, T, N = h["q"].shape
    burn = res.burn_in
    opp = np.asarray(h["opportunity"], bool)
    keep = np.zeros((B, T, N), bool)
    keep[:, burn:, :] = opp[:, burn:, :]
    if adaptive_only:
        sel = np.array([[f.selection == "Adaptive" for f in s.firms] for s in scns], bool)
        keep &= sel[:, None, :]
    b, t, i = np.nonzero(keep)
    if len(b) == 0:
        return pd.DataFrame(columns=list(STATE_COLUMNS))
    rng = np.random.default_rng(seed)
    pick = np.arange(len(b)) if len(b) <= n_states else rng.choice(len(b), n_states, replace=False)
    pick = np.sort(pick)
    b, t, i = b[pick], t[pick], i[pick]
    q = np.asarray(h["q"], float)
    q_prev = np.concatenate([np.full((B, 1, N), np.nan), q[:, :-1, :]], axis=1)
    rec = np.asarray(h["rec"], float)
    edges = np.asarray(scns[0].adaptive.bin_edges, float)
    own = q_prev[b, t, i]
    change = np.abs(rec[b, t, i] - own)
    rows = dict(
        market=b, firm=i, period=t, own_output=own,
        rivals_output=np.asarray(res.quantity)[b, t] - q[b, t, i],
        recommended=rec[b, t, i], adopted=np.asarray(h["deviate"], bool)[b, t, i],
        bin=np.clip(np.searchsorted(edges, change, side="right"), 0, len(edges)),
        predicted_advantage=_opt(h, "adv_pred", b, t, i), lower_bound=_opt(h, "adv_bound", b, t, i),
        evidence=_opt(h, "adv_neff", b, t, i),
        researcher_cost=np.asarray(res.cost)[b, t],
        researcher_true_intercept=np.asarray(res.p_max_path)[b, t],
        researcher_true_slope=np.asarray(res.slope_path)[b, t],
        researcher_fixed_cost=np.array([scns[bb].flex_cost(scns[bb].firms[ii].flex) for bb, ii in zip(b, i)]),
        researcher_realized_advantage=_opt(h, "gain_full", b, t, i),
    )
    return pd.DataFrame(rows)[list(STATE_COLUMNS)]


def _opt(h, key, b, t, i):
    v = h.get(key)
    return np.asarray(v, float)[b, t, i] if v is not None and getattr(v, "ndim", 0) == 3 else np.full(len(b), np.nan)


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in sorted(o.items())}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.integer, np.floating)):
        return float(o)
    return o


def researcher_only_columns(columns) -> List[str]:
    return sorted(c for c in columns if str(c).startswith(RESEARCHER_PREFIX))


# ================================================================================================ continuations
def _cost_paths(c0: np.ndarray, delta: float, lo: float, hi: float, steps: int, rng) -> np.ndarray:
    """Reflecting random walk, the raw-material cost process of heiner_abm.engine, continued from c0.

    Shape (len(c0), steps + 1); column 0 is c0 itself, the cost already realized at the decision period.
    """
    out = np.empty((len(c0), steps + 1))
    out[:, 0] = c0
    for k in range(1, steps + 1):
        v = out[:, k - 1] + delta * rng.standard_normal(len(c0))
        v = np.where(v <= lo, 2 * lo - v, v)
        v = np.where(v >= hi, 2 * hi - v, v)
        out[:, k] = v
    return out


def continuation_advantage(states: pd.DataFrame, scn: Scenario, spec: ContinuationSpec,
                           cost: Optional[float] = None) -> pd.DataFrame:
    """A*: conditional expected NET advantage of adopting, per sampled state, from independent continuations.

    For each state the two arms (adopt the recommendation, keep the current output) are continued under the same
    resampled cost paths - common random numbers, so the difference is far less noisy than either arm. Returns the
    mean, its Monte-Carlo standard error, and the draws used. Pure: `states` is not modified.
    """
    if states.empty:
        return states.assign(**{k: [] for k in ("A_star", "A_star_se", "draws")})
    H = spec.resolved_horizon(scn)
    c_adjust = float(scn.adaptive.adjust_cost if cost is None else cost)
    gam = float(spec.discount)
    m = scn.market
    lo, hi, delta = float(m.p_min), float(m.c_max), float(m.delta)
    n = len(states)
    adopt = states["recommended"].to_numpy(float)
    keep = states["own_output"].to_numpy(float)
    rivals = states["rivals_output"].to_numpy(float)
    pm = states["researcher_true_intercept"].to_numpy(float)
    sl = states["researcher_true_slope"].to_numpy(float)
    F = states["researcher_fixed_cost"].to_numpy(float)
    c_now = states["researcher_cost"].to_numpy(float)

    mean = np.zeros(n)
    m2 = np.zeros(n)
    for d in range(spec.draws):
        # one independent continuation per state; the seed depends only on the draw index and the state's identity,
        # so the estimate is reproducible and cannot coincide with any evaluation or oracle seed
        rng = np.random.default_rng([spec.seed_base, d])
        paths = _cost_paths(c_now, delta, lo, hi, H - 1, rng)
        adv = np.zeros(n)
        w = 1.0
        for k in range(H):
            c_k = paths[:, k]
            p_adopt = np.maximum(lo, pm - sl * (rivals + adopt))
            p_keep = np.maximum(lo, pm - sl * (rivals + keep))
            adv += w * (((p_adopt - c_k) * adopt - F) - ((p_keep - c_k) * keep - F))
            w *= gam
        adv -= c_adjust                                   # net of the charge for departing from the default
        prev = mean.copy()
        mean += (adv - mean) / (d + 1)
        m2 += (adv - prev) * (adv - mean)
    var = m2 / max(spec.draws - 1, 1)
    out = states.copy()
    out["A_star"] = mean
    out["A_star_se"] = np.sqrt(var / spec.draws)
    out["draws"] = spec.draws
    return out


def one_period_advantage(states: pd.DataFrame, scn: Scenario) -> np.ndarray:
    """The H = 1, zero-cost advantage in closed form: the engine's own one-period counterfactual.

    Used as the pinning invariant in tests. At the decision period no rival has yet reacted and no cost has yet
    moved, so a correct continuation must reproduce this exactly however it is implemented.
    """
    lo = float(scn.market.p_min)
    adopt = states["recommended"].to_numpy(float)
    keep = states["own_output"].to_numpy(float)
    rivals = states["rivals_output"].to_numpy(float)
    pm = states["researcher_true_intercept"].to_numpy(float)
    sl = states["researcher_true_slope"].to_numpy(float)
    F = states["researcher_fixed_cost"].to_numpy(float)
    c = states["researcher_cost"].to_numpy(float)
    p_adopt = np.maximum(lo, pm - sl * (rivals + adopt))
    p_keep = np.maximum(lo, pm - sl * (rivals + keep))
    return ((p_adopt - c) * adopt - F) - ((p_keep - c) * keep - F)


# ================================================================================================ evaluation
def predicted_net(df: pd.DataFrame, scn: Scenario, cost: Optional[float] = None) -> np.ndarray:
    """The gate's prediction on the same footing as A*: gross prediction minus the adaptation charge.

    gates.py keeps the charge out of the learned table and applies it at the threshold, so a gross prediction and a
    net estimand are not comparable until the charge is subtracted from one of them.
    """
    c = float(scn.adaptive.adjust_cost if cost is None else cost)
    return df["predicted_advantage"].to_numpy(float) - c


def calibration(df: pd.DataFrame, scn: Scenario, n_bins: int = 10, cost: Optional[float] = None):
    """Bias and calibration of the gate's prediction against A*. Returns (summary, binned curve for plotting)."""
    pred, truth = predicted_net(df, scn, cost), df["A_star"].to_numpy(float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[ok], truth[ok]
    if len(pred) < 3:
        return {"n": int(len(pred))}, pd.DataFrame(columns=["bin", "pred_mean", "truth_mean", "truth_se", "n"])
    bias = float(np.mean(pred - truth))
    slope, intercept = np.polyfit(pred, truth, 1) if np.ptp(pred) > 0 else (np.nan, np.nan)
    summary = {
        "n": int(len(pred)),
        "bias": bias,                                  # > 0: the gate is optimistic about adapting
        "mean_absolute_error": float(np.mean(np.abs(pred - truth))),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "slope": float(slope),                         # 1 = perfectly calibrated in the regression sense
        "intercept": float(intercept),
        "correlation": float(np.corrcoef(pred, truth)[0, 1]) if np.ptp(truth) > 0 else np.nan,
        "note": "slope 1 and bias 0 is calibration in the regression sense only; it does not imply the bound has "
                "nominal coverage (see coverage()).",
    }
    q = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(q[1:-1], pred, side="right"), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = idx == k
        if m.sum():
            rows.append(dict(bin=k, pred_mean=float(pred[m].mean()), truth_mean=float(truth[m].mean()),
                             truth_se=float(truth[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else np.nan,
                             n=int(m.sum())))
    return summary, pd.DataFrame(rows)


def coverage(df: pd.DataFrame, scn: Scenario, cost: Optional[float] = None) -> Dict[str, Any]:
    """Empirical one-sided coverage of the reported lower bound against A*. A measurement, never an assertion.

    The bound is reported at a nominal level, but gates.py claims no nominal coverage for it (overlapping judgement
    windows, nonstationarity, optional stopping, misspecification). This returns the observed share with a Wilson
    interval so the gap can be read directly, and states the target the number refers to.
    """
    c = float(scn.adaptive.adjust_cost if cost is None else cost)
    lb, truth = df["lower_bound"].to_numpy(float) - c, df["A_star"].to_numpy(float)
    ok = np.isfinite(lb) & np.isfinite(truth)
    lb, truth = lb[ok], truth[ok]
    n = len(lb)
    if n == 0:
        return {"n": 0, "nominal": float(scn.adaptive.confidence), "empirical": np.nan}
    hits = int(np.sum(lb <= truth))
    p = hits / n
    z = 1.959963984540054
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {
        "n": n, "nominal": float(scn.adaptive.confidence), "empirical": float(p),
        "ci_lo": float(max(0.0, centre - half)), "ci_hi": float(min(1.0, centre + half)),
        "target": "P(lower bound <= A*), with A* the conditional expected net advantage of this state under the "
                  "declared continuation policy and horizon",
        "claim": "measured, not asserted: no nominal coverage is claimed for this bound (see heiner_abm.gates)",
    }


def decision_errors(df: pd.DataFrame) -> Dict[str, Any]:
    """Both error types and what each costs, in payoff units.

    A* is already net of the adaptation charge, so the right decision at a state is to adopt exactly when A* > 0.
      false adaptation    adopted while A* <= 0: the payoff given up is -A*
      missed adaptation   kept while A* > 0: the payoff forgone is A*
    """
    truth = df["A_star"].to_numpy(float)
    adopted = df["adopted"].to_numpy(bool)
    ok = np.isfinite(truth)
    truth, adopted = truth[ok], adopted[ok]
    n = len(truth)
    if n == 0:
        return {"n": 0}
    beneficial = truth > 0
    false_adapt = adopted & ~beneficial
    missed = ~adopted & beneficial
    return {
        "n": n,
        "opportunities_beneficial": float(np.mean(beneficial)),
        "adoption_rate": float(np.mean(adopted)),
        "false_adaptation_rate": float(np.mean(false_adapt)),
        "missed_adaptation_rate": float(np.mean(missed)),
        # share of each error among the decisions that could have made it
        "false_given_adopted": float(np.mean(~beneficial[adopted])) if adopted.any() else np.nan,
        "missed_given_beneficial": float(np.mean(~adopted[beneficial])) if beneficial.any() else np.nan,
        # economic consequence, in payoff per sampled decision
        "cost_of_false_adaptation": float(-np.sum(truth[false_adapt]) / n),
        "cost_of_missed_adaptation": float(np.sum(truth[missed]) / n),
        "regret_total": float((-np.sum(truth[false_adapt]) + np.sum(truth[missed])) / n),
        "realized_vs_best": float(np.sum(truth[adopted]) / n), "best_possible": float(np.sum(truth[beneficial]) / n),
    }


def decompose(df: pd.DataFrame, scn: Scenario, cost: Optional[float] = None) -> Dict[str, Any]:
    """Separate the gap between what the gate predicts and what is true into its two parts.

        prediction - A*  =  (prediction - realized)  +  (realized - A*)
                            misspecification + luck    continuation noise of the single realized path
    The second term has mean zero if the realized path is a draw from the same continuation distribution; it is not
    exactly, because the engine's realized fork lets rivals re-decide while A* holds them. Reported, not assumed.
    """
    c = float(scn.adaptive.adjust_cost if cost is None else cost)
    pred = df["predicted_advantage"].to_numpy(float) - c
    realized = df["researcher_realized_advantage"].to_numpy(float) - c
    truth = df["A_star"].to_numpy(float)
    ok = np.isfinite(pred) & np.isfinite(realized) & np.isfinite(truth)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    pred, realized, truth = pred[ok], realized[ok], truth[ok]
    return {
        "n": int(len(pred)),
        "prediction_minus_truth": float(np.mean(pred - truth)),
        "prediction_minus_realized": float(np.mean(pred - realized)),
        "realized_minus_truth": float(np.mean(realized - truth)),
        "sd_realized_minus_truth": float(np.std(realized - truth, ddof=1)),
        "note": "realized - truth is not pure noise: the engine's realized fork lets rivals re-decide, while the "
                "continuation policy here holds them. The difference is part of this term.",
    }


# ================================================================================================ feedback regimes
# Each regime is an edit to the base scenario, isolating one property of the feedback a gate learns from. The
# estimator is the same in all of them; only what reaches it changes.
FEEDBACK_REGIMES: Dict[str, Dict[str, Any]] = {
    "clean": dict(note="One-period judgement, no delay, no observation noise, no regime shifts: consecutive "
                       "feedback values do not overlap and nothing is missing.",
                  window=1, obs_delay=0, obs_noise=0.0, hazard=0.0),
    "overlapping": dict(note="A judgement window of W > 1 means consecutive decisions' windows share periods, so "
                             "their feedback values are positively correlated and the standard error understates "
                             "the uncertainty.",
                        window=20, obs_delay=0, obs_noise=0.0, hazard=0.0),
    "delayed": dict(note="Every release is postponed, so the estimate acts on older evidence.",
                    window=20, obs_delay=5, obs_noise=0.0, hazard=0.0),
    "noisy": dict(note="Observation noise in what the firm sees when forming its counterfactual.",
                  window=20, obs_delay=0, obs_noise=4.0, hazard=0.0),
    "missing": dict(note="Demand-regime shifts invalidate pending judgements, so some decisions never produce any "
                         "feedback at all. A missing outcome is not a zero gain.",
                    window=20, obs_delay=0, obs_noise=0.0, hazard=0.02),
    "correlated": dict(note="High cost volatility makes consecutive judgements share a common drift on top of the "
                            "overlap, the hardest case for an independence assumption.",
                       window=20, obs_delay=0, obs_noise=0.0, hazard=0.0, delta=24.0),
}


def regime_scenario(base: Scenario, regime: str) -> Scenario:
    """The base scenario with one feedback regime applied. Unknown keys are refused rather than ignored."""
    if regime not in FEEDBACK_REGIMES:
        raise ValueError(f"unknown feedback regime {regime!r} (one of {', '.join(FEEDBACK_REGIMES)})")
    spec = FEEDBACK_REGIMES[regime]
    s = base.copy()
    import dataclasses
    s.adaptive = dataclasses.replace(s.adaptive, window=int(spec["window"]))
    s.info = dataclasses.replace(s.info, obs_delay=int(spec["obs_delay"]), obs_noise=float(spec["obs_noise"]))
    s.structural = dataclasses.replace(s.structural, hazard=float(spec["hazard"]),
                                       enabled=bool(spec["hazard"] > 0) if hasattr(s.structural, "enabled")
                                       else float(spec["hazard"]) > 0)
    if "delta" in spec:
        s.market = dataclasses.replace(s.market, delta=float(spec["delta"]))
    return s


# ================================================================================================ memory variants
# The existing rule forgets per update: every released feedback value multiplies the statistics by lambda, whether
# it arrived one period later or fifty. These are the alternatives, compared as ESTIMATORS on the same recorded
# evidence streams (see the module note on what that does and does not establish).
VARIANTS: Tuple[str, ...] = ("per_update_forgetting", "calendar_aging", "change_reset", "explore_continuing")
VARIANT_NOTES: Dict[str, str] = {
    "per_update_forgetting": "The existing rule: statistics decay by lambda once per released value, regardless of "
                             "how much calendar time passed.",
    "calendar_aging": "Statistics decay by lambda^(periods elapsed), so evidence ages with the environment rather "
                      "than with the arrival count. Equal to the existing rule when releases are one period apart.",
    "change_reset": "Per-update forgetting, plus discarding the bin's evidence when a demand-regime change is "
                    "detected. Sharp but destroys information if the detector fires spuriously.",
    "explore_continuing": "Per-update forgetting, plus continuing to randomize with probability `explore_after` "
                          "even once a bin has been classified decisively, so a changed bin can be re-learned. "
                          "The propensity and the payoff given up are logged.",
}


@dataclass(frozen=True)
class VariantSpec:
    name: str = "per_update_forgetting"
    memory: float = 0.97                   # lambda
    confidence: float = 0.9
    min_evidence: float = 5.0
    explore_rate: float = 0.2              # probability that an UNCERTAIN bin is randomized (all variants)
    explore_after: float = 0.0             # explore_continuing: probability of randomizing after a DECISIVE call
    seed: int = 0

    def __post_init__(self):
        if self.name not in VARIANTS:
            raise ValueError(f"unknown variant {self.name!r} (one of {', '.join(VARIANTS)})")
        if self.name == "explore_continuing" and not 0.0 < self.explore_after <= 1.0:
            raise ValueError("explore_continuing needs an exploration probability in (0, 1]")
        if self.name != "explore_continuing" and self.explore_after:
            raise ValueError("explore_after applies only to explore_continuing; keep the variants separate")


def evidence_streams(res, scns: Sequence[Scenario]) -> pd.DataFrame:
    """Released feedback, in the order each firm received it: one row per (market, firm, released value).

    Read from the engine's own record, so a variant is replayed over exactly the evidence the firm had, not over a
    reconstruction of it.
    """
    h = res.firm_hist
    if h is None or "fb_value" not in h or "fb_release" not in h:
        raise ValueError("evidence_streams needs a batch run with record_firm_history=True")
    val, rel = np.asarray(h["fb_value"], float), np.asarray(h["fb_release"], float)
    q = np.asarray(h["q"], float)
    rec = np.asarray(h["rec"], float)
    B, T, N = val.shape
    q_prev = np.concatenate([np.full((B, 1, N), np.nan), q[:, :-1, :]], axis=1)
    edges = np.asarray(scns[0].adaptive.bin_edges, float)
    shifts = np.asarray(res.shifts, bool) if res.shifts is not None else np.zeros((B, T), bool)
    b, t, i = np.nonzero(np.isfinite(val) & (rel >= 0))
    if len(b) == 0:
        return pd.DataFrame(columns=["market", "firm", "bin", "decided", "released", "g", "change_since_last"])
    change = np.abs(rec[b, t, i] - q_prev[b, t, i])
    df = pd.DataFrame(dict(market=b, firm=i, decided=t, released=rel[b, t, i].astype(int), g=val[b, t, i],
                           bin=np.clip(np.searchsorted(edges, change, side="right"), 0, len(edges))))
    df = df.sort_values(["market", "firm", "released", "decided"], kind="stable").reset_index(drop=True)
    # whether a regime shift occurred between this release and the previous one for the same firm and bin
    prev_rel = df.groupby(["market", "firm", "bin"])["released"].shift(1)
    seen = []
    for (mk, rl), pr in zip(zip(df["market"], df["released"]), prev_rel):
        seen.append(bool(shifts[mk, int(pr): int(rl) + 1].any()) if np.isfinite(pr) else False)
    df["change_since_last"] = seen
    return df


def replay_variant(streams: pd.DataFrame, spec: VariantSpec, cost: float = 0.0) -> pd.DataFrame:
    """Replay one memory variant over recorded evidence streams, one row per released value.

    Before each value is absorbed, the variant's current estimate of the bin is recorded together with the
    classification it would make and, where the variant randomizes, whether it did. The value then updates the
    statistics. The next value in the same bin is the one-step-ahead target: the estimate is scored against
    evidence it had not yet seen.
    """
    from scipy.special import stdtrit
    rng = np.random.default_rng([abs(hash(spec.name)) % (2 ** 31), spec.seed])
    stats: Dict[Tuple[int, int, int], List[float]] = {}        # (market, firm, bin) -> [S0, S1, S2, W2, last_period]
    lam = float(spec.memory)
    rows = []
    for r in streams.itertuples(index=False):
        key = (int(r.market), int(r.firm), int(r.bin))
        st = stats.get(key)
        if st is None:
            st = [0.0, 0.0, 0.0, 0.0, float(r.released)]
        if spec.name == "change_reset" and bool(r.change_since_last):
            st = [0.0, 0.0, 0.0, 0.0, float(r.released)]
        decay = lam
        if spec.name == "calendar_aging":
            elapsed = max(0.0, float(r.released) - st[4])
            decay = lam ** max(elapsed, 1.0)                   # lambda per period of calendar time, not per arrival
        S0, S1, S2, W2 = st[0], st[1], st[2], st[3]
        mean = S1 / S0 if S0 > 0 else np.nan
        neff = (S0 * S0 / W2) if W2 > 0 else 0.0
        den = S0 * S0 - W2
        var = max(S0 * S2 - S1 * S1, 0.0) / den if den > 0 else np.nan
        se = np.sqrt(var / neff) if neff > 0 and np.isfinite(var) else np.nan
        bound = mean - float(stdtrit(max(neff - 1, 1.0), spec.confidence)) * se if np.isfinite(se) else np.nan
        upper = mean + float(stdtrit(max(neff - 1, 1.0), spec.confidence)) * se if np.isfinite(se) else np.nan
        decided = bool(neff >= spec.min_evidence and np.isfinite(bound) and (bound > cost or upper <= cost))
        classification = ("adopt" if np.isfinite(bound) and bound > cost else
                          "keep" if np.isfinite(upper) and upper <= cost else "uncertain") if decided else "uncertain"
        # an uncertain bin randomizes at the shared exploration rate; only explore_continuing keeps
        # randomizing once the bin has been classified decisively
        propensity = ((float(spec.explore_after) if spec.name == "explore_continuing" else 0.0)
                      if decided else float(spec.explore_rate))
        explored = bool(rng.random() < propensity) if propensity > 0 else False
        rows.append(dict(market=int(r.market), firm=int(r.firm), bin=int(r.bin), released=int(r.released),
                         estimate=mean, se=se, evidence=neff, lower_bound=bound, upper_bound=upper,
                         decided=decided, classification=classification,
                         exploration_propensity=propensity, explored=explored, g=float(r.g)))
        # absorb the value
        st = [decay * S0 + 1.0, decay * S1 + float(r.g), decay * S2 + float(r.g) ** 2,
              (decay * decay) * W2 + 1.0, float(r.released)]
        stats[key] = st
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["variant"] = spec.name
    out["next_g"] = out.groupby(["market", "firm", "bin"])["g"].shift(-1)
    return out


def _finite_mean(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else 0.0


def score_variant(replayed: pd.DataFrame, cost: float = 0.0) -> Dict[str, Any]:
    """One-step-ahead predictive score of a variant, plus what its exploration cost.

    The target is the next feedback value in the same bin: evidence the estimate had not seen. That is an
    observable, so this comparison needs no counterfactual; it measures how well a memory rule tracks the stream
    it is fed, which is what the variants differ in.
    """
    if replayed.empty:
        return {"n": 0}
    d = replayed.dropna(subset=["estimate", "next_g"])
    if d.empty:
        return {"n": 0, "note": "no bin received two values, so nothing could be predicted one step ahead"}
    err = d["estimate"].to_numpy(float) - d["next_g"].to_numpy(float)
    lb = d["lower_bound"].to_numpy(float)
    nxt = d["next_g"].to_numpy(float)
    have_lb = np.isfinite(lb)
    explored = replayed["explored"].to_numpy(bool)
    return {
        "variant": str(replayed["variant"].iloc[0]),
        "n": int(len(d)),
        "bias": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean_absolute_error": float(np.mean(np.abs(err))),
        "coverage_vs_next_value": float(np.mean(lb[have_lb] <= nxt[have_lb])) if have_lb.any() else np.nan,
        "decided_share": float(np.mean(replayed["decided"].to_numpy(bool))),
        "exploration_propensity": float(np.mean(replayed["exploration_propensity"].to_numpy(float))),
        "exploration_share": float(np.mean(explored)),
        # what randomizing gave up: a trial that contradicts the classification forgoes the advantage of the
        # classified action, approximated by the absolute current estimate net of the charge
        "exploration_cost": _finite_mean(np.abs(replayed["estimate"].to_numpy(float) - cost)[explored]),
        "target": "the next released feedback value in the same bin (an observable, not a counterfactual)",
    }


def compare_variants(streams: pd.DataFrame, specs: Sequence[VariantSpec], cost: float = 0.0) -> pd.DataFrame:
    """Every variant on the same evidence streams. Separate variants, never combined in one estimator."""
    return pd.DataFrame([score_variant(replay_variant(streams, sp, cost), cost) for sp in specs])


# ================================================================================================ the study
@dataclass(frozen=True)
class ValidationPlan:
    """A frozen, hashed specification of a validation run, so a reported number identifies the run behind it."""
    version: str = "1.0"
    gates: Tuple[str, ...] = ("gain", "lcb")
    regimes: Tuple[str, ...] = ("clean", "overlapping", "delayed", "noisy", "missing", "correlated")
    markets: int = 6
    periods: int = 400
    burn_in: int = 50
    seed0: int = 1
    states_per_cell: int = 200
    state_seed: int = 20261012
    # the memory variants are compared in a regime that actually contains regime changes: with none, a
    # change-detection reset never fires and is identical to per-update forgetting by construction
    variant_regime: str = "missing"
    continuation: ContinuationSpec = field(default_factory=ContinuationSpec)
    variants: Tuple[Tuple[str, float], ...] = (("per_update_forgetting", 0.0), ("calendar_aging", 0.0),
                                               ("change_reset", 0.0), ("explore_continuing", 0.1))

    def variant_specs(self, memory: float, confidence: float, min_evidence: float,
                      explore_rate: float = 0.2) -> List[VariantSpec]:
        return [VariantSpec(name=n, memory=memory, confidence=confidence, min_evidence=min_evidence,
                            explore_rate=explore_rate, explore_after=p) for n, p in self.variants]

    def to_json(self) -> str:
        return json.dumps(_plain(asdict(self)), sort_keys=True, indent=1, default=str)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


@dataclass
class GateValidation:
    plan_hash: str
    states: pd.DataFrame            # sampled decision states with A* and its standard error
    calibration: pd.DataFrame       # binned prediction vs A*, per gate and regime (the plot data)
    summary: pd.DataFrame           # one row per gate and regime: bias, coverage, error rates, their cost
    variants: pd.DataFrame          # one row per memory variant
    settings: Dict[str, Any] = field(default_factory=dict)
    not_run: Dict[str, str] = field(default_factory=dict)   # gate x regime -> why it could not run here


def experiment_spec(plan: ValidationPlan, base: Scenario) -> Dict[str, Any]:
    """Everything needed to reproduce a validation run, including the scenario it ran on."""
    from .scenarios import DEFAULTS
    return {"plan": json.loads(plan.to_json()), "plan_hash": plan.digest,
            "estimands": dict(ESTIMANDS), "continuation_policies": dict(CONTINUATION_POLICIES),
            "regimes": {k: v["note"] for k, v in FEEDBACK_REGIMES.items()},
            "variant_notes": dict(VARIANT_NOTES),
            "scenario": {"periods": base.periods, "burn_in": base.burn_in, "seed": base.seed,
                         "n_firms": base.n_firms, "adaptive": asdict(base.adaptive),
                         "info": base.info.to_dict(), "market": asdict(base.market)},
            "seed_blocks": {"evaluation": f"{plan.seed0} and upward",
                            "oracle_benchmark": f"{ORACLE_SEED_BASE} and upward",
                            "researcher_continuations": f"{plan.continuation.seed_base} and upward"},
            "reproduce": "heiner_abm.gate_validation.run_validation(plan, base)",
            "defaults_digest": hashlib.sha256(json.dumps(_plain(DEFAULTS), sort_keys=True).encode()).hexdigest()[:16]}


def run_validation(plan: ValidationPlan, base: Scenario, gates: Optional[Sequence[str]] = None,
                   regimes: Optional[Sequence[str]] = None, progress=None) -> GateValidation:
    """Run the validation: sample decision states, estimate A*, and score each gate against it in each regime."""
    from .engine import run_batch
    from .gate_study import gate_scenario
    from .information import UnsupportedInformation
    note = progress or (lambda f, m: None)
    gates = list(gates or plan.gates)
    regimes = list(regimes or plan.regimes)
    all_states, curves, rows, not_run = [], [], [], {}
    total = max(1, len(gates) * len(regimes))
    done = 0
    for regime in regimes:
        reg_base = regime_scenario(base, regime)
        reg_base.periods, reg_base.burn_in = plan.periods, plan.burn_in
        for gate in gates:
            note(done / total, f"{regime}: {gate}")
            scns = [gate_scenario(reg_base, gate, plan.seed0 + k) for k in range(plan.markets)]
            try:
                res = run_batch(scns, record_firm_history=True, horizon=reg_base.adaptive.window)
            except UnsupportedInformation as err:
                # the same treatment gate_study gives a gate the information treatment cannot support: report it
                not_run[f"{gate} / {regime}"] = str(err)
                done += 1
                continue
            st = sample_decision_states(res, scns, plan.states_per_cell, seed=plan.state_seed)
            if st.empty:
                done += 1
                continue
            st = continuation_advantage(st, reg_base, plan.continuation)
            st.insert(0, "gate", gate)
            st.insert(1, "regime", regime)
            cal, curve = calibration(st, reg_base)
            curve.insert(0, "gate", gate)
            curve.insert(1, "regime", regime)
            cov = coverage(st, reg_base)
            err = decision_errors(st)
            dec = decompose(st, reg_base)
            rows.append(dict(gate=gate, regime=regime, states=len(st),
                             **{f"cal_{k}": v for k, v in cal.items() if k != "note"},
                             **{f"cov_{k}": v for k, v in cov.items() if k in ("nominal", "empirical", "ci_lo",
                                                                               "ci_hi", "n")},
                             **{k: v for k, v in err.items() if k != "n"},
                             **{f"gap_{k}": v for k, v in dec.items() if isinstance(v, float)}))
            all_states.append(st)
            curves.append(curve)
            done += 1
    note(0.9, "memory variants")
    specs = plan.variant_specs(base.adaptive.memory, base.adaptive.confidence, base.adaptive.min_evidence,
                               float(base.adaptive.explore_rate))
    try:
        scns = [gate_scenario(regime_scenario(base, plan.variant_regime), "lcb", plan.seed0 + k)
                for k in range(plan.markets)]
        for s in scns:
            s.periods, s.burn_in = plan.periods, plan.burn_in
        res = run_batch(scns, record_firm_history=True, horizon=base.adaptive.window)
        variants = compare_variants(evidence_streams(res, scns), specs, float(base.adaptive.adjust_cost))
    except UnsupportedInformation as err:
        not_run["memory variants"] = str(err)
        variants = pd.DataFrame()
    note(1.0, "done")
    return GateValidation(
        plan_hash=plan.digest,
        states=pd.concat(all_states, ignore_index=True) if all_states else pd.DataFrame(),
        calibration=pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
        summary=pd.DataFrame(rows), variants=variants,
        settings=experiment_spec(plan, base), not_run=not_run)
