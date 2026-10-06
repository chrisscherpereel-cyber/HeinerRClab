"""Selection gates of the Adaptive (learned reliability) rule: evidence statistics and decisions per size bin.

PROPOSED EXTENSION. Heiner (1983) states when restricting flexibility pays (the reliability condition); he did not
propose confidence bounds, minimum-evidence rules or exploration. The confidence-sensitive and exploration-enabled
gates below are the laboratory's own operationalizations of how an agent might act on uncertain estimates of its own
reliability. Results with them test these implementations, not Heiner's theory.

Target quantity
    A_t = E[ U_t(x*) - U_t(x_B) | I_t ]: expected payoff of the candidate action x* (the production rule's
    recommendation) minus expected payoff of the default x_B (rule B: keep the current output), over the judgement
    window of W periods starting at t (AdaptiveParams.window), conditional on the information I_t available when
    deciding. Every gate coarsens I_t to the size bin of the recommended change |x* - x_B| (AdaptiveParams.bin_edges)
    and estimates the bin-level advantage A_b = E[A_t | bin b] for each firm. Payoffs are gross market payoffs; the
    adjustment cost c (AdaptiveParams.adjust_cost) is kept separate and enters only the decision threshold and the
    researcher's net-payoff accounting.

Gates (candidate generation is identical for all of them; only the decision to adopt x* differs)
    gain          Existing estimated-gain gate: adopt if the learned gain >= c. The learned gain is the existing
                  exponentially weighted table, learned = (1 - lambda) * S1 (S1 below), so for a bin with evidence the
                  rule is exactly "weighted mean >= c". With no evidence it adopts (the existing optimistic start:
                  learned = 0 >= 0 when c = 0; for c > 0, 0 >= c * (1 - lambda) * S0 = 0 still holds). No minimum
                  evidence. With c = 0 it is bit-identical to the rule before this extension. With lambda = 1 the
                  existing table never moves (it weights new feedback by 1 - lambda), so the gate then uses
                  S1 >= c * S0, the same rule.
    lcb           Confidence-sensitive gate: adopt only if the bin has at least `min_evidence` effective feedback items
                  and the lower confidence bound on A_b exceeds c. Otherwise keep x_B.
    explore       Exploration-enabled gate: learns only from its own realized payoffs (chosen-action feedback), never
                  from counterfactuals. Per bin it keeps evidence for two arms, adopt and keep. A bin is decided when
                  both arms have `min_evidence` effective trials and the confidence interval for A_b excludes c (adopt
                  if the lower bound exceeds c, keep if the upper bound is at most c); otherwise it is uncertain. An
                  uncertain decision is a randomized trial with probability `explore_rate` (adopt or keep with
                  probability 1/2 each, from a separate random stream); otherwise it keeps x_B. Only randomized trials
                  are used as evidence. A trial's feedback is the firm's own payoff over the W periods minus an
                  action-independent baseline (its pre-decision payoff with the cost updated to each period's realized
                  cost), which removes the cost shocks common to both arms without biasing the randomized comparison.
    oracle_table  ORACLE benchmark: adopt if the true bin-level advantage, estimated by the researcher from
                  independent simulation runs (heiner_abm.gate_study.oracle_table), exceeds c. It does not learn. The
                  table is accepted only by this gate (Scenario.validate), so oracle values cannot enter ordinary
                  learning.

Evidence statistics (per firm and bin; for `explore` per arm as well)
    Each released feedback value g updates S0 <- lambda*S0 + 1, S1 <- lambda*S1 + g, S2 <- lambda*S2 + g^2,
    W2 <- lambda^2*W2 + 1 (lambda = AdaptiveParams.memory, forgetting per update, as in the existing rule). With
    weights w_k = lambda^age:
        mean      m = S1 / S0                                   (undefined, never 0, without evidence)
        n_eff     S0^2 / W2                                     (Kish effective sample size)
        variance  s^2 = (S0*S2 - S1^2) / (S0^2 - W2)            (unbiased with reliability weights; needs n_eff > 1)
        s.e.      se = sqrt(s^2 / n_eff)
    Lower bound (lcb): m - t(conf; n_eff - 1) * se. Explore: difference of the arms' means, se = sqrt(se1^2 + se0^2),
    Welch-Satterthwaite degrees of freedom.

Assumptions of the uncertainty estimator, and what is not claimed
    The bound would have its nominal one-sided coverage if, within a bin, the feedback values were independent draws
    with a common mean (and approximately normal, or n_eff large). None of this is guaranteed here:
      * dependence: with W > 1 consecutive decisions' judgement windows overlap, so their feedback values are
        positively correlated and se understates the uncertainty;
      * nonstationarity: costs drift and demand regimes shift, so the bin mean moves; forgetting (lambda < 1) trades
        bias for variance and makes n_eff at most (1 + lambda) / (1 - lambda);
      * adaptive sampling: the explore gate stops randomizing once its interval excludes c (optional stopping), and the
        states in which decisions arise depend on past decisions;
      * misspecification: estimated-counterfactual feedback uses the firm's believed demand curve and holds rivals'
        output at its observed path, so it can be biased for A_t.
    No nominal coverage is claimed. Coverage and calibration are measured empirically against the researcher's
    counterfactuals and the oracle benchmark (heiner_abm.gate_study).

Missing outcomes are never zero gains: a bin without evidence has no mean (NaN in the records), only released feedback
updates the statistics, and keep decisions outside randomized trials produce no evidence for the explore gate.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.special import stdtrit

GATES = ("gain", "lcb", "explore", "oracle_table")
GATE_CODES = {g: i for i, g in enumerate(GATES)}
GATE_LABELS = {"gain": "Estimated-gain gate (existing)", "lcb": "Confidence-sensitive gate (proposed extension)",
               "explore": "Exploration-enabled gate (proposed extension)",
               "oracle_table": "ORACLE benchmark gate (true bin advantage from independent runs)"}
ON_CHANGE = ("forget", "reset")
KEEP, ADAPT, UNCERTAIN = 0, 1, 2
N_STATS = 4                                     # S0, S1, S2, W2


def update(S4: np.ndarray, g, lam) -> np.ndarray:
    """One released feedback value g into the statistics S4[..., (S0, S1, S2, W2)]. Shared by both engines."""
    return np.stack([lam * S4[..., 0] + 1.0, lam * S4[..., 1] + g, lam * S4[..., 2] + g * g,
                     (lam * lam) * S4[..., 3] + 1.0], axis=-1)


def summary(S: np.ndarray):
    """Mean, standard error and effective sample size per bin from statistics S[..., 4, n_bins]."""
    S0, S1, S2, W2 = S[..., 0, :], S[..., 1, :], S[..., 2, :], S[..., 3, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(S0 > 0, S1 / S0, np.nan)
        neff = np.where(W2 > 0, S0 * S0 / W2, 0.0)
        den = S0 * S0 - W2
        var = np.where(den > 0, np.maximum(S0 * S2 - S1 * S1, 0.0) / den, np.nan)
        se = np.sqrt(var / neff)
    return mean, se, neff


def _t(df, conf):
    with np.errstate(invalid="ignore"):
        return stdtrit(np.where(df > 0, df, 1.0), conf)


def gate_table(gate: str, learned: np.ndarray, S: np.ndarray, XS: np.ndarray, lam, cost, conf, nmin,
               table: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Decision mode per bin (KEEP, ADAPT or UNCERTAIN) and the recorded prediction for one gate.

    learned (..., nb): existing table; S (..., 4, nb): estimated/oracle feedback statistics; XS (..., 2, 4, nb): explore
    trial statistics (arm 0 keep, arm 1 adopt); lam, cost, conf, nmin: scalars or arrays broadcastable to (..., nb).
    Returns mode, pred (predicted advantage A_b), se, bound (lower confidence bound) and neff (effective evidence)."""
    if gate == "explore":
        m1, se1, n1 = summary(XS[..., 1, :, :])
        m0, se0, n0 = summary(XS[..., 0, :, :])
        pred = m1 - m0
        v1, v0 = se1 * se1, se0 * se0
        se = np.sqrt(v1 + v0)
        with np.errstate(divide="ignore", invalid="ignore"):
            dden = v1 * v1 / (n1 - 1.0) + v0 * v0 / (n0 - 1.0)
            df = np.where(dden > 0, (v1 + v0) * (v1 + v0) / dden, n1 + n0 - 2.0)      # Welch-Satterthwaite
        q = _t(df, conf)
        bound, upper = pred - q * se, pred + q * se
        neff = np.minimum(n1, n0)
        enough = (n1 >= nmin) & (n0 >= nmin)
        mode = np.where(enough & (bound > cost), ADAPT, np.where(enough & (upper <= cost), KEEP, UNCERTAIN))
        return dict(mode=mode.astype(np.int8), pred=pred, se=se, bound=bound, neff=neff)
    if gate == "oracle_table":
        tab = np.asarray(table, dtype=float)
        nan = np.full(tab.shape, np.nan)
        return dict(mode=np.where(tab > cost, ADAPT, KEEP).astype(np.int8), pred=tab, se=nan, bound=nan, neff=nan)
    mean, se, neff = summary(S)
    q = _t(neff - 1.0, conf)
    bound = mean - q * se
    if gate == "gain":
        # learned = (1 - lam) * S1, so "learned >= c (1 - lam) S0" is "mean >= c" (and adopts without evidence); the
        # existing table never moves when lam = 1, so that case uses the sums directly
        mode = np.where(np.asarray(lam) < 1.0, learned >= cost * (1.0 - lam) * S[..., 0, :],
                        S[..., 1, :] >= cost * S[..., 0, :])
        mode = np.where(mode, ADAPT, KEEP)
    elif gate == "lcb":
        mode = np.where((neff >= nmin) & (bound > cost), ADAPT, KEEP)
    else:
        raise ValueError(f"Unknown gate {gate!r}")
    return dict(mode=mode.astype(np.int8), pred=mean, se=se, bound=bound, neff=neff)


def exploration_draws(seed: int, periods: int, n_firms: int) -> np.ndarray:
    """Separate stream (6th child of the scenario seed) for the explore gate's randomized trials, shape (2, periods,
    n_firms): [0] < explore_rate makes an uncertain decision a trial, [1] < 1/2 adopts in a trial. Enabling it leaves
    every other stream intact."""
    ss = np.random.SeedSequence(seed).spawn(6)[5]
    return np.random.default_rng(ss).random((2, periods, n_firms))


def change_signal(scn, belief_p_max: np.ndarray, belief_slope: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Per period, whether the firms learn at decision time that the environment changed: an announced shift (when
    shifts are announced) or a change of the demand curve they are given (believed curve updated after its lag, or
    the true curve). With no demand model only announcements count. Built from Observation contents only."""
    T = len(shift)
    sig = np.zeros(T, dtype=bool)
    info = scn.info
    if info.regime_announced:
        sig |= shift.astype(bool)
    if info.demand_knowledge != "none":
        sig[1:] |= (belief_p_max[1:] != belief_p_max[:-1]) | (belief_slope[1:] != belief_slope[:-1])
    sig[0] = False
    return sig
