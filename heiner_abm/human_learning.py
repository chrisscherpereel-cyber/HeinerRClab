"""Trial-by-trial mechanistic models of human adjustment, learned from what participants actually saw.

PROPOSED EXTENSION. `human_models` fits four hurdle models - inaction band, probabilistic adjustment, inertia and
rounding. All four are STATIC: the probability of adjusting is a function of the current reference signal, and
nothing carries from one decision to the next. They also take that signal from `shadow:opt_br`, the researcher's
filtered best reply, which is never shown to anybody. Those models are descriptions of a decision rule, not of a
learning process, and they are scored against a quantity the participant could not compute.

This module adds five models that ARE learning processes. Each carries a state across the decisions of a block,
updates it from feedback, and decides from that state. The hard constraint, checked by a test rather than
asserted here, is that every input is something the participant saw on screen.

================================================================================ what the participant sees

`stepper.InteractiveMarket.info()` is the screen. Before deciding in period t a participant has:

    cost_estimate   c_hat_t     a noisy estimate of THIS period's unit cost        (shown now)
    last_cost       c_{t-1}     the cost that actually applied last period         (shown now)
    last_price      P_{t-1}     the price that cleared last period
    own_profit      pi_{t-1}    their own profit last period
    own_q           q_{t-1}     their own output last period
    last_market_q   Q_{t-1}     total output last period
    period          t

and nothing else. In particular they are never shown the best reply, the rivals' individual choices, the true
cost, or any model's recommendation except in the aid arm. Two derived quantities below are built only from the
list above:

    u_t  = c_hat_t - c_{t-1}         the cost surprise, the one forward-looking signal on the screen
    r_t  = pi_t / PROFIT_SCALE       realized profit, the one outcome signal

================================================================================ the five models

All five share a lapse rate and a magnitude model, so that what distinguishes them is the ADAPTATION DECISION:

    lapse      with probability eps the decision is random (adjust with probability 1/2, uniform change)
    magnitude  given adjustment, Delta ~ N(-phi * u_t, sd^2) on whole units, conditioned on Delta != 0.
               The sign is negative because a higher cost estimate makes a smaller output better, and both
               terms are on the screen. phi is free, so a participant who ignores the cost estimate has phi = 0.

1. `advantage` - LEARNED EXPECTED ADVANTAGE
   state     V_adjust, V_keep : values of the two actions, both 0 at the start of a block
   update    after acting and seeing profit:  V_a <- V_a + alpha (r_t - V_a)   for the action actually taken
   signal    m_t = V_adjust - V_keep
   choice    P(adjust) = logistic(a0 + beta * m_t + c_condition)
   params    alpha, beta, a0, cV, cN, cR, lapse, phi, sd
   A delta rule on the binary decision itself. The participant never has to represent the market, only which of
   the two things they do tends to pay.

2. `confidence` - ESTIMATION UNCERTAINTY AND CONFIDENCE-SENSITIVE ADAPTATION
   state     V_adjust, V_keep and their posterior variances v_adjust, v_keep
   update    Kalman: v <- v + sigma_w^2 ; k = v / (v + sigma_r^2) ; V <- V + k (r_t - V) ; v <- (1 - k) v
             applied to the action taken; the other action's variance still grows, so not acting makes the
             agent less certain about acting
   signal    m_t = V_adjust - V_keep ;  U_t = sqrt(v_adjust + v_keep)
   choice    P(adjust) = logistic(a0 + beta * m_t - omega * U_t + c_condition)
   params    sigma_w, sigma_r, beta, omega, a0, cV, cN, cR, lapse, phi, sd
   omega > 0 is confidence-sensitive adaptation: act when the estimate is precise, wait when it is not.
   omega = 0 is model 1 with a Kalman learning rate instead of a fixed one, so the two are nested and the test
   of confidence sensitivity is a test on one parameter.
   (An alternative form divides by U_t rather than subtracting it. Subtracting keeps the two terms separable,
   which matters because the whole identifiability question below is whether they can be told apart.)

3. `changebayes` - BAYESIAN BELIEFS ABOUT ENVIRONMENTAL CHANGE
   state     mu_t, s_t : a Gaussian belief about the current cost level
   update    predictive variance s + sigma_obs^2 under no change, + sigma_jump^2 under change;
             posterior change probability r_t = Bayes(h, the two predictive densities | c_hat_t);
             gain is raised by the change belief: g = k + r_t (1 - k) ; mu <- mu + g (c_hat_t - mu)
   signal    the surprise |c_hat_t - mu_t| and the change belief r_t
   choice    P(adjust) = logistic(a0 + beta * |c_hat_t - mu_t| / COST_SCALE + lam * r_t + c_condition)
   params    h, sigma_jump, k, beta, lam, a0, cV, cN, cR, lapse, phi, sd
   The participant tracks the cost and adapts when they believe the world has moved, not when a payoff
   happens to disappoint.

4. `aspiration` - ASPIRATION-BASED SATISFICING
   state     A_t : an aspiration level on profit, started at the first profit seen in the block
   update    A <- A + eta (pi_{t-1} - A)
   choice    P(adjust) = logistic(a0 + beta * (A_t - pi_{t-1}) / PROFIT_SCALE + c_condition)
   params    eta, beta, a0, cV, cN, cR, lapse, phi, sd
   Search when below aspiration, stay when at or above it. beta > 0 is the satisficing prediction; the model is
   free to estimate beta <= 0, which would say the opposite.

5. `risk` - RISK PREFERENCE SEPARATED FROM ESTIMATION UNCERTAINTY
   state     as `confidence`, plus w_t : an exponentially weighted variance of observed profit
   choice    P(adjust) = logistic(a0 + beta m_t - omega U_t - gamma sqrt(w_t) / PROFIT_SCALE + c_condition)
   params    sigma_w, sigma_r, beta, omega, gamma, nu, a0, cV, cN, cR, lapse, phi, sd
   omega is reluctance to act on an IMPRECISE ESTIMATE; gamma is reluctance to act in a VARIABLE WORLD. They are
   different claims and they are separable only if the design moves outcome variability and estimation
   uncertainty independently - which is what the two new manipulations below are for.

================================================================================ block boundaries

Every state above is reset at the start of each block: values, variances, aspiration and cost belief all begin
again. Blocks are different markets with different cost paths, presented as such, so carrying a learned value
across one would be a modelling claim about the participant that the protocol does not support. A test asserts
that decisions in block 2 are unchanged when block 1 is replaced. If a future protocol deliberately runs blocks
as one continuous environment, `CARRY_STATE` documents what would have to change.

================================================================================ what this is not

Everything here is fitted to SYNTHETIC participants. Recovery and power results below describe whether the
design could identify these mechanisms if people behaved this way. They are not evidence that anybody does, and
no number produced by this module is a finding about human behaviour.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr

from . import human_models as HM

# ================================================================================================ constants
MODELS = ("advantage", "confidence", "changebayes", "aspiration", "risk", "inertia_learn")
MODEL_LABELS = {
    "advantage": "Learned expected advantage",
    "confidence": "Estimation uncertainty + confidence-sensitive adaptation",
    "changebayes": "Bayesian beliefs about environmental change",
    "aspiration": "Aspiration-based satisficing",
    "risk": "Risk preference separated from estimation uncertainty",
    "inertia_learn": "Learned advantage + inertia (the confound competitor)",
}
PROFIT_SCALE = 500.0            # profits are in the hundreds; this keeps learned values near unit scale
COST_SCALE = 10.0               # cost surprises are in single digits
LAPSE_RANGE = HM.LAPSE_RANGE
COND_INDEX = HM.COND_INDEX
CARRY_STATE = False             # blocks are separate markets; see the module note

# parameter -> (transform, weak prior mean, weak prior sd) on the unconstrained scale
PARAMS: Dict[str, Tuple[str, float, float]] = {
    "a0": ("id", 0.0, 3.0),
    "beta": ("id", 0.0, 2.0),              # signed: a model may estimate the opposite of its own prediction
    "alpha": ("logit", 0.0, 1.5),
    # omega and gamma are SIGNED. Forcing them positive would make "is there confidence sensitivity?"
    # unanswerable - every estimate would be above zero by construction - and a negative value is a real
    # possibility worth being able to see: adapting MORE when uncertain is exploration, not a bug.
    # The prior spread on these two is deliberately wide. At sd 1.0 - which looked weak - the two-pass
    # empirical Bayes shrank omega from a true mean of 0.92 to an estimate of 0.25, because the first pass was
    # pulled to zero and the second pass inherited that as its population mean. The correlation with the truth
    # was unaffected (0.67 either way), so the shrinkage was invisible in a rank-based check and would have been
    # reported as a recovered parameter with a badly wrong magnitude.
    "omega": ("id", 0.0, 2.5),             # confidence sensitivity: > 0 waits for precision, < 0 explores
    "gamma": ("id", 0.0, 2.5),             # risk aversion over outcome variability
    "nu": ("logit", 0.0, 1.5),             # memory of the outcome-variability estimate
    "sigma_w": ("exp", np.log(0.3), 1.0),  # drift of the action values
    "sigma_r": ("exp", np.log(1.0), 1.0),  # outcome noise the learner assumes
    "h": ("logit", -2.5, 1.0),             # believed hazard of a change
    "sigma_jump": ("exp", np.log(5.0), 1.0),
    "k": ("logit", -1.0, 1.5),             # baseline gain of the cost belief
    "lam": ("id", 0.0, 2.0),               # weight on the change belief
    "eta": ("logit", 0.0, 1.5),            # aspiration adaptation rate
    "g": ("id", 0.0, 2.0),                 # inertia: stickiness after keeping output last period
    "cV": ("id", 0.0, 2.0), "cN": ("id", 0.0, 2.0), "cR": ("id", 0.0, 2.0),
    "lapse": ("logit", -3.0, 1.5),
    "phi": ("phi", 0.0, 2.0), "sd": ("exp", np.log(15.0), 1.5),
}
_COND = ("cV", "cN", "cR")
_TAIL = ("lapse", "phi", "sd")
MODEL_PARAMS: Dict[str, Tuple[str, ...]] = {
    "advantage": ("a0", "beta", "alpha") + _COND + _TAIL,
    "confidence": ("a0", "beta", "omega", "sigma_w", "sigma_r") + _COND + _TAIL,
    "changebayes": ("a0", "beta", "lam", "h", "sigma_jump", "k") + _COND + _TAIL,
    "aspiration": ("a0", "beta", "eta") + _COND + _TAIL,
    "risk": ("a0", "beta", "omega", "gamma", "nu", "sigma_w", "sigma_r") + _COND + _TAIL,
    # Not one of the five the brief asks for: the competitor they have to be told apart from. Inertia predicts
    # "adapts less after keeping" and confidence sensitivity predicts "adapts less when uncertain", and early in
    # a block those two look nearly the same. It is in the family so the comparison is like for like, on the
    # same observables and the same magnitude model.
    "inertia_learn": ("a0", "beta", "alpha", "g") + _COND + _TAIL,
}

# Assumed population for SYNTHETIC work: design assumptions for checking the pipeline, not estimates of people.
DEFAULT_POPULATION = dict(
    a0=-0.4, a0_sd=0.7, beta=1.2, beta_sd=0.5, alpha=0.35, alpha_sd=0.6,
    omega=0.8, omega_sd=0.5, gamma=0.6, gamma_sd=0.5, nu=0.3, nu_sd=0.6, g=1.0, g_sd=0.5,
    sigma_w=0.30, sigma_w_sd=0.4, sigma_r=1.0, sigma_r_sd=0.4,
    h=0.08, h_sd=0.5, sigma_jump=6.0, sigma_jump_sd=0.4, k=0.30, k_sd=0.5, lam=1.5, lam_sd=0.6,
    eta=0.40, eta_sd=0.6,
    cV=0.5, cN=-0.3, cR=0.5, c_sd=0.3, lapse=0.03, lapse_sd=0.5, phi=0.6, phi_sd=0.4, sd=15.0, sd_sd=0.3,
)


def _to_nat(name: str, u: float):
    t = PARAMS[name][0]
    if t == "id":
        return u
    if t == "exp":
        return np.exp(u)
    if t == "logit":
        return expit(u)
    return 1.5 * expit(u)


def _to_unc(name: str, v: float):
    t = PARAMS[name][0]
    if t == "id":
        return v
    if t == "exp":
        return np.log(v)
    if t == "logit":
        return np.log(v / (1 - v))
    p = v / 1.5
    return np.log(p / (1 - p))


def param_names(model: str) -> Tuple[str, ...]:
    return MODEL_PARAMS[model]


# ================================================================================================ data
OBSERVED_COLUMNS = ("cost_estimate", "cost", "price", "profit", "q_prev", "q", "period", "block_index",
                    "condition")
FORBIDDEN_PREFIX = "shadow:"       # the researcher's reference designs: never an input to these models


@dataclass
class LData:
    """One participant, in decision order, holding ONLY what was on the screen.

    `c_hat` is this period's cost estimate, shown before the decision. `c_last`, `profit_last` and `price_last`
    are the previous period's realized values, shown as feedback. Everything is lagged WITHIN a block, so the
    first decision of a block has no feedback and the models start from their priors.
    """
    participant: str
    d: np.ndarray                # adjusted (0/1)
    dq: np.ndarray               # observed change
    c_hat: np.ndarray            # cost estimate shown for this period
    c_last: np.ndarray           # cost shown as last period's outcome (NaN on the first decision of a block)
    profit_last: np.ndarray      # own profit last period (NaN on the first decision of a block)
    cond: np.ndarray
    block: np.ndarray
    first: np.ndarray            # 1.0 where a new block starts: the state is reset here
    history: np.ndarray          # sampling-history factor, periods observed before the block was scored
    horizon: np.ndarray          # announced periods remaining (opportunity for further learning)
    scored: np.ndarray           # 1 where the participant chose; 0 in the observation-only lead-in
    # The lead-in rows are KEPT, not dropped. The participant watched those outcomes, so the model has to watch
    # them too: the state is advanced through them and they contribute no choice to the likelihood. Dropping
    # them makes the fitted learner start every block from its prior, which silently erases the sampling-history
    # manipulation while leaving the generated data unchanged - the whole factor then measures nothing.

    @property
    def kept(self) -> np.ndarray:
        """1 where the previous decision kept output; 0 at the first decision of a block."""
        prev = np.concatenate([[0.0], self.d[:-1]])
        return np.where(self.first > 0, 0.0, 1.0 - prev)

    @property
    def u(self) -> np.ndarray:
        """Cost surprise c_hat_t - c_{t-1}: zero where no previous cost was shown."""
        return np.nan_to_num(self.c_hat - self.c_last)

    def subset(self, mask) -> "LData":
        return LData(self.participant, *(np.asarray(v)[mask] for v in
                                         (self.d, self.dq, self.c_hat, self.c_last, self.profit_last, self.cond,
                                          self.block, self.first, self.history, self.horizon, self.scored)))


def prepare(df: pd.DataFrame) -> List[LData]:
    """Per-participant arrays from recorded decisions, lagged within blocks.

    Reads none of the `shadow:` columns. A test asserts that by deleting them before calling this.
    """
    d0 = df[~df["practice"].astype(bool)] if "practice" in df else df
    out = []
    for pid, g in d0.sort_values(["participant", "block_index", "period"]).groupby("participant", sort=True):
        g = g.reset_index(drop=True)
        blocks = g["block_index"].to_numpy(int)
        first = np.concatenate([[1.0], (np.diff(blocks) != 0).astype(float)])
        q_prev = g["q_prev"].to_numpy(float)
        dq = g["q"].to_numpy(float) - q_prev
        lag = lambda col: np.where(first > 0, np.nan, np.concatenate([[np.nan], g[col].to_numpy(float)[:-1]]))
        out.append(LData(
            participant=str(pid), d=(dq != 0).astype(float), dq=dq,
            c_hat=g["cost_estimate"].to_numpy(float), c_last=lag("cost"), profit_last=lag("profit"),
            cond=g["condition"].map(COND_INDEX).to_numpy(int), block=blocks, first=first,
            history=(g["history_periods"].to_numpy(float) if "history_periods" in g else np.zeros(len(g))),
            horizon=(g["horizon_left"].to_numpy(float) if "horizon_left" in g else np.zeros(len(g))),
            scored=(g["scored"].to_numpy(float) if "scored" in g else np.ones(len(g)))))
    return out


# ================================================================================================ the state
def _normal_pdf(x: float, mu: float, sd: float) -> float:
    z = (x - mu) / max(sd, 1e-9)
    return float(np.exp(-0.5 * z * z) / (max(sd, 1e-9) * np.sqrt(2.0 * np.pi)))


class Learner:
    """The learning state of one model, advanced one decision at a time.

    The likelihood and the simulator both drive this object, so a change to a mechanism cannot be made in one
    and forgotten in the other - which is the usual way a recovery study comes to recover a model nobody fitted.

    Order within a decision is fixed and matters:
        signals()                 read the state BEFORE this decision (what the choice rule may use)
        observe(profit, adjusted) fold in the feedback from the decision just resolved
    `reset()` is called at every block boundary.
    """
    __slots__ = ("model", "p", "V", "v", "wvar", "wmean", "seen", "mu", "s2", "A", "started", "_last_change",
                 "_last_surprise")

    def __init__(self, model: str, p: Dict[str, float]):
        self.model = model
        self.p = p
        self.reset()

    def reset(self) -> None:
        self.V = np.zeros(2)
        self.v = np.ones(2)
        self.wvar = 0.0
        self.wmean = 0.0
        self.seen = 0
        self.mu = 0.0
        self.s2 = 1.0
        self.A = 0.0
        self.started = False
        self._last_change = 0.0
        self._last_surprise = 0.0

    # ------------------------------------------------------------------ reading the state
    def signals(self, c_hat: float, profit_last: float) -> Dict[str, float]:
        """What the choice rule may use at this decision. For `changebayes` the cost estimate is shown now, so
        the belief is updated with it here, before the choice - that is the one update that precedes the
        decision, because the participant reads the number before choosing."""
        p = self.p
        out = {"m": 0.0, "uncertainty": 0.0, "surprise": 0.0, "change": 0.0, "shortfall": 0.0,
               "outcome_sd": 0.0}
        if self.model in ("advantage", "confidence", "risk", "inertia_learn"):
            out["m"] = float(self.V[1] - self.V[0])
            out["uncertainty"] = float(np.sqrt(self.v[0] + self.v[1]))
            out["outcome_sd"] = float(np.sqrt(max(self.wvar, 0.0)))
        elif self.model == "changebayes":
            sj2 = float(p["sigma_jump"]) ** 2
            if not self.started:
                self.mu, self.s2, self.started = float(c_hat), sj2, True
            c = float(c_hat)
            out["surprise"] = abs(c - self.mu) / COST_SCALE
            d_no = _normal_pdf(c, self.mu, np.sqrt(self.s2 + 1.0))
            d_ch = _normal_pdf(c, self.mu, np.sqrt(self.s2 + 1.0 + sj2))
            h = float(p["h"])
            post = h * d_ch / max(h * d_ch + (1.0 - h) * d_no, 1e-300)
            out["change"] = post
            gain = float(p["k"]) + post * (1.0 - float(p["k"]))
            self.mu = float(np.clip(self.mu + gain * (c - self.mu), -1e6, 1e6))
            self.s2 = float(np.clip((1.0 - gain) * self.s2 + post * sj2, 1e-12, 1e12))
        elif self.model == "aspiration":
            out["shortfall"] = ((self.A - float(profit_last)) / PROFIT_SCALE
                                if (self.started and np.isfinite(profit_last)) else 0.0)
        return out

    # ------------------------------------------------------------------ folding in feedback
    def observe(self, profit: float, adjusted: int) -> None:
        """Feedback from the decision that has just resolved. `adjusted` is what the participant did.

        Every state variable is clipped. During model recovery the optimiser evaluates parameters no
        participant would have, and an unclipped variance can overflow to inf and then to NaN, which reaches
        the Fortran optimiser as a non-finite objective.
        """
        if not np.isfinite(profit):
            return
        p = self.p
        r = float(profit) / PROFIT_SCALE
        if self.model in ("advantage", "confidence", "risk", "inertia_learn"):
            a = int(adjusted)
            if self.model in ("advantage", "inertia_learn"):
                alpha = float(p["alpha"])
                self.V[a] = self.V[a] + alpha * (r - self.V[a])
            else:
                sw2 = float(p["sigma_w"]) ** 2
                sr2 = max(float(p["sigma_r"]) ** 2, 1e-9)
                self.v = self.v + sw2                      # both actions drift: not acting costs certainty too
                g = self.v[a] / (self.v[a] + sr2)
                self.V[a] = self.V[a] + g * (r - self.V[a])
                self.v[a] = (1.0 - g) * self.v[a]
            self.V = np.clip(self.V, -1e6, 1e6)
            self.v = np.clip(self.v, 1e-12, 1e6)
            self.seen += 1
            nu = float(p.get("nu", 0.0))
            if self.seen == 1:
                self.wmean, self.wvar = r, 0.0
            else:
                self.wmean = float(np.clip(self.wmean + nu * (r - self.wmean), -1e6, 1e6))
                self.wvar = float(np.clip(self.wvar + nu * ((r - self.wmean) ** 2 - self.wvar), 0.0, 1e12))
        elif self.model == "aspiration":
            if not self.started:
                self.A, self.started = r * PROFIT_SCALE, True
            else:
                self.A = float(np.clip(self.A + float(p["eta"]) * (float(profit) - self.A), -1e9, 1e9))


def trial_signals(model: str, p: Dict[str, float], x: LData) -> Dict[str, np.ndarray]:
    """Run the state across a participant's decisions and collect the regressors, in decision order.

    The feedback of decision t-1 is folded in before decision t, and the state is reset at every block boundary,
    so nothing learned in one block reaches another.
    """
    n = len(x.d)
    keys = ("m", "uncertainty", "surprise", "change", "shortfall", "outcome_sd")
    out = {k: np.zeros(n) for k in keys}
    learner = Learner(model, p)
    for t in range(n):
        if x.first[t] and not CARRY_STATE:
            learner.reset()
        elif t > 0:
            learner.observe(x.profit_last[t], int(x.d[t - 1]))
        s = learner.signals(x.c_hat[t], x.profit_last[t])
        for k in keys:
            out[k][t] = s[k]
    return out


class _WithKept:
    """An LData view with `kept` supplied explicitly: during simulation the previous action is known directly,
    while in the likelihood it is derived from the recorded choices."""
    __slots__ = ("_x", "kept")

    def __init__(self, x: LData, kept: np.ndarray):
        self._x, self.kept = x, kept

    def __getattr__(self, name):
        return getattr(self._x, name)


def adapt_logit(model: str, p: Dict[str, float], x: LData, sig: Dict[str, np.ndarray]) -> np.ndarray:
    """The linear predictor of the adaptation decision, per model."""
    z = p["a0"] + np.array([0.0, p["cV"], p["cN"], p["cR"]])[x.cond]
    if model == "advantage":
        z = z + p["beta"] * sig["m"]
    elif model == "inertia_learn":
        z = z + p["beta"] * sig["m"] - p["g"] * x.kept
    elif model == "confidence":
        z = z + p["beta"] * sig["m"] - p["omega"] * sig["uncertainty"]
    elif model == "risk":
        z = (z + p["beta"] * sig["m"] - p["omega"] * sig["uncertainty"]
             - p["gamma"] * sig["outcome_sd"])
    elif model == "changebayes":
        z = z + p["beta"] * sig["surprise"] + p["lam"] * sig["change"]
    elif model == "aspiration":
        z = z + p["beta"] * sig["shortfall"]
    else:
        raise ValueError(model)
    return z


# ================================================================================================ likelihood
def _nat(model: str, u: np.ndarray) -> Dict[str, float]:
    return {n: float(_to_nat(n, u[i])) for i, n in enumerate(param_names(model))}


def components(model: str, p: Dict[str, float], x: LData):
    """Per decision: P(adjust), log P(observed change | adjust), expected change.

    The magnitude model is shared by all five, so model comparison is a comparison of adaptation decisions and
    not of how large a change somebody makes once they have decided to make one.
    """
    sig = trial_signals(model, p, x)
    core = expit(adapt_logit(model, p, x, sig))
    eps = p["lapse"]
    p_adj = np.clip(eps / 2 + (1 - eps) * core, 1e-9, 1 - 1e-9)

    mu, sd = -p["phi"] * x.u, max(p["sd"], 1e-6)
    D = x.dq
    gauss = ndtr((D + 0.5 - mu) / sd) - ndtr((D - 0.5 - mu) / sd)
    m0 = ndtr((0.5 - mu) / sd) - ndtr((-0.5 - mu) / sd)
    body = gauss / np.maximum(1 - m0, 1e-12)
    p_mag = eps / LAPSE_RANGE + (1 - eps) * body
    return p_adj, np.log(np.maximum(p_mag, 1e-300)), (1 - eps) * mu


def loglik(model: str, u: np.ndarray, x: LData) -> float:
    """Summed over SCORED decisions only. Observation-only periods advance the state and are not choices."""
    p_adj, lmag, _ = components(model, _nat(model, u), x)
    w = x.scored
    return float(np.sum(w * (x.d * np.log(p_adj) + (1 - x.d) * np.log(1 - p_adj))) + np.sum(w * x.d * lmag))


# Box constraints on the unconstrained scale. L-BFGS-B is Fortran: handed a non-finite objective or allowed to
# wander far enough that a state variable overflows, it does not raise - it segfaults the interpreter. That
# happened once here, during model recovery, where fitting a model to data generated by a different one sends
# the search somewhere no sensible parameter lives. Bounding the box and refusing to return a non-finite value
# are the two things that make that impossible rather than unlikely.
_BOUNDS = {"id": (-10.0, 10.0), "exp": (-8.0, 6.0), "logit": (-10.0, 10.0), "phi": (-10.0, 10.0)}


def bounds_for(model: str) -> List[Tuple[float, float]]:
    return [_BOUNDS[PARAMS[n][0]] for n in param_names(model)]


def _weak_prior(model: str):
    names = param_names(model)
    return (np.array([PARAMS[n][1] for n in names], float),
            np.array([PARAMS[n][2] for n in names], float))


def fit_one(model: str, x: LData, prior_mean: np.ndarray, prior_sd: np.ndarray) -> np.ndarray:
    bounds = bounds_for(model)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    start = np.clip(prior_mean, lo, hi)

    def nlp(u):
        if not np.all(np.isfinite(u)):
            return 1e12
        val = -loglik(model, u, x) + 0.5 * float(np.sum(((u - prior_mean) / prior_sd) ** 2))
        return float(val) if np.isfinite(val) else 1e12

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r = minimize(nlp, start.copy(), method="L-BFGS-B", bounds=bounds, options=dict(maxiter=250))
    u = np.asarray(r.x, float)
    return np.clip(u, lo, hi) if np.all(np.isfinite(u)) else start


def _group_prior(model: str, est: Dict[str, np.ndarray], exclude: Optional[str] = None):
    names = param_names(model)
    prior = {}
    for i, n in enumerate(names):
        v = np.array([u[i] for pid, u in est.items() if pid != exclude], float)
        m0, s0 = PARAMS[n][1], PARAMS[n][2]
        prior[n] = ((float(v.mean()), float(np.clip(v.std(ddof=1), 0.15, s0))) if len(v) >= 3 else (m0, s0))
    return prior


def _prior_vec(model: str, prior: Dict[str, Tuple[float, float]]):
    names = param_names(model)
    return (np.array([prior[n][0] for n in names], float), np.array([prior[n][1] for n in names], float))


@dataclass
class Fit:
    model: str
    params: pd.DataFrame
    unconstrained: Dict[str, np.ndarray]
    prior: Dict[str, Tuple[float, float]]
    loglik: float
    n_params: int
    n_obs: int

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.n_params * np.log(max(self.n_obs, 1))


def fit_model(model: str, data: Sequence[LData]) -> Fit:
    """Two-pass empirical Bayes with partial pooling across participants, as in `human_models`.

    Participant-level dependence is kept: every parameter is estimated per participant, and the population
    enters only through the prior. Pooling decisions across participants into one regression would treat 120
    decisions from one person as 120 independent observations, which they are not.
    """
    first = {x.participant: fit_one(model, x, *_weak_prior(model)) for x in data}
    prior = _group_prior(model, first)
    est = {x.participant: fit_one(model, x, *_prior_vec(model, prior)) for x in data}
    rows, ll, k, n = [], 0.0, 0, 0
    for x in data:
        u = est[x.participant]
        ll += loglik(model, u, x)
        k += len(u)
        n += int(x.scored.sum())
        rows.append(dict(participant=x.participant, **_nat(model, u)))
    return Fit(model, pd.DataFrame(rows), est, prior, ll, k, n)


def heldout_scores(model: str, data: Sequence[LData], full: Optional[Fit] = None) -> pd.DataFrame:
    """Leave-one-block-out per participant, with a prior computed without that participant.

    Held out by BLOCK rather than by decision, because the decisions of a block share a learning trajectory and
    leaving out single trials would let the fit see the state that produced them.
    """
    full = full or fit_model(model, data)
    rows = []
    for x in data:
        prior = _group_prior(model, full.unconstrained, exclude=x.participant)
        pm, ps = _prior_vec(model, prior)
        for b in np.unique(x.block):
            tr, te = x.block != b, x.block == b
            if te.sum() == 0 or tr.sum() < 10 or x.scored[te].sum() == 0:
                continue
            u = fit_one(model, x.subset(tr), pm, ps)
            xt = x.subset(te)
            p_adj, lmag, e_mag = components(model, _nat(model, u), xt)
            w = xt.scored.astype(bool)
            if not w.any():
                continue
            log_score = float(np.sum(xt.d[w] * np.log(p_adj[w]) + (1 - xt.d[w]) * np.log(1 - p_adj[w]))
                              + np.sum(xt.d[w] * lmag[w]))
            rows.append(dict(participant=x.participant, block=int(b), n=int(w.sum()),
                             log_score=log_score,
                             brier=float(np.mean((p_adj[w] - xt.d[w]) ** 2)),
                             rmse=float(np.sqrt(np.mean((p_adj[w] * e_mag[w] - xt.dq[w]) ** 2)))))
    return pd.DataFrame(rows)


def compare_models(data: Sequence[LData], models: Sequence[str] = MODELS) -> pd.DataFrame:
    rows = []
    for m in models:
        fit = fit_model(m, data)
        h = heldout_scores(m, data, fit)
        rows.append(dict(model=m, label=MODEL_LABELS[m], loglik=fit.loglik, bic=fit.bic,
                         n_params=fit.n_params, n_obs=fit.n_obs,
                         heldout_log_score=float(h["log_score"].sum()),
                         heldout_brier=float(h["brier"].mean())))
    return pd.DataFrame(rows).sort_values("heldout_log_score", ascending=False).reset_index(drop=True)


# ================================================================================================ synthetic agents
def draw_participant(model: str, pop: Dict[str, float], rng: np.random.Generator) -> Dict[str, float]:
    """Natural-scale parameters of one SYNTHETIC participant. Never an estimate of a person."""
    th = {}
    for n in param_names(model):
        mean = pop[n]
        sd = pop.get(f"{n}_sd", pop.get("c_sd", 0.3))
        if PARAMS[n][0] == "id":
            th[n] = float(mean + sd * rng.standard_normal())
        else:
            th[n] = float(_to_nat(n, _to_unc(n, mean) + sd * rng.standard_normal()))
    return th


def simulate_decision(model: str, theta: Dict[str, float], learner: "Learner", cond: str, c_hat: float,
                      c_last: float, profit_last: float, q_prev: float,
                      rng: np.random.Generator, kept_last: float = 0.0) -> Tuple[float, bool]:
    """One decision of a SYNTHETIC participant, driving the SAME Learner and the SAME choice rule.

    The linear predictor comes from `adapt_logit`, the function the likelihood uses, rather than from a second
    copy of the arithmetic. A duplicated choice rule is how a recovery study comes to recover a model nobody
    fitted: the generator drifts from the estimator and every confusion matrix after that is measuring the gap
    between two versions of the same name.
    """
    sig = learner.signals(c_hat, profit_last)
    one = LData(participant="sim", d=np.zeros(1), dq=np.zeros(1),
                c_hat=np.array([c_hat], float), c_last=np.array([c_last], float),
                profit_last=np.array([profit_last], float),
                cond=np.array([COND_INDEX[cond]], int), block=np.zeros(1, int), first=np.ones(1),
                history=np.zeros(1), horizon=np.zeros(1), scored=np.ones(1))
    one_kept = np.array([float(kept_last)])
    z = float(adapt_logit(model, theta, _WithKept(one, one_kept),
                          {k: np.array([v], float) for k, v in sig.items()})[0])

    if rng.random() < theta["lapse"]:
        if rng.random() < 0.5:
            return float(max(5.0, q_prev + rng.integers(-400, 401))), True
        return q_prev, False
    if rng.random() >= expit(z):
        return q_prev, False
    u = (c_hat - c_last) if np.isfinite(c_last) else 0.0
    for _ in range(50):
        delta = float(np.rint(-theta["phi"] * u + theta["sd"] * rng.standard_normal()))
        if delta != 0:
            return float(max(5.0, q_prev + delta)), True
    return float(max(5.0, q_prev + 1.0)), True


# ================================================================================================ the protocol
# Four manipulations, separated on purpose. The first two already exist in the registered protocol; the last two
# are new and are the reason the confidence parameter is identifiable at all.
#
#   OUTCOME VARIABILITY      condition V (delta 8 -> 24). The world pays more erratically. Moves the spread of
#                            realized profit; does NOT move how precisely the cost is reported.
#   OBSERVATION PRECISION    condition N (noise 2 -> 10). The cost estimate is noisier. Moves how precisely the
#                            participant can see; does NOT move the spread of the underlying world.
#   SAMPLING HISTORY         NEW. A block begins with `history` FORCED-SAMPLING periods: the output to submit is
#                            prescribed on screen and the outcome is shown, alternating between keeping and
#                            changing so that BOTH actions are sampled. Evidence accumulates without the
#                            participant choosing anything, so estimation uncertainty at the first scored
#                            decision is lower while outcome variability and observation precision are
#                            untouched.
#                            A purely passive lead-in - watch while output is held - does NOT do this. It only
#                            ever samples "keep", and in an action-value learner the uncertainty about adjusting
#                            falls only when adjusting is tried, so a passive phase leaves the advantage no
#                            better estimated than before. That version was built first and measured: the
#                            uncertainty at the first scored decision moved from 1.414 to 1.321 and the mean
#                            over the block did not move at all.
#   FURTHER LEARNING         NEW. The block is announced as short or long. The value of adapting now depends on
#                            how many periods remain to benefit from it, without changing anything the
#                            participant can currently see.
#
# Why the last two matter: confidence sensitivity (omega), inertia, risk preference (gamma) and a slow learning
# rate all predict "adapts less" in a plain design, and the registered conditions cannot separate them, because
# V and N both change what is on the screen at the moment of choice. Sampling history changes the PRECISION of
# the estimate while leaving the screen identical; a horizon change alters the VALUE of adapting while leaving
# both the screen and the precision identical. `identifiability()` below measures how much this helps rather
# than assuming it.
HISTORY_LEVELS = (0, 10)        # forced-sampling periods before the scored decisions of a block
FORCED_STEP = 25.0              # the prescribed change on a forced "adjust" period
FORCED_PLACEMENT = "split"      # "lead" puts every forced period at the start; "split" puts half of them in a
                                # second bout in the middle of the block. Placement matters more than it looks:
                                # with every forced period at the front, the uncertainty signal is a monotone
                                # decay in trial index, which is nearly collinear with the intercept, and the
                                # confidence parameter cannot be told from a participant who simply adapts less
                                # often. A second bout makes the signal drop twice and breaks that.
HORIZON_LEVELS = (12, 32)       # scored periods in a block, announced to the participant in advance
FACTOR_CELLS: Tuple[Tuple[int, int], ...] = tuple((h, k) for h in HISTORY_LEVELS for k in HORIZON_LEVELS)


def factor_cell(arm_index: int, position: int) -> Tuple[int, int]:
    """The (history, horizon) cell of the block at `position` for a participant in arm `arm_index`.

    The condition order is the Williams square, which for order 4 is built on the cyclic group and therefore has
    no orthogonal mate - there is no second square that crosses conditions with factor cells within a single
    participant. Rotating by the ARM index instead gives a design in which every condition meets every factor
    cell exactly once within each arm, and four times overall. A test checks both counts rather than trusting
    this paragraph.
    """
    return FACTOR_CELLS[(position + arm_index) % len(FACTOR_CELLS)]


@dataclass(frozen=True)
class LearnPlan:
    """Frozen specification of the extended protocol and the simulation studies."""
    version: str = "1.0"
    seed: int = 20261013
    models: Tuple[str, ...] = MODELS
    n_participants: int = 32
    history_levels: Tuple[int, ...] = HISTORY_LEVELS
    horizon_levels: Tuple[int, ...] = HORIZON_LEVELS
    recovery_n: int = 32
    model_recovery_n: int = 10
    power_sizes: Tuple[int, ...] = (16, 32, 48, 64)
    power_reps: int = 12
    alpha: float = 0.05

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "code": learning_code_digest(),
                           "params": {k: list(v) for k, v in PARAMS.items()},
                           "model_params": {k: list(v) for k, v in MODEL_PARAMS.items()},
                           "population": DEFAULT_POPULATION, "factor_cells": [list(c) for c in FACTOR_CELLS],
                           "carry_state": CARRY_STATE,
                           "scales": {"profit": PROFIT_SCALE, "cost": COST_SCALE}},
                          sort_keys=True, indent=1, default=str)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def learning_code_digest() -> str:
    """Digest of the code that decides what happens, so that changing a mechanism invalidates the plan.

    A plan hash over parameters alone would still match results produced by different update equations.
    """
    import inspect
    import sys as _sys
    mod = _sys.modules[__name__]
    funcs = (mod.Learner.signals, mod.Learner.observe, mod.Learner.reset, mod.trial_signals, mod.adapt_logit,
             mod.components, mod.simulate_decision, mod.prepare, mod.factor_cell)
    return hashlib.sha256("".join(inspect.getsource(f) for f in funcs).encode()).hexdigest()[:16]


# ================================================================================================ synthetic pilot
def forced_periods(hist: int, horizon: int, placement: str = FORCED_PLACEMENT) -> set:
    """Which step indices of a block are forced-sampling periods."""
    if hist <= 0:
        return set()
    if placement == "lead":
        return set(range(hist))
    half = hist // 2
    mid = half + horizon // 2
    return set(range(half)) | set(range(mid, mid + (hist - half)))


def synthetic_pilot(n: int, model: str = "confidence", seed: int = 0,
                    population: Optional[Dict[str, float]] = None,
                    use_new_factors: bool = True, placement: str = FORCED_PLACEMENT,
                    randomize_paths: bool = False,
                    plan: Optional[LearnPlan] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """SYNTHETIC participants generated by one of these models in the real interactive market.

    For design checks, recovery and power planning only: never evidence about people. Every row carries
    synthetic=True, and `use_new_factors=False` reproduces the registered protocol (no observation lead-in, one
    block length) so that the value of the new manipulations can be measured rather than asserted.
    """
    from dataclasses import replace as _replace
    from .experiment import PLAN, ARMS, allocation_schedule, block_market, decision_row

    rng = np.random.default_rng(seed)
    pop = {**DEFAULT_POPULATION, **(population or {})}
    frames, truth = [], []
    for a in allocation_schedule(n, PLAN):
        pid = f"synthetic-{a.slot:03d}"
        theta = draw_participant(model, pop, rng)
        truth.append(dict(participant=pid, model=model, **{f"true:{k}": v for k, v in theta.items()}))
        arm_index = ARMS.index((a.aid, a.elicit))
        # The registered protocol gives every participant the SAME four cost paths. That makes participants
        # non-independent: a path that happens to favour waiting shifts everybody in the sample the same way,
        # and no participant-level test can remove it. `randomize_paths` draws a fresh set per participant,
        # which is the design fix; `power_curve` measures what it is worth.
        seeds = (tuple(int(v) for v in rng.integers(1000, 999999, PLAN.n_paths))
                 if randomize_paths else PLAN.path_seeds)
        for bi, cond in enumerate(a.order(PLAN)):
            hist, horizon = factor_cell(arm_index, bi) if use_new_factors else (0, PLAN.periods_per_block)
            path = a.path(cond, PLAN)
            bplan = _replace(PLAN, periods_per_block=hist + horizon, path_seeds=seeds)
            m = block_market(cond, path, bplan, shadows=(PLAN.signal_design, PLAN.aid_design))
            forced = forced_periods(hist, horizon, placement)
            learner = Learner(model, theta)        # reset at the block boundary, by construction
            profit_last, c_last, kept_last = np.nan, np.nan, 0.0
            step = 0
            while not m.done:
                info = m.info()
                q_prev = float(info["own_q"])
                c_hat = float(info["cost_estimate"])
                watching = step in forced
                if watching:
                    # Forced sampling: the move is prescribed, and the two actions alternate so that both are
                    # sampled. Keeping only would leave the value of adjusting as uncertain as it started.
                    adj = bool(step % 2)
                    sign = 1.0 if (step // 2) % 2 == 0 else -1.0
                    q = float(max(5.0, q_prev + sign * FORCED_STEP)) if adj else q_prev
                else:
                    q, adj = simulate_decision(model, theta, learner, cond, c_hat, c_last, profit_last,
                                               q_prev, rng, kept_last)
                row = m.step(q)
                learner.observe(row["profit"], int(adj))
                profit_last, c_last = float(row["profit"]), float(row["cost"])
                kept_last = 0.0 if adj else 1.0
                step += 1
                extra = dict(participant=pid, synthetic=True, scored=not watching,
                             history_periods=float(hist), horizon_total=float(horizon),
                             forced=bool(watching),
                             horizon_left=float(max(0, hist + horizon - step)),
                             comprehension_passed=True, comprehension_attempts=1)
                frames.append(decision_row(m, row, a, bi, cond, path, extra, PLAN))
    # The observation-only rows are returned too: see the note on LData.scored.
    return pd.DataFrame(frames).reset_index(drop=True), pd.DataFrame(truth)


# ================================================================================================ recovery
def parameter_recovery(model: str = "confidence", n: int = 32, seed: int = 1,
                       use_new_factors: bool = True,
                       placement: str = FORCED_PLACEMENT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the generating model to SYNTHETIC participants and compare estimates with the truth.

    Reported per parameter: correlation, bias and RMSE across participants. A parameter whose correlation is
    near zero is not estimable by this design, whatever its point estimate looks like.
    """
    df, truth = synthetic_pilot(n, model=model, seed=seed, use_new_factors=use_new_factors,
                                placement=placement)
    fit = fit_model(model, prepare(df))
    m = fit.params.merge(truth, on="participant")
    rows = []
    for name in param_names(model):
        if f"true:{name}" not in m:
            continue
        t = m[f"true:{name}"].to_numpy(float)
        e = m[name].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(e)
        t, e = t[ok], e[ok]
        rows.append(dict(parameter=name, n=len(t),
                         correlation=float(np.corrcoef(t, e)[0, 1]) if len(t) > 2 and t.std() > 0 and e.std() > 0
                         else np.nan,
                         bias=float(np.mean(e - t)), rmse=float(np.sqrt(np.mean((e - t) ** 2))),
                         true_mean=float(t.mean()), est_mean=float(e.mean())))
    return pd.DataFrame(rows), m


def model_recovery(n: int = 12, seed: int = 2, models: Sequence[str] = MODELS,
                   criterion: str = "heldout",
                   progress: Optional[Callable[[float, str], None]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate from each model, fit all of them, and record which one each participant is assigned to.

    This is by far the slowest stage - every generating model is fitted by every candidate, and the held-out
    criterion refits once per participant per block - so it reports progress. Without it a stall here is
    indistinguishable from a crash, which is exactly what happened the first time it was run.
    """
    note = progress or (lambda f, m: None)
    rows = []
    total = max(1, len(models) ** 2)
    done = 0
    for j, gen in enumerate(models):
        df, _ = synthetic_pilot(n, model=gen, seed=seed * 100 + j)
        data = prepare(df)
        scores = {}
        for m in models:
            note(done / total, f"model recovery: data from {gen}, fitting {m}")
            done += 1
            fit = fit_model(m, data)
            if criterion == "heldout":
                scores[m] = heldout_scores(m, data, fit).groupby("participant")["log_score"].sum()
            else:
                scores[m] = pd.Series({x.participant: loglik(m, fit.unconstrained[x.participant], x)
                                       - 0.5 * len(fit.unconstrained[x.participant]) * np.log(len(x.d))
                                       for x in data})
        tab = pd.DataFrame(scores)
        for pid, r in tab.iterrows():
            rows.append(dict(generating=gen, participant=pid, selected=r.idxmax(),
                             **{f"score:{m}": r[m] for m in models}))
    per = pd.DataFrame(rows)
    conf = pd.crosstab(per["generating"], per["selected"], normalize="index").reindex(
        index=list(models), columns=list(models), fill_value=0.0)
    return conf, per


# ================================================================================================ identifiability
# The question the brief puts, and the reason the two new manipulations exist: can this design tell confidence
# sensitivity apart from inertia, from risk preference, and from simply learning slowly? All four predict "adapts
# less", and in the registered design all four are read off the same decisions. Four probes, each answering a
# different part of it, and none of them assuming the answer.
CONFOUNDS = ("inertia_learn", "risk", "advantage")
IDENTIFIABILITY_CLAIMS = {
    "I1": "Confidence sensitivity is recoverable: omega correlates with its true value across participants.",
    "I2": "Confidence sensitivity is not inertia: data generated with omega > 0 and no inertia is assigned to "
          "the confidence model rather than to the inertia model.",
    "I3": "Confidence sensitivity is not risk preference: in the risk model, which carries both, the estimation "
          "errors of omega and gamma are not collinear.",
    "I4": "Confidence sensitivity is not slow learning: a confident waiter and a slow learner, matched on how "
          "often they adapt, are distinguished.",
    "I5": "The new manipulations help: omega recovers better with the sampling-history and horizon factors "
          "than without them.",
}


def _fit_omega(df: pd.DataFrame, model: str = "confidence") -> pd.DataFrame:
    return fit_model(model, prepare(df)).params


def identifiability(n: int = 32, seed: int = 11, reps: int = 1) -> Dict[str, pd.DataFrame]:
    """Run the five probes. Returns one frame per claim plus a verdict table."""
    out: Dict[str, pd.DataFrame] = {}

    # --- I1 and I5: recovery of omega, with and without the new factors ------------------------------------
    rows = []
    for use_new in (True, False):
        for r in range(reps):
            rec, merged = parameter_recovery("confidence", n=n, seed=seed + 1000 * r, use_new_factors=use_new)
            om = rec[rec["parameter"] == "omega"].iloc[0]
            rows.append(dict(factors="new" if use_new else "registered", rep=r,
                             correlation=om["correlation"], rmse=om["rmse"], bias=om["bias"],
                             n=int(om["n"])))
    out["omega_recovery"] = pd.DataFrame(rows)

    # --- I2: generated with confidence and no inertia; which model is selected? -----------------------------
    pop_conf = {**DEFAULT_POPULATION, "omega": 1.2, "omega_sd": 0.3}
    df, _ = synthetic_pilot(n, model="confidence", seed=seed + 7, population=pop_conf)
    data = prepare(df)
    scores = {}
    for m in ("confidence", "inertia_learn", "advantage", "risk"):
        fit = fit_model(m, data)
        scores[m] = heldout_scores(m, data, fit).groupby("participant")["log_score"].sum()
    tab = pd.DataFrame(scores)
    tab["selected"] = tab.idxmax(axis=1)
    out["confidence_vs_inertia"] = tab.reset_index()

    # --- I3: do omega and gamma trade off inside the model that carries both? -------------------------------
    rec_r, merged_r = parameter_recovery("risk", n=n, seed=seed + 23)
    err = pd.DataFrame({"omega": merged_r["omega"] - merged_r["true:omega"],
                        "gamma": merged_r["gamma"] - merged_r["true:gamma"]}).dropna()
    out["risk_tradeoff"] = pd.DataFrame([dict(
        n=len(err),
        error_correlation=float(np.corrcoef(err["omega"], err["gamma"])[0, 1]) if len(err) > 2 else np.nan,
        omega_correlation=float(rec_r[rec_r["parameter"] == "omega"]["correlation"].iloc[0]),
        gamma_correlation=float(rec_r[rec_r["parameter"] == "gamma"]["correlation"].iloc[0]))])

    # --- I4: a confident waiter against a slow learner, matched on how often they adapt ----------------------
    waiter = {**DEFAULT_POPULATION, "omega": 1.2, "omega_sd": 0.25, "sigma_r": 1.0, "sigma_r_sd": 0.25}
    slow = {**DEFAULT_POPULATION, "omega": 0.0, "omega_sd": 0.25, "sigma_r": 6.0, "sigma_r_sd": 0.25,
            "a0": -1.1}
    rows = []
    for label, pop in (("confident_waiter", waiter), ("slow_learner", slow)):
        d, _ = synthetic_pilot(n, model="confidence", seed=seed + 31, population=pop)
        prepped = prepare(d)
        est = fit_model("confidence", prepped).params
        rows.append(dict(generating=label, adjust_rate=float(np.mean([x.d.mean() for x in prepped])),
                         omega_mean=float(est["omega"].mean()), omega_sd=float(est["omega"].std(ddof=1)),
                         sigma_r_mean=float(est["sigma_r"].mean()), n=len(est)))
    out["waiter_vs_slow"] = pd.DataFrame(rows)

    out["verdicts"] = _identifiability_verdicts(out)
    return out


def _identifiability_verdicts(res: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rec = res["omega_recovery"]
    new = rec[rec["factors"] == "new"]["correlation"].mean()
    old = rec[rec["factors"] == "registered"]["correlation"].mean()
    tab = res["confidence_vs_inertia"]
    share_conf = float((tab["selected"] == "confidence").mean())
    share_inert = float((tab["selected"] == "inertia_learn").mean())
    tr = res["risk_tradeoff"].iloc[0]
    wv = res["waiter_vs_slow"].set_index("generating")
    gap = float(wv.loc["confident_waiter", "omega_mean"] - wv.loc["slow_learner", "omega_mean"])
    pooled = float(np.sqrt((wv["omega_sd"] ** 2).mean()))
    rows = [
        dict(claim="I1", statement=IDENTIFIABILITY_CLAIMS["I1"], value=round(float(new), 3),
             supported=bool(new >= 0.5), detail=f"correlation of estimated with true omega = {new:.3f}"),
        dict(claim="I2", statement=IDENTIFIABILITY_CLAIMS["I2"], value=round(share_conf, 3),
             supported=bool(share_conf > share_inert),
             detail=f"confidence selected {share_conf:.0%}, inertia {share_inert:.0%}"),
        dict(claim="I3", statement=IDENTIFIABILITY_CLAIMS["I3"],
             value=round(float(tr["error_correlation"]), 3),
             supported=bool(abs(tr["error_correlation"]) < 0.7),
             detail=f"error correlation {tr['error_correlation']:+.3f}; omega r={tr['omega_correlation']:.2f}, "
                    f"gamma r={tr['gamma_correlation']:.2f}"),
        dict(claim="I4", statement=IDENTIFIABILITY_CLAIMS["I4"], value=round(gap / max(pooled, 1e-9), 3),
             supported=bool(gap / max(pooled, 1e-9) > 1.0),
             detail=f"omega gap {gap:+.3f} against a within-group s.d. of {pooled:.3f}"),
        dict(claim="I5", statement=IDENTIFIABILITY_CLAIMS["I5"], value=round(float(new - old), 3),
             supported=bool(new > old),
             detail=f"correlation {new:.3f} with the new factors, {old:.3f} without"),
    ]
    return pd.DataFrame(rows)


# ================================================================================================ power
def power_curve(sizes: Sequence[int] = (16, 32, 48, 64), reps: int = 12, seed: int = 5,
                effect: float = 0.8, alpha: float = 0.05, estimator: str = "first_pass",
                randomize_paths: bool = True,
                progress: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
    """Power to detect confidence-sensitive adaptation, with the false-positive rate beside it.

    A null row runs at every size. Without it a power curve says nothing, and here it is what caught two real
    problems - the first version of this function rejected the null 50-75% of the time at a nominal 5%:

    1. TWO-PASS POOLING BREAKS THE TEST. Empirical Bayes shrinks the spread of the per-participant estimates
       (s.d. 1.0 -> 0.3-0.5 under the null) while leaving the common offset of a replication untouched. A
       t-statistic divides the offset by the shrunken spread, so it rejects. The estimates share a prior fitted
       to the same data and are not independent. `estimator="first_pass"` uses the fixed-prior estimates, which
       are.
    2. A SHARED STIMULUS MAKES PARTICIPANTS NON-INDEPENDENT. The registered protocol gives everybody the same
       four cost paths, so a path that happens to favour waiting moves the whole sample together. Measured
       under the null: the across-replication s.d. of the mean estimate is 0.374 with shared paths and 0.220
       with paths re-randomised per participant. This offset does NOT shrink with sample size, so a
       participant-level test is miscalibrated at large n however many people are run.

    The default here is therefore the corrected procedure: first-pass estimates, paths re-randomised. Any
    reported power is only meaningful if the null row beside it is near alpha, which is why both are returned
    rather than only the one that was asked for.
    """
    from scipy import stats
    note = progress or (lambda f, m: None)
    idx = param_names("confidence").index("omega")
    rows = []
    total = max(1, len(sizes) * reps * 2)
    done = 0
    for n in sizes:
        for truth, label in ((effect, "effect"), (0.0, "null")):
            hits, means = 0, []
            for r in range(reps):
                pop = {**DEFAULT_POPULATION, "omega": truth, "omega_sd": 0.4}
                df, _ = synthetic_pilot(n, model="confidence", seed=seed + 100 * r + n, population=pop,
                                        randomize_paths=randomize_paths)
                data = prepare(df)
                if estimator == "first_pass":
                    est = np.array([fit_one("confidence", x, *_weak_prior("confidence"))[idx] for x in data])
                else:
                    est = fit_model("confidence", data).params["omega"].to_numpy(float)
                est = est[np.isfinite(est)]
                if len(est) > 2:
                    _, pval = stats.ttest_1samp(est, 0.0)
                    hits += int(pval < alpha)
                    means.append(float(est.mean()))
                done += 1
                note(done / total, f"n={n} {label} rep {r + 1}/{reps}")
            rows.append(dict(n=n, truth=label, true_omega=truth, reps=reps, estimator=estimator,
                             randomized_paths=randomize_paths,
                             rejection_rate=hits / max(reps, 1),
                             across_rep_sd=float(np.std(means, ddof=1)) if len(means) > 1 else np.nan))
    return pd.DataFrame(rows)


# ================================================================================================ the whole study
def run_study(plan: Optional[LearnPlan] = None,
              progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, pd.DataFrame]:
    plan = plan or LearnPlan()
    note = progress or (lambda f, m: None)

    note(0.02, "parameter recovery")
    recs = []
    for i, m in enumerate(plan.models):
        rec, _ = parameter_recovery(m, n=plan.recovery_n, seed=plan.seed + i)
        recs.append(rec.assign(model=m))
        note(0.02 + 0.33 * (i + 1) / len(plan.models), f"parameter recovery: {m}")
    note(0.36, "model recovery")
    conf, per = model_recovery(n=plan.model_recovery_n, seed=plan.seed, models=plan.models,
                               progress=lambda f, m: note(0.36 + 0.24 * f, m))

    note(0.60, "identifiability")
    ident = identifiability(n=plan.recovery_n, seed=plan.seed + 77)

    note(0.75, "power")
    power = power_curve(plan.power_sizes, plan.power_reps, seed=plan.seed + 5,
                        progress=lambda f, m: note(0.75 + 0.25 * f, m))
    note(1.0, "done")

    return {"parameter_recovery": pd.concat(recs, ignore_index=True),
            "model_recovery": conf.reset_index(), "model_recovery_detail": per,
            "omega_recovery": ident["omega_recovery"], "confidence_vs_inertia": ident["confidence_vs_inertia"],
            "risk_tradeoff": ident["risk_tradeoff"], "waiter_vs_slow": ident["waiter_vs_slow"],
            "identifiability": ident["verdicts"], "power": power,
            "models": model_table(), "protocol": protocol_table()}


def model_table() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        rows.append(dict(model=m, label=MODEL_LABELS[m], n_params=len(param_names(m)),
                         parameters=", ".join(param_names(m)),
                         role="confound competitor" if m == "inertia_learn" else "mechanism"))
    return pd.DataFrame(rows)


def protocol_table() -> pd.DataFrame:
    return pd.DataFrame([
        dict(manipulation="outcome variability", factor="condition V", levels="delta 8 -> 24",
             within_participant=True, new=False,
             moves="spread of realized profit", leaves_alone="precision of the cost estimate"),
        dict(manipulation="observation precision", factor="condition N", levels="noise 2 -> 10",
             within_participant=True, new=False,
             moves="precision of the cost estimate", leaves_alone="spread of the underlying world"),
        dict(manipulation="sampling history", factor="history", levels=f"{HISTORY_LEVELS} watched periods",
             within_participant=True, new=True,
             moves="estimation uncertainty at the first scored decision",
             leaves_alone="everything on the screen at the moment of choice"),
        dict(manipulation="opportunities for further learning", factor="horizon",
             levels=f"{HORIZON_LEVELS} scored periods, announced", within_participant=True, new=True,
             moves="value of adapting now", leaves_alone="the screen and the precision of the estimate"),
    ])


EXPORTS = {
    "models": "the five mechanisms, their parameters, and the inertia competitor",
    "protocol": "the four manipulations, what each moves and what it leaves alone",
    "parameter_recovery": "per model and parameter: correlation, bias and RMSE of estimates against truth",
    "model_recovery": "confusion matrix: generating model against selected model",
    "model_recovery_detail": "per participant held-out scores behind that matrix",
    "identifiability": "the five claims about telling confidence sensitivity apart, and whether each holds",
    "omega_recovery": "recovery of omega with and without the two new manipulations",
    "confidence_vs_inertia": "held-out scores when the truth is confidence sensitivity and no inertia",
    "risk_tradeoff": "whether omega and gamma are collinear inside the model that carries both",
    "waiter_vs_slow": "a confident waiter against a slow learner, matched on adjustment rate",
    "power": "rejection rate by sample size, with the false-positive rate beside it",
}
