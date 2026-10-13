"""Two information environments for the newsvendor: censored demand, and information you have to buy.

PROPOSED EXTENSION. The task, the costs and the critical fractile are the study's (heiner_abm.learnability);
what is new is what the agent gets to see and whether it chooses to see more.

WHY THIS IS A SEPARATE SIMULATOR
    The learnability study draws one exogenous path and runs every policy on it, which is what makes its
    comparisons paired. Censoring breaks that: when stock runs out the agent observes SALES, and sales depend on
    the order the agent itself placed, so two policies on the same demand path no longer see the same data. The
    loop here is therefore sequential and per policy. The demand path is still common across policies (common
    random numbers on the exogenous part), so the comparison stays paired on everything that can be.

1. CENSORED DEMAND
    Each period the agent holds stock S_t, demand d_t arrives, and
        sales_t      = min(d_t, S_t)                     always observed
        stocked_out  = sales_t == S_t                    always observed: the shelf was empty, which is itself a
                                                         signal that demand was at least S_t
        lost_t       = max(0, d_t - S_t)                 observed only with `unmet_access`
    `unmet_access` is the configurable part: 1.0 is a shop that records turned-away customers perfectly, 0.0 one
    that records nothing, and values between are a recording probability per period. Total demand d_t is kept by
    the researcher for evaluation and is never in any agent's contract.

2. ENDOGENOUS INFORMATION ACQUISITION
    Before ordering, a policy that is allowed to may buy one of the instruments in INSTRUMENTS. Each states a
    price, a precision, a delay and what it reveals. The point of the design is that the three are separable: an
    instrument can be cheap and imprecise, expensive and immediate, or precise but too late to use.

FREE-INFORMATION CONTROLS
    Buying information well has two parts that are easy to confuse: having better information, and choosing when
    to get it. A policy that buys usefully is doing both. The exogenous controls give the SAME instrument at the
    SAME rate for free and with no choice, so the difference between a chooser and its matched free control is
    the value of choosing, and the difference between the free control and no information is the value of the
    information itself. Without the control, the two are reported as one number and called the value of
    information, which it is not.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import learnability as L
from .information_contract import InfoContract, InfoView, InformationViolation, RESEARCHER_ONLY

Z_FRAC = L.Z_FRAC
FRACTILE = L.FRACTILE


# ================================================================================================ instruments
@dataclass(frozen=True)
class Instrument:
    """Something an agent can buy to learn more about this period's demand.

    price      charged per purchase, in payoff units, whatever the signal turns out to say
    sd         s.d. of the signal around the quantity it reports; 0 is a perfect reading
    delay      periods before the signal can be used; 0 arrives in time for this period's order
    reveals    "demand" (a reading of this period's demand) or "mean" (a reading of the current mean demand)
    """
    key: str
    name: str
    price: float
    sd: float
    delay: int
    reveals: str
    note: str = ""

    def __post_init__(self):
        if self.reveals not in ("demand", "mean"):
            raise ValueError(f"instrument {self.key!r} reveals {self.reveals!r}, not 'demand' or 'mean'")
        if self.price < 0 or self.sd < 0 or self.delay < 0:
            raise ValueError(f"instrument {self.key!r} has a negative price, s.d. or delay")


INSTRUMENTS: Dict[str, Instrument] = {i.key: i for i in (
    Instrument("none", "No instrument", 0.0, 0.0, 0, "demand", "The agent buys nothing."),
    Instrument("forecast", "Forecast", 1.0, 12.0, 0, "mean",
               "Cheap, immediate, imprecise: a noisy reading of the current mean demand, in time to order on."),
    Instrument("audit", "Audit", 6.0, 3.0, 0, "mean",
               "Expensive, immediate, precise: what a careful count of current conditions would show."),
    Instrument("pilot", "Pilot experiment", 3.0, 1.0, 3, "demand",
               "Precise but late: an almost exact reading of demand that arrives three periods after the "
               "decision it was bought for, so it informs the next decisions and not this one."),
)}


# ================================================================================================ contracts
# One contract per policy. A policy sees its period record through these fields and no others.
OBSERVED_FIELDS = ("sales", "stocked_out", "order", "lost_sales_reported", "signal", "signal_kind", "period")

CONTRACTS: Dict[str, InfoContract] = {
    "censored": InfoContract(
        "censored", "Censored observer", OBSERVED_FIELDS,
        "Sees its own sales, whether it stocked out, and unmet demand only where the environment records it."),
    "censored_buyer": InfoContract(
        "censored_buyer", "Censored observer that may buy information", OBSERVED_FIELDS,
        "As the censored observer, and may buy an instrument before ordering.", may_acquire=True),
    "free_info": InfoContract(
        "free_info", "Censored observer given an instrument free", OBSERVED_FIELDS,
        "As the censored observer, given the same instrument at the same rate with no choice and no charge."),
}


def contract_for(policy: str) -> InfoContract:
    if policy not in CONTRACTS:
        raise KeyError(f"no information contract declared for policy {policy!r}")
    return CONTRACTS[policy]


# ================================================================================================ environment
@dataclass(frozen=True)
class EnvConfig:
    """The environment. Demand and its regime come from the study's generator, unchanged."""
    family: str = "jump"
    sigma: float = 20.0
    tau: float = 0.0                   # measurement noise on a RECORDED observation (sales are recorded exactly)
    hazard: float = 0.01
    unmet_access: float = 0.0          # probability a period's unmet demand is recorded; 0 = fully censored
    cost: float = 1.0                  # default-departure overhead, as in the study
    switch_cost: float = 0.0
    magnitude_cost: float = 0.0
    periods: int = 600
    burn_in: int = 50

    def to_config(self) -> L.Config:
        """The study's Config for the exogenous part, so demand paths are generated by code already registered."""
        return L.Config(family=self.family, sigma=self.sigma, tau=self.tau, hazard=self.hazard, avail=1.0,
                        cost=self.cost, switch_cost=self.switch_cost, magnitude_cost=self.magnitude_cost)


def demand_path(env: EnvConfig, seed: int) -> Dict[str, np.ndarray]:
    """Exogenous demand and its regime, from heiner_abm.learnability. Common across policies."""
    p = L.paths(env.to_config(), env.periods, seed)
    return {"mu": p["mu"], "demand": p["d"], "change": p["change"]}


def record_access(env: EnvConfig, seed: int) -> np.ndarray:
    """Which periods record unmet demand. A separate stream, so changing the recording rate leaves demand alone."""
    return np.random.default_rng([seed, 515_151]).random(env.periods) < env.unmet_access


def signal_noise(env: EnvConfig, seed: int) -> np.ndarray:
    """Standard normal draws for purchased signals; its own stream for the same reason."""
    return np.random.default_rng([seed, 626_262]).standard_normal(env.periods)


# ================================================================================================ the agent
class CensoredLearner:
    """A newsvendor that learns from censored observations, through its contract and nothing else.

    Belief: an exponentially weighted mean and variance of demand. The censoring correction is the part that
    matters. A period that stocked out is a LOWER BOUND on demand, not a reading of it: treating sales as demand
    there biases the estimate down, and ordering less then makes the next stockout more likely, which is the
    classic failure of a censored newsvendor. Where unmet demand is not recorded, a stocked-out period is updated
    towards the censored-normal expectation of demand given that it exceeded the order, which is the simplest
    correction that is not simply wrong.
    """

    def __init__(self, contract: InfoContract, gain: float = 0.1, sd0: float = 25.0, mu0: float = L.MU0,
                 correct_censoring: bool = True):
        self.contract = contract
        self.gain = float(gain)
        self.mu = float(mu0)
        self.var = float(sd0) ** 2
        self.correct_censoring = bool(correct_censoring)
        self.pending: List[Tuple[int, float, str]] = []      # (ready period, value, kind) for delayed signals

    # ---------------------------------------------------------------- belief
    def believed_sd(self) -> float:
        return float(np.sqrt(max(self.var, 1.0)))

    def order(self) -> float:
        return float(max(0.0, self.mu + Z_FRAC * self.believed_sd()))

    def _update(self, value: float) -> None:
        self.var += self.gain * ((value - self.mu) ** 2 - self.var)
        self.mu += self.gain * (value - self.mu)

    def _censored_expectation(self, bound: float) -> float:
        """E[d | d >= bound] under the agent's own current normal belief: what a stockout says about demand."""
        sd = np.sqrt(max(self.var, 1.0))
        z = (bound - self.mu) / sd
        from scipy.special import ndtr
        tail = max(1.0 - float(ndtr(z)), 1e-6)
        lam = float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)) / tail      # inverse Mills ratio
        return float(self.mu + sd * lam)

    # ---------------------------------------------------------------- the period
    def observe(self, view: InfoView, t: int) -> None:
        """Absorb what this period's contract allows. Every read goes through the view."""
        for ready, value, kind in [s for s in self.pending if s[0] <= t]:
            self._update(value if kind == "demand" else value)                # a mean reading is used as a level
        self.pending = [s for s in self.pending if s[0] > t]
        reported = view["lost_sales_reported"]
        sales = float(view["sales"])
        if reported is not None and not np.isnan(reported):
            self._update(sales + float(reported))                             # unmet demand was recorded
        elif bool(view["stocked_out"]) and self.correct_censoring:
            self._update(self._censored_expectation(sales))                   # a bound, not a reading
        else:
            self._update(sales)
        sig = view["signal"]
        if sig is not None and not np.isnan(sig):
            self._update(float(sig))

    def receive_delayed(self, ready: int, value: float, kind: str) -> None:
        self.pending.append((int(ready), float(value), str(kind)))


# ================================================================================================ acquisition
AcquisitionRule = Callable[["CensoredLearner", int], str]


def never_buy(_learner: "CensoredLearner", _t: int) -> str:
    return "none"


def always_buy(instrument: str) -> AcquisitionRule:
    def rule(_learner, _t):
        return instrument
    return rule


def buy_when_uncertain(instrument: str, threshold: float) -> AcquisitionRule:
    """Buy when the agent's own belief is vague enough to be worth sharpening.

    The threshold is on the agent's believed s.d., which is a quantity it has; nothing here consults the truth.
    """
    def rule(learner: "CensoredLearner", _t: int) -> str:
        return instrument if np.sqrt(max(learner.var, 1.0)) > threshold else "none"
    return rule


# ================================================================================================ the run
def simulate(env: EnvConfig, policy: str, seed: int, gain: float = 0.1,
             acquisition: Optional[AcquisitionRule] = None,
             free_instrument: Optional[str] = None, free_rate: float = 0.0,
             correct_censoring: bool = True, band: float = 0.0) -> Dict[str, Any]:
    """One policy on one demand path.

    `acquisition` is consulted only if the contract allows buying. `free_instrument` is the exogenous control:
    the same instrument, handed over at `free_rate` per period, with no choice and no charge.

    `band` makes adaptation a DECISION rather than a reflex: the new target is taken only if it differs from the
    standing order by more than `band` times the agent's own believed s.d., otherwise the order is held. With
    band = 0 the agent re-orders every period, which is what it did before this parameter existed - and with no
    decision to make, an adjustment cost is a constant that cancels out of every comparison instead of changing
    behaviour. That is how the band came to be added: the interaction table showed adjustment cost doing nothing.
    """
    contract = contract_for(policy)
    if acquisition is not None and not contract.may_acquire:
        raise InformationViolation(
            f"policy {policy!r} was given an acquisition rule, but its contract does not allow buying "
            f"information. A policy that may not choose must not be measured as though it could.")
    path = demand_path(env, seed)
    demand, mu = path["demand"], path["mu"]
    recorded = record_access(env, seed)
    noise = signal_noise(env, seed)
    free_draw = np.random.default_rng([seed, 737_373]).random(env.periods)
    learner = CensoredLearner(contract, gain=gain, correct_censoring=correct_censoring)
    T = env.periods
    rows = []
    prev_order = None
    for t in range(T):
        # ---- acquisition, before ordering
        bought = "none"
        if contract.may_acquire and acquisition is not None:
            bought = acquisition(learner, t)
            if bought not in INSTRUMENTS:
                raise ValueError(f"acquisition rule asked for unknown instrument {bought!r}")
        given = "none"
        if free_instrument and free_instrument != "none" and free_draw[t] < free_rate:
            given = free_instrument
        info_cost = float(INSTRUMENTS[bought].price)            # the free control is free by construction
        # ---- the signal each instrument produces
        signal, signal_kind = np.nan, ""
        for key, charged in ((bought, True), (given, False)):
            if key == "none":
                continue
            ins = INSTRUMENTS[key]
            target = float(demand[t]) if ins.reveals == "demand" else float(mu[t])
            value = target + ins.sd * float(noise[t])
            if ins.delay == 0:
                signal, signal_kind = value, ins.reveals
            else:
                learner.receive_delayed(t + ins.delay, value, ins.reveals)
        # ---- order, then demand arrives
        target = learner.order()
        if prev_order is not None and abs(target - prev_order) <= band * learner.believed_sd():
            S = prev_order                                      # inside the band: hold, and pay no charge
        else:
            S = target
        adapted = prev_order is None or abs(S - prev_order) > L.ACTION_TOL
        d = float(demand[t])
        sales = float(min(d, S))
        lost = float(max(0.0, d - S))
        stocked_out = bool(sales >= S - L.ACTION_TOL)
        reported = float(lost) if (recorded[t] and env.unmet_access > 0) else np.nan
        gross = float(L.payoff(np.array([S]), np.array([d]))[0])
        charge = 0.0
        if prev_order is not None and abs(S - prev_order) > L.ACTION_TOL:
            charge += env.switch_cost + env.magnitude_cost * abs(S - prev_order)
        record = {"period": t, "order": S, "sales": sales, "stocked_out": stocked_out,
                  "lost_sales_reported": reported, "signal": signal, "signal_kind": signal_kind,
                  # researcher-only below: present in the record, absent from every contract
                  "demand": d, "mu": float(mu[t]), "lost_sales": lost,
                  "oracle_order": float(max(0.0, mu[t] + Z_FRAC * env.sigma))}
        view = InfoView(record, contract)
        learner.observe(view, t)
        rows.append({**record, "bought": bought, "given": given, "info_cost": info_cost,
                     "target": target, "adapted": bool(adapted),
                     "adjustment_cost": charge, "gross_payoff": gross,
                     "net_payoff": gross - charge - info_cost, "belief_mu": learner.mu,
                     "belief_sd": float(np.sqrt(max(learner.var, 1.0))), "reads": len(view.reads)})
        prev_order = S
    df = pd.DataFrame(rows)
    rec = df.iloc[env.burn_in:]
    return {
        "policy": policy, "seed": seed, "contract": contract.to_dict(), "path": df,
        "net_payoff": float(rec["net_payoff"].mean()), "gross_payoff": float(rec["gross_payoff"].mean()),
        # net of everything EXCEPT the price of information: the performance the agent reached with what it
        # knew, before paying for knowing it. Needed to separate timing skill from the bill.
        "net_before_info": float((rec["net_payoff"] + rec["info_cost"]).mean()),
        "info_cost": float(rec["info_cost"].mean()), "adjustment_cost": float(rec["adjustment_cost"].mean()),
        "purchase_rate": float((rec["bought"] != "none").mean()),
        "given_rate": float((rec["given"] != "none").mean()),
        "stockout_rate": float(rec["stocked_out"].mean()),
        "unmet_recorded_rate": float(rec["lost_sales_reported"].notna().mean()),
        "mean_order": float(rec["order"].mean()),
        "adaptation_rate": float(rec["adapted"].mean()),
        "researcher_mean_demand": float(rec["demand"].mean()),
        "researcher_lost_sales": float(rec["lost_sales"].mean()),
    }


# ================================================================================================ value of information
# A benchmark needs a case where the answer is known. The two-point newsvendor is that case: with demand on two
# points and a normal signal, the prior-optimal order, the perfect-information value and the value of the signal
# are all closed form, so the Monte-Carlo estimator used in the full environment can be CHECKED rather than
# trusted. Everything larger is an approximation, and is labelled one.
def newsvendor_expected_payoff(S: float, values: Sequence[float], probs: Sequence[float]) -> float:
    """Expected payoff of ordering S against a discrete demand distribution. Exact: a finite sum."""
    v, p = np.asarray(values, float), np.asarray(probs, float)
    return float(-np.sum(p * (L.H_OVER * np.maximum(S - v, 0.0) + L.P_SHORT * np.maximum(v - S, 0.0))))


def best_discrete_order(values: Sequence[float], probs: Sequence[float]) -> Tuple[float, float]:
    """The optimal order and its expected payoff. For a discrete demand the optimum is at a support point, so
    enumerating the support is exact rather than a search."""
    best = max(((float(s), newsvendor_expected_payoff(s, values, probs)) for s in values), key=lambda x: x[1])
    return best


def voi_exact_two_point(d_lo: float, d_hi: float, p_hi: float, signal_sd: float) -> Dict[str, float]:
    """EXACT value of information for the two-point newsvendor with one normal signal of this period's demand.

    Perfect information: order the demand that will occur, so the loss is zero.
    The signal x = d + N(0, signal_sd^2) gives a posterior on d_hi that is monotone in x, so the decision rule is
    a threshold on x, and the value integrates in closed form through the normal c.d.f.

    Returns the prior-optimal payoff, the perfect-information payoff (EVPI above the prior), and the value of
    the imperfect signal (EVSI). With signal_sd = 0 the signal is perfect and EVSI must equal EVPI; with
    signal_sd -> infinity it must fall to zero. Both are asserted in the tests.
    """
    from scipy.special import ndtr
    values, probs = (d_lo, d_hi), (1.0 - p_hi, p_hi)
    _, prior = best_discrete_order(values, probs)
    perfect = 0.0                                      # order exactly what arrives: no shortage, no leftover
    if signal_sd <= 0:
        return {"prior_payoff": prior, "perfect_payoff": perfect, "evpi": perfect - prior,
                "signal_payoff": perfect, "evsi": perfect - prior, "threshold": float("nan"),
                "exact": True}
    # posterior odds of d_hi after x are monotone in x; the rule switches to ordering d_hi once the posterior
    # probability crosses the critical fractile of the two-point problem
    pay_lo = {s: newsvendor_expected_payoff(s, (d_lo,), (1.0,)) for s in values}
    pay_hi = {s: newsvendor_expected_payoff(s, (d_hi,), (1.0,)) for s in values}
    # choose d_hi when q * pay_hi[d_hi] + (1-q) * pay_lo[d_hi] >= q * pay_hi[d_lo] + (1-q) * pay_lo[d_lo]
    num = pay_lo[d_lo] - pay_lo[d_hi]
    den = (pay_hi[d_hi] - pay_hi[d_lo]) + (pay_lo[d_lo] - pay_lo[d_hi])
    q_star = float(np.clip(num / den, 0.0, 1.0)) if den != 0 else 0.5
    # q(x) = p f_hi(x) / (p f_hi(x) + (1-p) f_lo(x)) >= q_star  <=>  x >= x_star, solved in logs
    if not 0.0 < q_star < 1.0 or p_hi in (0.0, 1.0):
        x_star = -np.inf if q_star <= 0 else np.inf
    else:
        lhs = np.log(q_star / (1.0 - q_star)) - np.log(p_hi / (1.0 - p_hi))
        x_star = float(lhs * signal_sd ** 2 / (d_hi - d_lo) + (d_hi + d_lo) / 2.0)
    value = 0.0
    for d, pr in ((d_lo, 1.0 - p_hi), (d_hi, p_hi)):
        above = float(1.0 - ndtr((x_star - d) / signal_sd))          # P(order d_hi | demand is d)
        value += pr * (above * newsvendor_expected_payoff(d_hi, (d,), (1.0,))
                       + (1.0 - above) * newsvendor_expected_payoff(d_lo, (d,), (1.0,)))
    return {"prior_payoff": prior, "perfect_payoff": perfect, "evpi": perfect - prior,
            "signal_payoff": value, "evsi": value - prior, "threshold": x_star, "exact": True}


def voi_monte_carlo(d_lo: float, d_hi: float, p_hi: float, signal_sd: float, draws: int = 200_000,
                    seed: int = 0) -> Dict[str, float]:
    """APPROXIMATION of the same quantity by simulation, with the estimator used in the full environment.

    This is the estimator that has to be used once the demand distribution is not two points and the belief is
    not conjugate. Here it can be compared with the exact answer, which is the only reason to run it on a case
    where the exact answer exists.
    """
    rng = np.random.default_rng(seed)
    d = np.where(rng.random(draws) < p_hi, d_hi, d_lo)
    values, probs = (d_lo, d_hi), (1.0 - p_hi, p_hi)
    s_prior, prior_exp = best_discrete_order(values, probs)
    prior = float(np.mean(-(L.H_OVER * np.maximum(s_prior - d, 0.0) + L.P_SHORT * np.maximum(d - s_prior, 0.0))))
    perfect = 0.0
    if signal_sd <= 0:
        chosen = d
    else:
        x = d + signal_sd * rng.standard_normal(draws)
        # posterior odds, then the same decision the exact solution makes
        lo_dens = np.exp(-0.5 * ((x - d_lo) / signal_sd) ** 2)
        hi_dens = np.exp(-0.5 * ((x - d_hi) / signal_sd) ** 2)
        q = p_hi * hi_dens / np.maximum(p_hi * hi_dens + (1.0 - p_hi) * lo_dens, 1e-300)
        pay_hi_if_hi = newsvendor_expected_payoff(d_hi, (d_hi,), (1.0,))
        pay_hi_if_lo = newsvendor_expected_payoff(d_hi, (d_lo,), (1.0,))
        pay_lo_if_hi = newsvendor_expected_payoff(d_lo, (d_hi,), (1.0,))
        pay_lo_if_lo = newsvendor_expected_payoff(d_lo, (d_lo,), (1.0,))
        take_hi = (q * pay_hi_if_hi + (1 - q) * pay_hi_if_lo) >= (q * pay_lo_if_hi + (1 - q) * pay_lo_if_lo)
        chosen = np.where(take_hi, d_hi, d_lo)
    loss = lambda S: -(L.H_OVER * np.maximum(S - d, 0.0) + L.P_SHORT * np.maximum(d - S, 0.0))
    per_draw_signal, per_draw_prior = loss(chosen), loss(s_prior)
    signal_payoff = float(np.mean(per_draw_signal))
    # the two terms share their demand draws, so the VALUE is a paired difference and its standard error must be
    # computed on the pairs. Taking the s.e. of the signal term alone overstates it, which first made the exact
    # and approximate answers look inconsistent when they were not.
    se = float(np.std(per_draw_signal - per_draw_prior, ddof=1) / np.sqrt(draws))
    return {"prior_payoff": prior, "perfect_payoff": perfect, "evpi": perfect - prior,
            "signal_payoff": signal_payoff, "evsi": signal_payoff - prior, "standard_error": se,
            "draws": int(draws), "exact": False,
            "approximation": "Monte Carlo over demand and signal draws; the decision rule is the posterior "
                             "comparison, not a closed form. Checked against voi_exact_two_point in the small "
                             "case; in the full environment no exact answer exists and this is an estimate."}


def voi_in_environment(env: EnvConfig, instrument: str, seed: int, gain: float = 0.1,
                       rate: float = 1.0) -> Dict[str, Any]:
    """APPROXIMATION of the value of an instrument in the full environment, by the free-information control.

    There is no exact answer here: the belief is not conjugate, the regime moves, and the agent's own orders
    determine what it observes. The quantity reported is the difference in net payoff between an agent GIVEN the
    instrument free at `rate` and the same agent given nothing, which is the value of the information alone with
    the cost of buying it and the skill of choosing when removed.
    """
    with_info = simulate(env, "free_info", seed, gain=gain, free_instrument=instrument, free_rate=rate)
    without = simulate(env, "censored", seed, gain=gain)
    return {"instrument": instrument, "rate": rate, "seed": seed,
            "with_information": with_info["net_payoff"], "without_information": without["net_payoff"],
            "value_of_information": with_info["net_payoff"] - without["net_payoff"],
            "price_if_bought": float(INSTRUMENTS[instrument].price) * rate,
            "exact": False,
            "approximation": "difference in mean net payoff between a free-information control and no "
                             "information, on a common demand path. Not an EVSI: the belief is not conjugate "
                             "and the agent's own orders shape what it observes."}


# ================================================================================================ the experiment
@dataclass(frozen=True)
class InfoPlan:
    """Frozen specification of the information experiment."""
    version: str = "1.0"
    seed: int = 20261013
    paths: int = 12
    periods: int = 600
    burn_in: int = 50
    gain: float = 0.1
    instrument: str = "audit"
    threshold_quantile: float = 0.75           # the buyer buys when its believed s.d. is above this quantile
    band: float = 0.35                         # inaction band, in believed s.d.: adaptation is a decision
    unmet_access: Tuple[float, ...] = (0.0, 0.5, 1.0)
    adjustment_costs: Tuple[float, ...] = (0.0, 2.0, 8.0)      # switching cost: what adaptation costs
    hazards: Tuple[float, ...] = (0.002, 0.01, 0.05)           # environmental persistence

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=1, default=str)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def _threshold(env: EnvConfig, plan: InfoPlan, seed: int) -> float:
    """The buyer's trigger, set from its OWN believed spread on a reference run: a quantity it has, not the
    truth. Taken on one path so that the rule is the same across the arms it is compared with."""
    r = simulate(env, "censored", seed, gain=plan.gain, band=plan.band)
    sd = r["path"]["belief_sd"].iloc[env.burn_in:]
    return float(np.quantile(sd, plan.threshold_quantile))


def run_arms(env: EnvConfig, plan: InfoPlan, seed: int) -> Dict[str, Any]:
    """The three arms on one demand path: no information, a buyer, and the matched free control.

    The control is given the same instrument at the rate the buyer actually bought it, so the comparison
    separates two things the literature routinely adds together:
        value of the information   free control  -  no information
        value of CHOOSING          buyer         -  free control
    A buyer that beats no information might be doing neither well: it might simply have been handed a better
    environment. Matching the rate is what makes the second line mean anything.
    """
    thr = _threshold(env, plan, seed)
    none = simulate(env, "censored", seed, gain=plan.gain, band=plan.band)
    buyer = simulate(env, "censored_buyer", seed, gain=plan.gain, band=plan.band,
                     acquisition=buy_when_uncertain(plan.instrument, thr))
    free = simulate(env, "free_info", seed, gain=plan.gain, band=plan.band,
                    free_instrument=plan.instrument, free_rate=buyer["purchase_rate"])
    # An additive decomposition, with each term answering one question:
    #   value of information   what the instrument is worth when it is free and its timing is not chosen
    #   value of choosing      what the buyer's TIMING adds at the same rate, before it pays for anything
    #   price                  what it paid
    # and the three sum to what buying actually got it. An earlier version charged the buyer inside the
    # choosing term, so a bad instrument and bad timing were indistinguishable.
    voi = free["net_payoff"] - none["net_payoff"]
    choosing = buyer["net_before_info"] - free["net_payoff"]
    price = buyer["info_cost"]
    return {"seed": seed, "threshold": thr,
            "none": none, "buyer": buyer, "free": free,
            "value_of_information": voi, "value_of_choosing": choosing, "price_paid": price,
            "net_of_buying": buyer["net_payoff"] - none["net_payoff"],
            "decomposition_residual": (buyer["net_payoff"] - none["net_payoff"]) - (voi + choosing - price),
            "purchase_rate": buyer["purchase_rate"], "info_cost": buyer["info_cost"]}


def run_experiment(plan: InfoPlan, progress: Optional[Callable[[float, str], None]] = None
                   ) -> Dict[str, pd.DataFrame]:
    """The full design: censoring x adjustment cost x persistence, three arms each, several paths.

    Returns four frames, kept apart deliberately (see the module note on exports).
    """
    note = progress or (lambda f, m: None)
    cells, decisions, availability, truth = [], [], [], []
    combos = [(a, c, h) for a in plan.unmet_access for c in plan.adjustment_costs for h in plan.hazards]
    total = max(1, len(combos) * plan.paths)
    done = 0
    for access, adj, hz in combos:
        env = EnvConfig(unmet_access=access, switch_cost=adj, hazard=hz,
                        periods=plan.periods, burn_in=plan.burn_in)
        for k in range(plan.paths):
            seed = plan.seed + 1000 * k
            note(done / total, f"access {access} cost {adj} hazard {hz}: path {k + 1}")
            arms = run_arms(env, plan, seed)
            base = dict(unmet_access=access, adjustment_cost=adj, hazard=hz, seed=seed)
            cells.append({**base, "threshold": arms["threshold"],
                          "net_none": arms["none"]["net_payoff"], "net_buyer": arms["buyer"]["net_payoff"],
                          "net_free": arms["free"]["net_payoff"],
                          "value_of_information": arms["value_of_information"],
                          "value_of_choosing": arms["value_of_choosing"],
                          "price_paid": arms["price_paid"],
                          "decomposition_residual": arms["decomposition_residual"],
                          "net_of_buying": arms["net_of_buying"],
                          "purchase_rate": arms["purchase_rate"], "info_cost": arms["info_cost"],
                          "adaptation_rate_none": arms["none"]["adaptation_rate"],
                          "adaptation_rate_buyer": arms["buyer"]["adaptation_rate"],
                          "stockout_rate_none": arms["none"]["stockout_rate"],
                          "stockout_rate_buyer": arms["buyer"]["stockout_rate"]})
            for arm in ("none", "buyer", "free"):
                p = arms[arm]["path"]
                rec = p.iloc[plan.burn_in:]
                decisions.append({**base, "arm": arm,
                                  "purchases": int((rec["bought"] != "none").sum()),
                                  "purchase_rate": float((rec["bought"] != "none").mean()),
                                  "given": int((rec["given"] != "none").sum()),
                                  "information_cost_total": float(rec["info_cost"].sum()),
                                  "information_cost_per_period": float(rec["info_cost"].mean()),
                                  "adjustment_cost_per_period": float(rec["adjustment_cost"].mean())})
                availability.append({**base, "arm": arm,
                                     "sales_observed_rate": 1.0,
                                     "stockout_rate": float(rec["stocked_out"].mean()),
                                     "unmet_recorded_rate": float(rec["lost_sales_reported"].notna().mean()),
                                     "signal_received_rate": float(rec["signal"].notna().mean())})
                truth.append({**base, "arm": arm,
                              "researcher_mean_demand": float(rec["demand"].mean()),
                              "researcher_mean_mu": float(rec["mu"].mean()),
                              "researcher_lost_sales": float(rec["lost_sales"].mean()),
                              "researcher_oracle_order": float(rec["oracle_order"].mean()),
                              "mean_order": float(rec["order"].mean())})
            done += 1
    note(1.0, "done")
    return {"cells": pd.DataFrame(cells), "decisions": pd.DataFrame(decisions),
            "availability": pd.DataFrame(availability), "truth": pd.DataFrame(truth)}


def interaction_table(cells: pd.DataFrame) -> pd.DataFrame:
    """How the value of information and the value of choosing move with adaptation cost and persistence."""
    g = cells.groupby(["unmet_access", "adjustment_cost", "hazard"], as_index=False).agg(
        value_of_information=("value_of_information", "mean"),
        value_of_choosing=("value_of_choosing", "mean"),
        price_paid=("price_paid", "mean"),
        net_of_buying=("net_of_buying", "mean"),
        purchase_rate=("purchase_rate", "mean"),
        info_cost=("info_cost", "mean"),
        paths=("seed", "count"))
    return g.sort_values(["unmet_access", "adjustment_cost", "hazard"]).reset_index(drop=True)


EXPORTS = {
    "cells": "one row per cell and path: the three arms and the decomposition",
    "decisions": "acquisition decisions and what they cost, per arm",
    "availability": "what was observable: sales, stockouts, recorded unmet demand, signals received",
    "truth": "researcher-only ground truth, kept in its own file so it cannot be mistaken for agent information",
}
