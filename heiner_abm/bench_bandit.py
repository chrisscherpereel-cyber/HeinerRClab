"""Decision benchmark C: nonstationary bandit learning, a chosen-action-feedback version of the learning task.

Environment (unchanged; heiner_abm.tasks._streams, task "learning")
    Two options whose mean payoffs (+1/2 and -1/2) swap without warning with probability equal to the environment's
    hazard each period; payoffs carry N(0, noise^2) outcome noise, and what is observed carries further N(0, obs_noise^2)
    noise. In the generalization task the agent observes both options' payoffs every period (full feedback). Here an
    agent observes ONLY the (noisy) payoff of the option it chose: obs[t, a_t]. It is scored on the true payoff of its
    choices. Every agent below chooses exactly once per period for the same periods, and every bandit agent receives
    exactly the chosen-action observation.

Agents (bandit: chosen-action feedback)
    ucb         UCB with index mean_i + c sqrt(log n / n_i), n plays so far (Auer, Cesa-Bianchi & Fischer 2002; c
                absorbs the reward scale). Assumes stationary payoffs: a baseline, not a nonstationary method.
    sw_ucb      sliding-window UCB (Garivier & Moulines 2011): mean and count over the last tau periods, padding
                c sqrt(log(min(n, tau)) / N_tau,i) (c = B sqrt(xi) in their notation).
    d_ucb       discounted UCB (Garivier & Moulines 2011): discounted mean and count with factor gamma, padding
                c sqrt(log n_gamma / N_gamma,i) (c = 2 B sqrt(xi)).
    eps_greedy  epsilon-greedy with a constant step size alpha, which weights recent payoffs more (Sutton & Barto
                2018, section 2.5); exploration draws come from a separate random stream.
    oracle      ORACLE: chooses the option with the higher mean payoff in the current regime. Not ranked.
    full_ref    REFERENCE, NOT A BANDIT: the learning task's flexible rule (fast smoothing of both options' payoffs),
                which observes both payoffs every period. Shown to measure the value of full feedback; it is not
                ranked with the bandit agents because its feedback is different.
Untried options are tried first (lowest index first); ties go to the lower index.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from . import tasks
from .bench_tuning import Family, Param, Tuned, budget_curve, ranking, tune
from .tasks import TaskEnv, learning_payoff

ARMS = 2


# Evaluation conditions. The default is the environment as it stands; "heavy" contaminates what agents OBSERVE
# with occasional large outliers, leaving the regime and the true payoffs untouched, so agents are still scored on
# the truth and only the tail of their evidence changes. That is the condition a location estimator has to
# survive, and the reason a robust one is offered at all.
#
# An earlier attempt simply drew the observation noise from a Student t. It did not produce a heavy tail: with
# obs_noise = 0.5 against payoff noise of 1.0, the payoff's own Gaussian noise dominates what is observed, and the
# kurtosis of obs barely moved (3.1 against 2.8). Contamination is applied at the scale of the payoff noise
# instead, which does.
TAILS = ("gaussian", "heavy")
HEAVY_DF = 3.0                       # Student t degrees of freedom for a contaminating draw (finite variance: >2)
HEAVY_RATE = 0.05                    # share of observations contaminated
HEAVY_SCALE = 8.0                    # contamination s.d., in units of the environment's payoff noise


@lru_cache(maxsize=256)
def _stream(env: TaskEnv, T: int, tail: str = "gaussian"):
    s = tasks._streams(env, T, env.seed)
    state, outcome, obs = s["state"], s["outcome"], s["obs"]
    if tail == "gaussian":
        return state, outcome, obs
    if tail != "heavy":
        raise ValueError(f"unknown evaluation condition {tail!r} (one of {', '.join(TAILS)})")
    rng = np.random.default_rng([env.seed, 30517])
    hit = rng.random(obs.shape) < HEAVY_RATE
    z = rng.standard_t(HEAVY_DF, size=obs.shape) / np.sqrt(HEAVY_DF / (HEAVY_DF - 2.0))
    return state, outcome, obs + np.where(hit, HEAVY_SCALE * env.noise * z, 0.0)


def _argmax(x) -> int:
    return int(np.argmax(x))              # first maximum: ties go to the lower index


def ucb_actions(obs: np.ndarray, c: float) -> np.ndarray:
    T = len(obs)
    a = np.zeros(T, int)
    n, s = np.zeros(ARMS), np.zeros(ARMS)
    for t in range(T):
        if (n == 0).any():
            a[t] = _argmax(n == 0)
        else:
            a[t] = _argmax(s / n + c * np.sqrt(np.log(n.sum()) / n))
        r = obs[t, a[t]]                                        # chosen-action feedback only
        n[a[t]] += 1.0
        s[a[t]] += r
    return a


def sw_ucb_actions(obs: np.ndarray, tau: int, c: float) -> np.ndarray:
    T = len(obs)
    a = np.zeros(T, int)
    win: deque = deque()
    n, s = np.zeros(ARMS), np.zeros(ARMS)
    for t in range(T):
        if (n == 0).any():
            a[t] = _argmax(n == 0)
        else:
            a[t] = _argmax(s / n + c * np.sqrt(np.log(min(t, tau)) / n))
        r = obs[t, a[t]]
        win.append((a[t], r))
        n[a[t]] += 1.0
        s[a[t]] += r
        if len(win) > tau:                                      # keep the last tau periods
            i, ri = win.popleft()
            n[i] -= 1.0
            s[i] -= ri
    return a


def d_ucb_actions(obs: np.ndarray, gamma: float, c: float) -> np.ndarray:
    T = len(obs)
    a = np.zeros(T, int)
    n, s = np.zeros(ARMS), np.zeros(ARMS)
    played = np.zeros(ARMS, bool)
    for t in range(T):
        if not played.all():
            a[t] = _argmax(~played)
        else:
            a[t] = _argmax(s / n + c * np.sqrt(np.log(n.sum()) / n))
        r = obs[t, a[t]]
        n *= gamma
        s *= gamma
        n[a[t]] += 1.0
        s[a[t]] += r
        played[a[t]] = True
    return a


# ---------------------------------------------------------------------------------------- CUSUM-UCB
# Liu, Lee & Shroff (2018), AAAI-18, section "CUSUM-UCB". Their two-sided CUSUM detector, per arm, with the first
# M samples after a reset forming the reference mean u0:
#     s_k+ = (y_k - u0 - eps) 1{k > M},      g_k+ = max(0, g_{k-1}+ + s_k+)
#     s_k- = (u0 - y_k - eps) 1{k > M},      g_k- = max(0, g_{k-1}- + s_k-)
# an alarm when g+ >= h or g- >= h, on which that arm's statistics and counts are reset. Arm choice is their
# CD-UCB rule: with probability alpha play uniformly at random (this is what keeps the detectors fed), otherwise
# play argmax of the UCB index xbar_i + sqrt(xi log n / N_i).
#
# ADAPTATIONS TO THIS TASK, none of them in Liu, Lee & Shroff:
#   scale        they assume rewards bounded in [0, 1]. Payoffs here are about +/-0.5 with unbounded Gaussian
#                (or, in the heavy condition, contaminated) noise, so the exploration constant c absorbs the
#                reward scale, as the suite's existing `ucb` family already does. It is tuned, not fixed at
#                their xi = 1.
#   eps          they call eps a KNOWN detectability parameter and assume shifts of at least 3 eps. Nothing
#                here is told the shift size, so eps is tuned on training environments like any other parameter.
#   robust       their u0 and xbar are sample means, which a heavy tail can drag a long way. Under the heavy
#                condition this laboratory substitutes a median-of-means estimator (see ROBUST_ESTIMATORS). That
#                is an adaptation, not a result of theirs, and it is reported as a separate family.
#   M, alarms    forced initial plays per arm via a countdown, as they prescribe; K = 2 here.
ROBUST_ESTIMATORS = ("mean", "median_of_means")


def median_of_means(x: np.ndarray, blocks: int = 5) -> float:
    """Median of block means: a location estimate whose breakdown does not depend on the tail of the payoff.

    With b blocks it tolerates fewer than half the blocks being dominated by outliers, so a few extreme draws move
    it far less than they move the sample mean. Falls back to the mean when there are too few observations to
    block, which is the honest behaviour early in a run rather than a sharper-looking estimate from nothing.
    """
    x = np.asarray(x, float)
    if len(x) == 0:
        return 0.0
    b = int(min(max(blocks, 1), len(x)))
    if b <= 1:
        return float(x.mean())
    cuts = np.array_split(x, b)
    return float(np.median([c.mean() for c in cuts if len(c)]))


def cusum_step(g_plus: float, g_minus: float, y: float, ref: float, eps: float) -> Tuple[float, float]:
    """One step of the two-sided CUSUM of Liu, Lee & Shroff (2018), written as their equations:

        s+ = y - u0 - eps,   g+ = max(0, g+ + s+)
        s- = u0 - y - eps,   g- = max(0, g- + s-)

    Separated out so the recursion can be checked against the published equations directly, not inferred from
    a policy's chosen actions.
    """
    return max(0.0, g_plus + (y - ref - eps)), max(0.0, g_minus + (ref - y - eps))


def _location(values: List[float], robust: str, blocks: int) -> float:
    if not values:
        return 0.0
    return median_of_means(np.asarray(values), blocks) if robust == "median_of_means" else float(np.mean(values))


def cusum_ucb_actions(obs: np.ndarray, eps: float, h: float, alpha: float, c: float, M: int,
                      draws: np.ndarray, robust: str = "mean", blocks: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """CUSUM-UCB (Liu, Lee & Shroff 2018) with the adaptations documented above.

    Returns the actions and, per period, whether an alarm fired, so detection behaviour can be inspected rather
    than inferred from the actions.
    """
    T = len(obs)
    a = np.zeros(T, int)
    alarms = np.zeros(T, bool)
    hist: List[List[float]] = [[] for _ in range(ARMS)]      # observations of this arm since its last reset
    ref = np.zeros(ARMS)                                     # u0, the reference mean, once M samples are in
    gp, gm = np.zeros(ARMS), np.zeros(ARMS)
    n, s = np.zeros(ARMS), np.zeros(ARMS)                    # valid plays and their sum since the last reset
    for t in range(T):
        if (n == 0).any():
            a[t] = _argmax(n == 0)                           # forced initial plays
        elif draws[0, t] < alpha:
            a[t] = int(draws[1, t] * ARMS) % ARMS            # uniform exploration: what feeds the detectors
        else:
            total = max(n.sum(), 1.0)
            a[t] = _argmax(s / n + c * np.sqrt(np.log(total) / n))
        i = int(a[t])
        y = float(obs[t, i])
        n[i] += 1.0
        s[i] += y
        hist[i].append(y)
        k = len(hist[i])
        if k == M:
            ref[i] = _location(hist[i], robust, blocks)
        elif k > M:
            gp[i], gm[i] = cusum_step(gp[i], gm[i], y, ref[i], eps)
            if gp[i] >= h or gm[i] >= h:                     # alarm: reset this arm and restart its detector
                alarms[t] = True
                hist[i], ref[i], gp[i], gm[i], n[i], s[i] = [], 0.0, 0.0, 0.0, 0.0, 0.0
    return a, alarms


# ---------------------------------------------------------------------------------------- Thompson sampling
# Action space: exactly one of the ARMS options per period, for the same periods as every other agent here.
# Observable feedback: obs[t, a_t] alone, the noisy payoff of the option actually chosen. No counterfactual, no
# payoff for the option not chosen, and no knowledge of the regime or of when it switches.
#
# Model: a Gaussian reward with unknown mean and a working reward s.d. `reward_sd`, under a N(0, prior_sd^2) prior
# per arm. The posterior of arm i after effective count n_i and effective sum s_i is Gaussian with
#     precision  1 / prior_sd^2 + n_i / reward_sd^2,     mean  (s_i / reward_sd^2) / precision
# and the agent plays argmax of one independent draw per arm (Thompson 1933; the Gaussian version analysed by
# Agrawal & Goyal 2013). reward_sd is a tuned working value, NOT the environment's noise: the agent is not told
# the generating parameters, and the posterior it uses is therefore its own, not the correct one.
#
# Nonstationarity is handled two ways, kept as separate families rather than combined:
#   ts_sw    sliding window: only the last `window` periods count (Trovo, Paladino, Restelli & Gatti 2020).
#   ts_disc  discounting: the sufficient statistics decay by `gamma` each period. This laboratory's adaptation of
#            the discount device Garivier & Moulines (2011) use for UCB to the Bayesian update; it is not taken
#            from a refereed source, and is labelled an adaptation rather than attributed to one.
def _ts_posterior_draw(s: np.ndarray, n: np.ndarray, reward_sd: float, prior_sd: float,
                       u: np.ndarray) -> np.ndarray:
    """One standard-normal draw per arm, mapped through each arm's Gaussian posterior."""
    prec = 1.0 / (prior_sd * prior_sd) + n / (reward_sd * reward_sd)
    mean = (s / (reward_sd * reward_sd)) / prec
    return mean + u / np.sqrt(prec)


def ts_sw_actions(obs: np.ndarray, window: int, reward_sd: float, prior_sd: float,
                  draws: np.ndarray) -> np.ndarray:
    """Sliding-window Thompson sampling: the posterior uses only the last `window` periods."""
    T = len(obs)
    a = np.zeros(T, int)
    win: deque = deque()
    n, s = np.zeros(ARMS), np.zeros(ARMS)
    for t in range(T):
        a[t] = _argmax(_ts_posterior_draw(s, n, reward_sd, prior_sd, draws[:, t]))
        r = obs[t, a[t]]                                        # chosen-action feedback only
        win.append((a[t], r))
        n[a[t]] += 1.0
        s[a[t]] += r
        if len(win) > window:
            i, ri = win.popleft()
            n[i] -= 1.0
            s[i] -= ri
    return a


def ts_disc_actions(obs: np.ndarray, gamma: float, reward_sd: float, prior_sd: float,
                    draws: np.ndarray) -> np.ndarray:
    """Discounted Thompson sampling: the sufficient statistics decay by gamma each period."""
    T = len(obs)
    a = np.zeros(T, int)
    n, s = np.zeros(ARMS), np.zeros(ARMS)
    for t in range(T):
        a[t] = _argmax(_ts_posterior_draw(s, n, reward_sd, prior_sd, draws[:, t]))
        r = obs[t, a[t]]
        n *= gamma
        s *= gamma
        n[a[t]] += 1.0
        s[a[t]] += r
    return a


def posterior_draws(env: TaskEnv, T: int) -> np.ndarray:
    """Thompson sampling's own random stream, shape (ARMS, T). Separate from the epsilon-greedy stream, so adding
    these families leaves every other agent's draws untouched."""
    return np.random.default_rng([env.seed, 20231]).standard_normal((ARMS, T))


def eps_greedy_actions(obs: np.ndarray, eps: float, alpha: float, draws: np.ndarray) -> np.ndarray:
    T = len(obs)
    a = np.zeros(T, int)
    q = np.zeros(ARMS)
    for t in range(T):
        a[t] = int(draws[1, t] < 0.5) if draws[0, t] < eps else _argmax(q)
        q[a[t]] += alpha * (obs[t, a[t]] - q[a[t]])
    return a


def exploration_draws(env: TaskEnv, T: int) -> np.ndarray:
    return np.random.default_rng([env.seed, 20111]).random((2, T))


def detector_draws(env: TaskEnv, T: int) -> np.ndarray:
    """CUSUM-UCB's uniform-exploration stream, separate from every other agent's."""
    return np.random.default_rng([env.seed, 30911]).random((2, T))


def actions(env: TaskEnv, family: str, cfg: Dict[str, float], T: int, tail: str = "gaussian") -> np.ndarray:
    state, pay, obs = _stream(env, T, tail)
    if family == "ucb":
        return ucb_actions(obs, cfg["c"])
    if family == "sw_ucb":
        return sw_ucb_actions(obs, int(cfg["window"]), cfg["c"])
    if family == "d_ucb":
        return d_ucb_actions(obs, 1.0 - cfg["one_minus_gamma"], cfg["c"])
    if family in ("cusum_ucb", "cusum_ucb_robust"):
        robust = "median_of_means" if family == "cusum_ucb_robust" else "mean"
        acts, _ = cusum_ucb_actions(obs, float(cfg["eps"]), float(cfg["h"]), float(cfg["alpha"]), float(cfg["c"]),
                                    int(cfg["M"]), detector_draws(env, T), robust)
        return acts
    if family == "ts_sw":
        return ts_sw_actions(obs, int(cfg["window"]), float(cfg["reward_sd"]), float(cfg["prior_sd"]),
                             posterior_draws(env, T))
    if family == "ts_disc":
        return ts_disc_actions(obs, 1.0 - float(cfg["one_minus_gamma"]), float(cfg["reward_sd"]),
                               float(cfg["prior_sd"]), posterior_draws(env, T))
    if family == "eps_greedy":
        return eps_greedy_actions(obs, cfg["eps"], cfg["alpha"], exploration_draws(env, T))
    if family == "oracle":
        return (state > 0).astype(int)
    if family == "full_ref":
        return tasks._decisions(replace(env, gain=float(cfg["gain"])), T, env.seed)["x"].astype(int)
    raise ValueError(family)


def run_agent(env: TaskEnv, family: str, cfg: Dict[str, float], T: int, burn: int,
              tail: str = "gaussian") -> Dict[str, float]:
    state, pay, obs = _stream(env, T, tail)
    a = actions(env, family, cfg, T, tail)
    assert a.shape == (T,)                                      # one choice per period for every agent
    got = learning_payoff(a, pay)[burn:]
    best = learning_payoff((state > 0).astype(int), pay)[burn:]
    return dict(payoff=float(got.mean()), regret=float((best - got).mean()),
                correct=float((a == (state > 0))[burn:].mean()), noise=env.noise, hazard=env.hazard,
                obs_noise=env.obs_noise)


FAMILIES: Dict[str, Family] = {f.key: f for f in [
    Family("ucb", "UCB (stationary baseline)", "competitor", {"c": Param(0.01, 5.0, log=True)},
           "Payoffs stationary; the index treats all past plays alike.", "O(K) per period.", ("auer2002",),
           "Chosen-action payoff"),
    Family("sw_ucb", "Sliding-window UCB", "competitor",
           {"window": Param(5, 500, log=True, integer=True), "c": Param(0.01, 5.0, log=True)},
           "Payoffs piecewise stationary; only the last τ periods are relevant.", "O(K) per period, O(τ) memory.",
           ("garivier2011",), "Chosen-action payoff"),
    Family("d_ucb", "Discounted UCB", "competitor",
           {"one_minus_gamma": Param(0.0005, 0.2, log=True), "c": Param(0.01, 5.0, log=True)},
           "Payoffs piecewise stationary; past plays lose weight geometrically.", "O(K) per period.",
           ("garivier2011",), "Chosen-action payoff"),
    Family("cusum_ucb", "Change-detecting UCB (CUSUM-UCB)", "competitor",
           {"eps": Param(0.01, 2.0, log=True), "h": Param(0.5, 100.0, log=True),
            "alpha": Param(0.005, 0.5, log=True), "c": Param(0.01, 5.0, log=True),
            "M": Param(5, 150, log=True, integer=True)},
           "Payoffs piecewise stationary with abrupt changes. A two-sided CUSUM per arm restarts that arm when it "
           "alarms. Liu, Lee & Shroff treat the detectability eps as known and rewards as bounded in [0, 1]; "
           "neither holds here, so eps and the exploration constant are tuned on training environments and the "
           "constant absorbs the reward scale. Its location estimates are sample means, which a heavy tail "
           "can drag.",
           "O(K) per period, O(M) memory per arm; one uniform draw per period.", ("liu2018", "auer2002"),
           "Chosen-action payoff"),
    Family("cusum_ucb_robust", "Change-detecting UCB, robust location (CUSUM-UCB)", "competitor",
           {"eps": Param(0.01, 2.0, log=True), "h": Param(0.5, 100.0, log=True),
            "alpha": Param(0.005, 0.5, log=True), "c": Param(0.01, 5.0, log=True),
            "M": Param(5, 150, log=True, integer=True)},
           "As CUSUM-UCB, with the detector's reference mean taken by median of block means instead of the "
           "sample mean, so a few extreme observations cannot move it far. This substitution is this "
           "laboratory's adaptation for the heavy-tailed evaluation condition, not a result of theirs.",
           "As CUSUM-UCB plus an O(M log M) median over blocks at each reference estimate.",
           ("liu2018", "auer2002"), "Chosen-action payoff"),
    Family("ts_sw", "Sliding-window Thompson sampling", "competitor",
           {"window": Param(5, 500, log=True, integer=True), "reward_sd": Param(0.1, 10.0, log=True),
            "prior_sd": Param(0.1, 10.0, log=True)},
           "Payoffs piecewise stationary; only the last W periods are relevant. Gaussian reward with a working "
           "reward s.d. and a zero-mean Gaussian prior, both TUNED: the agent is not told the environment's noise, "
           "so its posterior is its own and not the correct one.",
           "O(K) per period for K arms, O(W) memory; one normal draw per arm per period.",
           ("thompson1933", "agrawal2013", "trovo2020"), "Chosen-action payoff"),
    Family("ts_disc", "Discounted Thompson sampling", "competitor",
           {"one_minus_gamma": Param(1e-4, 0.3, log=True), "reward_sd": Param(0.1, 10.0, log=True),
            "prior_sd": Param(0.1, 10.0, log=True)},
           "As the sliding window, but older evidence decays geometrically instead of leaving a window. The "
           "discount device is this laboratory's adaptation of discounted UCB to the Bayesian update, not a result "
           "borrowed from a refereed source.",
           "O(K) per period, O(K) memory; one normal draw per arm per period.",
           ("thompson1933", "agrawal2013", "garivier2011"), "Chosen-action payoff"),
    Family("eps_greedy", "ε-greedy, constant step size", "competitor",
           {"eps": Param(0.001, 0.3, log=True), "alpha": Param(0.01, 0.8, log=True)},
           "Recent payoffs matter more (constant step size); exploration at a fixed rate.", "O(K) per period.",
           ("sutton2018",), "Chosen-action payoff"),
    Family("oracle", "ORACLE (knows the better option)", "oracle", {}, "Knows the current regime.", "O(1).", (),
           "The true state"),
    Family("full_ref", "Full-feedback learner (not a bandit)", "reference", {"gain": Param(0.02, 0.8, log=True)},
           "Observes both options' payoffs every period (the generalization task's design).", "O(K) per period.",
           ("behrens2007",), "Both options' payoffs (full feedback)"),
]}


# keys in heiner_abm.information.AGENTS
REGISTRY_KEY = {k: ("task_full_ref" if k == "full_ref" else f"bandit_{k}") for k in FAMILIES}


@dataclass(frozen=True)
class BanditPlan:
    periods: int = 1000
    burn_in: int = 100
    n_train: int = 10
    n_val: int = 10
    n_test: int = 20
    budgets: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    reps: int = 3
    seed: int = 5353
    families: Tuple[str, ...] = tuple(FAMILIES)

    def env_sets(self) -> Dict[str, List[TaskEnv]]:
        sets = {name: tasks._env_set("learning", n, self.seed * 10 + i)
                for i, (name, n) in enumerate((("train", self.n_train), ("validation", self.n_val),
                                               ("test", self.n_test)), 1)}
        seeds = [{e.seed for e in v} for v in sets.values()]
        assert not (seeds[0] & seeds[1] or seeds[0] & seeds[2] or seeds[1] & seeds[2]), "environment sets overlap"
        return sets

    def to_dict(self) -> Dict:
        return asdict(self)


# A quick check is meant to be quick, and it is now spread over more families: the inventory suite gained
# a Bayesian state space with estimated parameters and a moment-ambiguity robust order, the bandit suite
# gained two Thompson variants and two change-detecting ones. The periods were cut to keep an orientation
# run at roughly its previous cost; the Full scale, which is what any reported comparison uses, is
# unchanged.
QUICK_BANDIT = dict(periods=250, burn_in=40, n_train=3, n_val=3, n_test=4, budgets=(1, 2, 4), reps=2)


def run_bandit(plan: BanditPlan, progress=None) -> Dict:
    sets = plan.env_sets()
    results: Dict[str, Tuned] = {}
    for j, key in enumerate(plan.families):
        fam = FAMILIES[key]
        if progress:
            progress(j / len(plan.families), fam.name)
        run = lambda env, cfg, key=key: run_agent(env, key, cfg, plan.periods, plan.burn_in)
        results[key] = tune(fam, run, sets["train"], sets["validation"], sets["test"], plan.budgets, plan.reps,
                            plan.seed + 101 * j)
    if progress:
        progress(1.0, "Done")
    return dict(results=results, ranking=ranking(results, FAMILIES, baseline="ucb"), curve=budget_curve(results),
                plan=plan.to_dict(), environments={k: [asdict(e) for e in v] for k, v in sets.items()})
