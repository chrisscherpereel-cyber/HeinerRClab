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


@lru_cache(maxsize=256)
def _stream(env: TaskEnv, T: int):
    s = tasks._streams(env, T, env.seed)
    return s["state"], s["outcome"], s["obs"]


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


def actions(env: TaskEnv, family: str, cfg: Dict[str, float], T: int) -> np.ndarray:
    state, pay, obs = _stream(env, T)
    if family == "ucb":
        return ucb_actions(obs, cfg["c"])
    if family == "sw_ucb":
        return sw_ucb_actions(obs, int(cfg["window"]), cfg["c"])
    if family == "d_ucb":
        return d_ucb_actions(obs, 1.0 - cfg["one_minus_gamma"], cfg["c"])
    if family == "eps_greedy":
        return eps_greedy_actions(obs, cfg["eps"], cfg["alpha"], exploration_draws(env, T))
    if family == "oracle":
        return (state > 0).astype(int)
    if family == "full_ref":
        return tasks._decisions(replace(env, gain=float(cfg["gain"])), T, env.seed)["x"].astype(int)
    raise ValueError(family)


def run_agent(env: TaskEnv, family: str, cfg: Dict[str, float], T: int, burn: int) -> Dict[str, float]:
    state, pay, obs = _stream(env, T)
    a = actions(env, family, cfg, T)
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


QUICK_BANDIT = dict(periods=400, burn_in=50, n_train=3, n_val=3, n_test=5, budgets=(1, 2, 4), reps=2)


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
