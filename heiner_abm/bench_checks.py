"""Correctness checks of the decision benchmarks on analytically solvable or small enumerated cases. They are run
before any performance comparison (the Decision benchmarks page shows them; tests/test_benchmarks.py requires every
one to pass)."""
from __future__ import annotations

import itertools
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from . import bench_bandit as BB
from . import bench_inventory as BI
from . import tasks


def _ok(name: str, what: str, err: float, tol: float) -> Tuple[str, str, bool, float, float]:
    return (name, what, bool(err <= tol), float(err), tol)


# ------------------------------------------------------------------------------------------------ A: Bayes
def check_grid_filter_enumeration():
    """Three-point grid (30, 100, 170), four observations: the forward filter's posterior mean equals the posterior
    mean from enumerating all 3^4 paths of the mean demand."""
    grid = np.array([30.0, 100.0, 170.0])
    h, sigma, tau = 0.3, 25.0, 10.0
    P = BI.grid_transition(grid, h)
    y = np.array([95.0, 140.0, 160.0, 20.0])
    _, means = BI.bayes_true_orders(y, h, sigma, tau, grid_step=70.0, return_post=True)
    num = den = 0.0
    for path in itertools.product(range(3), repeat=len(y)):
        pr, prev = 1.0, 1                                      # start at MU0 = 100 (index 1)
        for t, j in enumerate(path):
            pr *= P[prev, j] * BI.obs_likelihood(y[t], grid[j:j + 1], sigma, tau)[0]
            prev = j
        num += pr * grid[path[-1]]
        den += pr
    return _ok("A1", "Grid filter = enumeration of all paths (3 states, T = 4)", abs(means[-1] - num / den), 1e-9)


def check_censored_quantile():
    """A point-mass predictive with negligible censoring: the order is mu + z sigma with z the 0.8 normal quantile;
    for a two-point mixture the order's predictive CDF is 0.8 and it beats nearby orders in exact expected cost."""
    S = BI.censored_mixture_quantile(np.array([1.0]), np.array([120.0]), 15.0)
    e1 = abs(S - (120.0 + BI.Z_FRAC * 15.0))
    w, mu, sd = np.array([0.3, 0.7]), np.array([40.0, 110.0]), 20.0
    S2 = BI.censored_mixture_quantile(w, mu, sd)
    d = np.linspace(0, 400, 400001)
    pdf = sum(wi * np.exp(-0.5 * ((d - m) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) for wi, m in zip(w, mu))
    mass0 = float(np.dot(w, ndtr(-mu / sd)))
    cost = lambda s: mass0 * BI.newsvendor_loss(s, 0.0) + np.trapezoid(BI.newsvendor_loss(s, d) * pdf, d)
    worse = min(cost(S2 - 1.0), cost(S2 + 1.0)) - cost(S2)
    e2 = abs(float(np.dot(w, ndtr((S2 - mu) / sd))) - BI.FRACTILE)
    return _ok("A2", "Posterior expected-cost order: point mass = μ + zσ; mixture CDF = 0.8 and a local minimum",
               max(e1, e2, 0.0 if worse > 0 else 1.0), 1e-6)


def _seg_marglik(ys, m0, v0, v):
    """Marginal likelihood of one segment under the conjugate normal model (sequential predictive product)."""
    mu, var, out = m0, v0, 1.0
    for y in ys:
        s2 = var + v
        out *= np.exp(-0.5 * (y - mu) ** 2 / s2) / np.sqrt(2 * np.pi * s2)
        prec = 1 / var + 1 / v
        mu, var = (mu / var + y / v) / prec, 1 / prec
    return out


def check_bocpd_enumeration():
    """Run-length posterior of BOCPD = enumeration of all 2^(T-1) change-point configurations (T = 6)."""
    H, m0, s0, sm = 0.2, 100.0, 30.0, 12.0
    y = np.array([98.0, 104.0, 150.0, 146.0, 155.0, 90.0])
    model = BI.BOCPD(H, m0, s0, sm, rmax=50)
    for v in y:
        model.update(v)
    T = len(y)
    post = np.zeros(T)
    for cps in itertools.product([0, 1], repeat=T - 1):        # cps[j] = 1: y_{j+2} starts a new segment
        starts = [0] + [j + 1 for j, c in enumerate(cps) if c]
        prior = H ** sum(cps) * (1 - H) ** (T - 1 - sum(cps))
        segs = list(zip(starts, starts[1:] + [T]))
        lik = np.prod([_seg_marglik(y[a:b], m0, s0 ** 2, sm ** 2) for a, b in segs])
        post[T - starts[-1] - 1] += prior * lik
    post /= post.sum()
    return _ok("A3", "BOCPD run-length posterior = enumeration of all change-point configurations (T = 6)",
               float(np.abs(model.w - post).max()), 1e-12)


def check_bocpd_no_change():
    """With hazard 0, BOCPD is the conjugate normal posterior of a single mean."""
    m0, s0, sm = 100.0, 30.0, 12.0
    y = np.array([98.0, 104.0, 110.0, 101.0])
    model = BI.BOCPD(0.0, m0, s0, sm)
    for v in y:
        model.update(v)
    prec = 1 / s0 ** 2 + len(y) / sm ** 2
    exact = (m0 / s0 ** 2 + y.sum() / sm ** 2) / prec
    return _ok("A4", "BOCPD with hazard 0 = conjugate normal posterior", abs(float(model.w @ model.mu) - exact)
               + abs(model.w[-1] - 1.0), 1e-9)


def check_environment_constants():
    """The documented generating process (MU0, jump s.d., bounds, censoring, observation noise) reproduces
    tasks._streams draw for draw."""
    env = tasks.TaskEnv("inventory", 20.0, 0.05, 7.0, 0.1, 12345)
    T = 500
    s = tasks._streams(env, T, env.seed)
    rng = np.random.default_rng(env.seed)
    jumps = rng.random(T) < env.hazard
    steps = np.where(jumps, rng.normal(0, BI.JUMP_SD, T), 0.0)
    mu, m = np.empty(T), BI.MU0
    for t in range(T):
        m = min(BI.MU_HI, max(BI.MU_LO, m + steps[t]))
        mu[t] = m
    d = np.maximum(0.0, mu + env.noise * rng.standard_normal(T))
    y = d + env.obs_noise * rng.standard_normal(T)
    err = max(np.abs(mu - s["state"]).max(), np.abs(d - s["outcome"]).max(), np.abs(y - s["obs"]).max())
    return _ok("A5", "Documented inventory process reproduces the task's generator exactly", float(err), 0.0)


# ------------------------------------------------------------------------------------------------ B: SAA and DRO
def check_zero_ambiguity():
    """ρ = 0: the worst-case loss equals the empirical average loss for every order, and the robust order attains the
    SAA optimum (the ceil(0.8 N)-th smallest observation)."""
    rng = np.random.default_rng(3)
    W = rng.normal(100, 25, (40, 23))
    S = rng.uniform(40, 160, (40, 1))
    e1 = np.abs(BI.dro_objective(S[:, 0], W, 0.0) - BI.newsvendor_loss(S, W).mean(axis=1)).max()
    s_saa = BI.saa_order_window(W)
    s_dro = BI.dro_orders_window(W, 0.0)
    v_saa = BI.newsvendor_loss(s_saa[:, None], W).mean(axis=1)
    v_dro = BI.newsvendor_loss(s_dro[:, None], W).mean(axis=1)
    cand = BI.newsvendor_loss(W[:, :, None], W[:, None, :]).mean(axis=2).min(axis=1)   # optimum among the data
    return _ok("B1", "Zero ambiguity reduces to SAA (objective identical; same optimal value; SAA = order statistic)",
               float(max(e1, np.abs(v_dro - v_saa).max(), np.abs(v_saa - cand).max())), 1e-6)


def check_closed_form_worst_case():
    """When no weight hits zero, the worst case is mean + sqrt(ρ · variance) (population variance)."""
    L = np.array([3.0, 5.0, 6.0, 9.0, 12.0])
    rho = 0.05
    exact = L.mean() + np.sqrt(rho * L.var())
    return _ok("B2", "Worst case = mean + √(ρ·variance) when the ball keeps every weight positive",
               abs(float(BI.chi2_worst_case(L, rho)) - exact), 1e-12)


def check_worst_case_numerical():
    """Exact worst case = numerical maximization (SLSQP) over the ball, including cases where weights hit zero, and
    = the largest loss once the ball contains a point mass (ρ ≥ N - 1)."""
    rng = np.random.default_rng(11)
    err = 0.0
    for N, rho in ((4, 0.3), (6, 1.5), (5, 3.0), (7, 0.8)):
        L = rng.uniform(0, 20, N)
        p = 1.0 / N
        cons = [{"type": "eq", "fun": lambda q: q.sum() - 1.0},
                {"type": "ineq", "fun": lambda q, rho=rho: rho * p - ((q - p) ** 2).sum()}]
        res = max((minimize(lambda q: -q @ L, x0, bounds=[(0, 1)] * N, constraints=cons, method="SLSQP",
                            options=dict(ftol=1e-14, maxiter=500)) for x0 in
                   (np.full(N, p), np.eye(N)[np.argmax(L)] * 0.5 + 0.5 * p)), key=lambda r: -r.fun)
        err = max(err, abs(-res.fun - float(BI.chi2_worst_case(L, rho))))
    L = rng.uniform(0, 20, 6)
    err = max(err, abs(float(BI.chi2_worst_case(L, 5.0)) - L.max()))
    return _ok("B3", "Exact worst case = numerical optimum (SLSQP), and = max loss for ρ ≥ N − 1", err, 1e-6)


def check_dro_minimizer():
    """The robust order minimizes the worst-case loss: no order on a fine grid does better."""
    rng = np.random.default_rng(5)
    W = rng.normal(100, 30, (1, 15))
    S = BI.dro_orders_window(W, 0.4)[0]
    grid = np.linspace(W.min(), W.max(), 4001)
    f = BI.dro_objective(grid, np.repeat(W, len(grid), 0), 0.4)
    return _ok("B4", "Robust order = minimizer of the worst-case loss (fine grid)",
               max(0.0, float(BI.dro_objective(np.array([S]), W, 0.4)[0] - f.min())), 1e-6)


def check_minimax_limit():
    """ρ ≥ N − 1: the ball contains every point mass, the worst case is the largest loss, and the robust order is the
    minimax order (h·min + p·max) / (h + p)."""
    W = np.random.default_rng(8).normal(100, 25, (1, 40))
    S = float(BI.dro_orders_window(W, 39.0)[0])
    exact = (BI.H_OVER * W.min() + BI.P_SHORT * W.max()) / (BI.H_OVER + BI.P_SHORT)
    return _ok("B5", "Full ambiguity (ρ ≥ N − 1) gives the minimax order (h·min + p·max)/(h + p)", abs(S - exact), 1e-6)


# ------------------------------------------------------------------------------------------------ C: bandits
def _bandit_obs(T=300, seed=2):
    env = tasks.TaskEnv("learning", 1.0, 0.02, 0.5, 0.1, seed)
    return env, BB._stream(env, T)[2]


def check_bandit_limits():
    """Sliding window longer than the run = UCB; discount factor 1 = UCB (identical choices)."""
    env, obs = _bandit_obs()
    a = BB.ucb_actions(obs, 0.7)
    d1 = int((BB.sw_ucb_actions(obs, 10_000, 0.7) != a).sum())
    d2 = int((BB.d_ucb_actions(obs, 1.0, 0.7) != a).sum())
    return _ok("C1", "SW-UCB with τ ≥ T and D-UCB with γ = 1 reduce to UCB", float(d1 + d2), 0.0)


def check_chosen_action_only():
    """Changing the payoff of the option NOT chosen changes no bandit agent's choices (chosen-action feedback only)."""
    env, obs = _bandit_obs()
    T = len(obs)
    draws = BB.exploration_draws(env, T)
    runs = {"ucb": lambda o: BB.ucb_actions(o, 0.5), "sw_ucb": lambda o: BB.sw_ucb_actions(o, 30, 0.5),
            "d_ucb": lambda o: BB.d_ucb_actions(o, 0.95, 0.5),
            "eps_greedy": lambda o: BB.eps_greedy_actions(o, 0.1, 0.2, draws)}
    changed = 0
    for f in runs.values():
        a = f(obs)
        alt = obs.copy()
        alt[np.arange(T), 1 - a] += np.random.default_rng(0).normal(0, 50, T)   # unseen payoffs perturbed
        changed += int((f(alt) != a).sum())
    return _ok("C2", "Bandit agents ignore payoffs of options they did not choose", float(changed), 0.0)


def check_deterministic_greedy():
    """Stationary, noise-free payoffs: with no exploration bonus UCB tries each option once and then always picks the
    better one; ε-greedy with ε = 0 and α = 1 does the same after trying the better option."""
    T = 50
    obs = np.tile([0.2, 0.9], (T, 1))
    a = BB.ucb_actions(obs, 0.0)
    a2 = BB.eps_greedy_actions(obs[:, ::-1].copy(), 0.0, 1.0, np.ones((2, T)))   # better option first here
    err = float((a[:2] != [0, 1]).sum() + (a[2:] != 1).sum() + (a2 != 0).sum())
    return _ok("C3", "Noise-free stationary case: greedy choices are the analytic ones", err, 0.0)


CHECKS: List[Callable] = [check_grid_filter_enumeration, check_censored_quantile, check_bocpd_enumeration,
                          check_bocpd_no_change, check_environment_constants, check_zero_ambiguity,
                          check_closed_form_worst_case, check_worst_case_numerical, check_dro_minimizer, check_minimax_limit,
                          check_bandit_limits, check_chosen_action_only, check_deterministic_greedy]


def run_checks():
    import pandas as pd
    return pd.DataFrame([c() for c in CHECKS], columns=["id", "check", "passed", "error", "tolerance"])
