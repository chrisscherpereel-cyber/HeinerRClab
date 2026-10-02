"""A single-firm benchmark that can be solved on paper: tracking a moving target observed with noise.

The target follows a random walk and is observed with noise:
    x_t = x_{t-1} + w_t,  w_t ~ N(0, q)        (the best action moves)
    y_t = x_t + v_t,      v_t ~ N(0, r)        (the firm sees it with error)
After seeing y_t the firm sets a_t and loses L(a_t - x_t).

Part 1 · exact result (Muth 1960; Kalman 1960). Under quadratic loss, within the partial-adjustment rules
    a_t = a_{t-1} + phi (y_t - a_{t-1})
the expected loss per period is
    E(phi) = [(1 - phi)^2 q + phi^2 r] / [phi (2 - phi)],
minimized at the steady-state Kalman gain
    k = P / (P + r),  P = (q + sqrt(q^2 + 4 q r)) / 2,
which depends only on the signal-to-noise ratio q / r. The simulation must reproduce this curve and its minimum.

Part 2 · lopsided stakes. Overshooting the target (a > x) costs c_o per squared unit and undershooting costs c_u, with
the stakes ratio rho = c_o / c_u (normalized so that (c_o + c_u) / 2 = 1, which keeps the loss of any symmetric error
distribution unchanged). The two theories now part ways:
    * optimal filtering (certainty equivalence): the weight on new information depends only on q / r; stakes shift the
      level of the action (an offset), never how much the firm responds to news;
    * Heiner's reliability condition: whether to act on a signal depends on the tolerance limit, which rises with the
      loss from a wrong move relative to the gain from a right one; lopsided stakes therefore call for restricting
      moves in the costly direction, holding the information (q / r) fixed.
Rules compared, all fed by the same observations:
    filter          Kalman filter mean (the optimal-filtering prescription; responds to every observation)
    filter_offset   Kalman mean shifted by the loss-minimizing offset (Bayes-optimal under these stakes)
    speed_sym       partial adjustment toward y_t with one speed, tuned for the stakes
    speed_asym      partial adjustment with separate speeds up and down, tuned for the stakes
    restrict        move to the Kalman mean only when the move exceeds a threshold, with separate thresholds up and
                    down, tuned for the stakes (reliability-based restriction of the optimal filter)
Parameters are tuned on training paths and evaluated on separate test paths with the same random numbers for every
rule.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .analysis import ols

NEVER = -1e9                  # threshold meaning "always move"

TRACK_HYPOTHESES = (
    ("T1", "Exactness: the simulation reproduces the analytic solution: its loss curve over adjustment speeds matches "
           "Muth's formula and its best speed is the Kalman gain, at every signal-to-noise ratio.",
     "Supported if the largest relative gap between simulated and analytic loss is below 5% and the simulated best "
     "speed is within one grid step of the Kalman gain, for every signal-to-noise ratio."),
    ("T2", "Agreement under symmetric stakes: no restriction or direction-specific speed beats the Kalman filter when "
           "overshooting and undershooting cost the same.",
     "At rho = 1, test-path loss of the best restricted rule and of the best direction-specific speeds minus the "
     "filter's loss; 95% bootstrap CI over test paths. Supported if neither CI lies below 0."),
    ("T3", "Heiner's unique prediction: with lopsided stakes, restricting the filter's moves pays, and pays more the "
           "more lopsided the stakes, although the information (q / r) is unchanged.",
     "Gain of the restricted rule over the filter per test path. Supported if the gain's CI lies above 0 at every "
     "rho > 1 and the OLS slope of the gain on ln rho is positive with p < α."),
    ("T4", "The limit of the prediction: once the stakes are built into the estimate (Kalman mean plus the optimal "
           "offset), restriction adds nothing.",
     "Loss of the restricted rule minus the loss of the filter with optimal offset, at every rho. Supported if no CI "
     "lies below 0 (the restricted rule never beats the Bayes-optimal rule)."),
)


@dataclass(frozen=True)
class TrackPlan:
    version: str = "1.0"
    snrs: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)        # q / r with r = 1
    phi_grid: int = 96                                     # speeds 0.05 ... 1.00 in steps of 0.01
    verify_reps: int = 8
    verify_periods: int = 50000
    stakes_snr: float = 0.1
    rhos: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    speed_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.05, 1.0, 20), 4))
    threshold_grid: Tuple[float, ...] = tuple(np.round(np.arange(0.0, 3.01, 0.25), 4))
    train_reps: int = 8
    test_reps: int = 16
    periods: int = 20000
    burn_in: int = 500
    seed: int = 8008
    alpha: float = 0.05
    n_boot: int = 2000
    hypotheses: Tuple[Tuple[str, str, str], ...] = TRACK_HYPOTHESES

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "code": track_code_digest()}, sort_keys=True, indent=1)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def track_code_digest() -> str:
    mod = sys.modules[__name__]
    src = "".join(inspect.getsource(f) for f in (mod.kalman_gain, mod.loss_of_speed, mod.weights, mod.optimal_offset,
                                                  mod.simulate_rules, mod.run_tracking))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


QUICK_TRACK = dict(verify_reps=3, verify_periods=4000, phi_grid=20, train_reps=3, test_reps=6, periods=4000,
                   burn_in=200, n_boot=300, speed_grid=tuple(np.round(np.linspace(0.1, 1.0, 10), 4)),
                   threshold_grid=tuple(np.round(np.arange(0.0, 3.01, 0.5), 4)))


# ================================================================================================ analytic results
def kalman_gain(q: float, r: float) -> float:
    """Steady-state Kalman gain for a random walk observed with noise (Muth 1960; Kalman 1960)."""
    P = (q + np.sqrt(q * q + 4 * q * r)) / 2          # prior variance
    return float(P / (P + r))


def posterior_var(q: float, r: float) -> float:
    return kalman_gain(q, r) * r


def loss_of_speed(phi, q: float, r: float):
    """Expected squared error per period of partial adjustment with speed phi (steady state)."""
    phi = np.asarray(phi, float)
    return ((1 - phi) ** 2 * q + phi ** 2 * r) / (phi * (2 - phi))


def weights(rho: float) -> Tuple[float, float]:
    """Loss weights (overshoot, undershoot) with c_o / c_u = rho and (c_o + c_u) / 2 = 1."""
    return 2 * rho / (1 + rho), 2 / (1 + rho)


def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + np.vectorize(erf)(np.asarray(x) / sqrt(2)))


def _norm_pdf(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / np.sqrt(2 * np.pi)


def expected_asym_loss(mu, rho: float):
    """E[c_o Z^2 1(Z > 0) + c_u Z^2 1(Z < 0)] for Z ~ N(mu, 1)."""
    co, cu = weights(rho)
    mu = np.asarray(mu, float)
    Phi, phi = _norm_cdf(mu), _norm_pdf(mu)
    pos = (mu ** 2 + 1) * Phi + mu * phi
    neg = (mu ** 2 + 1) * (1 - Phi) - mu * phi
    return co * pos + cu * neg


def optimal_offset(rho: float) -> Tuple[float, float]:
    """Loss-minimizing mean error (in posterior s.d.) under the stakes, and the minimal loss (per unit variance).
    Under certainty equivalence the firm sets a = posterior mean + offset * posterior s.d."""
    grid = np.linspace(-3, 3, 6001)
    L = expected_asym_loss(grid, rho)
    i = int(np.argmin(L))
    return float(grid[i]), float(L[i])


# ================================================================================================ simulation
def _paths(q: float, r: float, reps: int, T: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.cumsum(np.sqrt(q) * rng.standard_normal((T, reps)), axis=0)
    y = x + np.sqrt(r) * rng.standard_normal((T, reps))
    return x, y


def simulate_rules(x: np.ndarray, y: np.ndarray, q: float, r: float, rules: pd.DataFrame, burn: int
                   ) -> Dict[str, np.ndarray]:
    """Run many rules at once on the same paths.

    Each rule (row of `rules`) has: source ('y' = partial adjustment toward the observation, 'kf' = toward the Kalman
    mean), phi_up / phi_down (speed of upward and downward moves), th_up / th_down (a move is made only if it exceeds
    th · s, with s the posterior s.d.; NEVER = always move) and offset (added to the Kalman mean, in posterior s.d.).
    Returns per rule and path: squared error accumulated separately for overshooting and undershooting, and the share
    of periods with an upward and a downward move (after burn-in)."""
    T, R = y.shape
    n = len(rules)
    k = kalman_gain(q, r)
    s = np.sqrt(posterior_var(q, r))
    use_kf = (rules["source"].to_numpy() == "kf")[:, None]
    pu, pd_ = rules["phi_up"].to_numpy()[:, None], rules["phi_down"].to_numpy()[:, None]
    th_u, th_d = rules["th_up"].to_numpy()[:, None], rules["th_down"].to_numpy()[:, None]
    tu = np.where(th_u <= NEVER / 2, 0.0, th_u * s)          # NEVER: move whenever there is any difference
    td = np.where(th_d <= NEVER / 2, 0.0, th_d * s)
    off = rules["offset"].to_numpy()[:, None] * s
    a = np.zeros((n, R))
    m = np.zeros(R)
    over, under = np.zeros((n, R)), np.zeros((n, R))
    ups, downs = np.zeros((n, R)), np.zeros((n, R))
    for t in range(T):
        m = m + k * (y[t] - m)
        z = np.where(use_kf, m[None, :] + off, y[t][None, :])
        d = z - a
        up = d > tu
        down = -d > td
        a = a + np.where(up, pu * d, 0.0) + np.where(down, pd_ * d, 0.0)
        if t >= burn:
            e = a - x[t][None, :]
            over += np.where(e > 0, e * e, 0.0)
            under += np.where(e < 0, e * e, 0.0)
            ups += up
            downs += down
    n_rec = T - burn
    return dict(over=over / n_rec, under=under / n_rec, up=ups / n_rec, down=downs / n_rec)


def _loss(sim: Dict[str, np.ndarray], rho: float) -> np.ndarray:
    co, cu = weights(rho)
    return co * sim["over"] + cu * sim["under"]


def _rule_table(plan: TrackPlan) -> pd.DataFrame:
    rows = [dict(family="filter", source="kf", phi_up=1.0, phi_down=1.0, th_up=NEVER, th_down=NEVER, offset=0.0)]
    rows += [dict(family="speed_sym", source="y", phi_up=p, phi_down=p, th_up=NEVER, th_down=NEVER, offset=0.0)
             for p in plan.speed_grid]
    rows += [dict(family="speed_asym", source="y", phi_up=pu, phi_down=pd_, th_up=NEVER, th_down=NEVER, offset=0.0)
             for pu in plan.speed_grid for pd_ in plan.speed_grid]
    rows += [dict(family="restrict", source="kf", phi_up=1.0, phi_down=1.0, th_up=tu, th_down=td, offset=0.0)
             for tu in plan.threshold_grid for td in plan.threshold_grid]
    rows += [dict(family="filter_offset", source="kf", phi_up=1.0, phi_down=1.0, th_up=NEVER, th_down=NEVER,
                  offset=optimal_offset(rho)[0], rho=rho) for rho in plan.rhos]
    return pd.DataFrame(rows)


FAMILY_LABELS = {"filter": "Kalman filter (optimal filtering)", "filter_offset": "Kalman filter + optimal offset",
                 "speed_sym": "Partial adjustment, one speed", "speed_asym": "Partial adjustment, speeds up/down",
                 "restrict": "Restricted filter (thresholds up/down)"}


@dataclass
class TrackResult:
    verify: pd.DataFrame          # per snr and speed: simulated and analytic loss
    verify_summary: pd.DataFrame  # per snr: Kalman gain, simulated best speed, max relative gap
    stakes: pd.DataFrame          # per rho, family and test path: loss and move rates of the tuned rule
    chosen: pd.DataFrame          # per rho and family: tuned parameters
    verdicts: pd.DataFrame


def run_tracking(plan: TrackPlan, progress: Optional[Callable[[float, str], None]] = None) -> TrackResult:
    tick = (lambda f, m: progress(f, m)) if progress else (lambda f, m: None)
    # ---------------------------------------------------------------- part 1: exactness
    phis = np.round(np.linspace(0.05, 1.0, plan.phi_grid), 4)
    vrows, srows = [], []
    for i, snr in enumerate(plan.snrs):
        tick(0.4 * i / len(plan.snrs), f"Verifying against the analytic solution, q/r = {snr:g}")
        x, y = _paths(snr, 1.0, plan.verify_reps, plan.verify_periods, plan.seed + i)
        rules = pd.DataFrame(dict(source="y", phi_up=phis, phi_down=phis, th_up=NEVER, th_down=NEVER, offset=0.0))
        sim = simulate_rules(x, y, snr, 1.0, rules, plan.burn_in)
        mse = (sim["over"] + sim["under"]).mean(1)
        exact = loss_of_speed(phis, snr, 1.0)
        for p, a_, b_ in zip(phis, mse, exact):
            vrows.append(dict(snr=snr, phi=p, simulated=a_, analytic=b_))
        k = kalman_gain(snr, 1.0)
        srows.append(dict(snr=snr, kalman_gain=k, simulated_best=float(phis[np.argmin(mse)]),
                          analytic_best_on_grid=float(phis[np.argmin(exact)]), grid_step=float(phis[1] - phis[0]),
                          max_rel_gap=float(np.max(np.abs(mse / exact - 1))),
                          min_loss=float(loss_of_speed(k, snr, 1.0)), posterior_var=posterior_var(snr, 1.0)))
    # ---------------------------------------------------------------- part 2: lopsided stakes
    q, r = plan.stakes_snr, 1.0
    rules = _rule_table(plan)
    tick(0.45, "Lopsided stakes: tuning on training paths")
    xt, yt = _paths(q, r, plan.train_reps, plan.periods, plan.seed + 100)
    train = simulate_rules(xt, yt, q, r, rules, plan.burn_in)
    tick(0.7, "Lopsided stakes: evaluating on test paths")
    xe, ye = _paths(q, r, plan.test_reps, plan.periods, plan.seed + 200)
    test = simulate_rules(xe, ye, q, r, rules, plan.burn_in)
    srows2, crows = [], []
    fam = rules["family"].to_numpy()
    for rho in plan.rhos:
        tr = _loss(train, rho).mean(1)
        te = _loss(test, rho)
        for f in FAMILY_LABELS:
            mask = fam == f
            if f == "filter_offset":
                mask &= rules["rho"].to_numpy() == rho
            idx = np.flatnonzero(mask)
            j = int(idx[np.argmin(tr[idx])])
            row = rules.iloc[j]
            crows.append(dict(rho=rho, family=f, phi_up=row["phi_up"], phi_down=row["phi_down"],
                              th_up=None if row["th_up"] <= NEVER / 2 else row["th_up"],
                              th_down=None if row["th_down"] <= NEVER / 2 else row["th_down"],
                              offset=row["offset"], train_loss=float(tr[j])))
            for p in range(plan.test_reps):
                srows2.append(dict(rho=rho, family=f, path=p, loss=float(te[j, p]), up_rate=float(test["up"][j, p]),
                                   down_rate=float(test["down"][j, p]),
                                   move_rate=float(test["up"][j, p] + test["down"][j, p])))
    tick(1.0, "Done")
    res = TrackResult(verify=pd.DataFrame(vrows), verify_summary=pd.DataFrame(srows), stakes=pd.DataFrame(srows2),
                      chosen=pd.DataFrame(crows), verdicts=pd.DataFrame())
    res.verdicts = evaluate_tracking(res, plan)
    return res


# ================================================================================================ analysis
def _boot(x: np.ndarray, n_boot: int, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = rng.choice(x, (n_boot, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def gains(res: TrackResult, plan: TrackPlan, base: str = "filter") -> pd.DataFrame:
    """Per rho and family: mean test loss reduction relative to `base` (positive = better than base), with CI."""
    w = res.stakes.pivot_table(index=["rho", "path"], columns="family", values="loss")
    rows = []
    for rho, d in w.groupby(level=0):
        for f in FAMILY_LABELS:
            m, lo, hi = _boot((d[base] - d[f]).to_numpy(), plan.n_boot)
            rows.append(dict(rho=rho, family=f, gain=m, lo=lo, hi=hi, base_loss=float(d[base].mean())))
    return pd.DataFrame(rows)


def evaluate_tracking(res: TrackResult, plan: TrackPlan) -> pd.DataFrame:
    v = res.verify_summary
    ok1 = bool((v["max_rel_gap"] < 0.05).all() and
               (np.abs(v["simulated_best"] - v["kalman_gain"]) <= v["grid_step"] + 1e-9).all())
    t1 = "; ".join(f"q/r = {r.snr:g}: Kalman gain {r.kalman_gain:.3f}, simulated best {r.simulated_best:.2f}, "
                   f"largest gap {r.max_rel_gap:.1%}" for r in v.itertuples())
    g = gains(res, plan)
    sym = g[g["rho"] == 1.0].set_index("family")
    ok2 = bool(not (sym.loc["restrict", "lo"] > 0) and not (sym.loc["speed_asym", "lo"] > 0))
    t2 = (f"restricted − filter gain {sym.loc['restrict', 'gain']:+.4f} [{sym.loc['restrict', 'lo']:+.4f}, "
          f"{sym.loc['restrict', 'hi']:+.4f}]; speeds up/down {sym.loc['speed_asym', 'gain']:+.4f} "
          f"[{sym.loc['speed_asym', 'lo']:+.4f}, {sym.loc['speed_asym', 'hi']:+.4f}]")
    rs = g[(g["family"] == "restrict") & (g["rho"] > 1)]
    w = res.stakes.pivot_table(index=["rho", "path"], columns="family", values="loss").reset_index()
    w = w[w["rho"] > 1]
    tab, _ = ols((w["filter"] - w["restrict"]).to_numpy(float), [np.log(w["rho"].to_numpy(float))], ["ln_rho"])
    b, p = float(tab.loc[1, "coef"]), float(tab.loc[1, "p"])
    ok3 = bool((rs["lo"] > 0).all() and b > 0 and p < plan.alpha)
    t3 = "; ".join(f"ρ = {r.rho:g}: {r.gain:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]" for r in rs.itertuples()) + \
        f"; slope on ln ρ {b:+.4f} (p = {p:.3g})"
    g4 = gains(res, plan, base="filter_offset")
    r4 = g4[g4["family"] == "restrict"]
    ok4 = bool(not (r4["lo"] > 0).any())
    t4 = "; ".join(f"ρ = {r.rho:g}: {r.gain:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]" for r in r4.itertuples())
    v_ = [(ok1, t1), (ok2, t2), (ok3, t3), (ok4, t4)]
    return pd.DataFrame([dict(id=h[0], hypothesis=h[1], decision_rule=h[2],
                              verdict="supported" if ok else "not supported", result=txt)
                         for h, (ok, txt) in zip(plan.hypotheses, v_)])
