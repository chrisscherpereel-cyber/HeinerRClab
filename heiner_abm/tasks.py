"""Generalisation: the same selection problem in decision tasks outside the market.

Each task has the structure of the market problem:
    * a default: a rule-governed action that ignores most new information (here a slow, long-memory estimate, the
      analogue of a firm that keeps its established output);
    * a flexible alternative: a fast re-estimate that responds to every new observation;
    * a controllable gap between the difficulty of the problem (outcome noise, frequency of unannounced shifts) and the
      agent's competence (observation error, how strongly the flexible rule reacts);
    * every period the agent decides whether to deviate from the default to the flexible action.

Tasks
    inventory  Newsvendor with shifting demand (Arrow, Harris & Marschak 1951; Scarf 1960). Each period the agent sets
               an order level S; unmet demand costs p per unit, leftover stock h per unit. Both rules set S at the
               critical fractile of a smoothed demand forecast, the default with a slow and the flexible rule with a
               fast forecast.
    learning   Two options whose mean payoffs swap without warning (a restless two-armed task with full feedback;
               Behrens et al. 2007). Both rules pick the option with the higher smoothed payoff estimate, the default
               with slow and the flexible rule with fast smoothing.
    investment Irreversible investment (Dixit & Pindyck 1994). Each period one project arrives; its value is the current
               mean project quality (which shifts without warning between booms and busts) plus project-specific
               noise. Investing commits the firm: a project worth V pays V, and a project that turns out bad also costs
               a write-off of half its loss. Not investing pays 0. Both rules invest when the expected payoff at their
               estimate of mean quality is positive, the default with a slow and the flexible rule with a fast
               estimate.

Selection layers, identical in all tasks
    always       always take the flexible action (no restriction)
    band         deviate only when the flexible action differs from the default by more than b times the agent's own
                 noise estimate (inaction band; b tuned on separate training environments)
    rc_learned   reliability condition learned from experience: for each signal-strength bin the agent keeps a running
                 mean of the ex post gain of deviating (computable every period, whether or not it deviated, because
                 both actions' payoffs can be evaluated against the observed outcome; judged with its own noisy
                 observations) and deviates only in bins where that mean is positive
    rc_oracle    the same decision with the true mean gain per bin, estimated from a long, independent run of the same
                 environment (it knows its rule's reliability, not the future)
    ruleb        never deviate from the default

No cost is charged for deviating, so any gain from restriction comes from the reliability of the flexible rule alone.

The boundary test uses the same error-to-signal ratio as the market:
    K = sqrt( E[(flexible_t - optimum_t)^2] / E[(optimum_t - default_t)^2] ),
the flexible rule's error relative to how far the optimum lies from the default. Because the actions of both rules
depend only on what the agent observes, K is a property of the environment and the agent's competence, not of the
selection layer.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .analysis import ols
from .arena import cluster_ci, holm

N_BINS = 5
EDGES = np.array([0.5, 1.0, 2.0, 4.0])
LAYERS = ("always", "band", "rc_learned", "rc_oracle", "ruleb")
LAYER_LABELS = {"always": "Always deviate (flexible rule)", "band": "Inaction band", "rc_learned": "RC, learned",
                "rc_oracle": "RC, oracle", "ruleb": "Rule B (never deviate)"}
TASKS = {"inventory": "Inventory (newsvendor with shifting demand)",
         "learning": "Learning with shifting payoffs (restless two-option task)",
         "investment": "Irreversible investment (projects in shifting booms and busts)"}
RANGES = {
    "inventory": {"noise": (5.0, 40.0), "hazard": (0.002, 0.05), "obs_noise": (0.0, 40.0), "gain": (0.05, 0.6)},
    "learning": {"noise": (0.5, 5.0), "hazard": (0.002, 0.05), "obs_noise": (0.0, 5.0), "gain": (0.05, 0.6)},
    "investment": {"noise": (0.5, 4.0), "hazard": (0.002, 0.05), "obs_noise": (0.0, 4.0), "gain": (0.05, 0.6)},
}
K_CLIP = (0.05, 20.0)       # range of K used in the boundary analysis
WRITE_OFF = 0.5             # extra loss per unit of a bad investment's negative value (irreversibility)
SLOW_GAIN = 0.01            # the default rule's smoothing (memory of about 100 periods)
AXIS_LABELS = {"noise": "Difficulty: outcome noise", "hazard": "Difficulty: shift hazard",
               "obs_noise": "Competence: observation error (higher = less competent)",
               "gain": "Flexibility: gain of the fast estimate (higher = reacts more)"}

TASK_HYPOTHESES = (
    ("G1", "Boundary in every task: the gain from reliability-based restriction rises with the flexible rule's "
           "error-to-signal ratio K.",
     "Per task, OLS of (rc_oracle − always) on ln K across environments; Holm-adjusted over tasks. Supported if the "
     "slope is positive and significant in every task."),
    ("G2", "Restriction pays only where the flexible rule is unreliable: the gain is positive in the top third of K "
           "and not positive in the bottom third, in every task.",
     "Per task and K tercile, mean (rc_oracle − always) with a 95% bootstrap CI. Supported if the top-tercile CI lies "
     "above 0 and the bottom-tercile CI does not, in every task."),
    ("G3", "Cost of applying the principle: knowing the true reliability beats learning it, in every task.",
     "Per task, paired difference rc_oracle − rc_learned; Holm-adjusted over tasks. Supported if significantly "
     "positive in every task."),
    ("G4", "A learnable principle: where the flexible rule is unreliable (top third of K), the learned reliability "
           "condition beats always deviating, in every task.",
     "Per task, mean (rc_learned − always) in the top K tercile, 95% bootstrap CI. Supported if above 0 in every "
     "task."),
)


@dataclass(frozen=True)
class TaskEnv:
    task: str
    noise: float
    hazard: float
    obs_noise: float
    gain: float
    seed: int


@dataclass(frozen=True)
class TaskPlan:
    version: str = "1.1"
    tasks: Tuple[str, ...] = ("inventory", "learning", "investment")
    n_envs: int = 120
    n_train: int = 30
    periods: int = 3000
    burn_in: int = 300
    oracle_periods: int = 30000
    band_grid: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    rc_memory: float = 0.995
    seed: int = 7007
    alpha: float = 0.05
    n_boot: int = 2000
    hypotheses: Tuple[Tuple[str, str, str], ...] = TASK_HYPOTHESES

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "ranges": RANGES, "slow_gain": SLOW_GAIN, "write_off": WRITE_OFF, "k_clip": K_CLIP,
                           "code": task_code_digest()},
                          sort_keys=True, indent=1)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def task_code_digest() -> str:
    mod = sys.modules[__name__]
    src = "".join(inspect.getsource(f) for f in (mod._streams, mod._decisions, mod._layer, mod._env_set,
                                                  mod.run_tasks, mod.inventory_payoff, mod.learning_payoff,
                                                  mod.investment_payoff, mod.expected_investment_payoff))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


QUICK_TASKS = dict(n_envs=30, n_train=10, periods=1000, burn_in=150, oracle_periods=8000, n_boot=200)


def _env_set(task: str, n: int, seed: int) -> List[TaskEnv]:
    rng = np.random.default_rng(seed)
    cols = {}
    for k, (lo, hi) in RANGES[task].items():
        u = (rng.permutation(n) + rng.random(n)) / n
        cols[k] = lo + u * (hi - lo)
    return [TaskEnv(task, *(float(cols[k][j]) for k in ("noise", "hazard", "obs_noise", "gain")), seed=seed * 1000 + j)
            for j in range(n)]


# ================================================================================================ task worlds
P_SHORT, H_OVER = 4.0, 1.0
Z = 0.8416212335729143      # critical fractile p / (p + h) = 0.8


def inventory_payoff(S, d):
    return -(H_OVER * np.maximum(S - d, 0.0) + P_SHORT * np.maximum(d - S, 0.0))


def learning_payoff(a, pay):
    return pay[..., 1] * a + pay[..., 0] * (1 - a)


def investment_payoff(a, v):
    """Invest (a = 1) in a project worth v: v, plus a write-off on a bad project; not investing pays 0."""
    return a * (v + WRITE_OFF * np.minimum(v, 0.0))


def expected_investment_payoff(m, sd):
    """Expected payoff of investing when project value ~ N(m, sd^2)."""
    from math import erf, sqrt
    m = np.asarray(m, float)
    z = m / sd
    Phi_neg = 0.5 * (1 + np.vectorize(erf)(-z / sqrt(2)))
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return m + WRITE_OFF * (m * Phi_neg - sd * pdf)


def _streams(env: TaskEnv, T: int, seed: int) -> Dict[str, np.ndarray]:
    """Exogenous paths: the true state, realised outcomes and what the agent observes."""
    rng = np.random.default_rng(seed)
    jumps = rng.random(T) < env.hazard
    if env.task == "inventory":
        steps = np.where(jumps, rng.normal(0, 30.0, T), 0.0)
        mu = np.empty(T)
        m = 100.0
        for t in range(T):
            m = min(170.0, max(30.0, m + steps[t]))
            mu[t] = m
        d = np.maximum(0.0, mu + env.noise * rng.standard_normal(T))
        return dict(state=mu, outcome=d, obs=d + env.obs_noise * rng.standard_normal(T))
    if env.task == "investment":
        steps = np.where(jumps, rng.normal(0, 1.5, T), 0.0)
        mq = np.clip(0.5 + np.cumsum(steps), -3.0, 3.0)           # mean project quality (booms and busts)
        v = mq + env.noise * rng.standard_normal(T)
        return dict(state=mq, outcome=v, obs=v + env.obs_noise * rng.standard_normal(T))
    adv = np.where(np.cumsum(jumps) % 2 == 0, 1.0, -1.0)        # mean advantage of option 1 over option 0
    pay = np.stack([-adv / 2, adv / 2], 1) + env.noise * rng.standard_normal((T, 2))
    return dict(state=adv, outcome=pay, obs=pay + env.obs_noise * rng.standard_normal((T, 2)))


def _decisions(env: TaskEnv, T: int, seed: int) -> Dict[str, np.ndarray]:
    """Per period: default and flexible action, signal strength, payoffs of both actions (true and as observed), and
    the optimum. None of this depends on the selection layer."""
    s = _streams(env, T, seed)
    obs, g, gs = s["obs"], env.gain, SLOW_GAIN
    if env.task == "inventory":
        slow, fast, var = np.empty(T), np.empty(T), np.empty(T)
        ms, mf, v = 100.0, 100.0, env.noise ** 2 + env.obs_noise ** 2 + 1.0
        for t in range(T):                                      # forecasts made before period t's demand is seen
            slow[t], fast[t], var[t] = ms, mf, v
            o = obs[t]
            v += gs * ((o - ms) ** 2 - v)
            ms += gs * (o - ms)
            mf += g * (o - mf)
        sd = np.sqrt(np.maximum(var - env.obs_noise ** 2, 1.0))
        d_act, x_act = slow + Z * sd, fast + Z * sd
        opt = s["state"] + Z * env.noise
        sig = np.abs(x_act - d_act) / np.sqrt(var)
        pay_d, pay_x = inventory_payoff(d_act, s["outcome"]), inventory_payoff(x_act, s["outcome"])
        obs_gain = inventory_payoff(x_act, obs) - inventory_payoff(d_act, obs)
    elif env.task == "investment":
        slow, fast, var = np.empty(T), np.empty(T), np.empty(T)
        ms, mf, v = 0.5, 0.5, env.noise ** 2 + env.obs_noise ** 2 + 0.1
        for t in range(T):
            slow[t], fast[t], var[t] = ms, mf, v
            o = obs[t]
            v += gs * ((o - ms) ** 2 - v)
            ms += gs * (o - ms)
            mf += g * (o - mf)
        sd = np.sqrt(np.maximum(var - env.obs_noise ** 2, 0.01))      # the agent's estimate of project dispersion
        ev_d, ev_x = expected_investment_payoff(slow, sd), expected_investment_payoff(fast, sd)
        d_act, x_act = (ev_d > 0).astype(float), (ev_x > 0).astype(float)
        opt = (expected_investment_payoff(s["state"], env.noise) > 0).astype(float)
        sig = np.where(x_act != d_act, np.abs(ev_x) / np.sqrt(var), 0.0)
        pay_d, pay_x = investment_payoff(d_act, s["outcome"]), investment_payoff(x_act, s["outcome"])
        obs_gain = investment_payoff(x_act, obs) - investment_payoff(d_act, obs)
    else:
        slow, fast, var = np.empty((T, 2)), np.empty((T, 2)), np.empty(T)
        es, ef, v = np.zeros(2), np.zeros(2), 2 * (env.noise ** 2 + env.obs_noise ** 2) + 1.0
        for t in range(T):
            slow[t], fast[t], var[t] = es, ef, v
            o = obs[t]
            v += gs * (((o[1] - o[0]) - (es[1] - es[0])) ** 2 - v)
            es = es + gs * (o - es)
            ef = ef + g * (o - ef)
        d_act = (slow[:, 1] > slow[:, 0]).astype(float)
        x_act = (fast[:, 1] > fast[:, 0]).astype(float)
        opt = (s["state"] > 0).astype(float)
        sig = np.where(x_act != d_act, np.abs(fast[:, 1] - fast[:, 0]) / np.sqrt(var), 0.0)
        pay_d, pay_x = learning_payoff(d_act, s["outcome"]), learning_payoff(x_act, s["outcome"])
        obs_gain = learning_payoff(x_act, obs) - learning_payoff(d_act, obs)
    return dict(d=d_act, x=x_act, opt=opt, sig=sig, bin=(sig[:, None] > EDGES).sum(1), differ=x_act != d_act,
                pay_d=pay_d, pay_x=pay_x, true_gain=pay_x - pay_d, obs_gain=obs_gain)


def _layer(dec: Dict[str, np.ndarray], layer: str, burn: int, band: float = 1.0, table: Optional[np.ndarray] = None,
           rc_memory: float = 0.995) -> Dict[str, float]:
    T = len(dec["d"])
    if layer == "always":
        dev = dec["differ"].copy()
    elif layer == "ruleb":
        dev = np.zeros(T, bool)
    elif layer == "band":
        dev = dec["differ"] & (dec["sig"] > band)
    elif layer == "rc_oracle":
        dev = dec["differ"] & (table[dec["bin"]] > 0)
    else:                                                      # rc_learned: decide, then learn from period t's outcome
        dev = np.zeros(T, bool)
        lg, lc = np.zeros(N_BINS), np.zeros(N_BINS)
        b_, diff, og = dec["bin"], dec["differ"], dec["obs_gain"]
        for t in range(T):
            if diff[t]:
                b = b_[t]
                dev[t] = lc[b] < 5 or lg[b] > 0
                lc[b] += 1
                lg[b] += (og[t] - lg[b]) * max(1 - rc_memory, 1 / lc[b])
    pay = np.where(dev, dec["pay_x"], dec["pay_d"])[burn:]
    w = slice(burn, None)
    K = np.sqrt(np.mean((dec["x"][w] - dec["opt"][w]) ** 2) / max(np.mean((dec["opt"][w] - dec["d"][w]) ** 2), 1e-12))
    return dict(payoff=float(pay.mean()), deviation_rate=float(dev[w].mean()), K=float(K))


def oracle_table(env: TaskEnv, plan: "TaskPlan") -> np.ndarray:
    """True mean gain of deviating per signal bin, from a long independent run of the same environment."""
    dec = _decisions(env, plan.oracle_periods, env.seed + 777_001)
    m = dec["differ"]
    g = np.bincount(dec["bin"][m], dec["true_gain"][m], N_BINS)
    c = np.bincount(dec["bin"][m], minlength=N_BINS)
    return np.where(c > 0, g / np.maximum(c, 1), -np.inf)


@dataclass
class TaskResult:
    runs: pd.DataFrame          # one row per task × environment × layer
    bands: Dict[str, float]
    verdicts: pd.DataFrame


def run_tasks(plan: TaskPlan, progress: Optional[Callable[[float, str], None]] = None) -> TaskResult:
    rows, bands = [], {}
    total = len(plan.tasks) * (plan.n_train + plan.n_envs)
    done = 0

    def tick(msg):
        nonlocal done
        done += 1
        if progress:
            progress(min(done / total, 1.0), msg)

    for ti, task in enumerate(plan.tasks):
        # the band width is tuned on separate training environments
        score = np.zeros(len(plan.band_grid))
        for env in _env_set(task, plan.n_train, plan.seed + 100 + ti):
            dec = _decisions(env, plan.periods, env.seed)
            score += [_layer(dec, "band", plan.burn_in, band=b)["payoff"] for b in plan.band_grid]
            tick(f"{TASKS[task]}: tuning the band")
        bw = bands[task] = float(plan.band_grid[int(np.argmax(score))])
        for env in _env_set(task, plan.n_envs, plan.seed + ti):
            table = oracle_table(env, plan)
            dec = _decisions(env, plan.periods, env.seed)
            for layer in LAYERS:
                r = _layer(dec, layer, plan.burn_in, band=bw, table=table, rc_memory=plan.rc_memory)
                rows.append(dict(task=task, env=env.seed, noise=env.noise, hazard=env.hazard, obs_noise=env.obs_noise,
                                 gain=env.gain, layer=layer, **r))
            tick(f"{TASKS[task]}: environment {env.seed % 1000 + 1}")
    res = TaskResult(runs=pd.DataFrame(rows), bands=bands, verdicts=pd.DataFrame())
    res.verdicts = evaluate_tasks(res, plan)
    if progress:
        progress(1.0, "Done")
    return res


# ================================================================================================ analysis
def wide(runs: pd.DataFrame, task: str) -> pd.DataFrame:
    d = runs[runs["task"] == task]
    w = d.pivot(index="env", columns="layer", values="payoff")
    k = d[d["layer"] == "always"].set_index("env")["K"]
    # K is clipped to [0.05, 20]: with binary actions it is 0 or infinite when one rule is never wrong over a run
    w["lnK"] = np.log(k.clip(lower=K_CLIP[0], upper=K_CLIP[1]))
    for c in ("noise", "hazard", "obs_noise", "gain"):
        w[c] = d[d["layer"] == "always"].set_index("env")[c]
    w["tercile"] = pd.qcut(w["lnK"].rank(method="first"), 3, labels=["low K", "middle K", "high K"])
    return w


def _boot(x: np.ndarray, n_boot: int, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan
    b = rng.choice(x, (n_boot, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boundary(runs: pd.DataFrame, task: str) -> Dict:
    w = wide(runs, task)
    out = {}
    for v in ("rc_oracle", "rc_learned", "band"):
        y = (w[v] - w["always"]).to_numpy(float)
        tab, r2 = ols(y, [w["lnK"].to_numpy(float)], ["lnK"])
        a, b = float(tab.loc[0, "coef"]), float(tab.loc[1, "coef"])
        out[v] = dict(intercept=a, slope=b, p=float(tab.loc[1, "p"]), r2=r2,
                      K_star=float(np.exp(-a / b)) if b > 0 else np.nan)
    return out


def tercile_table(runs: pd.DataFrame, task: str, n_boot: int = 2000) -> pd.DataFrame:
    w = wide(runs, task)
    rows = []
    for terc, d in w.groupby("tercile", observed=True):
        for v in ("rc_oracle", "rc_learned", "band", "ruleb"):
            m, lo, hi = _boot((d[v] - d["always"]).to_numpy(), n_boot)
            rows.append(dict(task=task, tercile=str(terc), K_range=f"{np.exp(d['lnK'].min()):.2f}–"
                             f"{np.exp(d['lnK'].max()):.2f}", layer=v, gain=m, lo=lo, hi=hi, n=len(d)))
    return pd.DataFrame(rows)


def evaluate_tasks(res: TaskResult, plan: TaskPlan) -> pd.DataFrame:
    runs = res.runs
    tasks = list(plan.tasks)
    b = {t: boundary(runs, t) for t in tasks}
    terc = {t: tercile_table(runs, t, plan.n_boot) for t in tasks}
    verdicts = []
    # G1
    rej = holm([b[t]["rc_oracle"]["p"] for t in tasks], plan.alpha)
    ok = all(b[t]["rc_oracle"]["slope"] > 0 and rej[i] for i, t in enumerate(tasks))
    verdicts.append(("supported" if ok else "not supported",
                     "; ".join(f"{t}: slope {b[t]['rc_oracle']['slope']:+.3g} per ln K (p = {b[t]['rc_oracle']['p']:.3g}"
                               f"{', Holm-significant' if rej[i] else ''}), K* ≈ {b[t]['rc_oracle']['K_star']:.2f}"
                               for i, t in enumerate(tasks))))

    def cell(t, terc_name, layer):
        d = terc[t]
        return d[(d["tercile"] == terc_name) & (d["layer"] == layer)].iloc[0]
    # G2
    parts, ok = [], True
    for t in tasks:
        hi_, lo_ = cell(t, "high K", "rc_oracle"), cell(t, "low K", "rc_oracle")
        ok &= hi_["lo"] > 0 and not lo_["lo"] > 0
        parts.append(f"{t}: high K {hi_['gain']:+.3g} [{hi_['lo']:+.3g}, {hi_['hi']:+.3g}], low K {lo_['gain']:+.3g} "
                     f"[{lo_['lo']:+.3g}, {lo_['hi']:+.3g}]")
    verdicts.append(("supported" if ok else "not supported", "; ".join(parts)))
    # G3
    ps, parts = [], []
    for t in tasks:
        w = wide(runs, t)
        m, lo, hi, p = cluster_ci((w["rc_oracle"] - w["rc_learned"]).to_numpy(float), np.arange(len(w)),
                                  plan.n_boot, 0.95)
        ps.append(p)
        parts.append((t, m, lo, hi, p))
    rej = holm(ps, plan.alpha)
    ok = all(m > 0 and rej[i] for i, (_, m, _, _, _) in enumerate(parts))
    verdicts.append(("supported" if ok else "not supported",
                     "; ".join(f"{t}: {m:+.3g} [{lo:+.3g}, {hi:+.3g}] (p = {p:.3g}"
                               f"{', Holm-significant' if rej[i] else ''})" for i, (t, m, lo, hi, p) in enumerate(parts))))
    # G4
    parts, ok = [], True
    for t in tasks:
        c = cell(t, "high K", "rc_learned")
        ok &= c["lo"] > 0
        parts.append(f"{t}: {c['gain']:+.3g} [{c['lo']:+.3g}, {c['hi']:+.3g}]")
    verdicts.append(("supported" if ok else "not supported", "; ".join(parts)))
    return pd.DataFrame([dict(id=h[0], hypothesis=h[1], decision_rule=h[2], verdict=v[0], result=v[1])
                         for h, v in zip(plan.hypotheses, verdicts)])
