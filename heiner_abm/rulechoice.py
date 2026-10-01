"""Endogenous rule choice: firms switch between decision rules according to recent performance (Brock & Hommes 1997).

Every firm carries all candidate rules at once. Each rule observes the firm's actual output and proposes an output
every period, so a firm can switch rules without losing state. The firm executes the proposal of its current rule.
Each period a firm revises its rule with probability `revision`, choosing rule k with logit probability

    P(k) = exp(beta * U_k / s) / sum_j exp(beta * U_j / s),

where U_k is the rule's fitness (exponentially weighted average profit of the firms in the market currently using it,
minus a per-period cost of the rule), s is a running scale of profits (so that beta is dimensionless), and beta is the
intensity of choice: beta = 0 means random choice, large beta means everyone moves to the best-performing rule.

The question is Heiner's: do populations drift toward restricted, rule-governed behaviour as uncertainty rises, as
something that emerges rather than something imposed? Market-level outcomes (price volatility, distance from the
Cournot-Nash price) link rule choice to how far markets stay from equilibrium.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import arena
from .analysis import ols
from .arena import ALL_DESIGNS, P_MIN, Q_MIN, Env, Market, Tuned, cluster_ci, nash_output

RULES = ("ruleb", "options_p", "heiner_p", "opt_br", "cobweb_p", "heur_markup")
RESTRICTED = ("ruleb", "options_p", "heiner_p")          # keep the default unless a condition is met
RULE_LABELS = {"ruleb": "Rule B (keep output)", "options_p": "Inaction band (real options)",
               "heiner_p": "Reliability condition (Heiner)", "opt_br": "Filtered best reply (optimisation)",
               "cobweb_p": "Adaptive price expectations (cobweb)", "heur_markup": "Target-margin rule (heuristic)"}

RC_HYPOTHESES = (
    ("E1", "Emergent restriction: with intensity of choice above zero, the share of firms using restricted rules "
           "rises with uncertainty.",
     "OLS of the end-of-run restricted share on the uncertainty level, pooled over the positive intensities of "
     "choice. Supported if the slope is positive with p < α."),
    ("E2", "Emergent predictability: the share of periods in which firms change output falls with uncertainty.",
     "OLS of the population change rate on the uncertainty level, pooled over the positive intensities. Supported if "
     "the slope is negative with p < α."),
    ("E3", "Selection, not chance: the restricted share at the highest uncertainty is higher with intensity of choice "
           "than without it.",
     "Difference in the restricted share at the highest uncertainty level, largest intensity − zero intensity, 95% "
     "bootstrap CI over replications. Supported if the CI lies above 0."),
    ("E4", "Rule-governed markets stay away from equilibrium: across runs, a larger restricted share goes with a "
           "larger distance between the market price and the Cournot–Nash price.",
     "OLS of the mean |P − P_Nash| on the restricted share, controlling for the uncertainty level. Supported if the "
     "coefficient is positive with p < α."),
)


@dataclass(frozen=True)
class ChoicePlan:
    version: str = "1.0"
    tournament_plan: str = ""
    levels: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)     # uncertainty level u in [0, 1]
    betas: Tuple[float, ...] = (0.0, 1.0, 4.0, 16.0)
    reps: int = 8
    n_firms: int = 12
    periods: int = 1500
    burn_in: int = 100
    revision: float = 0.05
    memory: float = 0.95
    rule_costs: Tuple[Tuple[str, float], ...] = ()
    seed: int = 4004
    alpha: float = 0.05
    n_boot: int = 1000
    hypotheses: Tuple[Tuple[str, str, str], ...] = RC_HYPOTHESES

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "code_sha256": choice_code_digest()}, sort_keys=True, indent=1)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def choice_code_digest() -> str:
    src = "".join(inspect.getsource(o) for o in (level_env, simulate_choice, run_choice_study, evaluate_choice))
    return hashlib.sha256((src + arena.code_digest()).encode()).hexdigest()


def level_env(u: float, seed: int) -> Env:
    """One environment at uncertainty level u in [0, 1]: cost volatility, perception noise and unannounced demand
    shifts all rise together with u (each source can be studied separately on the Mechanisms page)."""
    return Env(delta=2.0 + 28.0 * u, c_max=80.0, q_range=1500.0, noise=8.0 * u, foresight=0.0, hazard=0.04 * u,
               belief_lag=20, seed=seed)


def simulate_choice(envs: Sequence[Env], params: Dict[str, Dict[str, float]], beta: np.ndarray, n_firms: int,
                    periods: int, burn_in: int, revision: float = 0.05, memory: float = 0.95,
                    rule_costs: Optional[Dict[str, float]] = None, rules: Sequence[str] = RULES,
                    keep_path: bool = False, keep_market: bool = False, rng_seed: Optional[int] = None
                    ) -> Dict[str, np.ndarray]:
    """Run B markets of n_firms firms that choose among `rules` (beta: intensity of choice per market). With a single
    rule and revision = 0 this is a homogeneous market of that rule; keep_market returns the full price, quantity,
    cost and output paths."""
    B, N, K = len(envs), n_firms, len(rules)
    beta = np.broadcast_to(np.asarray(beta, float), (B,))
    cost = np.array([(rule_costs or {}).get(k, 0.0) for k in rules])
    mk = Market(envs, N, periods)
    q0 = np.array([nash_output(e.q_range, N) for e in envs], float)
    q = np.repeat(q0[:, None], N, axis=1)
    Q = q.sum(1)
    P = np.maximum(P_MIN, mk.PM[:, 0] - mk.SL[:, 0] * Q)
    mk.P_prev, mk.Q_prev, mk.q_prev = P, Q, q
    mk.profit_prev = (P - mk.cost[:, 0])[:, None] * q
    mk.R_hist[:, 0] = Q[:, None] - q
    mk.q_hist[:, 0] = q
    idx = np.arange(B)
    agents = [[ALL_DESIGNS[k](idx, i, {n: np.full(B, float(params.get(k, {}).get(n, s.default)))
                                       for n, s in ALL_DESIGNS[k].SPACE.items()}, mk) for i in range(N)]
              for k in rules]
    rng = np.random.default_rng([int(envs[0].seed) if rng_seed is None else rng_seed, 99, B])
    choice = rng.integers(0, K, (B, N))
    U = np.zeros((B, K))
    scale = np.maximum(np.abs(mk.profit_prev).mean(1), 1.0)
    restricted = np.isin(np.array(rules), RESTRICTED)
    share_path = np.zeros((B, periods))
    rule_share = np.zeros((B, K))
    changes, n_rec = np.zeros(B), 0
    prices, dist_nash, dist_comp = [], np.zeros(B), np.zeros(B)
    for t in range(1, periods):
        mk.t = t
        c_prev, c_now = mk.cost[:, t - 1], mk.cost[:, t]
        mk.c_hat = (c_prev + mk.kappa * (c_now - c_prev))[:, None] + mk.sigma[:, None] * mk.eps[:, t, :]
        props = np.empty((K, B, N))
        for k in range(K):
            for i in range(N):
                props[k, :, i] = agents[k][i].act()
        new = np.take_along_axis(props, choice[None], axis=0)[0]
        new = np.maximum(Q_MIN, np.rint(np.nan_to_num(new, nan=Q_MIN, posinf=Q_MIN, neginf=Q_MIN)))
        Q = new.sum(1)
        P = np.maximum(P_MIN, mk.PM[:, t] - mk.SL[:, t] * Q)
        profit = (P - c_now)[:, None] * new
        mk.R_hist[:, t] = Q[:, None] - new
        mk.q_hist[:, t] = new
        changed = (new != mk.q_prev)
        mk.P_prev, mk.Q_prev, mk.q_prev, mk.profit_prev = P, Q, new, profit
        for k in range(K):
            for i in range(N):
                agents[k][i].update()
        # fitness: average profit of the current users of each rule, minus its cost
        users = choice[:, :, None] == np.arange(K)[None, None, :]                  # (B, N, K)
        cnt = users.sum(1)
        mean_k = np.where(cnt > 0, (profit[:, :, None] * users).sum(1) / np.maximum(cnt, 1), 0.0) - cost
        U = np.where(cnt > 0, memory * U + (1 - memory) * mean_k, U)
        scale = 0.99 * scale + 0.01 * np.maximum(np.abs(profit).mean(1), 1.0)
        rev = rng.random((B, N)) < revision
        if rev.any():
            z = beta[:, None] * U / scale[:, None]
            pz = np.exp(z - z.max(1, keepdims=True))
            pz /= pz.sum(1, keepdims=True)
            draw = (np.cumsum(pz, 1)[:, None, :] < rng.random((B, N, 1))).sum(2)
            choice = np.where(rev, np.minimum(draw, K - 1), choice)
        share_path[:, t] = restricted[choice].mean(1)
        if t >= periods - (periods - burn_in) // 3:          # final third: the population's settled state
            rule_share += users.mean(1)
            changes += changed.mean(1)
            pn = np.maximum(P_MIN, (mk.PM[:, t] + N * c_now) / (N + 1))
            dist_nash += np.abs(P - pn)
            dist_comp += P - c_now
            prices.append(P)
            n_rec += 1
    prices = np.array(prices).T
    out = dict(rule_share=rule_share / n_rec, restricted_share=(rule_share / n_rec)[:, restricted].sum(1),
               change_rate=changes / n_rec, price_sd=prices.std(1), dist_nash=dist_nash / n_rec,
               margin=dist_comp / n_rec, share_path=share_path if keep_path else None)
    if keep_market:
        Qp = mk.q_hist.sum(2)
        out.update(price=np.maximum(P_MIN, mk.PM - mk.SL * Qp), quantity=Qp, cost=mk.cost, p_max=mk.PM, slope=mk.SL,
                   q=mk.q_hist.copy())
    return out


@dataclass
class ChoiceResult:
    plan_hash: str
    runs: pd.DataFrame
    paths: Dict[Tuple[float, float], np.ndarray]
    verdicts: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_choice_study(plan: ChoicePlan, tuned: Tuned, progress: Optional[Callable[[float, str], None]] = None
                     ) -> ChoiceResult:
    rows, paths = [], {}
    costs = dict(plan.rule_costs)
    for b_i, beta in enumerate(plan.betas):
        if progress:
            progress(b_i / len(plan.betas), f"Intensity of choice {beta:g}")
        envs = [level_env(u, plan.seed + 1000 * j + r) for j, u in enumerate(plan.levels) for r in range(plan.reps)]
        out = simulate_choice(envs, tuned.params, beta, plan.n_firms, plan.periods, plan.burn_in, plan.revision,
                              plan.memory, costs, keep_path=True)
        k = 0
        for j, u in enumerate(plan.levels):
            paths[(beta, u)] = out["share_path"][k:k + plan.reps].mean(0)
            for r in range(plan.reps):
                rows.append(dict(beta=beta, level=u, rep=r, restricted_share=out["restricted_share"][k],
                                 change_rate=out["change_rate"][k], price_sd=out["price_sd"][k],
                                 dist_nash=out["dist_nash"][k], margin=out["margin"][k],
                                 **{f"share_{rk}": out["rule_share"][k, i] for i, rk in enumerate(RULES)}))
                k += 1
    if progress:
        progress(1.0, "Done")
    runs = pd.DataFrame(rows)
    res = ChoiceResult(plan_hash=plan.digest, runs=runs, paths=paths)
    res.verdicts = evaluate_choice(runs, plan)
    return res


def evaluate_choice(runs: pd.DataFrame, plan: ChoicePlan) -> pd.DataFrame:
    pos = runs[runs["beta"] > 0]
    t1, _ = ols(pos["restricted_share"].to_numpy(float), [pos["level"].to_numpy(float)], ["level"])
    e1 = (float(t1.loc[1, "coef"]), float(t1.loc[1, "p"]))
    t2, _ = ols(pos["change_rate"].to_numpy(float), [pos["level"].to_numpy(float)], ["level"])
    e2 = (float(t2.loc[1, "coef"]), float(t2.loc[1, "p"]))
    top = max(plan.levels)
    a = runs[(runs["level"] == top) & (runs["beta"] == max(plan.betas))]["restricted_share"].to_numpy(float)
    z = runs[(runs["level"] == top) & (runs["beta"] == 0)]["restricted_share"].to_numpy(float)
    rng = np.random.default_rng(0)
    boots = [rng.choice(a, len(a)).mean() - rng.choice(z, len(z)).mean() for _ in range(plan.n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    t4, _ = ols(runs["dist_nash"].to_numpy(float), [runs["restricted_share"].to_numpy(float),
                                                    runs["level"].to_numpy(float)], ["restricted", "level"])
    e4 = (float(t4.loc[1, "coef"]), float(t4.loc[1, "p"]))
    v = [("supported" if e1[0] > 0 and e1[1] < plan.alpha else "not supported",
          f"slope {e1[0]:+.3f} per unit of uncertainty (p = {e1[1]:.3g})"),
         ("supported" if e2[0] < 0 and e2[1] < plan.alpha else "not supported",
          f"slope {e2[0]:+.3f} per unit of uncertainty (p = {e2[1]:.3g})"),
         ("supported" if lo > 0 else "not supported", f"{a.mean() - z.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"),
         ("supported" if e4[0] > 0 and e4[1] < plan.alpha else "not supported",
          f"coefficient {e4[0]:+.2f} per unit of restricted share (p = {e4[1]:.3g})")]
    return pd.DataFrame([dict(id=h[0], hypothesis=h[1], decision_rule=h[2], verdict=x[0], result=x[1])
                         for h, x in zip(plan.hypotheses, v)])


def default_choice_plan(tournament_plan: str) -> ChoicePlan:
    return ChoicePlan(tournament_plan=tournament_plan)


QUICK_CHOICE = dict(levels=(0.0, 0.5, 1.0), betas=(0.0, 4.0), reps=3, n_firms=8, periods=400, burn_in=50,
                    n_boot=200)
