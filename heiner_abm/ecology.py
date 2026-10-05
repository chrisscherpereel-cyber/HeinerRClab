"""Organizational ecology as agents, and selection through exit and entry (Hannan & Freeman 1977, 1984).

Organizational ecology explains industry change mainly by *selection* rather than adaptation: organizations are
structurally inert, poorly fitted ones fail, and new organizations take their place. Two agent designs implement the
theory, in the same market and with the same information as every arena agent:

    eco_inert   Structural inertia: the firm keeps its founding output for as long as it survives. At founding it
                sets output to a multiple f of the output it enters with (f is tuned).
    eco_reorg   Inertia with rare reorganization: the firm keeps its output, but when its recent performance falls
                into crisis (an exponentially weighted profit below zero) and at least R periods have passed since
                its last reorganization, it reorganizes by jumping to its best reply on the believed demand curve.

Both designs are tuned with the arena's equal budget (Latin-hypercube search on the training environments, in a
mixed market against every theory's registered tuned design), so they enter on the same terms as the other theories.

Selection: every firm starts with the arena's survival buffer (CAPITAL_PERIODS periods of Nash profit) and
accumulates profit. A firm whose capital turns negative exits. An entrant takes its slot with the incumbents' mean
output and a fresh buffer. The entrant's form is the design of a randomly drawn surviving firm in that market
(selection of forms), or with probability `mutation` a design drawn uniformly from all candidate forms. Every
candidate design runs in every slot every period on the slot's actual history, so an entrant starts with a warm
state, as in the rule-choice study.

This module does not change the registered tournament: the ecology designs are kept out of arena.DESIGNS and the
study has its own plan hash.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import arena
from .analysis import ols
from .arena import (ALL_DESIGNS, CAPITAL_PERIODS, KEYS, P_MIN, PREREG, Q_MIN, THEORY_NAMES, Agent, Env, Market,
                    Prereg, Spec, Tuned, _lhs, cluster_ci, nash_output, nash_profit, train_envs)
from .rulechoice import level_env


# ================================================================================================ agents
class EcoInert(Agent):
    key, name, theory = "eco_inert", "Structural inertia", "Organizational ecology"
    rule = ("Keeps its founding output for as long as it survives; at founding it sets output to f times the output "
            "it enters with. Adaptation happens through selection (exit and entry), not within the firm.")
    sources = ("hannan1984",)
    SPACE = {"f": Spec(1.0, 0.6, 1.4, help="Founding output as a multiple of the entry output")}

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.qf = None

    def enter(self, rows: np.ndarray):
        """A new firm of this design is founded in markets `rows` (positions within idx)."""
        if self.qf is not None:
            self.qf[rows] = np.maximum(Q_MIN, np.rint(self.q[rows] * self.p["f"][rows]))

    def act(self):
        if self.qf is None:
            self.qf = np.maximum(Q_MIN, np.rint(self.q * self.p["f"]))
        return self.qf.copy()


class EcoReorg(Agent):
    key, name, theory = "eco_reorg", "Inertia with crisis reorganization", "Organizational ecology"
    rule = ("Keeps its output; when its exponentially weighted profit falls below zero and at least R periods have "
            "passed since its last reorganization, it reorganizes by jumping to its best reply on the believed demand "
            "curve.")
    sources = ("hannan1984", "cyert1963")
    SPACE = {"R": Spec(50, 5, 300, "int", "Minimum periods between reorganizations"),
             "memory": Spec(0.9, 0.5, 0.99, help="Memory of the crisis indicator (weighted profit)")}

    def __init__(self, idx, slot, p, mk):
        super().__init__(idx, slot, p, mk)
        self.perf = np.zeros(self.n)
        self.since = np.zeros(self.n)

    def enter(self, rows):
        self.perf[rows] = 0.0
        self.since[rows] = 0.0

    def act(self):
        lam = self.p["memory"]
        self.perf = lam * self.perf + (1 - lam) * self.profit
        self.since += 1
        crisis = (self.perf < 0) & (self.since >= self.p["R"])
        tgt = self.mk.br(self.idx, self.c_hat, self.rivals)
        self.since = np.where(crisis, 0.0, self.since)
        return np.where(crisis, tgt, self.q)


ECO_DESIGNS: Dict[str, type] = {d.key: d for d in (EcoInert, EcoReorg)}
DESIGN_POOL: Dict[str, type] = {**ALL_DESIGNS, **ECO_DESIGNS}
ECO_NAME = "Organizational ecology"
INERT_FORMS = ("eco_inert", "eco_reorg", "ruleb")


# ================================================================================================ plan
ECO_HYPOTHESES = (
    ("S1", "Organizational ecology: selection favors inert forms. At every uncertainty level the population share of "
           "inert forms (structural inertia, crisis reorganization, rule B) is higher at the end of the run than at "
           "the start.",
     "Mean change in the inert share at each uncertainty level, 95% bootstrap CI over replications. Supported if "
     "every level's CI lies above 0."),
    ("S2", "Heiner versus organizational ecology: the selection advantage of inert forms grows with uncertainty "
           "(Heiner), rather than being the same everywhere (ecology: 'no volatility gradient').",
     "OLS of the change in the inert share on the uncertainty level. Supported (Heiner's reading) if the slope is "
     "positive with p < α; a flat slope is ecology's reading."),
    ("S3", "Organizational ecology: inert organizations fail less often than flexible ones.",
     "Exits per 1,000 firm-periods, inert forms minus flexible forms, paired by market, 95% cluster-bootstrap CI. "
     "Supported if the CI lies below 0."),
    ("S4", "Head-to-head: organizational ecology's tuned design earns at least as much as the median theory in "
           "held-out mixed markets.",
     "Mean profit rank of the ecology design among all entrants (1 = best) with a 95% cluster-bootstrap CI. "
     "Supported if the CI's lower end is at or below the median rank."),
)


@dataclass(frozen=True)
class EcoPlan:
    version: str = "1.0"
    tournament_plan: str = ""
    budget: int = PREREG.budget
    n_train: int = 12
    n_test: int = 24
    levels: Tuple[float, ...] = (0.0, 0.33, 0.67, 1.0)
    reps: int = 8
    copies: int = 2                 # initial firms per form in the selection runs
    periods: int = 3000
    burn_in: int = 50
    capital_periods: float = float(CAPITAL_PERIODS)
    mutation: float = 0.02
    train_seed: int = 5005
    test_seed: int = 6006
    seed: int = 7007
    alpha: float = 0.05
    n_boot: int = 1000
    hypotheses: Tuple[Tuple[str, str, str], ...] = ECO_HYPOTHESES

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "code_sha256": eco_code_digest()}, sort_keys=True, indent=1)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


QUICK_ECO = EcoPlan(n_train=4, n_test=6, levels=(0.0, 1.0), reps=3, copies=1, periods=600, burn_in=30, budget=6,
                    n_boot=200)


def eco_code_digest() -> str:
    src = "".join(inspect.getsource(o) for o in (EcoInert, EcoReorg, simulate_eco, tune_ecology, head_to_head_eco,
                                                 selection_runs, evaluate_ecology))
    return hashlib.sha256((src + arena.code_digest()).encode()).hexdigest()


def _envs(n: int, seed: int, base: int) -> List[Env]:
    return arena.sample_envs(n, seed, PREREG.ranges, seed_base=base)


# ================================================================================================ simulation
def simulate_eco(envs: Sequence[Env], lineup: np.ndarray, params: Dict[str, Dict[str, np.ndarray]], periods: int,
                 burn_in: int, selection: bool = False, candidates: Optional[Sequence[str]] = None,
                 capital_periods: float = CAPITAL_PERIODS, mutation: float = 0.02, rng_seed: int = 0
                 ) -> Dict[str, np.ndarray]:
    """Run B markets with any designs from the arena or this module. Without selection it reproduces
    arena.simulate exactly (profit, change rate) and also reports ruin (capital ever below zero). With selection,
    failing firms exit and entrants take their slots (see module docstring). lineup: (B, N) design keys."""
    lineup = np.asarray(lineup, dtype=object)
    B, N = lineup.shape
    pool = list(dict.fromkeys(list(candidates or []) + [k for k in np.unique(lineup)]))
    kidx = {k: j for j, k in enumerate(pool)}
    mk = Market(envs, N, periods)
    q0 = np.array([nash_output(e.q_range, N) for e in envs], float)
    q = np.repeat(q0[:, None], N, axis=1)
    Q = q.sum(1)
    P = np.maximum(P_MIN, mk.PM[:, 0] - mk.SL[:, 0] * Q)
    mk.P_prev, mk.Q_prev, mk.q_prev = P, Q, q
    mk.profit_prev = (P - mk.cost[:, 0])[:, None] * q
    mk.R_hist[:, 0] = Q[:, None] - q
    mk.q_hist[:, 0] = q
    typ = np.vectorize(kidx.get)(lineup).astype(int)
    # agents: without selection, as arena.simulate (one object per slot and design present in that slot);
    # with selection, every candidate design in every slot over all markets
    agents: Dict[Tuple[int, str], Agent] = {}
    all_idx = np.arange(B)
    for i in range(N):
        keys_i = pool if selection else list(np.unique(lineup[:, i]))
        for key in keys_i:
            idx = all_idx if selection else np.flatnonzero(lineup[:, i] == key)
            cls = DESIGN_POOL[key]
            given = params.get(key, {})
            p = {n: np.broadcast_to(np.asarray(given.get(n, s.default), float), (B,))[idx].copy()
                 for n, s in cls.SPACE.items()}
            agents[(i, key)] = cls(idx, i, p, mk)
    cap0 = np.array([capital_periods * nash_profit(e.q_range, N) for e in envs])
    capital = np.repeat(cap0[:, None], N, axis=1) + mk.profit_prev
    ruined = np.zeros((B, N), bool)
    nk = len(pool)
    sum_profit = np.zeros((B, N)); changes = np.zeros((B, N))
    prof_k = np.zeros((B, nk)); per_k = np.zeros((B, nk)); exits_k = np.zeros((B, nk))
    types = np.zeros((B, periods, N), dtype=np.int16); types[:, 0] = typ
    rngs = [np.random.default_rng([rng_seed, int(e.seed), 31]) for e in envs]
    for t in range(1, periods):
        mk.t = t
        c_prev, c_now = mk.cost[:, t - 1], mk.cost[:, t]
        mk.c_hat = (c_prev + mk.kappa * (c_now - c_prev))[:, None] + mk.sigma[:, None] * mk.eps[:, t, :]
        new = np.empty((B, N))
        for (i, key), a in agents.items():
            x = a.act()
            if selection:
                m = typ[:, i] == kidx[key]
                new[m, i] = x[m]
            else:
                new[a.idx, a.i] = x
        new = np.maximum(Q_MIN, np.rint(np.nan_to_num(new, nan=Q_MIN, posinf=Q_MIN, neginf=Q_MIN)))
        Q = new.sum(1)
        P = np.maximum(P_MIN, mk.PM[:, t] - mk.SL[:, t] * Q)
        profit = (P - c_now)[:, None] * new
        mk.R_hist[:, t] = Q[:, None] - new
        mk.q_hist[:, t] = new
        if t >= burn_in:
            sum_profit += profit
            changes += new != mk.q_prev
            for j in range(nk):
                m = typ == j
                prof_k[:, j] += np.where(m, profit, 0).sum(1)
                per_k[:, j] += m.sum(1)
        mk.P_prev, mk.Q_prev, mk.q_prev, mk.profit_prev = P, Q, new, profit
        for a in agents.values():
            a.update()
        capital += profit
        dead = capital < 0
        ruined |= dead
        if selection and dead.any():
            for b, i in zip(*np.nonzero(dead)):
                if t >= burn_in:
                    exits_k[b, typ[b, i]] += 1
                alive = [typ[b, j] for j in range(N) if j != i and not dead[b, j]]
                rng = rngs[b]
                new_t = int(rng.integers(0, nk)) if (not alive or rng.random() < mutation) else \
                    int(alive[int(rng.integers(0, len(alive)))])
                typ[b, i] = new_t
                others = [j for j in range(N) if j != i]
                mk.q_prev[b, i] = max(Q_MIN, np.rint(mk.q_prev[b, others].mean()))
                mk.profit_prev[b, i] = 0.0
                capital[b, i] = cap0[b]
                ag = agents[(i, pool[new_t])]
                if hasattr(ag, "enter"):
                    ag.enter(np.array([b]))
            mk.Q_prev = mk.q_prev.sum(1)
        types[:, t] = typ
    n_rec = periods - burn_in
    return dict(profit=sum_profit / n_rec, change_rate=changes / n_rec, ruined=ruined, pool=pool, types=types,
                profit_by_form=prof_k, periods_by_form=per_k, exits_by_form=exits_k)


# ================================================================================================ study steps
def tune_ecology(plan: EcoPlan, tuned: Tuned, progress: Optional[Callable[[float, str], None]] = None
                 ) -> Tuple[str, Dict[str, Dict[str, float]], pd.DataFrame]:
    """Equal-budget tuning of both ecology designs against every theory's registered tuned design (the arena's
    second-round setting), on training environments. Returns the selected design, tuned parameters and a log."""
    envs = _envs(plan.n_train, plan.train_seed, 500_000)
    E, K = len(envs), plan.budget
    base = np.array(tuned.lineup() + ["eco_inert"], dtype=object)
    slot = len(base) - 1
    params_out, log, best = {}, [], (-np.inf, None)
    for j, design in enumerate(ECO_DESIGNS):
        if progress:
            progress(j / len(ECO_DESIGNS), f"Tuning {ECO_DESIGNS[design].name}")
        rng = np.random.default_rng([plan.train_seed, j])
        cand = _lhs(ECO_DESIGNS[design].SPACE, K, rng)
        lineup = np.tile(base, (K * E, 1)); lineup[:, slot] = design
        params = {d: dict(v) for d, v in tuned.params.items()}
        params[design] = {n: np.repeat(v, E) for n, v in cand.items()}
        out = simulate_eco(envs * K, lineup, params, 800, 50)
        score = out["profit"][:, slot].reshape(K, E).mean(1)
        b = int(np.argmax(score))
        params_out[design] = {n: float(v[b]) for n, v in cand.items()}
        log.append(dict(design=design, name=ECO_DESIGNS[design].name, best_score=float(score[b]),
                        default_score=float(score[0]), **{f"param_{n}": float(v[b]) for n, v in cand.items()}))
        if score[b] > best[0]:
            best = (float(score[b]), design)
    return best[1], params_out, pd.DataFrame(log)


def head_to_head_eco(plan: EcoPlan, tuned: Tuned, eco_design: str, eco_params: Dict[str, Dict[str, float]]
                     ) -> pd.DataFrame:
    """Held-out mixed markets: one firm per theory (registered tuned designs) plus the selected ecology design."""
    envs = _envs(plan.n_test, plan.test_seed, 600_000)
    theories = list(KEYS) + ["ecology"]
    lineup = np.tile(np.array(tuned.lineup() + [eco_design], dtype=object), (len(envs), 1))
    params = {**{d: dict(v) for d, v in tuned.params.items()}, **eco_params}
    out = simulate_eco(envs, lineup, params, 800, 50)
    rows = []
    for b, e in enumerate(envs):
        ranks = (-out["profit"][b]).argsort().argsort() + 1
        for i, th in enumerate(theories):
            rows.append(dict(market=b, theory=th, name=ECO_NAME if th == "ecology" else THEORY_NAMES[th],
                             profit=out["profit"][b, i], rank=ranks[i], ruined=bool(out["ruined"][b, i]),
                             change_rate=out["change_rate"][b, i], delta=e.delta, hazard=e.hazard))
    return pd.DataFrame(rows)


def selection_runs(plan: EcoPlan, tuned: Tuned, eco_design: str, eco_params: Dict[str, Dict[str, float]],
                   progress: Optional[Callable[[float, str], None]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Selection dynamics at each uncertainty level (rule-choice levels: volatility, noise and demand shifts rise
    together). Returns (per-run summary, share paths)."""
    forms = tuned.lineup() + [eco_design]
    names = {d: (ECO_NAME if d in ECO_DESIGNS else THEORY_NAMES[t])
             for d, t in zip(forms, list(KEYS) + ["ecology"])}
    params = {**{d: dict(v) for d, v in tuned.params.items()}, **eco_params}
    envs, keys = [], []
    for u in plan.levels:
        for r in range(plan.reps):
            envs.append(level_env(u, plan.seed + 1000 * r + int(round(100 * u))))
            keys.append((u, r))
    lineup = np.tile(np.array(forms * plan.copies, dtype=object), (len(envs), 1))
    if progress:
        progress(0.1, "Selection runs")
    out = simulate_eco(envs, lineup, params, plan.periods, plan.burn_in, selection=True, candidates=forms,
                       capital_periods=plan.capital_periods, mutation=plan.mutation, rng_seed=plan.seed)
    pool, types = out["pool"], out["types"]
    inert = np.isin(np.array(pool), INERT_FORMS)
    rows, paths = [], []
    T = types.shape[1]
    step = max(1, T // 120)
    for b, (u, r) in enumerate(keys):
        start = inert[types[b, plan.burn_in]].mean()
        end = inert[types[b, -1]].mean()
        per = out["periods_by_form"][b]; ex = out["exits_by_form"][b]
        ex_inert = ex[inert].sum() / max(per[inert].sum(), 1) * 1000
        ex_flex = ex[~inert].sum() / max(per[~inert].sum(), 1) * 1000
        rows.append(dict(level=u, rep=r, market=b, inert_start=start, inert_end=end, inert_change=end - start,
                         exits_inert=ex_inert, exits_flexible=ex_flex, total_exits=ex.sum(),
                         **{f"share_{d}": float((types[b, -1] == pool.index(d)).mean()) for d in forms}))
    for u in plan.levels:
        bs = [b for b, (uu, _) in enumerate(keys) if uu == u]
        for t in range(0, T, step):
            for d in forms:
                paths.append(dict(level=u, period=t, form=d, name=names[d],
                                  share=float((types[bs, t] == pool.index(d)).mean())))
    return pd.DataFrame(rows), pd.DataFrame(paths)


def evaluate_ecology(plan: EcoPlan, runs: pd.DataFrame, h2h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # S1: inert share rises at every level
    per_level = []
    for u, g in runs.groupby("level"):
        m, lo, hi, p = cluster_ci(g["inert_change"].to_numpy(), g["market"].to_numpy(), plan.n_boot, 0.95)
        per_level.append((u, m, lo, hi))
    ok = all(lo > 0 for _, _, lo, _ in per_level)
    rows.append(dict(id="S1", supported=ok, estimate=np.mean([m for _, m, _, _ in per_level]),
                     detail="; ".join(f"u={u:g}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}]" for u, m, lo, hi in per_level)))
    # S2: gradient
    tab, r2 = ols(runs["inert_change"].to_numpy(float), [runs["level"].to_numpy(float)], ["level"])
    rows.append(dict(id="S2", supported=bool(tab.loc[1, "coef"] > 0 and tab.loc[1, "p"] < plan.alpha),
                     estimate=float(tab.loc[1, "coef"]),
                     detail=f"slope {tab.loc[1, 'coef']:+.3f} per unit of uncertainty, p = {tab.loc[1, 'p']:.3f}"))
    # S3: inert forms fail less
    d = (runs["exits_inert"] - runs["exits_flexible"]).to_numpy()
    m, lo, hi, p = cluster_ci(d, runs["market"].to_numpy(), plan.n_boot, 0.95)
    rows.append(dict(id="S3", supported=bool(hi < 0), estimate=m,
                     detail=f"inert − flexible exits per 1,000 firm-periods {m:+.2f} [{lo:+.2f}, {hi:+.2f}]"))
    # S4: head-to-head rank
    e = h2h[h2h["theory"] == "ecology"]
    m, lo, hi, p = cluster_ci(e["rank"].to_numpy(float), e["market"].to_numpy(), plan.n_boot, 0.95)
    med = (h2h["theory"].nunique() + 1) / 2
    rows.append(dict(id="S4", supported=bool(lo <= med), estimate=m,
                     detail=f"mean rank {m:.2f} [{lo:.2f}, {hi:.2f}] of {h2h['theory'].nunique()} (median {med:g})"))
    text = {h: s for h, s, _ in plan.hypotheses}
    test = {h: t for h, _, t in plan.hypotheses}
    out = pd.DataFrame(rows)
    out["hypothesis"] = out["id"].map(text)
    out["test"] = out["id"].map(test)
    out["verdict"] = np.where(out["supported"], "supported", "not supported")
    return out


@dataclass
class EcoResult:
    plan_hash: str
    exploratory: bool
    design: str
    params: Dict[str, Dict[str, float]]
    tuning: pd.DataFrame
    head_to_head: pd.DataFrame
    runs: pd.DataFrame
    paths: pd.DataFrame
    verdicts: pd.DataFrame


def run_ecology_study(plan: EcoPlan, tuned: Tuned, registered_hash: Optional[str] = None,
                      progress: Optional[Callable[[float, str], None]] = None) -> EcoResult:
    say = progress or (lambda f, s: None)
    say(0.0, "Tuning the ecology designs (equal budget)")
    design, params, log = tune_ecology(plan, tuned)
    say(0.35, "Head-to-head in held-out mixed markets")
    h2h = head_to_head_eco(plan, tuned, design, params)
    say(0.5, "Selection through exit and entry")
    runs, paths = selection_runs(plan, tuned, design, params)
    say(0.95, "Evaluating hypotheses")
    v = evaluate_ecology(plan, runs, h2h)
    say(1.0, "Done")
    return EcoResult(plan_hash=plan.digest, exploratory=(registered_hash is None or plan.digest != registered_hash),
                     design=design, params=params, tuning=log, head_to_head=h2h, runs=runs, paths=paths, verdicts=v)
