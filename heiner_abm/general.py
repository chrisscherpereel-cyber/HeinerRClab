"""General simulations: fair comparisons of competing theories, all implemented as agents.

Fairness conventions (applied in every design):
  * every agent sees the same information and faces the same market, shocks and costs;
  * replication r uses seed base.seed + r in every environment and line-up (common random numbers);
  * initial output is scaled so total initial capacity equals the paper's 4-firm market (q0 * 4 / N);
  * flexibility is free unless the user sets flexibility costs (the decisive test in the paper);
  * each theory's parameters are literature defaults that the user can change in the sidebar.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .params import Scenario, StructuralParams
from .theories import (GENERAL_THEORIES, THEORIES, AgentSpec, SelectionParams, industry_table, run_theories,
                       theory_table)


@dataclass(frozen=True)
class Environment:
    name: str
    delta: float
    hazard: float = 0.0
    belief_lag: int = 20


STANDARD_ENVS = [Environment("Calm (Δ = 2)", 2.0), Environment("Risk (Δ = 25)", 25.0),
                 Environment("Knightian (Δ = 5, regime shifts λ = 0.05)", 5.0, 0.05)]


def env_scenario(base: Scenario, env: Environment, n_firms: int, seed: int, periods: Optional[int] = None) -> Scenario:
    s = base.copy(seed=seed)
    if periods:
        s.periods = periods
    s.market.delta = min(env.delta, s.market.c_max - s.market.p_min)
    s.structural = replace(base.structural, enabled=env.hazard > 0, hazard=env.hazard, belief_lag=env.belief_lag)
    s.firm_globals.q0 = round(base.firm_globals.q0 * 4 / n_firms)
    return s


def _batch(base, envs, specs, reps, selection=None, periods=None, record=False, candidates=None):
    scns, keys = [], []
    for e in envs:
        for r in range(reps):
            scns.append(env_scenario(base, e, len(specs), base.seed + r, periods))
            keys.append((e.name, r))
    res = run_theories(scns, specs, selection=selection, record=record, candidates=candidates)
    return res, keys


def tournament(base: Scenario, specs: Sequence[AgentSpec], envs: Sequence[Environment], reps: int,
               selection: Optional[SelectionParams] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mixed markets: all listed theories compete in the same market. Returns (theory rows, industry rows)."""
    res, keys = _batch(base, envs, specs, reps, selection)
    tt, it = theory_table(res), industry_table(res)
    k = pd.DataFrame(keys, columns=["environment", "rep"]); k["market"] = np.arange(len(keys))
    return tt.merge(k, on="market"), it.merge(k, on="market")


def homogeneous(base: Scenario, specs: Sequence[AgentSpec], n_firms: int, envs: Sequence[Environment],
                reps: int) -> pd.DataFrame:
    """Single-theory industries: n_firms copies of one theory. Industry outcomes per theory and environment."""
    rows = []
    for sp in specs:
        res, keys = _batch(base, envs, [sp] * n_firms, reps)
        it = industry_table(res)
        it["theory"], it["label"] = sp.theory, sp.name()
        it["environment"] = [k[0] for k in keys]; it["rep"] = [k[1] for k in keys]
        rows.append(it)
    return pd.concat(rows, ignore_index=True)


def selection_dynamics(base: Scenario, specs: Sequence[AgentSpec], envs: Sequence[Environment], reps: int,
                       selection: SelectionParams, periods: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ecological selection: firms that run out of capital exit; entrants' forms are drawn from survivors.
    Returns (population share of each theory over time, per-theory summary)."""
    res, keys = _batch(base, envs, specs, reps, selection=selection, periods=periods)
    T, nk = res.types.shape[1], len(res.theory_keys)
    env_of = np.array([k[0] for k in keys])
    step = max(1, T // 150)
    paths = []
    for e in dict.fromkeys(env_of):
        idx = np.flatnonzero(env_of == e)
        ty = res.types[idx]
        for t in range(0, T, step):
            for j, k in enumerate(res.theory_keys):
                paths.append(dict(environment=e, period=t, theory=k, label=res.templates[k].name(),
                                  share=float((ty[:, t] == j).mean())))
    tt = theory_table(res)
    kk = pd.DataFrame(keys, columns=["environment", "rep"]); kk["market"] = np.arange(len(keys))
    tt = tt.merge(kk, on="market")
    tt["exit_rate"] = tt["exits"] / np.maximum(tt["firm_periods"], 1) * 1000
    final = pd.DataFrame([dict(environment=env_of[b], theory=res.theory_keys[j],
                               final_share=float((res.types[b, -1] == j).mean()),
                               initial_share=float((res.types[b, 0] == j).mean()))
                          for b in range(len(keys)) for j in range(nk)])
    summ = tt.groupby(["environment", "theory", "label"]).agg(
        avg_profit=("avg_profit", "mean"), exits=("exits", "mean"), exit_rate=("exit_rate", "mean")).reset_index()
    summ = summ.merge(final.groupby(["environment", "theory"]).mean().reset_index(), on=["environment", "theory"],
                      how="outer")
    return pd.DataFrame(paths), summ


# ------------------------------------------------------------------------------------------------
# Scorecard: each theory's own claim, tested in a common battery of environments
# ------------------------------------------------------------------------------------------------

def _paired(a: pd.Series, b: pd.Series):
    d = (a - b).dropna()
    if len(d) < 2 or d.std(ddof=1) == 0:
        return d.mean() if len(d) else np.nan, np.nan, np.nan, np.nan
    t, p = stats.ttest_1samp(d, 0.0)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h, p


def scorecard(base: Scenario, specs: Dict[str, AgentSpec], reps: int, envs: Sequence[Environment] = STANDARD_ENVS,
              selection_periods: int = 2000, capital0: float = 20000.0,
              progress: Optional[Callable[[float, str], None]] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run the standard battery and evaluate one falsifiable claim per theory.

    Battery: (1) a tournament of all theories in calm, risky and Knightian environments; (2) the same
    tournament with the biased agent replaced by its debiased twin (paired control); (3) single-theory
    industries; (4) ecological selection with survivor-weighted entry."""
    keys = [k for k in GENERAL_THEORIES if k in specs]
    line = [specs[k] for k in keys]
    calm, risk, knight = envs[0].name, envs[1].name, envs[2].name
    say = progress or (lambda f, s: None)
    say(0.05, "Tournament")
    T, _ = tournament(base, line, envs, reps)
    rel = T.pivot_table(index=["environment", "rep"], columns="theory", values="rel_profit")
    act = T.pivot_table(index=["environment", "rep"], columns="theory", values="avg_absdq")
    say(0.35, "Debiased control")
    if "biases" in specs:
        deb = AgentSpec("biases", {**specs["biases"].params, "trend": 0.0, "loss_aversion": 1.0}, "Debiased twin")
        T2, _ = tournament(base, [deb if k == "biases" else specs[k] for k in keys], envs, reps)
        rel2 = T2.pivot_table(index=["environment", "rep"], columns="theory", values="rel_profit")
    say(0.55, "Single-theory industries")
    homo = homogeneous(base, [specs[k] for k in keys if k in ("neoclassical", "heiner", "ecology", "imitation")],
                       4, envs, max(3, reps // 2))
    say(0.75, "Ecological selection")
    sel = SelectionParams(enabled=True, capital0=capital0, mode="survivors", mutation=0.02)
    paths, selsum = selection_dynamics(base, line, [envs[0], envs[2]], max(4, reps // 2), sel, selection_periods)
    say(0.95, "Scoring")

    rows = []

    def add(theory, claim, stat, est, lo, hi, p, supported, note=""):
        rows.append(dict(theory=theory, label=THEORIES[theory].label, claim=claim, statistic=stat, estimate=est,
                         ci_low=lo, ci_high=hi, p_value=p, verdict="Supported" if supported else "Not supported",
                         note=note))

    def env(name):
        return rel.loc[name]

    # Neoclassical: earns the most in every environment
    if "neoclassical" in keys:
        worst = None
        for e in (calm, risk, knight):
            r = env(e)
            others = [k for k in keys if k != "neoclassical"]
            best_other = r[others].mean().idxmax()
            d, lo, hi, p = _paired(r["neoclassical"], r[best_other])
            if worst is None or d < worst[1]:
                worst = (e, d, lo, hi, p, best_other)
        e, d, lo, hi, p, bo = worst
        add("neoclassical", "Optimising firms earn the most in every environment.",
            f"Hardest case ({e}): profit vs best rival ({THEORIES[bo].label})", d, lo, hi, p, hi >= 0,
            "Supported if, in every environment, the optimiser is not significantly less profitable than the best "
            "rival (95% CI of the difference reaches 0 or above).")
        h = homo[homo["theory"] == "neoclassical"]
        if len(h):
            add("neoclassical", "Markets of optimisers sit at the Nash equilibrium.",
                "Mean |P − P_Nash| in an all-optimiser industry (calm)", h[h["environment"] == calm]["nash_gap"].mean(),
                np.nan, np.nan, np.nan, h[h["environment"] == calm]["nash_gap"].mean() < 1.0,
                "Supported if prices stay within 1 price unit of the Nash price.")
    # Real options: relative gains rise with volatility
    if "real_options" in keys:
        d, lo, hi, p = _paired(env(risk)["real_options"], env(calm)["real_options"])
        add("real_options", THEORIES["real_options"].claim, "Relative profit: risk − calm", d, lo, hi, p, lo > 0)
    # Bayesian: best adaptive updater under risk
    if "bayesian" in keys:
        adaptive = [k for k in ("real_options", "heiner", "satisficing", "imitation", "biases") if k in keys]
        r = env(risk)
        bo = r[adaptive].mean().idxmax()
        d, lo, hi, p = _paired(r["bayesian"], r[bo])
        add("bayesian", THEORIES["bayesian"].claim, f"Risk: profit vs best other adaptive agent ({THEORIES[bo].label})",
            d, lo, hi, p, hi >= 0, "Supported if not significantly below the best other adaptive agent.")
    # Heiner: relative gains rise with uncertainty; constrained behaviour pays under uncertainty
    if "heiner" in keys:
        unc = (env(risk)["heiner"] + env(knight)["heiner"]) / 2
        d, lo, hi, p = _paired(unc, env(calm)["heiner"])
        add("heiner", THEORIES["heiner"].claim, "Relative profit: mean(risk, Knightian) − calm", d, lo, hi, p, lo > 0)
        cors = {}
        for e in (calm, risk, knight):
            a = act.loc[e][keys].mean(); r = rel.loc[e][keys].mean()
            cors[e] = stats.spearmanr(a, r)[0]
        add("heiner", "Across theories, more active (flexible) agents earn less when uncertainty is high, but not "
            "when it is low.", "Spearman ρ(activity, profit): uncertain minus calm",
            np.mean([cors[risk], cors[knight]]) - cors[calm], np.nan, np.nan, np.nan,
            np.mean([cors[risk], cors[knight]]) < 0 and np.mean([cors[risk], cors[knight]]) < cors[calm],
            ", ".join(f"{k.split(' (')[0]}: ρ = {v:.2f}" for k, v in cors.items()))
    # Satisficing: survives at least as well as the optimiser, without maximising
    if "satisficing" in keys and "neoclassical" in keys:
        s = selsum[selsum["environment"] == knight].set_index("theory")
        er_s, er_n = s.loc["satisficing", "exit_rate"], s.loc["neoclassical", "exit_rate"]
        best = rel.groupby(level=0).mean()[keys].max(axis=1).mean()
        mean_s = rel.groupby(level=0).mean()["satisficing"].mean()
        add("satisficing", THEORIES["satisficing"].claim, "Exit rate per 1,000 firm-periods (Knightian): satisficer − optimiser",
            er_s - er_n, np.nan, np.nan, np.nan, (er_s <= er_n) and (mean_s < best),
            f"Satisficer mean relative profit {mean_s:,.0f} vs best theory {best:,.0f}.")
    # Biases: costly vs debiased twin, more so when volatile
    if "biases" in specs:
        d_all, lo, hi, p = _paired(rel["biases"], rel2["biases"])
        d_risk = (rel.loc[risk]["biases"] - rel2.loc[risk]["biases"]).mean()
        d_calm = (rel.loc[calm]["biases"] - rel2.loc[calm]["biases"]).mean()
        add("biases", THEORIES["biases"].claim, "Biased − debiased twin (all environments)", d_all, lo, hi, p,
            hi < 0 and d_risk < d_calm, f"Cost of bias: calm {d_calm:,.0f}, risk {d_risk:,.0f}.")
    # Imitation: at least as good as optimisation under Knightian uncertainty
    if "imitation" in keys and "neoclassical" in keys:
        d, lo, hi, p = _paired(env(knight)["imitation"], env(knight)["neoclassical"])
        add("imitation", THEORIES["imitation"].claim, "Knightian: imitator − optimiser", d, lo, hi, p, hi >= 0)
    # Ecology: inert forms favoured by selection under uncertainty
    if "ecology" in keys:
        s = selsum[selsum["environment"] == knight].set_index("theory")
        gain = s.loc["ecology", "final_share"] - s.loc["ecology", "initial_share"]
        flex_er = s.drop(index="ecology")["exit_rate"].mean()
        add("ecology", THEORIES["ecology"].claim, "Knightian: change in population share of inert firms", gain,
            np.nan, np.nan, np.nan, (gain > 0) and (s.loc["ecology", "exit_rate"] <= flex_er),
            f"Exit rate per 1,000 firm-periods: inert {s.loc['ecology', 'exit_rate']:.2f} vs others {flex_er:.2f}.")
    say(1.0, "Done")
    return pd.DataFrame(rows), dict(tournament=T, homogeneous=homo, selection_paths=paths, selection_summary=selsum)
