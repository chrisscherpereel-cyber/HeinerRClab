"""Agent track: every theory's own agent competes in every condition of the Competing-theories experiments.

The directional experiments (heiner_abm.theories) use the market model's firms (production rule plus flexibility φ),
so every theory is judged by what it predicts about them. The agent track asks the complementary question: when each
theory's own agent competes in the same market, how does it do in every experimental condition? Every theory enters
with its registered tuned design and parameters (one agent per theory, plus rule B as the rigid benchmark), faces the
same market, information and random draws (common random numbers across conditions), and each experiment changes
only the manipulated variable.

This is an exploratory analysis run on the page; it reports no registered numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .analysis import ols
from .params import Scenario


def _sign(coef: float, p: float, alpha: float = 0.05) -> str:
    if p is None or np.isnan(p) or p >= alpha:
        return "0"
    return "+" if coef > 0 else "-"


@dataclass(frozen=True)
class AgentExperiment:
    key: str
    title: str
    variable: str            # arena.Env field, or "copies" (agents per theory)
    label: str               # axis label
    levels: tuple
    hid: str
    directional: str = ""    # the directional experiment it parallels, if any


AGENT_EXPERIMENTS: List[AgentExperiment] = [
    AgentExperiment("profit", "A1 · Industry profitability falls (max cost rises)", "c_max", "Max raw-material cost",
                    (60.0, 70.0, 80.0, 90.0), "H2", "free_harm"),
    AgentExperiment("volatility", "A2 · Cost volatility Δ rises", "delta", "Cost volatility Δ",
                    (2.0, 8.0, 16.0, 24.0, 32.0), "H3", "volatility"),
    AgentExperiment("competence", "A3 · Competence κ (cost foresight) rises", "foresight", "Cost foresight κ",
                    (0.0, 0.25, 0.5, 0.75, 1.0), "H7", "competence"),
    AgentExperiment("noise", "A4 · Perception noise σ rises", "noise", "Perception noise σ",
                    (0.0, 5.0, 10.0, 15.0, 20.0), "H8", "noise"),
    AgentExperiment("rivals", "A5 · More rivals (one, then two agents per theory)", "copies", "Agents per theory",
                    (1, 2), "H11", "rivals"),
    AgentExperiment("shifts", "A6 · Unannounced demand-regime shifts (Knightian uncertainty)", "hazard",
                    "Shift probability λ", (0.0, 0.02, 0.05), "KNIGHT"),
]
AGENT_EXPERIMENT_BY_KEY = {e.key: e for e in AGENT_EXPERIMENTS}


def agent_env(base: Scenario, seed: int, **over):
    """An arena environment taken from the sidebar's base scenario (the arena fixes the demand intercept, price
    floor and initial cost at the baseline calibration)."""
    from .arena import C0, P_MIN, Env
    st = base.structural
    f0 = base.firms[0] if base.firms else None
    e = dict(delta=float(base.market.delta), c_max=float(base.market.c_max), q_range=float(base.market.q_range),
             noise=float(f0.noise) if f0 else 0.0, foresight=float(f0.foresight) if f0 else 0.0,
             hazard=float(st.hazard) if st.enabled else 0.0, belief_lag=int(st.belief_lag) if st.enabled else 20)
    e.update(over)
    e["c_max"] = max(e["c_max"], C0 + 1.0)
    e["delta"] = min(e["delta"], e["c_max"] - P_MIN)
    return Env(seed=seed, **e)


def agent_track(base: Scenario, reps: int, periods: int = 600, burn_in: int = 50, tuned=None,
                experiments: Optional[Sequence[str]] = None,
                progress: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
    """Every theory's agent in every condition of every agent experiment. One row per (experiment, level,
    replication, theory): mean profit per period of the theory's agent(s), its advantage over rule B in the same
    market, and its profit rank among all entries (1 = best)."""
    from .arena import KEYS, simulate
    from .registered import registered_tuned
    tuned = tuned or registered_tuned()
    exps = [AGENT_EXPERIMENT_BY_KEY[k] for k in (experiments or [e.key for e in AGENT_EXPERIMENTS])]
    designs = [tuned.design[k] for k in KEYS]
    rows = []
    for j, ex in enumerate(exps):
        if progress:
            progress(j / len(exps), ex.title)
        for lv in ex.levels:
            copies = int(lv) if ex.variable == "copies" else 1
            over = {} if ex.variable == "copies" else {ex.variable: lv}
            envs = [agent_env(base, base.seed + r, **over) for r in range(reps)]
            lineup = np.array([[d for d in designs for _ in range(copies)]] * reps, dtype=object)
            prof = simulate(envs, lineup, tuned.params, periods, burn_in)["profit"]
            per = prof.reshape(reps, len(KEYS), copies).mean(2)                 # (reps, theories incl. rule B)
            rank = (-per).argsort(1).argsort(1) + 1
            ruleb = per[:, KEYS.index("ruleb")]
            for r in range(reps):
                for i, k in enumerate(KEYS):
                    rows.append(dict(experiment=ex.key, level=float(lv), rep=r, theory=k, design=designs[i],
                                     profit=per[r, i], advantage=per[r, i] - ruleb[r], rank=int(rank[r, i])))
    if progress:
        progress(1.0, "done")
    return pd.DataFrame(rows)


def agent_scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    """Per theory, over every condition of every agent experiment: mean profit rank, conditions won (best mean
    profit), share of conditions in which it beats rule B, and its mean advantage over rule B."""
    from .arena import THEORY_NAMES
    cond = df.groupby(["experiment", "level", "theory"]).agg(profit=("profit", "mean"), adv=("advantage", "mean"),
                                                             rank=("rank", "mean")).reset_index()
    best = cond.loc[cond.groupby(["experiment", "level"])["profit"].idxmax(), "theory"].value_counts()
    out = cond.groupby("theory").agg(mean_rank=("rank", "mean"), beats_ruleb=("adv", lambda x: float((x > 0).mean())),
                                     advantage=("adv", "mean"), conditions=("rank", "size")).reset_index()
    out["wins"] = out["theory"].map(best).fillna(0).astype(int)
    out["name"] = out["theory"].map(THEORY_NAMES)
    out.loc[out["theory"] == "ruleb", "beats_ruleb"] = np.nan
    return out.sort_values(["mean_rank", "wins"], ascending=[True, False], ignore_index=True)[
        ["theory", "name", "mean_rank", "wins", "beats_ruleb", "advantage", "conditions"]]


def agent_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """Per experiment and theory: how the agent's advantage over rule B changes with the manipulated variable
    (OLS slope across markets, two-sided p), read as '+', '-' or '0' at the 5% level."""
    rows = []
    for (ex, k), g in df[df["theory"] != "ruleb"].groupby(["experiment", "theory"], sort=False):
        if g["level"].nunique() < 2:
            continue
        tab, _ = ols(g["advantage"].to_numpy(float), [g["level"].to_numpy(float)], ["level"])
        coef, p = float(tab.loc[1, "coef"]), float(tab.loc[1, "p"])
        rows.append(dict(experiment=ex, theory=k, slope=coef, p=p, observed=_sign(coef, p)))
    return pd.DataFrame(rows)
