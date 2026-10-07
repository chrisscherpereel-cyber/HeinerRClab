"""Endogenous rule choice (heiner_abm.rulechoice): firms switch between six tuned designs by recent performance.

Stages: inputs (tuned designs of a recorded tournament run; the environments of every level and replication),
compute (one row per intensity of choice, uncertainty level and replication; rule-share paths), analysis (frozen
decision rules E1-E4; mean shares and rates by intensity and level with 95% intervals), trace (period-by-period rule
shares of each market, the decision-level record this population model keeps).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ... import rulechoice as R
from ..store import Stage, Study
from ..traces import agent_spec, trace_choice
from .common import env_table, plan_from_dict, plan_to_dict, sources, tuned_from
from .registry_inputs import tournament_input

SRC = sources(__file__, "rulechoice.py")
TRACE_SRC = sources(__file__, "rulechoice.py", "research/traces.py")


def default_config() -> Dict[str, Any]:
    from ...registered import TOURNAMENT_PLAN
    return dict(plan=plan_to_dict(R.default_choice_plan(TOURNAMENT_PLAN)), inputs=dict(tuned=tournament_input()))


def _plan(cfg) -> R.ChoicePlan:
    return plan_from_dict(R.ChoicePlan, cfg["plan"])


def stages(cfg: Dict[str, Any]) -> List[Stage]:
    def inputs(ctx):
        plan = _plan(ctx.config)
        ctx.note("plan_hash", plan.digest)
        ctx.note("frozen_plan_hash", R.default_choice_plan(plan.tournament_plan).digest)
        ctx.seed("environment_seeds", f"{plan.seed} + 1000 * level index + replication")
        envs = [R.level_env(u, plan.seed + 1000 * j + r) for j, u in enumerate(plan.levels) for r in range(plan.reps)]
        return dict(tuned=ctx.input("tuned"), environments=env_table(envs, "evaluation"),
                    agents=agent_spec(list(R.RULES)))

    def compute(ctx):
        res = R.run_choice_study(_plan(ctx.config), tuned_from(ctx.load("inputs", "tuned")), ctx.progress)
        paths = pd.DataFrame([dict(beta=b, level=u, period=t, restricted_share=p[t])
                              for (b, u), p in res.paths.items() for t in range(len(p))])
        return dict(runs=res.runs, share_paths=paths)

    def analysis(ctx):
        plan = _plan(ctx.config)
        runs = ctx.load("compute", "runs")
        cols = [c for c in runs.columns if c not in ("beta", "level", "rep")]
        g = runs.groupby(["beta", "level"])[cols]
        mean, sd, n = g.mean(), g.std(), g.count()
        ci = (1.96 * sd / np.sqrt(n)).add_suffix("_ci95")
        return dict(verdicts=R.evaluate_choice(runs, plan), by_level=pd.concat([mean, ci], axis=1).reset_index())

    def trace(ctx):
        plan = _plan(ctx.config)
        tuned = tuned_from(ctx.load("inputs", "tuned"))
        beta = max(plan.betas)
        envs = [R.level_env(u, plan.seed + 1000 * j + r) for j, u in enumerate(plan.levels) for r in range(plan.reps)]
        rec = [0, len(envs) - 1]                    # lowest and highest uncertainty level, first/last replication
        res = trace_choice(envs, tuned.params, beta, plan.n_firms, plan.periods, plan.burn_in, plan.revision,
                           plan.memory, dict(plan.rule_costs), R.RULES, rec)
        ctx.note("traced", dict(beta=beta, markets=rec, levels=[plan.levels[0], plan.levels[-1]]))
        tr = res["trace"]
        tr["level"] = tr["market"].map(lambda b: plan.levels[b // plan.reps])
        return dict(trace=tr, rule_share_paths=ctx.load("compute", "share_paths"))

    return [Stage("inputs", inputs, SRC, role="inputs", description="tuned designs and environments"),
            Stage("compute", compute, SRC, ("inputs",), description="rule-choosing markets"),
            Stage("analysis", analysis, SRC, ("compute",), "analysis", "frozen decision rules E1-E4"),
            Stage("trace", trace, TRACE_SRC, ("inputs", "compute"), "trace",
                  "firm-level rule choices and proposals in two markets; rule-share paths")]


STUDY = Study("rule_choice", "Rule choice (emergence)", "simulation",
              "Firms choose between restricted and flexible rules by recent performance.", default_config, stages)
