"""Mechanism study (heiner_abm.mechanisms): principle versus implementation of reliability-based restriction.

The background rivals are the tuned designs of a recorded tournament run (config['inputs']['tuned'], checksummed).
Stages: splits (map environments and the independent long runs that estimate the oracle's values), compute (focal
variants, full line-up and oracle values), analysis (effects, decomposition, boundary, types of uncertainty, verdicts,
metamodel fit by theory), trace (decision-level trace of the learned reliability-condition agent on the model-based
target in the first map environments).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ... import arena as A
from ... import mechanisms as M
from ..store import Stage, Study
from ..traces import agent_spec, trace_markets
from .common import env_table, plan_from_dict, plan_to_dict, sources, tuned_from
from .registry_inputs import tournament_input

SRC = sources(__file__, "mechanisms.py")
TRACE_SRC = sources(__file__, "mechanisms.py", "research/traces.py")


def default_config() -> Dict[str, Any]:
    from ...registered import TOURNAMENT_PLAN
    return dict(plan=plan_to_dict(M.default_plan(TOURNAMENT_PLAN)), inputs=dict(tuned=tournament_input()),
                trace_markets=2, trace_periods=None)


def _plan(cfg) -> M.StudyPlan:
    return plan_from_dict(M.StudyPlan, cfg["plan"])


def stages(cfg: Dict[str, Any]) -> List[Stage]:
    def inputs(ctx):
        tuned = ctx.input("tuned")
        plan = _plan(ctx.config)
        ctx.note("plan_hash", plan.digest)
        ctx.note("frozen_plan_hash", M.default_plan(plan.tournament_plan).digest)
        envs = M.map_envs(plan)
        ctx.seed("map_seed", plan.seed)
        ctx.seed("environment_seeds", "300000 + j; oracle long runs 900000 + 300000 + j")
        return dict(tuned=tuned, environments=pd.concat(
            [env_table(envs, "map"), env_table([replace(e, seed=e.seed + 900_000) for e in envs], "oracle_estimation")],
            ignore_index=True), agents=agent_spec(list(A.ALL_DESIGNS)))

    def compute(ctx):
        res = M.run_study(_plan(ctx.config), tuned_from(ctx.load("inputs", "tuned")), ctx.progress)
        return dict(focal=res.focal, theories=res.theories, oracle_values=res.oracle_values)

    def analysis(ctx):
        plan = _plan(ctx.config)
        focal = ctx.load("compute", "focal")
        theories = ctx.load("compute", "theories")
        res = M.StudyResult(plan_hash=plan.digest, exploratory=False, focal=focal, theories=theories,
                            oracle_values=ctx.load("compute", "oracle_values"))
        res.effects = M.effects(focal, plan)
        res.decomposition = M.decomposition(focal, plan)
        res.boundary = M.boundary(focal, plan)
        res.types = M.uncertainty_types(focal, plan)
        res.verdicts = M.evaluate_study(res, plan)
        b = res.boundary
        bounds = pd.DataFrame([dict(target=k[0], variant=k[1], **v) for k, v in b.items() if isinstance(k, tuple)])
        r2 = pd.DataFrame([dict(theory=t, cv_r2=M.fit_metamodel(g, "rel_profit")["r2"])
                           for t, g in theories.groupby("theory", sort=False)])
        return dict(effects=res.effects, decomposition=res.decomposition, boundary=bounds,
                    boundary_pooled=b.get("pooled", {}), boundary_data=b.get("data", pd.DataFrame()),
                    types=res.types, verdicts=res.verdicts, metamodel_r2=r2)

    def trace(ctx):
        plan = _plan(ctx.config)
        tuned = tuned_from(ctx.load("inputs", "tuned"))
        k = int(ctx.config["trace_markets"])
        envs = M.map_envs(plan)[:k]
        d = M.TARGETS["model"]
        tgt = {n: v for n, v in tuned.params[d["always"]].items() if n in A.TARGET_SPACE["model"]}
        rc = {**tgt, **{n: v for n, v in tuned.params[d["rc"]].items() if n in A.SELECT_SPACE["rc"]}}
        lineup = np.tile(np.array(tuned.lineup(), dtype=object), (k, 1))
        lineup[:, A.KEYS.index("heiner")] = d["rc"]
        params = {n: dict(v) for n, v in tuned.params.items()}
        params[d["rc"]] = rc
        res = trace_markets(envs, lineup, params, int(ctx.config.get("trace_periods") or plan.periods), plan.burn_in)
        tr = res["trace"]
        tr["theory"] = tr["slot"].map(dict(enumerate(A.KEYS)))
        ctx.note("focal", f"{d['rc']} (learned reliability condition, model-based target) in the heiner slot")
        return dict(trace=tr)

    return [Stage("inputs", inputs, SRC, role="inputs", description="tuned rivals (from a recorded tournament run), "
                                                                     "map environments, agent specifications"),
            Stage("compute", compute, SRC, ("inputs",), description="focal variants, line-up and oracle values"),
            Stage("analysis", analysis, SRC, ("compute",), "analysis", "frozen decision rules M1-M5 and maps"),
            Stage("trace", trace, TRACE_SRC, ("inputs",), "trace", "decision trace of the learned RC agent")]


STUDY = Study("mechanisms", "Mechanism study: oracle versus learned reliability", "simulation",
              "Separates the value of reliability-based restriction from the cost of learning it.",
              default_config, stages)
