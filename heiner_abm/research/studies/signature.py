"""Signature tests by theory (heiner_abm.special): each rival theory's own prediction, tested with its own agents.

The tuned background rivals come from a recorded tournament run (config['inputs']['tuned']). One stage per theory,
then an analysis stage collecting the verdicts and a trace stage (each focal design among the background rivals in
one representative market).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ... import arena as A
from ... import special as S
from ..store import Stage, Study
from ..traces import agent_spec, trace_markets
from .common import sources, tuned_override
from .registry_inputs import tournament_input

SRC = sources(__file__, "special.py")
TRACE_SRC = sources(__file__, "special.py", "research/traces.py")
# the focal design of each signature test (the firm whose behavior the test reads)
FOCAL = dict(optimiser="opt_br", options="cobweb_p", cobweb="cobweb_q", heuristic="heur_markup",
             satisficing="satis_p", rl="rl_softmax", imitation="imit_best", ecology="ecol_crisis")


def default_config() -> Dict[str, Any]:
    return dict(scale="Full", scale_settings=S.SCALES["Full"], theories=[t.key for t in S.SPECIAL_TESTS],
                inputs=dict(tuned=tournament_input()), trace_periods=400)


def stages(cfg: Dict[str, Any]) -> List[Stage]:
    def inputs(ctx):
        if ctx.config["scale_settings"] != S.SCALES[ctx.config["scale"]]:
            raise ValueError("scale_settings differ from the code's definition of this scale")
        return dict(tuned=ctx.input("tuned"),
                    agents=agent_spec(sorted(set(FOCAL.values()) | set(S.BACKGROUND))),
                    tests=[dict(key=t.key, title=t.title, claim=t.claim, setup=t.setup, criterion=t.criterion,
                                sources=list(t.sources)) for t in S.SPECIAL_TESTS])

    def one(key):
        def fn(ctx):
            tuned = ctx.load("inputs", "tuned")
            with tuned_override(tuned["params"]):
                out = S.run_special(key, ctx.config["scale"])
            ctx.seed("environment_seeds", "fixed per test in heiner_abm.special (e.g. 9300 + j)")
            table = out.pop("table")
            return dict(table=table, result={k: v for k, v in out.items()})
        return fn

    def analysis(ctx):
        rows = []
        for k in ctx.config["theories"]:
            r = ctx.load(f"test_{k}", "result")
            rows.append(dict(theory=k, verdict=r["verdict"], ok=r["ok"], result=r["result"]))
        return dict(verdicts=pd.DataFrame(rows))

    def trace(ctx):
        tuned = ctx.load("inputs", "tuned")
        T = int(ctx.config["trace_periods"])
        keys = [k for k in ctx.config["theories"] if k in FOCAL]
        envs = [S._env(9000 + j) for j in range(len(keys))]
        lineup = np.array([[FOCAL[k], *S.BACKGROUND] for k in keys], dtype=object)
        res = trace_markets(envs, lineup, tuned["params"], T, min(100, T // 4))
        tr = res["trace"]
        tr["test"] = tr["market"].map(dict(enumerate(keys)))
        ctx.note("market", "Δ = 10, no noise, no shifts; focal design in slot 0, tuned background rivals")
        return dict(trace=tr)

    out = [Stage("inputs", inputs, SRC, role="inputs", description="tuned rivals and test definitions")]
    out += [Stage(f"test_{k}", one(k), SRC, ("inputs",), description=f"signature test: {k}")
            for k in cfg["theories"]]
    out.append(Stage("analysis", analysis, SRC, tuple(f"test_{k}" for k in cfg["theories"]), "analysis",
                     "verdicts"))
    out.append(Stage("trace", trace, TRACE_SRC, ("inputs",), "trace", "focal designs in a representative market"))
    return out


STUDY = Study("signature", "Signature tests by theory", "simulation",
              "Each rival theory's distinctive prediction, tested with its own agents.", default_config, stages)
