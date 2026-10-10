"""Field patterns (heiner_abm.patterns): pattern-oriented validation of the simulated market.

The tuned designs come from a recorded tournament run. Stages: inputs, compute (one row per pattern, with its
criterion and result), trace (firm decisions in the first market of the mixed population).
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List

from ... import arena as A
from ... import patterns as PT
from ..store import Stage, Study
from ..traces import agent_spec, trace_choice
from .common import sources, tuned_from
from .registry_inputs import tournament_input

SRC = sources(__file__, "patterns.py")
TRACE_SRC = sources(__file__, "patterns.py", "research/traces.py")
MIX = ("cobweb_p", "opt_br", "heur_markup", "options_p", "heiner_p", "ruleb")   # the mixed population (patterns.py)


def default_config() -> Dict[str, Any]:
    defaults = {k: v.default for k, v in inspect.signature(PT.run_patterns).parameters.items()
                if k not in ("tuned", "progress")}
    return dict(settings=defaults, inputs=dict(tuned=tournament_input()))


def stages(cfg: Dict[str, Any]) -> List[Stage]:
    def inputs(ctx):
        return dict(tuned=ctx.input("tuned"), agents=agent_spec(list(MIX) + ["imit_best"]),
                    patterns=[dict(key=p.key, name=p.name, evidence=p.evidence, criterion=p.criterion,
                                   population=p.population, sources=list(p.sources)) for p in PT.PATTERNS])

    def compute(ctx):
        s = ctx.config["settings"]
        ctx.seed("environment_seeds", f"{s['seed']} + j")
        df = PT.run_patterns(tuned_from(ctx.load("inputs", "tuned")), progress=ctx.progress, **s)
        df["sources"] = df["sources"].map(lambda x: ";".join(x))
        return dict(patterns=df)

    def trace(ctx):
        s = ctx.config["settings"]
        tuned = tuned_from(ctx.load("inputs", "tuned"))
        envs = [A.Env(**PT.BASE, seed=s["seed"] + j) for j in range(s["n_envs"])]
        # the mixed population of the excess-volatility pattern: no revision, fixed rule assignment (rng seed 12345)
        res = trace_choice_fixed(envs, tuned.params, s["n_firms"], s["periods"], s["burn"])
        return dict(trace=res["trace"])

    return [Stage("inputs", inputs, SRC, role="inputs", description="tuned designs and pattern definitions"),
            Stage("compute", compute, SRC, ("inputs",), description="the six patterns"),
            Stage("trace", trace, TRACE_SRC, ("inputs",), "trace", "mixed-population market decisions")]


def trace_choice_fixed(envs, params, n_firms, periods, burn):
    """The mixed market of patterns._run(..., mixed=True) (no revision, rule assignment from rng seed 12345),
    traced in its first market."""
    return trace_choice(envs, params, 0.0, n_firms, periods, burn, 0.0, 0.95, None, MIX, [0], rng_seed=12345)


STUDY = Study("patterns", "Field patterns", "simulation",
              "Documented field patterns reproduced (or not) by the simulated market.", default_config, stages)
