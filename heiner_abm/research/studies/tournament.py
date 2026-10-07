"""Agent tournament (heiner_abm.arena): the frozen plan's main run and its replications with fresh seeds.

Stages per run (main run unprefixed, replication r prefixed rep<r>_), each saved separately so an interrupted run
resumes at the next stage:
    splits        training and test environments (the assignment of environments to tuning and evaluation)
    tuning        tuned parameters and the selected design per theory, and the full tuning history
    head_to_head  one firm per theory in every held-out market (per market and agent: six criteria)
    factorial     target x selection-rule markets
    invasion      resident populations with one mutant
    analysis      ranking, pairwise tests, selection-rule effects, invasion matrix, sensitivity, verdicts
Then, once: agents (information and feedback specification of every design), trace (decision-level trace of the
selected line-up in the first held-out markets) and summary (main run and replications side by side).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ... import arena as A
from ..store import Stage, Study
from ..traces import agent_spec, trace_markets
from .common import env_table, sources, tuned_from

SRC = sources(__file__, "arena.py")
TRACE_SRC = sources(__file__, "arena.py", "research/traces.py")


def default_config() -> Dict[str, Any]:
    return dict(prereg=asdict(A.PREREG), replications=3, trace_markets=2, trace_periods=None)


def prereg(cfg: Dict[str, Any], rep: int = 0) -> A.Prereg:
    pr = A.prereg_from_dict(cfg["prereg"])
    if rep:
        d = asdict(pr)
        d.update(train_seed=pr.train_seed + 100 * rep, test_seed=pr.test_seed + 100 * rep)
        pr = A.prereg_from_dict(d)
    return pr


def _protocol(prefix: str, rep: int) -> List[Stage]:
    def splits(ctx):
        pr = prereg(ctx.config, rep)
        ctx.seed("train_seed", pr.train_seed)
        ctx.seed("test_seed", pr.test_seed)
        ctx.seed("environment_seed_bases", dict(train=100_000, test=200_000, test_replication_offset=50_000))
        ctx.note("plan_hash", pr.digest)
        ctx.note("frozen_plan_hash", A.PREREG.digest)
        ctx.note("exploratory", pr.digest != A.PREREG.digest)
        return dict(prereg=dict(plan=asdict(pr), digest=pr.digest, code_sha256=A.code_digest()),
                    environments=pd.concat([env_table(A.train_envs(pr), "train"),
                                            env_table(A.test_envs(pr), "test")], ignore_index=True))

    def tuning(ctx):
        pr = prereg(ctx.config, rep)
        tuned, log = A.tune(pr, ctx.progress)
        ctx.seed("candidate_streams", "numpy default_rng([train_seed, round, design index]) per design and round")
        return dict(tuned=dict(design=tuned.design, params=tuned.params), tuning_history=log)

    def _tuned(ctx):
        return tuned_from(ctx.load(f"{prefix}tuning", "tuned"))

    def h2h(ctx):
        return dict(head_to_head=A.head_to_head(prereg(ctx.config, rep), _tuned(ctx)))

    def fac(ctx):
        return dict(factorial=A.factorial(prereg(ctx.config, rep), _tuned(ctx)))

    def inv(ctx):
        return dict(invasion=A.invasion(prereg(ctx.config, rep), _tuned(ctx), ctx.progress))

    def analysis(ctx):
        pr = prereg(ctx.config, rep)
        h = ctx.load(f"{prefix}head_to_head", "head_to_head")
        f = ctx.load(f"{prefix}factorial", "factorial")
        i = ctx.load(f"{prefix}invasion", "invasion")
        return dict(ranking=A.ranking(h, pr), pairwise=A.pairwise(h, "heiner", pr),
                    factorial_criteria=A.criteria(f, "design", pr), selection=A.selection_effects(f, pr),
                    invasion_matrix=A.invasion_matrix(i, pr), robustness=A.robustness(h),
                    verdicts=A.evaluate(pr, h, i, f))

    t = f"{prefix}tuning"
    return [
        Stage(f"{prefix}splits", splits, SRC, role="splits", description="training and held-out test environments"),
        Stage(t, tuning, SRC, (f"{prefix}splits",), "tuning", "equal-budget tuning and design selection"),
        Stage(f"{prefix}head_to_head", h2h, SRC, (t,), description="mixed markets, one firm per theory"),
        Stage(f"{prefix}factorial", fac, SRC, (t,), description="target x selection-rule markets"),
        Stage(f"{prefix}invasion", inv, SRC, (t,), description="invasion markets"),
        Stage(f"{prefix}analysis", analysis, SRC,
              (f"{prefix}head_to_head", f"{prefix}factorial", f"{prefix}invasion"), "analysis",
              "frozen decision rules and criteria"),
    ]


def stages(cfg: Dict[str, Any]) -> List[Stage]:
    out = [Stage("agents", lambda ctx: dict(agents=agent_spec(list(A.DESIGNS))), TRACE_SRC, role="inputs",
                 description="information and feedback specification of every design")]
    out += _protocol("", 0)
    for r in range(1, int(cfg["replications"]) + 1):
        out += _protocol(f"rep{r}_", r)

    def trace(ctx):
        pr = prereg(ctx.config)
        tuned = tuned_from(ctx.load("tuning", "tuned"))
        k = int(ctx.config["trace_markets"])
        envs = A.test_envs(pr)[:k]
        T = int(ctx.config.get("trace_periods") or pr.periods)
        lineup = np.tile(np.array(tuned.lineup(), dtype=object), (k, 1))
        res = trace_markets(envs, lineup, tuned.params, T, pr.burn_in)
        ctx.note("markets", [e.seed for e in envs])
        tr = res["trace"]
        tr["theory"] = tr["slot"].map(dict(enumerate(A.KEYS)))
        return dict(trace=tr)

    def summary(ctx):
        rows, sel, ver = [], [], []
        n = int(ctx.config["replications"])
        for r in range(n + 1):
            p = f"rep{r}_" if r else ""
            rk = ctx.load(f"{p}analysis", "ranking")
            for x in rk.itertuples():
                rows.append(dict(replication=r, key=x.key, design=x.design, design_name=x.design_name,
                                 mean_rank=x.mean_rank, aggregate_rank=x.aggregate_rank, rel_profit=x.rel_profit,
                                 lo=x.lo, hi=x.hi, cvar5=x.cvar5, survived=x.survived, pareto=x.pareto))
            sel.append(ctx.load(f"{p}analysis", "selection").assign(replication=r))
            ver.append(ctx.load(f"{p}analysis", "verdicts").assign(replication=r))
        return dict(ranks=pd.DataFrame(rows), selection=pd.concat(sel, ignore_index=True),
                    verdicts=pd.concat(ver, ignore_index=True))

    out.append(Stage("trace", trace, TRACE_SRC, ("tuning",), "trace",
                     "decision-level trace of the selected line-up in the first held-out markets"))
    out.append(Stage("summary", summary, SRC,
                     tuple(f"rep{r}_analysis" if r else "analysis" for r in range(int(cfg["replications"]) + 1)),
                     "analysis", "main run and replications side by side"))
    return out


STUDY = Study("tournament", "Agent tournament (frozen plan, main run and replications)", "simulation",
              "Every theory's agents compete in held-out mixed markets after equal-budget tuning.",
              default_config, stages)
