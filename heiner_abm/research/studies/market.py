"""Market-lab studies (heiner_abm.engine): the directional tournament, the forecasting horse race and the presets.

directional  every theory's directional prediction scored on nine experiments, per production rule and seed
horse_race   out-of-sample forecasts of which firms beat their rigid twin (AUC) and the encompassing test
presets      typical results of the preset scenarios shown in the sidebar
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ... import scenarios as SC
from ..store import Stage, Study
from ..traces import trace_engine
from .common import sources

DIR_SRC = sources(__file__, "theories.py", "scenarios.py")
HR_SRC = sources(__file__, "experiments.py", "scenarios.py")
PRE_SRC = sources(__file__, "engine.py", "analysis.py", "scenarios.py")
TRACE_SRC = sources(__file__, "engine.py", "scenarios.py", "research/traces.py")


def _model_spec() -> List[Dict[str, Any]]:
    """Objective, information, actions, feedback, assumptions and limits of every agent (heiner_abm.model_spec)."""
    from ...model_spec import all_specs
    return [s.row() for s in all_specs()]


# ================================================================================================ directional
def directional_config() -> Dict[str, Any]:
    return dict(markets=[dict(rule="Bertrand", flex_slope=0.25, seed=1), dict(rule="Bertrand", flex_slope=0.25, seed=3),
                         dict(rule="Cournot", flex_slope=0.1, seed=1), dict(rule="Cournot", flex_slope=0.1, seed=42)],
                settings=dict(SC.DEFAULTS), reps=20, horizon=20)


def _dir_scn(cfg, m):
    return SC.build_scenario({**cfg["settings"], **{k: m[k] for k in ("rule", "flex_slope", "seed")}})


def directional_stages(cfg):
    from ... import theories as TH

    def inputs(ctx):
        return dict(theories=[asdict(t) for t in TH.THEORIES],
                    experiments=[dict(key=e.key, title=e.title, manipulation=e.manipulation, statistic=e.statistic,
                                      predictions=e.predictions, why=e.why, hid=e.hid) for e in TH.EXPERIMENTS],
                    agents=_model_spec())

    def one(m):
        def fn(ctx):
            scn = _dir_scn(ctx.config, m)
            ctx.seed("base_seed", m["seed"])
            ctx.seed("replication_seeds", "experiments.run_sweep: base seed + replication")
            out = TH.run_tournament(scn, int(ctx.config["reps"]), horizon=int(ctx.config["horizon"]),
                                    progress=ctx.progress)
            rows = [dict(experiment=k, statistic=o.stat, p=o.p, observed=o.observed, summary=o.summary)
                    for k, o in out.items()]
            res = dict(outcomes=pd.DataFrame(rows))
            for k, o in out.items():
                res[f"detail__{k}"] = o.detail.reset_index(drop=True)
            return res
        return fn

    def label(m):
        return f"{m['rule']}_{m['seed']}"

    def analysis(ctx):
        rows, grid = [], []
        for m in ctx.config["markets"]:
            o = ctx.load(f"market_{label(m)}", "outcomes")
            obs = dict(zip(o["experiment"], o["observed"].where(o["observed"].notna(), None)))
            for t in TH.THEORIES:
                res = [TH.score(e.predictions.get(t.key), obs.get(e.key)) for e in TH.EXPERIMENTS]
                made = [r for r in res if r is not None]
                mt, ct, it = made.count("match"), made.count("contradicted"), made.count("inconclusive")
                rows.append(dict(rule=m["rule"], seed=m["seed"], theory=t.key, name=t.name, predictions=len(made),
                                 matches=mt, contradicted=ct, inconclusive=it, net=mt - ct,
                                 record=f"{mt}/{ct}/{it}"))
                grid += [dict(rule=m["rule"], seed=m["seed"], theory=t.key, experiment=e.key, score=r)
                         for e, r in zip(TH.EXPERIMENTS, res)]
        return dict(records=pd.DataFrame(rows), scores=pd.DataFrame(grid))

    def trace(ctx):
        frames = []
        for m in ctx.config["markets"]:
            frames.append(trace_engine(_dir_scn(ctx.config, m), 1).assign(rule=m["rule"], seed=m["seed"]))
        return dict(trace=pd.concat(frames, ignore_index=True))

    out = [Stage("inputs", inputs, DIR_SRC, role="inputs", description="theories, predictions and agents")]
    out += [Stage(f"market_{label(m)}", one(m), DIR_SRC, ("inputs",), description=f"nine experiments: {label(m)}")
            for m in cfg["markets"]]
    out.append(Stage("analysis", analysis, DIR_SRC, tuple(f"market_{label(m)}" for m in cfg["markets"]), "analysis",
                     "each theory's record (matches / contradictions / inconclusive)"))
    out.append(Stage("trace", trace, TRACE_SRC, (), "trace", "base scenario of each market, every firm and period"))
    return out


DIRECTIONAL = Study("directional", "Competing theories: directional tournament", "simulation",
                    "Every theory's predicted sign on nine experiments in the market model.", directional_config,
                    directional_stages)


# ================================================================================================ horse race
def horse_config() -> Dict[str, Any]:
    return dict(settings=dict(SC.DEFAULTS), n_env=100, reps=2, rules=["Bertrand", "Cournot"], env_seed=12345,
                horizon=20, continuation="default", discount=1.0, oos_split=0.5, n_boot=300)


def horse_stages(cfg):
    from ... import experiments as X

    def _envs(c):
        return X.random_environments(SC.build_scenario(c["settings"]), int(c["n_env"]),
                                     X.EnvRanges(rules=tuple(c["rules"])), seed=int(c["env_seed"]))

    def splits(ctx):
        ctx.seed("env_seed", ctx.config["env_seed"])
        envs = _envs(ctx.config)
        rows = [dict(env=i, seed=s.seed, rule=s.firms[0].rule, n_firms=s.n_firms, delta=s.market.delta,
                     c_max=s.market.c_max, q_range=s.market.q_range, noise=s.firms[0].noise,
                     foresight=s.firms[0].foresight, hazard=s.structural.hazard if s.structural.enabled else 0.0,
                     estimation_share=ctx.config["oos_split"]) for i, s in enumerate(envs)]
        return dict(environments=pd.DataFrame(rows), windows=dict(
            estimation="first oos_split of recorded periods, ending H − 1 periods early (no overlap)",
            evaluation="remaining recorded periods"))

    def compute(ctx):
        c = ctx.config
        df = X.rc_validation(_envs(c), int(c["reps"]), horizon=int(c["horizon"]), continuation=c["continuation"],
                             discount=float(c["discount"]), oos_split=float(c["oos_split"]))
        return dict(firms=df)

    def analysis(ctx):
        df = ctx.load("compute", "firms")
        out = {}
        for meas in ("full", "static"):
            out[f"auc_{meas}"] = X.horse_race(df, meas, n_boot=int(ctx.config["n_boot"]))
            enc, gain = X.encompassing_test(df, meas, n_boot=int(ctx.config["n_boot"]))
            out[f"encompassing_{meas}"] = enc
            out[f"gain_{meas}"] = gain
        out["base_rate"] = dict(share=float((df["dyn_adv_eval"] > 0).mean()), firms=len(df))
        return out

    def trace(ctx):
        envs = _envs(ctx.config)
        return dict(trace=trace_engine(envs[0], int(ctx.config["horizon"]), ctx.config["continuation"]))

    return [Stage("splits", splits, HR_SRC, role="splits", description="random environments and the windows"),
            Stage("compute", compute, HR_SRC, ("splits",), description="firms and their rigid twins"),
            Stage("analysis", analysis, HR_SRC, ("compute",), "analysis", "AUC by forecast; encompassing test"),
            Stage("trace", trace, TRACE_SRC, ("splits",), "trace", "first environment")]


HORSE_RACE = Study("horse_race", "Competing theories: forecasting horse race", "simulation",
                   "Which theory's forecast identifies the firms whose flexibility beats their rigid twin "
                   "(out of sample within the simulation).", horse_config, horse_stages)


# ================================================================================================ presets
def presets_config() -> Dict[str, Any]:
    return dict(presets=list(SC.PRESETS), reps=6, seed0=1, overrides={})


def preset_stages(cfg):
    from ...analysis import market_table, reliability_table
    from ...engine import run_batch

    def one(name):
        def fn(ctx):
            vals = {**SC.preset_values(name), **ctx.config.get("overrides", {})}
            adaptive = vals["selection"] == "Adaptive"
            H = int(vals["horizon"]) if adaptive else 1           # H affects only Adaptive agents' learning
            scns = [SC.build_scenario({**vals, "seed": int(ctx.config["seed0"]) + r}) for r in range(int(ctx.config["reps"]))]
            ctx.seed("seeds", [s.seed for s in scns])
            ctx.note("horizon", H)
            res = run_batch(scns, horizon=H)
            mt, ft = market_table(res), reliability_table(res)
            step = float(vals["flex_slope"]) or 1.0
            slopes = [float(np.polyfit([f.flex for f in s.firms], ft[ft["market"] == i]["avg_profit"].to_numpy(), 1)[0])
                      * step for i, s in enumerate(scns)]
            adopt = (ft["deviations"] / ft["opportunities"].where(ft["opportunities"] > 0)).mean()
            summary = dict(preset=name, avg_price=float(mt["avg_price"].mean()), avg_cost=float(mt["avg_cost"].mean()),
                           avg_margin=float(mt["avg_margin"].mean()), sd_price=float(mt["sd_price"].mean()),
                           sd_cost=float(mt["sd_cost"].mean()), profit_per_firm=float(ft["avg_profit"].mean()),
                           slope=float(np.mean(slopes)), slope_sd=float(np.std(slopes, ddof=1)),
                           adoption_share=float(adopt), shifts_per_run=float(mt["n_shifts"].mean()),
                           regime=mt["regime"].mode().iat[0], reps=len(scns))
            return dict(markets=mt, firms=ft, summary=summary)
        return fn

    def analysis(ctx):
        return dict(summary=pd.DataFrame([ctx.load(f"preset_{i}", "summary") for i in range(len(ctx.config["presets"]))]))

    def trace(ctx):
        vals = {**SC.preset_values(ctx.config["presets"][0]), **ctx.config.get("overrides", {})}
        return dict(trace=trace_engine(SC.build_scenario({**vals, "seed": int(ctx.config["seed0"])}), 1))

    out = [Stage("inputs", lambda ctx: dict(presets={n: SC.PRESETS[n] for n in ctx.config["presets"]},
                                            defaults=SC.DEFAULTS, agents=_model_spec()),
                 PRE_SRC, role="inputs", description="preset settings")]
    out += [Stage(f"preset_{i}", one(n), PRE_SRC, ("inputs",), description=n) for i, n in enumerate(cfg["presets"])]
    out.append(Stage("analysis", analysis, PRE_SRC, tuple(f"preset_{i}" for i in range(len(cfg["presets"]))),
                     "analysis", "typical result of each preset"))
    out.append(Stage("trace", trace, TRACE_SRC, (), "trace", "baseline preset, first replication"))
    return out


PRESETS = Study("presets", "Preset scenarios: typical results", "simulation",
                "Typical outcomes of the sidebar's preset scenarios.", presets_config, preset_stages)
