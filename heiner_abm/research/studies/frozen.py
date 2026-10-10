"""Studies with a frozen plan of their own: tracking benchmark, generalization tasks, learnability, NK landscapes and the
decision benchmarks (inventory, bandit).

Every study saves its plan and hash, its training / validation / test assignment, its tuning history, its results, a
decision-level trace and its frozen verdicts. Where a trace recomputes decisions outside the study's own code, the
stage checks that the recomputed rule gives the study's own aggregate result for that path and stops otherwise.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..store import Stage, Study
from .common import plan_from_dict, plan_to_dict, sources, split_outputs


def _plan_inputs(ctx, plan, frozen):
    ctx.note("plan_hash", plan.digest if hasattr(plan, "digest") else None)
    ctx.note("frozen_plan_hash", frozen.digest if hasattr(frozen, "digest") else None)
    ctx.note("exploratory", getattr(plan, "digest", None) != getattr(frozen, "digest", None))


# ================================================================================================ tracking
def _tracking():
    from ... import tracking as TR
    SRC = sources(__file__, "tracking.py")

    def cfg():
        return dict(plan=plan_to_dict(TR.TrackPlan()), trace_periods=300)

    def plan(c):
        return plan_from_dict(TR.TrackPlan, c["plan"])

    def stages(c):
        def splits(ctx):
            p = plan(ctx.config)
            _plan_inputs(ctx, p, TR.TrackPlan())
            ctx.seed("verification_paths", [p.seed + i for i in range(len(p.snrs))])
            ctx.seed("training_paths", p.seed + 100)
            ctx.seed("test_paths", p.seed + 200)
            return dict(plan=plan_to_dict(p), assignment=pd.DataFrame([
                dict(role="verification", seed=p.seed + i, snr=s, reps=p.verify_reps, periods=p.verify_periods)
                for i, s in enumerate(p.snrs)] + [
                dict(role="training", seed=p.seed + 100, snr=p.stakes_snr, reps=p.train_reps, periods=p.periods),
                dict(role="test", seed=p.seed + 200, snr=p.stakes_snr, reps=p.test_reps, periods=p.periods)]),
                rules=TR._rule_table(p))

        def compute(ctx):
            res = TR.run_tracking(plan(ctx.config), ctx.progress)
            return split_outputs(res)

        def trace(ctx):
            p = plan(ctx.config)
            T = int(ctx.config["trace_periods"])
            q, r = p.stakes_snr, 1.0
            x, y = TR._paths(q, r, 1, T, p.seed + 200)
            k = TR.kalman_gain(q, r)
            m, a, rows = 0.0, 0.0, []
            for t in range(T):
                m = m + k * (y[t, 0] - m)                     # Kalman mean after observing y_t
                d = m - a
                a = a + d                                     # the filter rule: move to the Kalman mean
                rows.append(dict(period=t, obs_signal=y[t, 0], pred_kalman_mean=m, action=a,
                                 feedback_available_period=t + 1, researcher_target=x[t, 0]))
            df = pd.DataFrame(rows)
            rule = pd.DataFrame([dict(family="filter", source="kf", phi_up=1.0, phi_down=1.0, th_up=TR.NEVER,
                                      th_down=TR.NEVER, offset=0.0)])
            sim = TR.simulate_rules(x, y, q, r, rule, 0)
            mse = float(((df["action"] - df["researcher_target"]) ** 2).mean())
            if not np.isclose(mse, float(sim["over"][0, 0] + sim["under"][0, 0]), rtol=1e-12, atol=0):
                raise AssertionError("trace does not reproduce tracking.simulate_rules")
            ctx.note("checked", "squared error of the traced filter equals tracking.simulate_rules on this path")
            return dict(trace=df)

        return [Stage("splits", splits, SRC, role="splits", description="verification, training and test paths"),
                Stage("compute", compute, SRC, ("splits",), description="verification and lopsided stakes"),
                Stage("trace", trace, SRC, ("splits",), "trace", "the Kalman filter rule on one test path")]

    return Study("tracking", "Solvable benchmark: Heiner versus optimal filtering", "analytical",
                 "Reproduces the exact Muth–Kalman loss curve, then tests lopsided stakes.", cfg, stages)


# ================================================================================================ generalization
def _tasks():
    from ... import tasks as TK
    SRC = sources(__file__, "tasks.py", "workbench/environments.py")

    def cfg():
        return dict(plan=plan_to_dict(TK.TaskPlan()))

    def plan(c):
        return plan_from_dict(TK.TaskPlan, c["plan"])

    def stages(c):
        def splits(ctx):
            p = plan(ctx.config)
            _plan_inputs(ctx, p, TK.TaskPlan())
            rows = []
            for ti, task in enumerate(p.tasks):
                rows += [dict(asdict(e), role="training") for e in TK._env_set(task, p.n_train, p.seed + 100 + ti)]
                rows += [dict(asdict(e), role="test") for e in TK._env_set(task, p.n_envs, p.seed + ti)]
            ctx.seed("training_sets", f"{p.seed} + 100 + task index")
            ctx.seed("test_sets", f"{p.seed} + task index")
            return dict(plan=plan_to_dict(p), environments=pd.DataFrame(rows))

        def tuning(ctx):
            p = plan(ctx.config)
            rows = []
            for ti, task in enumerate(p.tasks):
                score = np.zeros(len(p.band_grid))
                for env in TK._env_set(task, p.n_train, p.seed + 100 + ti):
                    dec = TK._decisions(env, p.periods, env.seed)
                    score += [TK._layer(dec, "band", p.burn_in, band=b)["payoff"] for b in p.band_grid]
                rows += [dict(task=task, band=b, train_payoff_sum=s_, selected=bool(j == int(np.argmax(score))))
                         for j, (b, s_) in enumerate(zip(p.band_grid, score))]
            return dict(tuning_history=pd.DataFrame(rows))

        def compute(ctx):
            res = TK.run_tasks(plan(ctx.config), ctx.progress)
            hist = ctx.load("tuning", "tuning_history")
            sel = hist[hist["selected"]].set_index("task")["band"].to_dict()
            if {k: float(v) for k, v in res.bands.items()} != {k: float(v) for k, v in sel.items()}:
                raise AssertionError(f"tuning stage selected {sel}, run_tasks selected {res.bands}")
            return dict(runs=res.runs, bands=res.bands, verdicts=TK.evaluate_tasks(res, plan(ctx.config)))

        def trace(ctx):
            from ...workbench.environments import task_trace
            p = plan(ctx.config)
            bands = ctx.load("compute", "bands")
            frames = []
            for ti, task in enumerate(p.tasks):
                env = TK._env_set(task, p.n_envs, p.seed + ti)[0]
                dec = TK._decisions(env, p.periods, env.seed)
                for layer in ("always", "band", "rc_learned"):
                    tr = task_trace(dec, layer, float(bands[task]), None, p.rc_memory)
                    frames.append(tr.assign(task=task, layer=layer, env_seed=env.seed))
            return dict(trace=pd.concat(frames, ignore_index=True))

        return [Stage("splits", splits, SRC, role="splits", description="training and test environments per task"),
                Stage("tuning", tuning, SRC, ("splits",), "tuning", "band width tuned on training environments"),
                Stage("compute", compute, SRC, ("tuning",), description="every layer in every test environment"),
                Stage("trace", trace, SRC, ("compute",), "trace", "first test environment of each task")]

    return Study("generalization", "Generalization tasks", "simulation",
                 "The same selection layers and boundary test in inventory, learning and investment tasks.", cfg, stages)


# ================================================================================================ learnability
def _learnability():
    from ... import learnability as L
    SRC = sources(__file__, "learnability.py", "learnability_market.py")

    def cfg():
        return dict(plan=plan_to_dict(L.LearnPlan()), market=True, trace_policies=list(L.POLICIES))

    def plan(c):
        return plan_from_dict(L.LearnPlan, c["plan"])

    def stages(c):
        def tuning(ctx):
            p = plan(ctx.config)
            _plan_inputs(ctx, p, L.LearnPlan())
            hp, log = L.tune(p, ctx.progress)
            ctx.seed("training_configurations", p.seed + 1)
            return dict(plan=plan_to_dict(p), hyperparameters=hp, tuning_history=log)

        def compute(ctx):
            res = L.run_study(plan(ctx.config), ctx.progress, market=bool(ctx.config["market"]))
            if res.hyperparameters != ctx.load("tuning", "hyperparameters"):
                raise AssertionError("run_study tuned different hyperparameters than the tuning stage")
            out = split_outputs({k: getattr(res, k) for k in ("sweeps", "tests", "new_families", "controls",
                                                             "verdicts", "paths")})
            if res.market:
                out.update(split_outputs(res.market, "market_"))
            return out

        def trace(ctx):
            p = plan(ctx.config)
            hp = ctx.load("tuning", "hyperparameters")
            cfg_ = L.Config(**dict(p.baseline))
            seed = p.seed + 777
            pth = L.paths(cfg_, p.periods, seed)
            cand = L.candidate_orders(cfg_, pth, hp["gain"])
            frames = []
            for pol in ctx.config["trace_policies"]:
                S, pred, gobs = L.policy_orders(pol, cfg_, pth, cand, hp)
                frames.append(pd.DataFrame(dict(
                    policy=pol, period=np.arange(len(S)), action_order=S,
                    pred_gain=pred if pred is not None else np.nan,
                    feedback_gain=gobs if gobs is not None else np.nan,
                    **{f"obs_{k}": v for k, v in pth.items() if getattr(v, "shape", None) == S.shape})))
            ctx.seed("trace_seed", seed)
            ctx.note("timing", "orders for period t use observations up to t − 1; observed gains are released "
                               "after the judgement window (heiner_abm.learnability)")
            return dict(trace=pd.concat(frames, ignore_index=True))

        return [Stage("tuning", tuning, SRC, role="tuning", description="hyperparameters on training configurations"),
                Stage("compute", compute, SRC, ("tuning",), description="sweeps, tests, new families, controls"),
                Stage("trace", trace, SRC, ("tuning",), "trace", "baseline configuration, every policy")]

    return Study("learnability", "When can reliability be learned before the environment changes?", "simulation",
                 "Learning a reliability gate in a changing environment.", cfg, stages)


# ================================================================================================ NK landscapes
def _nk():
    from ... import nk as NK
    SRC = sources(__file__, "nk.py", "workbench/environments.py")

    def cfg():
        return dict(plan=plan_to_dict(NK.NKPlan()))

    def plan(c):
        return plan_from_dict(NK.NKPlan, c["plan"])

    def stages(c):
        def splits(ctx):
            p = plan(ctx.config)
            rows = [dict(condition=name, **{k: v for k, v in asdict(env).items() if not isinstance(v, (list, tuple))})
                    for name, env in NK.conditions(p)]
            return dict(plan=plan_to_dict(p), conditions=pd.DataFrame(rows))

        def compute(ctx):
            return dict(runs=NK.run_nk(plan(ctx.config), ctx.progress))

        def trace(ctx):
            from ...workbench.environments import NKLandscape
            p = plan(ctx.config)
            name, env = NK.conditions(p)[0]
            path = NK.landscape_path(env)
            st_ = NK.run_streams(env, path, env.seed + 1)
            obs = NK.Observer(env, path, st_.noise, st_.leader, st_.visible)
            frames = []
            for s in NK.SEARCHERS:
                xs, costs, log = NK.simulate(s, env, obs, st_.start, st_.U, NK.SearchParams())
                frames.append(NKLandscape()._trace(env, path, xs, log).assign(searcher=s, condition=name))
            return dict(trace=pd.concat(frames, ignore_index=True))

        return [Stage("splits", splits, SRC, role="splits", description="conditions and landscapes"),
                Stage("compute", compute, SRC, ("splits",), description="every searcher on every landscape"),
                Stage("trace", trace, SRC, ("splits",), "trace", "first condition, every searcher")]

    return Study("nk", "NK landscapes: interaction complexity", "simulation",
                 "Search with reliability learning on rugged, noisy and changing landscapes.", cfg, stages)


# ================================================================================================ benchmarks
def _bench(kind: str):
    if kind == "inventory":
        from ... import bench_inventory as B
        Plan, run, src = B.InventoryPlan, B.run_inventory, "bench_inventory.py"
    else:
        from ... import bench_bandit as B
        Plan, run, src = B.BanditPlan, B.run_bandit, "bench_bandit.py"
    SRC = sources(__file__, src)

    def cfg():
        return dict(plan=plan_to_dict(Plan()))

    def plan(c):
        return plan_from_dict(Plan, c["plan"])

    def stages(c):
        def splits(ctx):
            p = plan(ctx.config)
            sets = p.env_sets()
            return dict(plan=plan_to_dict(p), environments=pd.DataFrame(
                [dict(role=k, **asdict(e)) for k, v in sets.items() for e in v]))

        def compute(ctx):
            res = run(plan(ctx.config), ctx.progress)
            out = {}
            for fam, t in res["results"].items():
                out[f"tuning_history__{fam}"] = t.curve
                out[f"test__{fam}"] = t.test
                out[f"selected__{fam}"] = dict(final=t.final, evaluations=t.evaluations)
            out.update(split_outputs({k: v for k, v in res.items() if k != "results"}))
            return out

        def trace(ctx):
            p = plan(ctx.config)
            env = p.env_sets()["test"][0]
            T = p.periods
            st_ = B._stream(env, T)
            frames = []
            for fam in p.families:
                final = ctx.load("compute", f"selected__{fam}")["final"]
                if kind == "inventory":
                    S = B.orders(env, fam, final, T, p.grid_step)
                    mu, d, y, _ = st_
                    frames.append(pd.DataFrame(dict(family=fam, period=np.arange(T), action_order=S,
                                                    obs_demand_signal=y, feedback_demand=d,
                                                    feedback_available_period=np.arange(T) + 1,
                                                    researcher_mean_demand=mu)))
                else:
                    a = B.actions(env, fam, final, T)
                    state, pay, obs = st_
                    frames.append(pd.DataFrame(dict(family=fam, period=np.arange(T), action=a,
                                                    feedback_payoff=pay[np.arange(T), a],
                                                    feedback_available_period=np.arange(T) + 1,
                                                    researcher_state=state)))
            return dict(trace=pd.concat(frames, ignore_index=True))

        return [Stage("splits", splits, SRC, role="splits", description="training, validation and test environments"),
                Stage("compute", compute, SRC, ("splits",), description="random-search tuning and test evaluation"),
                Stage("trace", trace, SRC, ("compute",), "trace", "selected configurations, first test environment")]

    return Study(f"bench_{kind}", f"Decision benchmark: {kind}", "simulation",
                 "Established decision methods under a common tuning protocol.", cfg, stages)


TRACKING, GENERALIZATION, LEARNABILITY, NK, BENCH_INVENTORY, BENCH_BANDIT = (
    _tracking(), _tasks(), _learnability(), _nk(), _bench("inventory"), _bench("bandit"))
STUDIES: List[Study] = [TRACKING, GENERALIZATION, LEARNABILITY, NK, BENCH_INVENTORY, BENCH_BANDIT]
