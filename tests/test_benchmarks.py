"""Decision benchmarks (heiner_abm.bench_inventory, bench_bandit, bench_tuning): correctness on solvable and enumerated
cases, information timing, the tuning protocol, and the separation of benchmarks, oracles and references from the
rankings."""
import numpy as np
import pytest

from heiner_abm import bench_bandit as BB
from heiner_abm import bench_inventory as BI
from heiner_abm import tasks
from heiner_abm.bench_checks import CHECKS, run_checks
from heiner_abm.bench_tuning import Family, Param, sample_configs, tune
from heiner_abm.information import AGENT_BY_KEY, ENGINES, compatibility
from heiner_abm.literature import REFERENCES
from heiner_abm.model_spec import all_specs


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_correctness_check_passes(check):
    cid, what, passed, err, tol = check()
    assert passed, (cid, what, err, tol)


# ------------------------------------------------------------------------------------------------ information timing
def _y(T=160, seed=1):
    env = tasks.TaskEnv("inventory", 15.0, 0.03, 5.0, 0.1, seed)
    return env, BI._stream(env, T)[2]


@pytest.mark.parametrize("make", [
    lambda y: BI.window_orders(y, 30, BI.saa_order_window, np.full(len(y), 100.0)),
    lambda y: BI.window_orders(y, 30, lambda W: BI.dro_orders_window(W, 0.3), np.full(len(y), 100.0)),
    lambda y: BI.bocpd_orders(y, 0.02, 30.0, 20.0),
    lambda y: BI.bayes_true_orders(y, 0.03, 15.0, 5.0, grid_step=2.0),
], ids=["saa", "dro", "bocpd", "bayes_true"])
def test_inventory_orders_use_only_past_observations(make):
    _, y = _y()
    K = 90
    alt = y.copy()
    alt[K:] += np.random.default_rng(0).normal(0, 40, len(y) - K)
    a, b = make(y), make(alt)
    np.testing.assert_array_equal(a[:K + 1], b[:K + 1])          # the order of period K uses y up to K - 1
    assert not np.array_equal(a[K + 1:], b[K + 1:])


@pytest.mark.parametrize("family,cfg", [("ucb", {"c": 0.5}), ("sw_ucb", {"window": 25, "c": 0.5}),
                                        ("d_ucb", {"one_minus_gamma": 0.05, "c": 0.5}),
                                        ("eps_greedy", {"eps": 0.1, "alpha": 0.2})])
def test_bandit_choices_use_only_past_payoffs(family, cfg, monkeypatch):
    env = tasks.TaskEnv("learning", 1.0, 0.02, 0.5, 0.1, 3)
    T, K = 200, 120
    state, pay, obs = BB._stream(env, T)
    base = BB.actions(env, family, cfg, T)
    alt = obs.copy()
    alt[K:] += 30.0
    monkeypatch.setattr(BB, "_stream", lambda e, t: (state, pay, alt))
    pert = BB.actions(env, family, cfg, T)
    np.testing.assert_array_equal(base[:K + 1], pert[:K + 1])


def test_every_agent_makes_one_choice_per_period_with_matched_feedback():
    env = tasks.TaskEnv("learning", 1.0, 0.02, 0.5, 0.1, 3)
    for k, f in BB.FAMILIES.items():
        cfg = {p: (v.lo * v.hi) ** 0.5 if v.log else (v.lo + v.hi) / 2 for p, v in f.space.items()}
        assert BB.actions(env, k, cfg, 150).shape == (150,)
        if f.role == "competitor":
            assert f.feedback == "Chosen-action payoff"
            assert AGENT_BY_KEY[("bench_bandit", BB.REGISTRY_KEY[k])].feedback == ("chosen",)
    assert BB.FAMILIES["full_ref"].role == "reference"                  # full feedback: never ranked as a bandit
    assert AGENT_BY_KEY[("tasks", "task_full_ref")].feedback == ("full",)


# ------------------------------------------------------------------------------------------------ properties
def test_worst_case_grows_with_ambiguity_and_starts_at_the_empirical_loss():
    L = np.random.default_rng(4).uniform(0, 30, 12)
    vals = [float(BI.chi2_worst_case(L, r)) for r in (0.0, 0.01, 0.1, 0.5, 2.0, 11.0, 50.0)]
    assert vals[0] == pytest.approx(L.mean(), abs=1e-12)
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert vals[-1] == pytest.approx(L.max())


def test_full_ambiguity_gives_the_minimax_order():
    """Once the ball contains every point mass (rho >= N - 1) the worst case is the largest loss, so the robust order
    equalizes the overage loss at the smallest and the shortage loss at the largest observation:
    S = (h * min + p * max) / (h + p)."""
    W = np.random.default_rng(8).normal(100, 25, (1, 40))
    S = float(BI.dro_orders_window(W, 39.0)[0])
    exact = (BI.H_OVER * W.min() + BI.P_SHORT * W.max()) / (BI.H_OVER + BI.P_SHORT)
    assert S == pytest.approx(exact, abs=1e-6)


def test_grid_benchmark_is_insensitive_to_refining_the_grid():
    _, y = _y(T=300)
    a = BI.bayes_true_orders(y, 0.03, 15.0, 5.0, grid_step=1.0)
    b = BI.bayes_true_orders(y, 0.03, 15.0, 5.0, grid_step=0.5)
    assert np.mean(np.abs(a - b)) < 0.5


def test_correct_model_beats_its_misspecified_variant_on_average():
    """Sanity check on a few environments: the correctly specified filter earns at least as much as BOCPD at a
    reasonable configuration (not a reported result)."""
    envs = tasks._env_set("inventory", 4, 991)
    cfg = {"hazard": 0.02, "prior_sd": 40.0, "obs_sd": 25.0}
    true = np.mean([BI.run_agent(e, "bayes_true", {}, 500, 50)["payoff"] for e in envs])
    mis = np.mean([BI.run_agent(e, "bocpd", cfg, 500, 50)["payoff"] for e in envs])
    assert true >= mis


# ------------------------------------------------------------------------------------------------ tuning protocol
def test_environment_sets_are_disjoint_and_tuning_is_nested():
    plan = BI.InventoryPlan(**BI.QUICK_INVENTORY)
    sets = plan.env_sets()
    seeds = [{e.seed for e in v} for v in sets.values()]
    assert not (seeds[0] & seeds[1]) and not (seeds[1] & seeds[2]) and not (seeds[0] & seeds[2])
    fam = Family("toy", "toy", "competitor", {"x": Param(0.0, 1.0)}, "", "")
    run = lambda env, cfg: {"payoff": -(cfg["x"] - 0.3) ** 2 - 0.01 * (env % 3)}
    res = tune(fam, run, [0, 1, 2], [3, 4], [5, 6, 7], (1, 2, 4, 8), reps=2, seed=1)
    for _, d in res.curve.groupby("rep"):
        assert np.all(np.diff(d.sort_values("budget")["train"].to_numpy()) >= 0)   # more budget never trains worse
    cfgs = sample_configs(fam.space, 8, 1)
    assert cfgs[:4] == sample_configs(fam.space, 4, 1)                              # budgets are prefixes
    assert len(res.test) == 3 and abs(res.final["x"] - 0.3) < 0.3


def test_quick_runs_keep_benchmarks_and_oracles_out_of_the_rankings():
    for out in (BI.run_inventory(BI.InventoryPlan(**{**BI.QUICK_INVENTORY, "periods": 250, "burn_in": 30})),
                BB.run_bandit(BB.BanditPlan(**BB.QUICK_BANDIT))):
        rk = out["ranking"]
        assert rk.loc[rk["role"] != "competitor", "rank"].isna().all()
        ranked = rk[rk["role"] == "competitor"].sort_values("rank")
        assert list(ranked["rank"]) == sorted(ranked["rank"])
        assert {"budget", "train_runs", "validation"} <= set(out["curve"].columns)


# ------------------------------------------------------------------------------------------------ registry
def test_every_family_is_registered_documented_and_sourced():
    keys = {s.key for s in all_specs()}
    for mod, engine in ((BI, "bench_inventory"), (BB, "bench_bandit")):
        for k, f in mod.FAMILIES.items():
            rk = mod.REGISTRY_KEY[k]
            eng = "tasks" if rk == "task_full_ref" else engine
            agent = AGENT_BY_KEY[(eng, rk)]
            assert rk in keys
            assert all(s in REFERENCES for s in f.sources)
            assert f.assumptions and f.compute
            status = compatibility(agent, ENGINES[eng].native, eng).status
            assert status == ("oracle variant" if f.role in ("benchmark", "oracle") else "supported"), (k, status)
            assert agent.oracle == (f.role in ("benchmark", "oracle"))


def test_checks_table():
    df = run_checks()
    assert df["passed"].all() and len(df) == len(CHECKS)
