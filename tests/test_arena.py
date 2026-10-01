"""Agent tournament: market parity with the main model, fairness of the protocol, and the statistics."""
from dataclasses import replace

import numpy as np
import pytest

from heiner_abm.agents import cost_path, make_streams
from heiner_abm import arena
from heiner_abm.arena import (AGENTS, KEYS, PREREG, QUICK, TUNED, Env, Market, _lhs, _scenario, cluster_ci, holm,
                              run_protocol, simulate)

ENV = Env(delta=10, c_max=80, q_range=1500, noise=0, foresight=0, hazard=0, belief_lag=20, seed=3)


def test_market_uses_the_main_models_cost_process():
    mk = Market([ENV], 4, 200)
    scn = _scenario(ENV, 4, 200)
    np.testing.assert_allclose(mk.cost[0], cost_path(scn, make_streams(scn.seed, 4, 200)[0]))


def test_rule_b_never_changes_output_and_simulation_is_deterministic():
    lineup = np.array([KEYS], dtype=object)
    a = simulate([ENV], lineup, {}, 200, 20)
    b = simulate([ENV], lineup, {}, 200, 20)
    np.testing.assert_array_equal(a["profit"], b["profit"])
    assert a["change_rate"][0, KEYS.index("ruleb")] == 0


def test_per_market_parameters_match_separate_runs():
    """A tuning batch (parameters varying across markets) gives the same result as separate runs."""
    envs = [ENV, replace(ENV, seed=4)]
    lineup = np.tile(np.array(KEYS, dtype=object), (2, 1))
    batch = simulate(envs, lineup, {"heiner": {"theta": np.array([5.0, 30.0])}}, 200, 20)
    for j, th in enumerate((5.0, 30.0)):
        alone = simulate([envs[j]], lineup[:1], {"heiner": {"theta": th}}, 200, 20)
        np.testing.assert_allclose(batch["profit"][j], alone["profit"][0])


def test_equal_budget_and_disjoint_environments():
    for key in TUNED:
        cand = _lhs(AGENTS[key].SPACE, PREREG.budget, np.random.default_rng(0))
        assert all(len(v) == PREREG.budget for v in cand.values())
        assert all(v[0] == AGENTS[key].SPACE[n].default for n, v in cand.items())
        for n, v in cand.items():
            s = AGENTS[key].SPACE[n]
            assert v.min() >= s.lo - 1e-9 and v.max() <= s.hi + 1e-9
    assert not {e.seed for e in arena.train_envs(PREREG)} & {e.seed for e in arena.test_envs(PREREG)}


def test_heiner_optimiser_and_options_share_the_target():
    shared = {"a_cost", "a_rival"}
    for key in ("heiner", "optimiser", "options"):
        assert shared <= set(AGENTS[key].SPACE)


def test_plan_hash_changes_with_any_setting():
    assert PREREG.digest != replace(PREREG, budget=PREREG.budget + 1).digest
    assert PREREG.digest == replace(PREREG).digest


def test_statistics():
    assert holm([0.01, 0.04, 0.03], 0.05) == [True, False, False]
    assert holm([0.001, 0.01, 0.02], 0.05) == [True, True, True]
    m, lo, hi, p = cluster_ci(np.r_[np.ones(20), np.ones(20) * 3], np.repeat(np.arange(10), 4), 200, 0.95)
    assert m == pytest.approx(2.0) and lo <= m <= hi


def test_quick_protocol_runs_end_to_end():
    res = run_protocol(QUICK)
    assert res.exploratory and len(res.verdicts) == 4
    assert set(res.ranking["key"]) == set(KEYS)
    assert res.invasion.shape[0] == len(KEYS) * (len(KEYS) - 1)
    assert set(res.tuned) == set(KEYS) and res.tuned["ruleb"] == {}
