"""Agent tournament: market parity with the main model, fairness of the protocol, and the statistics."""
from dataclasses import replace

import numpy as np
import pytest

from heiner_abm.agents import cost_path, make_streams
from heiner_abm import arena
from heiner_abm.arena import (DESIGNS, FACTORIAL, KEYS, PREREG, QUICK, THEORY_DESIGNS, Env, Market, _lhs, _scenario,
                              TARGET_SPACE, cluster_ci, holm, run_protocol, simulate)

ENV = Env(delta=10, c_max=80, q_range=1500, noise=0, foresight=0, hazard=0, belief_lag=20, seed=3)


def test_market_uses_the_main_models_cost_process():
    mk = Market([ENV], 4, 200)
    scn = _scenario(ENV, 4, 200)
    np.testing.assert_allclose(mk.cost[0], cost_path(scn, make_streams(scn.seed, 4, 200)[0]))


def test_rule_b_never_changes_output_and_simulation_is_deterministic():
    lineup = np.array([list(DESIGNS)], dtype=object)
    a = simulate([ENV], lineup, {}, 200, 20)
    b = simulate([ENV], lineup, {}, 200, 20)
    np.testing.assert_array_equal(a["profit"], b["profit"])
    assert a["change_rate"][0, list(DESIGNS).index("ruleb")] == 0


def test_per_market_parameters_match_separate_runs():
    """A tuning batch (parameters varying across markets) gives the same result as separate runs."""
    envs = [ENV, replace(ENV, seed=4)]
    lineup = np.tile(np.array(list(DESIGNS), dtype=object), (2, 1))
    batch = simulate(envs, lineup, {"heiner_m": {"theta": np.array([5.0, 30.0])}}, 200, 20)
    for j, th in enumerate((5.0, 30.0)):
        alone = simulate([envs[j]], lineup[:1], {"heiner_m": {"theta": th}}, 200, 20)
        np.testing.assert_allclose(batch["profit"][j], alone["profit"][0])


def test_equal_budget_and_disjoint_environments():
    assert all(len(d) == 2 for t, d in THEORY_DESIGNS.items() if t != "ruleb")
    for key in DESIGNS:
        cand = _lhs(DESIGNS[key].SPACE, PREREG.budget, np.random.default_rng(0))
        assert all(len(v) == PREREG.budget for v in cand.values())
        assert all(v[0] == DESIGNS[key].SPACE[n].default for n, v in cand.items())
        for n, v in cand.items():
            s = DESIGNS[key].SPACE[n]
            assert v.min() >= s.lo - 1e-9 and v.max() <= s.hi + 1e-9
    assert not {e.seed for e in arena.train_envs(PREREG)} & {e.seed for e in arena.test_envs(PREREG)}


def test_selection_rule_experiment_shares_targets():
    """Two targets x four selection rules; designs with the same target have the same target parameters."""
    cells = {(DESIGNS[d].TARGET, DESIGNS[d].SELECT) for d in FACTORIAL}
    assert cells == {(t, s) for t in ("model", "price") for s in ("always", "band", "rc", "aspiration")}
    for d in FACTORIAL:
        assert set(TARGET_SPACE[DESIGNS[d].TARGET]) <= set(DESIGNS[d].SPACE)


def test_composite_always_equals_selection_with_band_zero():
    """An inaction band of width 0 never blocks a move, so it reproduces 'always' exactly."""
    lineup = np.array([["opt_br"] + ["ruleb"] * 3], dtype=object)
    a = simulate([ENV], lineup, {}, 200, 20)
    lineup2 = np.array([["options_m"] + ["ruleb"] * 3], dtype=object)
    b = simulate([ENV], lineup2, {"options_m": {"k": 0.0}}, 200, 20)
    np.testing.assert_allclose(a["profit"], b["profit"])


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
    assert res.exploratory and len(res.verdicts) == 6
    assert set(res.ranking["key"]) == set(KEYS)
    assert res.invasion.shape[0] == len(KEYS) * (len(KEYS) - 1)
    assert all(res.tuned.design[t] in THEORY_DESIGNS[t] for t in KEYS)
    assert res.ranking["survived"].between(0, 1).all() and res.ranking["pareto"].any()
    assert len(res.factorial) == len(FACTORIAL)


def test_ecology_agents_are_inert():
    """Organizational ecology's agents change output far less often than an always-adjusting rule on the same target,
    and the scheduled reorganizer moves only at its interval."""
    envs = [replace(ENV, seed=s) for s in range(4)]
    lineup = np.array([["ecol_periodic", "ecol_crisis", "cobweb_p", "ruleb"]] * 4, dtype=object)
    out = simulate(envs, lineup, {"ecol_periodic": {"interval": 10}}, 300, 20)
    cr = out["change_rate"].mean(0)
    assert cr[0] <= 0.1 + 1e-9 and cr[1] < cr[2] and cr[3] == 0
    assert set(THEORY_DESIGNS["ecology"]) == {"ecol_periodic", "ecol_crisis"}
