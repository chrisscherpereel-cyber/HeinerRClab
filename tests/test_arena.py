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
        k = arena.design_budget(PREREG, key)
        cand = _lhs(DESIGNS[key].SPACE, k, np.random.default_rng(0))
        assert all(len(v) == k for v in cand.values())
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


# ---------------------------------------------------------------- equal search density (8 October 2026)
def test_every_design_is_searched_at_the_same_density():
    """Designs carry one to six free parameters. The registered rule gives each the same number of candidates per
    free parameter, so no theory is penalized for a richer parameterization. Under the legacy per-design rule the
    density ranged over a factor of six, and the reliability-condition designs were the worst served."""
    per_param = {d: arena.design_budget(PREREG, d) / max(1, len(DESIGNS[d].SPACE)) for d in DESIGNS}
    assert len(set(per_param.values())) == 1
    assert set(per_param.values()) == {float(PREREG.budget_per_parameter)}
    # the problem this fixes: a flat budget is unequal, and worst for the design with the most parameters
    legacy = replace(PREREG, budget_rule="per_design")
    dens = {d: arena.design_budget(legacy, d) / max(1, len(DESIGNS[d].SPACE)) for d in DESIGNS}
    assert len(set(dens.values())) > 1
    assert min(dens, key=dens.get) == "heiner_m" and len(DESIGNS["heiner_m"].SPACE) == 6
    # no design is searched less thoroughly than it was under the legacy budget
    assert all(arena.design_budget(PREREG, d) >= legacy.budget for d in DESIGNS)


def test_legacy_budget_rule_is_the_old_fixed_budget_and_unknown_rules_are_refused():
    legacy = replace(PREREG, budget_rule="per_design", budget=24)
    assert {arena.design_budget(legacy, d) for d in DESIGNS} == {24}
    with pytest.raises(ValueError):
        arena.design_budget(replace(PREREG, budget_rule="nonsense"), "heiner_m")


def test_chunked_candidate_scoring_is_identical_to_one_batch():
    """Budgets now grow with the dimension, so candidates are scored in chunks. Every market is seeded from its
    environment alone, so chunking must not move a single number."""
    pr = replace(QUICK, periods=120, burn_in=20)
    envs = arena.train_envs(pr)
    tuned = arena.Tuned(design={t: d[0] for t, d in THEORY_DESIGNS.items()},
                        params={d: arena.default_params(d) for d in DESIGNS})
    design, slot = "heiner_m", KEYS.index("heiner")
    cand = _lhs(DESIGNS[design].SPACE, 7, np.random.default_rng(1))
    saved = arena.MAX_TUNING_MARKETS
    try:
        arena.MAX_TUNING_MARKETS = 10 ** 9
        one = arena._score_candidates(pr, envs, tuned, design, slot, cand, 7)
        for m in (len(envs) * 3, len(envs) * 2, len(envs)):
            arena.MAX_TUNING_MARKETS = m
            np.testing.assert_array_equal(arena._score_candidates(pr, envs, tuned, design, slot, cand, 7), one)
    finally:
        arena.MAX_TUNING_MARKETS = saved


def test_tuning_log_reports_the_budget_each_design_received():
    pr = replace(QUICK, budget_per_parameter=1, periods=120, burn_in=20, n_train=3)
    _, log = arena.tune(pr)
    assert {"n_params", "n_candidates", "candidates_per_param"} <= set(log.columns)
    assert log["candidates_per_param"].nunique() == 1
    for _, r in log.iterrows():
        assert r["n_candidates"] == arena.design_budget(pr, r["design"])
