"""The tuning protocol shared by the decision benchmarks and the workbench (heiner_abm.bench_tuning).

Two things are being pinned. That "the same budget" is a stated choice with a recorded cost rather than an
assumption; and that routing the workbench through the shared protocol did not change which hyperparameters it
picks, only what it tells you about picking them.
"""
import numpy as np
import pandas as pd
import pytest

from heiner_abm.bench_tuning import (BUDGET_RULES, BudgetRecord, Choice, Family, Param, boundary_report,
                                     candidate_configs, design_budget, is_enumerated, sensitivity_report, tune)
from heiner_abm.workbench import execution as X, presets
from heiner_abm.workbench.environments import ENVIRONMENTS, TuningStep

SPACE = {"a": Param(0.01, 1.0, log=True), "b": Param(1, 100, integer=True)}


# ============================================================================== equivalent training opportunities
def test_the_two_notions_of_an_equal_budget_differ_and_both_are_available():
    one = {"a": SPACE["a"]}
    assert design_budget(SPACE, "equal_candidates", 8) == design_budget(one, "equal_candidates", 8) == 8
    assert design_budget(SPACE, "per_parameter", 8) == 16
    assert design_budget(one, "per_parameter", 8) == 8


def test_a_family_with_nothing_to_tune_gets_one_evaluation_under_either_rule():
    for rule in BUDGET_RULES:
        assert design_budget({}, rule, 25) == 1


def test_an_unnamed_budget_rule_is_refused():
    with pytest.raises(ValueError):
        design_budget(SPACE, "whatever_is_cheapest", 4)


def test_the_budget_record_adds_up_and_carries_its_rule():
    r = BudgetRecord(rule="per_parameter", base=4, n_parameters=2, candidates=8, candidates_per_parameter=4.0,
                     train_runs=80, validation_runs=10, test_runs=5)
    assert r.total_runs == 95
    d = r.to_dict()
    assert d["total_runs"] == 95 and "equal density" in d["rule_note"]


# ============================================================================== candidates
def test_a_declared_list_is_enumerated_in_order_and_nests():
    space = {"x": Choice((1, 2, 3)), "y": Choice(("a", "b"))}
    assert is_enumerated(space)
    four = candidate_configs(space, 4, seed=0)
    two = candidate_configs(space, 2, seed=0)
    assert four[:2] == two, "a smaller budget must see a prefix of what a larger one sees"
    assert len(candidate_configs(space, 99, seed=0)) == 6, "the whole product and no more"


def test_a_sampled_space_is_not_enumerated_and_reproduces():
    assert not is_enumerated(SPACE)
    assert not is_enumerated({"x": Choice((1, 2)), "y": Param(0.0, 1.0)})
    assert candidate_configs(SPACE, 5, seed=3) == candidate_configs(SPACE, 5, seed=3)


# ============================================================================== boundary selections
def test_a_value_at_the_end_of_its_range_is_reported_as_pinned():
    b = boundary_report(SPACE, {"a": 0.01, "b": 50})
    row = b.set_index("parameter")
    assert bool(row.loc["a", "at_low"]) and bool(row.loc["a", "pinned"])
    assert not bool(row.loc["b", "pinned"])


def test_position_is_measured_on_the_scale_the_parameter_is_sampled_on():
    """A log-scaled range is judged in logs, so the midpoint of [0.01, 1] is 0.1 and not 0.5."""
    b = boundary_report({"a": Param(0.01, 1.0, log=True)}, {"a": 0.1}).iloc[0]
    assert b["position"] == pytest.approx(0.5, abs=1e-9)
    lin = boundary_report({"a": Param(0.01, 1.0)}, {"a": 0.1}).iloc[0]
    assert lin["position"] == pytest.approx((0.1 - 0.01) / 0.99)


def test_a_declared_list_is_pinned_at_its_first_or_last_candidate():
    space = {"g": Choice((0.25, 0.5, 1.0, 2.0))}
    assert bool(boundary_report(space, {"g": 0.25}).iloc[0]["at_low"])
    assert bool(boundary_report(space, {"g": 2.0}).iloc[0]["at_high"])
    assert not bool(boundary_report(space, {"g": 0.5}).iloc[0]["pinned"])


# ============================================================================== sensitivity
def test_sensitivity_reports_the_spread_across_repetitions_and_the_gain_from_budget():
    curve = pd.DataFrame([dict(family="f", rep=r, budget=b, validation=v)
                          for b, vs in ((2, [1.0, 3.0]), (8, [5.0, 5.0]))
                          for r, v in enumerate(vs)])
    s = sensitivity_report(curve).set_index("budget")
    assert s.loc[2, "validation_sd"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert s.loc[2, "validation_range"] == pytest.approx(2.0)
    assert s.loc[8, "gain_over_smallest_budget"] == pytest.approx(3.0)
    assert s.loc[8, "validation_sd"] == 0.0


def test_an_empty_curve_reports_nothing_rather_than_failing():
    assert sensitivity_report(pd.DataFrame()).empty


# ============================================================================== the benchmarks record all of it
def test_a_tuned_family_carries_its_budget_boundary_and_sensitivity():
    fam = Family("f", "F", "competitor", SPACE, "assumption", "cost")
    run = lambda e, cfg: {"payoff": -abs(cfg["a"] - 0.3) - abs(cfg["b"] - 50) / 100.0 + 0.01 * e}
    t = tune(fam, run, [0, 1], [2], [3], budgets=(2, 4, 8), reps=2, seed=1)
    assert t.budget is not None and t.budget.rule == "equal_candidates" and t.budget.candidates == 8
    assert t.budget.total_runs > 0
    assert set(t.boundary["parameter"]) == set(SPACE)
    assert not t.sensitivity.empty and set(t.sensitivity["budget"]) == {2, 4, 8}


def test_the_per_parameter_rule_buys_a_two_parameter_family_a_denser_search():
    fam = Family("f", "F", "competitor", SPACE, "assumption", "cost")
    run = lambda e, cfg: {"payoff": -abs(cfg["a"] - 0.3)}
    equal = tune(fam, run, [0], [1], [2], budgets=(4,), reps=1, seed=1, budget_rule="equal_candidates")
    dense = tune(fam, run, [0], [1], [2], budgets=(4,), reps=1, seed=1, budget_rule="per_parameter")
    assert dense.budget.candidates == 2 * equal.budget.candidates
    assert dense.evaluations > equal.evaluations, "a denser search must cost more, and be recorded as costing more"


# ============================================================================== the workbench, unified
@pytest.fixture(scope="module")
def spec():
    s = presets.load("learnability", "research")
    s.design.replications, s.design.periods, s.design.burn_in = 3, 240, 30
    s.design.tuning, s.design.train_replications, s.design.n_boot = "grid", 2, 50
    s.design.treatment.control, s.design.treatment.levels = "", ()
    return s


def test_the_shared_protocol_picks_what_the_hand_rolled_grids_picked(spec):
    """The unification must change what the workbench TELLS you, not what it chooses. The grids below are the
    ones the inventory adapter used before the protocol was shared."""
    from heiner_abm import learnability as L
    env = ENVIRONMENTS[spec.environment]
    params, seeds = env.params(spec, None), X.train_seeds(spec)
    cfg, T, burn = env._cfg(params), spec.design.periods, spec.design.burn_in

    def old_score(policy, hp):
        vals = []
        for sd in seeds:
            p = L.paths(cfg, T, sd)
            c = L.candidate_orders(cfg, p, hp["gain"])
            S, pr, go = L.policy_orders(policy, cfg, p, c, hp)
            vals.append(L.outcomes(policy, cfg, p, c, S, pr, go, burn)["net_payoff"])
        return float(np.mean(vals))

    base = env._hp(params, {})
    gain = max((0.05, 0.1, 0.2, 0.4), key=lambda g: old_score("always", {**base, "gain": g}))
    b = max((0.25, 0.5, 1.0, 2.0), key=lambda v: old_score("band", {**base, "gain": gain, "band": {"b": v}}))
    grid = ((0.9, 5.0), (0.8, 3.0), (0.9, 12.0), (0.95, 25.0))
    conf, nmin = max(grid, key=lambda v: old_score("gate_lcb", {**base, "gain": gain,
                                                                "gate": {"conf": v[0], "nmin": v[1]}}))
    assert env.tune(spec, params, seeds, lambda: False) == {"gain": gain, "band_b": b, "gate_conf": conf,
                                                            "gate_nmin": nmin}


def test_every_tunable_policy_is_offered_the_same_number_of_candidates(spec):
    """The property established when the unequal tuning budget was fixed: no policy is compared on a smaller
    search than another."""
    env = ENVIRONMENTS[spec.environment]
    counts = {step.name: len(candidate_configs(step.space, design_budget(step.space, "equal_candidates", 4), 0))
              for step in env.tuning_plan(spec, env.params(spec, None))}
    assert len(set(counts.values())) == 1, counts


def test_a_candidate_the_environment_cannot_read_is_caught_rather_than_ignored(spec):
    """Regression. The confidence gate is tuned over four PAIRS, which have to be expanded into the two keys the
    adapter reads. Without the expansion every candidate scored the same and the search looked decisive while
    doing nothing; the zero spread below is what that failure looks like."""
    env = ENVIRONMENTS[spec.environment]
    params, seeds = env.params(spec, None), X.train_seeds(spec)
    gate = [s for s in env.tuning_plan(spec, params) if s.name == "confidence gate"][0]
    assert gate.expand is not None, "paired candidates must declare how they reach the adapter"
    assert set(gate.to_hp({"gate": (0.8, 3.0)})) == {"gate_conf", "gate_nmin"}
    scores = [env.score_policy(spec, params, gate.policy, gate.to_hp(c), seeds)
              for c in candidate_configs(gate.space, 4, 0)]
    assert max(scores) - min(scores) > 0, "every gate candidate scored the same, so none of them reached the gate"


def test_the_run_manifest_reports_the_budget_the_boundaries_and_the_spread(spec):
    out = X.execute(spec, "research")
    t = out.manifest["tuning"]
    assert t["budget_rule"] in BUDGET_RULES
    assert "bench_tuning" in t["protocol"]
    steps = [st for level in t["steps"].values() for st in level]
    assert steps, "a tuned run reported no tuning steps"
    for st in steps:
        assert {"candidates", "candidates_per_parameter", "chosen", "chosen_score", "gain_over_first",
                "score_spread", "pinned", "boundary"} <= set(st)
    assert isinstance(t["pinned"], list)


def test_a_pinned_selection_is_surfaced_where_a_reader_will_see_it(spec):
    """The inaction band is chosen at the smallest value it is offered in this environment, which means the list
    is too narrow at the low end. The point of reporting it is that this is visible without rerunning anything."""
    out = X.execute(spec, "research")
    pinned = out.manifest["tuning"]["pinned"]
    assert any("band_b" in p for p in pinned), pinned
    assert "statement about the candidate list" in out.manifest["tuning"]["pinned_note"]
