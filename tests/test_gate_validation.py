"""Validation of the reliability gates (heiner_abm.gate_validation).

Two things have to hold before any number here means anything. The continuation estimator must describe the same
market the agents played, and it must be impossible for it to reach them. The first is pinned by exact equality
against the engine; the second has a section of its own.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from heiner_abm import gate_validation as V
from heiner_abm.engine import run_batch
from heiner_abm.gate_study import ORACLE_SEED_BASE, gate_scenario
from heiner_abm.scenarios import DEFAULTS, build_scenario


def _base(periods=240, burn_in=40, regime="clean"):
    s = build_scenario(dict(DEFAULTS))
    s = dataclasses.replace(s, periods=periods, burn_in=burn_in)
    for f in s.firms:
        f.selection = "Adaptive"
    s = V.regime_scenario(s, regime)
    s.periods, s.burn_in = periods, burn_in
    return s


def _run(scn, gate="lcb", markets=2, seed0=1):
    scns = [gate_scenario(scn, gate, seed0 + k) for k in range(markets)]
    return run_batch(scns, record_firm_history=True, horizon=scn.adaptive.window), scns


@pytest.fixture(scope="module")
def clean_states():
    scn = _base()
    res, scns = _run(scn)
    return scn, res, scns, V.sample_decision_states(res, scns, 60, seed=1)


# ============================================================================== the estimand
def test_the_three_quantities_are_named_and_distinct():
    assert set(V.ESTIMANDS) == {"observed_feedback", "realized_advantage", "conditional_expected_net"}
    assert len({v for v in V.ESTIMANDS.values()}) == 3
    for text in V.ESTIMANDS.values():
        assert len(text) > 60, "an estimand stated too briefly to be checked against"


def test_the_continuation_policy_and_horizon_are_declared():
    spec = V.ContinuationSpec()
    assert spec.policy in V.CONTINUATION_POLICIES and V.CONTINUATION_POLICIES[spec.policy]
    assert spec.regime in V.REGIMES
    scn = _base()
    assert spec.resolved_horizon(scn) == scn.adaptive.window
    assert V.ContinuationSpec(horizon=7).resolved_horizon(scn) == 7


@pytest.mark.parametrize("bad", [dict(policy="teleport"), dict(regime="whatever"), dict(draws=1),
                                 dict(seed_base=ORACLE_SEED_BASE)])
def test_an_undefined_continuation_is_refused(bad):
    with pytest.raises(ValueError):
        V.ContinuationSpec(**bad)


# ============================================================================== pinning the continuation
def test_at_one_period_the_continuation_is_the_closed_form_advantage(clean_states):
    """No rival has reacted and no cost has moved, so a correct continuation must reproduce this exactly."""
    scn, _, _, st = clean_states
    got = V.continuation_advantage(st, scn, V.ContinuationSpec(horizon=1, draws=4), cost=0.0)
    assert np.array_equal(got["A_star"].to_numpy(), V.one_period_advantage(st, scn))
    assert float(np.nanmax(got["A_star_se"])) == 0.0, "a one-period advantage has no continuation noise"


def test_the_continuation_reproduces_the_engine_fork_exactly(clean_states):
    """With a one-period judgement window the engine's own realized fork and A* are the same quantity. If the
    continuation dynamics ever drift from the engine's, this is where it shows."""
    scn, _, _, st = clean_states
    assert scn.adaptive.window == 1
    got = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=8), cost=0.0)
    d = got.dropna(subset=["researcher_realized_advantage"])
    assert len(d) > 10
    assert np.allclose(d["A_star"], d["researcher_realized_advantage"], rtol=0, atol=1e-9)


def test_the_estimate_is_reproducible_and_does_not_mutate_its_input(clean_states):
    scn, _, _, st = clean_states
    before = st.copy()
    a = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=8))
    b = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=8))
    pd.testing.assert_frame_equal(st, before)
    pd.testing.assert_frame_equal(a, b)


def test_more_draws_shrink_the_reported_uncertainty(clean_states):
    scn = _base(regime="overlapping")
    res, scns = _run(scn)
    st = V.sample_decision_states(res, scns, 40, seed=1)
    few = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=4))
    many = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=64))
    assert np.nanmean(many["A_star_se"]) < np.nanmean(few["A_star_se"])


def test_the_adjustment_charge_enters_the_estimand(clean_states):
    scn, _, _, st = clean_states
    free = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=4), cost=0.0)
    charged = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=4), cost=3.0)
    assert np.allclose(free["A_star"] - 3.0, charged["A_star"]), "A* must be net of the adaptation charge"


# ============================================================================== leakage
def test_continuation_seeds_cannot_collide_with_evaluation_or_oracle_seeds():
    assert V.CONTINUATION_SEED_BASE > ORACLE_SEED_BASE
    plan = V.ValidationPlan()
    evaluation = set(range(plan.seed0, plan.seed0 + plan.markets))
    assert max(evaluation) < ORACLE_SEED_BASE < plan.continuation.seed_base


def test_running_a_validation_leaves_the_market_bit_identical():
    """The decisive leakage check: if a continuation could touch an agent, replaying the market after a validation
    would differ."""
    scn = _base()
    before, scns = _run(scn)
    st = V.sample_decision_states(before, scns, 40, seed=1)
    st = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=16))
    V.calibration(st, scn)
    V.coverage(st, scn)
    after, _ = _run(scn)
    for key in ("price", "cost", "quantity"):
        assert np.array_equal(getattr(before, key), getattr(after, key)), f"{key} changed after a validation"
    for key in ("q", "deviate", "adv_pred", "adv_bound"):
        a, b = before.firm_hist.get(key), after.firm_hist.get(key)
        if a is not None:
            assert np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True), f"{key} changed"


def test_researcher_only_truths_are_named_as_such(clean_states):
    _, _, _, st = clean_states
    hidden = V.researcher_only_columns(st.columns)
    assert {"researcher_cost", "researcher_true_intercept", "researcher_true_slope",
            "researcher_realized_advantage"} <= set(hidden)
    agent_visible = [c for c in st.columns if not c.startswith(V.RESEARCHER_PREFIX)]
    assert "A_star" not in agent_visible, "A* must not be produced before the continuations run"


def test_the_truth_is_never_passed_to_a_gate(clean_states):
    """A gate reads a scenario, and no scenario field can carry A*. The one researcher quantity the scenario can
    carry is the ORACLE table, and only the benchmark gate may hold it: two guards, both checked here."""
    scn, _, _, st = clean_states
    st2 = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=4))
    assert "A_star" in st2.columns
    assert not any("a_star" in f.lower() or "truth" in f.lower() for f in vars(scn.adaptive))

    table = [[0.0] * (len(scn.adaptive.bin_edges) + 1) for _ in range(scn.n_firms)]
    # guard 1: gate_scenario drops a table handed to an ordinary gate
    assert gate_scenario(scn, "lcb", 1, table).adaptive.oracle_table == ()
    # guard 2: setting one directly is rejected by the scenario itself
    sneaky = gate_scenario(scn, "lcb", 1)
    sneaky.adaptive.oracle_table = tuple(tuple(r) for r in table)
    assert any("ORACLE table" in e for e in sneaky.validate())


# ============================================================================== calibration and coverage
def test_calibration_reports_bias_and_a_curve_to_plot(clean_states):
    scn, _, _, st = clean_states
    st = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=8))
    summary, curve = V.calibration(st, scn, n_bins=5)
    assert summary["n"] > 10 and np.isfinite(summary["bias"]) and np.isfinite(summary["rmse"])
    assert set(curve.columns) == {"bin", "pred_mean", "truth_mean", "truth_se", "n"}
    assert len(curve) <= 5 and curve["n"].sum() == summary["n"]


def test_a_perfect_prediction_has_zero_bias_and_unit_slope():
    scn = _base()
    truth = np.linspace(-20, 20, 50)
    df = pd.DataFrame({"predicted_advantage": truth + scn.adaptive.adjust_cost, "A_star": truth})
    summary, _ = V.calibration(df, scn)
    assert abs(summary["bias"]) < 1e-9 and abs(summary["slope"] - 1) < 1e-9


def test_coverage_is_reported_as_measured_and_never_asserted(clean_states):
    scn, _, _, st = clean_states
    st = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=8))
    cov = V.coverage(st, scn)
    assert cov["nominal"] == scn.adaptive.confidence
    assert 0.0 <= cov["empirical"] <= 1.0
    assert cov["ci_lo"] <= cov["empirical"] <= cov["ci_hi"]
    assert "measured, not asserted" in cov["claim"]
    assert "A*" in cov["target"] or "conditional expected" in cov["target"]


def test_coverage_of_a_bound_that_always_holds_is_one():
    scn = _base()
    df = pd.DataFrame({"lower_bound": np.full(40, -1e6), "A_star": np.linspace(-5, 5, 40)})
    assert V.coverage(df, scn)["empirical"] == 1.0


# ============================================================================== decisions and their cost
def test_both_error_types_and_their_cost_are_counted():
    df = pd.DataFrame({"A_star": [10.0, -4.0, 6.0, -2.0], "adopted": [True, True, False, False]})
    e = V.decision_errors(df)
    assert e["adoption_rate"] == 0.5
    assert e["false_adaptation_rate"] == 0.25 and e["missed_adaptation_rate"] == 0.25
    assert e["cost_of_false_adaptation"] == pytest.approx(4.0 / 4)
    assert e["cost_of_missed_adaptation"] == pytest.approx(6.0 / 4)
    assert e["regret_total"] == pytest.approx((4.0 + 6.0) / 4)
    assert e["best_possible"] == pytest.approx((10.0 + 6.0) / 4)


def test_a_perfect_decision_rule_has_no_regret():
    df = pd.DataFrame({"A_star": [10.0, -4.0, 6.0, -2.0], "adopted": [True, False, True, False]})
    e = V.decision_errors(df)
    assert e["false_adaptation_rate"] == 0 and e["missed_adaptation_rate"] == 0
    assert e["regret_total"] == 0


def test_the_gap_is_split_into_misspecification_and_continuation_noise(clean_states):
    scn, _, _, st = clean_states
    st = V.continuation_advantage(st, scn, V.ContinuationSpec(draws=8), cost=0.0)
    d = V.decompose(st, scn, cost=0.0)
    assert d["prediction_minus_truth"] == pytest.approx(d["prediction_minus_realized"] + d["realized_minus_truth"],
                                                        abs=1e-6)
    # with a one-period window the realized fork IS the estimand, so the whole gap is the firm's misspecification
    assert abs(d["realized_minus_truth"]) < 1e-9


# ============================================================================== feedback regimes
@pytest.mark.parametrize("regime", sorted(V.FEEDBACK_REGIMES))
def test_every_regime_is_documented_and_applies_its_edit(regime):
    spec = V.FEEDBACK_REGIMES[regime]
    assert spec["note"].strip()
    s = V.regime_scenario(build_scenario(dict(DEFAULTS)), regime)
    assert s.adaptive.window == spec["window"]
    assert s.info.obs_delay == spec["obs_delay"] and s.info.obs_noise == spec["obs_noise"]
    assert s.structural.hazard == spec["hazard"]


def test_an_unknown_regime_is_refused():
    with pytest.raises(ValueError):
        V.regime_scenario(build_scenario(dict(DEFAULTS)), "sunshine")


def test_the_regimes_cover_what_the_brief_asks_about():
    for name in ("overlapping", "delayed", "noisy", "missing", "correlated"):
        assert name in V.FEEDBACK_REGIMES


# ============================================================================== memory variants
def _streams(n=60, gap=1, change=False):
    rng = np.random.default_rng(0)
    return pd.DataFrame(dict(market=0, firm=0, bin=1, decided=np.arange(n) * gap,
                             released=np.arange(n) * gap, g=rng.normal(5.0, 2.0, n),
                             change_since_last=[bool(change and k and k % 10 == 0) for k in range(n)]))


def test_the_four_variants_are_separate_and_documented():
    assert set(V.VARIANTS) == set(V.VARIANT_NOTES)
    assert len(V.VARIANTS) == 4
    for note in V.VARIANT_NOTES.values():
        assert note.strip()


def test_calendar_aging_equals_per_update_forgetting_when_releases_are_one_period_apart():
    """The variants must differ only where they are meant to: with one period between releases, ageing by calendar
    time and ageing by arrival count are the same rule."""
    s = _streams(gap=1)
    a = V.replay_variant(s, V.VariantSpec("per_update_forgetting", memory=0.9))
    b = V.replay_variant(s, V.VariantSpec("calendar_aging", memory=0.9))
    assert np.allclose(a["estimate"].to_numpy(float), b["estimate"].to_numpy(float), equal_nan=True)


def test_calendar_aging_differs_when_releases_are_far_apart():
    s = _streams(gap=7)
    a = V.replay_variant(s, V.VariantSpec("per_update_forgetting", memory=0.9))
    b = V.replay_variant(s, V.VariantSpec("calendar_aging", memory=0.9))
    assert not np.allclose(a["estimate"].to_numpy(float)[5:], b["estimate"].to_numpy(float)[5:], equal_nan=True)


def test_a_reset_only_fires_where_a_change_was_detected():
    same = V.replay_variant(_streams(change=False), V.VariantSpec("change_reset"))
    base = V.replay_variant(_streams(change=False), V.VariantSpec("per_update_forgetting"))
    assert np.allclose(same["estimate"].to_numpy(float), base["estimate"].to_numpy(float), equal_nan=True)
    reset = V.replay_variant(_streams(change=True), V.VariantSpec("change_reset"))
    assert not np.allclose(reset["evidence"].to_numpy(float), base["evidence"].to_numpy(float), equal_nan=True)


def test_continuing_exploration_changes_the_propensity_but_not_the_estimate():
    s = _streams()
    base = V.replay_variant(s, V.VariantSpec("per_update_forgetting"))
    cont = V.replay_variant(s, V.VariantSpec("explore_continuing", explore_after=0.5))
    assert np.allclose(base["estimate"].to_numpy(float), cont["estimate"].to_numpy(float), equal_nan=True)
    assert cont["exploration_propensity"].mean() > base["exploration_propensity"].mean()


def test_exploration_propensity_and_cost_are_logged():
    s = _streams()
    r = V.replay_variant(s, V.VariantSpec("explore_continuing", explore_after=0.5))
    assert {"exploration_propensity", "explored"} <= set(r.columns)
    score = V.score_variant(r)
    for k in ("exploration_propensity", "exploration_share", "exploration_cost", "decided_share"):
        assert k in score and np.isfinite(score[k])


def test_a_variant_refuses_a_setting_that_belongs_to_another():
    with pytest.raises(ValueError):
        V.VariantSpec("per_update_forgetting", explore_after=0.3)
    with pytest.raises(ValueError):
        V.VariantSpec("explore_continuing", explore_after=0.0)
    with pytest.raises(ValueError):
        V.VariantSpec("exponential_decay_but_nicer")


def test_the_variant_score_states_the_target_it_measures():
    score = V.score_variant(V.replay_variant(_streams(), V.VariantSpec()))
    assert "observable" in score["target"] and "counterfactual" in score["target"]


# ============================================================================== the plan
def test_the_plan_is_frozen_and_hashed():
    a, b = V.ValidationPlan(), V.ValidationPlan()
    assert a.digest == b.digest and len(a.digest) == 16
    assert V.ValidationPlan(markets=99).digest != a.digest
    assert V.ValidationPlan(continuation=V.ContinuationSpec(draws=8)).digest != a.digest
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.markets = 2


def test_the_experiment_specification_is_enough_to_reproduce_a_run():
    plan = V.ValidationPlan()
    spec = V.experiment_spec(plan, _base())
    for key in ("plan", "plan_hash", "estimands", "continuation_policies", "regimes", "variant_notes",
                "scenario", "seed_blocks", "reproduce"):
        assert key in spec
    assert spec["plan_hash"] == plan.digest
    assert "researcher_continuations" in spec["seed_blocks"]


def test_a_small_validation_runs_end_to_end():
    plan = V.ValidationPlan(markets=2, periods=240, burn_in=40, states_per_cell=30, gates=("lcb",),
                            regimes=("clean",), variant_regime="missing",
                            continuation=V.ContinuationSpec(draws=8))
    out = V.run_validation(plan, _base())
    assert out.plan_hash == plan.digest
    assert len(out.summary) == 1 and len(out.states) > 10
    assert not out.calibration.empty and len(out.variants) == len(V.VARIANTS)
    assert {"cov_nominal", "cov_empirical", "false_adaptation_rate", "cost_of_false_adaptation"} <= set(
        out.summary.columns)


# ============================================================================== the committed results
def test_the_committed_results_were_produced_by_the_committed_plan():
    """The plan hash exists so a reported number identifies the run behind it. A stale hash in the documentation
    defeats that, so the recorded hash is checked against the defaults rather than trusted.

    Regression: docs/gate_validation.md once quoted a hash captured before `variant_regime` was added to the plan.
    """
    import json
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec_path = os.path.join(here, "docs", "gate_validation_results", "spec.json")
    if not os.path.exists(spec_path):
        pytest.skip("no committed validation results in this checkout")
    with open(spec_path) as fh:
        spec = json.load(fh)
    assert spec["plan_hash"] == V.ValidationPlan().digest, (
        "the committed results were produced by a different plan than the current defaults: rerun "
        "tools/run_gate_validation.py, or say in the documentation which plan produced them")
    doc = open(os.path.join(here, "docs", "gate_validation.md"), encoding="utf-8").read()
    assert spec["plan_hash"] in doc, "the note quotes a plan hash that is not the one the results were run under"
