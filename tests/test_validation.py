"""Rule choice, field patterns, calibration, the human experiment and the generalization tasks."""
import os
from dataclasses import replace

import numpy as np
import pytest

from heiner_abm import registered
from heiner_abm.arena import Env, simulate
from heiner_abm.calibration import (FORECAST_RULES, QUANTITY_RULES, fit_subjects, prepare_forecasts,
                                    prepare_quantities, summarise, synthetic_cournot_data, synthetic_forecast_data)
from heiner_abm.experiment import SHADOWS
from heiner_abm.patterns import PATTERNS, run_patterns
from heiner_abm.rulechoice import (QUICK_CHOICE, RESTRICTED, RULES, default_choice_plan, level_env, run_choice_study,
                                   simulate_choice)
from heiner_abm.stepper import InteractiveMarket
from heiner_abm.tasks import LAYERS, QUICK_TASKS, TaskPlan, _decisions, _env_set, _layer, oracle_table, run_tasks


# ------------------------------------------------------------------------------------------------ stepped market
@pytest.mark.parametrize("design", ["heiner_p", "opt_br", "cobweb_p", "ruleb"])
def test_stepper_reproduces_simulate(design):
    """A participant who always plays a design's proposal reproduces the batch simulator exactly."""
    env = Env(delta=20.0, c_max=80.0, q_range=1500.0, noise=5.0, foresight=0.0, hazard=0.03, belief_lag=10, seed=11)
    rivals = ("cobweb_p", "heur_markup", "options_p")
    T = 60
    m = InteractiveMarket(env, rivals, registered.TUNED_PARAMS, [design], T)
    while not m.done:
        m.info()
        m.step(m.pending["shadows"][design])
    log = m.frame()
    params = {k: {n: np.asarray(v, float) for n, v in p.items()} for k, p in registered.TUNED_PARAMS.items()}
    out = simulate([env], np.array([[design, *rivals]], dtype=object), params, T, burn_in=1)
    assert np.isclose(log["profit"].mean(), out["profit"][0, 0])


# ------------------------------------------------------------------------------------------------ rule choice
def test_rule_choice_random_at_zero_intensity():
    """With beta = 0 rule choice is random, so the restricted share stays near its expected value of one half."""
    envs = [level_env(1.0, 50 + j) for j in range(6)]
    out = simulate_choice(envs, registered.TUNED_PARAMS, 0.0, 12, 400, 50, 0.05, 0.95, {})
    assert abs(out["restricted_share"].mean() - len(RESTRICTED) / len(RULES)) < 0.1
    assert out["rule_share"].shape == (6, len(RULES))
    assert np.allclose(out["rule_share"].sum(1), 1.0)


def test_rule_choice_selection_moves_shares():
    """With strong selection, rule shares depart from uniform."""
    envs = [level_env(0.0, 70 + j) for j in range(4)]
    out = simulate_choice(envs, registered.TUNED_PARAMS, 16.0, 12, 500, 50, 0.1, 0.9, {})
    assert out["rule_share"].max(1).mean() > 1.5 / len(RULES)


def test_rule_choice_study_quick():
    plan = replace(default_choice_plan(registered.TOURNAMENT_PLAN), **QUICK_CHOICE)
    res = run_choice_study(plan, registered.registered_tuned())
    assert list(res.verdicts["id"]) == ["E1", "E2", "E3", "E4"]
    assert len(res.runs) == len(plan.levels) * len(plan.betas) * plan.reps


def test_rule_choice_plan_registered():
    assert default_choice_plan(registered.TOURNAMENT_PLAN).digest == registered.CHOICE_PLAN


# ------------------------------------------------------------------------------------------------ field patterns
def test_patterns_run():
    res = run_patterns(registered.registered_tuned(), n_envs=3, periods=250)
    assert list(res["key"]) == [p.key for p in PATTERNS]
    assert res.set_index("key").loc["cycles", "passed"]        # naive expectations on steep demand cycle


# ------------------------------------------------------------------------------------------------ calibration
def test_calibration_recovers_forecast_rules():
    df, truth = synthetic_forecast_data(n_groups=3, seed=1)
    _, best = summarise(fit_subjects(prepare_forecasts(df), FORECAST_RULES, "forecast", {}))
    acc = (best.merge(truth, on=["group", "subject"]).eval("rule == true_rule")).mean()
    assert acc > 3 / len(FORECAST_RULES)


def test_calibration_recovers_quantity_rules():
    df, truth = synthetic_cournot_data(n_groups=4, seed=1)
    fits = fit_subjects(prepare_quantities(df, 100.0, 1.0, 1.0), QUANTITY_RULES, "quantity",
                        dict(a=100.0, b=1.0, c=1.0, n=4))
    rank, best = summarise(fits)
    acc = (best.merge(truth, on=["group", "subject"]).eval("rule == true_rule")).mean()
    assert acc > 3 / len(QUANTITY_RULES)
    assert (fits["test_rmse"] >= 0).all() and set(rank["rule"]) == {r.key for r in QUANTITY_RULES}


# ------------------------------------------------------------------------------------------------ generalization
@pytest.mark.parametrize("task", ["inventory", "learning"])
def test_task_layers_consistent(task):
    env = _env_set(task, 4, 99)[1]
    dec = _decisions(env, 800, env.seed)
    r = {k: _layer(dec, k, 100, band=1.0, table=np.full(5, 1.0)) for k in ("always", "ruleb", "rc_oracle", "band")}
    # an oracle that finds every deviation worthwhile is the always-deviating rule
    assert r["rc_oracle"]["payoff"] == pytest.approx(r["always"]["payoff"])
    assert r["ruleb"]["deviation_rate"] == 0.0
    assert r["band"]["deviation_rate"] <= r["always"]["deviation_rate"]
    # an oracle that never deviates is rule B
    assert _layer(dec, "rc_oracle", 100, table=np.full(5, -1.0))["payoff"] == pytest.approx(r["ruleb"]["payoff"])
    assert np.isfinite(oracle_table(env, replace(TaskPlan(), oracle_periods=2000))).any()


def test_tasks_quick_study():
    plan = replace(TaskPlan(), **{**QUICK_TASKS, "n_envs": 12, "n_train": 4})
    res = run_tasks(plan)
    assert set(res.runs["layer"]) == set(LAYERS)
    assert list(res.verdicts["id"]) == ["G1", "G2", "G3", "G4"]


def test_task_plan_registered():
    assert TaskPlan().digest == registered.TASK_PLAN


# ------------------------------------------------------------------------------------------------ focal theory
def test_every_theory_has_a_prediction_for_every_hypothesis():
    from heiner_abm.focal import FOCAL_KEYS, STATEMENTS, alternative_owners, prediction
    import heiner_abm.literature as lit
    for h in lit.HYPOTHESES:
        for k in FOCAL_KEYS:
            text, src = prediction(h.hid, k)
            assert text and src in {"registry", "tournament", "theory", "general"}
        assert prediction(h.hid, "heiner") == (h.rc_prediction, "registry")
        for k in alternative_owners(h.alt_label):
            assert prediction(h.hid, k) == (h.alt_prediction, "registry")
    for theory, d in STATEMENTS.items():
        assert theory in FOCAL_KEYS and set(d) <= set(lit.HYPOTHESIS_BY_ID), theory


def test_focal_theory_changes_hypothesis_card():
    from streamlit.testing.v1 import AppTest
    app = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
    at = AppTest.from_file(app, default_timeout=120)
    at.session_state["focal_theory"] = "heuristic"
    at.run()
    at.switch_page("app_pages/tracking.py").run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Simple heuristics" in md and "(theory under test)" in md


# ------------------------------------------------------------------------------------------------ tracking benchmark
def test_kalman_gain_minimises_analytic_loss():
    from heiner_abm.tracking import kalman_gain, loss_of_speed
    for snr in (0.01, 0.1, 1.0, 10.0):
        grid = np.linspace(0.001, 1.0, 100000)
        assert abs(grid[np.argmin(loss_of_speed(grid, snr, 1.0))] - kalman_gain(snr, 1.0)) < 1e-3


def test_optimal_offset_symmetric_and_lopsided():
    from heiner_abm.tracking import optimal_offset
    assert optimal_offset(1.0)[0] == pytest.approx(0.0, abs=1e-3)
    assert optimal_offset(1.0)[1] == pytest.approx(1.0, abs=1e-3)
    off, loss = optimal_offset(8.0)
    assert off < 0 and loss < 1.0


def test_tracking_quick_study():
    from heiner_abm.tracking import QUICK_TRACK, TrackPlan, run_tracking
    res = run_tracking(replace(TrackPlan(), **QUICK_TRACK))
    assert list(res.verdicts["id"]) == ["T1", "T2", "T3", "T4"]
    s = res.verify_summary
    assert (np.abs(s["simulated_best"] - s["kalman_gain"]) <= 2 * s["grid_step"] + 0.06).all()
    assert set(res.stakes["family"]) == {"filter", "filter_offset", "speed_sym", "speed_asym", "restrict"}


def test_tracking_plan_registered():
    from heiner_abm.tracking import TrackPlan
    assert TrackPlan().digest == registered.TRACK_PLAN
