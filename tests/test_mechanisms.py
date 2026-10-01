"""Mechanism study: oracle and true-model variants, shared targets, metamodel honesty, end-to-end run."""
from dataclasses import replace

import numpy as np
import pandas as pd

from heiner_abm.arena import QUICK, Env, simulate, train_envs, tune
from heiner_abm.mechanisms import (MAP_RANGES, QUICK_STUDY, StudyPlan, default_plan, fit_metamodel, map_envs,
                                   run_study, theory_maps)


def test_oracle_that_always_deviates_equals_always():
    envs = train_envs(QUICK)[:4]
    bg = ["ruleb"] * 3
    ones = {f"e{b}": 1.0 for b in range(5)}
    for always, oracle in (("opt_br", "heiner_m_oracle"), ("cobweb_p", "heiner_p_oracle")):
        a = simulate(envs, np.array([[always] + bg] * 4, dtype=object), {}, 200, 20)
        o = simulate(envs, np.array([[oracle] + bg] * 4, dtype=object), {oracle: ones}, 200, 20)
        np.testing.assert_allclose(a["profit"], o["profit"])


def test_true_model_learner_equals_believed_without_regime_shifts():
    """Without demand shifts the believed and the true model coincide, so both learners behave identically."""
    env = Env(delta=10, c_max=80, q_range=1500, noise=2, foresight=0, hazard=0, belief_lag=20, seed=5)
    a = simulate([env], np.array([["heiner_m", "ruleb", "ruleb"]], dtype=object), {}, 250, 20)
    b = simulate([env], np.array([["heiner_m_true", "ruleb", "ruleb"]], dtype=object), {}, 250, 20)
    np.testing.assert_allclose(a["profit"], b["profit"])
    assert np.isfinite(a["K"][0, 0]) and np.isnan(a["K"][0, 1])


def test_map_envs_cover_the_ranges():
    plan = StudyPlan(n_envs=40)
    envs = map_envs(plan)
    for k, (lo, hi) in MAP_RANGES.items():
        v = np.array([getattr(e, k) for e in envs], float)
        assert v.min() >= lo - 0.5 and v.max() <= hi + 0.5
        assert np.histogram(v, bins=4, range=(lo, hi))[0].min() >= 8      # stratified: every quarter is covered


def test_metamodel_reports_out_of_sample_fit():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({k: rng.uniform(lo, hi, 80) for k, (lo, hi) in MAP_RANGES.items()})
    df["signal"] = df["delta"] / 35 + rng.normal(0, 0.05, 80)
    df["noise_only"] = rng.normal(0, 1, 80)
    assert fit_metamodel(df, "signal")["r2"] > 0.8
    assert fit_metamodel(df, "noise_only")["r2"] < 0.2


def test_plan_hash_and_quick_study_end_to_end():
    assert default_plan("a").digest != default_plan("b").digest
    tuned, _ = tune(QUICK)
    plan = replace(default_plan(QUICK.digest), **QUICK_STUDY)
    res = run_study(plan, tuned)
    assert res.exploratory and len(res.verdicts) == 5
    assert set(res.focal["variant"]) >= {"always", "band", "rc_learned", "rc_true_model", "rc_oracle"}
    gx, gy, preds, win, r2 = theory_maps(res.theories, "delta", "hazard", n=10)
    assert win.shape == (10, 10) and all(np.isfinite(v) for v in r2.values())
