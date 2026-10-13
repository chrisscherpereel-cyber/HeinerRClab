"""Incremental construct validity of the proposed learnability index (heiner_abm.learnability_validity).

The module makes claims about an index it must not change, using data it must not peek at. Those two properties
are what most of these tests pin; the rest check that the incremental-validity test can actually tell a useful
covariate from a useless one, which is the thing a prediction contest is worthless without.
"""
import numpy as np
import pandas as pd
import pytest

from heiner_abm import learnability as L
from heiner_abm import learnability_validity as V

HP = {"gain": 0.1, "band": {"b": 0.5}, "gate": {"conf": 0.9, "nmin": 5.0},
      "bocpd": {"hazard": 0.005, "prior_sd": 30.0, "obs_sd": 25.0}, "dro": {"window": 20, "rho": 0.01}}
SMALL = V.ValidityPlan(n_configs=8, periods=300, burn_in=40, pilot_paths=3, eval_paths=3, n_boot=40,
                       matched_pairs=4)


# ============================================================================== the index is not redefined
def test_the_index_is_imported_unchanged_and_not_reimplemented():
    """The module analyses the study's index. If it quietly computed its own, every claim would be about a
    different quantity than the one the study registered."""
    assert V.INDEX_KEY == "log10_ratio"
    assert "learnability_ratio" in V.INDEX_NOTE and "unchanged" in V.INDEX_NOTE
    cfg = L.Config(family="jump", sigma=20.0, tau=10.0, hazard=0.01, avail=0.6)
    sds = V.seeds(SMALL, V.EXPLORATORY_TAG, 1, 3)
    pilot = V.pilot_summary(cfg, 0.1, SMALL, sds)
    direct = L.learnability_ratio(cfg, 0.1, SMALL.periods, sds, SMALL.burn_in)
    assert pilot[V.INDEX_KEY] == direct["log10_ratio"], "the measured index differs from the study's own"


def test_the_clip_probe_reproduces_the_index_exactly_at_the_registered_clip():
    """ratio_with_clip exists to vary the clip. At the registered clip it must BE the index, bit for bit, or the
    sensitivity analysis is of some other quantity."""
    for cfg in (L.Config(family="jump", sigma=20.0, tau=10.0, hazard=0.01, avail=0.6),
                L.Config(family="switch", sigma=8.0, tau=2.0, hazard=0.02, avail=1.0, default="slow"),
                L.Config(family="drift", sigma=30.0, tau=20.0, hazard=0.005, avail=0.4)):
        sds = V.seeds(SMALL, V.EXPLORATORY_TAG, 2, 3)
        a = L.learnability_ratio(cfg, 0.1, SMALL.periods, sds, SMALL.burn_in)["log10_ratio"]
        b = V.ratio_with_clip(cfg, 0.1, SMALL.periods, sds, SMALL.burn_in, 1.0, 1e6)
        assert a == b, (cfg.family, a, b)


def test_the_plan_hash_covers_the_index_definition():
    """Changing the index without changing the plan would otherwise leave the hash untouched."""
    plan = V.ValidityPlan()
    first = plan.digest
    saved = V.INDEX_NOTE
    try:
        V.INDEX_NOTE = saved + " (altered)"
        assert V.ValidityPlan().digest != first
    finally:
        V.INDEX_NOTE = saved
    assert V.ValidityPlan().digest == first


# ============================================================================== the confirmatory set is separate
def test_the_two_arms_use_disjoint_seed_blocks():
    plan = V.ValidityPlan()
    expl = set(V.seeds(plan, V.EXPLORATORY_TAG, 0, 50)) | set(V.seeds(plan, V.EXPLORATORY_TAG, 7, 50))
    conf = set(V.seeds(plan, V.CONFIRMATORY_TAG, 0, 50)) | set(V.seeds(plan, V.CONFIRMATORY_TAG, 7, 50))
    assert not expl & conf


def test_neither_arm_touches_the_registered_study_blocks():
    """heiner_abm.learnability uses tags 1 training, 2 pilot, 3 test, 4 negative controls."""
    plan = V.ValidityPlan()
    study = set()
    for tag in (1, 2, 3, 4):
        for j in (0, 1, 7, 50):
            study |= set(L._seeds(L.LearnPlan(seed=plan.seed), tag, j, 40))
    for tag in (V.EXPLORATORY_TAG, V.CONFIRMATORY_TAG):
        mine = set()
        for j in (0, 1, 7, 50, 50_000, 70_000, 80_000):
            mine |= set(V.seeds(plan, tag, j, 40))
        assert not mine & study, f"tag {tag} overlaps the registered study"


def test_a_tag_outside_this_modules_blocks_is_refused():
    with pytest.raises(ValueError):
        V.seeds(V.ValidityPlan(), 3, 0, 4)


def test_the_two_arms_draw_different_configurations():
    plan = V.ValidityPlan(n_configs=6)
    a = V._configs(plan, 6, V.EXPLORATORY_CONFIG_SEED, "jump")
    b = V._configs(plan, 6, V.CONFIRMATORY_CONFIG_SEED, "jump")
    assert [c.sigma for c in a] != [c.sigma for c in b]


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError):
        V.measure(SMALL, HP, "whichever_looks_better")


# ============================================================================== the comparison model
def test_every_quantity_the_brief_names_is_in_the_comparison_model():
    keys = set(V.COVARIATE_KEYS)
    for needed in ("adv_noise", "log_regime_len", "avail", "delay", "log_memory_window", "log_evidence",
                   "default_gap", "cost", "switch_cost", "magnitude_cost"):
        assert needed in keys, needed
    assert len(V.INTERACTIONS) >= 3
    for a, b in V.INTERACTIONS:
        assert a in keys and b in keys


def test_the_design_matrix_adds_exactly_one_column_for_the_index():
    df = _frame(24, seed=1)
    X0, n0 = V.design_matrix(df, with_index=False)
    X1, n1 = V.design_matrix(df, with_index=True)
    assert X1.shape[1] == X0.shape[1] + 1
    assert set(n1) - set(n0) == {V.INDEX_KEY}


def test_a_factor_held_fixed_is_dropped_rather_than_breaking_the_fit():
    df = _frame(20, seed=2)
    df["switch_cost"] = 0.0
    _, names = V.design_matrix(df, with_index=False)
    assert "switch_cost" not in names


# ============================================================================== the test can tell signal from noise
def _frame(n, seed=0, index_is="noise"):
    """A synthetic design: the comparison covariates, an outcome built from two of them, and an index column
    that is either pure noise or genuinely informative."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({k: rng.normal(size=n) for k in V.COVARIATE_KEYS})
    d["family"] = ["jump"] * (n // 2) + ["switch"] * (n - n // 2)
    d["config"] = list(range(n))
    extra = rng.normal(size=n)
    d["advantage"] = 2.0 * d["adv_noise"] - 1.5 * d["log_regime_len"] + 0.3 * rng.normal(size=n) + (
        3.0 * extra if index_is == "signal" else 0.0)
    d[V.INDEX_KEY] = extra if index_is == "signal" else rng.normal(size=n)
    for c in ("sigma", "tau", "hazard", "memory"):
        d[c] = np.abs(rng.normal(size=n)) + 0.1
    return d


def test_a_useless_index_is_not_reported_as_supported():
    """The negative control for the test itself. A column of noise must not buy a supported verdict."""
    d = _frame(60, seed=3, index_is="noise")
    out = V.incremental_validity(d, V.ValidityPlan(n_boot=120))
    assert not out["supported"], out["delta_r2"]
    assert out["delta_r2"] < 0.05


def test_a_genuinely_informative_index_is_detected():
    """The positive control. If the test cannot find a covariate that explains most of the outcome, a null
    result from it would mean nothing."""
    d = _frame(60, seed=4, index_is="signal")
    out = V.incremental_validity(d, V.ValidityPlan(n_boot=120))
    assert out["supported"] and out["delta_r2"] > 0.2, out


def test_the_verdict_never_claims_a_mechanism():
    d = _frame(60, seed=5, index_is="signal")
    inc = V.incremental_validity(d, V.ValidityPlan(n_boot=60))
    assert "NOT establish a distinct mechanism" in inc["claim"]
    rows = V.verdict_table(inc, {}, {}, {}, {}, "exploratory")
    v1 = rows[rows["id"] == "V1"].iloc[0]
    assert "summary index" in v1["reading"] and "mechanism" not in v1["reading"]


def test_an_exploratory_run_is_labelled_as_one():
    rows = V.verdict_table({}, {}, {}, {}, {}, "exploratory")
    assert (rows["status"] == "EXPLORATORY (not a confirmatory result)").all()
    assert (V.verdict_table({}, {}, {}, {}, {}, "confirmatory")["status"] == "confirmatory").all()


# ============================================================================== transfer and hierarchy
def test_transfer_fits_on_one_family_and_scores_on_the_others():
    d = _frame(60, seed=6, index_is="signal")
    out = V.family_transfer(d, V.ValidityPlan(n_boot=60), train_family="jump")
    assert out["n_train"] == (d["family"] == "jump").sum()
    assert out["test_families"] == ["switch"]
    assert out["n_train"] + out["n_test"] == len(d)


def test_pilot_resampling_widens_the_interval_rather_than_narrowing_it():
    """Propagating the uncertainty of estimating the index cannot make the answer more certain. If the inner
    loop ever tightened the interval, it would be averaging the index rather than resampling it."""
    plan = V.ValidityPlan(n_configs=6, periods=300, burn_in=40, pilot_paths=3, eval_paths=3, n_boot=80,
                          families=("jump",))
    df = V.measure(plan, HP, "exploratory")
    none = V.hierarchical_incremental_validity(plan, HP, df, "exploratory", inner=0, seed=1)
    some = V.hierarchical_incremental_validity(plan, HP, df, "exploratory", inner=4, seed=1)
    w0 = none["ci_hi"] - none["ci_lo"]
    w1 = some["ci_hi"] - some["ci_lo"]
    assert w1 >= w0 * 0.98, f"the interval narrowed when uncertainty was added: {w0:.4f} -> {w1:.4f}"


# ============================================================================== matched designs
def test_matched_pairs_hold_the_right_thing_fixed():
    d = _frame(40, seed=7, index_is="signal")
    plan = V.ValidityPlan(n_boot=60, matched_pairs=5)
    d1 = V.matched_simple_uncertainty(d, plan)
    assert d1["pairs"] > 0
    t = d1["table"]
    assert (t["index_gap"] > 0).all()
    d2 = V.matched_same_index(d, plan)
    assert d2["pairs"] > 0
    assert (d2["table"]["family_a"] != d2["table"]["family_b"]).all(), "a pair came from one family"


def test_sufficiency_is_reported_and_never_asserted():
    d = _frame(40, seed=8, index_is="noise")
    out = V.matched_same_index(d, V.ValidityPlan(n_boot=60, matched_pairs=5))
    assert "supported" not in out, "design 2 must not report a pass/fail verdict"
    assert "relative_to_outcome_sd" in out
    # the reading must keep sufficiency and usefulness apart rather than collapsing them into one verdict
    assert "SUFFICIENT" in out["reading"] and "predicts" in out["reading"]


# ============================================================================== robustness probes
def test_every_probe_the_brief_names_is_present():
    plan = V.ValidityPlan()
    probes = {k for k, _ in plan.robustness} | {"near_zero", "clip"}
    for needed in ("obs_corr", "obs_bias", "memory", "near_zero", "clip"):
        assert needed in probes, needed
    for k in probes:
        assert V.ROBUSTNESS_NOTES[k].strip()


def test_the_feedback_properties_the_probes_need_are_inert_by_default():
    c = L.Config()
    assert c.obs_delay == 0 and c.obs_bias == 0.0 and c.obs_corr == 0.0


@pytest.mark.parametrize("bad", [dict(obs_delay=-1), dict(obs_corr=1.5), dict(obs_corr=-2.0)])
def test_an_impossible_feedback_setting_is_refused(bad):
    with pytest.raises(ValueError):
        L.Config(**bad)


def test_correlated_noise_keeps_its_scale_and_gains_its_dependence():
    base = L.paths(L.Config(tau=10.0, avail=1.0, sigma=1e-9), 4000, 5)
    corr = L.paths(L.Config(tau=10.0, avail=1.0, sigma=1e-9, obs_corr=0.9), 4000, 5)
    e0, e1 = base["y"] - base["d"], corr["y"] - corr["d"]
    assert abs(float(np.std(e1)) - float(np.std(e0))) < 1.5, "the AR(1) filter changed the marginal scale"
    assert float(np.corrcoef(e1[:-1], e1[1:])[0, 1]) > 0.8
    assert abs(float(np.corrcoef(e0[:-1], e0[1:])[0, 1])) < 0.1


def test_a_delay_withholds_the_earliest_observations_and_leaves_the_truth_alone():
    a = L.paths(L.Config(avail=1.0), 200, 6)
    b = L.paths(L.Config(avail=1.0, obs_delay=4), 200, 6)
    assert np.array_equal(a["mu"], b["mu"]) and np.array_equal(a["d"], b["d"])
    assert np.isnan(b["y"][:4]).all()
    assert np.allclose(b["y"][4:], a["y"][:-4], equal_nan=True)


def test_a_bias_moves_the_observation_and_not_the_demand():
    a = L.paths(L.Config(avail=1.0, tau=0.0), 500, 7)
    b = L.paths(L.Config(avail=1.0, tau=0.0, obs_bias=7.0), 500, 7)
    assert np.array_equal(a["d"], b["d"])
    assert float(np.nanmean(b["y"] - a["y"])) == pytest.approx(7.0)
