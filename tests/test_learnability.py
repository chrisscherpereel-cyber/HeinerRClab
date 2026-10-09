"""Learnability study (heiner_abm.learnability, heiner_abm.learnability_market): shared exogenous paths, information
timing, missing feedback, the ratio construct's measurement, disjoint data sets, negative-control setups and the frozen
specification."""
from dataclasses import replace

import numpy as np
import pytest

from heiner_abm import learnability as L
from heiner_abm import learnability_market as LM
from heiner_abm import registered

HP = {"gain": 0.2, "band": {"b": 0.5}, "gate": {"conf": 0.9, "nmin": 5.0},
      "bocpd": {"hazard": 0.01, "prior_sd": 40.0, "obs_sd": 25.0}, "dro": {"window": 30, "rho": 0.05}}


def test_specification_is_frozen_and_not_claimed_as_external_preregistration():
    assert L.LEARN_PREREG.digest == registered.LEARN_PLAN, "study code or plan changed: register a new plan"
    doc = L.registration_document(L.LEARN_PREREG, registered.LEARN_PLAN)
    assert registered.LEARN_PLAN in doc and "not** been preregistered" in doc
    for h in ("H1", "H2", "H3", "H4", "NC1", "NC2", "NC3"):
        assert h in doc


@pytest.mark.parametrize("family", L.FAMILIES)
def test_paths_are_reproducible_and_shared(family):
    cfg = L.Config(family=family, avail=0.5)
    a, b = L.paths(cfg, 400, 7), L.paths(cfg, 400, 7)
    for k in ("mu", "d", "seen", "change"):
        np.testing.assert_array_equal(a[k], b[k])
    assert np.all((a["mu"] >= L.MU_LO) & (a["mu"] <= L.MU_HI))
    assert np.isnan(a["y"][~a["seen"]]).all() and not np.isnan(a["y"][a["seen"]]).any()


def test_factors_change_only_their_own_part_of_the_path():
    base = L.paths(L.Config(), 300, 3)
    for kw in (dict(tau=30.0), dict(avail=0.3)):
        other = L.paths(L.Config(**kw), 300, 3)
        np.testing.assert_array_equal(base["mu"], other["mu"])
        np.testing.assert_array_equal(base["d"], other["d"])
    np.testing.assert_array_equal(base["mu"], L.paths(L.Config(sigma=40.0), 300, 3)["mu"])


@pytest.mark.parametrize("policy", ["band", "gate_gain", "gate_lcb", "bocpd", "dro", "always"])
def test_orders_use_only_past_observations(policy):
    cfg = L.Config(avail=0.7, hazard=0.02)
    p = L.paths(cfg, 300, 5)
    K = 180
    q = {k: v.copy() for k, v in p.items()}
    q["y"][K:] = np.where(np.isnan(q["y"][K:]), np.nan, q["y"][K:] + 40.0)
    q["d"][K:] += 40.0
    a = L.policy_orders(policy, cfg, p, L.candidate_orders(cfg, p, HP["gain"]), HP)[0]
    b = L.policy_orders(policy, cfg, q, L.candidate_orders(cfg, q, HP["gain"]), HP)[0]
    np.testing.assert_array_equal(a[:K + 1], b[:K + 1])


def test_unobserved_periods_give_the_gates_no_evidence():
    cfg = L.Config(avail=0.3)
    p = L.paths(cfg, 400, 9)
    c = L.candidate_orders(cfg, p, HP["gain"])
    _, pred, gobs = L.gated(cfg, p, c, "lcb")
    assert np.all(np.isnan(gobs[~p["seen"]]))
    informative = p["seen"] & (c["S_F"] != c["S_D"])
    assert np.isfinite(gobs[informative]).all()


def test_ratio_reacts_to_feedback_availability_and_regime_frequency_as_constructed():
    seeds = [101, 102, 103]
    r = lambda **kw: L.learnability_ratio(L.Config(**kw), 0.2, 1200, seeds, 100)
    full, scarce = r(avail=1.0), r(avail=0.2)
    assert scarce["n_avail"] < full["n_avail"] and scarce["ratio"] > full["ratio"]
    assert r(hazard=0.05)["n_avail"] < r(hazard=0.003)["n_avail"]
    # the sign of a large advantage needs few observations: a dominated default makes R tiny
    assert r(default="dominated")["n_needed"] < r(default="slow")["n_needed"]


def test_seed_blocks_are_disjoint():
    plan = L.LEARN_PREREG
    blocks = [set(L._seeds(plan, tag, j, 50)) for tag in (1, 2, 3, 4) for j in (0, 1, 999, 2000, 3100)]
    allseeds = [s for b in blocks for s in b]
    assert len(allseeds) == len(set(allseeds))
    mplan = LM.MarketPlan()
    m = [s for tag in (1, 2, 3) for j in range(20) for s in LM._seeds(mplan, tag, j, 10)]
    assert len(m) == len(set(m))


def test_negative_control_setups():
    cfg = L.Config(perfect=True, cost=0.0)
    p = L.paths(cfg, 300, 4)
    c = L.candidate_orders(cfg, p, HP["gain"])
    oracle = L.policy_orders("oracle", cfg, p, c, HP)[0]
    np.testing.assert_array_equal(L.policy_orders("always", cfg, p, c, HP)[0], oracle)
    dom = L.candidate_orders(replace(cfg, default="dominated"), p, HP["gain"])
    assert np.all(dom["S_D"] == 0)


def test_quick_study_runs_end_to_end():
    plan = replace(L.LEARN_PREREG, **{**L.QUICK_LEARN, "periods": 300, "paths": 2, "n_test_configs": 6,
                                      "n_new_family_configs": 3, "n_boot": 100})
    res = L.run_study(plan, market=False)
    assert res.plan_hash != registered.LEARN_PLAN                  # quick runs are exploratory
    assert set(res.verdicts["id"]) == {"H1", "H2", "H3"}
    assert set(res.controls["id"]) == {"NC1", "NC2", "NC3"}
    cols = {"net_payoff", "regret", "cvar5", "adaptation_rate", "missed", "calib_slope", "recovery_delay"}
    assert cols <= set(res.paths.columns)
    assert set(res.paths["policy"]) == set(L.POLICIES)


def test_market_replication_runs():
    mp = LM.MarketPlan(**{**LM.QUICK_MARKET, "periods": 250, "n_test_configs": 3, "n_train_configs": 1, "paths": 2,
                          "pilot_paths": 2, "band_grid": (25.0,)})
    out = LM.run_market(mp)
    t = out["tests"]
    assert len(t) == 3 and {"log10_ratio", "adv_gate_lcb", "adv_gate_gain", "adv_band"} <= set(t.columns)
    assert set(out["not_applicable"]) == {"bocpd", "dro"}


# ---------------------------------------------------------------- equal tuning budget (7 October 2026)
def test_every_tunable_policy_gets_the_same_number_of_candidates():
    """Policies are compared on equivalent computational budgets: one candidate count for every policy that has
    hyperparameters. Until 6 October 2026 the two gates got none while their four comparators got four each."""
    plan = replace(L.LEARN_PREREG, **{**L.QUICK_LEARN, "periods": 200, "n_train_configs": 2, "train_paths": 1})
    _, log = L.tune(plan)
    assert set(log["policy"]) == set(L.TUNABLE)
    assert log["n_candidates"].nunique() == 1 and log["n_candidates"].iloc[0] == len(dict(plan.tuning_grid)["gain"])
    # the grid itself must offer the same number of settings to each of them
    grid = dict(plan.tuning_grid)
    assert len({len(grid[k]) for k in L.TUNABLE.values()}) == 1
    # candidate 0 of the gate is the setting used before it had a budget, so the search can only help
    assert tuple(grid["gate"][0]) == (L.GATE_HP0["conf"], L.GATE_HP0["nmin"])
    assert (log["gain_over_candidate0"] >= -1e-9).all()


def test_the_plans_gate_settings_are_read_rather_than_shadowed_by_defaults():
    """LearnPlan.gate_confidence / gate_min_evidence were in the frozen plan but never reached the gate; the legacy
    path must now actually use them."""
    plan = replace(L.LEARN_PREREG, **{**L.QUICK_LEARN, "periods": 200, "n_train_configs": 1, "train_paths": 1},
                   tune_gate=False, gate_confidence=0.95, gate_min_evidence=1e9)
    hp, log = L.tune(plan)
    assert hp["gate"] == {"conf": 0.95, "nmin": 1e9}
    assert int(log.loc[log["policy"] == "gate_lcb", "n_candidates"].iloc[0]) == 0
    cfg = L.Config()
    p = L.paths(cfg, 300, 11)
    c = L.candidate_orders(cfg, p, hp["gain"])
    # an unreachable minimum evidence means the confidence gate can never adopt: it is exactly the default
    S = L.policy_orders("gate_lcb", cfg, p, c, hp)[0]
    np.testing.assert_array_equal(S, c["S_D"])


def test_gate_hyperparameters_change_behavior_through_policy_orders():
    cfg = L.Config(hazard=0.02)
    p = L.paths(cfg, 400, 13)
    c = L.candidate_orders(cfg, p, HP["gain"])
    loose = L.policy_orders("gate_lcb", cfg, p, c, {**HP, "gate": {"conf": 0.6, "nmin": 2.0}})[0]
    strict = L.policy_orders("gate_lcb", cfg, p, c, {**HP, "gate": {"conf": 0.99, "nmin": 50.0}})[0]
    assert (loose != c["S_D"]).sum() > (strict != c["S_D"]).sum()
    # omitting the key falls back to the registered candidate 0, not to an arbitrary value
    np.testing.assert_array_equal(L.policy_orders("gate_lcb", cfg, p, c, {k: v for k, v in HP.items() if k != "gate"})[0],
                                  L.policy_orders("gate_lcb", cfg, p, c, {**HP, "gate": L.GATE_HP0})[0])


def test_market_replication_tunes_the_gate_with_the_same_budget_as_the_band():
    mp = LM.MarketPlan(**{**LM.QUICK_MARKET, "periods": 200, "n_test_configs": 1, "n_train_configs": 1, "paths": 2,
                          "pilot_paths": 2, "band_grid": (10.0, 25.0), "gate_grid": ((0.9, 5.0), (0.8, 3.0))})
    assert len(mp.band_grid) == len(mp.gate_grid)
    out = LM.run_market(mp)
    assert len(out["band_scores"]) == len(out["gate_scores"]) == 2
    assert out["gate_hp"] in mp.gate_grid
    assert out["tests"]["gate_confidence"].iloc[0] == out["gate_hp"][0]


def test_market_gate_settings_reach_the_scenario():
    cfg = LM.MarketConfig()
    s = LM.scenario(cfg, "gate_lcb", 1, 200, 20, 5, 25.0, (0.8, 7.0))
    assert (s.adaptive.gate, s.adaptive.confidence, s.adaptive.min_evidence) == ("lcb", 0.8, 7.0)


# ================================================================ defect repairs (8 October 2026)
# Each test below fails on the code as it stood before the repair it names.

def test_bocpd_applies_exactly_one_hazard_transition_per_period():
    """Calendar time advances once per period. After k unobserved periods the probability that the original run is
    still unbroken at the next prediction must be (1 - H)^(k+1): no missed and no duplicated transition."""
    from heiner_abm.bench_inventory import BOCPD
    H = 0.01
    for k in (0, 1, 2, 5, 20):
        m = BOCPD(H, 100.0, 30.0, 25.0, rmax=500)
        m.update(100.0)
        for _ in range(k):
            m.advance()
        w = m.predictive()[0]
        assert np.isclose(w.sum(), 1.0)
        assert np.isclose(w[k], (1 - H) ** (k + 1), rtol=1e-12), f"k={k}"


def test_bocpd_zero_hazard_makes_missing_periods_a_no_op():
    """With no hazard a change cannot happen, so unobserved periods carry no information either way: the predictive
    distribution over the next demand must be untouched. (The run length still advances with calendar time, which is
    why this compares the predictive distribution rather than the run-length index.)"""
    from heiner_abm.bench_inventory import BOCPD

    def moments(m):
        w, mu, var = m.predictive()
        mean = float((w * mu).sum())
        return mean, float((w * (var + mu ** 2)).sum() - mean ** 2)

    a = BOCPD(0.0, 100.0, 30.0, 25.0, rmax=500)
    b = BOCPD(0.0, 100.0, 30.0, 25.0, rmax=500)
    for m in (a, b):
        m.update(123.0)
    before = moments(a)
    for _ in range(10):
        b.advance()
    np.testing.assert_allclose(moments(b), before, rtol=1e-12)
    assert np.isclose(b.predictive()[0][10], 1.0)        # all the mass is still on the one unbroken run
    assert np.isclose(b.predictive()[0].sum(), 1.0)


def test_bocpd_consecutive_missing_periods_accumulate_regime_uncertainty():
    """The defect: with positive hazard a gap in the data left the run-length posterior frozen, so predictive
    uncertainty did not grow and the model acted as if no time had passed."""
    from heiner_abm.bench_inventory import BOCPD

    def predictive_sd(k):
        m = BOCPD(0.05, 100.0, 30.0, 25.0, rmax=500)
        m.update(100.0)
        for _ in range(k):
            m.advance()
        w, mu, var = m.predictive()
        return float(np.sqrt((w * (var + mu ** 2)).sum() - (w * mu).sum() ** 2))

    sds = [predictive_sd(k) for k in (0, 1, 3, 10)]
    assert all(b > a for a, b in zip(sds, sds[1:])), sds


def test_bocpd_orders_unchanged_on_a_fully_observed_path_and_react_to_gaps():
    """A fully observed path must be unaffected by the repair (no advance() is ever reached), while a path with
    missing demands must now give different orders from one where the gaps are simply skipped."""
    cfg = L.Config(avail=1.0, hazard=0.02)
    p = L.paths(cfg, 300, 21)
    assert not np.isnan(p["y"]).any()
    full = L.bocpd_orders(p["y"], 0.01, 30.0, 25.0)

    def skipping(y, hazard, s0, sm, rmax=150):           # the pre-repair behavior
        from heiner_abm.bench_inventory import BOCPD, gaussian_mixture_quantile
        model = BOCPD(hazard, L.MU0, s0, sm, rmax)
        out = np.empty(len(y))
        for t in range(len(y)):
            out[t] = max(0.0, gaussian_mixture_quantile(*model.predictive()))
            if not np.isnan(y[t]):
                model.update(float(y[t]))
        return out

    np.testing.assert_array_equal(full, skipping(p["y"], 0.01, 30.0, 25.0))
    gappy = L.paths(L.Config(avail=0.4, hazard=0.02), 300, 21)
    assert np.isnan(gappy["y"]).any()
    assert not np.array_equal(L.bocpd_orders(gappy["y"], 0.01, 30.0, 25.0),
                              skipping(gappy["y"], 0.01, 30.0, 25.0))


def test_nc3_is_a_noninferiority_test_on_the_lower_confidence_limit():
    """The defect: the control passed whenever the *upper* limit exceeded the margin, which only fails when the gate
    is confidently much worse. The estimand is the gate's advantage; noninferiority needs the lower limit."""
    import inspect
    src = inspect.getsource(L.negative_controls)
    assert "lo > -NC3_MARGIN" in src and "hi > -0.25" not in src
    assert L.NC3_MARGIN == 0.25
    # the rule must separate these three cases correctly
    decide = lambda lo: bool(lo > -L.NC3_MARGIN)
    assert decide(+0.00) and decide(-0.10)                       # clearly within the margin
    assert not decide(-0.30) and not decide(-5.0)                # worse than the margin, or far worse
    spec = next(c for c in L.LEARN_PREREG.negative_controls if c[0] == "NC3")[2]
    assert "lower" in spec and "margin" in spec


def test_market_ratio_counts_feedback_by_release_period_within_the_regime():
    """The defect: feedback was counted in the regime of the *decision*, so a judgement window that only matured
    after the regime ended still counted as available within it. With a long observation delay, less feedback can
    arrive in time, so n_avail must fall and R must rise."""
    plan = LM.MarketPlan(periods=500, burn_in=50, window=5)
    seeds = LM._seeds(plan, 2, 0, 2)
    prompt = LM.ratio(LM.MarketConfig(hazard=0.03, delay=0), seeds, plan, 25.0)
    delayed = LM.ratio(LM.MarketConfig(hazard=0.03, delay=10), seeds, plan, 25.0)
    assert delayed["n_avail"] < prompt["n_avail"], (prompt["n_avail"], delayed["n_avail"])
    assert delayed["ratio"] > prompt["ratio"], (prompt["ratio"], delayed["ratio"])
    import inspect
    doc = inspect.getdoc(LM.ratio)
    assert "released" in doc and "crosses a regime boundary" in doc
