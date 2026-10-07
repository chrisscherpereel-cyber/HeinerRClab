"""Learnability study (heiner_abm.learnability, heiner_abm.learnability_market): shared exogenous paths, information
timing, missing feedback, the ratio construct's measurement, disjoint data sets, negative-control setups and the frozen
specification."""
from dataclasses import replace

import numpy as np
import pytest

from heiner_abm import learnability as L
from heiner_abm import learnability_market as LM
from heiner_abm import registered

HP = {"gain": 0.2, "band": {"b": 0.5}, "bocpd": {"hazard": 0.01, "prior_sd": 40.0, "obs_sd": 25.0},
      "dro": {"window": 30, "rho": 0.05}}


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
