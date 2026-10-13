"""Censored demand, bought information, and the contract that keeps them honest.

The failure this module exists to prevent is silent: a policy written for full feedback keeps reading total
demand in an experiment where total demand is supposed to be unobservable, and the experiment measures nothing
while looking fine. Most of what follows is about making that failure loud.
"""
import numpy as np
import pandas as pd
import pytest

from heiner_abm import information_env as E
from heiner_abm.information_contract import (InfoContract, InformationViolation, InfoView, RESEARCHER_ONLY)

ENV = E.EnvConfig(periods=300, burn_in=40, unmet_access=0.0)


# ============================================================================== the contract
def test_a_contract_cannot_grant_the_truth():
    for field in RESEARCHER_ONLY:
        with pytest.raises(InformationViolation):
            InfoContract("greedy", "Greedy", (field,))


def test_every_declared_contract_is_clean():
    for key, c in E.CONTRACTS.items():
        assert c.key == key and c.name.strip() and c.note.strip()
        assert not set(c.fields) & set(RESEARCHER_ONLY)


def test_a_view_refuses_what_the_contract_does_not_grant():
    record = {"sales": 10.0, "demand": 14.0, "mu": 12.0}
    view = InfoView(record, InfoContract("c", "C", ("sales",)))
    assert view["sales"] == 10.0
    for field in ("demand", "mu"):
        with pytest.raises(InformationViolation) as e:
            _ = view[field]
        assert "does not grant" in str(e.value)


def test_a_view_is_not_a_side_door():
    """Indexing is refused; so must be attribute access, iteration and membership, or the refusal is cosmetic."""
    record = {"sales": 10.0, "demand": 14.0}
    view = InfoView(record, InfoContract("c", "C", ("sales",)))
    with pytest.raises(InformationViolation):
        _ = view.demand
    with pytest.raises(InformationViolation):
        _ = view._record                       # the name the record used to live under is not reachable
    assert "demand" not in view._granted, "a view must not carry what it refuses to hand over"
    assert list(view) == ["sales"]
    assert "demand" not in view and "sales" in view
    assert dict(view) == {"sales": 10.0}, "copying the view must not copy the truth out with it"


def test_a_view_records_what_was_actually_read():
    record = {"sales": 1.0, "stocked_out": True, "demand": 9.0}
    view = InfoView(record, InfoContract("c", "C", ("sales", "stocked_out")))
    _ = view["sales"]
    assert view.reads == ("sales",), "a view must report what was used, not what was permitted"


# ============================================================================== the leak the brief names
def test_a_full_feedback_policy_cannot_keep_privileged_access_in_a_censored_run():
    """The requirement, tested directly: a policy that reaches for total demand in a censored experiment is
    stopped, not quietly served."""
    class GreedyLearner(E.CensoredLearner):
        def observe(self, view, t):
            self._update(float(view["demand"]))        # what a full-feedback policy would do

    original = E.CensoredLearner
    try:
        E.CensoredLearner = GreedyLearner
        with pytest.raises(InformationViolation) as e:
            E.simulate(ENV, "censored", 1)
        assert "demand" in str(e.value) and "censored" in str(e.value)
    finally:
        E.CensoredLearner = original


def test_a_policy_that_may_not_buy_cannot_be_given_a_buying_rule():
    with pytest.raises(InformationViolation):
        E.simulate(ENV, "censored", 1, acquisition=E.always_buy("audit"))
    assert E.simulate(ENV, "censored_buyer", 1, acquisition=E.always_buy("audit"))["purchase_rate"] > 0


def test_the_researcher_keeps_total_demand_for_evaluation():
    r = E.simulate(ENV, "censored", 1)
    p = r["path"]
    assert {"demand", "mu", "lost_sales", "oracle_order"} <= set(p.columns)
    assert r["researcher_mean_demand"] > 0


# ============================================================================== censoring
def test_sales_are_censored_by_the_agents_own_order():
    p = E.simulate(ENV, "censored", 2)["path"]
    assert np.allclose(p["sales"], np.minimum(p["demand"], p["order"]))
    assert np.allclose(p["lost_sales"], np.maximum(0.0, p["demand"] - p["order"]))
    out = p["stocked_out"].to_numpy(bool)
    assert (p.loc[out, "sales"] < p.loc[out, "demand"] + 1e-9).all()
    assert out.any() and not out.all(), "a useful censored run stocks out sometimes and not always"


@pytest.mark.parametrize("access", [0.0, 0.5, 1.0])
def test_unmet_demand_is_recorded_at_the_configured_rate(access):
    env = E.replace(ENV, unmet_access=access)
    p = E.simulate(env, "censored", 3)["path"]
    rate = float(p["lost_sales_reported"].notna().mean())
    if access == 0.0:
        assert rate == 0.0
    else:
        assert abs(rate - access) < 0.1


def test_ignoring_censoring_drives_the_order_down():
    """The classic failure: treating sales as demand when the shelf emptied biases the estimate down, which
    makes the next stockout likelier. The correction is what stops it."""
    good = [E.simulate(ENV, "censored", s, correct_censoring=True) for s in (1, 2, 3)]
    bad = [E.simulate(ENV, "censored", s, correct_censoring=False) for s in (1, 2, 3)]
    assert np.mean([r["mean_order"] for r in bad]) < np.mean([r["mean_order"] for r in good]) - 5
    assert np.mean([r["net_payoff"] for r in bad]) < np.mean([r["net_payoff"] for r in good])


def test_recording_unmet_demand_helps():
    poor = [E.simulate(E.replace(ENV, unmet_access=0.0), "censored", s)["net_payoff"] for s in (1, 2, 3, 4)]
    rich = [E.simulate(E.replace(ENV, unmet_access=1.0), "censored", s)["net_payoff"] for s in (1, 2, 3, 4)]
    assert np.mean(rich) > np.mean(poor)


# ============================================================================== instruments
def test_an_instrument_states_price_precision_delay_and_what_it_reveals():
    for key, ins in E.INSTRUMENTS.items():
        assert ins.key == key and ins.name.strip()
        assert ins.price >= 0 and ins.sd >= 0 and ins.delay >= 0
        assert ins.reveals in ("demand", "mean")
    assert {"forecast", "audit", "pilot"} <= set(E.INSTRUMENTS)


@pytest.mark.parametrize("bad", [dict(reveals="vibes"), dict(price=-1.0), dict(sd=-1.0), dict(delay=-1)])
def test_an_impossible_instrument_is_refused(bad):
    kw = dict(key="x", name="X", price=1.0, sd=1.0, delay=0, reveals="mean")
    kw.update(bad)
    with pytest.raises(ValueError):
        E.Instrument(**kw)


def test_a_delayed_instrument_arrives_late_and_still_arrives():
    """The pilot is precise but late: it cannot inform the decision it was bought for."""
    env = E.replace(ENV, periods=200)
    r = E.simulate(env, "censored_buyer", 4, acquisition=E.always_buy("pilot"))
    p = r["path"]
    assert (p["bought"] == "pilot").all()
    assert p["signal"].isna().all(), "a delayed instrument must not deliver a same-period signal"
    assert r["info_cost"] == pytest.approx(E.INSTRUMENTS["pilot"].price)


def test_an_immediate_instrument_delivers_in_time():
    r = E.simulate(ENV, "censored_buyer", 4, acquisition=E.always_buy("audit"))
    assert r["path"]["signal"].notna().mean() > 0.9


# ============================================================================== value of information
@pytest.mark.parametrize("sd", [0.0, 5.0, 10.0, 25.0, 40.0, 100.0])
def test_the_approximation_agrees_with_the_exact_small_case(sd):
    ex = E.voi_exact_two_point(80.0, 140.0, 0.4, sd)
    mc = E.voi_monte_carlo(80.0, 140.0, 0.4, sd, draws=200_000, seed=1)
    assert abs(mc["evsi"] - ex["evsi"]) <= 4 * mc["standard_error"] + 1e-9, (ex["evsi"], mc["evsi"])


def test_a_perfect_signal_is_worth_perfect_information():
    ex = E.voi_exact_two_point(80.0, 140.0, 0.4, 0.0)
    assert ex["evsi"] == pytest.approx(ex["evpi"])
    assert ex["evpi"] > 0


def test_a_useless_signal_is_worth_nothing():
    assert E.voi_exact_two_point(80.0, 140.0, 0.4, 1e6)["evsi"] == pytest.approx(0.0, abs=1e-3)


def test_value_falls_as_the_signal_gets_noisier():
    vals = [E.voi_exact_two_point(80.0, 140.0, 0.4, sd)["evsi"] for sd in (1.0, 10.0, 30.0, 60.0, 120.0)]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])), vals


def test_the_exact_solution_is_labelled_exact_and_the_others_are_not():
    assert E.voi_exact_two_point(80.0, 140.0, 0.4, 5.0)["exact"] is True
    mc = E.voi_monte_carlo(80.0, 140.0, 0.4, 5.0, draws=5_000)
    assert mc["exact"] is False and "approximation" in mc
    env_voi = E.voi_in_environment(E.replace(ENV, periods=200), "audit", 1)
    assert env_voi["exact"] is False and "Not an EVSI" in env_voi["approximation"]


# ============================================================================== the decomposition
def test_the_decomposition_is_additive():
    plan = E.InfoPlan(paths=2, periods=250, burn_in=40)
    env = E.EnvConfig(periods=250, burn_in=40, unmet_access=0.0)
    for seed in (plan.seed, plan.seed + 1000):
        a = E.run_arms(env, plan, seed)
        assert a["decomposition_residual"] == pytest.approx(0.0, abs=1e-9)
        assert a["net_of_buying"] == pytest.approx(
            a["value_of_information"] + a["value_of_choosing"] - a["price_paid"], abs=1e-9)


def test_the_free_control_is_matched_and_free():
    plan = E.InfoPlan(paths=1, periods=250, burn_in=40)
    env = E.EnvConfig(periods=250, burn_in=40)
    a = E.run_arms(env, plan, plan.seed)
    assert a["free"]["given_rate"] == pytest.approx(a["buyer"]["purchase_rate"], abs=0.05)
    assert a["free"]["info_cost"] == 0.0, "the control is given the instrument, not sold it"
    assert a["buyer"]["info_cost"] > 0


def test_adaptation_is_a_decision_so_an_adjustment_cost_can_act_on_it():
    """With no band the agent re-orders every period, an adjustment charge is a constant, and it cancels out of
    every comparison. The band is what makes the interaction measurable at all."""
    reflex = E.simulate(ENV, "censored", 1, band=0.0)
    deciding = E.simulate(ENV, "censored", 1, band=0.35)
    assert reflex["adaptation_rate"] > 0.95
    assert deciding["adaptation_rate"] < 0.5


# ============================================================================== exports
def test_the_four_exports_are_kept_apart():
    plan = E.InfoPlan(paths=1, periods=250, burn_in=40, unmet_access=(0.0,), adjustment_costs=(0.0,),
                      hazards=(0.01,))
    out = E.run_experiment(plan)
    assert set(out) == set(E.EXPORTS)
    truth_cols = {c for c in out["truth"].columns if c.startswith("researcher_")}
    assert truth_cols, "the truth export carries no researcher-only quantity"
    for name in ("decisions", "availability"):
        leaked = {c for c in out[name].columns if c.startswith("researcher_")}
        assert not leaked, f"{name} carries researcher-only fields: {leaked}"
    assert {"purchases", "information_cost_total"} <= set(out["decisions"].columns)
    assert {"unmet_recorded_rate", "signal_received_rate"} <= set(out["availability"].columns)


def test_the_plan_is_frozen_and_hashed():
    a, b = E.InfoPlan(), E.InfoPlan()
    assert a.digest == b.digest and len(a.digest) == 16
    assert E.InfoPlan(instrument="forecast").digest != a.digest
    with pytest.raises(Exception):
        a.paths = 3
