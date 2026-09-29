"""The readable agent model and the vectorised engine must produce identical trajectories."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heiner_abm.agents import Industry, reflect  # noqa: E402
from heiner_abm.engine import run_batch  # noqa: E402
from heiner_abm.params import FirmSpec, Scenario, default_scenario, EvolutionParams  # noqa: E402


def _mixed():
    firms = [
        FirmSpec("Bertrand", 0.5, "Always"),
        FirmSpec("Bertrand", 1.0, "Small", threshold=10),
        FirmSpec("Cournot", 0.6, "Large", threshold=8, foresight=0.5),
        FirmSpec("Cournot", 0.3, "Adaptive", noise=2.0),
        FirmSpec("Bertrand", 0.8, "Never"),
    ]
    s = Scenario(firms=firms, periods=400, burn_in=25, seed=7)
    s.firm_globals.flex_cost_slope = 50
    s.firm_globals.fixed_cost = 20
    s.firm_globals.margin_includes_fixed = True
    return s


CASES = [
    default_scenario(periods=300, seed=1),
    default_scenario(periods=300, seed=2).copy(firms=[FirmSpec("Cournot", f) for f in (0.25, 0.5, 0.75, 1.0)]),
    _mixed(),
]


@pytest.mark.parametrize("cont", ["default", "rules"])
@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("scn", CASES)
def test_agent_engine_equivalence(scn, horizon, cont):
    scn = scn.copy(periods=min(scn.periods, 120))
    ind = Industry(scn, horizon=horizon, continuation=cont).run()
    res = run_batch([scn], record_firm_history=True, horizon=horizon, continuation=cont)
    np.testing.assert_allclose(res.price[0], ind.price)
    np.testing.assert_allclose(res.quantity[0], ind.quantity)
    np.testing.assert_allclose(res.cost[0], ind.market.costs)
    for k in ("q", "rec", "profit", "profit_rule", "profit_default"):
        np.testing.assert_allclose(res.firm_hist[k][0, 1:], ind.h[k][1:], err_msg=k)
    np.testing.assert_array_equal(res.firm_hist["deviate"][0, 1:], ind.h["deviate"][1:])


def test_evolution_equivalence():
    scn = default_scenario(periods=600, seed=3)
    scn.evolution = EvolutionParams(enabled=True, every=40, imitation_prob=0.6, mutation_sd=0.05)
    ind = Industry(scn).run()
    res = run_batch([scn], record_firm_history=True)
    np.testing.assert_allclose(res.flex_path[0], ind.h["flex"])
    np.testing.assert_allclose(res.price[0], ind.price)


def test_batch_equals_individual_runs():
    scns = [default_scenario(periods=200, seed=s) for s in range(4)]
    together = run_batch(scns)
    for i, s in enumerate(scns):
        alone = run_batch([s])
        np.testing.assert_allclose(together.price[i], alone.price[0])
        np.testing.assert_allclose(together.acc["sum_profit"][i], alone.acc["sum_profit"][0])


def test_reflecting_boundary_matches_vba_example():
    # paper: cost 75, +15 change, upper bound 80 -> 5 up to the bound, 10 back down = 70
    assert reflect(90.0, 10.0, 80.0) == 70.0
    assert reflect(5.0, 10.0, 80.0) == 15.0
    assert reflect(50.0, 10.0, 80.0) == 50.0


def test_cost_stays_in_bounds():
    s = default_scenario(periods=5000, seed=11)
    s.market.delta = 30
    r = run_batch([s])
    assert r.cost.min() >= s.market.p_min and r.cost.max() <= s.market.c_max


def test_never_firm_never_moves():
    s = default_scenario(periods=200, seed=5)
    s.firms[0].selection = "Never"
    r = run_batch([s], record_firm_history=True)
    assert np.all(r.firm_hist["q"][0, :, 0] == s.firm_globals.q0)
    assert r.acc["n_dev"][0, 0] == 0


@pytest.mark.parametrize("horizon", [1, 5])
def test_rc_accounting_identity(horizon):
    """Advantage of deviating over rule B equals correct-deviation gains minus wrong-deviation losses."""
    s = _mixed()
    r = run_batch([s], horizon=horizon)
    a = r.acc
    np.testing.assert_allclose(a["sum_realized_adv"], a["sum_gain_dev_pe"] - a["sum_loss_dev_npe"], atol=1e-6)


def test_horizon_one_matches_static_counterfactual():
    s = _mixed()
    a = run_batch([s], record_firm_history=True, horizon=1)
    h = a.firm_hist
    own = np.where(h["deviate"], h["profit_rule"], h["profit_default"])
    np.testing.assert_allclose(own[0, 1:], h["profit"][0, 1:])


def test_evolution_equivalence_with_horizon():
    scn = default_scenario(periods=200, seed=4)
    scn.firms[1].selection = "Adaptive"
    scn.evolution = EvolutionParams(enabled=True, every=30, imitation_prob=0.6, mutation_sd=0.05)
    ind = Industry(scn, horizon=3, continuation="rules").run()
    res = run_batch([scn], record_firm_history=True, horizon=3, continuation="rules")
    np.testing.assert_allclose(res.flex_path[0], ind.h["flex"])
    np.testing.assert_allclose(res.firm_hist["profit_rule"][0, 1:], ind.h["profit_rule"][1:])
