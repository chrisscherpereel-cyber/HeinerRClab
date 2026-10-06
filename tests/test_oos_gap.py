"""Out-of-sample RC tests: estimates use nothing from the evaluation window.

A decision's H-period gain covers periods t .. t + H - 1, so for the measures that look ahead ("persist", "full") the
estimation window ends H - 1 periods before the evaluation window starts."""
import numpy as np
import pytest

from heiner_abm import engine
from heiner_abm.agents import make_streams as _make_streams
from heiner_abm.engine import MEASURES, run_batch
from heiner_abm.params import FirmSpec, Scenario

H, T, BURN, SPLIT = 8, 300, 20, 0.5


def _scenario():
    firms = [FirmSpec("Bertrand", 0.8, "Always"), FirmSpec("Cournot", 0.6, "Large", threshold=6, noise=2.0),
             FirmSpec("Bertrand", 1.2, "Small", threshold=12), FirmSpec("Cournot", 0.4, "Adaptive")]
    s = Scenario(firms=firms, periods=T, burn_in=BURN, seed=9)
    s.market.delta = 15.0
    return s


def _perturbed_streams(cut):
    def make(seed, n, periods):
        u, eps, rng = _make_streams(seed, n, periods)
        u = u.copy()
        u[cut:] = np.random.default_rng(1).uniform(-1, 1, periods - cut)       # different costs from the split on
        return u, eps, rng
    return make


def test_estimation_window_ends_h_minus_1_periods_before_the_split():
    r = run_batch([_scenario()], horizon=H, oos_split=SPLIT)
    split = r.meta["split_t"]
    assert r.meta["est_end"] == {"static": split, "persist": split - (H - 1), "full": split - (H - 1)}


@pytest.mark.parametrize("continuation", ["default", "rules"])
def test_estimates_do_not_depend_on_the_evaluation_window(monkeypatch, continuation):
    base = run_batch([_scenario()], horizon=H, oos_split=SPLIT, continuation=continuation)
    monkeypatch.setattr(engine, "make_streams", _perturbed_streams(base.meta["split_t"]))
    pert = run_batch([_scenario()], horizon=H, oos_split=SPLIT, continuation=continuation)
    assert not np.array_equal(base.cost[0, base.meta["split_t"]:], pert.cost[0, pert.meta["split_t"]:])
    for m in MEASURES:
        for k, v in base.accs[(m, "est")].items():
            np.testing.assert_array_equal(v, pert.accs[(m, "est")][k], err_msg=f"{m} {k}")
    # the evaluation window does change, so the test has teeth
    assert not np.array_equal(base.accs[("full", "eval")]["sum_profit"], pert.accs[("full", "eval")]["sum_profit"])


def test_one_period_measure_keeps_the_whole_first_window():
    r = run_batch([_scenario()], horizon=H, oos_split=SPLIT)
    n_static = r.accs[("static", "est")]["n_rec"][0, 0]
    n_full = r.accs[("full", "est")]["n_rec"][0, 0]
    assert n_static - n_full == H - 1
