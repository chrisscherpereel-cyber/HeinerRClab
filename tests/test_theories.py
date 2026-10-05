"""Tests for the theory agents and the general (multi-theory) market."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heiner_abm.engine import run_batch  # noqa: E402
from heiner_abm.general import STANDARD_ENVS, scorecard, tournament  # noqa: E402
from heiner_abm.params import FirmSpec, StructuralParams, default_scenario  # noqa: E402
from heiner_abm.theories import (GENERAL_THEORIES, SEL_NAMES, AgentSpec, SelectionParams, run_theories,  # noqa: E402
                                 theory_table)


def _paper_specs(rule, flexes, sels, thr=8.0):
    return [AgentSpec("paper", dict(rule=0.0 if rule == "Cournot" else 1.0, flex=f, selection=float(SEL_NAMES.index(s)),
                                    threshold=thr, desired_margin=5.0)) for f, s in zip(flexes, sels)]


@pytest.mark.parametrize("rule", ["Cournot", "Bertrand"])
@pytest.mark.parametrize("struct", [False, True])
def test_paper_agents_reproduce_engine_exactly(rule, struct):
    """The general multi-theory market embeds the paper's model: identical prices and profits."""
    flexes, sels = (0.25, 0.5, 0.75, 1.0), ("Always", "Never", "Small", "Large")
    s = default_scenario(periods=250, seed=9)
    s.firms = [FirmSpec(rule, f, sel, threshold=8) for f, sel in zip(flexes, sels)]
    s.firm_globals.flex_cost_slope, s.firm_globals.fixed_cost = 120.0, 40.0
    s.firms[0].foresight, s.firms[0].noise = 0.0, 0.0
    if struct:
        s.structural = StructuralParams(enabled=True, hazard=0.04, belief_lag=6)
    e = run_batch([s], record_firm_history=True)
    r = run_theories([s], _paper_specs(rule, flexes, sels), record=True)
    np.testing.assert_allclose(r.price[0], e.price[0])
    np.testing.assert_allclose(r.hist["profit"][0, 1:], e.firm_hist["profit"][0, 1:])


def _scn(seed=1, periods=300, delta=10.0, hazard=0.0, q0=100.0):
    s = default_scenario(periods=periods, seed=seed)
    s.market.delta = delta
    s.firm_globals.q0 = q0
    if hazard:
        s.structural = StructuralParams(enabled=True, hazard=hazard, belief_lag=10)
    return s


def test_all_theories_run_and_are_deterministic():
    specs = [AgentSpec(k) for k in GENERAL_THEORIES]
    a = run_theories([_scn(s) for s in range(3)], specs)
    b = run_theories([_scn(s) for s in range(3)], specs)
    np.testing.assert_array_equal(a.price, b.price)
    t = theory_table(a)
    assert set(t["theory"]) == set(GENERAL_THEORIES)
    assert np.isfinite(t["avg_profit"]).all()


def test_ecology_firm_never_changes_output():
    r = run_theories([_scn()], [AgentSpec("ecology"), AgentSpec("neoclassical")], record=True)
    q = r.hist["q"][0, :, 0]
    assert np.all(q == q[0])


def test_imitator_copies_most_profitable_firm():
    r = run_theories([_scn()], [AgentSpec("imitation"), AgentSpec("neoclassical"), AgentSpec("ecology")], record=True)
    q, pr = r.hist["q"][0], r.hist["profit"][0]
    for t in range(1, 200):
        best = int(np.argmax(pr[t - 1]))
        assert q[t, 0] == max(15.0, np.rint(q[t - 1, best]))


def test_neoclassical_plays_nash_quantity():
    s = _scn()
    r = run_theories([s], [AgentSpec("neoclassical")] * 3, record=True)
    c = r.cost[0]
    expect = np.maximum(15.0, np.rint((s.market.p_max - c[:-1]) / (s.market.slope * 4)))
    np.testing.assert_allclose(r.hist["q"][0, 1:, 0], expect)


def test_heiner_agent_speed_respects_bound():
    r = run_theories([_scn(delta=25)], [AgentSpec("heiner")] * 4, record=True)
    flex = r.hist["flex"][0, 1:, 0]
    assert flex.max() <= 1.0 / (1.0 * (1 + 1.5)) + 1e-9        # beta0 <= 1/(1 - f') with K >= 0, n = 4


def test_satisficer_holds_when_satisfied_and_bayes_gain_in_unit_interval():
    r = run_theories([_scn(delta=1.0)], [AgentSpec("satisficing"), AgentSpec("bayesian"), AgentSpec("ecology")],
                     record=True)
    g = r.hist["flex"][0, 1:, 1]
    assert (g > 0).all() and (g <= 1).all()
    sat = r.hist["flex"][0, 1:, 0] == 0          # flex = 1 when dissatisfied
    dq = np.abs(np.diff(r.hist["q"][0, :, 0]))
    assert np.all(dq[sat] == 0)


def test_selection_exits_and_entry_keep_candidates():
    specs = [AgentSpec(k) for k in ("neoclassical", "biases", "ecology", "imitation")]
    sel = SelectionParams(enabled=True, capital0=2000.0, mode="survivors", mutation=0.1)
    r = run_theories([_scn(seed=s, delta=30, hazard=0.05, periods=600) for s in range(4)], specs, selection=sel)
    assert r.exits.sum() > 0
    assert set(np.unique(r.types)) <= set(range(len(r.theory_keys)))


def test_tournament_and_scorecard_run_small():
    base = default_scenario(periods=200)
    specs = {k: AgentSpec(k) for k in GENERAL_THEORIES}
    T, I = tournament(base, list(specs.values()), STANDARD_ENVS[:2], 2)
    assert len(T) == 2 * 2 * len(GENERAL_THEORIES)
    sc, det = scorecard(base, specs, 4, selection_periods=300)
    assert set(sc["verdict"]) <= {"Supported", "Not supported"}
    assert len(sc) >= 8
