"""Agents decide and learn only from information available at decision time.

The researcher's counterfactual (forked markets over the measurement horizon H, priced on the true demand curve)
measures pi, r, w, G and D. It must never reach the agents unless the researcher-only "lookahead" feedback is chosen
explicitly."""
import numpy as np
import pytest

from heiner_abm.agents import Industry
from heiner_abm.engine import run_batch
from heiner_abm.params import FirmSpec, Scenario


def _adaptive_market(periods, feedback="observable", structural=False, seed=5):
    firms = [FirmSpec("Bertrand", 0.8, "Adaptive"), FirmSpec("Cournot", 0.6, "Adaptive", noise=3.0),
             FirmSpec("Bertrand", 1.2, "Always"), FirmSpec("Cournot", 0.4, "Adaptive", foresight=0.3)]
    s = Scenario(firms=firms, periods=periods, burn_in=10, seed=seed)
    s.market.delta = 20.0
    s.adaptive.feedback = feedback
    s.adaptive.window = 8
    s.adaptive.memory = 0.8
    if structural:
        s.structural.enabled, s.structural.hazard, s.structural.belief_lag = True, 0.05, 3
    return s


@pytest.mark.parametrize("horizon", [1, 6])
def test_observable_decisions_do_not_depend_on_later_periods(horizon):
    """Truncating the run cannot change any earlier decision: nothing after period t is used at t."""
    short, long_ = _adaptive_market(150), _adaptive_market(260)
    a = run_batch([short], record_firm_history=True, horizon=horizon)
    b = run_batch([long_], record_firm_history=True, horizon=horizon)
    np.testing.assert_array_equal(a.firm_hist["q"][0], b.firm_hist["q"][0, :150])
    np.testing.assert_array_equal(a.firm_hist["deviate"][0], b.firm_hist["deviate"][0, :150])


def test_lookahead_feedback_lets_researcher_information_reach_agents():
    """Contrast for the test below: with the researcher-only feedback, the measurement horizon H (and with it future
    periods and rivals' simulated reactions) changes what the agents do. That is why it is not the default."""
    s = _adaptive_market(200, "lookahead")
    a = run_batch([s], record_firm_history=True, horizon=1)
    b = run_batch([s], record_firm_history=True, horizon=15)
    assert not np.array_equal(a.firm_hist["q"][0], b.firm_hist["q"][0])


@pytest.mark.parametrize("structural", [False, True])
def test_measurement_horizon_does_not_change_agent_behavior(structural):
    """H is the researcher's measurement horizon; with observable feedback it changes the measured gains only."""
    s = _adaptive_market(200, structural=structural)
    runs = [run_batch([s], record_firm_history=True, horizon=H) for H in (1, 5, 15)]
    for r in runs[1:]:
        np.testing.assert_array_equal(r.firm_hist["q"][0], runs[0].firm_hist["q"][0])
        np.testing.assert_array_equal(r.price[0], runs[0].price[0])


def test_adaptive_agents_still_learn_to_hold_back():
    """The observable rule is a working learner, not a constant: it deviates in some bins and not in others."""
    s = _adaptive_market(600)
    ind = Industry(s, horizon=1).run()
    learned = np.array([f.learned_gain for f, spec in zip(ind.firms, s.firms) if spec.selection == "Adaptive"])
    assert np.abs(learned).sum() > 0
    dev = ind.h["deviate"][100:][:, [0, 1, 3]]
    opp = ind.h["opportunity"][100:][:, [0, 1, 3]]
    rate = dev[opp].mean()
    assert 0.0 < rate < 1.0


def test_unknown_feedback_is_rejected():
    s = _adaptive_market(50, feedback="oracle")
    assert any("feedback" in e for e in s.validate())


@pytest.mark.parametrize("horizon,digest,n_dev", [(1, "f5e29f7d37722f76", 967), (6, "bded79d4403071d9", 943)])
def test_lookahead_feedback_reproduces_the_pre_october_2026_model(horizon, digest, n_dev):
    """The researcher-only option is kept to reproduce earlier results: these digests were computed with the code of
    commit 469619d, before observable feedback was introduced."""
    import hashlib
    s = _adaptive_market(300, "lookahead")
    s.adaptive.window = 20          # unused by look-ahead feedback
    r = run_batch([s], record_firm_history=True, horizon=horizon)
    h = hashlib.sha256(np.round(r.firm_hist["q"][0], 6).tobytes() + np.round(r.price[0], 6).tobytes()).hexdigest()
    assert (h[:16], int(r.firm_hist["deviate"][0].sum())) == (digest, n_dev)
