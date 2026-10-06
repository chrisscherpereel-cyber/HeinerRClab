"""Information timing: agents decide and learn only from information available at the time.

The schedule is documented in heiner_abm.agents.DECISION_SCHEDULE. Feedback about a decision is queued when the decision
is made and released only after the last period it covers has occurred. The researcher's counterfactual (forked
markets over the measurement horizon H) measures pi, r, w, G and D; it reaches agents only under the explicitly
labeled "oracle" treatment, and then only at maturity."""
import hashlib

import numpy as np
import pytest

from heiner_abm import arena, stepper
from heiner_abm.agents import DECISION_SCHEDULE, Industry, Market, belief_index
from heiner_abm.engine import run_batch
from heiner_abm.params import AdaptiveParams, FirmSpec, Scenario, default_scenario

K = 120          # cut-off period for the future-shock tests: everything after K is perturbed


def _adaptive_market(periods, feedback="observable", structural=False, seed=5, window=8):
    firms = [FirmSpec("Bertrand", 0.8, "Adaptive"), FirmSpec("Cournot", 0.6, "Adaptive", noise=3.0),
             FirmSpec("Bertrand", 1.2, "Always"), FirmSpec("Cournot", 0.4, "Adaptive", foresight=0.3)]
    s = Scenario(firms=firms, periods=periods, burn_in=10, seed=seed)
    s.market.delta = 20.0
    s.adaptive.feedback = feedback
    s.adaptive.window = window
    s.adaptive.memory = 0.8
    if structural:
        s.structural.enabled, s.structural.hazard, s.structural.belief_lag = True, 0.05, 3
    return s


def _digest(r):
    return hashlib.sha256(np.round(r.firm_hist["q"][0], 6).tobytes() + np.round(r.price[0], 6).tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ future shocks
def _perturbed_industry(scn, horizon, k=K):
    """An Industry whose cost shocks, perception noise and true demand after period k differ from the original."""
    ind = Industry(scn, horizon=horizon)
    rng = np.random.default_rng(99)
    ind.u[k + 1:] = rng.uniform(-1, 1, len(ind.u) - k - 1)
    ind.eps[k + 1:] = rng.standard_normal(ind.eps[k + 1:].shape)
    mk = Market(scn, ind.u)
    mk.p_max[k + 1:] += 7.0
    mk.slope[k + 1:] *= 1.15
    bi = belief_index(scn)       # the believed curve at s reflects the true curve of period s - 1 - lag at the latest
    mk.belief_p_max = np.where(bi < 0, scn.market.p_max, mk.p_max[np.maximum(bi, 0)])
    mk.belief_slope = np.where(bi < 0, scn.market.slope, mk.slope[np.maximum(bi, 0)])
    ind.market = mk
    return ind


@pytest.mark.parametrize("feedback,horizon", [("observable", 1), ("observable", 6), ("oracle", 1), ("oracle", 6)])
@pytest.mark.parametrize("structural", [False, True])
def test_future_shocks_change_no_earlier_decision_or_learned_state(feedback, horizon, structural):
    scn = _adaptive_market(260, feedback, structural)
    base = Industry(scn, horizon=horizon).run()
    pert = _perturbed_industry(scn, horizon).run()
    assert not np.array_equal(base.market.costs[K + 1:], pert.market.costs[K + 1:])
    for key in ("q", "rec", "deviate", "c_hat"):
        np.testing.assert_array_equal(base.h[key][:K + 1], pert.h[key][:K + 1], err_msg=key)
    np.testing.assert_array_equal(base.learned_path[:K + 1], pert.learned_path[:K + 1])
    assert not np.array_equal(base.h["q"][K + 1:], pert.h["q"][K + 1:])      # the perturbation does bite later


@pytest.mark.parametrize("feedback,horizon", [("observable", 1), ("observable", 6), ("oracle", 1), ("oracle", 12)])
def test_vectorized_engine_ignores_periods_after_the_run(feedback, horizon):
    """Truncating the run cannot change any earlier decision (the vectorized engine has no learned-state history, so
    the reference engine's future-shock test above is the stronger check; the two engines agree exactly)."""
    short, long_ = _adaptive_market(150, feedback), _adaptive_market(260, feedback)
    a = run_batch([short], record_firm_history=True, horizon=horizon)
    b = run_batch([long_], record_firm_history=True, horizon=horizon)
    np.testing.assert_array_equal(a.firm_hist["q"][0], b.firm_hist["q"][0, :150])
    np.testing.assert_array_equal(a.firm_hist["deviate"][0], b.firm_hist["deviate"][0, :150])


def _env(hazard=0.03):
    return arena.Env(delta=15.0, c_max=80.0, q_range=1500.0, noise=4.0, foresight=0.2, hazard=hazard, belief_lag=5,
                     seed=11)


class _PerturbedMarket(arena.Market):
    def __init__(self, envs, n, periods):
        super().__init__(envs, n, periods)
        rng = np.random.default_rng(99)
        self.cost[:, K + 1:] = np.clip(self.cost[:, K + 1:] + rng.normal(0, 8, self.cost[:, K + 1:].shape), 11, 79)
        self.eps[:, K + 1:] = rng.standard_normal(self.eps[:, K + 1:].shape)
        self.PM[:, K + 1:] += 7.0


LINEUP = ["heiner_m", "heiner_p", "options_p", "cobweb_p"]     # deterministic designs, incl. both learned RC agents


def test_arena_future_shocks_change_no_earlier_outcome(monkeypatch):
    params = {k: arena.default_params(k) for k in LINEUP}
    for k in ("heiner_m", "heiner_p"):
        params[k]["horizon"] = 9.0
    lineup = np.array([LINEUP])
    base = arena.simulate([_env()], lineup, params, 260, 20, keep_path=True)["path"]
    monkeypatch.setattr(arena, "Market", _PerturbedMarket)
    pert = arena.simulate([_env()], lineup, params, 260, 20, keep_path=True)["path"]
    np.testing.assert_array_equal(base[:, :K + 1 - 20], pert[:, :K + 1 - 20])
    assert not np.array_equal(base[:, K + 1 - 20:], pert[:, K + 1 - 20:])


def test_interactive_market_future_shocks_change_no_earlier_outcome(monkeypatch):
    def play(market_cls):
        monkeypatch.setattr(stepper, "Market", market_cls)
        im = stepper.InteractiveMarket(_env(), ["heiner_m", "options_p", "cobweb_p"],
                                       {"heiner_m": {"horizon": 9.0}}, ["heiner_p", "cobweb_p"], 200)
        rows = []
        while not im.done:
            info = im.info()
            rows.append((info["last_price"], info["cost_estimate"], im.step(150.0 + (im.t % 7) * 5)["price"]))
        return np.array(rows), im.frame()
    a, fa = play(arena.Market)
    b, fb = play(_PerturbedMarket)
    np.testing.assert_array_equal(a[:K], b[:K])          # row j is period j + 1
    shadow = [c for c in fa.columns if c.startswith("shadow:")]
    np.testing.assert_array_equal(fa[shadow].to_numpy()[:K], fb[shadow].to_numpy()[:K])
    assert not np.array_equal(a[K:], b[K:])


# ------------------------------------------------------------------------------------------------ maturity
@pytest.mark.parametrize("feedback,horizon,window", [("observable", 1, 1), ("observable", 6, 8), ("oracle", 6, 8),
                                                     ("oracle", 1, 8)])
def test_feedback_is_released_exactly_at_maturity(feedback, horizon, window):
    scn = _adaptive_market(200, feedback, window=window)
    ind = Industry(scn, horizon=horizon).run()
    span = window if feedback == "observable" else horizon
    assert ind.feedback_log
    for released, firm, decided, matures, b, gain, kind in ind.feedback_log:
        assert kind == feedback
        assert matures == decided + span - 1 == released
        assert ind.h["opportunity"][decided, firm]
    released = {(f, d) for _, f, d, *_ in ind.feedback_log}
    expected = {(f, d) for d in range(1, scn.periods - span + 1) for f in range(scn.n_firms) if ind.h["opportunity"][d, f]}
    assert released == expected                       # everything that matured, and nothing that did not
    assert all(item.matures > scn.periods - 1 for item in ind.pending)     # no partial terminal feedback


@pytest.mark.parametrize("feedback,horizon", [("observable", 1), ("oracle", 25)])
def test_h_period_feedback_is_unavailable_before_maturity(feedback, horizon):
    """With a 25-period span no learned table can move before period 25, however much has been decided."""
    scn = _adaptive_market(120, feedback, window=25)
    ind = Industry(scn, horizon=horizon if feedback == "oracle" else 1).run()
    assert np.all(ind.learned_path[:25] == 0.0)          # decisions of periods 1-24 have not matured yet
    assert np.any(ind.learned_path[25] != 0.0)           # the decision of period 1 matures at period 25
    for released, firm, decided, matures, *_ in ind.feedback_log:
        assert released >= decided + 24


def test_schedule_is_documented():
    for step in ("Decide", "Clear", "Researcher evaluation", "pending queue", "Release", "no partial"):
        assert step in DECISION_SCHEDULE


# ------------------------------------------------------------------------------------------------ separation
@pytest.mark.parametrize("structural", [False, True])
def test_measurement_horizon_does_not_change_agent_behavior(structural):
    """H is the researcher's measurement horizon; with observable feedback it changes the measured gains only."""
    s = _adaptive_market(200, structural=structural)
    runs = [run_batch([s], record_firm_history=True, horizon=H) for H in (1, 5, 15)]
    for r in runs[1:]:
        np.testing.assert_array_equal(r.firm_hist["q"][0], runs[0].firm_hist["q"][0])
        np.testing.assert_array_equal(r.price[0], runs[0].price[0])


def test_oracle_is_opt_in_and_cannot_affect_ordinary_agents():
    assert AdaptiveParams().feedback == "observable"
    plain = default_scenario(periods=200, seed=4)                  # no Adaptive firms
    oracle = plain.copy()
    oracle.adaptive.feedback = "oracle"
    a = run_batch([plain], record_firm_history=True, horizon=6)
    b = run_batch([oracle], record_firm_history=True, horizon=6)
    np.testing.assert_array_equal(a.firm_hist["q"], b.firm_hist["q"])


def test_removed_immediate_lookahead_is_rejected():
    s = _adaptive_market(50, feedback="lookahead")
    assert any("removed" in e for e in s.validate())
    s = _adaptive_market(50, feedback="nonsense")
    assert any("Unknown Adaptive feedback" in e for e in s.validate())


def test_adaptive_agents_still_learn_to_hold_back():
    """The observable rule is a working learner, not a constant: it deviates in some bins and not in others."""
    s = _adaptive_market(600)
    ind = Industry(s, horizon=1).run()
    learned = np.array([f.learned_gain for f, spec in zip(ind.firms, s.firms) if spec.selection == "Adaptive"])
    assert np.abs(learned).sum() > 0
    dev = ind.h["deviate"][100:][:, [0, 1, 3]]
    opp = ind.h["opportunity"][100:][:, [0, 1, 3]]
    assert 0.0 < dev[opp].mean() < 1.0


# ------------------------------------------------------------------------------------------------ regressions
def test_one_period_oracle_feedback_is_unchanged_since_before_october_2026():
    """With H = 1 the old immediate feedback already had valid timing (released at the end of the decision period).
    Digest computed with the code of commit 469619d (feedback then called 'lookahead')."""
    s = _adaptive_market(300, "oracle")
    s.adaptive.window = 20          # unused by oracle feedback
    r = run_batch([s], record_firm_history=True, horizon=1)
    assert (_digest(r), int(r.firm_hist["deviate"][0].sum())) == ("f5e29f7d37722f76", 967)


def test_observable_feedback_is_unchanged_by_the_queue():
    """The pending queue reproduces the earlier at-maturity computation exactly (digest from commit 3dc9646)."""
    s = _adaptive_market(300)
    r = run_batch([s], record_firm_history=True, horizon=6)
    assert (_digest(r), int(r.firm_hist["deviate"][0].sum())) == ("8d7c09f66cb6952c", 851)
