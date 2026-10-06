"""Selection gates of the Adaptive rule (heiner_abm.gates, a proposed extension) and their comparison
(heiner_abm.gate_study): information timing, initialization, limiting cases, evidence accumulation, the separation of
the ORACLE benchmark from ordinary learning, and agreement of the two engines."""
import numpy as np
import pytest

from heiner_abm import gates
from heiner_abm.agents import Industry, Market, belief_index
from heiner_abm.engine import run_batch
from heiner_abm.gate_study import ORACLE_SEED_BASE, compare_gates, gate_scenario, oracle_table
from heiner_abm.information import UnsupportedInformation, compatibility, AGENT_BY_KEY, export_record, market_agents
from heiner_abm.params import FirmSpec, Scenario

K = 110          # cut-off period for the future-shock tests
ADAPTIVE = [0, 1, 3]


def _scn(gate="lcb", periods=220, delay=0, reset=False, structural=True, feedback="estimated", cost=0.0, W=4,
         memory=0.9, nmin=3.0, rate=0.5, conf=0.9, seed=7, announced=False):
    firms = [FirmSpec("Bertrand", 0.8, "Adaptive"), FirmSpec("Cournot", 0.6, "Adaptive", noise=3.0),
             FirmSpec("Bertrand", 1.2, "Always"), FirmSpec("Cournot", 0.4, "Adaptive", foresight=0.3)]
    s = Scenario(firms=firms, periods=periods, burn_in=10, seed=seed)
    s.market.delta = 20.0
    s.info.feedback, s.info.obs_delay, s.info.regime_announced = feedback, delay, announced
    ad = s.adaptive
    ad.window, ad.memory, ad.gate, ad.adjust_cost = W, memory, gate, cost
    ad.min_evidence, ad.explore_rate, ad.confidence = nmin, rate, conf
    ad.on_change = "reset" if reset else "forget"
    if gate == "oracle_table":
        ad.oracle_table = tuple(tuple(r) for r in np.random.default_rng(3).normal(0, 50, (4, 5)))
    if structural:
        s.structural.enabled, s.structural.hazard, s.structural.belief_lag = True, 0.05, 3
    return s


def _released_before(ind, t, firm, b):
    """Items about firm's decisions in bin b released at the end of a period before t."""
    return [x for x in ind.feedback_log if x[1] == firm and x[4] == b and x[0] < t]


# ------------------------------------------------------------------------------------------------ engine agreement
@pytest.mark.parametrize("gate", gates.GATES)
@pytest.mark.parametrize("delay,reset,H", [(0, False, 1), (2, True, 3)])
def test_reference_and_vectorized_engines_agree_for_every_gate(gate, delay, reset, H):
    s = _scn(gate, delay=delay, reset=reset, cost=0.0 if gate == "gain" else 30.0)
    ind = Industry(s.copy(), horizon=H).run()
    r = run_batch([s.copy()], record_firm_history=True, horizon=H)
    for k in ("q", "deviate", "adv_pred", "adv_se", "adv_bound", "adv_neff", "fb_value", "fb_release", "explored"):
        np.testing.assert_array_equal(ind.h[k], r.firm_hist[k][0], err_msg=k)
    np.testing.assert_array_equal(ind.h["reset"], r.firm_hist["reset"][0])


# ------------------------------------------------------------------------------------------------ information timing
def _perturbed(scn, horizon, k=K):
    ind = Industry(scn, horizon=horizon)
    rng = np.random.default_rng(99)
    ind.u[k + 1:] = rng.uniform(-1, 1, len(ind.u) - k - 1)
    ind.eps[k + 1:] = rng.standard_normal(ind.eps[k + 1:].shape)
    mk = Market(scn, ind.u)
    mk.p_max[k + 1:] += 7.0
    mk.slope[k + 1:] *= 1.15
    bi = belief_index(scn)
    mk.belief_p_max = np.where(bi < 0, scn.market.p_max, mk.p_max[np.maximum(bi, 0)])
    mk.belief_slope = np.where(bi < 0, scn.market.slope, mk.slope[np.maximum(bi, 0)])
    ind.market = mk
    if ind.xdraw is not None:                    # future exploration draws differ too
        ind.xdraw = ind.xdraw.copy()
        ind.xdraw[:, k + 1:] = rng.random(ind.xdraw[:, k + 1:].shape)
    return ind


@pytest.mark.parametrize("gate", ["lcb", "explore", "gain"])
@pytest.mark.parametrize("delay,reset", [(0, False), (2, True)])
def test_future_shocks_change_no_earlier_decision_prediction_or_evidence(gate, delay, reset):
    scn = _scn(gate, delay=delay, reset=reset, cost=20.0)
    base = Industry(scn, horizon=3).run()
    pert = _perturbed(scn, 3).run()
    for key in ("q", "deviate", "adv_pred", "adv_se", "adv_neff", "explored"):
        np.testing.assert_array_equal(base.h[key][:K + 1], pert.h[key][:K + 1], err_msg=key)
    # feedback released by period K is identical; feedback released later may differ
    early = lambda ind: [x for x in ind.feedback_log if x[0] <= K]
    assert early(base) == early(pert)
    assert not np.array_equal(base.h["q"][K + 1:], pert.h["q"][K + 1:])


@pytest.mark.parametrize("gate,delay", [("lcb", 0), ("lcb", 3), ("explore", 0), ("explore", 2)])
def test_feedback_is_released_at_maturity_plus_delay_and_used_only_afterwards(gate, delay):
    scn = _scn(gate, delay=delay, memory=1.0, structural=False, W=5)
    ind = Industry(scn, horizon=1).run()
    released = np.argwhere(ind.h["fb_release"] >= 0)
    assert len(released) > 20
    for d, i in released:
        assert ind.h["fb_release"][d, i] == d + 5 - 1 + delay
    if gate == "lcb":            # with memory 1 the effective evidence is the count of items released before t
        for t in range(1, scn.periods):
            for i in ADAPTIVE:
                b = ind.firms[i]._bin(abs(ind.h["rec"][t, i] - ind.h["q"][t - 1, i]))
                assert ind.h["adv_neff"][t, i] == pytest.approx(len(_released_before(ind, t, i, b)), abs=1e-9)


def test_trial_feedback_is_the_firms_own_payoff_minus_an_action_independent_baseline():
    W, delay = 4, 2
    scn = _scn("explore", delay=delay, W=W, structural=False)
    ind = Industry(scn, horizon=1).run()
    trials = np.argwhere(ind.h["fb_release"] >= 0)
    assert len(trials) > 10
    for d, i in trials:
        assert ind.h["explored"][d, i]
        s = max(d - 1 - delay, 0)
        c = ind.market.costs
        expected = 0.0 - W * ind.h["profit"][s, i]
        for t in range(d, d + W):          # own payoff vs the pre-decision payoff at the realized (exogenous) cost
            expected = expected + (ind.h["profit"][t, i] + ind.h["q"][s, i] * (c[t] - c[s]))
        assert ind.h["fb_value"][d, i] == expected


def test_explore_gate_never_uses_counterfactual_feedback():
    """Its decisions are the same whatever counterfactual feedback the specification offers."""
    runs = [run_batch([_scn("explore", feedback=fb)], record_firm_history=True, horizon=4)
            for fb in ("chosen", "estimated", "oracle")]
    for r in runs[1:]:
        np.testing.assert_array_equal(r.firm_hist["q"], runs[0].firm_hist["q"])


def test_gates_that_need_counterfactuals_are_not_run_under_chosen_action_feedback():
    for gate in ("gain", "lcb"):
        s = _scn(gate, feedback="chosen")
        assert s.info_errors()
        with pytest.raises(UnsupportedInformation):
            run_batch([s])
    assert not _scn("explore", feedback="chosen").info_errors()


# ------------------------------------------------------------------------------------------------ initialization
def test_initialization_and_sparse_data():
    for gate in ("gain", "lcb", "explore"):
        ind = Industry(_scn(gate, structural=False), horizon=1).run()
        h = ind.h
        first = 1
        assert np.all(np.isnan(h["adv_pred"][first, ADAPTIVE]))       # no evidence: no prediction, never a zero
        assert np.all(h["adv_neff"][first, ADAPTIVE] == 0.0)
        opp = h["opportunity"][first, ADAPTIVE]
        if gate == "gain":       # the existing gate starts optimistic: it adopts without evidence
            assert np.all(h["deviate"][first, ADAPTIVE][opp])
        if gate == "lcb":        # the confidence-sensitive gate keeps x_B until it has the minimum evidence
            ad = h["deviate"][:, ADAPTIVE] & h["opportunity"][:, ADAPTIVE]
            assert np.all(h["adv_neff"][:, ADAPTIVE][ad] >= 3.0)
        if gate == "explore":    # without evidence every decision is uncertain: trials or keeps, never exploitation
            sparse = h["opportunity"][:, ADAPTIVE] & (h["adv_neff"][:, ADAPTIVE] < 3.0)
            dev = h["deviate"][:, ADAPTIVE]
            assert np.all(h["explored"][:, ADAPTIVE][sparse & dev])
            assert 0.2 < h["explored"][:, ADAPTIVE][sparse].mean() < 0.8      # explore_rate 0.5


def test_missing_outcomes_are_never_counted_as_zero_gains():
    ind = Industry(_scn("explore", memory=1.0, structural=False), horizon=1).run()
    for i in ADAPTIVE:
        f = ind.firms[i]
        counts = np.zeros((2, 5))
        for released, firm, decided, *_ in ind.feedback_log:
            if firm == i:
                counts[int(ind.h["deviate"][decided, i]), f._bin(abs(ind.h["rec"][decided, i] -
                                                                     ind.h["q"][decided - 1, i]))] += 1
        np.testing.assert_array_equal(f.trials[:, 0, :], counts)       # S0 counts released trials only
    # decisions whose feedback would mature after the run are never released, so never enter the statistics
    assert all(item.release > 219 for item in ind.pending)
    mean, se, neff = gates.summary(np.zeros((4, 3)))
    assert np.all(np.isnan(mean)) and np.all(neff == 0)


def test_one_item_gives_a_mean_but_no_uncertainty_estimate():
    S = gates.update(np.zeros(4), 12.0, 0.9)[:, None]
    mean, se, neff = gates.summary(S)
    assert mean[0] == 12.0 and neff[0] == 1.0 and np.isnan(se[0])
    t = gates.gate_table("lcb", np.zeros(1), S, np.zeros((2, 4, 1)), 0.9, 0.0, 0.9, 2.0)
    assert t["mode"][0] == gates.KEEP


# ------------------------------------------------------------------------------------------------ limiting cases
def _price(s, H=1):
    return run_batch([s], record_firm_history=True, horizon=H).price[0]


def test_unreachable_evidence_requirement_or_zero_exploration_equals_rule_b():
    never = _scn("lcb")
    for i in ADAPTIVE:
        never.firms[i].selection = "Never"
    np.testing.assert_array_equal(_price(_scn("lcb", nmin=1e9)), _price(never))
    np.testing.assert_array_equal(_price(_scn("explore", rate=0.0)), _price(never))


def test_oracle_table_extremes_equal_always_and_never():
    hi, lo = _scn("oracle_table"), _scn("oracle_table")
    hi.adaptive.oracle_table = tuple((1e12,) * 5 for _ in range(4))
    lo.adaptive.oracle_table = tuple((-np.inf,) * 5 for _ in range(4))
    always, never = _scn("gain"), _scn("gain")
    for i in ADAPTIVE:
        always.firms[i].selection, never.firms[i].selection = "Always", "Never"
    np.testing.assert_array_equal(_price(hi), _price(always))
    np.testing.assert_array_equal(_price(lo), _price(never))


def test_lower_bound_at_confidence_one_half_is_the_point_estimate():
    """With confidence 0.5 the lower bound is the mean, so the lcb gate decides by mean > c once it has evidence."""
    ind = Industry(_scn("lcb", conf=0.5, nmin=2.0, cost=15.0), horizon=1).run()
    h = ind.h
    m = h["opportunity"] & (h["adv_neff"] >= 2.0) & (h["adv_pred"] != 15.0)
    m[:, 2] = False
    assert m.sum() > 50
    np.testing.assert_array_equal(h["deviate"][m], h["adv_pred"][m] > 15.0)
    np.testing.assert_allclose(h["adv_bound"][m], h["adv_pred"][m])


def test_a_prohibitive_adjustment_cost_stops_adaptation_once_there_is_evidence():
    for gate in ("gain", "lcb"):
        h = Industry(_scn(gate, cost=1e9), horizon=1).run().h
        ad = (h["deviate"] & h["opportunity"])[:, ADAPTIVE]
        assert np.all(h["adv_neff"][:, ADAPTIVE][ad] == 0.0)    # gain: only its optimistic start; lcb: never
        if gate == "lcb":
            assert not ad.any()


def test_existing_gate_is_unchanged_by_the_extension():
    """The default gate with no adjustment cost reproduces the rule as it was before the gates were added (digests
    from tests/test_information.py and the market equivalence tests also cover this)."""
    s = _scn("gain", structural=False)
    s2 = s.copy()
    s2.adaptive.confidence, s2.adaptive.min_evidence, s2.adaptive.explore_rate = 0.99, 50.0, 0.9   # unused by "gain"
    np.testing.assert_array_equal(_price(s), _price(s2))


def test_existing_gate_still_learns_without_forgetting():
    """With memory 1 the existing table would never move; the gate then uses the equivalent sums."""
    h = Industry(_scn("gain", memory=1.0, cost=40.0, structural=False), horizon=1).run().h
    with_evidence = h["opportunity"][:, ADAPTIVE] & (h["adv_neff"][:, ADAPTIVE] > 0)
    assert with_evidence.sum() > 50
    np.testing.assert_array_equal(h["deviate"][:, ADAPTIVE][with_evidence],
                                  h["adv_pred"][:, ADAPTIVE][with_evidence] >= 40.0)


# ------------------------------------------------------------------------------------------------ evidence accumulation
def test_uncertainty_shrinks_and_the_bound_rises_with_evidence():
    rng = np.random.default_rng(0)
    S = np.zeros(4)
    se, bound = [], []
    for n in range(1, 201):
        S = gates.update(S, 10.0 + 30.0 * rng.standard_normal(), 1.0)
        mean, s_e, neff = gates.summary(S[:, None])
        assert neff[0] == pytest.approx(n)
        if n >= 2:
            se.append(s_e[0])
            bound.append(gates.gate_table("lcb", np.zeros(1), S[:, None], np.zeros((2, 4, 1)), 1.0, 0.0, 0.9,
                                          2.0)["bound"][0])
    assert se[-1] < se[10] / 3                               # about 30 / sqrt(n)
    assert np.mean(bound[-20:]) > np.mean(bound[5:25])
    assert abs(se[-1] - 30.0 / np.sqrt(200)) < 0.6


def test_forgetting_caps_the_effective_evidence():
    S = np.zeros(4)
    for _ in range(2000):
        S = gates.update(S, 1.0, 0.95)
    assert gates.summary(S[:, None])[2][0] == pytest.approx((1 + 0.95) / (1 - 0.95), rel=1e-6)


def test_lower_bound_has_nominal_coverage_only_under_its_assumptions():
    """Independent normal draws with a common mean: the one-sided bound covers the mean at about its level. (No such
    claim is made for the market, where draws are dependent and sampling is adaptive.)"""
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(3000):
        S = np.zeros(4)
        for g in 5.0 + 4.0 * rng.standard_normal(8):
            S = gates.update(S, g, 1.0)
        b = gates.gate_table("lcb", np.zeros(1), S[:, None], np.zeros((2, 4, 1)), 1.0, 0.0, 0.9, 2.0)["bound"][0]
        hits += b <= 5.0
    assert abs(hits / 3000 - 0.9) < 0.02


def test_market_bounds_tighten_as_evidence_accumulates():
    h = Industry(_scn("lcb", memory=1.0, periods=400, structural=False), horizon=1).run().h
    width = (h["adv_pred"] - h["adv_bound"])[:, ADAPTIVE]
    neff = h["adv_neff"][:, ADAPTIVE]
    ok = np.isfinite(width)
    lo, hi = ok & (neff < 10), ok & (neff > 40)
    assert lo.any() and hi.any()
    assert np.median(width[hi]) < np.median(width[lo])


# ------------------------------------------------------------------------------------------------ change response
def test_reset_discards_evidence_when_the_firms_see_a_change():
    ind = Industry(_scn("lcb", reset=True, announced=True, memory=1.0, periods=300), horizon=1).run()
    resets = np.flatnonzero(ind.h["reset"])
    assert len(resets) >= 3
    np.testing.assert_array_equal(resets, np.flatnonzero(ind.market.shift))     # announced: at the shift itself
    for t in resets:
        assert np.all(ind.h["adv_neff"][t, ADAPTIVE] == 0.0)
        # no feedback about a decision made before the reset is released after it
        assert not any(dec < t <= rel for rel, f, dec, *_ in ind.feedback_log)


def test_without_a_demand_model_only_announcements_signal_a_change():
    s = _scn("explore", reset=True)
    s.info.demand_knowledge = "none"
    for f in s.firms:
        f.rule = "Bertrand"
    mk = Market(s, Industry(s).u)
    assert not gates.change_signal(s, mk.belief_p_max, mk.belief_slope, mk.shift).any()
    s.info.regime_announced = True
    np.testing.assert_array_equal(gates.change_signal(s, mk.belief_p_max, mk.belief_slope, mk.shift), mk.shift)


# ------------------------------------------------------------------------------------------------ oracle separation
def test_oracle_table_is_accepted_only_by_the_oracle_gate():
    s = _scn("lcb")
    s.adaptive.oracle_table = tuple((0.0,) * 5 for _ in range(4))
    assert any("ordinary gates must not" in e for e in s.validate())
    s = _scn("oracle_table")
    s.adaptive.oracle_table = ()
    assert any("needs a table" in e for e in s.validate())


def test_oracle_gate_is_labeled_as_an_oracle_variant():
    s = _scn("oracle_table")
    agent = AGENT_BY_KEY[("market", "Adaptive:oracle_table")]
    assert agent in market_agents(s)
    assert compatibility(agent, s.info, "market").status == "oracle variant"
    assert export_record("market", s.info, market_agents(s))["oracle_information"]


def test_oracle_benchmark_uses_independent_seeds_and_does_not_reach_ordinary_gates():
    base = _scn("gain", periods=160, structural=False)
    small = dict(reps=2, oracle_runs=2, oracle_passes=1, horizon=4, adjust_cost=10.0)
    tab = oracle_table(gate_scenario(base, "gain", 1), runs=2, horizon=4, passes=1)
    assert min(tab["seeds"]) >= ORACLE_SEED_BASE
    a = compare_gates(base, ("lcb", "explore"), **small)
    b = compare_gates(base, ("lcb", "explore", "oracle_table"), **{**small, "oracle_runs": 3})
    for g in ("lcb", "explore"):
        x = a.markets[a.markets["gate"] == g]["net_payoff"].to_numpy()
        y = b.markets[b.markets["gate"] == g]["net_payoff"].to_numpy()
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError):
        compare_gates(base, ("lcb",), reps=2, seed0=ORACLE_SEED_BASE - 1, oracle_runs=1)


def test_gate_study_reports_every_metric():
    base = _scn("gain", periods=200)
    st = compare_gates(base, reps=3, oracle_runs=2, horizon=4, adjust_cost=10.0, min_evidence=3.0,
                       shift_window=20)
    assert list(st.summary["gate"]) == list(gates.GATES)
    for col in ("adoption_rate", "false_adapt_realized", "false_adapt_oracle", "missed_realized", "missed_oracle",
                "net_payoff", "bias", "calibration_slope", "coverage_lower", "learning_delay", "post_shift_change",
                "net_payoff_vs_oracle"):
        assert col in st.summary, col
    o = st.summary.set_index("gate").loc["oracle_table"]
    assert o["false_adapt_oracle"] == 0.0 and o["missed_oracle"] == 0.0 and o["net_payoff_vs_oracle"] == 0.0
    assert not st.calibration.empty and not st.after_shift.empty
    assert st.settings["info"]["feedback"] == "estimated"


def test_unsupported_gates_are_reported_not_run():
    base = _scn("gain", periods=150, structural=False, feedback="chosen")
    st = compare_gates(base, ("gain", "explore"), reps=2, oracle_runs=1, oracle_passes=1, horizon=3)
    assert "gain" in st.not_run and list(st.summary["gate"]) == ["explore"]
