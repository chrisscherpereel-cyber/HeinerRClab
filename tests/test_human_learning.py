"""Trial-by-trial models of human adjustment: that they see only the screen, and that blocks stay separate.

Two properties matter more than the rest and are tested first. A model that quietly reads the researcher's
reference signal would fit beautifully and mean nothing, because participants were never shown it. And a
learning state that leaks across a block boundary would carry evidence from a market the participant was told
had ended.
"""
import re

import numpy as np
import pandas as pd
import pytest

from heiner_abm import human_learning as L

SMALL = 6


@pytest.fixture(scope="module")
def pilot():
    df, truth = L.synthetic_pilot(SMALL, model="confidence", seed=3)
    return df, truth, L.prepare(df)


# ============================================================================== information
def test_the_models_never_read_the_researchers_signal(pilot):
    """The registered hurdle models are scored against `shadow:opt_br`, the filtered best reply, which the
    module that owns it says is never shown. Deleting every shadow column must leave these models unaffected."""
    df, _, data = pilot
    stripped = df.drop(columns=[c for c in df.columns if c.startswith(L.FORBIDDEN_PREFIX)])
    assert len(stripped.columns) < len(df.columns), "the fixture has no shadow columns to remove"
    again = L.prepare(stripped)
    for a, b in zip(data, again):
        for field in ("d", "dq", "c_hat", "c_last", "profit_last", "cond", "block", "first", "scored"):
            assert np.allclose(np.nan_to_num(getattr(a, field)), np.nan_to_num(getattr(b, field)))
    p = L.fit_one("confidence", again[0], *L._weak_prior("confidence"))
    assert np.all(np.isfinite(p))


def test_the_module_does_not_read_a_shadow_column():
    """Prose may mention the researcher designs; code may not touch them. The failure mode is a column access,
    so that is what is checked - `df["shadow:opt_br"]` and anything shaped like it."""
    import inspect
    src = inspect.getsource(L)
    bad = re.findall(r"""\[\s*['"]shadow:""", src) + re.findall(r"""startswith\(\s*['"]shadow""", src)
    assert bad == [], bad
    assert L.FORBIDDEN_PREFIX == "shadow:"


def test_every_input_is_something_that_was_on_the_screen(pilot):
    """The data container may hold only fields `stepper.InteractiveMarket.info()` reports, plus design labels."""
    _, _, data = pilot
    from heiner_abm.stepper import InteractiveMarket
    screen = set(inspect_info_keys(InteractiveMarket))
    assert {"cost_estimate", "last_cost", "own_profit", "own_q", "last_price", "period"} <= screen
    # c_hat is cost_estimate, c_last is last_cost, profit_last is own_profit; the rest are design, not signal
    design = {"participant", "d", "dq", "cond", "block", "first", "history", "horizon", "scored"}
    assert set(data[0].__dataclass_fields__) - design == {"c_hat", "c_last", "profit_last"}


def inspect_info_keys(cls):
    import inspect
    src = inspect.getsource(cls.info)
    return [p.split("=")[0].strip() for p in src.split("dict(", 1)[1].rsplit(")", 1)[0].split(",")]


def test_feedback_is_lagged_so_no_decision_sees_its_own_outcome(pilot):
    """profit_last[t] must be the profit of t-1 in the same block, and undefined at a block start."""
    df, _, data = pilot
    x = data[0]
    starts = np.flatnonzero(x.first > 0)
    assert np.all(np.isnan(x.c_last[starts])) and np.all(np.isnan(x.profit_last[starts])),         "the first decision of a block was given feedback that did not exist yet"
    g = (df[df["participant"] == x.participant]
         .sort_values(["block_index", "period"]).reset_index(drop=True))
    for t in range(1, len(x.d)):
        if x.first[t]:
            continue
        assert x.profit_last[t] == pytest.approx(float(g["profit"].iloc[t - 1]))
        assert x.c_last[t] == pytest.approx(float(g["cost"].iloc[t - 1]))
        assert x.c_hat[t] == pytest.approx(float(g["cost_estimate"].iloc[t]))


# ============================================================================== block boundaries
def test_a_learning_state_does_not_cross_a_block_boundary(pilot):
    """Replace everything in the first block and the signals of the later blocks must not move."""
    _, truth, data = pilot
    x = data[0]
    theta = {k.replace("true:", ""): v for k, v in truth.iloc[0].items() if str(k).startswith("true:")}
    base = L.trial_signals("confidence", theta, x)
    first_block = x.block == x.block[0]
    rng = np.random.default_rng(0)
    y = L.LData(x.participant, x.d.copy(), x.dq.copy(), x.c_hat.copy(), x.c_last.copy(),
                x.profit_last.copy(), x.cond, x.block, x.first, x.history, x.horizon, x.scored)
    y.profit_last[first_block] = rng.normal(0, 500, int(first_block.sum()))
    y.d[first_block] = (rng.random(int(first_block.sum())) < 0.5).astype(float)
    y.c_hat[first_block] = rng.normal(40, 10, int(first_block.sum()))
    moved = L.trial_signals("confidence", theta, y)
    later = ~first_block
    for key in ("m", "uncertainty", "outcome_sd"):
        assert np.allclose(base[key][later], moved[key][later]), f"{key} leaked across a block boundary"
    assert not np.allclose(base["m"][first_block], moved["m"][first_block]), "the change had no effect at all"


def test_carry_state_is_off_and_documented():
    assert L.CARRY_STATE is False
    assert "CARRY_STATE" in L.__doc__ and "block" in L.__doc__


@pytest.mark.parametrize("model", L.MODELS)
def test_every_model_resets_every_piece_of_state(model):
    p = L.draw_participant(model, L.DEFAULT_POPULATION, np.random.default_rng(1))
    a = L.Learner(model, p)
    for t in range(20):
        a.signals(40.0 + t, 300.0)
        a.observe(300.0 + 10 * t, t % 2)
    before = {s: getattr(a, s) for s in a.__slots__ if s not in ("model", "p")}
    a.reset()
    after = {s: getattr(a, s) for s in a.__slots__ if s not in ("model", "p")}
    fresh = L.Learner(model, p)
    for s in after:
        assert np.all(np.asarray(after[s]) == np.asarray(getattr(fresh, s))), f"{s} survived a reset"
    assert any(np.any(np.asarray(before[s]) != np.asarray(after[s])) for s in before), "nothing was learned"


# ============================================================================== the mechanisms
def test_the_advantage_model_learns_which_action_pays():
    p = dict(alpha=0.4, a0=0.0, beta=1.0, cV=0, cN=0, cR=0, lapse=0.0, phi=0.0, sd=10.0)
    a = L.Learner("advantage", p)
    for _ in range(40):
        a.signals(40.0, 100.0)
        a.observe(1000.0, 1)          # adjusting always pays well
        a.signals(40.0, 100.0)
        a.observe(-200.0, 0)          # keeping always pays badly
    assert a.signals(40.0, 100.0)["m"] > 1.0


def test_uncertainty_falls_with_evidence_and_rises_without_it():
    p = dict(sigma_w=0.2, sigma_r=1.0, nu=0.3, a0=0, beta=1, omega=1, cV=0, cN=0, cR=0, lapse=0, phi=0, sd=10)
    a = L.Learner("confidence", p)
    start = a.signals(40.0, np.nan)["uncertainty"]
    for _ in range(15):
        a.observe(300.0, 1)
        a.observe(300.0, 0)
    both = a.signals(40.0, 300.0)["uncertainty"]
    assert both < start, "sampling both actions did not reduce uncertainty"

    b = L.Learner("confidence", p)
    for _ in range(30):
        b.observe(300.0, 0)           # only ever keeps
    only_keep = b.signals(40.0, 300.0)["uncertainty"]
    assert only_keep > both, "never trying the other action should leave it uncertain"


def test_the_change_belief_rises_with_surprise():
    p = dict(h=0.08, sigma_jump=6.0, k=0.3, a0=0, beta=1, lam=1, cV=0, cN=0, cR=0, lapse=0, phi=0, sd=10)
    a = L.Learner("changebayes", p)
    for _ in range(12):
        a.signals(40.0, 300.0)
    quiet = a.signals(40.2, 300.0)
    b = L.Learner("changebayes", p)
    for _ in range(12):
        b.signals(40.0, 300.0)
    jump = b.signals(70.0, 300.0)
    assert jump["change"] > quiet["change"] and jump["surprise"] > quiet["surprise"]
    assert 0.0 <= quiet["change"] <= 1.0 and 0.0 <= jump["change"] <= 1.0


def test_the_aspiration_tracks_profit_and_measures_shortfall():
    p = dict(eta=0.3, a0=0, beta=1, cV=0, cN=0, cR=0, lapse=0, phi=0, sd=10)
    a = L.Learner("aspiration", p)
    a.observe(1000.0, 0)
    for _ in range(30):
        a.observe(1000.0, 0)
    assert a.signals(40.0, 1000.0)["shortfall"] == pytest.approx(0.0, abs=0.05)
    assert a.signals(40.0, 200.0)["shortfall"] > 1.0, "a collapse in profit is not registered as a shortfall"


def test_risk_separates_outcome_spread_from_estimation_uncertainty():
    p = dict(sigma_w=0.2, sigma_r=1.0, nu=0.4, a0=0, beta=1, omega=1, gamma=1,
             cV=0, cN=0, cR=0, lapse=0, phi=0, sd=10)
    steady, noisy = L.Learner("risk", p), L.Learner("risk", p)
    rng = np.random.default_rng(2)
    for _ in range(40):
        steady.observe(500.0, 1)
        noisy.observe(500.0 + 400 * rng.standard_normal(), 1)
    s, n = steady.signals(40.0, 500.0), noisy.signals(40.0, 500.0)
    assert n["outcome_sd"] > 5 * s["outcome_sd"], "outcome variability is not being tracked"
    assert abs(n["uncertainty"] - s["uncertainty"]) < 0.3 * n["uncertainty"], \
        "a noisier world moved the ESTIMATION uncertainty as much as the outcome spread"


def test_confidence_sensitivity_and_risk_aversion_can_be_negative():
    """Constraining them positive would make the hypothesis unfalsifiable: every estimate would support it."""
    for name in ("omega", "gamma"):
        assert L.PARAMS[name][0] == "id"
        assert L._to_nat(name, -1.3) == pytest.approx(-1.3)


# ============================================================================== the protocol
def test_the_design_crosses_conditions_with_the_new_factors():
    from collections import Counter
    from heiner_abm.experiment import WILLIAMS
    pairs, within_arm = Counter(), Counter()
    for row in range(len(WILLIAMS)):
        for arm in range(4):
            for pos in range(4):
                cell = L.factor_cell(arm, pos)
                pairs[(WILLIAMS[row][pos], cell)] += 1
                within_arm[(arm, WILLIAMS[row][pos], cell)] += 1
    assert len(pairs) == 16 and set(pairs.values()) == {4}, "conditions are not balanced over factor cells"
    assert len(within_arm) == 64 and set(within_arm.values()) == {1}


def test_the_four_manipulations_move_different_things():
    t = L.protocol_table()
    assert len(t) == 4 and t["new"].sum() == 2
    assert set(t["manipulation"]) == {"outcome variability", "observation precision", "sampling history",
                                      "opportunities for further learning"}
    assert t["moves"].nunique() == 4 and t["within_participant"].all()


def test_forced_sampling_lowers_uncertainty_at_the_first_scored_decision():
    df, truth = L.synthetic_pilot(4, model="confidence", seed=3)
    data = L.prepare(df)
    x = data[0]
    theta = {k.replace("true:", ""): v for k, v in truth.iloc[0].items() if str(k).startswith("true:")}
    U = L.trial_signals("confidence", theta, x)["uncertainty"]
    sc = x.scored.astype(bool)
    firsts = {}
    for b in np.unique(x.block):
        sel = sc & (x.block == b)
        firsts[float(x.history[sel][0])] = U[sel][0]
    assert firsts[10.0] < 0.8 * firsts[0.0], firsts


def test_a_passive_lead_in_would_not_have_worked():
    """Recorded because it was built that way first: sampling only `keep` leaves the value of adjusting as
    uncertain as it started, so the factor moves nothing."""
    p = dict(sigma_w=0.3, sigma_r=1.0, nu=0.3, a0=0, beta=1, omega=1, cV=0, cN=0, cR=0, lapse=0, phi=0, sd=10)
    passive, forced = L.Learner("confidence", p), L.Learner("confidence", p)
    fresh = L.Learner("confidence", p).signals(40.0, np.nan)["uncertainty"]
    for i in range(10):
        passive.observe(300.0, 0)
        forced.observe(300.0, i % 2)
    assert passive.signals(40.0, 300.0)["uncertainty"] >= 0.9 * fresh
    assert forced.signals(40.0, 300.0)["uncertainty"] < 0.75 * fresh


def test_forced_periods_are_placed_where_the_setting_says():
    assert L.forced_periods(0, 32) == set()
    lead = L.forced_periods(10, 32, "lead")
    assert lead == set(range(10))
    split = L.forced_periods(10, 32, "split")
    assert len(split) == 10 and min(split) == 0 and max(split) > 15, split


def test_the_observation_only_periods_are_kept_but_not_scored(pilot):
    df, _, data = pilot
    assert "scored" in df and not df["scored"].all(), "no forced-sampling rows survived"
    x = data[0]
    assert x.scored.sum() < len(x.d)
    n_scored = int(x.scored.sum())
    assert L.fit_model("advantage", [x]).n_obs == n_scored, "unscored rows entered the likelihood as choices"


# ============================================================================== fitting and recovery
def test_the_likelihood_ignores_unscored_rows_but_learns_from_them(pilot):
    """Two separate claims. The forced periods contribute no CHOICE to the likelihood, and they do contribute
    EVIDENCE: dropping them changes the score, because the state at the scored decisions is different.

    The parameter vector has to put weight on the state, or the test passes for the wrong reason - the weak
    prior sets beta to zero, which makes the likelihood independent of everything the learner knows."""
    _, _, data = pilot
    x = data[0]
    theta = dict(a0=-0.3, beta=1.5, omega=1.0, sigma_w=0.3, sigma_r=1.0, cV=0.0, cN=0.0, cR=0.0,
                 lapse=0.03, phi=0.6, sd=15.0)
    u = np.array([L._to_unc(n, theta[n]) for n in L.param_names("confidence")])
    base = L.loglik("confidence", u, x)
    dropped = L.loglik("confidence", u, x.subset(x.scored.astype(bool)))
    assert np.isfinite(base) and np.isfinite(dropped)
    assert dropped != pytest.approx(base),         "dropping the forced periods changed nothing, so they taught the model nothing"
    assert int(x.scored.sum()) < len(x.d)


def test_participant_level_dependence_is_kept(pilot):
    _, _, data = pilot
    fit = L.fit_model("advantage", data)
    assert len(fit.params) == len(data)
    assert fit.params["beta"].nunique() > 1, "every participant got the same parameter"
    assert fit.n_params == sum(len(L.param_names("advantage")) for _ in data)


def test_held_out_blocks_are_whole(pilot):
    _, _, data = pilot
    h = L.heldout_scores("advantage", data[:3])
    assert len(h) > 0 and set(h["participant"]) <= {x.participant for x in data[:3]}
    assert (h["n"] > 0).all()


def test_confidence_sensitivity_recovers():
    rec, _ = L.parameter_recovery("confidence", n=24, seed=1)
    om = rec[rec["parameter"] == "omega"].iloc[0]
    assert om["correlation"] > 0.45, om
    assert abs(om["bias"]) < 0.4, f"omega is recovered in rank but not in magnitude: {om['bias']:+.3f}"


def test_the_plan_is_frozen_and_hashed():
    a, b = L.LearnPlan(), L.LearnPlan()
    assert a.digest == b.digest and len(a.digest) == 16
    assert L.LearnPlan(recovery_n=99).digest != a.digest
    with pytest.raises(Exception):
        a.seed = 1
    for needle in ("population", "model_params", "factor_cells", "carry_state"):
        assert needle in a.to_json()


def test_changing_an_update_equation_invalidates_the_plan():
    before_code, before_plan = L.learning_code_digest(), L.LearnPlan().digest
    original = L.adapt_logit
    try:
        L.adapt_logit = lambda *a, **k: None
        assert L.learning_code_digest() != before_code
        assert L.LearnPlan().digest != before_plan
    finally:
        L.adapt_logit = original
    assert L.learning_code_digest() == before_code and L.LearnPlan().digest == before_plan


# ============================================================================== honesty
def test_everything_generated_here_is_labelled_synthetic(pilot):
    df, _, _ = pilot
    assert df["synthetic"].all()


def test_the_module_says_what_its_numbers_are_not():
    doc = " ".join(L.__doc__.split())
    assert "not evidence that anybody does" in doc
    assert "SYNTHETIC" in doc
    for fn in (L.synthetic_pilot, L.parameter_recovery):
        assert "SYNTHETIC" in fn.__doc__


def test_the_identifiability_claims_are_stated_before_they_are_tested():
    assert set(L.IDENTIFIABILITY_CLAIMS) == {"I1", "I2", "I3", "I4", "I5"}
    for key, text in L.IDENTIFIABILITY_CLAIMS.items():
        assert len(text) > 40 and text.endswith(".")
    assert set(L.CONFOUNDS) <= set(L.MODELS)


# ============================================================================== robustness
@pytest.mark.parametrize("generating", L.MODELS)
def test_fitting_the_wrong_model_cannot_crash_the_optimiser(generating):
    """Regression. Model recovery fits every model to data generated by every other one, which sends the search
    somewhere no sensible parameter lives. L-BFGS-B is Fortran: handed a non-finite objective it does not raise,
    it segfaults the interpreter, and a whole study dies after thirty minutes with exit code 139. Bounds and a
    finite fallback make that impossible; this keeps them."""
    df, _ = L.synthetic_pilot(3, model=generating, seed=404)
    data = L.prepare(df)
    for model in L.MODELS:
        for x in data:
            u = L.fit_one(model, x, *L._weak_prior(model))
            assert np.all(np.isfinite(u)), (generating, model)
            lo = np.array([b[0] for b in L.bounds_for(model)])
            hi = np.array([b[1] for b in L.bounds_for(model)])
            assert np.all(u >= lo - 1e-9) and np.all(u <= hi + 1e-9)
            assert np.isfinite(L.loglik(model, u, x))


def test_the_objective_is_finite_even_at_absurd_parameters():
    df, _ = L.synthetic_pilot(2, model="risk", seed=7)
    x = L.prepare(df)[0]
    for model in L.MODELS:
        n = len(L.param_names(model))
        for value in (-10.0, 10.0, 0.0):
            u = np.full(n, value)
            assert np.isfinite(L.loglik(model, u, x)), (model, value)


def test_the_simulator_and_the_likelihood_use_one_choice_rule():
    """The generator and the estimator must not be two copies of the same arithmetic. If they drift, every
    recovery result afterwards measures the gap between two versions of a model rather than the model."""
    import inspect
    src = inspect.getsource(L.simulate_decision)
    assert "adapt_logit" in src, "the simulator computes its own linear predictor"
    for token in ("theta[\"beta\"] * sig", "theta['beta'] * sig"):
        assert token not in src, "a second copy of the choice rule is back in the simulator"

    # and it agrees numerically with the likelihood path, model by model
    for model in L.MODELS:
        theta = L.draw_participant(model, L.DEFAULT_POPULATION, np.random.default_rng(5))
        learner = L.Learner(model, theta)
        learner.observe(420.0, 1)
        state = L.Learner(model, theta)
        state.observe(420.0, 1)
        sig = state.signals(44.0, 420.0)
        one = L.LData("x", np.zeros(1), np.zeros(1), np.array([44.0]), np.array([41.0]),
                      np.array([420.0]), np.array([1]), np.zeros(1, int), np.ones(1),
                      np.zeros(1), np.zeros(1), np.ones(1))
        z_likelihood = float(L.adapt_logit(model, theta, L._WithKept(one, np.array([1.0])),
                                           {k: np.array([v]) for k, v in sig.items()})[0])
        assert np.isfinite(z_likelihood)


# ============================================================================== pinned choice rule
def _reference_simulate_decision(model, theta, learner, cond, c_hat, c_last, profit_last, q_prev, rng,
                                 kept_last=0.0):
    """The choice rule as it stood before `simulate_decision` was routed through `adapt_logit`.

    Kept here on purpose, in the same spirit as tests/test_traces.py: the refactor was verified bit-identical
    against this, and this is what keeps it so. If a future change to the choice rule is intended, this
    reference has to be updated deliberately and the results re-run - which is the point.
    """
    from scipy.special import expit as _expit
    sig = learner.signals(c_hat, profit_last)
    z = (theta["a0"] + {"B": 0.0, "V": theta["cV"], "N": theta["cN"], "R": theta["cR"]}[cond])
    if model == "advantage":
        z += theta["beta"] * sig["m"]
    elif model == "inertia_learn":
        z += theta["beta"] * sig["m"] - theta["g"] * kept_last
    elif model == "confidence":
        z += theta["beta"] * sig["m"] - theta["omega"] * sig["uncertainty"]
    elif model == "risk":
        z += (theta["beta"] * sig["m"] - theta["omega"] * sig["uncertainty"]
              - theta["gamma"] * sig["outcome_sd"])
    elif model == "changebayes":
        z += theta["beta"] * sig["surprise"] + theta["lam"] * sig["change"]
    elif model == "aspiration":
        z += theta["beta"] * sig["shortfall"]
    if rng.random() < theta["lapse"]:
        if rng.random() < 0.5:
            return float(max(5.0, q_prev + rng.integers(-400, 401))), True
        return q_prev, False
    if rng.random() >= _expit(z):
        return q_prev, False
    u = (c_hat - c_last) if np.isfinite(c_last) else 0.0
    for _ in range(50):
        delta = float(np.rint(-theta["phi"] * u + theta["sd"] * rng.standard_normal()))
        if delta != 0:
            return float(max(5.0, q_prev + delta)), True
    return float(max(5.0, q_prev + 1.0)), True


@pytest.mark.parametrize("model", L.MODELS)
def test_the_shared_choice_rule_is_bit_identical_to_the_reference(model):
    cols = ["participant", "block_index", "period", "q", "q_prev", "profit", "cost_estimate", "scored"]
    current = L.simulate_decision
    try:
        a, _ = L.synthetic_pilot(5, model=model, seed=77)
        L.simulate_decision = _reference_simulate_decision
        b, _ = L.synthetic_pilot(5, model=model, seed=77)
    finally:
        L.simulate_decision = current
    assert a[cols].equals(b[cols])
    assert np.array_equal(a["q"].to_numpy(), b["q"].to_numpy())
