"""Rival-theory machinery: stability index, scoring rules, logistic fit, tournament and horse race."""
import numpy as np
import pytest

from heiner_abm.analysis import auc, instability_index, logistic_fit, logistic_predict
from heiner_abm.experiments import EnvRanges, encompassing_test, horse_race, random_environments, rc_validation
from heiner_abm.params import default_scenario
from heiner_abm.theories import EXPERIMENTS, THEORIES, run_tournament, score, scoreboard


@pytest.mark.parametrize("n", [2, 4, 8])
def test_cournot_stability_matches_theocharis(n):
    """Symmetric Cournot partial adjustment is stable exactly when phi < 4/(n+1)."""
    s = default_scenario()
    s.firms = s.firms[:1] * n
    lim = 4.0 / (n + 1)
    for phi, stable in ((0.9 * lim, True), (1.1 * lim, False)):
        s = s.copy()
        for f in s.firms:
            f.rule, f.flex = "Cournot", min(phi, 1.0)
        if phi > 1.0:
            continue
        assert (instability_index(s) < 1) == stable


def test_bertrand_stability_uses_aggregate_mode():
    s = default_scenario()          # Bertrand, phi = 0.25..1, slope 0.06: 1 - s * sum(phi) = 0.85
    assert instability_index(s) == pytest.approx(1 - s.market.slope * sum(f.flex for f in s.firms))


def test_score_rules():
    assert score("-", "-") == "match" and score("-", "+") == "contradicted" and score("-", "0") == "inconclusive"
    assert score("0", "0") == "match" and score("0", "+") == "contradicted"
    assert score(">=0", "0") == "match" and score(">=0", "-") == "contradicted"
    assert score(None, "+") is None


def test_every_experiment_has_a_prediction_slot_per_theory():
    keys = {t.key for t in THEORIES}
    for e in EXPERIMENTS:
        assert set(e.predictions) == keys
        assert e.predictions["heiner"] is not None


def test_every_theory_is_tested_in_every_experiment_and_is_an_agent():
    """No theory is tested only on a convenient subset: each states a reasoned prediction for every experiment, and
    every theory of the laboratory is in the directional tournament and implemented as agents."""
    from heiner_abm.arena import THEORY_DESIGNS
    from heiner_abm.theory_content import THEORIES as CONTENT
    for e in EXPERIMENTS:
        assert all(e.predictions[t.key] is not None and e.why.get(t.key) for t in THEORIES), e.key
    assert {t.tournament_key for t in CONTENT} == {t.key for t in THEORIES}
    assert {t.key for t in CONTENT} == set(THEORY_DESIGNS) - {"ruleb"}


def test_agent_track_runs_every_theory_in_every_condition():
    from heiner_abm.agent_track import AGENT_EXPERIMENT_BY_KEY, agent_scoreboard, agent_slopes, agent_track
    from heiner_abm.arena import KEYS
    from heiner_abm.params import linear_flex_firms
    base = default_scenario()
    base.firms = linear_flex_firms(n=4)
    df = agent_track(base, 2, periods=150, burn_in=20, experiments=["volatility", "rivals"])
    for ex in ("volatility", "rivals"):
        assert len(df[df["experiment"] == ex]) == len(AGENT_EXPERIMENT_BY_KEY[ex].levels) * 2 * len(KEYS)
    assert set(df["theory"]) == set(KEYS)
    assert (df.loc[df["theory"] == "ruleb", "advantage"] == 0).all()
    sb = agent_scoreboard(df)
    assert set(sb["theory"]) == set(KEYS) and sb["mean_rank"].between(1, len(KEYS)).all()
    assert set(agent_slopes(df)["theory"]) == set(KEYS) - {"ruleb"}


def test_logistic_recovers_direction():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 1))
    y = rng.random(2000) < 1 / (1 + np.exp(-(0.5 + 2 * x[:, 0])))
    beta = logistic_fit(x, y)
    assert beta[1] == pytest.approx(2, abs=0.3)
    assert auc(logistic_predict(x, beta), y) > 0.75


def test_tournament_and_horse_race_run():
    b = default_scenario()
    b.periods, b.seed = 250, 1
    out = run_tournament(b, 4, horizon=2)
    assert set(out) == {e.key for e in EXPERIMENTS}
    assert all(o.observed in "+-0" for o in out.values())
    sb = scoreboard(out)
    assert len(sb) == len(THEORIES) and (sb["matches"] + sb["contradicted"] + sb["inconclusive"]
                                          == sb["predictions"]).all()
    df = rc_validation(random_environments(b, 12, EnvRanges(), seed=3), 1, horizon=3)
    hr = horse_race(df, n_boot=20)
    assert {"rc", "past", "neoclassical"} <= set(hr["key"])
    assert hr["auc"].between(0, 1).all()
    enc, gain = encompassing_test(df, n_boot=20)
    assert len(enc) == 3 and np.isfinite(gain["gain"])
