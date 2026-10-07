"""Human experiment protocol 2.0 (heiner_abm.experiment), its analysis (heiner_abm.human_analysis) and mechanism models
(heiner_abm.human_models): separated treatments, balanced assignment and counterbalancing, recording, estimators that
respect the dependence structure, simulation-based power, recovery studies and a full participant session."""
import os
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from heiner_abm import experiment as E
from heiner_abm import human_analysis as HA
from heiner_abm import human_models as HM
from heiner_abm import registered

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_protocol_is_frozen():
    assert E.PLAN.digest == registered.EXPERIMENT_PLAN


def test_each_treatment_changes_exactly_one_mechanism():
    base = E.CONDITIONS["B"]
    for c in ("V", "N", "R"):
        diff = [k for k in base if E.CONDITIONS[c][k] != base[k]]
        assert len(diff) == 1, (c, diff)
    assert {E.CONDITIONS[c][k] != base[k] for c, k in (("V", "delta"), ("N", "noise"), ("R", "hazard"))} == {True}


def test_williams_square_is_balanced():
    W = E.WILLIAMS
    for pos in range(4):
        assert sorted(r[pos] for r in W) == [0, 1, 2, 3]
    pairs = Counter((r[i], r[i + 1]) for r in W for i in range(3))
    assert len(pairs) == 12 and set(pairs.values()) == {1}


@pytest.mark.parametrize("n", [16, 64])
def test_allocation_is_balanced_and_never_confounds_condition_with_path(n):
    alloc = E.allocation_schedule(n)
    a = pd.DataFrame([x.to_dict() for x in alloc])
    for start in range(0, n, 16):
        blk = a.iloc[start:start + 16]
        assert (pd.crosstab(blk["williams_row"], [blk["aid"], blk["elicit"]]) == 1).all().all()
        assert (pd.crosstab(blk["williams_row"], blk["path_offset"]) == 1).all().all()
        assert (pd.crosstab([blk["aid"], blk["elicit"]], blk["path_offset"]) == 1).all().all()
    cells = Counter((c, x.path(c)) for x in alloc for c in E.PLAN.conditions)
    assert len(cells) == 16 and len(set(cells.values())) == 1          # every condition meets every path equally
    for x in alloc:
        assert sorted(x.path(c) for c in E.PLAN.conditions) == [0, 1, 2, 3]
    assert E.allocation_schedule(n) == alloc                            # reproducible


def test_paths_share_draws_across_conditions():
    b, n = E.block_market("B", 2), E.block_market("N", 2)
    np.testing.assert_array_equal(b.mk.eps, n.mk.eps)                   # same noise draws, different scale
    np.testing.assert_array_equal(b.mk.cost, n.mk.cost)                 # noise does not touch the cost path
    other = E.block_market("B", 3)
    assert not np.array_equal(b.mk.cost, other.mk.cost)


def test_recorded_fields_and_unaided_control():
    aided = next(x for x in E.allocation_schedule(16) if x.aid)
    control = next(x for x in E.allocation_schedule(16) if not x.aid)
    for a in (aided, control):
        m = E.block_market("V", a.path("V"))
        m.info()
        row = m.step(m.pending["shadows"]["ruleb"])
        r = E.decision_row(m, row, a, 1, "V", a.path("V"), dict(rt_decision=2.0, info_history=False, info_cost=True))
        for k in ("condition", "block_index", "path", "order", "aid", "elicit", "aid_recommendation", "aid_shown",
                  "feedback_price", "feedback_profit", "rt_decision", "info_cost", "q", "plan"):
            assert k in r
        assert r["aid_shown"] == a.aid and np.isfinite(r["aid_recommendation"])   # computed in both arms


@pytest.fixture(scope="module")
def pilot():
    return E.synthetic_pilot(32, seed=7)


def test_synthetic_pilot_is_labeled_and_complete(pilot):
    df, truth = pilot
    assert df["synthetic"].all() and df["participant"].str.startswith("synthetic").all()
    q = HA.quality(df)
    assert not q["excluded"].any() and q["synthetic"].all()
    s = HA.block_summary(df)
    assert (s.groupby("participant").size() == 4).all()
    assert df[df["elicit"]]["belief_price"].notna().sum() == 32 // 2 * 4 * (E.PLAN.periods_per_block // 3)


def test_within_effects_recover_a_known_contrast_and_refuse_confounding():
    comp = dict(outcome="adjust_rate", mean=0.4, V=0.15, N=0.0, R=-0.1, aid=0.05, elicit=0.0, sd_participant=0.15,
                sd_path=0.05, sd_residual=0.1, pilot_participants=0)
    s = HA.simulate_summary(comp, 128, np.random.default_rng(3))
    w = HA.within_effects(s, "adjust_rate").set_index("contrast")
    for c, v in (("V", 0.15), ("N", 0.0), ("R", -0.1)):
        assert w.loc[f"{c} − B", "lo"] < v < w.loc[f"{c} − B", "hi"]
    bad = s.copy()
    bad["path"] = bad["condition"].map({"B": 0, "V": 1, "N": 2, "R": 3})
    with pytest.raises(ValueError):
        HA.within_effects(bad, "adjust_rate")


def test_power_is_reproducible_and_null_effects_have_low_power():
    comp = dict(outcome="adjust_rate", mean=0.4, V=0.2, N=0.0, R=0.0, aid=0.0, elicit=0.0, sd_participant=0.15,
                sd_path=0.05, sd_residual=0.1, pilot_participants=0)
    a = HA.power_analysis(comp, (32,), reps=60, seed=5)
    b = HA.power_analysis(comp, (32,), reps=60, seed=5)
    pd.testing.assert_frame_equal(a, b)
    p = a.set_index("effect")["power"]
    assert p["V − B"] > 0.9 and p["N − B"] <= 0.15 and p["Decision aid (vs unaided control)"] <= 0.15
    rq = HA.required_n(a)
    assert rq.set_index("effect").loc["V − B", "required_n"] == 32


def test_associational_result_is_labeled(pilot):
    s = HA.block_summary(pilot[0])
    a = HA.associational(s)
    assert len(a) and a["status"].str.startswith("ASSOCIATIONAL").all()


def test_models_fit_predict_and_compare(pilot):
    data = HM.prepare(pilot[0])
    comp, fits, per = HM.compare_models(data[:12])
    assert set(comp["model"]) == set(HM.MODELS)
    assert np.isfinite(comp[["log_score_per_decision", "brier", "rmse_output"]].to_numpy()).all()
    assert (comp["brier"] < 0.25).all()                                   # better than a coin
    mech = HA.mechanism_effects(fits["full"])
    assert {"cV", "cN", "cR", "lapse", "rho"} <= set(mech["parameter"])


def test_parameter_recovery_and_confusion_matrix():
    pr, _ = HM.parameter_recovery(24, seed=1)
    r = pr.set_index("parameter")["correlation"]
    assert r["a"] > 0.6                                             # weakly identified parameters are shown, not tested
    assert set(HM.MODEL_PARAMS["full"]) <= set(pr["parameter"])
    conf, per = HM.model_recovery(4, seed=2, criterion="bic")
    assert list(conf.index) == list(HM.MODELS) and list(conf.columns) == list(HM.MODELS)
    np.testing.assert_allclose(conf.sum(axis=1), 1.0)


def test_analysis_plan_document():
    comp = [dict(outcome="adjust_rate", mean=0.4, V=0.1, N=0.0, R=0.0, aid=0.0, elicit=0.0, sd_participant=0.1,
                 sd_path=0.0, sd_residual=0.1, pilot_participants=16)]
    pw = HA.power_analysis(comp[0], (16, 32), reps=20)
    doc = HA.analysis_plan(E.PLAN, comp, pw, "SYNTHETIC pilot", registered.EXPERIMENT_PLAN)
    assert "**not** been preregistered" in doc and "The pilot is synthetic" in doc
    for h in E.PLAN.hypotheses:
        assert h[0] in doc
    assert "associat" in doc.lower()


def test_full_participant_session():
    """Consent, comprehension (one failed attempt), practice, four blocks with the aid and belief screens, download."""
    from streamlit.testing.v1 import AppTest
    slot = next(x.slot for x in E.allocation_schedule(16) if x.aid and x.elicit)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120)
    at.run()
    at.switch_page("app_pages/play_market.py").run()
    at.text_input(key="play_pid").set_value("tester-1").run()
    at.text_input(key="play_slot").set_value(str(slot)).run()
    at.checkbox(key="play_consent").check().run()
    at.button(key="play_start").click().run()
    qs = E.COMPREHENSION + (E.AID_QUESTION,)
    for k, _, opts, right in qs:                                         # first attempt: one wrong answer
        at.radio(key=f"play_comp_{k}_0").set_value(opts[(right + 1) % len(opts)] if k == "price" else opts[right])
    btn = lambda label: next(b for b in at.button if b.label.startswith(label))
    btn("Check answers").click().run()
    assert at.session_state["play"]["stage"] == "comprehension"
    for k, _, opts, right in qs:
        at.radio(key=f"play_comp_{k}_1").set_value(opts[right])
    btn("Check answers").click().run()
    assert at.session_state["play"]["stage"] == "practice"
    steps = 0
    while (at.session_state["play"]["stage"] != "done" or at.session_state["play"].get("belief_for") is not None) \
            and steps < 400:
        steps += 1
        if at.session_state["play"].get("belief_for") is not None:
            at.number_input[0].set_value(50.0)
            btn("Continue").click().run()
        else:
            btn("Keep").click().run()
        assert not at.exception, [e.value for e in at.exception]
    recs = pd.DataFrame(at.session_state["play"]["records"])
    assert len(recs) == E.PLAN.practice_periods + 4 * E.PLAN.periods_per_block
    main = recs[~recs["practice"]]
    assert sorted(main["condition"].unique()) == sorted(E.PLAN.conditions)
    assert main["aid_shown"].all() and (main["rt_decision"] > 0).all()
    assert main["elicited"].sum() == 4 * (E.PLAN.periods_per_block // E.PLAN.elicit_every)
    assert (main.loc[main["elicited"], "rt_belief"] > 0).all()
    assert at.session_state["play"]["attempts"] == 2
    assert any(b.label.startswith("Download the data") for b in at.get("download_button"))
    q = HA.quality(recs.assign(rt_decision=2.0))
    assert not q["excluded"].any()
