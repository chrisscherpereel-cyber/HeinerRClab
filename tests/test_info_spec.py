"""The information-and-feedback specification: restrictions, delays and unsupported agent-treatment combinations."""
import dataclasses

import numpy as np
import pytest

from heiner_abm.agents import Industry, Market
from heiner_abm.engine import run_batch
from heiner_abm.information import (AGENTS, AGENT_BY_KEY, ENGINES, OBSERVATION_FIELDS, RESEARCHER_ONLY, InfoSpec,
                                    Observation, compatibility, export_record)
from heiner_abm.params import EvolutionParams, FirmSpec, Scenario

SPECS = [InfoSpec(), InfoSpec(obs_delay=2), InfoSpec(obs_noise=3.0), InfoSpec(obs_delay=1, obs_noise=2.0),
         InfoSpec(demand_knowledge="true"), InfoSpec(regime_announced=True), InfoSpec(rivals_visible=False),
         InfoSpec(feedback="oracle", obs_delay=2)]


def _market(spec, periods=140, adaptive=True, structural=True, seed=3, kappa=0.3):
    sel = "Adaptive" if adaptive else "Large"
    firms = [FirmSpec("Bertrand", 0.8, sel, threshold=6), FirmSpec("Cournot", 0.5, sel, noise=2.0, threshold=6),
             FirmSpec("Bertrand", 1.1, "Always"), FirmSpec("Cournot", 0.4, "Small", foresight=kappa, threshold=12)]
    s = Scenario(firms=firms, periods=periods, burn_in=10, seed=seed, info=dataclasses.replace(spec))
    s.market.delta = 15.0
    s.adaptive.window, s.adaptive.memory = 5, 0.8
    if structural:
        s.structural.enabled, s.structural.hazard, s.structural.belief_lag = True, 0.05, 4
    return s


# ------------------------------------------------------------------------------------------------ engines agree
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: str(s.to_dict()))
@pytest.mark.parametrize("horizon,cont", [(1, "default"), (4, "default"), (4, "rules")])
def test_reference_and_vectorized_engines_agree_under_every_spec(spec, horizon, cont):
    scn = _market(spec)
    ind = Industry(scn, horizon=horizon, continuation=cont).run()
    res = run_batch([scn], record_firm_history=True, horizon=horizon, continuation=cont)
    np.testing.assert_allclose(res.price[0], ind.price)
    for k in ("q", "rec", "profit_rule", "profit_default"):
        np.testing.assert_allclose(res.firm_hist[k][0, 1:], ind.h[k][1:], err_msg=k)
    np.testing.assert_array_equal(res.firm_hist["deviate"][0, 1:], ind.h["deviate"][1:])
    assert res.meta["info"][0] == spec.to_dict()


def test_mixed_specs_in_one_batch_match_individual_runs():
    scns = [_market(s, seed=i) for i, s in enumerate(SPECS)]
    together = run_batch(scns, record_firm_history=True, horizon=3)
    for i, s in enumerate(scns):
        alone = run_batch([s], record_firm_history=True, horizon=3)
        np.testing.assert_allclose(together.firm_hist["q"][i], alone.firm_hist["q"][0])


# ------------------------------------------------------------------------------------------------ observations
def test_observation_carries_no_researcher_only_or_future_field():
    assert not set(OBSERVATION_FIELDS) & set(RESEARCHER_ONLY)
    # 'none' rejects the Cournot rule, so check the observation on a margin-feedback-only market
    scn = Scenario(firms=[FirmSpec("Bertrand", 0.8), FirmSpec("Bertrand", 1.2)], periods=40, burn_in=5, seed=1,
                   info=InfoSpec(obs_delay=3, rivals_visible=False, demand_knowledge="none"))
    ind = Industry(scn).run()
    from heiner_abm.agents import _History
    hist = _History(ind.price, ind.quantity, ind.h["q"], ind.h["profit"])
    obs = ind.observe(0, 20, hist)
    assert isinstance(obs, Observation)
    assert obs.observed_period == 16 and obs.price == ind.price[16] and obs.market_quantity == ind.quantity[16]
    assert obs.rivals is None and obs.demand_model is None and obs.regime_announced is None


@pytest.mark.parametrize("delay", [0, 2, 5])
def test_delayed_observations_ignore_the_last_d_periods(delay):
    """With delay d and no cost foresight, a cost shock after period K cannot change any decision before K + 2 + d."""
    K = 60
    scn = _market(InfoSpec(obs_delay=delay), adaptive=False, structural=False, kappa=0.0)
    base = Industry(scn, horizon=1).run()
    pert = Industry(scn, horizon=1)
    pert.u[K + 1:] = np.random.default_rng(5).uniform(-1, 1, len(pert.u) - K - 1)
    pert.market = Market(scn, pert.u)
    pert.run()
    last_same = K + 1 + delay
    np.testing.assert_array_equal(base.h["q"][:last_same + 1], pert.h["q"][:last_same + 1])
    assert not np.array_equal(base.h["q"], pert.h["q"])


@pytest.mark.parametrize("delay", [1, 4])
@pytest.mark.parametrize("feedback,span", [("estimated", 5), ("oracle", 3)])
def test_feedback_is_released_after_maturity_plus_delay(delay, feedback, span):
    scn = _market(InfoSpec(obs_delay=delay, feedback=feedback), structural=False)
    ind = Industry(scn, horizon=3).run()
    assert ind.feedback_log
    for released, firm, decided, matures, *_ in ind.feedback_log:
        assert matures == decided + span - 1 and released == matures + delay
    assert all(item.release > scn.periods - 1 for item in ind.pending)       # never released past the end


def test_observation_noise_uses_its_own_stream():
    quiet = run_batch([_market(InfoSpec(), adaptive=False)], record_firm_history=True)
    noisy = run_batch([_market(InfoSpec(obs_noise=4.0), adaptive=False)], record_firm_history=True)
    np.testing.assert_array_equal(quiet.cost, noisy.cost)                     # cost shocks are unchanged
    assert not np.array_equal(quiet.firm_hist["q"], noisy.firm_hist["q"])    # decisions see the noise


# ------------------------------------------------------------------------------------------------ unsupported
@pytest.mark.parametrize("spec,firm,evo,needle", [
    (InfoSpec(feedback="chosen"), FirmSpec("Bertrand", 1.0, "Adaptive"), False, "does not provide"),
    (InfoSpec(feedback="full"), FirmSpec("Bertrand", 1.0, "Always"), False, "full feedback is not available"),
    (InfoSpec(demand_knowledge="none"), FirmSpec("Cournot", 0.5, "Always"), False, "needs a demand model"),
    (InfoSpec(demand_knowledge="none"), FirmSpec("Bertrand", 1.0, "Adaptive"), False, "estimates counterfactual"),
    (InfoSpec(rivals_visible=False), FirmSpec("Bertrand", 1.0, "Always"), True, "rivals' individual"),
])
def test_unsupported_combinations_are_rejected_not_run(spec, firm, evo, needle):
    scn = Scenario(firms=[firm, FirmSpec("Bertrand", 0.5)], periods=60, burn_in=5, seed=1, info=spec,
                   evolution=EvolutionParams(enabled=evo))
    errs = scn.validate()
    assert any(needle in e for e in errs), errs
    with pytest.raises(ValueError):
        run_batch([scn])
    with pytest.raises(ValueError):
        Industry(scn)


def test_chosen_action_feedback_runs_agents_that_do_not_need_counterfactuals():
    scn = _market(InfoSpec(feedback="chosen"), adaptive=False)
    assert scn.validate() == []
    run_batch([scn])


def test_fixed_engines_reject_non_native_specs_and_label_oracle_variants():
    arena, tasks = ENGINES["arena"], ENGINES["tasks"]
    assert arena.check(arena.native) == [] and tasks.check(tasks.native) == []
    assert arena.check(InfoSpec(obs_delay=2)) and arena.check(InfoSpec(feedback="full"))
    imit = AGENT_BY_KEY[("arena", "imit_best")]
    assert compatibility(imit, InfoSpec(rivals_visible=False), "tasks").status == "unsupported"
    rc_task = AGENT_BY_KEY[("tasks", "rc_learned")]
    assert compatibility(rc_task, InfoSpec(feedback="chosen")).status == "unsupported"
    assert compatibility(AGENT_BY_KEY[("arena", "heiner_m_oracle")], arena.native).status == "oracle variant"
    assert compatibility(AGENT_BY_KEY[("tasks", "rc_oracle")], tasks.native).status == "oracle variant"
    for a in AGENTS:                       # under its own engine's native spec every agent runs, or is a labeled oracle
        assert compatibility(a, ENGINES[a.engine].native).status in ("supported", "oracle variant"), a.key


def test_export_record_describes_spec_engine_and_every_agent():
    rec = export_record("market", InfoSpec(obs_delay=2, feedback="oracle"))
    assert rec["spec"]["obs_delay"] == 2 and rec["oracle_information"]
    assert {a["key"] for a in rec["agents"]} == {a.key for a in AGENTS if a.engine == "market"}
    assert all("access" in a and "status" in a for a in rec["agents"])


def test_participant_sees_only_the_completed_period_without_demand_model_or_rivals():
    from heiner_abm import arena, stepper
    env = arena.Env(delta=10.0, c_max=80.0, q_range=1500.0, noise=2.0, foresight=0.0, hazard=0.0, belief_lag=5, seed=2)
    im = stepper.InteractiveMarket(env, ["cobweb_p", "options_p"], {}, ["heiner_p"], 20)
    im.step(150.0)
    obs = im.observation()
    assert obs.period == 2 and obs.observed_period == 1
    assert obs.demand_model is None and obs.rivals is None
    assert obs.price == float(im.mk.P_prev[0]) and im.info()["last_price"] == obs.price


def test_saved_scenarios_from_before_the_specification_still_load():
    import json
    from dataclasses import asdict
    from ui.common import from_json, to_json
    d = asdict(_market(InfoSpec()))
    d.pop("info")
    d["adaptive"]["feedback"] = "observable"            # the earlier field and value
    s = from_json(json.dumps(d))
    assert s.info.feedback == "estimated" and s.validate() == []
    d["adaptive"]["feedback"] = "oracle"
    assert from_json(json.dumps(d)).info.feedback == "oracle"
    assert from_json(to_json(_market(InfoSpec(obs_delay=2)))).info.obs_delay == 2
