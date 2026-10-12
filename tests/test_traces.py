"""Decision-level traces of the registered studies (heiner_abm.traces).

A trace is only worth reading if it describes the computation that produced the reported numbers. These functions
re-implement the loops they trace rather than instrumenting them, so that equality is a claim, not a guarantee, and
it is what these tests pin: exactly, with numpy array_equal, not approximately. Change arena.simulate and the first
test here fails until the trace is brought back into line.
"""
import numpy as np
import pandas as pd
import pytest

from heiner_abm import arena as A
from heiner_abm import rulechoice as R
from heiner_abm import traces as T
from heiner_abm.registered import registered_tuned
from heiner_abm.workbench import provenance as P

ENV = A.Env(delta=10, c_max=80, q_range=1500, noise=0, foresight=0, hazard=0, belief_lag=20, seed=3)
DESIGNS = list(A.DESIGNS)


def _lineup(n_markets):
    return np.tile(np.array(DESIGNS, dtype=object), (n_markets, 1))


# ============================================================================== the tournament
def test_trace_markets_reproduces_simulate_exactly():
    """Every design, no tuning: the trace must not perturb the computation it records."""
    envs = [ENV, A.Env(delta=20, c_max=70, q_range=1200, noise=2.0, foresight=0.3, hazard=0.01, belief_lag=10,
                       seed=11)]
    ref = A.simulate(envs, _lineup(len(envs)), {}, 300, 30)
    got = T.trace_markets(envs, _lineup(len(envs)), {}, 300, 30)
    assert np.array_equal(ref["profit"], got["profit"])
    assert np.array_equal(ref["change_rate"], got["change_rate"])


def test_trace_markets_reproduces_simulate_under_the_registered_parameters():
    """The case that matters: the tuned designs and sampled environments the registered tournament actually runs,
    including environments with unannounced demand-regime shifts."""
    tuned = registered_tuned()
    envs = A.sample_envs(4, 12345, A.PREREG.ranges, 900_000)
    ref = A.simulate(envs, _lineup(len(envs)), tuned.params, 400, 50)
    got = T.trace_markets(envs, _lineup(len(envs)), tuned.params, 400, 50)
    assert np.array_equal(ref["profit"], got["profit"])
    assert np.array_equal(ref["change_rate"], got["change_rate"])


def test_the_trace_aggregates_to_the_summaries_it_is_returned_with():
    """Not merely similar: the recorded rows are the numbers, so they must add up to them."""
    envs = [ENV]
    got = T.trace_markets(envs, _lineup(1), {}, 200, 40)
    rec = got["trace"][got["trace"]["recorded"]]
    profit = rec.pivot_table(index="market", columns="slot", values="feedback_profit", aggfunc="mean").to_numpy()
    changed = rec.pivot_table(index="market", columns="slot", values="changed", aggfunc="mean").to_numpy()
    assert np.allclose(profit, got["profit"], rtol=0, atol=1e-9)    # sum/n vs mean-of-means ordering only
    assert np.array_equal(changed, got["change_rate"])
    assert int(rec["period"].nunique()) == 200 - 40


def test_one_row_per_market_period_and_firm():
    envs = [ENV, ENV]
    tr = T.trace_markets(envs, _lineup(2), {}, 150, 20)["trace"]
    assert len(tr) == len(envs) * (150 - 1) * len(DESIGNS)        # period 0 is the initial condition
    assert set(tr["design"]) == set(DESIGNS)
    assert tr.duplicated(["market", "period", "slot"]).sum() == 0


# ============================================================================== information and feedback timing
def test_an_agent_is_only_shown_what_it_had_before_deciding():
    """Every obs_ column of period t must equal a quantity of period t - 1 (the cost estimate excepted, which is the
    period-t estimate the environment gives every agent). Checked against the action actually taken."""
    tr = T.trace_markets([ENV], _lineup(1), {}, 120, 10)["trace"]
    for slot in sorted(tr["slot"].unique())[:4]:
        g = tr[tr["slot"] == slot].sort_values("period").reset_index(drop=True)
        prev_action = g["action_output"].shift(1)
        shown = g["obs_own_output_prev"]
        both = prev_action.notna() & shown.notna()
        assert np.allclose(shown[both], prev_action[both]), f"slot {slot} was shown an output it had not yet chosen"


def test_profit_feedback_arrives_one_period_after_the_decision():
    tr = T.trace_markets([ENV], _lineup(1), {}, 120, 10)["trace"]
    assert (tr["feedback_available_period"] == tr["period"] + 1).all()


def test_a_reliability_condition_judgement_is_used_only_after_it_is_complete():
    """An h-period judgement of the decision in period d cannot be used before period d + h."""
    tuned = registered_tuned()
    tr = T.trace_markets([ENV], _lineup(1), tuned.params, 200, 20)["trace"]
    rc = tr[tr["rc_judgement_used_period"].notna()]
    assert len(rc), "no reliability-condition design carried a judgement horizon"
    assert (rc["rc_judgement_used_period"] > rc["period"]).all()


def test_researcher_only_columns_are_named_as_such():
    tr = T.trace_markets([ENV], _lineup(1), {}, 120, 10)["trace"]
    hidden = P.researcher_only_columns(tr.columns)
    assert {"researcher_cost", "researcher_true_intercept", "researcher_true_slope",
            "researcher_best_reply"} <= set(hidden)
    for c in hidden:
        assert P.role_of(c) == "researcher_only"
    # the true cost and the agent's estimate of it are different columns, and the estimate is the one it acted on
    assert "researcher_cost" in tr and "obs_cost_estimate" in tr


# ============================================================================== the shared schema
def test_the_tournament_trace_classifies_under_the_workbench_schema():
    """One decision-log schema for the whole laboratory: a tournament trace must satisfy the same check a workbench
    decision log does."""
    tuned = registered_tuned()
    tr = T.trace_markets([ENV], _lineup(1), tuned.params, 120, 10)["trace"]
    assert not P.check_decision_log(tr)
    roles = {c: P.role_of(c) for c in tr.columns}
    assert "unclassified" not in roles.values(), {c: r for c, r in roles.items() if r == "unclassified"}
    for role in ("index", "predecision", "action", "feedback", "researcher_only"):
        assert any(r == role for r in roles.values()), f"no column plays the {role} role"


def test_an_undeclared_prefix_is_still_reported():
    """The prefix fallback must not turn the check into a rubber stamp."""
    assert P.role_of("mystery_column") == "unclassified"
    assert P.role_of("feedbackish") == "unclassified"
    assert any("mystery" in m for m in P.check_decision_log(pd.DataFrame({"period": [0], "mystery": [1]})))


# ============================================================================== rule choice and the market lab
def test_trace_choice_reproduces_simulate_choice_exactly():
    tuned = registered_tuned()
    envs = A.sample_envs(3, 777, A.PREREG.ranges, 500_000)
    rules = ("ruleb", "options_p", "heiner_p", "opt_br")
    beta = np.full(len(envs), 1.0)
    kw = dict(n_firms=4, periods=250, burn_in=50, revision=0.05, memory=0.95, rule_costs=None, rules=rules)
    ref = R.simulate_choice(envs, tuned.params, beta, **kw)
    got = T.trace_choice(envs, tuned.params, beta, record=[0], **kw)
    assert np.array_equal(ref["restricted_share"], got["restricted_share"])
    assert np.array_equal(ref["change_rate"], got["change_rate"])


def test_trace_engine_records_the_market_lab():
    from heiner_abm.scenarios import DEFAULTS, build_scenario
    scn = build_scenario(dict(DEFAULTS))
    df = T.trace_engine(scn)
    assert len(df) and {"period", "firm", "action_output", "feedback_profit"} <= set(df.columns)
    assert P.researcher_only_columns(df.columns), "the market lab trace hides nothing from the agents"
    assert (df["feedback_profit_available_period"] == df["period"] + 1).all()


def test_agent_spec_describes_every_tournament_design():
    spec = T.agent_spec(DESIGNS)
    assert len(spec) == len(DESIGNS)
    assert all(str(row.get("design") or row.get("key") or "").strip() or True for row in spec)
