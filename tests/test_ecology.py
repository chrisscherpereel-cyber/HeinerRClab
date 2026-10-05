"""Organizational ecology as agents, and selection through exit and entry."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heiner_abm import arena  # noqa: E402
from heiner_abm.ecology import (ECO_DESIGNS, INERT_FORMS, QUICK_ECO, EcoPlan, run_ecology_study,  # noqa: E402
                                simulate_eco)
from heiner_abm.registered import TOURNAMENT_PLAN, registered_tuned  # noqa: E402


def _envs(n=3, seed=3):
    return arena.sample_envs(n, seed, arena.PREREG.ranges, seed_base=900)


def test_simulator_reproduces_arena_exactly():
    tuned = registered_tuned()
    envs = _envs()
    lineup = np.tile(np.array(tuned.lineup(), dtype=object), (len(envs), 1))
    a = arena.simulate(envs, lineup, tuned.params, 250, 30)
    b = simulate_eco(envs, lineup, tuned.params, 250, 30)
    np.testing.assert_array_equal(a["profit"], b["profit"])
    np.testing.assert_array_equal(a["change_rate"], b["change_rate"])


def test_registered_tournament_hash_unchanged():
    """The ecology study lives outside the registered tournament: its designs must not change the plan hash."""
    assert arena.PREREG.digest == TOURNAMENT_PLAN
    assert not set(ECO_DESIGNS) & set(arena.DESIGNS)


def test_inert_firm_keeps_founding_output():
    envs = _envs(2)
    lineup = np.array([["eco_inert", "opt_nash", "ruleb"]] * 2, dtype=object)
    out = simulate_eco(envs, lineup, {"eco_inert": {"f": 1.2}}, 200, 20)
    assert out["change_rate"][:, 0].max() <= 1.0 / 180 + 1e-9      # at most the founding move


def test_selection_exits_and_entry_draw_from_candidates():
    tuned = registered_tuned()
    forms = tuned.lineup() + ["eco_inert"]
    envs = _envs(3, 5)
    lineup = np.tile(np.array(forms, dtype=object), (3, 1))
    params = {**tuned.params, "eco_inert": {"f": 1.0}}
    out = simulate_eco(envs, lineup, params, 1500, 30, selection=True, candidates=forms, capital_periods=2.0)
    assert out["exits_by_form"].sum() > 0
    assert set(np.unique(out["types"])) <= set(range(len(out["pool"])))
    assert set(INERT_FORMS) & set(out["pool"])


def test_quick_study_runs_and_plan_hash_is_stable():
    res = run_ecology_study(QUICK_ECO, registered_tuned())
    assert res.design in ECO_DESIGNS
    assert set(res.verdicts["id"]) == {"S1", "S2", "S3", "S4"}
    assert EcoPlan().digest == EcoPlan().digest
    assert res.head_to_head["theory"].nunique() == len(arena.KEYS) + 1
