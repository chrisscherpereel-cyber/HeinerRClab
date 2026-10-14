"""The replication specification and the rule that turns two verdicts into a claim's fate.

The whole exercise is worth nothing if the claim table misquotes what was originally found, or if the decision
rule could have been chosen after the results arrived. Most of what follows checks those two things.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heiner_abm import adjudication as A

ROOT = Path(A.__file__).resolve().parents[1]


# ============================================================================== the frozen plan
def test_the_plan_is_frozen_and_covers_the_rule():
    a, b = A.AdjudicationPlan(), A.AdjudicationPlan()
    assert a.digest == b.digest and len(a.digest) == 16
    blob = a.to_json()
    for needle in ("claim_table", "outcomes", "seed_offset", "retain_fraction", "code"):
        assert needle in blob
    with pytest.raises(Exception):
        a.seed_offset = 1


def test_changing_the_rule_or_the_claims_invalidates_the_plan():
    """A decision rule that could be edited without moving the hash could be edited after seeing results."""
    before_code, before_plan = A.adjudication_code_digest(), A.AdjudicationPlan().digest
    original = A.RETAIN_FRACTION
    try:
        A.RETAIN_FRACTION = 0.9
        assert A.adjudication_code_digest() != before_code
        assert A.AdjudicationPlan().digest != before_plan
    finally:
        A.RETAIN_FRACTION = original
    assert A.adjudication_code_digest() == before_code


def test_the_seed_offset_is_the_only_thing_that_differs():
    """If the offset were zero this would be reproduction, which cannot move a claim."""
    assert A.SEED_OFFSET != 0
    assert A.AdjudicationPlan().seed_offset == A.SEED_OFFSET


# ============================================================================== the claim table is honest
def test_every_claim_names_a_study_that_has_a_runner():
    for c in A.CLAIMS:
        assert c.study in A.STUDIES
        assert c.study in A.RUNNERS
        assert c.statement.strip().endswith(".") and len(c.statement) > 30
        assert c.original_verdict in ("supported", "not supported")
        assert c.quantity.strip()


def test_the_claim_keys_are_unique_and_listed_in_the_plan():
    keys = [c.key for c in A.CLAIMS]
    assert len(keys) == len(set(keys))
    assert tuple(keys) == A.AdjudicationPlan().claims


@pytest.mark.parametrize("family,policy,expected_rank", [
    ("base", "bayes_true", 1), ("multimodal", "saa", 1), ("outside_support", "dro_moment", 4)])
def test_the_stress_originals_match_the_stored_results(family, policy, expected_rank):
    """The frozen table quotes the first run. If it quotes it wrongly, every outcome below is wrong."""
    path = ROOT / "docs" / "stress_results" / "summary.csv"
    if not path.exists():
        pytest.skip("stress results not present in this checkout")
    s = pd.read_csv(path)
    g = s[s["family"] == family].sort_values("regret").reset_index(drop=True)
    assert int(g.index[g["policy"] == policy][0]) + 1 == expected_rank


def test_the_information_originals_match_the_stored_results():
    path = ROOT / "docs" / "information_results" / "cells.csv"
    if not path.exists():
        pytest.skip("information results not present in this checkout")
    c = pd.read_csv(path)
    voi = c.groupby("unmet_access")["value_of_information"].mean()
    gap = float(voi.loc[min(voi.index)] - voi.loc[max(voi.index)])
    assert gap == pytest.approx(A.CLAIM_BY_KEY["I1"].original_effect, abs=0.002)
    net = c.groupby("hazard")["net_of_buying"].mean()
    assert float(net.iloc[0]) == pytest.approx(A.CLAIM_BY_KEY["I2"].original_effect, abs=0.002)


def test_the_learnability_original_matches_its_verdict_table():
    path = ROOT / "docs" / "learnability_results" / "verdicts.csv"
    if not path.exists():
        pytest.skip("learnability results not present in this checkout")
    v = pd.read_csv(path).set_index("id")
    assert str(v.loc["H1", "verdict"]).startswith("support")
    assert A.CLAIM_BY_KEY["H1"].original_verdict == "supported"
    assert A._leading_number(str(v.loc["H1", "result"])) == pytest.approx(
        A.CLAIM_BY_KEY["H1"].original_effect, abs=0.02)


def test_the_tournament_originals_match_the_stored_run():
    path = ROOT / "docs" / "tournament_results" / "main.json"
    if not path.exists():
        pytest.skip("tournament results not present in this checkout")
    verdicts = json.loads(path.read_text(encoding="utf-8"))["verdicts"]
    for key in ("PR1", "PR2", "PR3", "PR4", "PR5", "PR6"):
        assert A.CLAIM_BY_KEY[key].original_verdict == verdicts[key]


# ============================================================================== the decision rule
def test_the_five_outcomes_are_named_and_explained():
    assert len(A.OUTCOMES) == 5
    assert set(A.OUTCOME_KEYS) == {"retained", "narrowed", "rejected", "rejection_confirmed", "overturned"}
    for k, meaning in A.OUTCOMES:
        assert len(meaning) > 30


@pytest.mark.parametrize("before,after,oe,re_,expected", [
    ("supported", "supported", -0.95, -0.72, "retained"),
    ("supported", "supported", -0.95, -0.20, "narrowed"),
    ("supported", "not supported", 0.21, -0.19, "rejected"),
    ("not supported", "not supported", 4.0, 5.0, "rejection_confirmed"),
    ("not supported", "supported", None, None, "overturned"),
    ("supported", "supported", 0.21, -0.19, "rejected"),      # a sign flip is a rejection even if "supported"
    ("supported", "supported", None, None, "retained"),
])
def test_the_rule_is_mechanical(before, after, oe, re_, expected):
    assert A.decide(before, after, oe, re_)[0] == expected


def test_a_sign_flip_can_never_be_a_retention():
    for mag in (0.1, 1.0, 10.0):
        out, _ = A.decide("supported", "supported", mag, -mag)
        assert out == "rejected"


def test_the_retention_boundary_is_the_declared_fraction():
    assert A.decide("supported", "supported", 1.0, A.RETAIN_FRACTION + 0.01)[0] == "retained"
    assert A.decide("supported", "supported", 1.0, A.RETAIN_FRACTION - 0.01)[0] == "narrowed"


def test_overturned_says_a_third_run_is_needed():
    _, why = A.decide("not supported", "supported", None, None)
    assert "third run" in why


# ============================================================================== the results as stored
def test_the_stored_verdicts_follow_the_frozen_rule():
    """Re-derive every outcome from the rule. A stored outcome the rule does not produce would mean the table
    was edited by hand."""
    path = ROOT / "docs" / "adjudication_results" / "verdicts.csv"
    if not path.exists():
        pytest.skip("adjudication results not present in this checkout")
    v = pd.read_csv(path)
    assert len(v) == len(A.CLAIMS)
    for _, r in v.iterrows():
        oe = None if pd.isna(r["original_effect"]) else float(r["original_effect"])
        re_ = None if pd.isna(r["replication_effect"]) else float(r["replication_effect"])
        expected, _ = A.decide(r["original_verdict"], r["replication_verdict"], oe, re_)
        assert r["outcome"] == expected, (r["claim"], r["outcome"], expected)


def test_every_claim_was_adjudicated():
    path = ROOT / "docs" / "adjudication_results" / "verdicts.csv"
    if not path.exists():
        pytest.skip("adjudication results not present in this checkout")
    v = pd.read_csv(path)
    assert set(v["claim"]) == {c.key for c in A.CLAIMS}
    assert not v["replication_verdict"].isin(["not run"]).any()


def test_the_note_does_not_overclaim():
    doc = ROOT / "docs" / "adjudication.md"
    if not doc.exists():
        pytest.skip("note not present")
    text = " ".join(doc.read_text(encoding="utf-8").split())
    assert "survived sampling variation and nothing else" in text
    assert "post-hoc" in text, "the clustering diagnosis must be labelled as post-hoc"
    assert "not a vindication" in text, "the overturned claim must not be reported as support"


def test_the_module_states_what_a_single_replication_cannot_do():
    doc = " ".join(A.__doc__.split()).lower()
    assert "sampling variation and nothing else" in doc
    assert "never been run on people" in doc
