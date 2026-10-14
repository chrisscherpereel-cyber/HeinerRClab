"""The pre-run disclosure and the integration audit.

The disclosure exists so that the decisions determining what a result MEANS are visible before the run, not
reconstructed afterwards. The audit exists because unit tests check that functions work, not that the parts are
wired to each other. Both are only worth having if they fail when they should, so most of what follows breaks
something on purpose and checks that it is noticed.
"""
import numpy as np
import pandas as pd
import pytest

from heiner_abm import integration_audit as A
from heiner_abm.workbench import disclosure as D, presets
from heiner_abm.workbench.environments import ENVIRONMENTS
from heiner_abm.workbench.spec import STEPS, STEP_LABELS, Issue, issue_step, step_status


# ============================================================================== the seven stages
def test_the_stages_are_the_seven_the_protocol_names():
    assert STEPS == ("question", "environment", "agents", "design", "run", "results", "export")
    for s in STEPS:
        assert s in STEP_LABELS and STEP_LABELS[s][0].isdigit()
    assert [STEP_LABELS[s][0] for s in STEPS] == list("1234567")


def test_a_legacy_issue_key_still_finds_a_stage():
    """`information` is no longer a stage of its own. Environment code still raises issues against it, and
    those must appear somewhere rather than vanishing or crashing."""
    assert issue_step("information") == "environment"
    st = step_status([Issue("information", "error", "x")])
    assert st["environment"] == "invalid"


def test_an_issue_against_an_unknown_stage_is_not_lost():
    st = step_status([Issue("nowhere", "error", "x")])
    assert "invalid" in st.values(), "an issue against an unknown stage disappeared"


# ============================================================================== disclosure
SPEC_PRESETS = [k for k, v in presets.PRESETS.items() if v.spec is not None]


def test_all_seven_items_are_produced_for_every_preset():
    assert SPEC_PRESETS, "no preset carries a specification"
    for key in SPEC_PRESETS:
        items = D.disclosure(presets.load(key, mode="research"))
        assert [i.key for i in items] == list(D.ITEM_KEYS)
        for i in items:
            assert i.level in D.LEVELS and i.summary.strip()


def test_an_unchosen_outcome_is_flagged_rather_than_silently_defaulted():
    """The environment has a default, so a run still happens. A reader seeing the result will assume the
    outcome was selected for the question; it was not, and the disclosure says so."""
    spec = presets.load("restriction", mode="research")
    spec.question.primary_outcome = ""
    item = next(i for i in D.disclosure(spec) if i.key == "estimand")
    assert item.level == "caution"
    assert "no outcome was chosen" in item.summary.lower()
    assert any("NOT chosen" in d for d in item.detail)


def test_no_outcome_at_all_blocks_the_run():
    spec = presets.load("restriction", mode="research")
    spec.question.primary_outcome = ""
    spec.environment = "does-not-exist"
    item = next(i for i in D.disclosure(spec) if i.key == "estimand")
    assert item.level == "blocking"
    assert D.blocking(D.disclosure(spec))


def test_overlapping_seed_blocks_block_the_run():
    spec = presets.load("restriction", mode="research")
    spec.design.tuning = "grid"
    spec.design.train_replications = 0
    item = next(i for i in D.disclosure(spec) if i.key == "separation")
    assert item.level == "blocking", item
    assert "tuning" in item.summary.lower() or "training" in item.summary.lower()


def test_an_untuned_run_does_not_claim_a_training_block_it_never_uses():
    spec = presets.load("restriction", mode="research")
    spec.design.tuning = "none"
    spec.design.train_replications = 5
    item = next(i for i in D.disclosure(spec) if i.key == "separation")
    assert item.level == "ok"
    assert not any(d.startswith("training:") for d in item.detail)
    assert any("unused" in d for d in item.detail), item.detail


def test_a_privileged_benchmark_is_surfaced_not_hidden():
    spec = presets.load("learnability", mode="research")
    item = next(i for i in D.disclosure(spec) if i.key == "information")
    assert item.level == "caution"
    assert "oracle" in item.summary


def test_a_free_adaptation_run_says_so():
    spec = presets.load("restriction", mode="research")
    env = ENVIRONMENTS[spec.environment]
    for c in env.controls:
        if "cost" in c.key and c.kind in ("float", "int"):
            spec.env_params[c.key] = 0
    item = next(i for i in D.disclosure(spec) if i.key == "costs")
    assert item.level == "caution"
    assert "free" in item.summary


def test_the_cost_item_names_every_charge_and_its_rate():
    spec = presets.load("learnability", mode="research")
    item = next(i for i in D.disclosure(spec) if i.key == "costs")
    assert len(item.detail) >= 2
    assert any("charged" in d for d in item.detail)
    assert any("when" in d for d in item.detail), "a charge is listed without saying when it applies"


def test_calibration_limits_are_always_shown():
    """These are never 'ok': a reader who is told nothing about calibration will assume the default meaning."""
    for key in SPEC_PRESETS:
        item = next(i for i in D.disclosure(presets.load(key, mode="research")) if i.key == "calibration")
        assert item.level == "caution" and item.detail
    joined = " ".join(n for _, n in D.CALIBRATION_NOTES)
    assert "45-62%" in joined, "the measured gate coverage is no longer quoted"
    assert "0.50-0.75" in joined, "the measured false-positive rate is no longer quoted"


def test_the_disclosure_is_read_from_the_specification_not_invented():
    a = presets.load("restriction", mode="research")
    b = a.copy()
    b.design.replications = 3
    ia = next(i for i in D.disclosure(a) if i.key == "calibration")
    ib = next(i for i in D.disclosure(b) if i.key == "calibration")
    assert len(ib.detail) > len(ia.detail), "shrinking the design did not change what the reader is told"


def test_summary_line_reports_blocking_first():
    spec = presets.load("restriction", mode="research")
    spec.question.primary_outcome = ""
    spec.environment = "does-not-exist"
    assert "resolved before" in D.summary_line(D.disclosure(spec))


# ============================================================================== statuses
def test_the_four_named_statuses_exist_and_are_explained():
    for key in ("exploratory", "confirmatory", "unsupported", "requires_replication"):
        s = D.status(key)
        assert len(s.meaning) > 60 and s.may_claim and s.may_not_claim
        assert s.may_claim != s.may_not_claim
    with pytest.raises(KeyError):
        D.status("probably_fine")


def test_the_weakest_status_comes_first():
    assert D.STATUS_ORDER[0] == "exploratory", "the default offered should be the one that claims least"


def test_every_status_says_what_it_may_not_claim():
    for row in D.status_table():
        assert row["may_not_claim"].strip()


# ============================================================================== audit: reachability
def test_the_reachability_probe_respects_environment_validation():
    """NK declares interactions up to 15 but requires K <= N - 1, which is 7 at the default N. The environment
    blocks that, so a probe of 15 is the audit being wrong, not the wiring. Caught this way once."""
    env = ENVIRONMENTS["nk"]
    spec = presets.blank("nk", mode="explore")
    chosen = A._first_valid(env, spec, "K_NK", A._probe_values(env.control("K_NK")))
    assert chosen is not None and 0 <= int(chosen) <= int(env.params(spec)["N"]) - 1


def test_a_disconnected_control_is_reported_as_a_failure():
    """The audit must fail when a control reaches nothing. Simulated by probing a control the environment
    ignores, which is exactly the defect shape it exists to find."""
    base = {("p", "x"): 1.0}
    assert A._moved(base, {("p", "x"): 1.0}) == set()
    assert A._moved(base, {("p", "x"): 2.0}) == {"x"}


@pytest.mark.parametrize("env_key", ["inventory"])
def test_every_control_of_an_environment_reaches_something(env_key):
    findings = A.reachability([env_key], periods=90, burn_in=15, seeds=(1, 2))
    fails = [f for f in findings if f.status == "fail"]
    assert not fails, [f"{f.item}: {f.detail}" for f in fails]
    assert len(findings) >= 10


# ============================================================================== audit: parity
def test_every_policy_declares_what_it_reads():
    fails = [f for f in A.parity() if f.status == "fail"]
    assert not fails, [f"{f.item}: {f.detail}" for f in fails]


def test_the_charges_do_not_depend_on_who_is_compared():
    cost = [f for f in A.parity() if f.item.endswith(".costs")]
    assert cost and all(f.status == "pass" for f in cost), [f.detail for f in cost if f.status != "pass"]


def test_privileged_benchmarks_are_marked_rather_than_hidden():
    info = [f for f in A.parity() if f.status == "info"]
    assert info, "no environment declares a privileged benchmark, which would be surprising"
    assert all("benchmark" in f.detail for f in info)


# ============================================================================== audit: provenance
def test_provenance_checks_pass():
    fails = [f for f in A.provenance_checks() if f.status == "fail"]
    assert not fails, [f"{f.item}: {f.detail}" for f in fails]


def test_a_changed_specification_cannot_reuse_a_cached_run():
    from heiner_abm.workbench.execution import run_key
    spec = presets.load("restriction", mode="explore")
    other = spec.copy()
    other.design.periods += 1
    assert run_key(spec, "research") != run_key(other, "research")


# ============================================================================== audit: outstanding
def test_outstanding_lists_every_study_and_its_command():
    found = A.outstanding()
    assert len(found) == len(A.STUDIES)
    for f in found:
        assert f.status in ("pass", "warn", "info")
    keys = {s["key"] for s in A.STUDIES}
    assert {"tournament", "learnability", "stress", "human_learning", "human_study"} <= keys
    human = next(s for s in A.STUDIES if s["key"] == "human_study")
    assert "participants" in human["command"], "the study needing people must say so"


def test_a_study_with_no_results_is_flagged_not_assumed_done(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "STUDIES", (dict(key="ghost", label="Never run", results="docs/nothing",
                                            command="python tools/nothing.py", scale="?"),))
    f = A.outstanding()[0]
    assert f.status == "warn" and "NOT EXECUTED" in f.detail


# ============================================================================== audit: shape
def test_the_audit_summarises_into_a_verdict_per_check():
    findings = pd.DataFrame([
        A.Finding("parity", "a", "pass", "").to_dict(),
        A.Finding("parity", "b", "fail", "").to_dict(),
        A.Finding("tests", "c", "pass", "").to_dict(),
    ])
    s = A.summarise(findings).set_index("check")
    assert s.loc["parity", "verdict"] == "FAIL"
    assert s.loc["tests", "verdict"] == "pass"


def test_the_six_checks_are_named_and_described():
    assert len(A.CHECKS) == 6
    for key, desc in A.CHECKS:
        assert key and len(desc) > 20


def test_measured_resources_are_excluded_from_reproducibility():
    """compute_ms is wall-clock time: comparing it would report a failure on every machine."""
    assert "compute_ms" in A.MEASURED_RESOURCES


def test_the_audit_does_not_claim_more_than_it_checks():
    doc = " ".join(A.__doc__.split())
    assert "Nothing here is a scientific result" in doc
    assert "precondition" in doc
