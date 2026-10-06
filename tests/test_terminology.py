"""Terms, kinds of uncertainty and kinds of evidence are defined once and used consistently."""
import glob
import os
import re

from heiner_abm.terminology import (EVIDENCE_BY_RESULT, EVIDENCE_LABELS, KNIGHT_NOTE, RULE_LABELS, UNCERTAINTY_TYPES,
                                    rule_option)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
PAGES = {p: open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(ROOT, "app_pages", "*.py"))}


def test_rule_labels_keep_config_values_and_say_what_the_rule_is():
    assert set(RULE_LABELS) == {"Bertrand", "Cournot"}
    assert "margin-feedback" in RULE_LABELS["Bertrand"].lower() and "Bertrand" in rule_option("Bertrand")


def test_five_kinds_of_uncertainty_and_ambiguity_is_not_claimed():
    kinds = {k for k, *_ in UNCERTAINTY_TYPES}
    assert kinds == {"risk", "parameter", "ambiguity", "misspecification", "change"}
    ambiguity = next(t for t in UNCERTAINTY_TYPES if t[0] == "ambiguity")
    assert ambiguity[3].startswith("Not modeled")
    assert "not unknowable" in KNIGHT_NOTE


def test_evidence_labels_are_defined_and_used_on_every_results_page():
    assert set(EVIDENCE_LABELS) == {"analytical", "simulation", "behavioral", "causal"}
    assert {k for _, k, _ in EVIDENCE_BY_RESULT} <= set(EVIDENCE_LABELS)
    for path, src in PAGES.items():
        for call in re.findall(r"evidence_note\(([^)]*)\)", src):
            for key in re.findall(r'"(\w+)"', call):
                assert key in EVIDENCE_LABELS, (path, key)


def test_every_readme_results_section_states_its_evidence_type():
    sections = re.split(r"\n## ", README)
    for sec in sections:
        title = sec.split("\n", 1)[0]
        if any(w in title for w in ("results", "Signature tests", "Validation against data")):
            assert "Evidence type" in sec, title


def test_no_wholesale_rejection_language():
    banned = ("hypothesis rejected", "is rejected", "signature fails", "principle is sound")
    for name, text in [("README.md", README), *PAGES.items()]:
        low = text.lower()
        for phrase in banned:
            assert phrase not in low, (name, phrase)


def test_frozen_plans_are_distinguished_from_external_preregistration():
    assert "Externally timestamped prospective" in README or "externally timestamped prospective" in README
    assert "no record of external preregistration" in README
