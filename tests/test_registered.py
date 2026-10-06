"""Reported findings without a frozen plan cannot go stale silently.

Each is tied to a fingerprint of the code that produced it. When that code changes, the finding must either be
rerun (and re-fingerprinted) or be listed as requiring replication, which the app and the README then show."""
import os

from heiner_abm import registered as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_finding_has_sources_and_a_fingerprint():
    assert set(R.FINDING_FINGERPRINTS) == set(R.FINDING_SOURCES)
    for key, (title, files) in R.FINDING_SOURCES.items():
        assert title and files, key
        for f in files:
            assert f == "#tuned" or os.path.exists(os.path.join(ROOT, "heiner_abm", f)), (key, f)


def test_changed_code_marks_findings_for_replication():
    stale = [k for k in R.FINDING_SOURCES if not R.finding_is_current(k) and k not in R.REPLICATION_REQUIRED]
    assert not stale, (
        f"The code behind {stale} changed. Rerun the finding and update FINDING_FINGERPRINTS (with a revision note), "
        "or add it to REPLICATION_REQUIRED in heiner_abm/registered.py and mark it in the README.")


def test_replication_entries_are_explained_and_shown_in_the_readme():
    assert set(R.REPLICATION_REQUIRED) <= set(R.FINDING_SOURCES)
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    for key, why in R.REPLICATION_REQUIRED.items():
        assert len(why) > 40, key
        assert R.replication_note(key) == why
        assert f"<!-- requires-replication:{key} -->" in readme, f"README does not flag '{key}' as requiring replication"



def test_a_listed_or_changed_finding_reports_a_note(monkeypatch):
    monkeypatch.setitem(R.REPLICATION_REQUIRED, "calibration", "Model changed on a test date; rerun required.")
    assert R.replication_note("calibration").startswith("Model changed")
    monkeypatch.delitem(R.REPLICATION_REQUIRED, "calibration")
    monkeypatch.setitem(R.FINDING_FINGERPRINTS, "calibration", "0" * 16)
    assert not R.finding_is_current("calibration")
    assert "has changed" in R.replication_note("calibration")
