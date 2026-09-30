"""Literature registry integrity: every citation resolves, every hypothesis has research on both sides, and the app
text cites research only through the registry (no stray references to other project documents)."""
import pathlib
import re

import pytest

from heiner_abm.literature import (CONTRIBUTIONS, HYPOTHESES, REFERENCES, THEORY_SOURCES, bibliography, cite,
                                   cited_keys)
from heiner_abm.theories import EXPERIMENTS, THEORIES

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_every_cited_key_exists_and_every_reference_is_cited():
    assert set(cited_keys()) == set(REFERENCES)


@pytest.mark.parametrize("h", HYPOTHESES, ids=lambda h: h.hid)
def test_hypothesis_has_research_on_both_sides(h):
    assert len(h.support) >= 2 and len(h.alternative) >= 2
    assert h.rc_prediction and h.alt_prediction and h.alt_label and h.contribution
    assert all(note.strip() for _, note in h.support + h.alternative)


def test_theories_and_experiments_link_to_the_registry():
    assert {t.key for t in THEORIES} == set(THEORY_SOURCES)
    ids = {h.hid for h in HYPOTHESES}
    assert all(e.hid in ids for e in EXPERIMENTS)
    assert all(keys for _, _, keys in CONTRIBUTIONS)


def test_reference_formatting():
    for r in REFERENCES.values():
        assert str(r.year) in r.apa and r.title in r.apa
        assert r.bibtex.startswith("@") and r.key in r.bibtex
    assert cite("dixit1994") == "Dixit & Pindyck (1994)"
    assert cite("aghion2005") == "Aghion et al. (2005)"
    assert [r.key for r in bibliography(["kreps1979", "arrow1951"])] == ["arrow1951", "kreps1979"]


def test_app_text_has_no_stray_document_references():
    banned = re.compile(r"the paper|paper's|\bVBA\b|workbook", re.I)
    files = [ROOT / "app.py", ROOT / "README.md"] + sorted((ROOT / "app_pages").glob("*.py")) + \
        sorted((ROOT / "ui").glob("*.py")) + sorted((ROOT / "heiner_abm").glob("*.py"))
    hits = [f"{f.name}:{i}" for f in files for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
            if banned.search(line.replace('yref="paper"', ""))]
    assert not hits, hits
