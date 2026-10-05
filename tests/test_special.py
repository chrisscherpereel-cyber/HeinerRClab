"""Signature tests by theory, the sidebar layout and the neutral presentation of hypotheses."""
import os

import pytest
from streamlit.testing.v1 import AppTest

from heiner_abm.special import SPECIAL_TESTS, run_special

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.mark.parametrize("key", [t.key for t in SPECIAL_TESTS])
def test_signature_tests_run(key):
    res = run_special(key, "Quick")
    assert res["verdict"] in {"supported", "not supported"}
    assert len(res["table"]) and res["result"]


def test_cobweb_boundary_matches_theory():
    res = run_special("cobweb", "Quick")
    d = res["table"].set_index("firms")
    assert abs(d.loc[4, "simulated_boundary"] - 0.8) <= 0.1


def test_sidebar_puts_theory_selector_under_theories_and_collapses_groups():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    md = [m.value for m in at.sidebar.markdown]
    assert md.index("### Theories") < md.index("### Agents") < md.index("### Theory under test") < \
        md.index("### Base scenario")
    assert all(not e.proto.expanded for e in at.sidebar.expander)       # overview page: every group collapsed
    labels = [e.label for e in at.sidebar.expander]
    from heiner_abm.arena import THEORY_NAMES
    assert all(name in labels for name in THEORY_NAMES.values())       # every theory's agents are described
    assert labels.index("General simulations") < labels.index("Special simulations")


def test_hypothesis_titles_are_neutral_questions():
    import heiner_abm.literature as lit
    for h in lit.HYPOTHESES:
        assert h.title.endswith("?"), h.hid
