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


def test_sidebar_has_five_destinations_mode_and_theory_selector():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    labels = [m.proto.label for m in at.sidebar.get("page_link")]
    assert labels[:5] == ["Start", "Experiment", "Results", "Validation", "Reference"]
    assert any(r.label == "Mode" for r in at.sidebar.radio)
    assert any(s.key == "focal_theory" for s in at.sidebar.selectbox)              # theory under test still offered
    exp = {e.proto.label: e.proto.expanded for e in at.sidebar.expander}
    assert exp["More in Start"] and not exp["All pages"]                          # current section open, rest folded
    md = [m.value for m in at.sidebar.markdown]
    assert "### Base market scenario" in md
    assert md.index("### Agents") < md.index("### Base market scenario")
    from heiner_abm.arena import THEORY_NAMES
    assert all(name in exp and not exp[name] for name in THEORY_NAMES.values())   # every theory's agents, folded
    assert "**General simulations**" in md and "**Special simulations**" in md   # simulations split by fairness


def test_every_grouped_simulation_page_exists():
    import pathlib
    from ui.common import SIM_GROUPS
    app = pathlib.Path(APP).read_text(encoding="utf-8")
    assert all(f'"app_pages/{name}.py"' in app for name in SIM_GROUPS)


def test_hypothesis_titles_are_neutral_questions():
    import heiner_abm.literature as lit
    for h in lit.HYPOTHESES:
        assert h.title.endswith("?"), h.hid
