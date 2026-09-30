"""Headless smoke test: every page renders and every experiment button runs without exceptions."""
import os

import pytest
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
PAGES = ["app_pages/theory.py", "app_pages/market_lab.py", "app_pages/hypotheses.py", "app_pages/rc_validation.py",
         "app_pages/dynamic_rc.py", "app_pages/uncertainty.py",
         "app_pages/cd_gap.py", "app_pages/evolution.py", "app_pages/theories.py", "app_pages/designer.py",
         "app_pages/literature.py", "app_pages/model_docs.py"]


def _fast(at):
    at.session_state["cfg_reps"] = 3
    at.session_state["cfg_periods"] = 300
    return at


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("preset", [{"cfg_horizon": 1}, {"cfg_rule": "Cournot", "cfg_horizon": 3},
                                    {"cfg_selection": "Adaptive", "cfg_horizon": 3},
                                    {"cfg_struct_on": True, "cfg_rule": "Cournot", "cfg_horizon": 3}])
def test_page_runs(page, preset):
    at = _fast(AppTest.from_file(APP, default_timeout=600))
    for k, v in preset.items():
        at.session_state[k] = v
    at.run()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    # press every run/submit button once
    for i in range(len(at.button)):
        b = at.button[i]
        if b.label in ("Apply preset", "Reset"):
            continue
        b.click().run()
        assert not at.exception, (b.label, [e.value for e in at.exception])
    for form_btn in at.get("form_submit_button") if hasattr(at, "get") else []:
        form_btn.click().run()
        assert not at.exception, [e.value for e in at.exception]
