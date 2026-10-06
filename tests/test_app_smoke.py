"""Headless smoke test: every page renders and every experiment button runs without exceptions."""
import os

import pytest
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
PAGES = ["app_pages/theory_overview.py", "app_pages/theory_heiner.py", "app_pages/theory_optimiser.py",
         "app_pages/theory_options.py", "app_pages/theory_cobweb.py", "app_pages/theory_heuristics.py",
         "app_pages/theory_satisficing.py", "app_pages/theory_rl.py", "app_pages/theory_imitation.py", "app_pages/market_lab.py", "app_pages/hypotheses.py", "app_pages/rc_validation.py",
         "app_pages/dynamic_rc.py", "app_pages/uncertainty.py",
         "app_pages/cd_gap.py", "app_pages/evolution.py", "app_pages/theories.py", "app_pages/arena.py", "app_pages/mechanisms.py", "app_pages/rule_choice.py", "app_pages/designer.py",
         "app_pages/field_patterns.py", "app_pages/calibration.py", "app_pages/empirical.py", "app_pages/play_market.py",
         "app_pages/experiment_analysis.py", "app_pages/generalisation.py", "app_pages/tracking.py",
         "app_pages/literature.py", "app_pages/model_docs.py", "app_pages/special_tests.py",
         "app_pages/agents_reference.py", "app_pages/information.py", "app_pages/reliability_gates.py",
         "app_pages/benchmarks.py"]


def _fast(at):
    at.session_state["cfg_reps"] = 3
    at.session_state["cfg_periods"] = 300
    at.session_state["arena_scale"] = "Quick check"   # the registered plans take a minute or more
    at.session_state["mech_scale"] = "Quick check"
    at.session_state["choice_scale"] = "Quick check"
    at.session_state["gen_scale"] = "Quick check"
    at.session_state["trk_scale"] = "Quick check"
    for k in ("cournot", "forecast", "nv", "time"):
        at.session_state[f"emp_{k}_src"] = "Synthetic demonstration"
    at.session_state["emp_cournot_groups"] = 2
    at.session_state["emp_cournot_gen"] = False
    at.session_state["emp_forecast_groups"] = 2
    at.session_state["emp_nv_subjects"] = 6
    at.session_state["pat_envs"] = 4
    at.session_state["pat_periods"] = 200
    at.session_state["exa_src"] = "Synthetic demonstration"
    at.session_state["exa_n"] = 6
    at.session_state["play_pid"] = "smoke-test"
    at.session_state["play_consent"] = True
    return at


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("preset", [{"cfg_horizon": 1}, {"cfg_rule": "Cournot", "cfg_horizon": 3},
                                    {"cfg_selection": "Adaptive", "cfg_horizon": 3, "focal_theory": "options"},
                                    {"cfg_struct_on": True, "cfg_rule": "Cournot", "cfg_horizon": 3,
                                     "focal_theory": "rl"},
                                    {"cfg_selection": "Adaptive", "cfg_horizon": 3, "cfg_info_delay": 2,
                                     "cfg_info_noise": 2.0, "cfg_info_announced": True},
                                    {"cfg_info_feedback": "chosen", "cfg_info_rivals": False, "cfg_horizon": 1},
                                    {"cfg_selection": "Adaptive", "cfg_gate": "explore", "cfg_info_feedback": "chosen",
                                     "cfg_adjust_cost": 15.0, "cfg_struct_on": True, "cfg_horizon": 2}])
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
