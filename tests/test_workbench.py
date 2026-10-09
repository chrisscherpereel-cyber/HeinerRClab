"""Experiment workbench (heiner_abm.workbench): one specification, validity checks, reproducible paired execution,
identical interface and command-line runs, the run store, cancellation, background runs and the workspace pages."""
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
import io

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("HEINER_LAB_DIR", tempfile.mkdtemp(prefix="heiner_lab_test_"))

from heiner_abm import tasks as TK                                             # noqa: E402
from heiner_abm.workbench import analysis, execution, presets, store            # noqa: E402
from heiner_abm.workbench.environments import ENVIRONMENTS, task_trace         # noqa: E402
from heiner_abm.workbench.spec import ExperimentSpec, Treatment, changed_fields  # noqa: E402
from heiner_abm.workbench.validate import validate                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_PRESETS = [k for k, p in presets.PRESETS.items() if p.spec is not None]


def _small(spec, reps=3, periods=200):
    spec.design.replications, spec.design.periods, spec.design.burn_in = reps, periods, 20
    return spec


def errors(spec):
    return [i for i in validate(spec) if i.level == "error"]


def test_spec_round_trip_and_digest():
    s = presets.load("complexity")
    t = ExperimentSpec.from_json(s.to_json())
    assert t.to_dict() == s.to_dict() and t.digest() == s.digest()
    t.name, t.mode = "other name", "research"
    assert t.digest() == s.digest()                          # name and interface mode do not change results
    t.design.replications += 1
    assert t.digest() != s.digest() and changed_fields(s, t) == ["design.replications"]


@pytest.mark.parametrize("key", SIM_PRESETS)
def test_presets_are_valid_and_name_their_comparison(key):
    s = presets.load(key)
    assert errors(s) == []
    env = ENVIRONMENTS[s.environment]
    a, b = s.question.comparison
    assert a in s.policies and b in s.policies and env.outcome(s.question.primary_outcome)
    assert env.policy(a).kind != "benchmark" and env.policy(b).kind != "benchmark"


def test_human_question_is_not_simulated():
    p = presets.PRESETS["decision_aid"]
    assert p.spec is None and "human" in p.note.lower()


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_blank_specs_need_only_a_question(env):
    s = presets.blank(env)
    assert any(i.step == "question" for i in errors(s))
    s.question.text = "q"
    assert errors(s) == []


def test_validity_checks():
    s = presets.load("learnability")
    s.question.comparison = ("oracle", "always")
    assert any("benchmark" in i.message for i in errors(s))
    s = presets.load("learnability")
    s.policies = ("always",)
    assert any(i.step == "agents" for i in errors(s))
    s = presets.load("learnability")
    s.design.replications = 1
    assert any(i.step == "design" for i in errors(s))
    s = presets.load("learnability")
    s.design.treatment = Treatment("hazard", (0.01, 0.5))
    assert any("outside" in i.message for i in errors(s))
    s = presets.load("complexity")
    s.design.treatment = Treatment("K_NK", (0, 9))
    assert any("K_NK" in i.message for i in errors(s))
    s = presets.load("restriction")
    s.candidate, s.env_params["demand_knowledge"] = "Cournot", "none"
    assert any("demand model" in i.message for i in errors(s))
    s = presets.load("restriction")
    s.mode, s.design.replications = "research", 3
    assert not errors(s) and any(i.level == "warning" and i.step == "design" for i in validate(s))
    with pytest.raises(ValueError):
        execution.execute(presets.load("learnability").__class__())       # an empty spec never runs


def test_seed_blocks_are_disjoint_and_preview_is_reduced():
    s = presets.load("learnability")
    s.design.tuning, s.design.train_replications = "grid", 5
    assert not set(execution.train_seeds(s)) & set(execution.test_seeds(s))
    p = execution.effective_spec(s, "preview")
    assert p.design.replications <= 3 and p.design.periods <= 300
    assert execution.run_key(s, "preview") != execution.run_key(s, "research")


@pytest.mark.parametrize("key", SIM_PRESETS)
def test_execution_is_reproducible_paired_and_traced(key):
    s = _small(presets.load(key))
    a, b = execution.execute(s), execution.execute(s)
    pd.testing.assert_frame_equal(a.trials.drop(columns="compute_ms"), b.trials.drop(columns="compute_ms"))
    t = a.trials
    levels = max(1, len(s.design.treatment.levels))
    assert len(t) == len(s.policies) * s.design.replications * levels
    # every policy in a condition saw the same environments (paired by replication)
    for (lvl, rep), g in t.groupby(["level", "replication"]):
        assert g["seed"].nunique() == 1 and set(g["policy"]) == set(s.policies)
    assert set(a.traces) == set(t["level"])
    for pols in a.traces.values():
        for pol, tr in pols.items():
            assert len(tr) == s.design.periods
            assert {"acted", "reason", "feedback_available"} <= set(tr.columns)
    eff = analysis.overview(a)["effects"]
    assert len(eff) == levels and (eff["lo"] <= eff["effect"] + 1e-12).all()


@pytest.mark.parametrize("task", ["learning", "investment"])
def test_task_trace_reproduces_the_engine_layer(task):
    env = TK.TaskEnv(task, 1.5, 0.02, 1.0, 0.3, 11)
    dec = TK._decisions(env, 600, 11)

    class P:
        oracle_periods = 4000

    table = TK.oracle_table(env, P())
    for layer in TK.LAYERS:
        tr = task_trace(dec, layer, 1.0, table, 0.995)
        out = TK._layer(dec, layer, 50, band=1.0, table=table, rc_memory=0.995)
        assert tr["acted"].to_numpy()[50:].mean() == pytest.approx(out["deviation_rate"])
        pay = np.where(tr["acted"], dec["pay_x"], dec["pay_d"])[50:].mean()
        assert pay == pytest.approx(out["payoff"])


def test_store_cache_export_and_saved_experiments():
    s = _small(presets.load("noise_simple"))
    out = execution.execute(s, "preview")
    store.save_run(out)
    hit = store.cached(s, "preview")
    assert hit is not None and hit.key == out.key
    pd.testing.assert_frame_equal(hit.trials.drop(columns="level_value"), out.trials.drop(columns="level_value"),
                                  check_dtype=False)
    s2 = s.copy()
    s2.design.seed += 1
    assert store.cached(s2, "preview") is None                       # a different configuration is never reused
    z = zipfile.ZipFile(io.BytesIO(store.export_zip(out.key)))
    names = set(z.namelist())
    assert {"spec.json", "provenance.json", "trials.csv", "README.txt"} <= names
    assert any(n.startswith("traces/") for n in names)
    prov = json.loads(z.read("provenance.json"))
    assert prov["code_version"] == execution.code_version() and prov["status"] == "complete"
    s.name = "Saved test experiment"
    store.save_experiment(s)
    assert store.load_experiment("saved-test-experiment").to_dict() == s.to_dict()
    dup = store.duplicate_experiment("saved-test-experiment", "Saved copy")
    assert dup.digest() == s.digest() and any(e["name"] == "Saved copy" for e in store.list_experiments())
    store.delete_experiment("saved-copy")
    assert not any(e["name"] == "Saved copy" for e in store.list_experiments())


def test_command_line_runs_the_same_experiment(tmp_path):
    s = _small(presets.load("complexity"), reps=2, periods=120)
    f = tmp_path / "spec.json"
    f.write_text(s.to_json())
    env = {**os.environ, "HEINER_LAB_DIR": str(tmp_path / "lab")}
    r = subprocess.run([sys.executable, "-m", "heiner_abm.workbench", "run", str(f), "--kind", "research"], cwd=ROOT,
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    key = json.loads(r.stdout.strip().splitlines()[-1])["key"]
    cli = pd.read_csv(os.path.join(str(tmp_path / "lab"), "runs", key, "trials.csv"))
    here = execution.execute(s, "research")
    assert key == here.key
    cols = [c for c in cli.columns if c not in ("compute_ms", "level_value")]
    pd.testing.assert_frame_equal(cli[cols].reset_index(drop=True), here.trials[cols].reset_index(drop=True),
                                  check_dtype=False)


def test_cancellation_stops_between_blocks():
    s = presets.load("complexity")
    calls = []
    out = execution.execute(s, cancel=lambda: calls.append(1) or len(calls) > 2)
    assert out.provenance["status"] == "cancelled" and not out.complete
    assert len(out.trials) < len(s.policies) * s.design.replications * len(s.design.treatment.levels)


def test_background_run_completes_outside_the_page():
    s = _small(presets.load("noise_simple"), reps=2, periods=150)
    s.design.seed = 4242
    key = execution.start_background(s, "research")
    # A wall-clock budget, not a fixed poll count: the suite runs its tests in parallel worker processes, and a
    # starved subprocess can take much longer than the run itself needs. 240 polls (120 s) was tight under any load.
    deadline, p = time.time() + 600, None
    while time.time() < deadline:
        p = store.progress(key)
        if p and p["status"] != "running":
            break
        time.sleep(0.5)
    assert p is not None, "the background run never reported any progress"
    assert p["status"] == "complete", (p["status"], open(os.path.join(store.run_dir(key), "worker.log")).read())
    assert store.cached(s, "research").key == key


def test_workspace_steps_and_results_pages_run():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=600)
    for mode in ("explore", "research"):
        at.session_state["wb_mode"] = mode
        at.run()
        at.switch_page("app_pages/start.py").run()
        next(b for b in at.button if b.key == "start_use_noise_simple").click().run()
        assert not at.exception, [e.value for e in at.exception]
        at.switch_page("app_pages/experiment.py").run()       # AppTest does not keep an app-initiated page switch
        for step in ("question", "environment", "agents", "information", "design", "run"):
            at.session_state["wb_step"] = step
            at.run()
            assert not at.exception, (mode, step, [e.value for e in at.exception])
        next(b for b in at.button if b.key == "wb_preview").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert at.session_state["wb_run_key"]
        at.switch_page("app_pages/results.py").run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("Comparison" in m.value for m in at.markdown)


def test_choosing_a_question_loads_it_at_once():
    """Regression: picking a guided question must load its preset without a separate button, and 'Use this question'
    on the Start page must open the workspace with that question."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=600)
    at.run()
    next(b for b in at.button if b.key == "start_use_complexity").click().run()
    assert not at.exception, [e.value for e in at.exception]
    spec = at.session_state["wb_spec"]
    assert spec.question.preset == "complexity" and spec.environment == "nk"
    assert any(t.value == presets.PRESETS["complexity"].title for t in at.text_area)   # now on the Experiment page
    at.switch_page("app_pages/experiment.py").run()             # AppTest does not keep an app-initiated page switch
    pick = next(s for s in at.selectbox if s.label == "Guided questions")
    pick.set_value("noise_simple").run()
    assert not at.exception, [e.value for e in at.exception]
    spec = at.session_state["wb_spec"]
    assert spec.question.preset == "noise_simple" and spec.environment == "inventory"
    assert spec.question.comparison == ("band", "always")
    assert any(t.value == presets.PRESETS["noise_simple"].title for t in at.text_area)
    pick = next(s for s in at.selectbox if s.label == "Guided questions")
    pick.set_value("custom").run()                              # custom keeps the current settings
    assert at.session_state["wb_spec"].environment == "inventory"


def test_inventory_workbench_forwards_the_gate_hyperparameters():
    """Defect (repaired 8 October 2026): the workbench built its hyperparameters without a "gate" entry, so the
    confidence-sensitive gate silently fell back to learnability.GATE_HP0 and neither the control nor a tuned value
    reached gated() through policy_orders()."""
    import numpy as np
    from heiner_abm import learnability as L
    from heiner_abm.workbench.environments import Inventory

    env = Inventory()
    params = {c.key: c.default for c in env.controls}
    for cand in env.candidates:
        params.update({c.key: c.default for c in cand.params})
    assert {"gate_conf", "gate_nmin"} <= set(params), "the gate's parameters are not exposed as controls"

    assert env._hp(params, {})["gate"] == {"conf": params["gate_conf"], "nmin": params["gate_nmin"]}
    custom = {**params, "gate_conf": 0.75, "gate_nmin": 11.0}
    assert env._hp(custom, {})["gate"] == {"conf": 0.75, "nmin": 11.0}          # the control reaches the gate
    assert env._hp(custom, {"gate_conf": 0.95, "gate_nmin": 3.0})["gate"] == {"conf": 0.95, "nmin": 3.0}  # tuned wins

    # and the value actually changes the policy's behavior end to end
    cfg = env._cfg(params)
    p = L.paths(cfg, 300, 5)
    c = L.candidate_orders(cfg, p, params["gain"])
    loose = L.policy_orders("gate_lcb", cfg, p, c, env._hp({**params, "gate_conf": 0.6, "gate_nmin": 2.0}, {}))[0]
    strict = L.policy_orders("gate_lcb", cfg, p, c, env._hp({**params, "gate_conf": 0.99, "gate_nmin": 50.0}, {}))[0]
    assert (loose != c["S_D"]).sum() > (strict != c["S_D"]).sum()


def test_inventory_workbench_tunes_the_gate_with_the_same_budget():
    """The gate must get as many candidate settings as the forecast gain and the band, not zero."""
    import inspect
    from heiner_abm.workbench.environments import Inventory
    src = inspect.getsource(Inventory.tune)
    assert "gate_lcb" in src and "gate_grid" in src
    grid = [ln for ln in src.splitlines() if "gate_grid = " in ln][0]
    assert grid.count("(") - 1 == 4, grid                      # four candidate pairs, as for gain and band_b
    out = Inventory.tune.__doc__ or ""
    assert isinstance(Inventory().tuning_note, str) and "four candidates" in Inventory().tuning_note


def test_every_preset_asks_for_outcomes_its_environment_reports():
    """A preset naming an outcome the environment does not report fails validation, and the page then renders no
    preview button, which surfaces only as a StopIteration far away. Renaming the inventory outcomes on
    9 October 2026 broke two presets exactly this way."""
    from heiner_abm.workbench import presets
    from heiner_abm.workbench.environments import ENVIRONMENTS

    checked = 0
    for key, preset in presets.PRESETS.items():
        if preset.spec is None:
            continue
        spec = presets.load(key)
        have = {o.key for o in ENVIRONMENTS[spec.environment].outcomes}
        missing = [o for o in spec.design.outcomes if o not in have]
        assert not missing, f"preset {key!r} ({spec.environment}) asks for {missing}, which it does not report"
        primary = spec.question.primary_outcome
        assert primary in have, f"preset {key!r} primary outcome {primary!r} is not reported"
        checked += 1
    assert checked >= 4


def test_inventory_cost_controls_reach_the_config_and_are_exported():
    """The three charges must be selectable in the workbench and must reach the economics, including a zero-cost
    control. They are separate frictions, so each must move the right component and only that one."""
    from heiner_abm import learnability as L
    from heiner_abm.workbench.environments import Inventory

    env = Inventory()
    keys = {c.key for c in env.controls}
    assert {"cost", "switch_cost", "magnitude_cost", "cost_model"} <= keys
    reported = {o.key for o in env.outcomes}
    assert {"gross_payoff", "cost_departure", "cost_switching", "cost_magnitude", "cost_total",
            "departure_rate", "adjustment_rate", "adjustment_magnitude"} <= reported

    params = {c.key: c.default for c in env.controls}
    for cand in env.candidates:
        params.update({c.key: c.default for c in cand.params})
    base = env._cfg({**params, "cost": 0.0, "switch_cost": 0.0, "magnitude_cost": 0.0})
    assert base.zero_cost
    assert env._cfg({**params, "switch_cost": 4.0}).switch_cost == 4.0
    assert env._cfg({**params, "magnitude_cost": 0.3}).magnitude_cost == 0.3
    assert env._cfg({**params, "cost_model": "legacy_departure_only"}).cost_model == "legacy_departure_only"

    S = np.array([10.0, 10.0, 25.0]); S_D = np.array([10.0, 20.0, 20.0])
    only_switch = L.cost_components(env._cfg({**params, "cost": 0.0, "switch_cost": 2.0, "magnitude_cost": 0.0}), S, S_D)
    only_mag = L.cost_components(env._cfg({**params, "cost": 0.0, "switch_cost": 0.0, "magnitude_cost": 1.0}), S, S_D)
    assert only_switch["switching"].sum() > 0 and only_switch["magnitude"].sum() == 0
    assert only_mag["magnitude"].sum() > 0 and only_mag["switching"].sum() == 0
