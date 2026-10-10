"""Provenance and auditability of workbench runs.

What a stored run has to be able to answer: which code produced it, in what software, on which seeds and for what
purpose, what each agent knew when it decided, and which recorded fields no agent ever had. These tests pin those
answers, and the two properties that make a cache trustworthy: a relevant code change invalidates it, and an
unchanged run reproduces.
"""
import json
import os
import zipfile

import pandas as pd
import pytest

from heiner_abm.workbench import execution, provenance as P, store
from heiner_abm.workbench.environments import ENVIRONMENTS
from heiner_abm.workbench.spec import Design, ExperimentSpec, Question
from heiner_abm.workbench.validate import validate


def _spec(environment="inventory", **design):
    env = ENVIRONMENTS[environment]
    d = dict(replications=2, periods=80, burn_in=10, seed=1, n_boot=50)
    d.update(design)
    policies = [p.key for p in env.policies][:2]
    return ExperimentSpec(name="provenance test", mode="research", environment=environment,
                          candidate=env.candidates[0].key, policies=tuple(policies),
                          question=Question(text="Does the gate pay?", primary_outcome=env.primary,
                                            comparison=(policies[0], policies[1])),
                          design=Design(**d))


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("HEINER_LAB_DIR", str(tmp_path))
    return tmp_path


# ============================================================================== 1. the code fingerprint
def test_the_fingerprint_covers_the_agent_code_and_the_workbench_analysis():
    """The old fingerprint hashed a hand-written module list that left these out, so a change to an agent or to the
    analysis did not invalidate a stored run."""
    covered = set(P._closure(P.FINGERPRINT_ROOTS))
    for name in ("heiner_abm.agents", "heiner_abm.arena", "heiner_abm.analysis", "heiner_abm.gates",
                 "heiner_abm.engine", "heiner_abm.learnability", "heiner_abm.nk", "heiner_abm.tasks",
                 "heiner_abm.information", "heiner_abm.params",
                 "heiner_abm.workbench.analysis", "heiner_abm.workbench.environments",
                 "heiner_abm.workbench.execution", "heiner_abm.workbench.provenance"):
        assert name in covered, f"{name} is not covered by the code fingerprint"


def test_the_closure_follows_imports_written_inside_functions():
    """The adapters import their engines inside run_cell, in the form `from .. import learnability as L`. A
    fingerprint built by walking imported module objects, or one that ignored relative imports, would miss them."""
    import ast
    import heiner_abm.workbench.environments as E
    tree = ast.parse(open(E.__file__, "rb").read())
    named = set()
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for n in ast.walk(fn):
            if isinstance(n, ast.ImportFrom):
                named |= {a.name for a in n.names} | ({n.module} if n.module else set())
    assert "learnability" in named, f"expected a function-level engine import, found {sorted(named)}"
    assert "heiner_abm.learnability" in set(P._closure(("heiner_abm.workbench.environments",)))


@pytest.mark.parametrize("module", ["heiner_abm.agents", "heiner_abm.workbench.analysis", "heiner_abm.nk"])
def test_changing_any_covered_module_changes_the_fingerprint(module, monkeypatch):
    before = P.code_fingerprint()
    real = P._hash_source

    def altered(path):
        h = real(path)
        return ("f" * 64) if path.endswith(module.split(".")[-1] + ".py") else h

    P.code_fingerprint.cache_clear()
    monkeypatch.setattr(P, "_hash_source", altered)
    after = P.code_fingerprint()
    monkeypatch.undo()
    P.code_fingerprint.cache_clear()
    assert after != before, f"a change to {module} left the fingerprint unchanged"
    assert P.code_fingerprint() == before, "the fingerprint did not return to its real value"


def test_the_fingerprint_is_stable_when_nothing_changes():
    P.code_fingerprint.cache_clear()
    a = P.code_fingerprint()
    P.code_fingerprint.cache_clear()
    assert P.code_fingerprint() == a


def test_execution_reports_the_same_fingerprint():
    assert execution.code_version() == P.code_fingerprint()


# ============================================================================== 2. cache invalidation and reuse
def test_a_code_change_invalidates_a_cached_run(lab, monkeypatch):
    spec = _spec()
    out = execution.execute(spec, "research")
    store.save_run(out)
    assert store.cached(spec, "research") is not None, "the run should be reused while the code is unchanged"

    real = P._hash_source
    P.code_fingerprint.cache_clear()
    monkeypatch.setattr(P, "_hash_source", lambda path: "0" * 64 if path.endswith("agents.py") else real(path))
    try:
        assert execution.run_key(spec, "research") != out.key, "the run key ignored a change to the agent code"
        assert store.cached(spec, "research") is None, "a stale run was reused after the agent code changed"
    finally:
        P.code_fingerprint.cache_clear()


def test_an_unchanged_run_reproduces_exactly(lab):
    spec = _spec()
    first = execution.execute(spec, "research")
    second = execution.execute(spec, "research")
    assert first.key == second.key
    # compute_ms is wall-clock time; everything a result depends on must be identical
    drop = ["compute_ms"]
    pd.testing.assert_frame_equal(first.trials.drop(columns=drop, errors="ignore"),
                                  second.trials.drop(columns=drop, errors="ignore"))
    assert first.manifest["seeds"] == second.manifest["seeds"]
    assert first.manifest["costs"] == second.manifest["costs"]


def test_a_specification_change_changes_the_run_key(lab):
    a, b = _spec(), _spec(periods=81)
    assert execution.run_key(a, "research") != execution.run_key(b, "research")


# ============================================================================== 3. software environment
def test_the_software_environment_records_dependency_versions():
    env = P.software_environment()
    assert env["python"] and env["platform"]
    for name in ("numpy", "pandas"):
        assert env["packages"][name], f"no version recorded for {name}"
    assert set(P.DEPENDENCIES) <= set(env["packages"])


def test_the_software_environment_holds_no_absolute_paths():
    """A stored run may be published; an interpreter path carries the user's account name."""
    blob = json.dumps(P.software_environment())
    assert ":\\" not in blob and "/home/" not in blob and "/Users/" not in blob


# ============================================================================== 4. seed namespaces
def test_the_four_namespaces_never_overlap():
    base = 7
    drawn = {ns: set(P.seeds(base, ns, P.SEED_STRIDE)) for ns in P.SEED_NAMESPACES}
    names = sorted(drawn)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not drawn[a] & drawn[b], f"{a} and {b} share seeds"


def test_the_legacy_offsets_are_preserved_so_existing_runs_reproduce():
    """Training and evaluation keep the offsets the workbench has always used."""
    spec = _spec(replications=3, train_replications=3, tuning="grid")
    assert execution.train_seeds(spec) == [spec.design.seed * 1000 + 100_000_000 + r for r in range(3)]
    assert execution.test_seeds(spec) == [spec.design.seed * 1000 + 300_000_000 + r for r in range(3)]


def test_overlapping_requests_are_refused_rather_than_silently_wrapped():
    assert P.disjoint(1, {"evaluation": 10}) is None
    assert P.disjoint(1, {"evaluation": P.SEED_STRIDE + 1}) is not None
    assert P.disjoint(P.SEED_BLOCK, {"evaluation": 1}) is not None
    with pytest.raises(ValueError):
        P.seeds(1, "not_a_namespace", 3)


def test_a_design_whose_namespaces_would_collide_does_not_run(lab):
    spec = _spec(replications=P.SEED_STRIDE + 1)
    assert any(i.level == "error" and "overlap" in i.message for i in validate(spec))
    with pytest.raises(ValueError):
        execution.execute(spec, "research")


def test_every_namespace_states_its_purpose():
    assert set(P.SEED_PURPOSE) == set(P.SEED_NAMESPACES)
    assert all(P.SEED_PURPOSE[n].strip() for n in P.SEED_NAMESPACES)


def test_pilot_and_validation_passes_are_kept_apart_from_the_reported_result(lab):
    spec = _spec(replications=2, pilot_replications=2, validation_replications=2)
    out = execution.execute(spec, "research")
    assert set(out.trials["stage"]) == {"evaluation"}
    assert set(out.pilot["stage"]) == {"pilot"} and set(out.validation["stage"]) == {"validation"}
    assert not set(out.pilot["seed"]) & set(out.trials["seed"])
    assert not set(out.validation["seed"]) & set(out.trials["seed"])
    plan = out.manifest["seeds"]["namespaces"]
    assert plan["pilot"]["count"] == 2 and plan["validation"]["count"] == 2


# ============================================================================== 5. trace retention
@pytest.mark.parametrize("retention,expected", [("summary", 0), ("illustrative", 1), ("all", 3)])
def test_retention_controls_how_many_replications_are_kept(lab, retention, expected):
    spec = _spec(replications=3, trace_retention=retention)
    out = execution.execute(spec, "research")
    if expected == 0:
        assert not out.traces
        return
    df = next(iter(next(iter(out.traces.values())).values()))
    assert df["replication"].nunique() == expected


def test_retention_is_recorded_and_round_trips_through_the_store(lab):
    spec = _spec(replications=3, trace_retention="all")
    out = execution.execute(spec, "research")
    store.save_run(out)
    back = store.load_run(out.key)
    assert back.provenance["trace_retention"] == "all"
    for level, pols in out.traces.items():
        for pol, df in pols.items():
            pd.testing.assert_frame_equal(df.reset_index(drop=True), back.traces[level][pol].reset_index(drop=True),
                                          check_dtype=False)


def test_large_logs_are_stored_column_wise(lab):
    spec = _spec(replications=3, trace_retention="all")
    out = execution.execute(spec, "research")
    d = store.save_run(out)
    written = [f for f in store.TRACE_FILES if os.path.exists(os.path.join(d, f))]
    assert written, "no decision-log file was written"
    assert written[0] in ("traces.parquet", "traces.pkl.gz"), f"uncompressed storage used: {written}"


def test_an_unknown_retention_is_refused():
    with pytest.raises(ValueError):
        P.keeps_trace("everything", 0)


# ============================================================================== 6. decision logs
@pytest.mark.parametrize("environment", sorted(ENVIRONMENTS))
def test_every_environment_writes_a_classified_decision_log(lab, environment):
    spec = _spec(environment, replications=2, periods=60)
    out = execution.execute(spec, "research")
    assert out.traces, f"{environment} produced no decision log"
    for pols in out.traces.values():
        for pol, df in pols.items():
            problems = P.check_decision_log(df)
            assert not problems, f"{environment}/{pol}: {problems}"
            roles = {P.role_of(c) for c in df.columns}
            assert "predecision" in roles and "action" in roles, f"{environment}/{pol} roles: {roles}"


@pytest.mark.parametrize("environment", sorted(ENVIRONMENTS))
def test_feedback_timing_is_recorded_and_ordered(lab, environment):
    """An outcome cannot reach the agent before it exists, and -1 means it never reached the agent at all."""
    spec = _spec(environment, replications=2, periods=60)
    out = execution.execute(spec, "research")
    for pols in out.traces.values():
        for pol, df in pols.items():
            if "feedback" not in df:
                continue
            rel = df["feedback_release_period"].astype(float)
            mat = df["feedback_matured_period"].astype(float)
            released = rel >= 0
            assert (rel[released] >= mat[released]).all(), f"{environment}/{pol}: feedback released before it matured"
            assert set(df.loc[~released, "feedback_origin"]) <= {"none"}, \
                f"{environment}/{pol}: an unreleased feedback value claims an origin"


def test_researcher_only_fields_are_named_and_never_mixed_in(lab):
    spec = _spec("inventory", replications=2, periods=60)
    out = execution.execute(spec, "research")
    df = next(iter(next(iter(out.traces.values())).values()))
    hidden = P.researcher_only_columns(df.columns)
    assert "researcher_demand" in hidden and "researcher_mean" in hidden
    assert all(P.role_of(c) == "researcher_only" for c in hidden)
    assert set(hidden) == set(out.manifest["researcher_only_fields"])
    assert not set(hidden) & set(out.manifest["decision_log"]["roles"]["predecision"])


def test_the_market_hides_the_counterfactual_profits_from_the_firm(lab):
    """profit_rule and profit_default are researcher-only by heiner_abm.information.RESEARCHER_ONLY: a firm sees the
    gain the engine releases to it, never both branches."""
    from heiner_abm.information import RESEARCHER_ONLY
    spec = _spec("market", replications=2, periods=60)
    out = execution.execute(spec, "research")
    df = next(iter(next(iter(out.traces.values())).values()))
    for name in ("profit_rule", "profit_default"):
        assert name in RESEARCHER_ONLY
        assert P.RESEARCHER_PREFIX + name in df.columns
        assert name not in df.columns, "a researcher-only quantity is recorded as if the agent could see it"


def test_an_unclassified_column_is_reported():
    df = pd.DataFrame({"period": [0], "mystery": [1.0]})
    assert any("mystery" in p for p in P.check_decision_log(df))


def test_feedback_without_its_timing_is_reported():
    df = pd.DataFrame({"period": [0], "feedback": [1.0]})
    problems = P.check_decision_log(df)
    assert any("feedback_origin" in p for p in problems)
    assert any("feedback_release_period" in p for p in problems)


def test_an_undeclared_feedback_origin_is_reported():
    df = pd.DataFrame({"period": [0], "feedback": [1.0], "feedback_origin": ["rumour"],
                       "feedback_matured_period": [0], "feedback_release_period": [0]})
    assert any("rumour" in p for p in P.check_decision_log(df))


# ============================================================================== 7. the manifest
def test_the_manifest_answers_every_required_question(lab):
    spec = _spec(replications=2, pilot_replications=2, validation_replications=2)
    out = execution.execute(spec, "research")
    man = out.manifest
    for section in ("configuration", "policies", "tuning", "costs", "information", "code", "software", "seeds",
                    "decision_log", "researcher_only_fields", "run"):
        assert section in man, f"the manifest has no {section}"
    assert man["run"]["status"] == "complete"
    assert man["code"]["fingerprint"] == P.code_fingerprint()
    assert man["code"]["n_modules"] >= 15
    assert man["software"]["packages"]["numpy"]
    assert [p["key"] for p in man["policies"]] == list(spec.policies)
    assert man["decision_log"]["retention"] == spec.design.trace_retention


@pytest.mark.parametrize("environment", sorted(ENVIRONMENTS))
def test_every_environment_states_its_cost_definitions(lab, environment):
    """Separately, one entry per friction: a reader must see which charges applied and at what rate."""
    env = ENVIRONMENTS[environment]
    defs = env.cost_definitions(_spec(environment), env.defaults())
    assert defs, f"{environment} states no cost definitions"
    for d in defs:
        assert {"name", "parameter", "rate", "charged_when", "in_force"} <= set(d)
        assert d["charged_when"].strip()


def test_the_inventory_manifest_separates_the_three_charges(lab):
    spec = _spec("inventory", replications=2, periods=60)
    out = execution.execute(spec, "research")
    names = [c["name"] for c in out.manifest["costs"]["baseline"]]
    assert names == ["Default-departure overhead", "Fixed switching cost", "Magnitude cost"]


def test_a_zero_cost_control_is_visible_in_the_manifest(lab):
    spec = _spec("inventory", replications=2, periods=60)
    spec.env_params.update(cost=0.0, switch_cost=0.0, magnitude_cost=0.0)
    out = execution.execute(spec, "research")
    assert not any(c["in_force"] for c in out.manifest["costs"]["baseline"])


def test_the_manifest_is_stored_and_reloaded(lab):
    spec = _spec()
    out = execution.execute(spec, "research")
    d = store.save_run(out)
    assert os.path.exists(os.path.join(d, "manifest.json"))
    assert store.load_run(out.key).manifest == json.loads(json.dumps(out.manifest, default=str, sort_keys=True))


# ============================================================================== 8. the export bundle
def test_the_export_carries_the_manifest_and_names_the_hidden_fields(lab):
    spec = _spec(replications=2, pilot_replications=2, validation_replications=2)
    out = execution.execute(spec, "research")
    store.save_run(out)
    z = zipfile.ZipFile(__import__("io").BytesIO(store.export_zip(out.key)))
    names = set(z.namelist())
    for fn in ("manifest.json", "spec.json", "provenance.json", "trials.csv", "pilot.csv", "validation.csv",
               "README.txt"):
        assert fn in names, f"{fn} missing from the bundle"
    readme = z.read("README.txt").decode()
    assert P.code_fingerprint() in readme
    assert "researcher_" in readme and "never released" not in readme.split("Seed namespaces")[0]
    for ns in P.SEED_NAMESPACES:
        assert ns in readme
    for field in out.manifest["researcher_only_fields"]:
        assert field in readme
