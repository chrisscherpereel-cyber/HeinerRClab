"""An integration audit: whether the parts of this laboratory are actually wired to each other.

Unit tests check that a function does what it says. They do not check that the number a user typed into a
control reaches the component it names, that two policies being compared are charged the same way, or that a
cached result is thrown away when the code behind it changes. Those are integration properties, they fail
silently, and in this repository they have failed before:

    a tuning step scored the same configuration four times, because the hyperparameter key the search varied
    was not a key the environment read - every candidate scored identically and the search reported a gain of
    +0.000 while looking like it had worked

    a sampling-history manipulation moved nothing, because the rows carrying the manipulation were dropped
    before the model saw them

    a seed derived from a string hash varied between processes, so a frozen plan reproduced nothing

Each of those is the same shape: a configured input that never arrives. The audit below looks for that shape
directly rather than waiting for it to show up as a strange result.

    1  REACHABILITY   does changing a control change the outcome?
    2  PARITY         do the policies being compared face the same costs and declared information?
    3  PROVENANCE     does a code change invalidate a cached result, and does an unchanged run reproduce?
    4  TESTS          the engine and interface suites, actually run
    5  END TO END     small complete experiments, run twice, compared
    6  OUTSTANDING    which full studies have not been executed

Nothing here is a scientific result. It is a statement about whether the software is wired the way the
documentation says, which is a precondition for a scientific result and not a substitute for one.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CHECKS: Tuple[Tuple[str, str], ...] = (
    ("reachability", "Configuration parameters reach the intended simulation components"),
    ("parity", "Policies face the same cost definitions and declared information"),
    ("provenance", "Caching and provenance: code changes invalidate, unchanged runs reproduce"),
    ("tests", "Engine and interface tests"),
    ("end_to_end", "Small reproducible end-to-end experiments"),
    ("outstanding", "Full studies that still require execution"),
)


@dataclass
class Finding:
    check: str
    item: str
    status: str                      # "pass" | "fail" | "warn" | "info"
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.check, "item": self.item, "status": self.status, "detail": self.detail,
                **{f"evidence_{k}": v for k, v in self.evidence.items()}}


# ================================================================================================ 1. reachability
def _probe_values(control) -> List[Any]:
    """Candidate values clearly different from the default, furthest first.

    More than one, because a control may declare a range wider than what is valid at the other controls'
    defaults - NK interactions go to 15 but require K <= N - 1, which is 7 at the default N. The environment
    validates that and blocks the run, so the audit must offer a value the environment would accept rather than
    reporting its own invalid probe as a defect. That mistake was made first and is what this list prevents.
    """
    d = control.default
    if control.kind == "bool":
        return [not bool(d)]
    if control.kind == "choice":
        return [o for o in control.options if o != d]
    if control.kind in ("float", "int"):
        lo, hi = control.lo, control.hi
        if lo is None or hi is None:
            return [d * 2 + 1] if isinstance(d, (int, float)) else []
        out = []
        for frac in (1.0, 0.6, 0.3):
            for end in (hi, lo):
                v = d + frac * (end - d)
                v = int(round(v)) if control.kind == "int" else float(v)
                if v != d and v not in out:
                    out.append(v)
        return out
    return []


def reachability(env_keys: Optional[Sequence[str]] = None, periods: int = 120, burn_in: int = 20,
                 seeds: Sequence[int] = (1, 2), outcome: Optional[str] = None,
                 progress: Optional[Callable[[float, str], None]] = None) -> List[Finding]:
    """Change one control at a time and see whether any reported outcome moves.

    A control that moves nothing is not automatically a defect - it may legitimately not apply to the selected
    candidate or policy set, which is what `Environment.not_applicable` declares. What the audit objects to is
    an UNDECLARED control that moves nothing, because that is indistinguishable from a wire that was never
    connected.

    Every outcome is compared, not just the primary one: a cost parameter that changes the cost column but not
    the net payoff is reaching its component.
    """
    from .workbench import presets
    from .workbench.environments import ENVIRONMENTS

    note = progress or (lambda f, m: None)
    out: List[Finding] = []
    keys = list(env_keys or ENVIRONMENTS)
    total = max(1, sum(len([c for c in ENVIRONMENTS[k].controls if c.treatable]) for k in keys))
    done = 0

    for ek in keys:
        env = ENVIRONMENTS[ek]
        spec = presets.blank(ek, mode="explore")
        spec.design.periods, spec.design.burn_in, spec.design.seed = periods, burn_in, 1
        base_params = env.params(spec)
        try:
            base_rows, _ = env.run_cell(spec, base_params, list(seeds), {}, lambda: False, set(), 0)
        except Exception as exc:
            out.append(Finding("reachability", f"{ek}", "fail",
                               f"the baseline cell did not run: {type(exc).__name__}: {exc}"))
            continue
        base = _summarise(base_rows)

        for c in env.controls:
            if not c.treatable:
                continue
            done += 1
            note(done / total, f"{ek}.{c.key}")
            candidates = _probe_values(c)
            probe = _first_valid(env, spec, c.key, candidates)
            if probe is None:
                out.append(Finding("reachability", f"{ek}.{c.key}", "info",
                                   "no alternative value inside the declared range is valid at the other "
                                   "controls' defaults" if candidates else
                                   "no alternative value could be constructed for this control"))
                continue
            params = {**base_params, c.key: probe}
            try:
                rows, _ = env.run_cell(spec, params, list(seeds), {}, lambda: False, set(), 0)
            except Exception as exc:
                out.append(Finding("reachability", f"{ek}.{c.key}", "fail",
                                   f"setting {c.key}={probe!r} passed validation but raised "
                                   f"{type(exc).__name__}: {exc}", {"value": probe}))
                continue
            moved = _moved(base, _summarise(rows))
            declared = c.key in getattr(env, "not_applicable", {})
            if moved:
                out.append(Finding("reachability", f"{ek}.{c.key}", "pass",
                                   f"moved {len(moved)} outcome(s): {', '.join(sorted(moved)[:4])}",
                                   {"value": probe, "n_moved": len(moved)}))
            elif declared:
                out.append(Finding("reachability", f"{ek}.{c.key}", "info",
                                   f"no effect, and declared not applicable: "
                                   f"{env.not_applicable[c.key]}", {"value": probe}))
            else:
                out.append(Finding("reachability", f"{ek}.{c.key}", "fail",
                                   f"setting {c.key} from {c.default!r} to {probe!r} changed no reported "
                                   f"outcome, and the environment does not declare it inapplicable",
                                   {"value": probe}))
    return out


def _first_valid(env, spec, key: str, candidates: Sequence[Any]) -> Optional[Any]:
    """The first candidate the environment would accept. A control the environment rejects at these defaults
    is guarded, not unwired, and the audit must not confuse the two."""
    for v in candidates:
        probe_spec = spec.copy()
        probe_spec.env_params = {**dict(spec.env_params), key: v}
        try:
            issues = env.validate(probe_spec)
        except Exception:
            continue
        if not any(level == "error" for _, level, _ in issues):
            return v
    return None


def _summarise(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for r in rows:
        pol = str(r.get("policy", ""))
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                out[(pol, k)] = out.get((pol, k), 0.0) + float(v)
    return out


def _moved(a: Dict[Tuple[str, str], float], b: Dict[Tuple[str, str], float], tol: float = 1e-9) -> set:
    keys = set(a) | set(b)
    return {k[1] for k in keys
            if abs(a.get(k, float("nan")) - b.get(k, float("nan"))) > tol
            or (k in a) != (k in b)}


# ================================================================================================ 2. parity
PRIVILEGED_KINDS = ("benchmark",)

# Columns that record how long something took, not what it produced. Identical code gives different values.
MEASURED_RESOURCES = frozenset({"compute_ms", "seconds", "wall_time", "elapsed"})


def parity(env_keys: Optional[Sequence[str]] = None) -> List[Finding]:
    """Every policy compared in an environment must face the same charges and have its information declared.

    Cost parity is a property of the environment here: charges are applied by the environment, not by the
    policy, so the check is that no policy is exempted and that the charge set does not depend on which policy
    is selected. Information parity is the weaker but more easily broken property: that every policy declares
    what it reads, and that anything privileged is marked as a benchmark rather than competing.
    """
    from .workbench import presets
    from .workbench.environments import ENVIRONMENTS

    out: List[Finding] = []
    for ek in list(env_keys or ENVIRONMENTS):
        env = ENVIRONMENTS[ek]
        spec = presets.blank(ek, mode="explore")

        for p in env.policies:
            if not p.uses.strip():
                out.append(Finding("parity", f"{ek}.{p.key}", "fail",
                                   "the policy does not declare what information it reads"))
            elif p.kind in PRIVILEGED_KINDS:
                out.append(Finding("parity", f"{ek}.{p.key}", "info",
                                   f"privileged benchmark, reported beside the ranking: reads {p.uses}"))
            else:
                out.append(Finding("parity", f"{ek}.{p.key}", "pass", f"declares: {p.uses}"))

        # the charge set must not depend on which policies are being compared
        try:
            allp = tuple(p.key for p in env.policies)
            a = _cost_signature(env, _with_policies(spec, allp))
            b = _cost_signature(env, _with_policies(spec, allp[:1]))
        except Exception as exc:
            out.append(Finding("parity", f"{ek}.costs", "warn",
                               f"cost definitions could not be compared: {type(exc).__name__}: {exc}"))
            continue
        if a == b:
            out.append(Finding("parity", f"{ek}.costs", "pass",
                               f"the same {len(a)} charge(s) apply whichever policies are compared"))
        else:
            out.append(Finding("parity", f"{ek}.costs", "fail",
                               "the charges in force depend on which policies are selected",
                               {"all_policies": str(a), "one_policy": str(b)}))
    return out


def _with_policies(spec, policies):
    s = spec.copy()
    s.policies = tuple(policies)
    return s


def _cost_signature(env, spec) -> Tuple[Tuple[str, float], ...]:
    defs = env.cost_definitions(spec, env.params(spec))
    return tuple(sorted((str(d.get("name", d.get("parameter", "?"))), float(d.get("rate", 0.0)))
                        for d in defs if d.get("in_force", True)))


# ================================================================================================ 3. provenance
def provenance_checks() -> List[Finding]:
    """A cached result must die when the code behind it changes, and an unchanged run must reproduce."""
    from .workbench import presets, provenance
    from .workbench.execution import code_version, run_key

    out: List[Finding] = []
    spec = presets.load("restriction", mode="explore")

    k1 = run_key(spec, "research")
    k2 = run_key(spec, "research")
    out.append(Finding("provenance", "run_key.stable", "pass" if k1 == k2 else "fail",
                       "the run key is stable for an unchanged specification" if k1 == k2
                       else f"the run key changed between two calls: {k1} vs {k2}"))

    moved = spec.copy()
    moved.design.periods += 1
    out.append(Finding("provenance", "run_key.sensitive",
                       "pass" if run_key(moved, "research") != k1 else "fail",
                       "changing the specification changes the run key" if run_key(moved, "research") != k1
                       else "a changed specification produced the same run key, so a cache would be reused"))

    name_only = spec.copy()
    name_only.name = "a different name"
    out.append(Finding("provenance", "run_key.content_only",
                       "pass" if run_key(name_only, "research") == k1 else "warn",
                       "renaming an experiment does not invalidate its results"
                       if run_key(name_only, "research") == k1 else
                       "renaming an experiment changed the run key, so results are recomputed needlessly"))

    fp = provenance.code_fingerprint()
    detail = provenance.code_fingerprint_detail()
    n = len(detail.get("modules", detail.get("files", [])) or [])
    out.append(Finding("provenance", "code_fingerprint", "pass" if fp and n else "fail",
                       f"fingerprint {fp} over {n} module(s)", {"modules": n}))
    out.append(Finding("provenance", "code_version", "pass" if code_version() else "warn",
                       f"code version recorded as {code_version()!r}"))

    env = provenance.software_environment()
    pkgs = env.get("packages", {}) or {}
    missing = ([k for k in ("python", "platform") if not env.get(k)]
               + [k for k in ("numpy", "pandas", "scipy", "streamlit") if not pkgs.get(k)])
    out.append(Finding("provenance", "software_environment", "pass" if not missing else "warn",
                       f"python {env.get('python')} on {env.get('platform')}, {len(pkgs)} package versions "
                       f"recorded" if not missing else f"missing: {missing}",
                       {"packages": len(pkgs)}))

    clash = provenance.disjoint(1, {"training": 3, "evaluation": 8, "pilot": 2, "validation": 2})
    out.append(Finding("provenance", "seed_namespaces", "pass" if clash is None else "fail",
                       "the four seed blocks are disjoint at a representative size" if clash is None
                       else f"overlap: {clash}"))
    return out


# ================================================================================================ 4. tests
DEFAULT_TEST_TARGETS: Tuple[Tuple[str, str], ...] = (
    ("engine", "tests/test_arena.py tests/test_scenarios.py tests/test_information.py"),
    ("workbench", "tests/test_provenance.py tests/test_equivalence.py"),
    ("interface", "tests/test_app_smoke.py tests/test_theory_pages.py"),
    ("registered", "tests/test_registered.py"),
)


def run_tests(targets: Sequence[Tuple[str, str]] = DEFAULT_TEST_TARGETS,
              progress: Optional[Callable[[float, str], None]] = None) -> List[Finding]:
    """Actually run the suites and report what happened, including when they fail."""
    note = progress or (lambda f, m: None)
    out: List[Finding] = []
    for i, (name, paths) in enumerate(targets):
        note(i / max(len(targets), 1), f"tests: {name}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", *paths.split()],
                              cwd=str(ROOT), capture_output=True, text=True)
        tail = (proc.stdout or "").strip().splitlines()
        line = tail[-1] if tail else "(no output)"
        out.append(Finding("tests", name, "pass" if proc.returncode == 0 else "fail", line,
                           {"seconds": round(time.time() - t0, 1), "returncode": proc.returncode}))
    return out


# ================================================================================================ 5. end to end
def end_to_end(preset_keys: Sequence[str] = ("restriction",),
               progress: Optional[Callable[[float, str], None]] = None) -> List[Finding]:
    """Run a small complete experiment twice and compare the numbers, not just the exit status."""
    from .workbench import presets
    from .workbench.execution import execute
    from .workbench.validate import preview_spec

    note = progress or (lambda f, m: None)
    out: List[Finding] = []
    for i, key in enumerate(preset_keys):
        note(i / max(len(preset_keys), 1), f"end to end: {key}")
        spec = preview_spec(presets.load(key, mode="explore"))
        try:
            a = execute(spec, kind="preview")
            b = execute(spec, kind="preview")
        except Exception as exc:
            out.append(Finding("end_to_end", key, "fail",
                               f"the experiment did not complete: {type(exc).__name__}: {exc}"))
            continue
        fa, fb = a.trials, b.trials
        if fa is None or fb is None or len(fa) == 0:
            out.append(Finding("end_to_end", key, "fail", "the run produced no results frame"))
            continue
        # Measured resources are not results. compute_ms is wall-clock time and differs between two runs of
        # identical code by tens of milliseconds; comparing it would report a reproducibility failure on every
        # machine. It is excluded by name, and the exclusion is reported rather than silent.
        num = [c for c in fa.columns if pd.api.types.is_numeric_dtype(fa[c]) and c not in MEASURED_RESOURCES]
        skipped = sorted(set(fa.columns) & MEASURED_RESOURCES)
        same_shape = fa.shape == fb.shape
        differing = [c for c in num
                     if not np.allclose(fa[c].to_numpy(float), fb[c].to_numpy(float), equal_nan=True)]
        identical = bool(same_shape and not differing)
        out.append(Finding("end_to_end", key, "pass" if identical else "fail",
                           f"{fa.shape[0]} rows x {len(num)} substantive columns reproduced exactly on a "
                           f"second run (excluding measured resources: {skipped})" if identical
                           else f"two runs disagreed on {differing}",
                           {"rows": int(fa.shape[0]), "columns": len(num), "excluded": str(skipped)}))
        man = getattr(a, "manifest", None) or {}
        have = sorted(set(man))
        wanted = {"code", "seeds", "costs", "information", "status", "spec", "configuration", "policies"}
        got = sorted(wanted & set(man))
        out.append(Finding("end_to_end", f"{key}.manifest", "pass" if len(got) >= 4 else "warn",
                           f"manifest carries {len(have)} sections, including {got}" if man
                           else "no manifest was attached", {"sections": len(have)}))
        prov = getattr(a, "provenance", {}) or {}
        out.append(Finding("end_to_end", f"{key}.status",
                           "pass" if prov.get("status") == "complete" else "warn",
                           f"run status {prov.get('status')!r}"))
    return out


# ================================================================================================ 6. outstanding
# Full studies in this repository, and what it takes to run each. A study is "executed" when results exist in
# the repository; several are deliberately not run here because they need hours, and one needs people.
STUDIES: Tuple[Dict[str, Any], ...] = (
    dict(key="tournament", label="Registered agent tournament", results="docs/tournament_results",
         command="python tools/rerun_tournament_findings.py", scale="hours"),
    dict(key="learnability", label="Learnability study", results="docs/learnability_results",
         command="python tools/rerun_learnability_study.py", scale="~16 min"),
    dict(key="learnability_validity", label="Incremental construct validity",
         results="docs/learnability_validity_results", command="(module run_validity)", scale="~20 min"),
    dict(key="gate_validation", label="Reliability-gate validation", results="docs/gate_validation_results",
         command="python tools/run_gate_validation.py", scale="~25 min"),
    dict(key="information_env", label="Censored demand and information acquisition",
         results="docs/information_results", command="python tools/run_information_experiment.py",
         scale="~12 s"),
    dict(key="stress", label="Registered stress-test suite", results="docs/stress_results",
         command="python tools/run_stress_suite.py", scale="~35 min"),
    dict(key="human_learning", label="Trial-by-trial human models (synthetic)",
         results="docs/human_learning_results", command="python tools/run_human_learning.py",
         scale="~83 min"),
    dict(key="human_study", label="The human experiment itself", results="",
         command="(requires participants; protocol only)", scale="not executable here"),
)


def outstanding() -> List[Finding]:
    out: List[Finding] = []
    for s in STUDIES:
        path = ROOT / s["results"] if s["results"] else None
        files = sorted(p.name for p in path.glob("*")) if path and path.exists() else []
        if not s["results"]:
            out.append(Finding("outstanding", s["key"], "info",
                               f"{s['label']}: {s['command']}", {"scale": s["scale"]}))
        elif files:
            out.append(Finding("outstanding", s["key"], "pass",
                               f"{s['label']}: {len(files)} result file(s) present",
                               {"path": s["results"], "scale": s["scale"]}))
        else:
            out.append(Finding("outstanding", s["key"], "warn",
                               f"{s['label']}: NOT EXECUTED - run `{s['command']}` ({s['scale']})",
                               {"path": s["results"], "scale": s["scale"]}))
    return out


# ================================================================================================ the audit
def run_audit(checks: Sequence[str] = tuple(k for k, _ in CHECKS), quick: bool = False,
              progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, pd.DataFrame]:
    note = progress or (lambda f, m: None)
    found: List[Finding] = []
    envs = ("inventory",) if quick else None
    presets_to_run = ("restriction",)
    order = [c for c in checks]
    for i, c in enumerate(order):
        lo, hi = i / len(order), (i + 1) / len(order)
        step = lambda f, m, lo=lo, hi=hi: note(lo + (hi - lo) * f, m)
        if c == "reachability":
            found += reachability(envs, progress=step)
        elif c == "parity":
            found += parity(envs)
        elif c == "provenance":
            found += provenance_checks()
        elif c == "tests":
            found += run_tests(DEFAULT_TEST_TARGETS[:2] if quick else DEFAULT_TEST_TARGETS, progress=step)
        elif c == "end_to_end":
            found += end_to_end(presets_to_run, progress=step)
        elif c == "outstanding":
            found += outstanding()
        else:
            raise ValueError(f"unknown check {c!r}")
    note(1.0, "done")
    df = pd.DataFrame([f.to_dict() for f in found])
    return {"findings": df, "summary": summarise(df), "checks": pd.DataFrame(
        [{"check": k, "description": d} for k, d in CHECKS])}


def summarise(findings: pd.DataFrame) -> pd.DataFrame:
    if findings.empty:
        return pd.DataFrame(columns=["check", "pass", "fail", "warn", "info", "verdict"])
    g = findings.pivot_table(index="check", columns="status", values="item", aggfunc="count").fillna(0)
    for col in ("pass", "fail", "warn", "info"):
        if col not in g:
            g[col] = 0
    g = g[["pass", "fail", "warn", "info"]].astype(int).reset_index()
    g["verdict"] = np.where(g["fail"] > 0, "FAIL", np.where(g["warn"] > 0, "attention", "pass"))
    return g


EXPORTS = {
    "findings": "every audit finding: check, item, status and detail",
    "summary": "pass/fail/warn/info counts and a verdict per check",
    "checks": "what each of the six checks is for",
}
