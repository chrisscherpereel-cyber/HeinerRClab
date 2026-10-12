"""Execution service: seeds, tuning, scheduling, progress, cancellation and provenance for one experiment.

The same function runs an experiment from the interface (preview in-process; research runs in a separate process, see
`start_background`) and from the command line (python -m heiner_abm.workbench run spec.json).

Random streams come from the four disjoint namespaces in heiner_abm.workbench.provenance: `training` for tuning,
`pilot` for pilot estimates, `evaluation` for the reported results and `validation` for confirming a result on fresh
environments. Nothing estimated on one namespace is reported from another, and every policy in a cell sees the same
evaluation seeds (common random numbers), so policy differences are paired. Disjointness is checked before a run
starts rather than assumed.

A completed run carries a manifest (`manifest.json` in the run directory): the configuration, the policies compared,
the tuning choices and the budget behind them, the cost definitions in force, the information assumptions, the code
fingerprint and software environment, the seed plan, the decision-log schema with its researcher-only fields, and the
run status. The manifest is what makes a stored result auditable without re-reading the code that produced it.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import provenance
from .environments import ENVIRONMENTS
from .spec import ExperimentSpec
from .validate import preview_spec, validate

Progress = Callable[[float, str], None]
CHUNK = 4                    # replications per scheduling unit (progress and cancellation granularity)
KINDS = ("preview", "research")


def seed_counts(spec: ExperimentSpec) -> Dict[str, int]:
    """How many seeds each namespace supplies for this specification."""
    d = spec.design
    return {"training": d.train_replications if d.tuning == "grid" else 0,
            "pilot": max(0, int(d.pilot_replications)),
            "evaluation": d.replications,
            "validation": max(0, int(d.validation_replications))}


def namespace_seeds(spec: ExperimentSpec, namespace: str) -> List[int]:
    return provenance.seeds(spec.design.seed, namespace, seed_counts(spec)[namespace])


def train_seeds(spec: ExperimentSpec) -> List[int]:
    """Tuning seeds. Never reported as a result."""
    return provenance.seeds(spec.design.seed, "training", spec.design.train_replications)


def test_seeds(spec: ExperimentSpec) -> List[int]:
    """The evaluation seeds: the reported results."""
    return provenance.seeds(spec.design.seed, "evaluation", spec.design.replications)


def pilot_seeds(spec: ExperimentSpec) -> List[int]:
    return namespace_seeds(spec, "pilot")


def validation_seeds(spec: ExperimentSpec) -> List[int]:
    return namespace_seeds(spec, "validation")


def code_version() -> str:
    """Fingerprint of every first-party module the workbench can reach; a change invalidates cached runs.

    Delegates to provenance.code_fingerprint(). The name is kept because it appears in stored provenance records.
    """
    return provenance.code_fingerprint()


def run_key(spec: ExperimentSpec, kind: str) -> str:
    body = json.dumps({"spec": spec.content(), "kind": kind, "code": code_version()}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def effective_spec(spec: ExperimentSpec, kind: str) -> ExperimentSpec:
    return preview_spec(spec) if kind == "preview" else spec.copy()


def estimate(spec: ExperimentSpec, kind: str = "research") -> Dict[str, Any]:
    """Workload before execution: simulated policy-periods and a rough wall-time estimate."""
    s = effective_spec(spec, kind)
    env = ENVIRONMENTS[s.environment]
    levels = max(1, len(s.design.treatment.levels)) if s.design.treatment.control else 1
    units = len(s.policies) * s.design.replications * s.design.periods * levels
    tune = 0
    if s.design.tuning == "grid":
        tune = 8 * s.design.train_replications * s.design.periods * levels
    secs = env.seconds_per_unit * (units + tune)
    return dict(cells=levels, replications=s.design.replications, policies=len(s.policies),
                policy_periods=int(units), tuning_periods=int(tune), seconds=float(secs))


@dataclass
class RunOutput:
    key: str
    kind: str
    spec: ExperimentSpec
    trials: pd.DataFrame                                                       # evaluation seeds: the reported result
    traces: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)   # level label -> policy -> decision log
    provenance: Dict[str, Any] = field(default_factory=dict)
    pilot: pd.DataFrame = field(default_factory=pd.DataFrame)                  # pilot seeds: never a reported result
    validation: pd.DataFrame = field(default_factory=pd.DataFrame)             # validation seeds: confirmation only
    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.provenance.get("status") == "complete"


def level_label(spec: ExperimentSpec, level) -> str:
    return "baseline" if level is None else f"{spec.design.treatment.control} = {level}"


def _git_commit() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def execute(spec: ExperimentSpec, kind: str = "research", progress: Optional[Progress] = None,
            cancel: Optional[Callable[[], bool]] = None) -> RunOutput:
    """Run an experiment. Errors in the specification raise ValueError; nothing runs."""
    if kind not in KINDS:
        raise ValueError(kind)
    errors = [i for i in validate(spec) if i.level == "error"]
    if errors:
        raise ValueError("; ".join(f"{i.step}: {i.message}" for i in errors))
    note = progress or (lambda f, m: None)
    stop = cancel or (lambda: False)
    s = effective_spec(spec, kind)
    env = ENVIRONMENTS[s.environment]
    tr = s.design.treatment
    levels = list(tr.levels) if tr.control else [None]
    counts = seed_counts(s)
    clash = provenance.disjoint(s.design.seed, counts)
    if clash:
        raise ValueError("seed namespaces would overlap: " + clash)
    retention = s.design.trace_retention
    if retention not in provenance.RETENTION:
        raise ValueError("unknown trace retention %r; expected one of %s"
                         % (retention, list(provenance.RETENTION)))
    passes = [("evaluation", test_seeds(s))]
    if counts["pilot"]:
        passes.insert(0, ("pilot", pilot_seeds(s)))
    if counts["validation"]:
        passes.append(("validation", validation_seeds(s)))
    t_start = time.time()
    collected: Dict[str, List[pd.DataFrame]] = {name: [] for name, _ in passes}
    traces: Dict[str, Dict[str, List[pd.DataFrame]]] = {}
    hps: Dict[str, Any] = {}
    total = len(levels) * (sum(len(sd) for _, sd in passes) + (1 if s.design.tuning == "grid" else 0))
    done = 0
    status = "complete"
    for level in levels:
        label = level_label(s, level)
        params = env.params(s, level)
        hp: Dict[str, Any] = {}
        if s.design.tuning == "grid":
            # Tuning sees training seeds only. Nothing selected here is ever an evaluation or validation seed.
            note(done / total, f"{label}: tuning on training seeds")
            hp = env.tune(s, params, train_seeds(s), stop)
            done += 1
        hps[label] = hp
        for stage, stage_seeds in passes:
            # Decision logs are kept for the reported pass only: a pilot or a validation pass is a check, not a record.
            keep = {r for r in range(len(stage_seeds))
                    if stage == "evaluation" and provenance.keeps_trace(retention, r)}
            for c0 in range(0, len(stage_seeds), CHUNK):
                if stop():
                    status = "cancelled"
                    break
                chunk = stage_seeds[c0:c0 + CHUNK]
                note(done / total, f"{label}: {stage} replications {c0 + 1}-{c0 + len(chunk)} of {len(stage_seeds)}")
                rows, tr_ = env.run_cell(s, params, chunk, hp, stop, trace=keep, base=c0)
                for r in rows:
                    r["replication"] = stage_seeds.index(r["seed"])
                    r["level"] = label
                    r["level_value"] = level
                    r["stage"] = stage
                collected[stage].append(pd.DataFrame(rows))
                for pol, frames_ in (tr_ or {}).items():
                    traces.setdefault(label, {}).setdefault(pol, []).extend(frames_)
                done += len(chunk)
            if status == "cancelled":
                break
        if status == "cancelled":
            break
    note(1.0, "Done" if status == "complete" else "Cancelled")

    def _frame(stage: str) -> pd.DataFrame:
        parts = [f for f in collected.get(stage, []) if len(f)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    trials, pilot, validation = _frame("evaluation"), _frame("pilot"), _frame("validation")
    logs = {lvl: {pol: pd.concat(fr, ignore_index=True) for pol, fr in pols.items()}
            for lvl, pols in traces.items()}
    trace_columns = sorted({c for pols in logs.values() for df in pols.values() for c in df.columns})
    prov = dict(status=status, kind=kind, key=run_key(spec, kind), spec_digest=spec.digest(),
                code_version=code_version(), git_commit=_git_commit(), started=t_start, finished=time.time(),
                elapsed_seconds=round(time.time() - t_start, 2), python=platform.python_version(),
                numpy=np.__version__, pandas=pd.__version__, hyperparameters=hps,
                test_seeds=test_seeds(s), train_seeds=train_seeds(s) if s.design.tuning == "grid" else [],
                seed_plan=provenance.seed_plan(s.design.seed, counts),
                environment=env.label, information=env.information(s), engine=env.engine,
                trace_retention=retention, trace_retention_note=provenance.RETENTION_LABELS[retention],
                tuning_note=env.tuning_note if s.design.tuning == "grid" else "No tuning (documented fixed "
                                                                               "parameters).",
                cluster_note=env.cluster_note)
    man = manifest(s, kind, env, prov, trace_columns)
    return RunOutput(prov["key"], kind, s, trials, logs, prov, pilot, validation, man)


def manifest(spec: ExperimentSpec, kind: str, env, prov: Dict[str, Any],
             trace_columns: Sequence[str]) -> Dict[str, Any]:
    """Everything needed to read a stored result without re-reading the code that produced it.

    Configuration, the policies compared, the tuning choices and the budget behind them, the cost definitions in
    force, the information assumptions, the code fingerprint and software environment, the seed plan, the decision-log
    schema with its researcher-only fields, and the status of the run.
    """
    levels = list(spec.design.treatment.levels) if spec.design.treatment.control else [None]
    costs = {level_label(spec, lv): env.cost_definitions(spec, env.params(spec, lv)) for lv in levels}
    return {
        "manifest_version": "1.0",
        "run": {"key": prov["key"], "kind": kind, "status": prov["status"], "spec_digest": prov["spec_digest"],
                "started": prov["started"], "finished": prov["finished"],
                "elapsed_seconds": prov["elapsed_seconds"]},
        "question": {"text": spec.question.text, "primary_outcome": spec.question.primary_outcome,
                     "comparison": list(spec.question.comparison),
                     "smallest_effect_of_interest": spec.question.smallest_effect},
        "configuration": {"environment": env.key, "environment_label": env.label, "engine": env.engine,
                          "parameters": {level_label(spec, lv): env.params(spec, lv) for lv in levels},
                          "treatment": {"control": spec.design.treatment.control,
                                        "levels": list(spec.design.treatment.levels)},
                          "periods": spec.design.periods, "burn_in": spec.design.burn_in,
                          "replications": spec.design.replications},
        "policies": [{"key": k, "label": env.policy(k).label, "kind": env.policy(k).kind,
                      "uses": env.policy(k).uses, "assumptions": env.policy(k).assumptions,
                      "proposed_extension": env.policy(k).extension} for k in spec.policies],
        "candidate_rule": {"key": env.candidate_for(spec).key, "label": env.candidate_for(spec).label,
                           "description": env.candidate_for(spec).description},
        "tuning": {"mode": spec.design.tuning, "note": prov["tuning_note"],
                   "training_replications": spec.design.train_replications,
                   "chosen": prov["hyperparameters"],
                   "budget_note": "Every tunable policy is given the same number of candidates; see the "
                                  "environment adapter's tune()."},
        "costs": costs,
        "information": {"assumptions": prov["information"], "engine": env.engine,
                        "not_applicable": dict(env.not_applicable)},
        "code": provenance.code_fingerprint_detail(),
        "software": provenance.software_environment(),
        "git_commit": prov["git_commit"],
        "seeds": prov["seed_plan"],
        "decision_log": {"retention": prov["trace_retention"], "retention_note": prov["trace_retention_note"],
                         **provenance.decision_log_schema(trace_columns)},
        "researcher_only_fields": provenance.researcher_only_columns(trace_columns),
    }


# ================================================================================================ background runs
def start_background(spec: ExperimentSpec, kind: str = "research") -> str:
    """Run outside the interactive page process: writes the specification to the run store and starts
    `python -m heiner_abm.workbench run-dir <dir>`. Returns the run key; poll with store.progress(key)."""
    from . import store
    key = run_key(spec, kind)
    d = store.run_dir(key)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "request.json"), "w") as f:
        json.dump({"spec": spec.to_dict(), "kind": kind}, f)
    store.write_progress(key, 0.0, "Queued", "running")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log = open(os.path.join(d, "worker.log"), "w")
    proc = subprocess.Popen([sys.executable, "-m", "heiner_abm.workbench", "run-dir", d], cwd=root, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    with open(os.path.join(d, "pid"), "w") as f:
        f.write(str(proc.pid))
    return key


def run_from_dir(d: str) -> RunOutput:
    """Worker entry point: execute the request stored in a run directory, writing progress and outputs there."""
    from . import store
    with open(os.path.join(d, "request.json")) as f:
        req = json.load(f)
    spec = ExperimentSpec.from_dict(req["spec"])
    key = os.path.basename(os.path.normpath(d))
    cancel_file = os.path.join(d, "CANCEL")
    try:
        out = execute(spec, req["kind"], progress=lambda f, m: store.write_progress(key, f, m, "running"),
                      cancel=lambda: os.path.exists(cancel_file))
    except Exception as e:                                        # recorded for the interface, then re-raised
        store.write_progress(key, 1.0, f"Failed: {e}", "failed")
        raise
    store.save_run(out)
    store.write_progress(key, 1.0, "Done" if out.complete else "Cancelled", out.provenance["status"])
    return out
