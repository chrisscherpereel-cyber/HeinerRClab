"""Execution service: seeds, tuning, scheduling, progress, cancellation and provenance for one experiment.

The same function runs an experiment from the interface (preview in-process; research runs in a separate process, see
`start_background`) and from the command line (python -m heiner_abm.workbench run spec.json). Random streams:
    training seeds (tuning only)   seed * 1000 + 100_000_000 + r
    test seeds (reported results)  seed * 1000 + 300_000_000 + r
The two blocks never overlap, and every policy in a cell sees the same test seeds (common random numbers), so policy
differences are paired.
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
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .environments import ENVIRONMENTS
from .spec import ExperimentSpec
from .validate import preview_spec, validate

Progress = Callable[[float, str], None]
CHUNK = 4                    # replications per scheduling unit (progress and cancellation granularity)
KINDS = ("preview", "research")


def train_seeds(spec: ExperimentSpec) -> List[int]:
    return [spec.design.seed * 1000 + 100_000_000 + r for r in range(spec.design.train_replications)]


def test_seeds(spec: ExperimentSpec) -> List[int]:
    return [spec.design.seed * 1000 + 300_000_000 + r for r in range(spec.design.replications)]


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """Hash of the workbench and of the engines it calls: a change invalidates cached runs."""
    from .. import bench_inventory, engine, gates, information, learnability, nk, params, tasks
    from . import environments, spec as spec_mod
    mods = [environments, spec_mod, sys.modules[__name__], engine, params, gates, information, learnability, nk, tasks,
            bench_inventory]
    src = "".join(inspect.getsource(m) for m in mods)
    return hashlib.sha256(src.encode().replace(b"\r\n", b"\n")).hexdigest()[:12]


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
    trials: pd.DataFrame
    traces: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)   # level label -> policy -> trace
    provenance: Dict[str, Any] = field(default_factory=dict)

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
    seeds = test_seeds(s)
    t_start = time.time()
    frames, traces, hps = [], {}, {}
    total = len(levels) * (len(seeds) + (1 if s.design.tuning == "grid" else 0))
    done = 0
    status = "complete"
    for li, level in enumerate(levels):
        label = level_label(s, level)
        params = env.params(s, level)
        hp: Dict[str, Any] = {}
        if s.design.tuning == "grid":
            note(done / total, f"{label}: tuning on training seeds")
            hp = env.tune(s, params, train_seeds(s), stop)
            done += 1
        hps[label] = hp
        for c0 in range(0, len(seeds), CHUNK):
            if stop():
                status = "cancelled"
                break
            chunk = seeds[c0:c0 + CHUNK]
            note(done / total, f"{label}: replications {c0 + 1}–{c0 + len(chunk)} of {len(seeds)}")
            rows, tr_ = env.run_cell(s, params, chunk, hp, stop, trace=(c0 == 0))
            for r in rows:
                r["replication"] = seeds.index(r["seed"])
                r["level"] = label
                r["level_value"] = level
            frames.append(pd.DataFrame(rows))
            if tr_:
                traces[label] = tr_
            done += len(chunk)
        if status == "cancelled":
            break
    note(1.0, "Done" if status == "complete" else "Cancelled")
    trials = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    prov = dict(status=status, kind=kind, key=run_key(spec, kind), spec_digest=spec.digest(),
                code_version=code_version(), git_commit=_git_commit(), started=t_start, finished=time.time(),
                elapsed_seconds=round(time.time() - t_start, 2), python=platform.python_version(),
                numpy=np.__version__, pandas=pd.__version__, hyperparameters=hps,
                test_seeds=seeds, train_seeds=train_seeds(s) if s.design.tuning == "grid" else [],
                environment=env.label, information=env.information(s), engine=env.engine,
                tuning_note=env.tuning_note if s.design.tuning == "grid" else "No tuning (documented fixed "
                                                                               "parameters).",
                cluster_note=env.cluster_note)
    return RunOutput(prov["key"], kind, s, trials, traces, prov)


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
