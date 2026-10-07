"""Run store: saved experiments, completed runs with their provenance, progress of running ones, and exports.

Location: the directory in the environment variable HEINER_LAB_DIR, else ~/.heiner_lab. Layout:
    experiments/<slug>.json          saved, named specifications
    runs/<key>/spec.json             the executed specification (after preview reduction)
    runs/<key>/provenance.json       status, code version, git commit, seeds, timing, tuned parameters
    runs/<key>/trials.csv            one row per policy, replication and treatment level
    runs/<key>/traces.pkl            decision traces (first replication of each level)
    runs/<key>/progress.json         progress of a running job
A run's key hashes its specification content, its kind (preview or research) and the code version, so a stored run is
reused only for the same configuration and code. Exports are built from these stored files.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import re
import shutil
import time
import zipfile
from typing import Any, Dict, List, Optional

import pandas as pd

from .spec import ExperimentSpec


def base_dir() -> str:
    d = os.environ.get("HEINER_LAB_DIR") or os.path.join(os.path.expanduser("~"), ".heiner_lab")
    os.makedirs(os.path.join(d, "experiments"), exist_ok=True)
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    return d


def run_dir(key: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{16}", key):
        raise ValueError("invalid run key")
    return os.path.join(base_dir(), "runs", key)


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:60] or "experiment"


# ================================================================================================ experiments
def save_experiment(spec: ExperimentSpec) -> str:
    path = os.path.join(base_dir(), "experiments", slug(spec.name) + ".json")
    with open(path, "w") as f:
        f.write(spec.to_json())
    return path


def list_experiments() -> List[Dict[str, Any]]:
    d = os.path.join(base_dir(), "experiments")
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            p = os.path.join(d, fn)
            try:
                spec = ExperimentSpec.from_json(open(p).read())
            except Exception:
                continue
            out.append(dict(slug=fn[:-5], name=spec.name, question=spec.question.text, environment=spec.environment,
                            modified=os.path.getmtime(p)))
    return sorted(out, key=lambda r: -r["modified"])


def load_experiment(slug_: str) -> ExperimentSpec:
    with open(os.path.join(base_dir(), "experiments", slug(slug_) + ".json")) as f:
        return ExperimentSpec.from_json(f.read())


def duplicate_experiment(slug_: str, new_name: str) -> ExperimentSpec:
    spec = load_experiment(slug_)
    spec.name = new_name
    save_experiment(spec)
    return spec


def delete_experiment(slug_: str) -> None:
    p = os.path.join(base_dir(), "experiments", slug(slug_) + ".json")
    if os.path.exists(p):
        os.remove(p)


# ================================================================================================ runs
def save_run(out) -> str:
    d = run_dir(out.key)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "spec.json"), "w") as f:
        f.write(out.spec.to_json())
    with open(os.path.join(d, "provenance.json"), "w") as f:
        json.dump(out.provenance, f, indent=1, default=str)
    out.trials.to_csv(os.path.join(d, "trials.csv"), index=False)
    with open(os.path.join(d, "traces.pkl"), "wb") as f:
        pickle.dump(out.traces, f)
    return d


def load_run(key: str):
    from .execution import RunOutput
    d = run_dir(key)
    if not os.path.exists(os.path.join(d, "provenance.json")):
        return None
    with open(os.path.join(d, "provenance.json")) as f:
        prov = json.load(f)
    spec = ExperimentSpec.from_json(open(os.path.join(d, "spec.json")).read())
    trials = pd.read_csv(os.path.join(d, "trials.csv"))
    traces = {}
    tp = os.path.join(d, "traces.pkl")
    if os.path.exists(tp):
        with open(tp, "rb") as f:
            traces = pickle.load(f)
    return RunOutput(key, prov.get("kind", "research"), spec, trials, traces, prov)


def cached(spec: ExperimentSpec, kind: str):
    """A completed stored run of this exact configuration, kind and code version, or None."""
    from .execution import run_key
    out = load_run(run_key(spec, kind))
    return out if out is not None and out.complete else None


def list_runs() -> List[Dict[str, Any]]:
    d = os.path.join(base_dir(), "runs")
    rows = []
    for k in sorted(os.listdir(d)):
        p = os.path.join(d, k, "provenance.json")
        if os.path.exists(p):
            prov = json.load(open(p))
            spec = json.load(open(os.path.join(d, k, "spec.json")))
            rows.append(dict(key=k, name=spec.get("name"), kind=prov.get("kind"), status=prov.get("status"),
                             finished=prov.get("finished"), question=spec.get("question", {}).get("text")))
    return sorted(rows, key=lambda r: -(r["finished"] or 0))


# ================================================================================================ progress
def write_progress(key: str, fraction: float, message: str, status: str) -> None:
    d = run_dir(key)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "progress.json.tmp")
    with open(tmp, "w") as f:
        json.dump(dict(fraction=float(fraction), message=message, status=status, time=time.time()), f)
    os.replace(tmp, os.path.join(d, "progress.json"))


def progress(key: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(run_dir(key), "progress.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def request_cancel(key: str) -> None:
    with open(os.path.join(run_dir(key), "CANCEL"), "w") as f:
        f.write(str(time.time()))


def discard(key: str) -> None:
    shutil.rmtree(run_dir(key), ignore_errors=True)


# ================================================================================================ export
def export_zip(key: str) -> bytes:
    """A reproducible bundle of a stored run: specification, provenance, trial data, traces as CSV and a README."""
    d = run_dir(key)
    out = load_run(key)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in ("spec.json", "provenance.json", "trials.csv"):
            z.write(os.path.join(d, fn), fn)
        for level, pols in (out.traces or {}).items():
            for pol, df in pols.items():
                z.writestr(f"traces/{slug(level)}__{pol}.csv", df.to_csv(index=False))
        z.writestr("README.txt",
                   "Reproduce with: python -m heiner_abm.workbench run spec.json "
                   f"--kind {out.kind}\nRun key {key}; code version {out.provenance.get('code_version')}; git commit "
                   f"{out.provenance.get('git_commit')}.\nResearcher-only columns in traces start with 'researcher_' "
                   "and were never available to the agents.\n")
    return buf.getvalue()
