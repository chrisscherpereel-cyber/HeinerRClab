"""Run store: saved experiments, completed runs with their provenance, progress of running ones, and exports.

Location: the directory in the environment variable HEINER_LAB_DIR, else ~/.heiner_lab. Layout:
    experiments/<slug>.json          saved, named specifications
    runs/<key>/spec.json             the executed specification (after preview reduction)
    runs/<key>/manifest.json         configuration, policies, tuning, costs, information, code fingerprint, seeds,
                                     decision-log schema, researcher-only fields and run status
    runs/<key>/provenance.json       status, code fingerprint, git commit, seed plan, timing, tuned parameters
    runs/<key>/trials.csv            one row per policy, replication and treatment level (evaluation seeds)
    runs/<key>/pilot.csv             the pilot pass, if any. Never a reported result
    runs/<key>/validation.csv        the validation pass, if any
    runs/<key>/traces.parquet        decision logs, column-wise and compressed (traces.pkl.gz without pyarrow)
    runs/<key>/progress.json         progress of a running job
A run's key hashes its specification content, its kind (preview or research) and the code fingerprint, so a stored
run is reused only for the same configuration and the same code. Exports are built from these stored files.

Decision logs are stored column-wise because they are the largest artifact: retention "all" keeps every replication
of every condition, which is periods x replications x policies x conditions rows. Parquet is used when pyarrow is
installed (it is not a declared dependency) and gzipped pickle otherwise; both round-trip the same frames.
"""
from __future__ import annotations

import gzip
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
TRACE_FILES = ("traces.parquet", "traces.pkl.gz", "traces.pkl")


def _have_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_traces(d: str, traces: Dict[str, Dict[str, pd.DataFrame]]) -> Optional[str]:
    """Write the decision logs column-wise. Returns the file written, or None when there is nothing to write."""
    for fn in TRACE_FILES:
        q = os.path.join(d, fn)
        if os.path.exists(q):
            os.remove(q)
    frames = [df.assign(**{"__level": lvl, "__policy": pol})
              for lvl, pols in (traces or {}).items() for pol, df in pols.items() if len(df)]
    if not frames:
        return None
    if _have_parquet():
        path = os.path.join(d, "traces.parquet")
        pd.concat(frames, ignore_index=True).to_parquet(path, index=False, compression="snappy")
        return path
    path = os.path.join(d, "traces.pkl.gz")
    with gzip.open(path, "wb") as f:
        pickle.dump(traces, f, protocol=4)
    return path


def load_traces(d: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Read decision logs back, whichever of the supported layouts they were written in."""
    path = os.path.join(d, "traces.parquet")
    if os.path.exists(path):
        flat = pd.read_parquet(path)
        out: Dict[str, Dict[str, pd.DataFrame]] = {}
        for (lvl, pol), g in flat.groupby(["__level", "__policy"], sort=False):
            out.setdefault(str(lvl), {})[str(pol)] = g.drop(columns=["__level", "__policy"]).reset_index(drop=True)
        return out
    for fn, opener in (("traces.pkl.gz", gzip.open), ("traces.pkl", open)):
        q = os.path.join(d, fn)
        if os.path.exists(q):
            with opener(q, "rb") as f:
                return pickle.load(f)
    return {}


def save_run(out) -> str:
    d = run_dir(out.key)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "spec.json"), "w") as f:
        f.write(out.spec.to_json())
    with open(os.path.join(d, "provenance.json"), "w") as f:
        json.dump(out.provenance, f, indent=1, default=str)
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(getattr(out, "manifest", {}) or {}, f, indent=1, default=str, sort_keys=True)
    out.trials.to_csv(os.path.join(d, "trials.csv"), index=False)
    for name in ("pilot", "validation"):
        df = getattr(out, name, None)
        q = os.path.join(d, name + ".csv")
        if df is not None and len(df):
            df.to_csv(q, index=False)
        elif os.path.exists(q):
            os.remove(q)
    save_traces(d, out.traces)
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

    def _optional(name):
        q = os.path.join(d, name + ".csv")
        return pd.read_csv(q) if os.path.exists(q) else pd.DataFrame()

    man = {}
    mp = os.path.join(d, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as f:
            man = json.load(f)
    return RunOutput(key, prov.get("kind", "research"), spec, trials, load_traces(d), prov,
                     _optional("pilot"), _optional("validation"), man)


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
    man = getattr(out, "manifest", {}) or {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in ("spec.json", "manifest.json", "provenance.json", "trials.csv", "pilot.csv", "validation.csv"):
            q = os.path.join(d, fn)
            if os.path.exists(q):
                z.write(q, fn)
        for level, pols in (out.traces or {}).items():
            for pol, df in pols.items():
                z.writestr(f"traces/{slug(level)}__{pol}.csv", df.to_csv(index=False))
        z.writestr("README.txt", _export_readme(key, out, man))
    return buf.getvalue()


def _export_readme(key: str, out, man: Dict[str, Any]) -> str:
    """What a reader of the bundle needs: how to reproduce it, what the seeds meant, and what the agents never saw."""
    from . import provenance
    nl = chr(10)
    seeds = (man.get("seeds") or {}).get("namespaces", {})
    lines = [
        "Run " + key + " from the Heiner reliability laboratory.",
        "",
        "Reproduce with: python -m heiner_abm.workbench run spec.json --kind " + str(out.kind),
        "Code fingerprint " + str(out.provenance.get("code_version")) + "; git commit "
        + str(out.provenance.get("git_commit")) + "; status " + str(out.provenance.get("status")) + ".",
        "manifest.json records the configuration, policies, tuning choices, cost definitions, information "
        "assumptions, software versions and seed plan behind these numbers.",
        "",
        "Files",
        "  trials.csv      the reported results (evaluation seeds)",
        "  pilot.csv       pilot estimates, if a pilot pass was run. NOT a reported result",
        "  validation.csv  confirmation on fresh environments, if a validation pass was run",
        "  traces/         per-period decision logs, one file per condition and policy",
        "",
        "Seed namespaces (disjoint by construction)",
    ]
    for name in ("training", "pilot", "evaluation", "validation"):
        info = seeds.get(name) or {}
        lines.append("  %-11s %4s seeds  %s" % (name, info.get("count", 0),
                                                provenance.SEED_PURPOSE.get(name, "")))
    lines += [
        "",
        "Decision-log columns are grouped by role in manifest.json under decision_log.roles:",
        "  predecision   what the agent had when it chose",
        "  action        what it chose, and why",
        "  postdecision  consequences, known only afterwards",
        "  feedback      what it was later told, where that came from (feedback_origin), when the outcome matured",
        "                (feedback_matured_period) and when it reached the agent (feedback_release_period; "
        + str(provenance.NEVER_RELEASED) + " means never)",
        "",
        "Researcher-only columns start with '" + provenance.RESEARCHER_PREFIX + "' and were never available to any "
        "agent at any time. In this run: " + (", ".join(man.get("researcher_only_fields") or []) or "none recorded")
        + ".",
        "",
    ]
    return nl.join(lines)
