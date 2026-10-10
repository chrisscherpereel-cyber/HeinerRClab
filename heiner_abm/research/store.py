"""Run directories: write-once stages with checksums, resumable execution, verification and export.

A study is a sequence of stages (heiner_abm.research.studies). Running a study creates a run directory:

    <root>/runs/<run id>/
        run.json                  study, evidence category, status, stage list and the resume log
        config.json               the complete configuration (canonical JSON); its hash is part of the run id
        environment.json          Python, platform, dependency versions and git state when the run was created
        code/<file>               a copy of every source file the stages depend on (the analysis scripts)
        code.json                 their checksums and per-stage source fingerprints
        stages/<stage>/<output>   outputs (.csv.gz tables, .json objects, .md/.html/.txt text)
        stages/<stage>/stage.json checksums of the outputs, seeds, timing, source fingerprint, inputs, notes
        MANIFEST.sha256           checksums of every file, written when the run completes

The run id hashes the study, the configuration and every stage's source fingerprint, so the same configuration and
code always map to the same directory and re-running the same command resumes it: completed stages are verified
against their checksums and skipped, never recomputed or rewritten. A stage is written to a temporary directory and
renamed into place only when complete, so an interruption leaves either a complete stage or nothing. Changing any
source file a stage depends on changes the run id: the old run and its results stay untouched, and a new run must be
recorded before new numbers can be reported.

Roots searched for existing runs: the directory in HEINER_RESEARCH_DIR (default ~/.heiner_lab/research), then the
repository's evidence/ directory, where the runs behind reported results are committed.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import provenance as pv

Progress = Callable[[float, str], None]
EVIDENCE_DIR = os.path.join(pv.REPO_DIR, "evidence")
CATEGORIES = ("analytical", "simulation", "behavioral", "causal")       # heiner_abm.terminology.EVIDENCE_TYPES
ROLES = ("inputs", "splits", "tuning", "compute", "trace", "analysis", "figures")


class IntegrityError(RuntimeError):
    """A stored file does not match its recorded checksum, or a completed result would be overwritten."""


def research_dir() -> str:
    d = os.environ.get("HEINER_RESEARCH_DIR") or os.path.join(os.path.expanduser("~"), ".heiner_lab", "research")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    return d


def roots() -> List[str]:
    return [research_dir(), EVIDENCE_DIR]


def find_run(run_id: str) -> Optional[str]:
    """Directory of a stored run (complete or not) in any root, or None."""
    for r in roots():
        d = os.path.join(r, "runs", run_id)
        if os.path.exists(os.path.join(d, "run.json")):
            return d
    return None


# ================================================================================================ serialization
def _gzip(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as g:   # mtime 0: identical bytes every time
        g.write(data)
    return buf.getvalue()


def _prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Lossless CSV form of a table and its schema. Numeric and boolean columns are written as they are. Text columns
    are written as text, and the schema lists the rows holding a missing value (CSV writes both a missing value and an
    empty string as an empty cell). Object columns that mix types (e.g. True/False with gaps) are written as one JSON
    value per cell."""
    out = df.copy()
    schema: Dict[str, Any] = dict(types={}, missing={})
    for c in out.columns:
        col = out[c]
        if col.dtype != object:
            schema["types"][c] = str(col.dtype)
            continue
        vals = [v for v in col if not (v is None or (isinstance(v, float) and v != v))]
        miss = [int(i) for i, v in enumerate(col) if v is None or (isinstance(v, float) and v != v)]
        if all(isinstance(v, str) for v in vals):
            schema["types"][c] = "str"
            if miss:
                schema["missing"][c] = miss
            out[c] = col.where(col.notna(), "")
        else:
            schema["types"][c] = "json"
            out[c] = [json.dumps(pv._plain(v), sort_keys=True) for v in col]
    return out, schema


def _parse_csv(text: str, schema: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """CSV to DataFrame with the recorded column types (see _prepare)."""
    types = (schema or {}).get("types", {})
    special = {c: str for c, t in types.items() if t in ("str", "json")}
    df = pd.read_csv(io.StringIO(text), float_precision="round_trip", keep_default_na=False, na_values=[""],
                     dtype=special or None)
    for c, t in types.items():
        if t == "str":
            col = df[c].fillna("").astype(object)
            for i in schema.get("missing", {}).get(c, []):
                col.iat[i] = np.nan
            df[c] = col
        elif t == "json":
            df[c] = [json.loads(v) if isinstance(v, str) else None for v in df[c]]
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


def read_table(path: str) -> pd.DataFrame:
    with gzip.open(path, "rb") as f:
        text = f.read().decode()
    sp = path[:-len(".csv.gz")] + ".schema.json"
    schema = None
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            schema = json.load(f)
    return _parse_csv(text, schema)


def encode(name: str, value: Any) -> List[Tuple[str, bytes]]:
    """Files for one stage output. DataFrames become gzip CSV plus a column-type schema, and are re-read to prove that
    the stored table equals the computed one exactly; dicts and lists become canonical JSON; str is stored as text
    with the extension given in the name (default .txt)."""
    if isinstance(value, pd.DataFrame):
        df = value.reset_index(drop=True)
        df.columns = [str(c) for c in df.columns]
        if df.columns.duplicated().any():
            raise ValueError(f"table '{name}' has duplicate column names")
        csv_df, schema = _prepare(df)
        text = csv_df.to_csv(index=False, lineterminator="\n")
        back = _parse_csv(text, schema)
        try:
            pd.testing.assert_frame_equal(df, back, check_dtype=False, check_exact=True)
        except AssertionError as e:
            raise IntegrityError(f"table '{name}' does not round-trip through CSV exactly: {e}") from None
        return [(f"{name}.csv.gz", _gzip(text.encode())),
                (f"{name}.schema.json", (pv.canonical_json(schema) + "\n").encode())]
    if isinstance(value, bytes):
        return [(name, value)]
    if isinstance(value, str):
        return [((name if "." in name else f"{name}.txt"), value.encode())]
    return [(f"{name}.json", (pv.canonical_json(value) + "\n").encode())]


def decode(path: str) -> Any:
    if path.endswith(".csv.gz"):
        return read_table(path)
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode()
    except UnicodeDecodeError:
        return data


def _matching(files, name: str) -> List[str]:
    """The data file of output `name` (schema files accompany tables and are not outputs of their own)."""
    return [f for f in files if not f.endswith(".schema.json") and (f == name or f.split(".")[0] == name)]


def _write_atomic(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


# ================================================================================================ stages and studies
@dataclass(frozen=True)
class Stage:
    name: str
    fn: Callable[["StageContext"], Dict[str, Any]]
    sources: Tuple[str, ...]            # heiner_abm files the computation depends on (relative to heiner_abm/)
    deps: Tuple[str, ...] = ()          # stages whose outputs it reads
    role: str = "compute"               # one of ROLES
    description: str = ""


@dataclass
class Study:
    key: str
    title: str
    category: str                       # one of CATEGORIES
    description: str
    default_config: Callable[[], Dict[str, Any]]
    stages: Callable[[Dict[str, Any]], List[Stage]]
    agents: Callable[[Dict[str, Any]], Dict[str, Any]] = field(default=lambda cfg: {})   # agent info & feedback spec


class StageContext:
    """What a stage function sees: the configuration, completed upstream outputs, cross-run inputs, and recorders
    for seeds and notes (both saved in stage.json)."""

    def __init__(self, run: "Run", stage: Stage, progress: Optional[Progress]):
        self.run, self.stage, self.config = run, stage, run.config
        self.seeds: Dict[str, Any] = {}
        self.notes: Dict[str, Any] = {}
        self.inputs: Dict[str, Any] = {}
        self._progress = progress

    def load(self, stage: str, name: str) -> Any:
        if stage not in self.stage.deps:
            raise KeyError(f"stage '{self.stage.name}' does not declare a dependency on '{stage}'")
        return self.run.output(stage, name)

    def input(self, key: str) -> Any:
        """A file from another recorded run, named in config['inputs'][key] as {run, stage, file, sha256}; the
        checksum is verified, and the reference is recorded in stage.json."""
        ref = self.config["inputs"][key]
        d = find_run(ref["run"])
        if d is None:
            raise FileNotFoundError(f"input run {ref['run']} not found in {roots()}")
        path = os.path.join(d, "stages", ref["stage"], ref["file"])
        got = pv.sha256_file(path)
        if got != ref["sha256"]:
            raise IntegrityError(f"input {key}: {path} has checksum {got}, expected {ref['sha256']}")
        self.inputs[key] = dict(ref, path=os.path.relpath(path, pv.REPO_DIR) if path.startswith(pv.REPO_DIR) else None)
        return decode(path)

    def seed(self, label: str, value: Any) -> Any:
        self.seeds[label] = value
        return value

    def note(self, label: str, value: Any) -> None:
        self.notes[label] = value

    def progress(self, fraction: float, message: str) -> None:
        if self._progress:
            self._progress(fraction, f"{self.stage.name}: {message}")


# ================================================================================================ runs
class Run:
    def __init__(self, study: Study, config: Optional[Dict[str, Any]] = None, root: Optional[str] = None):
        self.study = study
        self.config = json.loads(pv.canonical_json(config if config is not None else study.default_config()))
        self.stages = study.stages(self.config)
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("duplicate stage names")
        for s in self.stages:
            if s.role not in ROLES:
                raise ValueError(f"stage {s.name}: unknown role {s.role}")
            missing = [d for d in s.deps if d not in names[:names.index(s.name)]]
            if missing:
                raise ValueError(f"stage {s.name} depends on later or unknown stages {missing}")
        self.fingerprints = {s.name: pv.source_fingerprint(s.sources) for s in self.stages}
        self.config_hash = pv.digest(self.config)
        self.run_id = pv.digest(dict(study=study.key, config=self.config, code=self.fingerprints))
        existing = find_run(self.run_id)
        self.dir = existing or os.path.join(root or research_dir(), "runs", self.run_id)

    # ------------------------------------------------------------------------------------------ state
    def stage_dir(self, name: str) -> str:
        return os.path.join(self.dir, "stages", name)

    def stage_record(self, name: str) -> Optional[Dict[str, Any]]:
        p = os.path.join(self.stage_dir(name), "stage.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def output(self, stage: str, name: str) -> Any:
        rec = self.stage_record(stage)
        if rec is None:
            raise FileNotFoundError(f"stage {stage} of run {self.run_id} is not complete")
        files = _matching(rec["outputs"], name)
        if len(files) != 1:
            raise KeyError(f"output {name} of stage {stage}: found {files}")
        return decode(os.path.join(self.stage_dir(stage), files[0]))

    def complete(self) -> bool:
        return os.path.exists(os.path.join(self.dir, "MANIFEST.sha256"))

    # ------------------------------------------------------------------------------------------ setup
    def _initialize(self) -> None:
        os.makedirs(os.path.join(self.dir, "stages"), exist_ok=True)
        cfg = os.path.join(self.dir, "config.json")
        if os.path.exists(cfg):
            with open(cfg, encoding="utf-8") as f:
                if pv.digest(json.load(f)) != self.config_hash:
                    raise IntegrityError(f"{cfg} does not match this run's configuration")
            return
        _write_atomic(cfg, (pv.canonical_json(self.config) + "\n").encode())
        _write_atomic(os.path.join(self.dir, "environment.json"), (pv.canonical_json(pv.environment()) + "\n").encode())
        code_dir = os.path.join(self.dir, "code")
        files = sorted({f for s in self.stages for f in s.sources})
        sums = {}
        for f in files:
            dst = os.path.join(code_dir, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(pv.PACKAGE_DIR, f), dst)
            sums[f] = pv.sha256_file(dst)
        _write_atomic(os.path.join(self.dir, "code.json"), (pv.canonical_json(dict(
            files=sums, stage_fingerprints=self.fingerprints,
            stage_sources={s.name: sorted(s.sources) for s in self.stages})) + "\n").encode())
        self._write_run_json(status="running", resumes=[])

    def _run_json(self) -> Dict[str, Any]:
        p = os.path.join(self.dir, "run.json")
        if not os.path.exists(p):
            return {}
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def _write_run_json(self, **kw) -> None:
        cur = self._run_json()
        cur.update(dict(run_id=self.run_id, study=self.study.key, title=self.study.title,
                        category=self.study.category, config_hash=self.config_hash,
                        created=cur.get("created", pv.now()),
                        stages=[dict(name=s.name, role=s.role, deps=list(s.deps), description=s.description,
                                     fingerprint=self.fingerprints[s.name],
                                     status="complete" if self.stage_record(s.name) else "pending")
                                for s in self.stages]))
        cur.update(kw)
        _write_atomic(os.path.join(self.dir, "run.json"), (pv.canonical_json(cur) + "\n").encode())

    # ------------------------------------------------------------------------------------------ execution
    def execute(self, progress: Optional[Progress] = None, stop_after: Optional[str] = None) -> "Run":
        """Run every stage that is not complete yet, in order. Completed stages are verified and skipped. With
        stop_after, stop once that stage is complete (used to test resumption)."""
        self._initialize()
        rj = self._run_json()
        if any(self.stage_record(s.name) for s in self.stages):
            rj.setdefault("resumes", []).append(dict(at=pv.now(), git=pv.git_state()))
            self._write_run_json(resumes=rj["resumes"])
        n = len(self.stages)
        for i, s in enumerate(self.stages):
            rec = self.stage_record(s.name)
            if rec is not None:
                self.verify_stage(s.name)
                if progress:
                    progress((i + 1) / n, f"{s.name}: complete (verified)")
            else:
                self._run_stage(s, lambda f, m, i=i: progress((i + f) / n, m) if progress else None)
            if stop_after == s.name:
                self._write_run_json(status="interrupted")
                return self
        self._write_run_json(status="complete", finished=pv.now())
        self._write_manifest()
        return self

    def _run_stage(self, s: Stage, progress: Progress) -> None:
        final = self.stage_dir(s.name)
        if os.path.exists(final):
            raise IntegrityError(f"{final} exists without a completion record; refusing to overwrite it")
        tmp = os.path.join(self.dir, "stages", f".{s.name}.partial")
        shutil.rmtree(tmp, ignore_errors=True)               # leftovers of an interrupted attempt
        os.makedirs(tmp)
        ctx = StageContext(self, s, progress)
        t0, started = time.time(), pv.now()
        outputs = s.fn(ctx) or {}
        files = {}
        for name, value in outputs.items():
            for fn, data in encode(name, value):
                if fn in files:
                    raise ValueError(f"stage {s.name}: duplicate output file {fn}")
                _write_atomic(os.path.join(tmp, fn), data)
                files[fn] = pv.sha256_bytes(data)
        rec = dict(stage=s.name, role=s.role, description=s.description, run_id=self.run_id,
                   fingerprint=self.fingerprints[s.name], sources=sorted(s.sources), deps={
                       d: self.stage_record(d)["output_digest"] for d in s.deps},
                   outputs=files, output_digest=pv.digest(files), seeds=ctx.seeds, notes=ctx.notes,
                   inputs=ctx.inputs, started=started, finished=pv.now(), seconds=round(time.time() - t0, 3))
        _write_atomic(os.path.join(tmp, "stage.json"), (pv.canonical_json(rec) + "\n").encode())
        os.replace(tmp, final)
        self._write_run_json(status="running")

    # ------------------------------------------------------------------------------------------ verification
    def verify_stage(self, name: str) -> None:
        rec = self.stage_record(name)
        d = self.stage_dir(name)
        present = {f for f in os.listdir(d) if f != "stage.json"}
        if present != set(rec["outputs"]):
            raise IntegrityError(f"stage {name}: files {sorted(present ^ set(rec['outputs']))} differ from the record")
        for fn, h in rec["outputs"].items():
            got = pv.sha256_file(os.path.join(d, fn))
            if got != h:
                raise IntegrityError(f"stage {name}: {fn} has checksum {got}, recorded {h}")
        if pv.digest(rec["outputs"]) != rec["output_digest"]:
            raise IntegrityError(f"stage {name}: output digest does not match its files")

    def _all_files(self) -> List[str]:
        out = []
        for base, _, files in os.walk(self.dir):
            for f in files:
                rel = os.path.relpath(os.path.join(base, f), self.dir).replace(os.sep, "/")
                if rel != "MANIFEST.sha256" and not rel.endswith(".tmp"):
                    out.append(rel)
        return sorted(out)

    def _write_manifest(self) -> None:
        lines = [f"{pv.sha256_file(os.path.join(self.dir, f))}  {f}" for f in self._all_files()]
        _write_atomic(os.path.join(self.dir, "MANIFEST.sha256"), ("\n".join(lines) + "\n").encode())


def verify_dir(d: str) -> List[str]:
    """Problems found in a stored run directory (empty list = every checksum matches)."""
    problems = []
    with open(os.path.join(d, "code.json"), encoding="utf-8") as f:
        code = json.load(f)
    for fn, h in code["files"].items():
        p = os.path.join(d, "code", fn)
        if not os.path.exists(p) or pv.sha256_file(p) != h:
            problems.append(f"code/{fn}: missing or changed")
    sdir = os.path.join(d, "stages")
    for name in sorted(os.listdir(sdir)) if os.path.exists(sdir) else []:
        rp = os.path.join(sdir, name, "stage.json")
        if name.startswith("."):
            problems.append(f"stages/{name}: incomplete stage left by an interruption")
            continue
        if not os.path.exists(rp):
            problems.append(f"stages/{name}: no completion record")
            continue
        with open(rp, encoding="utf-8") as f:
            rec = json.load(f)
        for fn, h in rec["outputs"].items():
            p = os.path.join(sdir, name, fn)
            if not os.path.exists(p) or pv.sha256_file(p) != h:
                problems.append(f"stages/{name}/{fn}: missing or changed")
    mp = os.path.join(d, "MANIFEST.sha256")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            for line in f:
                h, fn = line.rstrip("\n").split("  ", 1)
                p = os.path.join(d, fn)
                if not os.path.exists(p) or pv.sha256_file(p) != h:
                    problems.append(f"{fn}: does not match MANIFEST.sha256")
    return problems


def load_run_record(run_id: str) -> Dict[str, Any]:
    d = find_run(run_id)
    if d is None:
        raise FileNotFoundError(f"run {run_id} not found in {roots()}")
    with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
        rec = json.load(f)
    rec["dir"] = d
    return rec


def stored_output(run_id: str, stage: str, name: str) -> Any:
    d = find_run(run_id)
    if d is None:
        raise FileNotFoundError(f"run {run_id} not found in {roots()}")
    with open(os.path.join(d, "stages", stage, "stage.json"), encoding="utf-8") as f:
        rec = json.load(f)
    files = _matching(rec["outputs"], name)
    if len(files) != 1:
        raise KeyError(f"{run_id}/{stage}: output {name} not found ({sorted(rec['outputs'])})")
    return decode(os.path.join(d, "stages", stage, files[0]))


def stage_record(run_id: str, stage: str) -> Dict[str, Any]:
    d = find_run(run_id)
    with open(os.path.join(d, "stages", stage, "stage.json"), encoding="utf-8") as f:
        return json.load(f)


def file_ref(run_id: str, stage: str, file: str) -> Dict[str, str]:
    """A cross-run input reference (for config['inputs']), with the file's checksum."""
    rec = stage_record(run_id, stage)
    return dict(run=run_id, stage=stage, file=file, sha256=rec["outputs"][file])


def export_zip(run_id: str) -> bytes:
    """The whole run directory as a zip with sorted entries and fixed timestamps (byte-identical for equal runs)."""
    d = find_run(run_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in sorted(os.walk(d)):
            for f in sorted(files):
                p = os.path.join(base, f)
                info = zipfile.ZipInfo(os.path.join(run_id, os.path.relpath(p, d)).replace(os.sep, "/"),
                                       date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                with open(p, "rb") as fh:
                    z.writestr(info, fh.read())
    return buf.getvalue()


def publish(run_id: str) -> str:
    """Copy a complete run into the repository's evidence/runs/ (the runs behind reported results are committed).
    An existing published copy is never overwritten."""
    src = find_run(run_id)
    if src is None or not os.path.exists(os.path.join(src, "MANIFEST.sha256")):
        raise FileNotFoundError(f"run {run_id} is not complete")
    dst = os.path.join(EVIDENCE_DIR, "runs", run_id)
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst
    if os.path.exists(dst):
        raise IntegrityError(f"{dst} already exists; published runs are never overwritten")
    problems = verify_dir(src)
    if problems:
        raise IntegrityError(f"run {run_id} fails verification: {problems[:5]}")
    shutil.copytree(src, dst)
    return dst


def list_runs(root: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = []
    for r in ([root] if root else roots()):
        d = os.path.join(r, "runs")
        for k in sorted(os.listdir(d)) if os.path.exists(d) else []:
            p = os.path.join(d, k, "run.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    j = json.load(f)
                rows.append(dict(run_id=k, study=j.get("study"), status=j.get("status"), created=j.get("created"),
                                 root=r))
    return rows
