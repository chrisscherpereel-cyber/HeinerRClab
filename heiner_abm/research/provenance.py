"""Provenance primitives: canonical JSON, checksums, source fingerprints and the computing environment.

A *source fingerprint* is the SHA-256 of the listed heiner_abm source files (line endings normalized). Every stage of a
study declares the files its computation depends on; the fingerprint is recorded with the stage's outputs, so a later
change to any of those files is detected and the outputs are reported as requiring replication.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # heiner_abm/
REPO_DIR = os.path.dirname(PACKAGE_DIR)
DEPENDENCIES = ("numpy", "pandas", "scipy", "plotly", "streamlit", "openpyxl", "xlrd")


def _plain(o: Any) -> Any:
    """JSON-compatible form: numpy scalars and arrays, tuples, sets, NaN (as null) and dataclass-like objects."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return sorted(_plain(v) for v in o)
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if math.isnan(f) else (str(f) if math.isinf(f) else f)
    return o


def canonical_json(obj: Any, indent: Optional[int] = 1) -> str:
    """Deterministic JSON (sorted keys, NaN as null): equal content gives equal text and therefore equal checksums."""
    return json.dumps(_plain(obj), sort_keys=True, indent=indent, ensure_ascii=False, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj: Any, n: int = 16) -> str:
    """Short SHA-256 of an object's canonical JSON."""
    return sha256_bytes(canonical_json(obj, indent=None).encode())[:n]


def source_fingerprint(sources: Iterable[str]) -> str:
    """SHA-256 (16 hex digits) of the listed source files, relative to heiner_abm/ (e.g. "arena.py",
    "research/traces.py"), line endings normalized. Order-independent."""
    h = hashlib.sha256()
    for name in sorted(set(sources)):
        with open(os.path.join(PACKAGE_DIR, name), "rb") as f:
            h.update(name.encode() + b"\0" + f.read().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> Dict[str, Any]:
    """Commit, branch and whether the working tree has uncommitted changes to tracked files (the source fingerprints,
    not the commit, are what identify the code that produced a result; the commit locates it in history)."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return dict(commit=commit, branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=None if status is None else bool(status),
                dirty_files=sorted(line[3:] for line in (status or "").splitlines())[:200])


def dependency_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata
    out = {}
    for name in DEPENDENCIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment() -> Dict[str, Any]:
    """Everything about the computing environment that could affect a result, and nothing personal (no user name,
    host name or paths outside the repository)."""
    return dict(python=sys.version.split()[0], implementation=platform.python_implementation(),
                platform=dict(system=platform.system(), release=platform.release(), machine=platform.machine()),
                numpy_blas=_blas(), dependencies=dependency_versions(), git=git_state(),
                recorded_at=now())


def _blas() -> Optional[str]:
    try:
        cfg = np.show_config(mode="dicts")              # numpy >= 1.26
        return (cfg.get("Build Dependencies", {}).get("blas", {}) or {}).get("name")
    except Exception:
        return None


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


# Infrastructure excluded from source fingerprints: the run store and these primitives (their correctness is enforced
# by checksums and exact round-trip checks, and a refactor of them must not invalidate every result), and the
# registered-results module, whose tuned parameters a study receives as a checksummed input from a recorded run
# instead (see heiner_abm.research.studies.common.tuned_override).
FINGERPRINT_EXCLUDE = frozenset({"research/store.py", "research/provenance.py", "research/__init__.py",
                                 "research/claims.py", "research/manuscript.py", "research/__main__.py",
                                 "registered.py", "__init__.py"})


def _resolve(base_dir: str, level: int, module: Optional[str], names: Iterable[str]) -> Iterable[str]:
    d = base_dir
    for _ in range(level - 1):
        d = os.path.dirname(d)
    parts = module.split(".") if module else []
    target = os.path.join(d, *parts)
    out = []
    if parts:
        if os.path.exists(target + ".py"):
            out.append(target + ".py")
        elif os.path.isdir(target):
            out.append(os.path.join(target, "__init__.py"))
    for n in names:                           # `from . import x` or `from .pkg import submodule`
        cand = os.path.join(target, n + ".py")
        if os.path.exists(cand):
            out.append(cand)
        elif os.path.isdir(os.path.join(target, n)):
            out.append(os.path.join(target, n, "__init__.py"))
    return out


def module_closure(files: Iterable[str], exclude: Iterable[str] = FINGERPRINT_EXCLUDE) -> List[str]:
    """Every heiner_abm source file the given files import, directly or transitively (relative imports and absolute
    heiner_abm imports, including imports inside functions), relative to heiner_abm/, minus `exclude`."""
    import ast
    excl = set(exclude)
    todo = [os.path.join(PACKAGE_DIR, f) for f in files]
    seen = set()
    while todo:
        path = os.path.normpath(todo.pop())
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        base = os.path.dirname(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [a.name for a in node.names]
                if node.level:
                    todo += _resolve(base, node.level, node.module, names)
                elif node.module and node.module.split(".")[0] == "heiner_abm":
                    todo += _resolve(PACKAGE_DIR, 1, ".".join(node.module.split(".")[1:]) or None, names)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "heiner_abm":
                        todo += _resolve(PACKAGE_DIR, 1, ".".join(a.name.split(".")[1:]) or None, [])
    rel = sorted(os.path.relpath(p, PACKAGE_DIR).replace(os.sep, "/") for p in seen)
    return [r for r in rel if r not in excl and not r.endswith("/__init__.py") or r in files]
