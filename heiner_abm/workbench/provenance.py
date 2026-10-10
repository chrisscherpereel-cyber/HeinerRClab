"""Provenance: what code ran, in what software environment, on which seeds, and what it was allowed to see.

An experiment is auditable when a reader can tell, from the stored run alone, (a) exactly which source code produced
it, (b) what the surrounding software was, (c) which random streams served which purpose, (d) what each agent knew at
the moment it decided, and (e) which recorded fields the agents never had. This module supplies those five things to
the workbench; it holds no simulation logic.

Code fingerprint
    code_fingerprint() hashes the transitive closure of first-party imports reachable from the workbench's entry
    points, found by parsing the source rather than by importing it, so imports inside functions count too. The
    previous fingerprint hashed a hand-written module list, which omitted the agent code in heiner_abm.agents and
    heiner_abm.arena and the workbench's own analysis module: a change to any of them left stored runs looking
    current. Any change to any reachable module now changes the fingerprint, which changes every run key, which
    invalidates the cached runs.

Seed namespaces
    Four disjoint blocks, so a seed used for tuning can never reappear as an evaluation seed:
        training    base*1000 + 100_000_000 + r   tuning only
        pilot       base*1000 + 200_000_000 + r   pilot estimates, never reported as results
        evaluation  base*1000 + 300_000_000 + r   the reported results
        validation  base*1000 + 400_000_000 + r   confirmation of an evaluation result on fresh environments
    The training and evaluation offsets are the ones the workbench has always used, so existing runs reproduce; pilot
    and validation are new. disjoint() states the conditions under which the blocks cannot collide and is checked
    before a run starts.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
import os
import platform
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PACKAGE = "heiner_abm"

# Entry points of the workbench. Everything these reach, directly or indirectly, is part of the fingerprint.
FINGERPRINT_ROOTS: Tuple[str, ...] = (
    "heiner_abm.workbench.environments",     # the environment adapters and their decision traces
    "heiner_abm.workbench.execution",        # seeds, scheduling, provenance
    "heiner_abm.workbench.spec",             # what a specification means
    "heiner_abm.workbench.validate",         # which specifications are allowed to run
    "heiner_abm.workbench.analysis",         # the summaries and comparisons reported from a run
    "heiner_abm.workbench.presets",          # the shipped experiments
    "heiner_abm.workbench.provenance",       # this module: seeds and the manifest are part of the result
)

FINGERPRINT_DIGITS = 12

# ------------------------------------------------------------------------------------------------ seed namespaces
SEED_NAMESPACES: Dict[str, int] = {
    "training": 100_000_000,
    "pilot": 200_000_000,
    "evaluation": 300_000_000,
    "validation": 400_000_000,
}
SEED_PURPOSE: Dict[str, str] = {
    "training": "Tuning only. Never reported as a result.",
    "pilot": "Pilot estimates: effect size, variance and workload. Never reported as a result.",
    "evaluation": "The reported results. Every policy in a cell sees these same seeds (common random numbers).",
    "validation": "Confirmation of an evaluation result on fresh environments, drawn after the result exists.",
}
SEED_STRIDE = 1000            # replications addressable per base seed before two base seeds could collide
SEED_BLOCK = 100_000_000      # distance between namespaces


def seeds(base_seed: int, namespace: str, n: int) -> List[int]:
    """The first n seeds of one namespace for this base seed."""
    if namespace not in SEED_NAMESPACES:
        raise ValueError("unknown seed namespace %r; expected one of %s" % (namespace, sorted(SEED_NAMESPACES)))
    if n < 0:
        raise ValueError("n must not be negative")
    return [int(base_seed) * SEED_STRIDE + SEED_NAMESPACES[namespace] + r for r in range(int(n))]


def disjoint(base_seed: int, counts: Dict[str, int]) -> Optional[str]:
    """None if the requested namespaces cannot overlap, else why they can.

    Two conditions. Within a namespace, a base seed addresses SEED_STRIDE replications, so more than that would run
    into the next base seed's block. Across namespaces, base_seed * SEED_STRIDE must stay below SEED_BLOCK, or one
    namespace's seeds reach into the next namespace's.
    """
    for name, n in counts.items():
        if name not in SEED_NAMESPACES:
            return "unknown seed namespace %r" % (name,)
        if n > SEED_STRIDE:
            return ("%d %s replications exceed the %d addressable per base seed; seeds would collide with base "
                    "seed %d" % (n, name, SEED_STRIDE, int(base_seed) + 1))
    if not 0 <= int(base_seed) * SEED_STRIDE < SEED_BLOCK:
        return ("base seed %s reaches outside the %d block each namespace owns; use a base seed below %d"
                % (base_seed, SEED_BLOCK, SEED_BLOCK // SEED_STRIDE))
    return None


def seed_plan(base_seed: int, counts: Dict[str, int]) -> Dict[str, Any]:
    """The record of which stream served which purpose, for the manifest."""
    plan: Dict[str, Any] = {"base_seed": int(base_seed), "stride": SEED_STRIDE, "block": SEED_BLOCK,
                            "namespaces": {}}
    for name in SEED_NAMESPACES:
        n = int(counts.get(name, 0))
        s = seeds(base_seed, name, n)
        plan["namespaces"][name] = {"offset": SEED_NAMESPACES[name], "purpose": SEED_PURPOSE[name], "count": n,
                                    "seeds": s, "first": s[0] if s else None, "last": s[-1] if s else None}
    plan["disjoint"] = disjoint(base_seed, {k: int(v) for k, v in counts.items()}) is None
    return plan


# ------------------------------------------------------------------------------------------------ code fingerprint
def _module_file(name: str) -> Optional[str]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return spec.origin


def _resolve(module: str, node: ast.AST) -> List[str]:
    """First-party module names an import statement refers to, relative imports resolved against `module`."""
    out: List[str] = []
    if isinstance(node, ast.Import):
        out += [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            parts = module.split(".")
            # a module's own package is parts[:-1]; each further level strips one more package
            base = parts[:max(0, len(parts) - node.level)]
            prefix = ".".join(base + ([node.module] if node.module else []))
        else:
            prefix = node.module or ""
        if prefix:
            out.append(prefix)
            # `from package import submodule` names a module, not an attribute, when that submodule exists
            out += ["%s.%s" % (prefix, a.name) for a in node.names if a.name != "*"]
    return [n for n in out if n == PACKAGE or n.startswith(PACKAGE + ".")]


def _closure(roots: Sequence[str]) -> List[str]:
    """Every first-party module reachable from the roots, by parsing source (so imports inside functions count)."""
    seen: Dict[str, str] = {}
    queue = list(roots)
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        path = _module_file(name)
        if path is None:
            continue
        seen[name] = path
        try:
            with open(path, "rb") as fh:
                tree = ast.parse(fh.read(), filename=path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                queue += [n for n in _resolve(name, node) if n not in seen]
    return sorted(seen)


def _hash_source(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


@functools.lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """Hash of every first-party module the workbench can reach. A change to any of them changes this."""
    h = hashlib.sha256()
    for name in _closure(FINGERPRINT_ROOTS):
        path = _module_file(name)
        if path:
            h.update(name.encode() + b"\0" + _hash_source(path).encode() + b"\0")
    return h.hexdigest()[:FINGERPRINT_DIGITS]


def code_fingerprint_detail() -> Dict[str, Any]:
    """The fingerprint with the module list behind it, so a reader can see what it covers."""
    modules = {}
    for name in _closure(FINGERPRINT_ROOTS):
        path = _module_file(name)
        if path:
            modules[name] = _hash_source(path)[:FINGERPRINT_DIGITS]
    return {"fingerprint": code_fingerprint(), "digits": FINGERPRINT_DIGITS, "roots": list(FINGERPRINT_ROOTS),
            "n_modules": len(modules), "modules": modules,
            "method": "sha256 over the sorted first-party modules reachable from the roots by static import "
                      "analysis, each module hashed with line endings normalized"}


# ------------------------------------------------------------------------------------------------ software
DEPENDENCIES: Tuple[str, ...] = ("numpy", "pandas", "scipy", "streamlit", "plotly", "statsmodels", "pyarrow",
                                 "openpyxl", "xlrd", "pytest")


def _version(dist: str) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(dist)
    except Exception:
        mod = sys.modules.get(dist)
        return getattr(mod, "__version__", None) if mod is not None else None


def software_environment() -> Dict[str, Any]:
    """Dependency versions and the interpreter, recorded with the run.

    Absolute paths are deliberately excluded: a stored run may be published, and an interpreter path carries the
    user's account name. Versions and the platform string are what a reader needs in order to reproduce.
    """
    env: Dict[str, Any] = {"python": platform.python_version(),
                           "python_implementation": platform.python_implementation(),
                           "platform": platform.platform(), "machine": platform.machine(),
                           "packages": {name: _version(name) for name in DEPENDENCIES}}
    env["threading"] = {v: os.environ.get(v) for v in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    return env


# ------------------------------------------------------------------------------------------------ decision logs
RESEARCHER_PREFIX = "researcher_"

# Role of each decision-log column. A trace answers, for one period: what did the agent see before deciding, what did
# it choose, what happened afterwards, and when (if ever) did it learn the consequence.
DECISION_LOG_ROLES: Dict[str, str] = {
    # identifies the decision
    "period": "index", "policy": "index", "replication": "index", "seed": "index", "level": "index",
    # what the agent had before it decided
    "observed": "predecision", "observed_price": "predecision", "observed_candidate": "predecision",
    "signal": "predecision", "bin": "predecision", "configuration": "predecision",
    "belief_sd": "predecision", "belief_cost": "predecision", "belief_best_reply": "predecision",
    "default_order": "predecision", "default_action": "predecision", "current_output": "predecision",
    "proposed": "predecision", "opportunity": "predecision", "predicted_gain": "predecision",
    "lower_bound": "predecision",
    # what it did
    "chosen": "action", "acted": "action", "reason": "action",
    # what happened because of it
    "realized_payoff": "postdecision", "realized_cost": "postdecision",
    # what it was later told, and when
    "feedback": "feedback", "feedback_origin": "feedback", "feedback_matured_period": "feedback",
    "feedback_release_period": "feedback",
}
ROLES: Tuple[str, ...] = ("index", "predecision", "action", "postdecision", "feedback", "researcher_only")
ROLE_NOTES: Dict[str, str] = {
    "index": "Identifies the decision; not information.",
    "predecision": "Available to the agent at the moment it chose. Everything it could have used.",
    "action": "The action selected and the stated reason for selecting it.",
    "postdecision": "Consequences of the action, known only after it was taken.",
    "feedback": "What the agent was later told, where it came from, when the outcome matured and when it was "
                "released to the agent. A release period of -1 means it was never released.",
    "researcher_only": "Recorded for analysis. Never available to any agent at any time.",
}

# Where a feedback value comes from. Stated per environment so that 'feedback' is never ambiguous between an agent's
# own realized outcome and something it was shown about the world or about others.
FEEDBACK_ORIGINS: Dict[str, str] = {
    "own_outcome": "The realized outcome of this agent's own action in this period.",
    "own_counterfactual": "A comparison between this agent's action and the alternative it did not take, formed from "
                          "quantities the agent itself observed.",
    "observed_world": "An observation of the environment that is not an outcome of this agent's action.",
    "none": "No feedback arises from this decision.",
}
NEVER_RELEASED = -1


def role_of(column: str) -> str:
    """The role of a decision-log column. Unknown columns are 'unclassified', which the tests reject."""
    if column.startswith(RESEARCHER_PREFIX):
        return "researcher_only"
    return DECISION_LOG_ROLES.get(column, "unclassified")


def decision_log_schema(columns: Iterable[str]) -> Dict[str, Any]:
    """Columns grouped by role, for the manifest and the export."""
    by_role: Dict[str, List[str]] = {r: [] for r in ROLES}
    unclassified: List[str] = []
    for c in columns:
        r = role_of(c)
        (by_role[r] if r in by_role else unclassified).append(c)
    return {"roles": by_role, "unclassified": unclassified, "notes": ROLE_NOTES,
            "feedback_origins": FEEDBACK_ORIGINS, "never_released": NEVER_RELEASED,
            "researcher_prefix": RESEARCHER_PREFIX}


def researcher_only_columns(columns: Iterable[str]) -> List[str]:
    return sorted(c for c in columns if c.startswith(RESEARCHER_PREFIX))


def check_decision_log(df) -> List[str]:
    """Problems with a decision log: unclassified columns, or feedback recorded without its timing. Empty is good."""
    cols = list(df.columns)
    problems = ["column %r has no declared role" % (c,) for c in cols if role_of(c) == "unclassified"]
    if "feedback" in cols:
        for required in ("feedback_origin", "feedback_matured_period", "feedback_release_period"):
            if required not in cols:
                problems.append("'feedback' is recorded without %r" % (required,))
    if "feedback_origin" in cols:
        unknown = sorted({str(v) for v in df["feedback_origin"].unique()} - set(FEEDBACK_ORIGINS))
        problems += ["feedback_origin %r is not a declared origin" % (v,) for v in unknown]
    return problems


# ------------------------------------------------------------------------------------------------ trace retention
RETENTION: Tuple[str, ...] = ("summary", "illustrative", "all")
RETENTION_LABELS: Dict[str, str] = {
    "summary": "Summary only: no per-period decision log is kept.",
    "illustrative": "One illustrative replication per condition (the first). The default.",
    "all": "Every replication of every condition. Large; stored column-wise and compressed.",
}


def keeps_trace(retention: str, replication: int) -> bool:
    """Whether the decision log of this replication is retained."""
    if retention not in RETENTION:
        raise ValueError("unknown trace retention %r; expected one of %s" % (retention, list(RETENTION)))
    if retention == "summary":
        return False
    if retention == "all":
        return True
    return replication == 0
