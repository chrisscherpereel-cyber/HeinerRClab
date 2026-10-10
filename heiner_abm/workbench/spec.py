"""The experiment specification: one object that every page edits and every run executes.

An ExperimentSpec holds the research question, the environment and its uncertainty settings, the candidate-generation
rule, the policies compared, the information treatment and the design (treatments, replications, tuning, outcomes).
The interface, the command line (python -m heiner_abm.workbench) and saved experiments all use this structure, so the
same specification gives the same run wherever it is executed.

Four elements are kept apart (see the Experiment page):
    environment          what the world does (market, inventory, learning task, investment, NK landscape)
    candidate generation how the alternative to the default is produced (best reply, forecast target, local search)
    selection policy     whether to act on the candidate (always, inaction band, reliability gate, aspiration)
    information          what is observed and when feedback arrives (noise, delay, missing observations)
Complete policies that do not split this way (Bayesian change detection, robust ordering, annealing search) are listed
separately with their assumptions.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SPEC_VERSION = "1.0"
STEPS = ("question", "environment", "agents", "information", "design", "run")
STEP_LABELS = {"question": "1. Question", "environment": "2. Environment", "agents": "3. Agents",
               "information": "4. Information", "design": "5. Design", "run": "6. Run"}
UNCERTAINTY_GROUPS = ("outcome", "observation", "change", "model", "complexity")
GROUP_LABELS = {"outcome": "Outcome variation", "observation": "Observation quality",
                "change": "Environmental change", "model": "Model knowledge", "complexity": "Decision complexity"}
MODES = ("explore", "research")


@dataclass
class Question:
    text: str = ""
    preset: str = "custom"                       # key of the preset it was loaded from, or "custom"
    primary_outcome: str = ""
    comparison: Tuple[str, str] = ("", "")       # (treatment policy, reference policy)
    smallest_effect: float = 0.0                 # smallest effect of practical interest (outcome units)
    rationale: str = ""


@dataclass
class Treatment:
    """One manipulated setting and its levels (empty: a single condition)."""
    control: str = ""
    levels: Tuple[Any, ...] = ()


@dataclass
class Design:
    """The design, including which random streams serve which purpose.

    Replication counts name a seed namespace each (heiner_abm.workbench.provenance): train_replications draws from
    `training` and is used for tuning only, `replications` draws from `evaluation` and is what gets reported,
    pilot_replications draws from `pilot` and never enters a reported result, and validation_replications draws from
    `validation` to confirm a result on environments it was not estimated on. The four blocks cannot overlap.
    """
    replications: int = 10                       # independent environments (paths, markets or landscapes) per cell
    periods: int = 600
    burn_in: int = 50
    seed: int = 1
    tuning: str = "none"                         # "none" (documented fixed parameters) or "grid" (training seeds)
    train_replications: int = 3
    pilot_replications: int = 0                  # 0 = no pilot pass
    validation_replications: int = 0             # 0 = no validation pass
    trace_retention: str = "illustrative"        # "summary", "illustrative" or "all" (provenance.RETENTION)
    treatment: Treatment = field(default_factory=Treatment)
    outcomes: Tuple[str, ...] = ()               # secondary outcomes reported beside the primary one
    n_boot: int = 1000


@dataclass
class ExperimentSpec:
    name: str = "Untitled experiment"
    mode: str = "explore"
    question: Question = field(default_factory=Question)
    environment: str = "inventory"
    env_params: Dict[str, Any] = field(default_factory=dict)
    candidate: str = ""
    candidate_params: Dict[str, Any] = field(default_factory=dict)
    policies: Tuple[str, ...] = ()
    policy_params: Dict[str, Any] = field(default_factory=dict)
    information: Dict[str, Any] = field(default_factory=dict)
    design: Design = field(default_factory=Design)
    version: str = SPEC_VERSION

    # ---------------------------------------------------------------------------------------- serialization
    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=_jsonable))

    def to_json(self, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentSpec":
        d = copy.deepcopy(d)
        q = d.pop("question", {}) or {}
        q["comparison"] = tuple(q.get("comparison", ("", "")))
        des = d.pop("design", {}) or {}
        tr = des.pop("treatment", {}) or {}
        tr["levels"] = tuple(tr.get("levels", ()))
        des["outcomes"] = tuple(des.get("outcomes", ()))
        d["policies"] = tuple(d.get("policies", ()))
        return cls(question=Question(**q), design=Design(treatment=Treatment(**tr), **des), **d)

    @classmethod
    def from_json(cls, s: str) -> "ExperimentSpec":
        return cls.from_dict(json.loads(s))

    def copy(self) -> "ExperimentSpec":
        return ExperimentSpec.from_dict(self.to_dict())

    def content(self) -> Dict[str, Any]:
        """What determines the results: everything except the name and the interface mode."""
        d = self.to_dict()
        d.pop("name", None)
        d.pop("mode", None)
        d["question"] = {k: d["question"][k] for k in ("primary_outcome", "comparison")}
        return d

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.content(), sort_keys=True).encode()).hexdigest()[:16]


def _jsonable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    return str(x)


def changed_fields(a: ExperimentSpec, b: ExperimentSpec) -> List[str]:
    """Dotted paths whose values differ in what determines results (for 'results outdated' notices)."""
    out: List[str] = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
        elif x != y:
            out.append(path)

    walk(a.content(), b.content(), "")
    return out


@dataclass
class Issue:
    step: str
    level: str                     # "error" (blocks a run) or "warning"
    message: str


def step_status(issues: List[Issue]) -> Dict[str, str]:
    """'invalid', 'incomplete' (warnings) or 'ok' per step."""
    st = {s: "ok" for s in STEPS}
    for i in issues:
        if i.level == "error":
            st[i.step] = "invalid"
        elif st[i.step] == "ok":
            st[i.step] = "incomplete"
    return st
