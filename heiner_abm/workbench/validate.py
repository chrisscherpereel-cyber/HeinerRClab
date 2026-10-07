"""Validity checks of an experiment specification, per workspace step (errors block a run; warnings do not)."""
from __future__ import annotations

from typing import List

from .environments import ENVIRONMENTS
from .spec import MODES, ExperimentSpec, Issue

RESEARCH_MIN_REPLICATIONS = 8
PREVIEW = dict(replications=3, periods=300, train_replications=1)


def validate(spec: ExperimentSpec) -> List[Issue]:
    out: List[Issue] = []
    add = lambda step, level, msg: out.append(Issue(step, level, msg))
    if spec.mode not in MODES:
        add("question", "error", f"Unknown mode {spec.mode!r}.")
    q = spec.question
    if not q.text.strip():
        add("question", "error", "State the research question.")
    env = ENVIRONMENTS.get(spec.environment)
    if env is None:
        add("environment", "error", f"Unknown environment {spec.environment!r}.")
        return out
    if spec.candidate not in {c.key for c in env.candidates}:
        add("environment", "error", "Choose how candidates are generated.")
    keys = {p.key for p in env.policies}
    unknown = [p for p in spec.policies if p not in keys]
    if unknown:
        add("agents", "error", f"Not available in this environment: {', '.join(unknown)}.")
    if len([p for p in spec.policies if p in keys]) < 2:
        add("agents", "error", "Select at least two policies to compare.")
    a, b = q.comparison
    if not a or not b or a == b:
        add("question", "error", "Name the comparison: a treatment policy and a different reference policy.")
    else:
        for p in (a, b):
            if p not in spec.policies:
                add("agents", "error", f"The comparison uses '{p}', which is not among the selected policies.")
            elif p in keys and env.policy(p).kind == "benchmark":
                add("question", "error", f"'{env.policy(p).label}' is a researcher benchmark with information the "
                                         "agents do not have; it is shown beside the comparison, never in it.")
        kinds = {env.policy(p).kind for p in (a, b) if p in keys}
        if "complete" in kinds and kinds != {"complete"}:
            add("question", "warning", "The comparison mixes a complete policy with a selection policy: they differ "
                                       "in candidate generation as well as in selection.")
    if q.primary_outcome not in {o.key for o in env.outcomes}:
        add("question", "error", "Choose a primary outcome defined for this environment.")
    elif env.outcome(q.primary_outcome).higher_better is None:
        add("question", "warning", "The primary outcome is descriptive (no better direction); state how to read it.")
    d = spec.design
    if d.replications < 2:
        add("design", "error", "At least two replications are needed for an uncertainty interval.")
    elif spec.mode == "research" and d.replications < RESEARCH_MIN_REPLICATIONS:
        add("design", "warning", f"Research runs should use at least {RESEARCH_MIN_REPLICATIONS} replications per "
                                 "condition.")
    if d.periods < 50:
        add("design", "error", "Use at least 50 periods.")
    if not 0 <= d.burn_in < d.periods - 10:
        add("design", "error", "The burn-in must be non-negative and end well before the last period.")
    if d.tuning not in ("none", "grid"):
        add("design", "error", "Tuning must be 'none' or 'grid'.")
    if d.tuning == "grid" and d.train_replications < 1:
        add("design", "error", "Grid tuning needs at least one training replication.")
    tr = d.treatment
    if tr.control:
        try:
            c = env.control(tr.control)
        except KeyError:
            add("design", "error", f"Unknown treatment setting {tr.control!r}.")
        else:
            if not c.treatable:
                add("design", "error", f"'{c.label}' cannot be a treatment.")
            if len(tr.levels) < 2:
                add("design", "error", "A treatment needs at least two levels.")
            if len(set(map(str, tr.levels))) != len(tr.levels):
                add("design", "error", "Treatment levels must differ.")
            if c.kind in ("float", "int") and c.lo is not None:
                bad = [v for v in tr.levels if not (c.lo <= float(v) <= c.hi)]
                if bad:
                    add("design", "error", f"Levels outside [{c.lo}, {c.hi}]: {bad}.")
    for o in d.outcomes:
        if o not in {x.key for x in env.outcomes}:
            add("design", "warning", f"Secondary outcome {o!r} is not defined here and will be skipped.")
    for c in env.controls:
        v = spec.env_params.get(c.key)
        if v is not None and c.kind in ("float", "int") and c.lo is not None and not (c.lo <= float(v) <= c.hi):
            add("environment", "error", f"{c.label}: {v} lies outside [{c.lo}, {c.hi}].")
    for step, level, msg in env.validate(spec):
        add(step, level, msg)
    out += _information(spec, env)
    return out


def _information(spec, env) -> List[Issue]:
    """Information compatibility of each selected agent (heiner_abm.information)."""
    from ..information import AGENT_BY_KEY, InfoSpec, compatibility
    out = []
    if env.key != "market":
        return out
    info = InfoSpec(**env.information(spec))
    for e in info.validate():
        out.append(Issue("information", "error", e))
    agent_key = {"always": "Always", "default": "Never", "band": "Large", "gate_gain": "Adaptive",
                 "gate_lcb": "Adaptive:lcb", "gate_explore": "Adaptive:explore"}
    for p in spec.policies:
        k = agent_key.get(p)
        a = AGENT_BY_KEY.get(("market", k)) if k else None
        if a is None:
            continue
        c = compatibility(a, info, "market")
        if c.status == "unsupported":
            out.append(Issue("information", "error", f"{env.policy(p).label}: {'; '.join(c.reasons)}"))
    rule = AGENT_BY_KEY.get(("market", spec.candidate))
    if rule is not None:
        c = compatibility(rule, info, "market")
        if c.status == "unsupported":
            out.append(Issue("information", "error", f"Production rule: {'; '.join(c.reasons)}"))
    return out


def preview_spec(spec: ExperimentSpec) -> ExperimentSpec:
    """A reduced copy for a quick preview: fewer replications and periods (marked as a preview in its provenance)."""
    s = spec.copy()
    d = s.design
    d.replications = min(d.replications, PREVIEW["replications"])
    d.periods = min(d.periods, PREVIEW["periods"])
    d.burn_in = min(d.burn_in, d.periods // 5)
    d.train_replications = min(d.train_replications, PREVIEW["train_replications"])
    d.n_boot = min(d.n_boot, 300)
    return s
