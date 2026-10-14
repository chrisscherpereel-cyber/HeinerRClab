"""What a reader must be told BEFORE a research run, and what a result is allowed to claim afterwards.

A run that is cheap to start is easy to misread. By the time results are on screen the decisions that determine
what they mean - which quantity is being estimated, who pays which cost, what each agent was allowed to see,
whether the parameters were tuned on the same data they are reported on - have already been made, and nothing
on a results page reliably reminds anybody of them. This module computes those seven things from the
specification itself, so they can be shown before the run rather than reconstructed afterwards.

Nothing here changes what a run does. It is a reading of a specification, and if it cannot determine an item it
says so rather than filling in a plausible default: an unanswered question is information, a guessed answer is
not.

    1  PRIMARY ESTIMAND AND OUTCOME   what is being estimated, on what scale, against which comparison
    2  COST SEMANTICS                 which economic frictions are charged, at what rate, and to whom
    3  AGENT INFORMATION RESTRICTIONS what each policy may read, and whether any is privileged
    4  TRAINING AND EVALUATION        which seed blocks serve which purpose, and whether they are disjoint
    5  POLICY TUNING STATUS           what was tuned, on which data, under what budget
    6  REPLICATION REQUIREMENTS       whether a registered finding this run touches needs re-running
    7  CALIBRATION LIMITATIONS        where a reported interval or bound is not known to mean what it says

The statuses below are the other half. A result is not a finding until somebody says which kind of claim it is,
and the four words are chosen so that the weakest ones are available and not embarrassing to use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ================================================================================================ statuses
@dataclass(frozen=True)
class Status:
    key: str
    label: str
    meaning: str
    may_claim: str
    may_not_claim: str


STATUSES: Tuple[Status, ...] = (
    Status(
        "exploratory", "Exploratory",
        "The analysis was chosen after seeing these data, or the data were used to pick what to look at. "
        "Every comparison here is a hypothesis for a later test, not a test.",
        "that a pattern is present in these runs, and that it is worth testing",
        "a tested effect, a p-value that means what it usually means, or support for a hypothesis"),
    Status(
        "confirmatory", "Confirmatory",
        "The quantity, the comparison and the analysis were fixed in a registered plan before these data were "
        "generated, and the plan hash matches the code that produced them.",
        "that a prespecified prediction was or was not borne out, at the stated interval",
        "anything the plan did not name; a post-hoc subgroup is exploratory even inside a confirmatory run"),
    Status(
        "unsupported", "Not supported",
        "A prespecified prediction was tested and the result did not bear it out. This is a result, not a "
        "failure of the run, and it is reported with the same prominence as a supported one.",
        "that the prediction was not borne out, with the interval that shows how precisely",
        "that the opposite is true; failing to support a claim is not evidence for its negation"),
    Status(
        "requires_replication", "Requires replication",
        "Code that this result depends on has changed since it was produced, or the result was produced under "
        "a plan hash that no longer matches. The number may still be right; nobody has checked.",
        "that the result existed under the earlier code, naming the fingerprint it was produced under",
        "the result as current, or any comparison between it and a result produced after the change"),
    Status(
        "descriptive", "Descriptive",
        "A summary of what happened in these runs with no comparison claimed - a distribution, a trace, a "
        "count. Useful, and not a test of anything.",
        "what the runs did",
        "that any difference between conditions is an effect"),
)
STATUS_BY_KEY: Dict[str, Status] = {s.key: s for s in STATUSES}
STATUS_ORDER: Tuple[str, ...] = tuple(s.key for s in STATUSES)


def status(key: str) -> Status:
    if key not in STATUS_BY_KEY:
        raise KeyError(f"unknown result status {key!r}; expected one of {STATUS_ORDER}")
    return STATUS_BY_KEY[key]


def status_table() -> List[Dict[str, str]]:
    return [{"status": s.key, "label": s.label, "meaning": s.meaning,
             "may_claim": s.may_claim, "may_not_claim": s.may_not_claim} for s in STATUSES]


# ================================================================================================ disclosure
@dataclass
class Item:
    """One thing a reader must be told. `level` is how loudly: ok, caution, or blocking."""
    key: str
    title: str
    level: str                       # "ok" | "caution" | "unknown" | "blocking"
    summary: str
    detail: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "title": self.title, "level": self.level, "summary": self.summary,
                "detail": list(self.detail)}


ITEM_KEYS: Tuple[str, ...] = ("estimand", "costs", "information", "separation", "tuning", "replication",
                              "calibration")
LEVELS: Tuple[str, ...] = ("ok", "caution", "unknown", "blocking")


def _env_for(spec):
    from .environments import ENVIRONMENTS
    return ENVIRONMENTS.get(spec.environment)


def estimand_item(spec, env) -> Item:
    q = spec.question
    primary = q.primary_outcome or (env.primary if env else "")
    if not primary:
        return Item("estimand", "Primary estimand and outcome", "blocking",
                    "No primary outcome is set and the environment names no default, so there is nothing to "
                    "estimate.", ["Set the outcome in the research-question stage before running."])
    defaulted = not q.primary_outcome
    label, better = primary, None
    if env:
        try:
            o = env.outcome(primary)
            label, better = o.label, o.higher_better
        except StopIteration:
            pass
    a, b = q.comparison
    detail = [f"Outcome: {label} ({primary})."]
    detail.append("Higher is better." if better else ("Lower is better." if better is False
                                                      else "Descriptive: no direction is claimed."))
    if a and b:
        detail.append(f"Estimand: the difference in {label} between {a} and {b}, averaged over the "
                      f"replications of each cell.")
        level, summary = "ok", f"Difference in {label}: {a} minus {b}."
    else:
        detail.append("No comparison is named, so the run reports levels rather than an effect. "
                      "A difference read off two levels afterwards is exploratory.")
        level, summary = "caution", f"{label}, reported as levels (no comparison named)."
    if spec.design.treatment.control:
        detail.append(f"Manipulated: {spec.design.treatment.control} over "
                      f"{list(spec.design.treatment.levels)}.")
    if defaulted:
        level = "caution"
        summary = f"Defaulting to {label}: no outcome was chosen."
        detail.append("The outcome was NOT chosen for this question; it is the environment's default. A "
                      "reader will assume it was selected on purpose.")
    if q.smallest_effect:
        detail.append(f"Smallest effect of interest: {q.smallest_effect} {primary} units.")
    else:
        detail.append("No smallest effect of interest was declared, so 'no difference' cannot be "
                      "distinguished from 'too small a sample to tell'.")
    return Item("estimand", "Primary estimand and outcome", level, summary, detail)


def cost_item(spec, env) -> Item:
    if env is None:
        return Item("costs", "Cost semantics", "unknown", "No environment is selected.", [])
    try:
        defs = env.cost_definitions(spec, env.params(spec))
    except Exception as exc:                                  # a broken environment must not hide the warning
        return Item("costs", "Cost semantics", "unknown",
                    f"The environment could not report its cost definitions ({type(exc).__name__}).", [])
    live = [d for d in defs if d.get("in_force", True)]
    detail = []
    for d in defs:
        mark = "charged" if d.get("in_force", True) else "not in force"
        detail.append(f"{d.get('name', d.get('parameter', '?'))} [{mark}]: {d.get('rate', '?')} "
                      f"{d.get('units', '')}, when {d.get('charged_when', 'unspecified')}.")
    detail.append("Every policy in the comparison faces the same charges; the audit checks that rather than "
                  "assuming it.")
    if not live:
        return Item("costs", "Cost semantics", "caution",
                    "No economic friction is charged: every policy adapts for free.",
                    detail + ["With adaptation free, a policy that changes every period is never penalised "
                              "for it, and any advantage of inaction is understated. Several results in this "
                              "repository move substantially once a charge is switched on."])
    return Item("costs", "Cost semantics", "ok",
                f"{len(live)} of {len(defs)} charge(s) in force: "
                + ", ".join(str(d.get("name", d.get("parameter", "?"))) for d in live), detail)


def information_item(spec, env) -> Item:
    if env is None:
        return Item("information", "Agent information restrictions", "unknown", "No environment is selected.",
                    [])
    detail, privileged = [], []
    for key in spec.policies:
        try:
            p = env.policy(key)
        except StopIteration:
            detail.append(f"{key}: NOT DECLARED by this environment.")
            privileged.append(key)
            continue
        detail.append(f"{p.label}: reads {p.uses}." + (f" {p.assumptions}" if p.assumptions else ""))
        if p.kind == "benchmark":
            privileged.append(key)
    info = spec.information or {}
    if info:
        detail.append("Information treatment: " + ", ".join(f"{k}={v}" for k, v in sorted(info.items())))
    if privileged:
        return Item("information", "Agent information restrictions", "caution",
                    f"{len(spec.policies)} policies; {len(privileged)} privileged or undeclared: "
                    + ", ".join(privileged),
                    detail + ["A benchmark that is given the truth belongs beside the ranking, not in it. "
                              "Check that it is reported as a reference."])
    return Item("information", "Agent information restrictions", "ok",
                f"{len(spec.policies)} policies, all on declared information.", detail)


def separation_item(spec, env=None) -> Item:
    from . import provenance
    d = spec.design
    counts = {"training": d.train_replications, "evaluation": d.replications,
              "pilot": d.pilot_replications, "validation": d.validation_replications}
    if d.tuning == "none":
        counts["training"] = 0                 # an untuned run never draws from the training block
    used = {k: v for k, v in counts.items() if v}
    clash = provenance.disjoint(d.seed, counts)
    detail = [f"{k}: {v} replications from the {k!r} seed block" for k, v in used.items()]
    if d.tuning == "none" and d.train_replications:
        detail.append(f"({d.train_replications} training replications are configured but unused: "
                      f"nothing is being tuned.)")
    if clash:
        return Item("separation", "Training and evaluation separation", "blocking",
                    f"Seed blocks overlap: {clash}.",
                    detail + ["A policy tuned on a seed it is then scored on reports its own fit."])
    if d.tuning != "none" and not d.train_replications:
        return Item("separation", "Training and evaluation separation", "blocking",
                    "Tuning is on, but no training replications are allocated, so tuning would use the "
                    "evaluation seeds.", detail)
    if d.tuning == "none":
        return Item("separation", "Training and evaluation separation", "ok",
                    "Nothing is tuned, so there is nothing to separate; evaluation seeds only.", detail)
    return Item("separation", "Training and evaluation separation", "ok",
                f"Disjoint seed blocks: {', '.join(f'{k} ({v})' for k, v in used.items())}.", detail)


def tuning_item(spec, env) -> Item:
    d = spec.design
    if d.tuning == "none":
        note = getattr(env, "tuning_note", "") if env else ""
        return Item("tuning", "Policy tuning status", "caution",
                    "No tuning: documented fixed parameters are used for every policy.",
                    [note or "", "A policy whose published settings suit this environment better than "
                                 "another's has an advantage that is not a property of the rule."])
    steps = []
    if env is not None:
        try:
            steps = list(env.tuning_plan(spec, env.params(spec)))
        except Exception:
            steps = []
    if not steps:
        return Item("tuning", "Policy tuning status", "caution",
                    "Tuning is requested but this environment declares no tuning steps.",
                    ["Nothing will be searched; the fixed parameters will be used."])
    detail = [f"{s.name}: tunes {s.policy} over {sorted(s.space)}" for s in steps]
    tuned = {s.policy for s in steps}
    untuned = [p for p in spec.policies if p not in tuned]
    detail.append(f"Budget rule: {getattr(d, 'budget_rule', 'equal_candidates')}, "
                  f"base {getattr(d, 'tuning_base', 4)} candidates per parameter.")
    detail.append(f"Training replications: {d.train_replications} (training seed block).")
    if untuned:
        detail.append(f"NOT tuned: {', '.join(untuned)}. A tuned policy compared with an untuned one is "
                      f"partly a comparison of effort.")
        return Item("tuning", "Policy tuning status", "caution",
                    f"{len(tuned)} of {len(spec.policies)} policies tuned; {len(untuned)} not.", detail)
    return Item("tuning", "Policy tuning status", "ok",
                f"All {len(tuned)} policies tuned on the training block.", detail)


def replication_item(spec, env=None) -> Item:
    try:
        from .. import registered
    except Exception:
        return Item("replication", "Replication requirements", "unknown",
                    "The registered results could not be read.", [])
    pending = list(getattr(registered, "REPLICATION_REQUIRED", ()) or ())
    if pending:
        return Item("replication", "Replication requirements", "caution",
                    f"{len(pending)} registered finding(s) are marked as requiring replication.",
                    [str(p) for p in pending]
                    + ["These are not this run. They are flagged because a reader comparing this run with a "
                       "registered number should know that number is awaiting a re-run."])
    return Item("replication", "Replication requirements", "ok",
                "No registered finding is currently marked as requiring replication.",
                ["This run is not itself registered: it produces a result, not a registered finding."])


# Known limits on what a reported interval or bound means. Stated here rather than discovered by a reader,
# because every one of them was found the hard way somewhere in this repository.
CALIBRATION_NOTES: Tuple[Tuple[str, str], ...] = (
    ("bootstrap", "Intervals resample whole replications, so they cover sampling variation in the environments "
                  "drawn. They do not cover uncertainty in a tuned hyperparameter, which is treated as fixed "
                  "once chosen."),
    ("gate_bound", "The reliability gate reports a lower bound on its advantage. Measured coverage of the "
                   "nominal 90% bound is 45-62% across feedback regimes (docs/gate_validation.md). It is not a "
                   "confidence bound and is not described as one."),
    ("multiplicity", "A run that reports several outcomes and several treatment levels tests many hypotheses. "
                     "Only a prespecified primary comparison carries its nominal error rate."),
    ("participant_tests", "For human-study models, a t-test over per-participant estimates from a pooled "
                          "empirical-Bayes fit is anti-conservative: measured false-positive rate 0.50-0.75 "
                          "against a nominal 0.05 (docs/human_learning.md)."),
)


def calibration_item(spec, env=None) -> Item:
    notes = [n for _, n in CALIBRATION_NOTES]
    if spec.design.n_boot and spec.design.n_boot < 500:
        notes.append(f"n_boot = {spec.design.n_boot} is small; interval endpoints will be noisy.")
    if spec.design.replications < 8:
        notes.append(f"{spec.design.replications} evaluation replications is below the research minimum of 8; "
                     f"any interval will be wide and its coverage is not checked at this size.")
    return Item("calibration", "Calibration limitations", "caution",
                f"{len(notes)} limitation(s) apply to how the reported intervals should be read.", notes)


BUILDERS = {"estimand": estimand_item, "costs": cost_item, "information": information_item,
            "separation": separation_item, "tuning": tuning_item, "replication": replication_item,
            "calibration": calibration_item}


def disclosure(spec) -> List[Item]:
    """The seven items, in the order a reader needs them."""
    env = _env_for(spec)
    return [BUILDERS[k](spec, env) for k in ITEM_KEYS]


def blocking(items: Sequence[Item]) -> List[Item]:
    return [i for i in items if i.level == "blocking"]


def to_records(items: Sequence[Item]) -> List[Dict[str, Any]]:
    return [i.to_dict() for i in items]


def summary_line(items: Sequence[Item]) -> str:
    n = {lv: sum(1 for i in items if i.level == lv) for lv in LEVELS}
    if n["blocking"]:
        return f"{n['blocking']} item(s) must be resolved before a research run."
    if n["caution"] or n["unknown"]:
        return f"{n['caution'] + n['unknown']} item(s) need a reader's attention; none blocks the run."
    return "All seven disclosure items are settled."
