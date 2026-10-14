"""A frozen replication specification, and what its results are allowed to do to the claims.

PROPOSED EXTENSION. The studies in this laboratory were each run once, under their own frozen plan, and their
verdicts were registered. This specification runs them again on INDEPENDENT SEEDS and uses the second set of
verdicts to decide which theoretical claims are retained, narrowed or rejected.

================================================================================ why fresh seeds

Re-running a frozen plan on its own seeds reproduces its numbers exactly. That is worth doing - it is what the
integration audit checks - but it is REPRODUCTION, and reproduction cannot change the status of a claim,
because it contains no information the first run did not. A claim is only tested again by data the first test
did not see.

Every plan here therefore moves its seed by a fixed offset declared below and nothing else. The design, the
estimator, the decision rule and the analysis are untouched. That makes each study an independent draw from the
same specification, which is the narrow thing a replication can be, and it is the only thing that licenses
moving a claim.

================================================================================ why no new decision rules

Each study already carries its own prespecified decision rule - the `decision_rule` column of its verdict table,
fixed before its first run. Those rules are re-applied unchanged to the new data. Inventing a second set of
rules now, with the first results in view, would be choosing the test after seeing the answer.

The only rule frozen HERE is the one that maps a pair of verdicts onto a claim's fate, and it is mechanical:

    RETAINED             supported before and supported again, with a replication effect at least
                         RETAIN_FRACTION of the original magnitude
    NARROWED             supported before and supported again, but the effect is smaller than that; or
                         supported before and inconclusive now with the sign unchanged
    REJECTED             supported before, and not supported now
    REJECTION_CONFIRMED  not supported before, and not supported again
    OVERTURNED           not supported before, and supported now

OVERTURNED is listed because it can happen, not because it is expected. If it occurs it is a finding about
sampling variation in the first run at least as much as about the claim, and it is reported as needing a third
look rather than as a vindication.

================================================================================ what this cannot do

A single replication of a simulation study is a weaker instrument than it looks. Both runs share the generating
code, the environment definitions, the policy implementations and every modelling assumption. What differs is
the random draw. So a claim that survives here has survived SAMPLING variation and nothing else: it has not
been tested against a different operationalisation, a different environment family, or a different research
group. A claim rejected here was fragile to sampling alone, which is a stronger statement than survival.

None of this touches the human experiment, which has never been run on people, and nothing here is evidence
about human behaviour.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ================================================================================================ outcomes
OUTCOMES: Tuple[Tuple[str, str], ...] = (
    ("retained", "Supported before and again, at a comparable magnitude."),
    ("narrowed", "Supported before and again, but weaker; or supported before and inconclusive now with the "
                 "sign unchanged. The claim survives in a smaller form than it was stated in."),
    ("rejected", "Supported before, not supported on independent seeds. The original was fragile to sampling."),
    ("rejection_confirmed", "Not supported before, not supported again."),
    ("overturned", "Not supported before, supported now. Needs a third run before anybody relies on it."),
)
OUTCOME_KEYS: Tuple[str, ...] = tuple(k for k, _ in OUTCOMES)
OUTCOME_MEANING: Dict[str, str] = dict(OUTCOMES)

RETAIN_FRACTION = 0.5      # a replicated effect below this share of the original is a narrowing, not a retention
SEED_OFFSET = 770_000      # added to every study seed; the only thing that differs from the registered plans


@dataclass(frozen=True)
class Claim:
    """One theoretical claim, the study that tests it, and what it said the first time.

    `original_verdict` and `original_effect` are transcribed from the registered results and are checked
    against those files by a test, so this table cannot drift from what was actually found.
    """
    key: str
    statement: str
    study: str
    quantity: str
    original_verdict: str          # "supported" | "not supported"
    original_effect: Optional[float] = None
    effect_units: str = ""
    note: str = ""


CLAIMS: Tuple[Claim, ...] = (
    # ---------------------------------------------------------------- the reliability condition, head to head
    Claim("PR1", "In held-out mixed markets the reliability-condition agent earns more than each rival.",
          "tournament", "PR1 verdict of the registered tournament protocol", "not supported"),
    Claim("PR2", "The reliability-condition agent's advantage grows with environmental uncertainty.",
          "tournament", "PR2 verdict", "not supported"),
    Claim("PR3", "The reliability-condition agent adapts less often than the always-adjust rule.",
          "tournament", "PR3 verdict", "not supported"),
    Claim("PR4", "The reliability-condition agent's selectivity is what earns its advantage.",
          "tournament", "PR4 verdict", "not supported"),
    Claim("PR5", "The reliability-condition agent is robust across market protocols.",
          "tournament", "PR5 verdict", "not supported"),
    Claim("PR6", "The reliability-condition agent carries less downside risk than always adjusting.",
          "tournament", "PR6 verdict", "not supported"),

    # ---------------------------------------------------------------- learnability
    Claim("H1", "The confidence-sensitive gate's advantage over the better fixed rule falls as the "
                "learnability ratio R rises.",
          "learnability", "OLS slope of configuration-mean advantage on log10 R, 95% bootstrap CI below 0",
          "supported", -0.95, "net payoff per period per decade of R"),
    Claim("H2", "log10 R explains the gate's advantage beyond outcome volatility and observation noise.",
          "learnability", "gain in leave-one-out cross-validated R-squared from adding log10 R",
          "supported", None, "cross-validated R-squared"),

    # ---------------------------------------------------------------- is the index worth its name
    Claim("V1", "The learnability index adds predictive accuracy beyond the prespecified comparison model, on "
                "held-out configurations.",
          "validity", "difference in cross-validated R-squared, bootstrap lower limit above 0",
          "not supported", 0.0027, "cross-validated R-squared"),
    Claim("V2", "The index transfers to held-out process families.",
          "validity", "difference in cross-validated R-squared on unseen families", "not supported", -0.0349,
          "cross-validated R-squared"),

    # ---------------------------------------------------------------- the gates
    Claim("G1", "The confidence-sensitive gate's reported lower bound has its nominal 90% coverage.",
          "gate_validation", "measured coverage of the 90% bound across feedback regimes", "not supported",
          None, "coverage share",
          note="Rejected on the first run at 45-62%. The replication asks whether that was sampling."),

    # ---------------------------------------------------------------- the stress suite
    Claim("S1", "On the control process, the correctly specified model wins.",
          "stress", "rank of bayes_true by mean regret on the `base` family", "supported", 1.0, "rank"),
    Claim("S2", "Where distributional shape is the problem, shape-free window models win.",
          "stress", "rank of saa by mean regret on the `multimodal` family", "supported", 1.0, "rank"),
    Claim("S3", "The moment-ambiguity model, the only one able to price outcomes never observed, is the one "
                "suited to the outside-support family.",
          "stress", "rank of dro_moment by mean regret on the `outside_support` family", "not supported", 4.0,
          "rank",
          note="Contradicted on the first run: it finished behind both models that provably cannot represent "
               "the stress."),
    Claim("S4", "Knowing the true nuisance parameters beats having the right functional form.",
          "stress", "number of the five stress families on which bayes_true ranks first", "supported", 3.0,
          "families out of 5"),

    # ---------------------------------------------------------------- information
    Claim("I1", "Bought information and recorded unmet demand are substitutes.",
          "information", "value of information at unmet_access 0.0 minus at 1.0", "supported", 0.206,
          "net payoff per period"),
    Claim("I2", "Buying information pays only when the environment changes fast enough.",
          "information", "net of buying at the lowest hazard", "supported", -0.213, "net payoff per period",
          note="Supported in the sense that the prediction was of a NEGATIVE net at low hazard."),
)
CLAIM_BY_KEY: Dict[str, Claim] = {c.key: c for c in CLAIMS}
STUDIES: Tuple[str, ...] = ("tournament", "learnability", "validity", "gate_validation", "stress",
                            "information")


# ================================================================================================ the rule
def decide(original_verdict: str, replication_verdict: str, original_effect: Optional[float],
           replication_effect: Optional[float]) -> Tuple[str, str]:
    """Map a pair of verdicts onto a claim's fate. Mechanical, and frozen before the replication was run."""
    before = str(original_verdict).strip().lower().startswith("support")
    after = str(replication_verdict).strip().lower().startswith("support")
    if not before and not after:
        return "rejection_confirmed", "not supported on either set of seeds"
    if not before and after:
        return "overturned", ("supported on the new seeds having failed on the old; a third run is needed "
                              "before this is relied on")
    if before and not after:
        return "rejected", "supported on the original seeds and not on independent ones"
    if original_effect is None or replication_effect is None:
        return "retained", "supported on both sets of seeds"
    if abs(float(original_effect)) < 1e-12:
        return "retained", "supported on both sets of seeds"
    share = abs(float(replication_effect)) / abs(float(original_effect))
    if np.sign(float(replication_effect)) != np.sign(float(original_effect)):
        return "rejected", (f"the replicated effect has the opposite sign "
                            f"({replication_effect:+.4g} against {original_effect:+.4g})")
    if share < RETAIN_FRACTION:
        return "narrowed", (f"supported again, but at {share:.0%} of the original magnitude "
                            f"({replication_effect:+.4g} against {original_effect:+.4g})")
    return "retained", (f"supported again at {share:.0%} of the original magnitude "
                        f"({replication_effect:+.4g} against {original_effect:+.4g})")


# ================================================================================================ the plan
def adjudication_code_digest() -> str:
    """Digest of the claim table and the decision rule, so neither can change without invalidating the plan."""
    import inspect
    mod = sys.modules[__name__]
    src = inspect.getsource(mod.decide) + repr(CLAIMS) + repr(OUTCOMES) + repr(RETAIN_FRACTION)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class AdjudicationPlan:
    version: str = "1.0"
    seed_offset: int = SEED_OFFSET
    retain_fraction: float = RETAIN_FRACTION
    studies: Tuple[str, ...] = STUDIES
    claims: Tuple[str, ...] = tuple(c.key for c in CLAIMS)
    # study scales: the registered scale is kept wherever the run is affordable, and any reduction is declared
    stress_paths: int = 4
    learnability_scale: str = "registered"
    human_models_included: bool = False     # the human study has no data; its models are not adjudicated here

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "code": adjudication_code_digest(),
                           "claim_table": [asdict(c) for c in CLAIMS],
                           "outcomes": dict(OUTCOMES)},
                          sort_keys=True, indent=1, default=str)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def claim_table() -> pd.DataFrame:
    return pd.DataFrame([{**asdict(c), "outcome_if_repeats": "retained" if
                          str(c.original_verdict).startswith("support") else "rejection_confirmed"}
                         for c in CLAIMS])


def outcome_table() -> pd.DataFrame:
    return pd.DataFrame([{"outcome": k, "meaning": v} for k, v in OUTCOMES])


def verdict_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Apply the frozen rule to a set of replication records."""
    rows = []
    for r in records:
        c = CLAIM_BY_KEY[r["claim"]]
        outcome, why = decide(c.original_verdict, r["replication_verdict"], c.original_effect,
                              r.get("replication_effect"))
        rows.append({"claim": c.key, "study": c.study, "statement": c.statement,
                     "original_verdict": c.original_verdict, "original_effect": c.original_effect,
                     "replication_verdict": r["replication_verdict"],
                     "replication_effect": r.get("replication_effect"),
                     "outcome": outcome, "reason": why, "units": c.effect_units,
                     "detail": r.get("detail", "")})
    return pd.DataFrame(rows)




# ================================================================================================ the replication
# One runner per study. Each moves the seed by SEED_OFFSET and changes nothing else, then reads off the same
# quantity the original read, under the same rule. The rule is stated beside each extraction so that a reader
# can see it was not chosen after the number arrived.
def replicate_information(offset: int = SEED_OFFSET,
                          progress: Optional[Callable[[float, str], None]] = None) -> List[Dict[str, Any]]:
    from . import information_env as I
    plan = I.InfoPlan(seed=I.InfoPlan().seed + offset)
    frames = I.run_experiment(plan, progress=progress)
    cells = frames["cells"]
    voi = cells.groupby("unmet_access")["value_of_information"].mean()
    gap = float(voi.loc[min(voi.index)] - voi.loc[max(voi.index)])
    net = cells.groupby("hazard")["net_of_buying"].mean()
    low = float(net.loc[min(net.index)])
    return [
        dict(claim="I1", replication_verdict="supported" if gap > 0 else "not supported",
             replication_effect=gap,
             detail=f"value of information {voi.loc[min(voi.index)]:.3f} at no recorded unmet demand against "
                    f"{voi.loc[max(voi.index)]:.3f} at full recording; supported if the difference is positive",
             plan=plan.digest),
        dict(claim="I2", replication_verdict="supported" if low < 0 else "not supported",
             replication_effect=low,
             detail=f"net of buying at the lowest hazard {min(net.index)}: {low:+.3f}; supported if negative",
             plan=plan.digest),
    ]


def replicate_stress(offset: int = SEED_OFFSET,
                     progress: Optional[Callable[[float, str], None]] = None) -> List[Dict[str, Any]]:
    from . import stress as S
    plan = S.StressPlan(seed=S.StressPlan().seed + offset)
    frames = S.run_suite(plan, progress=progress)
    s = frames["summary"]

    def rank_of(family: str, policy: str) -> int:
        g = s[s["family"] == family].sort_values("regret").reset_index(drop=True)
        return int(g.index[g["policy"] == policy][0]) + 1

    def first_of(family: str) -> str:
        return str(s[s["family"] == family].sort_values("regret").iloc[0]["policy"])

    r_base = rank_of("base", "bayes_true")
    r_multi = rank_of("multimodal", "saa")
    r_moment = rank_of("outside_support", "dro_moment")
    r_emp = rank_of("outside_support", "dro_emp")
    r_saa = rank_of("outside_support", "saa")
    wins = sum(1 for f in ("reversal", "heavy", "outside_support", "multimodal", "decision_dependent")
               if first_of(f) == "bayes_true")
    return [
        dict(claim="S1", replication_verdict="supported" if r_base == 1 else "not supported",
             replication_effect=float(r_base),
             detail=f"bayes_true ranks {r_base} of 7 on the control family; supported if first", plan=plan.digest),
        dict(claim="S2", replication_verdict="supported" if r_multi == 1 else "not supported",
             replication_effect=float(r_multi),
             detail=f"saa ranks {r_multi} of 7 on multimodal demand; supported if first", plan=plan.digest),
        dict(claim="S3",
             replication_verdict="supported" if (r_moment < r_emp and r_moment < r_saa) else "not supported",
             replication_effect=float(r_moment),
             detail=f"dro_moment ranks {r_moment}, dro_emp {r_emp}, saa {r_saa} on the outside-support family; "
                    f"supported if the model that CAN price unseen outcomes beats both that cannot",
             plan=plan.digest),
        dict(claim="S4", replication_verdict="supported" if wins >= 3 else "not supported",
             replication_effect=float(wins),
             detail=f"bayes_true ranks first on {wins} of the 5 stress families, all of which it is "
                    f"misspecified for; supported if at least 3", plan=plan.digest),
    ]


def replicate_learnability(offset: int = SEED_OFFSET,
                           progress: Optional[Callable[[float, str], None]] = None) -> List[Dict[str, Any]]:
    from . import learnability as L
    plan = L.LearnPlan(seed=L.LearnPlan().seed + offset)
    res = L.run_study(plan, progress=progress, market=False)
    v = res.verdicts.set_index("id")
    out = []
    for key in ("H1", "H2"):
        if key not in v.index:
            continue
        row = v.loc[key]
        effect = _leading_number(str(row.get("result", "")))
        out.append(dict(claim=key, replication_verdict=str(row["verdict"]), replication_effect=effect,
                        detail=f"{row['result']} (study rule: {str(row['decision_rule'])[:110]})",
                        plan=plan.digest))
    return out


def replicate_validity(offset: int = SEED_OFFSET,
                       progress: Optional[Callable[[float, str], None]] = None) -> List[Dict[str, Any]]:
    import json as _json
    from pathlib import Path as _Path
    from . import learnability_validity as V
    plan = V.ValidityPlan(seed=V.ValidityPlan().seed + offset)
    # the same hyperparameters the registered learnability study used, read from its stored run rather than
    # re-tuned: re-tuning would change two things at once and the replication would not isolate the seed
    hp = _json.loads((_Path(__file__).resolve().parents[1] / "docs" / "learnability_results" /
                      "run.json").read_text(encoding="utf-8"))["hyperparameters"]
    res = V.run_validity(plan, hp=hp, arm="confirmatory", progress=progress)
    v = res.verdicts.set_index("id")
    out = []
    for key in ("V1", "V2"):
        if key not in v.index:
            continue
        row = v.loc[key]
        out.append(dict(claim=key,
                        replication_verdict="supported" if bool(row.get("supported", False))
                        else "not supported",
                        replication_effect=_leading_number(str(row.get("result", ""))),
                        detail=f"{row.get('result', '')} ({str(row.get('reading', ''))[:90]})",
                        plan=plan.digest))
    return out


def replicate_gate(offset: int = SEED_OFFSET,
                   progress: Optional[Callable[[float, str], None]] = None) -> List[Dict[str, Any]]:
    """G1: does the gate's 90% lower bound actually cover 90% of the time?"""
    from . import gate_validation as G
    from .scenarios import DEFAULTS, build_scenario
    plan = G.ValidationPlan(seed0=G.ValidationPlan().seed0 + offset,
                            state_seed=G.ValidationPlan().state_seed + offset)
    base = build_scenario(DEFAULTS)
    res = G.run_validation(plan, base, progress=progress)
    summary = res.summary
    # the claim is about the confidence-sensitive gate's own bound, so only its rows count
    rows = summary[summary["gate"].astype(str) == "lcb"]
    if rows.empty or "cov_empirical" not in summary.columns:
        return [dict(claim="G1", replication_verdict="not supported", replication_effect=None,
                     detail="no coverage was produced for the lcb gate", plan=res.plan_hash)]
    value = float(rows["cov_empirical"].mean())
    nominal = float(rows["cov_nominal"].iloc[0])
    lo = float(rows["cov_empirical"].min())
    hi = float(rows["cov_empirical"].max())
    return [dict(claim="G1",
                 replication_verdict="supported" if value >= nominal - 0.02 else "not supported",
                 replication_effect=value,
                 detail=f"measured coverage of the nominal {nominal:.0%} bound averages {value:.1%} across "
                        f"{len(rows)} feedback regimes (range {lo:.1%} to {hi:.1%}); supported only if it "
                        f"reaches the nominal level",
                 plan=res.plan_hash)]


def replicate_tournament(offset: int = SEED_OFFSET,
                         progress: Optional[Callable[[float, str], None]] = None) -> List[Dict[str, Any]]:
    from dataclasses import asdict as _asdict
    from . import arena
    pr = arena.Prereg()
    pr = arena.Prereg(**{**_asdict(pr), "train_seed": pr.train_seed + offset,
                         "test_seed": pr.test_seed + offset})
    res = arena.run_protocol(pr)
    v = res.verdicts.set_index(res.verdicts.columns[0])
    out = []
    for key in ("PR1", "PR2", "PR3", "PR4", "PR5", "PR6"):
        if key not in v.index:
            continue
        row = v.loc[key]
        out.append(dict(claim=key, replication_verdict=str(row["verdict"]), replication_effect=None,
                        detail=str(row.get("result", ""))[:160], plan=getattr(res, "plan_hash", "")))
    return out


def _leading_number(text: str) -> Optional[float]:
    import re
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(m.group()) if m else None


RUNNERS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "information": replicate_information,
    "stress": replicate_stress,
    "learnability": replicate_learnability,
    "validity": replicate_validity,
    "gate_validation": replicate_gate,
    "tournament": replicate_tournament,
}


def run_adjudication(studies: Sequence[str] = STUDIES, offset: int = SEED_OFFSET,
                     progress: Optional[Callable[[float, str], None]] = None
                     ) -> Dict[str, pd.DataFrame]:
    note = progress or (lambda f, m: None)
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for i, s in enumerate(studies):
        lo, hi = i / len(studies), (i + 1) / len(studies)
        note(lo, f"replicating {s}")
        try:
            records += RUNNERS[s](offset, progress=lambda f, m, lo=lo, hi=hi: note(lo + (hi - lo) * f, m))
        except Exception as exc:                       # a study that cannot run is reported, not skipped
            errors.append(dict(study=s, error=f"{type(exc).__name__}: {exc}"))
            for c in CLAIMS:
                if c.study == s:
                    records.append(dict(claim=c.key, replication_verdict="not run", replication_effect=None,
                                        detail=f"the study did not complete: {type(exc).__name__}: {exc}"))
    note(1.0, "done")
    verdicts = verdict_frame([r for r in records if r["replication_verdict"] != "not run"])
    not_run = pd.DataFrame([r for r in records if r["replication_verdict"] == "not run"])
    return {"verdicts": verdicts, "claims": claim_table(), "outcomes": outcome_table(),
            "not_run": not_run, "errors": pd.DataFrame(errors)}


EXPORTS = {
    "claims": "the claim table as frozen, with each claim's original verdict and effect",
    "outcomes": "what each of the five outcomes means",
    "verdicts": "per claim: original verdict, replication verdict, and the outcome the frozen rule assigns",
    "plan": "the frozen specification and its hash",
}
