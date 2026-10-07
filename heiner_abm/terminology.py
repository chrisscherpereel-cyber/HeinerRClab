"""Terms the laboratory uses, defined once, with what each does and does not claim.

The app, the README and the reference pages take their wording from here, so that a term means the same thing
everywhere. Configuration values are unchanged for compatibility (a production rule is still configured as "Bertrand"
or "Cournot"); only the descriptions change.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# ------------------------------------------------------------------------------------------------ production rules
# Config value -> label. "Bertrand" is kept as the configuration value so that saved scenarios, plan files and
# registered results still load; it names a quantity rule, not price competition.
RULE_LABELS: Dict[str, str] = {
    "Bertrand": "Margin-feedback quantity rule",
    "Cournot": "Cournot best-reply quantity rule",
}
RULE_SHORT: Dict[str, str] = {"Bertrand": "margin feedback", "Cournot": "Cournot best reply"}


def rule_label(code: str) -> str:
    return RULE_LABELS.get(code, code)


def rule_option(code: str) -> str:
    """Label for a selector whose stored value is the configuration code."""
    return {"Bertrand": "Margin-feedback quantity rule (config 'Bertrand')",
            "Cournot": "Cournot best-reply quantity rule"}.get(code, code)


MARGIN_FEEDBACK_NOTE = (
    "The *margin-feedback quantity rule* (configuration value 'Bertrand', kept for compatibility) is not "
    "conventional Bertrand price competition. Firms choose quantities, never prices; the market clears one price on "
    "the demand curve; and each firm moves its output up or down in proportion to how far the observed margin "
    "(last price minus its cost estimate) is from a desired margin m*. It needs no demand model. The name refers only "
    "to the price-based signal the rule reacts to.")

COURNOT_NOTE = (
    "The *Cournot best-reply quantity rule* moves a share φ of the way toward the static Cournot best reply to "
    "rivals' last output, computed on the firm's believed demand curve. It is a partial-adjustment rule, not an "
    "equilibrium solution; firms do not anticipate rivals' reactions.")

# ------------------------------------------------------------------------------------------------ inaction band
INACTION_BAND_LABEL = "inaction-band heuristic inspired by real options"
INACTION_BAND_NOTE = (
    "The *inaction-band heuristic inspired by real options* moves toward its target only when the gap exceeds k "
    "standard deviations of recent target changes, so the band widens with measured volatility, as real-options "
    "theory predicts for optimal inaction zones. It is not a dynamic real-options model: there is no adjustment cost, "
    "no irreversibility, no option value computed from a known stochastic process and no optimal-stopping solution; "
    "the band width k is a tuned parameter. Results about it test this heuristic, not real-options theory as such.")

# ------------------------------------------------------------------------------------------------ uncertainty
# (key, name, definition, where it appears in the laboratory, what the agents know)
UNCERTAINTY_TYPES: List[Tuple[str, str, str, str, str]] = [
    ("risk", "Known stochastic risk",
     "Outcomes are random but drawn from a probability distribution that is fully specified and, in principle, "
     "known.",
     "Raw-material cost follows a reflecting random walk with uniform steps of at most Δ; perception noise σ.",
     "The simulator specifies the distribution exactly. No agent is given it; agents react to realized costs, so "
     "for them it is a fixed process they could learn, not a known distribution they optimize against."),
    ("parameter", "Parameter uncertainty",
     "The form of the process is known but some of its parameters are not, so they must be estimated.",
     "Environments draw Δ, σ, κ, demand slope, hazard and lag from ranges; agents' filters, aspirations, bands and "
     "learned gains estimate what they need from experience.",
     "Agents are not told the parameters of their environment; tuned agent parameters are fixed across "
     "environments."),
    ("ambiguity", "Distributional ambiguity",
     "The decision maker cannot commit to a single probability distribution (for example, a set of priors).",
     "Not modeled. No agent represents a set of distributions or uses an ambiguity-averse decision rule.",
     "Results here say nothing direct about ambiguity in this sense."),
    ("misspecification", "Structural model misspecification",
     "The decision maker's model of how the world works has the wrong form or outdated structure.",
     "Model-based (Cournot) firms and model-based tournament targets use a believed demand curve that lags the true "
     "curve after a shift; every agent's rule is a simplification of the true market; fitted rules are "
     "simplifications of human behavior.",
     "Agents never see the true demand curve; model-free rules avoid a demand model altogether."),
    ("change", "Unannounced environmental change",
     "The process itself changes at times that are not announced.",
     "Demand-regime shifts with hazard λ per period: a new intercept and slope drawn around the baseline.",
     "Agents are not told when a shift occurs, its size distribution or the hazard; model-based agents update their "
     "demand model only after a lag L."),
]

KNIGHT_NOTE = (
    "A simulator must specify every process it runs, including the hazard and size distribution of regime shifts. "
    "Those processes are unknown to the agents, but they are not unknowable: a long-lived agent could in principle "
    "learn the hazard and the distribution of shifts. Unannounced regime shifts therefore model structural change "
    "and model misspecification that the agents have not been told about. They do not, by themselves, establish "
    "Knightian uncertainty in the unrestricted sense of outcomes to which no probabilities can be assigned. Where "
    "the app says 'Knightian' it means this restricted, agent-relative sense.")

# ------------------------------------------------------------------------------------------------ evidence
# (key, label, what it establishes, what it does not)
EVIDENCE_TYPES: List[Tuple[str, str, str, str]] = [
    ("analytical", "Analytical verification",
     "The simulation reproduces a result derived mathematically (a closed-form loss curve, a stability boundary, an "
     "accounting identity), so the code implements the model as specified.",
     "Nothing about whether the model describes real behavior; nothing about rival theories."),
    ("simulation", "Simulation comparison",
     "Within the specified model, one rule, agent or theory's prediction does better or worse than another under "
     "stated environments, settings and criteria.",
     "Results hold for the implementations and environments tested; they are not evidence about real firms or "
     "people, and a different operationalization of a theory could do differently."),
    ("behavioral", "Out-of-sample behavioral prediction",
     "Rules fitted to part of a person's recorded choices predict that person's later choices better or worse than "
     "other rules, in published experimental data.",
     "Fit is not cause: a rule that predicts well need not be what people do, and the data were collected by others "
     "for other purposes."),
    ("causal", "Causal experimental evidence",
     "Randomized or counterbalanced manipulation of a condition, with outcomes measured afterwards, identifies the "
     "condition's effect on behavior.",
     "The laboratory has a protocol for such an experiment (Play the market) but has collected no data, so it "
     "currently reports no causal experimental evidence of its own."),
]
EVIDENCE_LABELS: Dict[str, str] = {k: label for k, label, _, _ in EVIDENCE_TYPES}

# Every reported result family and the kind of evidence it is.
EVIDENCE_BY_RESULT: List[Tuple[str, str, str]] = [
    ("Agent/engine equivalence and the RC accounting identity (test suite)", "analytical",
     "Internal consistency checks of the code."),
    ("Solvable benchmark T1 (exact Muth–Kalman loss curve and optimum)", "analytical", ""),
    ("Cobweb signature test (stability boundary 4/(n + 1))", "analytical",
     "The simulated boundary reproduces the derived one."),
    ("Solvable benchmark T2–T4 (lopsided stakes)", "simulation", ""),
    ("Other signature tests", "simulation", ""),
    ("Hypothesis tests H1–H12, Competing theories (directional tournament)", "simulation", ""),
    ("Competing theories: out-of-sample forecasts of which simulated firms benefit", "simulation",
     "Out of sample within the simulation, not behavioral prediction."),
    ("Agent tournament, mechanism study, rule choice, generalization", "simulation", ""),
    ("Decision benchmarks: correctness checks on solvable and enumerated cases", "analytical", ""),
    ("Decision benchmarks: inventory and bandit comparisons", "simulation",
     "Tuned on training environments, selected on validation environments, reported on test environments."),
    ("NK landscapes (interaction complexity, noise and change)", "simulation",
     "Exact benchmarks by enumeration for small N, best-known otherwise; intervals clustered by landscape."),
    ("When can reliability be learned? (frozen study, proposed construct)", "simulation",
     "Untouched test paths and new process families; effects as paired differences with intervals."),
    ("Reliability gates under uncertainty (proposed extension)", "simulation",
     "Compares the laboratory's own selection gates; lower-bound coverage is measured, not assumed."),
    ("Field patterns", "simulation",
     "Qualitative comparison of simulated output with documented empirical patterns; not a fit to data."),
    ("Calibration recovery on synthetic subjects", "simulation", "Checks the fitting method, not people."),
    ("Empirical validation V1–V6 and calibration on human data", "behavioral", ""),
    ("Empirical validation V7 (time pressure)", "behavioral",
     "Compares fitted-rule shares across a condition manipulated within subjects in the original experiment. The "
     "manipulation was the original authors'; the shares are model-based classifications."),
    ("Human experiment E1–E5 (protocol 2.0)", "causal", "Randomized treatments; protocol only, no human data "
     "collected. A1 (adjustment frequency and profit) is associational."),
]

# ------------------------------------------------------------------------------------------------ scope of conclusions
SCOPE_NOTE = (
    "Every verdict here concerns a specific implementation of a theory in a specific simulated environment, under the "
    "stated settings, criteria and sample sizes. 'Not supported' means the tested implementation did not show the "
    "predicted pattern there. It does not refute the theory: a different operationalization, environment or "
    "criterion could give a different result, and the rival theories' predictions are stylized readings of their "
    "literatures.")

# ------------------------------------------------------------------------------------------------ verification vs validation
IDENTITY_NOTE = (
    "The RC accounting identity (the realized gain from deviations equals correct-deviation gains minus "
    "wrong-deviation losses) is checked by the test suite. That verifies the bookkeeping is internally consistent; it "
    "is true by construction and says nothing about whether the reliability condition predicts anything. Predictive "
    "validation needs estimates fixed before the outcomes they predict: the RC validation page estimates the "
    "condition in the first part of each run and predicts the advantage of flexibility in the second part.")

OOS_GAP_NOTE = (
    "The H-period measures (persistence and full) value each decision over periods t … t + H − 1, so their estimation "
    "window ends H − 1 periods before the evaluation window starts: only decisions whose horizon is complete before "
    "the split are used to estimate. The estimates therefore use nothing from the evaluation window. The one-period "
    "measure, realized profits, K and the CD-gap need no gap. (Until 6 October 2026 the windows overlapped by H − 1 "
    "periods: 19 of about 490 estimation decisions at H = 20 and 1,000 periods.)")

# ------------------------------------------------------------------------------------------------ frozen plans
FROZEN_PLAN_NOTE = (
    "A *frozen hashed specification* is a plan fixed in the code whose hash identifies the plan and the code that "
    "produced a result; any change gives a new hash. *Externally timestamped prospective preregistration* deposits "
    "the plan with an independent registry (such as OSF or AsPredicted) before the data are generated, so that the "
    "timing can be verified by others. The plans in this repository are frozen hashed specifications; the "
    "repository contains no record of external preregistration. 'Registered' and 'pre-registered' in the app refer "
    "to the former.")
