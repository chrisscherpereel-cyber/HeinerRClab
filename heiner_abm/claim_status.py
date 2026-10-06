"""What the laboratory takes from published theory, what it simplifies, and what it adds.

Every component is in exactly one of three classes:

* established  - a published result, used or reproduced as stated in its source;
* reduced_form - the laboratory's stylized stand-in for a market feature or for a theory's mechanism; a modeling
                 choice, not a claim of the cited work;
* extension    - a construct the laboratory proposes; its results test the laboratory's operationalization, not the
                 original theory.

Sources are keys of heiner_abm.literature.REFERENCES (tests/test_claim_status.py checks that they resolve). A
component with no source is the laboratory's own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

STATUS_LABELS = {
    "established": "Established theory",
    "reduced_form": "Reduced-form implementation",
    "extension": "Proposed extension",
}


@dataclass(frozen=True)
class Component:
    status: str
    name: str
    what: str
    sources: Tuple[str, ...] = ()
    where: str = ""          # where in the laboratory it is used


COMPONENTS: List[Component] = [
    # ---------------------------------------------------------------------------------------- established theory
    Component("established", "Reliability condition",
              "Flexibility pays only if r/w > (D/G)·(1 − π)/π. The laboratory measures π, r, w, G and D for every firm "
              "and decision.", ("heiner1983",), "Market lab, RC validation, CD-gap explorer"),
    Component("established", "Partial-adjustment bound",
              "The maximal reliable adjustment speed is β₀ = 1/((1 + K)(1 − f′)), with K the error-to-signal ratio.",
              ("heiner1989",), "Dynamic RC page"),
    Component("established", "Optimal filtering of a random walk",
              "With a random-walk target observed with noise, the loss-minimizing adjustment speed is the steady-state "
              "Kalman gain. The laboratory reproduces the loss curve and the optimum.", ("muth1960", "kalman1960"),
              "Solvable benchmark"),
    Component("established", "Cobweb cycles and their damping",
              "Producers who commit output on last period's price generate alternating cycles; adaptive expectations "
              "dampen them.", ("ezekiel1938", "nerlove1958"), "Field patterns, cobweb theory page"),
    Component("established", "Logit rule choice",
              "Agents switch between rules according to recent performance with intensity of choice β.",
              ("brock1997",), "Rule choice"),
    Component("established", "Imitation of the best",
              "Imitating the most profitable rival drives Cournot output above the Cournot–Nash level.",
              ("vegaredondo1997",), "Signature tests, field patterns"),
    # ---------------------------------------------------------------------------------------- reduced form
    Component("reduced_form", "The market",
              "One homogeneous product, linear demand with a price floor, and a raw-material cost that follows a "
              "reflecting random walk. Stylized and not calibrated to any commodity, so it is judged by qualitative "
              "patterns.", ("grimm2005",), "Every simulation page"),
    Component("reduced_form", "Production rules",
              "A model-based rule (partial adjustment toward the Cournot best reply on the believed demand curve) and a "
              "model-free margin-feedback quantity rule (output adjusts to the observed margin relative to a "
              "desired margin; configured as 'Bertrand' but not price-setting Bertrand competition).", (),
              "Market lab, hypothesis tests"),
    Component("reduced_form", "Difficulty and competence",
              "Difficulty is cost volatility Δ, the number of rivals and unannounced demand shifts; competence is cost "
              "foresight κ and perception noise σ. Heiner's CD-gap is measured as the error of each firm's cost "
              "perception.", ("heiner1983",), "Hypothesis tests, CD-gap explorer"),
    Component("reduced_form", "Unannounced structural change",
              "Unannounced demand-regime shifts that model-based firms learn about only after a lag. The simulator "
              "specifies the shift process, so the shifts are unknown to the agents but not unknowable; this is a "
              "restricted, agent-relative stand-in for Knightian uncertainty.", ("knight1921",),
              "Risk vs structural change"),
    Component("reduced_form", "Rival theories as agents",
              "Each rival theory enters as one or two agent designs: real options as an inaction-band heuristic (no "
              "option valuation or adjustment cost), satisficing as aspiration search, reinforcement learning as softmax or Erev–Roth learners, simple heuristics as a "
              "target margin or win-stay/lose-shift, optimization as a filtered best reply or rational expectations. "
              "The theories' directional predictions are stylized.",
              ("dixit1989", "simon1955", "erev1998", "gigerenzer2009"), "Agent tournament, Competing theories"),
    # ---------------------------------------------------------------------------------------- proposed extensions
    Component("extension", "Dynamic measurement of the reliability condition",
              "Heiner's condition values a deviation as a one-shot bet. The laboratory instead forks the market at each "
              "decision and compares the branches over H periods, and decomposes the value into immediate, "
              "persistence and strategic-feedback parts. H is a researcher's measurement; agents never see it.", (),
              "Measurement settings, Dynamic RC page"),
    Component("extension", "The reliability condition as a learned decision rule",
              "The theory says when restriction pays, not how an agent can know its own r, w, π, G and D. The Adaptive "
              "selection rule and the tournament's reliability-condition agents learn these from their own observed "
              "experience.", (), "Market lab (Adaptive rule), agent tournament"),
    Component("extension", "Oracle versus learned reliability",
              "The cost of applying the principle is decomposed into estimation from limited experience and judging "
              "with a misspecified model, by comparing an agent that knows its true reliability with learning agents.",
              (), "Mechanisms page"),
    Component("extension", "Error-to-signal boundary",
              "Restriction is predicted to pay once the flexible rule's error-to-signal ratio K is about 1 or more; "
              "tested in the market and in three other decision tasks.", ("heiner1989",),
              "Mechanisms, Generalization"),
    Component("extension", "Uncertainty-aware reliability gates",
              "The Adaptive rule's decision to adopt a recommendation is made by an estimated-gain gate (existing), a "
              "confidence-sensitive gate (lower confidence bound on the advantage above an adjustment cost, with a "
              "minimum evidence requirement) or an exploration-enabled gate (randomized trials, learning from its own "
              "payoffs only), compared with an ORACLE benchmark from independent runs. Heiner did not propose these "
              "gates; results test the laboratory's operationalization.", (), "Reliability gates page, Market lab"),
    Component("extension", "Lopsided stakes against certainty equivalence",
              "With asymmetric losses and unchanged information, restricting moves in the costly direction is compared "
              "with optimal filtering, which predicts that stakes shift the level of the action but not its response "
              "to news.", ("muth1960", "kalman1960"), "Solvable benchmark"),
]


def by_status(status: str) -> List[Component]:
    return [c for c in COMPONENTS if c.status == status]
