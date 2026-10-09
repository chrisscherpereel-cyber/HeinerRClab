"""Guided research questions: each loads a documented, editable experiment specification.

A preset states the question, the primary outcome and the comparison it is about. The values are starting points for a
first investigation, not optimal or registered designs; registered studies keep their own pages (linked as
`registered_page`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .spec import Design, ExperimentSpec, Question, Treatment


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    summary: str
    spec: Optional[ExperimentSpec]             # None: not a simulation (e.g. a human experiment)
    registered_page: Optional[Tuple[str, str]] = None   # (page path, label) of a related registered study
    note: str = ""


def _spec(**kw) -> ExperimentSpec:
    return ExperimentSpec(**kw)


PRESETS: Dict[str, Preset] = {}


def _add(p: Preset):
    PRESETS[p.key] = p


_add(Preset(
    "restriction", "When does restricting adaptation improve performance?",
    "Firms in the shared market act on a margin-feedback production rule. Compare acting on every recommendation with "
    "a confidence-sensitive reliability gate, as cost volatility rises.",
    _spec(name="Restricting adaptation in the market",
          question=Question("When does restricting adaptation improve performance?", "restriction", "net_profit",
                            ("gate_lcb", "always"), 0.0,
                            "Heiner's reliability condition predicts that restriction pays when the flexible rule's "
                            "errors are large relative to the signal; volatility raises both."),
          environment="market", candidate="Bertrand", policies=("always", "default", "band", "gate_gain", "gate_lcb"),
          design=Design(replications=8, periods=600, burn_in=50,
                        treatment=Treatment("delta", (5.0, 15.0, 30.0)),
                        outcomes=("adaptation_rate", "false_share", "missed"))),
    ("app_pages/hypotheses.py", "Hypothesis tests (registered market experiments)")))

_add(Preset(
    "learnability", "Can reliability be learned before the environment changes?",
    "Inventory decisions with unannounced demand changes. Compare the confidence-sensitive gate with always adapting "
    "as the change frequency rises.",
    _spec(name="Learning reliability before change",
          question=Question("Can reliability be learned before the environment changes?", "learnability",
                            "net_payoff", ("gate_lcb", "always"), 0.0,
                            "A gate needs evidence; frequent changes leave fewer informative observations per regime."),
          environment="inventory", candidate="fast_forecast",
          policies=("always", "default", "band", "gate_gain", "gate_lcb", "bocpd", "oracle"),
          design=Design(replications=10, periods=1000, burn_in=100, treatment=Treatment("hazard", (0.002, 0.01, 0.05)),
                        outcomes=("regret", "departure_rate", "adjustment_rate", "missed", "recovery_delay"))),
    ("app_pages/learnability.py", "Registered learnability study (frozen plan)")))

_add(Preset(
    "noise_simple", "Does observation noise favor simpler policies?",
    "Inventory decisions observed with increasing error. Compare an inaction band (a simple restriction) with always "
    "acting on the forecast.",
    _spec(name="Observation noise and simple policies",
          question=Question("Does observation noise favor simpler policies?", "noise_simple", "net_payoff",
                            ("band", "always"), 0.0,
                            "Noisier observations make the forecast-based target less reliable; a band ignores small, "
                            "noise-driven changes."),
          environment="inventory", candidate="fast_forecast", policies=("always", "default", "band", "gate_gain"),
          design=Design(replications=10, periods=1000, burn_in=100, treatment=Treatment("tau", (0.0, 15.0, 35.0)),
                        outcomes=("departure_rate", "adjustment_rate", "adjustment_magnitude", "regret"))),
    ("app_pages/generalisation.py", "Generalization tasks (registered)")))

_add(Preset(
    "complexity", "When does strategic complexity make coordinated change unreliable?",
    "NK landscapes with more interacting decisions. Compare a confidence-sensitive gate with acting on every apparent "
    "improvement, on identical proposals and budgets.",
    _spec(name="Complexity and coordinated change",
          question=Question("When does strategic complexity make coordinated change unreliable?", "complexity",
                            "payoff", ("gate_lcb", "gate_none"), 0.0,
                            "With more interactions, a change that looks better in one evaluation is more often an "
                            "artifact of noise or of interdependent effects."),
          environment="nk", candidate="stochastic", policies=("gate_none", "gate_gain", "gate_lcb", "hill"),
          design=Design(replications=12, periods=300, burn_in=0, treatment=Treatment("K_NK", (0, 2, 4, 7)),
                        outcomes=("regret", "false_share", "missed_share", "escapes"))),
    ("app_pages/nk.py", "NK landscapes (full sweeps)")))

_add(Preset(
    "decision_aid", "Does a decision aid improve human performance?",
    "A question about people, not simulated agents: it needs a human experiment. Participants play the market with and "
    "without an aid; the analysis compares them with the agents' shadow decisions.",
    None, ("app_pages/play_market.py", "Play the market (human experiment)"),
    note="The workspace cannot answer this with simulation alone. Use Validation → Human experiments to collect "
         "decisions (the decision aid is randomized against an unaided control), then Human experiments: analysis to estimate its effect. A synthetic pilot can only pre-test the design."))


def load(key: str, mode: str = "explore") -> ExperimentSpec:
    spec = PRESETS[key].spec.copy()
    spec.mode = mode
    return spec


def blank(environment: str = "inventory", mode: str = "explore") -> ExperimentSpec:
    from .environments import ENVIRONMENTS
    env = ENVIRONMENTS[environment]
    spec = ExperimentSpec(name="Untitled experiment", mode=mode, environment=environment,
                          candidate=env.candidates[0].key,
                          policies=tuple(p.key for p in env.policies if p.kind == "selection")[:3])
    sel = [p for p in spec.policies]
    spec.question = Question("", "custom", env.primary, (sel[-1], sel[0]) if len(sel) > 1 else ("", ""))
    return spec


# Explore mode: simplified uncertainty levels. Each lists exactly which underlying settings it changes.
EXPLORE_LEVELS = {
    "inventory": {"low": {"sigma": 10.0, "tau": 2.0, "hazard": 0.002, "avail": 1.0},
                  "medium": {"sigma": 20.0, "tau": 10.0, "hazard": 0.01, "avail": 1.0},
                  "high": {"sigma": 40.0, "tau": 30.0, "hazard": 0.04, "avail": 0.5}},
    "market": {"low": {"delta": 3.0, "noise": 1.0, "hazard": 0.0},
               "medium": {"delta": 10.0, "noise": 5.0, "hazard": 0.0},
               "high": {"delta": 25.0, "noise": 12.0, "hazard": 0.02}},
    "learning": {"low": {"noise": 0.8, "obs_noise": 0.2, "hazard": 0.002},
                 "medium": {"noise": 2.0, "obs_noise": 1.5, "hazard": 0.01},
                 "high": {"noise": 4.5, "obs_noise": 4.0, "hazard": 0.04}},
    "investment": {"low": {"noise": 0.8, "obs_noise": 0.2, "hazard": 0.002},
                   "medium": {"noise": 2.0, "obs_noise": 1.5, "hazard": 0.01},
                   "high": {"noise": 3.5, "obs_noise": 3.5, "hazard": 0.04}},
    "nk": {"low": {"obs_noise": 0.0, "hazard": 0.0}, "medium": {"obs_noise": 0.02, "hazard": 0.005},
           "high": {"obs_noise": 0.06, "hazard": 0.02}},
}
