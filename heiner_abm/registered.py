"""Registered results reported on the theory pages.

These numbers come from the registered runs documented in the README. They are tied to the plan hashes below:
tests/test_theory_pages.py fails when the code changes the hashes, so the numbers cannot silently go stale.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

TOURNAMENT_PLAN = "15193603c3f8359b"
STUDY_PLAN = "fafb771393aab7fe"
CHOICE_PLAN = "e6df56611a66c23f"       # endogenous rule choice (Rule choice page)
TASK_PLAN = "ee7faf1cff253ace"         # generalisation tasks (Generalisation page)
TRACK_PLAN = "e6c4f642f296c19b"        # single-firm tracking benchmark (Solvable benchmark page)
EXPERIMENT_PLAN = "01f956595e90e5ab"   # human experiment protocol (Play the market / Experiment analysis)

# Agent tournament: mean profit rank and aggregate rank over six criteria (1 = best of 9), main run and three
# replications with fresh seeds, and the design selected on training data in the main run.
TOURNAMENT: Dict[str, Dict] = {
    "cobweb": dict(design="Adaptive price expectations", profit=(1.89, 2.91, 2.74, 2.64), aggregate=(3.50, 2.67, 2.33, 2.83)),
    "heuristic": dict(design="Target-margin rule", profit=(2.79, 3.30, 1.91, 1.65), aggregate=(4.00, 4.17, 4.00, 3.83)),
    "options": dict(design="Inaction band · price-based target", profit=(3.41, 2.99, 3.24, 4.21),
                    aggregate=(3.33, 3.00, 3.83, 3.67)),
    "heiner": dict(design="Reliability condition · price-based target", profit=(3.98, 4.89, 6.00, 5.42),
                   aggregate=(4.17, 5.00, 3.33, 6.00)),
    "optimiser": dict(design="Rational expectations (Cournot–Nash)", profit=(5.84, 5.62, 5.90, 5.36),
                      aggregate=(4.83, 5.50, 6.83, 4.67)),
    "rl": dict(design="Softmax value learner", profit=(6.20, 6.65, 5.85, 6.20), aggregate=(8.17, 6.33, 5.17, 6.67)),
    "imitation": dict(design="Imitate the average (imitate the best in two replications)",
                      profit=(6.24, 4.35, 5.32, 5.29), aggregate=(6.00, 6.83, 7.00, 5.00)),
    "satisficing": dict(design="Aspiration search · price-based target (model-based in one replication)",
                        profit=(8.39, 7.88, 8.31, 8.60), aggregate=(5.50, 5.33, 7.17, 6.00)),
    "ruleb": dict(design="Rule B (rigid)", profit=(6.28, 6.41, 5.72, 5.62), aggregate=(5.50, 5.83, 5.33, 5.67)),
}

# Directional tournament of experiments (Competing theories page): matches / contradictions / inconclusive, for the
# Bertrand and the Cournot reference market, two seeds each (1,000 periods, 20 replications, H = 20).
DIRECTIONAL: Dict[str, Dict[str, Tuple[str, str]]] = {
    "heiner": {"Bertrand": ("6/1/2", "5/1/3"), "Cournot": ("5/1/3", "6/1/2")},
    "neo": {"Bertrand": ("2/4/1", "1/4/2"), "Cournot": ("3/1/3", "3/2/2")},
    "options": {"Bertrand": ("2/4/1", "2/3/2"), "Cournot": ("1/3/3", "2/3/2")},
    "cobweb": {"Bertrand": ("1/5/1", "2/4/1"), "Cournot": ("4/3/0", "4/3/0")},
    "biasvar": {"Bertrand": ("4/2/0", "3/2/1"), "Cournot": ("3/2/1", "3/2/1")},
    "satisficing": {"Bertrand": ("0/1/0", "0/1/0"), "Cournot": ("0/0/1", "0/1/0")},
    "ecology": {"Bertrand": ("2/0/0", "2/0/0"), "Cournot": ("1/1/0", "1/1/0")},
}

# Mechanism study findings that bear on each theory.
MECHANISMS: Dict[str, str] = {
    "heiner": (
        "With known reliability the reliability condition beats always adjusting toward the error-prone model-based "
        "target by +158 per period (95% CI 91–233) and breaks even near an error-to-signal ratio K ≈ 1. An agent that "
        "must learn its own reliability loses the whole gain (−14 on the model-based target, −132 on the price-based "
        "target), mostly through noisy estimates from limited, shifting experience or through judging its past "
        "decisions with a misspecified model."),
    "optimiser": (
        "Always adjusting toward the filtered best reply is the reference rule. On the model-based target it is beaten "
        "by a reliability-based restriction that knows the true reliability (+158 per period), because a misspecified "
        "demand model makes the best reply unreliable after unannounced shifts."),
    "options": (
        "An inaction band on the model-based target cut downside risk in every run of the tournament and broke even "
        "near K ≈ 1.06 in the mechanism study; on the reliable price-based target it cost profit (−144 per period)."),
    "cobweb": (
        "The adaptive price-expectations design needs no demand model; it ranked first or second on profit in every "
        "tournament run (first in the main run and replication 1). In the mechanism study, restricting it (inaction "
        "band or reliability condition) lowered profit."),
}


def tournament(key: str) -> Optional[Dict]:
    return TOURNAMENT.get(key)


# Tuned designs and parameters from the registered tournament run (plan TOURNAMENT_PLAN). Used as realistic rivals in
# the human experiment and as the default rule set for the rule-choice and validation pages, so that those pages do
# not need to re-run the one-minute tuning. Re-running the registered plan reproduces these values.
TUNED_DESIGN: Dict[str, str] = {'heiner': 'heiner_p',
     'optimiser': 'opt_nash',
     'options': 'options_p',
     'cobweb': 'cobweb_p',
     'heuristic': 'heur_markup',
     'satisficing': 'satis_p',
     'rl': 'rl_softmax',
     'imitation': 'imit_avg',
     'ruleb': 'ruleb'}
TUNED_PARAMS: Dict[str, Dict[str, float]] = {'heiner_m': {'a_cost': 0.262729,
              'a_rival': 0.063353,
              'phi': 0.239951,
              'theta': 3.44545,
              'memory': 0.886542,
              'horizon': 13.0},
 'heiner_p': {'lam': 0.708243, 'phi': 0.477099, 'theta': 1.585432, 'memory': 0.883274, 'horizon': 6.0},
 'opt_br': {'a_cost': 0.180077, 'a_rival': 0.050491, 'phi': 0.638447},
 'opt_nash': {'a_cost': 0.163401, 'phi': 0.241701},
 'options_m': {'a_cost': 0.059442, 'a_rival': 0.6927, 'phi': 0.112801, 'k': 1.764088},
 'options_p': {'lam': 0.977468, 'phi': 0.520112, 'k': 0.476872},
 'cobweb_p': {'lam': 0.698148, 'phi': 0.879722},
 'cobweb_q': {'a_rival': 0.931346, 'phi': 0.560706},
 'heur_wsls': {'step': 1.724622},
 'heur_markup': {'phi': 2.095768, 'm': 6.055212},
 'satis_m': {'a_cost': 0.633627, 'a_rival': 0.069392, 'phi': 0.134541, 'alpha': 0.085199},
 'satis_p': {'lam': 0.060947, 'phi': 0.086688, 'alpha': 0.014058},
 'rl_softmax': {'step': 4.87352, 'eta': 0.046256, 'temp': 1.876073},
 'rl_erevroth': {'step': 1.055022, 'forget': 0.001978, 'gain': 5.001642},
 'imit_best': {'p': 0.13636, 'noise': 4.569344},
 'imit_avg': {'p': 0.104148, 'noise': 7.527318},
 'ruleb': {}}


def registered_tuned():
    """The registered tournament's tuned designs and parameters as an arena.Tuned object."""
    from .arena import Tuned
    return Tuned(design=dict(TUNED_DESIGN), params={k: dict(v) for k, v in TUNED_PARAMS.items()})
