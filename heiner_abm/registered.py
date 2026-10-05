"""Registered results reported on the theory pages.

These numbers come from the registered runs documented in the README. They are tied to the plan hashes below:
tests/test_theory_pages.py fails when the code changes the hashes, so the numbers cannot silently go stale.

Revision note (2 October 2026): the text of the agents and hypotheses was converted from British to American spelling.
Plan hashes include that text, so five hashes changed (tournament 15193603c3f8359b, mechanism study fafb771393aab7fe,
rule choice e6df56611a66c23f, generalization ee7faf1cff253ace, tracking benchmark e6c4f642f296c19b). The computation
did not change: every registered study was rerun under the new hashes (the tournament main run and its three
replications, the mechanism study, rule choice, generalization and the tracking benchmark) and reproduced every
registered number exactly.

Revision note (5 October 2026): organizational ecology (structural inertia) became the ninth theory, implemented as two
agent designs. The tournament now has ten entries, which changes the agent code and therefore the tournament,
mechanism-study and rule-choice plans (new hashes below). All three registered studies, the three tournament
replications, the field patterns and the signature tests were rerun under the new plans, and every number below
comes from those runs. The directional tournament was rescored with a prediction from every theory for every
experiment.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

TOURNAMENT_PLAN = "110b3146bb072c2c"
STUDY_PLAN = "11a279e507f246e2"
CHOICE_PLAN = "f15f62149d08e800"       # endogenous rule choice (Rule choice page)
TASK_PLAN = "b46c1f64a8250547"         # generalization tasks (Generalization page)
TRACK_PLAN = "f582721595727105"        # single-firm tracking benchmark (Solvable benchmark page)
EXPERIMENT_PLAN = "01f956595e90e5ab"   # human experiment protocol (Play the market / Experiment analysis)

# Agent tournament: mean profit rank and aggregate rank over six criteria (1 = best of 10), main run and three
# replications with fresh seeds, and the design selected on training data in the main run.
TOURNAMENT: Dict[str, Dict] = {
    "cobweb": dict(design="Adaptive price expectations", profit=(2.10, 3.52, 3.11, 3.24),
             aggregate=(3.50, 4.17, 3.00, 3.50)),
    "heuristic": dict(design="Target-margin rule", profit=(2.96, 2.41, 2.29, 2.79),
             aggregate=(4.50, 5.00, 3.83, 4.67)),
    "options": dict(design="Inaction band · price-based target", profit=(3.50, 2.85, 4.10, 6.16),
             aggregate=(3.67, 2.83, 4.00, 6.33)),
    "heiner": dict(design="Reliability condition · price-based target", profit=(4.29, 5.00, 6.10, 3.98),
             aggregate=(4.17, 6.17, 4.50, 3.17)),
    "ecology": dict(design="Structural inertia · reorganize under threat of failure (scheduled reorganization in one "
                   "replication)", profit=(5.30, 4.64, 6.25, 3.60),
             aggregate=(3.33, 5.00, 6.83, 5.00)),
    "optimiser": dict(design="Rational expectations (Cournot–Nash)", profit=(6.33, 6.53, 5.80, 5.85),
             aggregate=(5.83, 6.17, 6.67, 6.00)),
    "imitation": dict(design="Imitate the best (imitate the average in two replications)", profit=(6.35, 6.00, 5.72, 6.00),
             aggregate=(9.00, 6.33, 7.67, 6.33)),
    "ruleb": dict(design="Rule B (rigid)", profit=(6.71, 7.49, 5.74, 6.89),
             aggregate=(6.00, 6.67, 4.17, 6.17)),
    "rl": dict(design="Softmax value learner", profit=(8.71, 7.67, 6.97, 7.69),
             aggregate=(9.17, 6.83, 5.67, 7.33)),
    "satisficing": dict(design="Aspiration search · price-based target (model-based in one replication)",
                        profit=(8.75, 8.89, 8.91, 8.81),
             aggregate=(5.33, 5.50, 8.50, 6.17)),
}

# Directional tournament of experiments (Competing theories page): matches / contradictions / inconclusive, for the
# Bertrand and the Cournot reference market, seeds 1 and 2 (1,000 periods, 20 replications, H = 20). Every theory states
# a prediction for every one of the nine experiments.
DIRECTIONAL: Dict[str, Dict[str, Tuple[str, str]]] = {
    "heiner": {"Bertrand": ("6/1/2", "6/1/2"), "Cournot": ("5/1/3", "5/1/3")},
    "optimiser": {"Bertrand": ("3/5/1", "3/5/1"), "Cournot": ("4/2/3", "4/2/3")},
    "options": {"Bertrand": ("3/5/1", "3/5/1"), "Cournot": ("1/5/3", "1/5/3")},
    "cobweb": {"Bertrand": ("1/7/1", "1/7/1"), "Cournot": ("5/4/0", "5/4/0")},
    "heuristic": {"Bertrand": ("5/2/2", "5/2/2"), "Cournot": ("4/2/3", "4/2/3")},
    "satisficing": {"Bertrand": ("5/3/1", "5/3/1"), "Cournot": ("4/2/3", "4/2/3")},
    "rl": {"Bertrand": ("3/4/2", "3/4/2"), "Cournot": ("4/2/3", "4/2/3")},
    "imitation": {"Bertrand": ("0/7/2", "0/7/2"), "Cournot": ("3/4/2", "3/4/2")},
    "ecology": {"Bertrand": ("4/5/0", "4/5/0"), "Cournot": ("2/6/1", "2/6/1")},
}

# Mechanism study findings that bear on each theory.
MECHANISMS: Dict[str, str] = {
    "heiner": (
        "With known reliability the reliability condition beats always adjusting toward the error-prone model-based "
        "target by +119 per period (95% CI 72–175) and breaks even near an error-to-signal ratio K ≈ 1. An agent that "
        "must learn its own reliability keeps little of the gain (+13 on the model-based target, −170 on the "
        "price-based target), mostly because it judges its past decisions with a misspecified demand model."),
    "optimiser": (
        "Always adjusting toward the filtered best reply is the reference rule. On the model-based target it is beaten "
        "by a reliability-based restriction that knows the true reliability (+119 per period), because a misspecified "
        "demand model makes the best reply unreliable after unannounced shifts."),
    "options": (
        "An inaction band on the model-based target cut downside risk in the main tournament run and all three "
        "replications, gained +216 per period in the mechanism study and broke even near K ≈ 1.03; on the reliable "
        "price-based target it cost profit (−137 per period)."),
    "cobweb": (
        "The adaptive price-expectations design needs no demand model; it ranked first on profit in the main "
        "tournament run and in the top three in every replication. In the mechanism study, restricting it (inaction "
        "band or reliability condition) lowered profit."),
    "ecology": (
        "Reorganizing only under threat of failure ranked fifth on profit in the main tournament run but first on the "
        "aggregate of six criteria, on the strength of low profit volatility and high survival; in the signature test "
        "it was less volatile than its always-adjusting twin at every level of volatility."),
}


def tournament(key: str) -> Optional[Dict]:
    return TOURNAMENT.get(key)


# Tuned designs and parameters from the registered tournament run (plan TOURNAMENT_PLAN). Used as realistic rivals in
# the human experiment and as the default rule set for the rule-choice and validation pages, so that those pages do
# not need to re-run the one-minute tuning. Re-running the registered plan reproduces these values.
TUNED_DESIGN: Dict[str, str] = {'cobweb': 'cobweb_p',
 'ecology': 'ecol_crisis',
 'heiner': 'heiner_p',
 'heuristic': 'heur_markup',
 'imitation': 'imit_best',
 'optimiser': 'opt_nash',
 'options': 'options_p',
 'rl': 'rl_softmax',
 'ruleb': 'ruleb',
 'satisficing': 'satis_p'}
TUNED_PARAMS: Dict[str, Dict[str, float]] = {'heiner_m': {'a_cost': 0.262729,
              'a_rival': 0.063353,
              'phi': 0.239951,
              'theta': 3.44545,
              'memory': 0.886542,
              'horizon': 13.0},
 'heiner_p': {'lam': 0.892012, 'phi': 0.445786, 'theta': 4.142132, 'memory': 0.937095, 'horizon': 10.0},
 'opt_br': {'a_cost': 0.230705, 'a_rival': 0.134275, 'phi': 0.083533},
 'opt_nash': {'a_cost': 0.150325, 'phi': 0.200264},
 'options_m': {'a_cost': 0.059442, 'a_rival': 0.6927, 'phi': 0.112801, 'k': 1.764088},
 'options_p': {'lam': 0.977468, 'phi': 0.520112, 'k': 0.476872},
 'cobweb_p': {'lam': 0.698148, 'phi': 0.879722},
 'cobweb_q': {'a_rival': 0.474026, 'phi': 0.054956},
 'heur_wsls': {'step': 2.654714},
 'heur_markup': {'phi': 0.434838, 'm': 2.534681},
 'satis_m': {'a_cost': 0.633627, 'a_rival': 0.069392, 'phi': 0.134541, 'alpha': 0.085199},
 'satis_p': {'lam': 0.060947, 'phi': 0.086688, 'alpha': 0.014058},
 'rl_softmax': {'step': 8.154122, 'eta': 0.131628, 'temp': 0.970332},
 'rl_erevroth': {'step': 1.055022, 'forget': 0.001978, 'gain': 5.001642},
 'imit_best': {'p': 0.13636, 'noise': 4.569344},
 'imit_avg': {'p': 0.256161, 'noise': 3.704685},
 'ecol_periodic': {'lam': 0.092515, 'phi': 0.434964, 'interval': 11.0},
 'ecol_crisis': {'lam': 0.82682, 'phi': 0.150208, 'floor': 0.827044, 'alpha': 0.016794},
 'ruleb': {}}


def registered_tuned():
    """The registered tournament's tuned designs and parameters as an arena.Tuned object."""
    from .arena import Tuned
    return Tuned(design=dict(TUNED_DESIGN), params={k: dict(v) for k, v in TUNED_PARAMS.items()})


# Empirical validation on the five public datasets (registered run of the protocol on the Empirical validation page).
EMPIRICAL_PLAN = "96b15ef4d9b78d56"
EMPIRICAL_VERDICTS: Dict[str, str] = {'V1': 'not supported',
 'V2': 'not supported',
 'V3': 'not supported',
 'V4': 'supported',
 'V5': 'supported',
 'V6': 'not supported',
 'V7': 'supported'}
EMPIRICAL_DETAILS: List[Tuple[str, str, str]] = [('V1',
  'Cournot, aggregate information',
  'best reply − RC test RMSE -0.061 [-0.152, +0.028], RC better for 46% of 72 participants'),
 ('V2',
  'Cournot, aggregate information',
  'change rate: human 0.615, simulated [0.880, 0.903]; output vs nash: human 0.979, simulated [0.972, 1.045]'),
 ('V1',
  'Cournot, individual information',
  'best reply − RC test RMSE -0.133 [-0.277, +0.010], RC better for 31% of 72 participants'),
 ('V2',
  'Cournot, individual information',
  'change rate: human 0.671, simulated [0.907, 0.931]; output vs nash: human 1.059, simulated [1.030, 1.110]'),
 ('V3',
  'EGM, negative feedback',
  'adaptive − RC test RMSE -0.332 [-0.644, -0.096], RC better for 47% of 252 participants'),
 ('V4', 'EGM, negative feedback', "most common best rule 'imitate' fits 22% [17%, 27%] of participants"),
 ('V3',
  'EGM, positive feedback',
  'adaptive − RC test RMSE -0.318 [-0.598, -0.110], RC better for 32% of 120 participants'),
 ('V4', 'EGM, positive feedback', "most common best rule 'naive' fits 24% [17%, 32%] of participants"),
 ('V3',
  'Anufriev–Hommes asset markets',
  'adaptive − RC test RMSE -1.647 [-4.176, +1.032], RC better for 32% of 120 participants'),
 ('V4', 'Anufriev–Hommes asset markets', "most common best rule 'trend' fits 37% [28%, 46%] of participants"),
 ('V5',
  'Newsvendor, low cost (optimal 75)',
  'pull-to-center ratio 0.16 [0.01, 0.31]; demand-chasing slope +0.34 [+0.27, +0.41]; n = 26'),
 ('V6',
  'Newsvendor, low cost (optimal 75)',
  'chasing − RC test RMSE -1.413 [-2.292, -0.654], RC better for 12% of 26 participants'),
 ('V5',
  'Newsvendor, high cost (optimal 25)',
  'pull-to-center ratio 0.29 [0.14, 0.46]; demand-chasing slope +0.30 [+0.23, +0.37]; n = 26'),
 ('V6',
  'Newsvendor, high cost (optimal 25)',
  'chasing − RC test RMSE -1.023 [-1.989, -0.138], RC better for 23% of 26 participants'),
 ('V7',
  'Time pressure (high vs low)',
  'share best predicted by simple/restricted rules, high − low time pressure +0.11 [+0.01, +0.21] (high 46%, low '
  '35%, paired by participant)')]
# Exploratory (not pre-registered): share of participants best predicted by a restricted rule (keep, inaction band,
# reliability condition) and the rule with the lowest out-of-sample error in each dataset.
EMPIRICAL_RULES: Dict[str, Dict] = {'Cournot, aggregate information': {'participants': 72,
                                    'restricted': 0.306,
                                    'top_rule': 'Move toward the best reply only below aspiration',
                                    'top_rmse': 4.738},
 'Cournot, individual information': {'participants': 72,
                                     'restricted': 0.111,
                                     'top_rule': 'Imitate the most profitable firm',
                                     'top_rmse': 5.841},
 'EGM, negative feedback': {'participants': 252,
                            'restricted': 0.313,
                            'top_rule': 'Imitate the most accurate group member',
                            'top_rmse': 5.183},
 'EGM, positive feedback': {'participants': 120,
                            'restricted': 0.358,
                            'top_rule': 'Naive expectations',
                            'top_rmse': 11.968},
 'Anufriev–Hommes asset markets': {'participants': 120,
                                   'restricted': 0.1,
                                   'top_rule': 'Trend following',
                                   'top_rmse': 54.209},
 'Newsvendor, low cost (optimal 75)': {'participants': 26,
                                       'restricted': 0.231,
                                       'top_rule': 'Anchor on the pulled-to-center order, adjust toward last demand',
                                       'top_rmse': 19.975},
 'Newsvendor, high cost (optimal 25)': {'participants': 26,
                                        'restricted': 0.115,
                                        'top_rule': 'Anchor on the pulled-to-center order, adjust toward last demand',
                                        'top_rmse': 17.374}}


# Signature tests by theory (heiner_abm.special), reference run at the Full scale. Not a hashed plan: the tests are
# deterministic given the code, so rerunning them on the page at the Full scale reproduces these lines.
SPECIAL_RESULTS: Dict[str, Tuple[str, str]] = {
    'optimiser': ('not supported', "profit slope on foresight -10.6 per unit (p = 0.0807); with full information the "
                                   "rational market's price is within 0.13% of Cournot–Nash on average"),
    'options': ('not supported', 'value of flexibility changes by -3.7 per unit of volatility (p = 0.0289)'),
    'cobweb': ('supported', 'n = 3: theory 1.00, simulated 0.95; n = 4: theory 0.80, simulated 0.80; n = 6: theory '
                            '0.57, simulated 0.55; n = 8: theory 0.44, simulated 0.45'),
    'heuristic': ('not supported', "heuristic's advantage changes by -8.8 per unit of noise (p = 0.00777); at the "
                                   'highest noise +409 [+311, +525]'),
    'satisficing': ('not supported', 'change rate changes by -0.0011 per unit of volatility (p = 0.00213)'),
    'rl': ('not supported', 'relative profit improves by -73 [-283, +156] from the first to the last third in a '
                            'stationary market, and by +148 [-163, +424] with unannounced shifts'),
    'imitation': ('supported', 'imitate-the-best markets produce 1.24 [1.22, 1.26] × the Cournot–Nash output'),
    'ecology': ('supported', 'Δ = 2: -6 [-9, -3]; Δ = 10: -94 [-101, -87]; Δ = 20: -310 [-326, -293]; Δ = 30: -753 '
                             '[-778, -728]'),
}
