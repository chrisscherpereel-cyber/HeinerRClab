"""Registered results reported on the theory pages.

These numbers come from the registered runs documented in the README. They are tied to the plan hashes below:
tests/test_theory_pages.py fails when the code changes the hashes, so the numbers cannot silently go stale.

Revision note (2 October 2026): the text of the agents and hypotheses was converted from British to American spelling.
Plan hashes include that text, so five hashes changed (tournament 15193603c3f8359b, mechanism study fafb771393aab7fe,
rule choice e6df56611a66c23f, generalization ee7faf1cff253ace, tracking benchmark e6c4f642f296c19b). The computation
did not change: every registered study was rerun under the new hashes (the tournament main run and its three
replications, the mechanism study, rule choice, generalization and the tracking benchmark) and reproduced every
registered number exactly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

TOURNAMENT_PLAN = "9699f86a6a899cf7"
STUDY_PLAN = "e9b5cbda8a4c4ab7"
CHOICE_PLAN = "0f7bb98f8f7e8e44"       # endogenous rule choice (Rule choice page)
TASK_PLAN = "b46c1f64a8250547"         # generalization tasks (Generalization page)
TRACK_PLAN = "f582721595727105"        # single-firm tracking benchmark (Solvable benchmark page)
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
# Bertrand and the Cournot reference market, two seeds each (1,000 periods, 20 replications, H = 20). Seeds: 1 and 3
# (Bertrand), 1 and 42 (Cournot). Rerun on 6 October 2026 with the observable Adaptive rule; only the Cournot seed-1
# records changed (see the revision notes under FINDING_FINGERPRINTS).
DIRECTIONAL: Dict[str, Dict[str, Tuple[str, str]]] = {
    "heiner": {"Bertrand": ("6/1/2", "5/1/3"), "Cournot": ("6/1/2", "6/1/2")},
    "neo": {"Bertrand": ("2/4/1", "1/4/2"), "Cournot": ("3/2/2", "3/2/2")},
    "options": {"Bertrand": ("2/4/1", "2/3/2"), "Cournot": ("2/3/2", "2/3/2")},
    "cobweb": {"Bertrand": ("1/5/1", "2/4/1"), "Cournot": ("4/3/0", "4/3/0")},
    "biasvar": {"Bertrand": ("4/2/0", "3/2/1"), "Cournot": ("3/2/1", "3/2/1")},
    "satisficing": {"Bertrand": ("0/1/0", "0/1/0"), "Cournot": ("0/1/0", "0/1/0")},
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
SPECIAL_RESULTS: Dict[str, Tuple[str, str]] = {'cobweb': ('supported',
            'n = 3: theory 1.00, simulated 0.95; n = 4: theory 0.80, simulated 0.80; n = 6: theory 0.57, simulated '
            '0.55; n = 8: theory 0.44, simulated 0.45'),
 'heuristic': ('not supported',
               "heuristic's advantage changes by -26.0 per unit of noise (p = 8.1e-06); at the highest noise +684 "
               '[+564, +826]'),
 'imitation': ('supported', 'imitate-the-best markets produce 1.24 [1.22, 1.26] × the Cournot–Nash output'),
 'optimiser': ('supported',
               "profit slope on foresight +40.6 per unit (p = 0.0116); with full information the rational market's "
               'price is within 0.13% of Cournot–Nash on average'),
 'options': ('not supported', 'value of flexibility changes by -11.2 per unit of volatility (p = 1.03e-15)'),
 'rl': ('not supported',
        'relative profit improves by -138 [-488, +230] from the first to the last third in a stationary market, and '
        'by -190 [-487, +112] with unannounced shifts'),
 'satisficing': ('not supported', 'change rate changes by -0.0002 per unit of volatility (p = 0.386)')}


# ================================================================================================ unhashed findings
# The findings above that come from a frozen plan are protected by its hash. The findings below are reported in the
# README and the app but were produced without a frozen plan, so they are tied instead to a fingerprint of the source
# files that produce them (FINDING_FINGERPRINTS). tests/test_registered.py fails when one of those files changes and
# the finding is neither re-fingerprinted (allowed only after a rerun reproduces or replaces the reported numbers) nor
# listed in REPLICATION_REQUIRED. A listed finding is shown in the app and the README as requiring replication; its
# numbers are left as reported, never silently updated.
#
# Provenance of the baseline fingerprints: they were taken from the code at commit 469619d (6 October 2026), when this
# mechanism was introduced. The findings were reported for that code; they were not rerun when the fingerprints were
# recorded.
#
# Revision notes
# * 6 October 2026, observable feedback for the Adaptive rule (engine.py, agents.py, params.py changed).
#   - presets: re-fingerprinted. Without Adaptive agents the change does not touch any code path, and the old
#     (469619d) and new engines gave bit-identical prices, outputs and measured gains for Bertrand and Cournot
#     scenarios, with and without demand shifts, at H = 1 and H = 6. The Adaptive preset is a separate finding.
#   - special, patterns: re-fingerprinted after a rerun. They use the arena agents, not the Adaptive rule; rerun
#     with the new code (signature tests at the Full scale, field patterns at the registered settings), every
#     signature-test line in SPECIAL_RESULTS and every field-pattern number in the README was reproduced exactly.
#   - directional, horse_race, preset_adaptive: involve Adaptive agents; listed in REPLICATION_REQUIRED at first.
# * 6 October 2026, rerun of directional, horse_race and preset_adaptive under the observable Adaptive rule. The
#   protocol was first run on the old code (469619d), which reproduced every previously reported number: the
#   directional records with seeds 1 and 3 (Bertrand) and 1 and 42 (Cournot), the forecasting AUCs and encompassing
#   gain, and the baseline preset (price 48.6, cost 43.7, slope +23; seeds 1-6). Then rerun on the new code:
#   - directional: only E7 (Adaptive agents) changed; every other experiment was bit-identical. The Cournot seed-1
#     records changed (Heiner 5/1/3 -> 6/1/2, neoclassical 3/1/3 -> 3/2/2, real options 1/3/3 -> 2/3/2, satisficing
#     0/0/1 -> 0/1/0); DIRECTIONAL updated. Over eight seeds per market Heiner keeps the best net record in every run.
#   - horse_race: 100 environments x 2 replications, sampling seed 12345, H = 20. Dynamic RC 0.612 -> 0.602,
#     track record 0.859 -> 0.851, encompassing gain -0.001 -> +0.000, one-shot RC 0.546 -> 0.537; ranking and
#     conclusions unchanged. README and the Competing theories page updated.
#   - preset_adaptive: price 48.56 -> 48.16, share of recommended changes adopted 0.75 -> 0.74, slope +67 -> +58;
#     typical result in ui/common.PRESET_INFO updated.
# * 6 October 2026, information-timing correction (agents.py, engine.py, params.py changed). Adaptive feedback now passes
#   through an explicit pending queue and is released only at maturity (agents.DECISION_SCHEDULE). The immediate
#   "lookahead" option, which with H > 1 let agents learn from periods that had not yet occurred, was replaced by the
#   "oracle" treatment, released at t + H - 1. No reported finding used it: every finding listed here uses observable
#   feedback or no Adaptive agents. All six were rerun with the corrected code (tools/rerun_adaptive_findings.py for
#   directional seeds B1, B3, C1, C42, horse_race and the presets; signature tests at the Full scale; field patterns at
#   the registered settings) and reproduced the previously recorded results exactly. Re-fingerprinted.

FINDING_SOURCES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "directional": ("Competing theories: directional tournament records (matches / contradictions / inconclusive)",
                    ("agents.py", "engine.py", "params.py", "analysis.py", "experiments.py", "theories.py")),
    "horse_race": ("Competing theories: out-of-sample forecasts of which firms beat their rigid twin (AUC)",
                   ("agents.py", "engine.py", "params.py", "analysis.py", "experiments.py")),
    "presets": ("Typical results of the preset scenarios without Adaptive agents (sidebar preset descriptions)",
                ("agents.py", "engine.py", "params.py", "analysis.py", "experiments.py")),
    "preset_adaptive": ("Typical result of the reliability-learning (Adaptive) preset",
                        ("agents.py", "engine.py", "params.py", "analysis.py", "experiments.py")),
    "special": ("Signature tests by theory",
                ("arena.py", "rulechoice.py", "special.py", "analysis.py", "agents.py", "params.py", "#tuned")),
    "patterns": ("Field patterns (pattern-oriented validation)",
                 ("arena.py", "rulechoice.py", "patterns.py", "agents.py", "params.py", "#tuned")),
    "calibration": ("Calibration to experiments: recovery of generating rules on synthetic subjects",
                    ("calibration.py",)),
}

FINDING_FINGERPRINTS: Dict[str, str] = {
    "directional": "f31ab19bd2cfda91",
    "horse_race": "95aabe1ec99f0ac6",
    "presets": "95aabe1ec99f0ac6",
    "preset_adaptive": "95aabe1ec99f0ac6",
    "special": "4ada44342edf8f67",
    "patterns": "f0fa87069a9a1781",
    "calibration": "854400653fe42159",
}

# finding key -> why it requires replication (the model change and its date). Remove an entry only after the finding
# has been rerun and its numbers (and fingerprint) updated with a revision note.
REPLICATION_REQUIRED: Dict[str, str] = {}


def source_fingerprint(sources: Tuple[str, ...]) -> str:
    """SHA-256 (16 hex digits) of the listed heiner_abm source files, line endings normalized. '#tuned' stands for the
    registered tuned designs and parameters (kept in this file, which cannot fingerprint itself)."""
    import hashlib
    import json
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in sources:
        if name == "#tuned":
            h.update(json.dumps([TUNED_DESIGN, TUNED_PARAMS], sort_keys=True).encode())
            continue
        with open(os.path.join(here, name), "rb") as fh:
            h.update(name.encode() + b"\0" + fh.read().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def finding_is_current(key: str) -> bool:
    """True if the code that produced the finding is unchanged since the finding was reported."""
    return source_fingerprint(FINDING_SOURCES[key][1]) == FINDING_FINGERPRINTS.get(key)


def replication_note(key: str) -> Optional[str]:
    """Why a reported finding requires replication, or None if it is current. A finding whose code changed without an
    entry in REPLICATION_REQUIRED is also reported (the test suite fails in that case)."""
    if key in REPLICATION_REQUIRED:
        return REPLICATION_REQUIRED[key]
    if not finding_is_current(key):
        return "The code that produced this finding has changed since it was reported; it has not been rerun."
    return None
