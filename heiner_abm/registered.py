"""Registered results reported on the theory pages.

These numbers come from the registered runs documented in the README. They are tied to the plan hashes below:
tests/test_theory_pages.py fails when the code changes the hashes, so the numbers cannot silently go stale.

Revision note (7 October 2026, equal tuning budgets): the learnability study gave four comparator policies a
four-candidate hyperparameter search on training configurations and the two reliability gates none, and the frozen
plan's `gate_confidence` / `gate_min_evidence` never reached the gate (the market replication tuned the inaction band
only). Both studies now tune every policy that has hyperparameters with the same number of candidates
(`LearnPlan.tune_gate`, `MarketPlan.tune_gate` restore the old behavior). The study code changed, so the plan hash
changed, ec9b781e125e289d → 8f43bedc7dae10bc, and the whole study was rerun under the new plan; the numbers below and
in docs/learnability_results/ are from that rerun. The results obtained under ec9b781e125e289d are kept in the README's
revision notes and REQUIRE REPLICATION: they were produced with an unequal tuning budget and are not comparable
one-for-one with the new run.

Revision note (6 October 2026): organizational ecology became the ninth theory, with two tournament designs, so
the tournament's code hash changed and every study built on it was rerun: tournament 9699f86a6a899cf7 →
110b3146bb072c2c (main run and three replications), mechanism study e9b5cbda8a4c4ab7 → 11a279e507f246e2, rule choice
0f7bb98f8f7e8e44 → f15f62149d08e800; the signature tests, field patterns and directional records were rerun too. The
numbers below are from those runs; the earlier values are kept in the README's revision notes.

Revision note (2 October 2026): the text of the agents and hypotheses was converted from British to American spelling.
Plan hashes include that text, so five hashes changed (tournament 15193603c3f8359b, mechanism study fafb771393aab7fe,
rule choice e6df56611a66c23f, generalization ee7faf1cff253ace, tracking benchmark e6c4f642f296c19b). The computation
did not change: every registered study was rerun under the new hashes (the tournament main run and its three
replications, the mechanism study, rule choice, generalization and the tracking benchmark) and reproduced every
registered number exactly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

TOURNAMENT_PLAN = "110b3146bb072c2c"
STUDY_PLAN = "11a279e507f246e2"
CHOICE_PLAN = "f15f62149d08e800"       # endogenous rule choice (Rule choice page)
TASK_PLAN = "b46c1f64a8250547"         # generalization tasks (Generalization page)
TRACK_PLAN = "f582721595727105"        # single-firm tracking benchmark (Solvable benchmark page)
# Human experiment protocol 2.0 (Play the market / Human experiments: analysis). It replaced protocol 1.0
# ("01f956595e90e5ab") on 7 October 2026, before any data were collected under either; not preregistered externally.
EXPERIMENT_PLAN = "e39ff2f6582cc591"
# Learnability study (heiner_abm.learnability): frozen in the repository on 7 October 2026, before its registered run;
# not preregistered with any external registry. Re-registered the same day when every policy was given the same tuning
# budget (see the revision note above); the run under the previous plan "ec9b781e125e289d" requires replication.
LEARN_PLAN = "8f43bedc7dae10bc"

# Agent tournament: mean profit rank and aggregate rank over six criteria (1 = best of 10), main run and three
# replications with fresh seeds, and the design selected on training data in the main run.
TOURNAMENT: Dict[str, Dict] = {
    "cobweb": dict(design="Adaptive price expectations", profit=(2.10, 3.53, 3.11, 3.24), aggregate=(3.50, 4.17, 3.00, 3.50)),
    "heuristic": dict(design="Target-margin rule", profit=(2.96, 2.41, 2.29, 2.79), aggregate=(4.50, 5.00, 3.83, 4.67)),
    "options": dict(design="Inaction-band heuristic · price-based target", profit=(3.50, 2.85, 4.10, 6.16),
                    aggregate=(3.67, 2.83, 4.00, 6.33)),
    "heiner": dict(design="Reliability condition · price-based target", profit=(4.29, 5.00, 6.10, 3.98),
                   aggregate=(4.17, 6.17, 4.50, 3.17)),
    "ecology": dict(design="Reorganize under threat of failure (scheduled reorganization in one replication)",
                    profit=(5.30, 4.64, 6.25, 3.60), aggregate=(3.33, 5.00, 6.83, 5.00)),
    "optimiser": dict(design="Rational expectations (Cournot–Nash)", profit=(6.33, 6.53, 5.80, 5.85),
                      aggregate=(5.83, 6.17, 6.67, 6.00)),
    "imitation": dict(design="Imitate the best (imitate the average in two replications)",
                      profit=(6.35, 6.00, 5.73, 6.00), aggregate=(9.00, 6.33, 7.67, 6.33)),
    "ruleb": dict(design="Rule B (rigid)", profit=(6.71, 7.49, 5.74, 6.89), aggregate=(6.00, 6.67, 4.17, 6.17)),
    "rl": dict(design="Softmax value learner", profit=(8.71, 7.68, 6.98, 7.69), aggregate=(9.17, 6.83, 5.67, 7.33)),
    "satisficing": dict(design="Aspiration search · price-based target (model-based in one replication)",
                        profit=(8.75, 8.89, 8.91, 8.81), aggregate=(5.33, 5.50, 8.50, 6.17)),
}

# Directional tournament of experiments (Competing theories page): matches / contradictions / inconclusive, for the
# Bertrand and the Cournot reference market, two seeds each (1,000 periods, 20 replications, H = 20). Seeds: 1 and 3
# (Bertrand), 1 and 42 (Cournot). Rerun on 6 October 2026 with the observable Adaptive rule; only the Cournot seed-1
# records changed. Rescored on 7 October 2026 with a prediction from all nine theories in every experiment
# (reinforcement learning and imitation added); see the revision notes under FINDING_FINGERPRINTS.
DIRECTIONAL: Dict[str, Dict[str, Tuple[str, str]]] = {
    "heiner": {"Bertrand": ("6/1/2", "5/1/3"), "Cournot": ("6/1/2", "6/1/2")},
    "neo": {"Bertrand": ("3/5/1", "2/5/2"), "Cournot": ("4/3/2", "4/3/2")},
    "options": {"Bertrand": ("3/5/1", "3/4/2"), "Cournot": ("2/5/2", "2/5/2")},
    "cobweb": {"Bertrand": ("1/7/1", "2/6/1"), "Cournot": ("4/5/0", "4/5/0")},
    "biasvar": {"Bertrand": ("5/2/2", "4/2/3"), "Cournot": ("5/2/2", "5/2/2")},
    "satisficing": {"Bertrand": ("5/3/1", "4/3/2"), "Cournot": ("4/3/2", "4/3/2")},
    "rl": {"Bertrand": ("3/4/2", "2/4/3"), "Cournot": ("4/3/2", "4/3/2")},
    "imitation": {"Bertrand": ("0/7/2", "1/6/2"), "Cournot": ("2/5/2", "2/5/2")},
    "ecology": {"Bertrand": ("4/5/0", "5/4/0"), "Cournot": ("3/6/0", "3/6/0")},
}

# Mechanism study findings that bear on each theory.
MECHANISMS: Dict[str, str] = {
    "heiner": (
        "With known reliability the reliability condition beats always adjusting toward the error-prone model-based "
        "target by +119 per period (95% CI 72–175) and breaks even near an error-to-signal ratio K ≈ 1. An agent that "
        "must learn its own reliability keeps little of the gain (+13 on the model-based target, −170 on the "
        "price-based target), mostly because it judges its past decisions with a misspecified model."),
    "optimiser": (
        "Always adjusting toward the filtered best reply is the reference rule. On the model-based target it is beaten "
        "by a reliability-based restriction that knows the true reliability (+119 per period), because a misspecified "
        "demand model makes the best reply unreliable after unannounced shifts."),
    "options": (
        "An inaction band on the model-based target cut downside risk in every run of the tournament; in the mechanism "
        "study it raised profit by +216 per period (95% CI 141–291) and broke even near K ≈ 1.03. On the reliable "
        "price-based target it cost profit (−137 per period)."),
    "cobweb": (
        "The adaptive price-expectations design needs no demand model; it ranked first on profit in the main "
        "tournament run and among the top three in every replication. In the mechanism study, restricting it "
        "(inaction band or reliability condition) lowered profit."),
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
 'imitation': 'imit_best',
 'ecology': 'ecol_crisis',
 'ruleb': 'ruleb'}
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
SPECIAL_RESULTS: Dict[str, Tuple[str, str]] = {'cobweb': ('supported',
            'n = 3: theory 1.00, simulated 0.95; n = 4: theory 0.80, simulated 0.80; n = 6: theory 0.57, simulated '
            '0.55; n = 8: theory 0.44, simulated 0.45'),
 'ecology': ('supported',
             'Δ = 2: -6 [-9, -3]; Δ = 10: -94 [-101, -87]; Δ = 20: -310 [-326, -293]; Δ = 30: -753 [-778, -728]'),
 'heuristic': ('not supported',
               "heuristic's advantage changes by -8.8 per unit of noise (p = 0.00777); at the highest noise +409 "
               '[+311, +525]'),
 'imitation': ('supported', 'imitate-the-best markets produce 1.24 [1.22, 1.26] × the Cournot–Nash output'),
 'optimiser': ('not supported',
               "profit slope on foresight -10.6 per unit (p = 0.0807); with full information the rational market's "
               'price is within 0.13% of Cournot–Nash on average'),
 'options': ('not supported', 'value of flexibility changes by -3.7 per unit of volatility (p = 0.0289)'),
 'rl': ('not supported',
        'relative profit improves by -73 [-283, +156] from the first to the last third in a stationary market, and '
        'by +148 [-163, +424] with unannounced shifts'),
 'satisficing': ('not supported', 'change rate changes by -0.0011 per unit of volatility (p = 0.00213)')}


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
# * 6 October 2026, terminology audit (wording in arena.py display names and docstring, params.py, experiments.py,
#   theories.py, agents.py, analysis.py; no computation changed; plan hashes unchanged). Every finding above was rerun
#   with tools/rerun_adaptive_findings.py, the signature tests (Full scale) and the field patterns (registered
#   settings), and every output was identical to the previous rerun. Re-fingerprinted.
# * 6 October 2026, estimation gap (engine.py, experiments.py): for the H-period measures the out-of-sample estimation
#   window now ends H - 1 periods before the evaluation window (before, the last H - 1 estimation decisions used costs
#   from the evaluation window). Rerun: horse_race changed slightly (dynamic RC AUC 0.602 -> 0.601, stakes 0.550 ->
#   0.548, accuracy 0.536 -> 0.539; encompassing gain +0.000 unchanged); README updated with the earlier values in a
#   revision note. directional, presets, preset_adaptive, special and patterns were rerun and are identical.
# * 6 October 2026, encompassing test (experiments.py): forecasts are mapped onto training-fold ranks before the logit
#   is fitted, and a single forecast is scored directly in the direction learned on the training folds. Rerun of
#   horse_race: every forecast AUC identical; encompassing table changed (rivals 0.851 -> 0.866, RC margin alone
#   0.515 -> 0.610, gain +0.000 [-0.0045, +0.0045] -> -0.001 [-0.003, +0.000]); README updated with a revision note.
#   directional, presets and preset_adaptive do not use encompassing_test; rerun and identical.
# * 6 October 2026, information-and-feedback specification (agents.py, engine.py, params.py, theories.py): firms decide
#   from an Observation built under Scenario.info; Adaptive feedback settings moved to InfoSpec. With the default
#   specification every finding was rerun (tools/rerun_adaptive_findings.py: directional B1, B3, C1, C42, presets,
#   horse_race; signature tests at the Full scale; field patterns at the registered settings) and every output was
#   identical. Re-fingerprinted.
# * 6 October 2026, selection gates for the Adaptive rule (gates.py added; agents.py, engine.py, params.py changed;
#   gates.py added to the sources of the four market findings). The default gate ("gain", no adjustment cost) is meant
#   to reproduce the earlier rule; the six findings were listed in REPLICATION_REQUIRED until rerun. Rerun on the
#   committed code against the code of commit 2bcff32 (tools/rerun_adaptive_findings.py: directional B1, B3, C1, C42,
#   horse_race, presets; signature tests at the Full scale; field patterns at the page defaults): every output was
#   byte-identical. Re-fingerprinted and removed from REPLICATION_REQUIRED.
# * 6 October 2026, params.py: the validation rule rejecting an adjustment cost together with evolution was removed
#   (no computation changed). Rerun on commit 1f96ca1 as above: every output byte-identical. Re-fingerprinted.
# * 6 October 2026, organizational ecology as the ninth theory (arena.py: two designs and selection rules; theories.py:
#   ecology's directional predictions for every experiment; special.py: ecology signature test; registered tuned
#   designs and parameters from the rerun ten-agent tournament). Rerun: directional B1, B3, C1, C42 — every experiment
#   bit-identical, only ecology's record changed (2/0/0, 2/0/0, 1/1/0, 1/1/0 -> 4/5/0, 5/4/0, 3/6/0, 3/6/0);
#   signature tests at the Full scale — optimization no longer supported, the other verdicts unchanged, ecology
#   supported, numbers updated in SPECIAL_RESULTS and the README with the earlier values; field patterns — excess
#   volatility no longer reproduced (4 of 6), README updated with the earlier values. horse_race, presets and
#   preset_adaptive do not depend on the changed files. Re-fingerprinted.
# * 7 October 2026, every theory tested in every experiment (theories.py: reinforcement learning and imitation added to
#   the directional tournament; a reasoned prediction filled in for every theory and experiment that had none). No
#   computation changed. Rerun with tools/rerun_adaptive_findings.py (directional B1, B3, C1, C42) on the committed code
#   and on the code of commit 4ce3132: the old code reproduced every record in DIRECTIONAL, and every experiment's
#   result was bit-identical on the new code; only the scoring changed. DIRECTIONAL updated, earlier records in the
#   README revision note. Heiner keeps the best net record in all four runs. Re-fingerprinted.

FINDING_SOURCES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "directional": ("Competing theories: directional tournament records (matches / contradictions / inconclusive)",
                    ("agents.py", "engine.py", "params.py", "gates.py", "analysis.py", "experiments.py", "theories.py")),
    "horse_race": ("Competing theories: out-of-sample forecasts of which firms beat their rigid twin (AUC)",
                   ("agents.py", "engine.py", "params.py", "gates.py", "analysis.py", "experiments.py")),
    "presets": ("Typical results of the preset scenarios without Adaptive agents (sidebar preset descriptions)",
                ("agents.py", "engine.py", "params.py", "gates.py", "analysis.py", "experiments.py")),
    "preset_adaptive": ("Typical result of the reliability-learning (Adaptive) preset",
                        ("agents.py", "engine.py", "params.py", "gates.py", "analysis.py", "experiments.py")),
    "special": ("Signature tests by theory",
                ("arena.py", "rulechoice.py", "special.py", "analysis.py", "agents.py", "params.py", "#tuned")),
    "patterns": ("Field patterns (pattern-oriented validation)",
                 ("arena.py", "rulechoice.py", "patterns.py", "agents.py", "params.py", "#tuned")),
    "calibration": ("Calibration to experiments: recovery of generating rules on synthetic subjects",
                    ("calibration.py",)),
}

FINDING_FINGERPRINTS: Dict[str, str] = {
    "directional": "b77ae980f52f8e79",
    "horse_race": "94b4811784863c0a",
    "presets": "94b4811784863c0a",
    "preset_adaptive": "94b4811784863c0a",
    "special": "71d28315d7da1800",
    "patterns": "c18e9f0d5c24e544",
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
