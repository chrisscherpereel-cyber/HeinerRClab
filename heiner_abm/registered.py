"""Registered results reported on the theory pages.

These numbers come from the registered runs documented in the README. They are tied to the plan hashes below:
tests/test_theory_pages.py fails when the code changes the hashes, so the numbers cannot silently go stale.

Revision note (9 October 2026, three separate adaptation costs): the inventory study charged one cost, for departing
from the default, and the same indicator also *defined* the reported adaptation rate, so "departed from the default"
and "actually changed the order" could not be told apart. Three distinct frictions are now modeled and reported
separately: a default-departure overhead (Config.cost), a fixed switching cost (Config.switch_cost) and a magnitude
cost per unit moved (Config.magnitude_cost); `Config.cost_model = "legacy_departure_only"` reproduces the previous
calculation exactly. Outcomes now report gross payoff, each charge, net payoff, departure_rate, adjustment_rate and
adjustment_magnitude; tuning scores the same net payoff it is evaluated on. The same vocabulary is applied across
tasks, with their differences stated rather than glossed: in the market the default is rule B ("keep last output"),
so departure and switching are the same event there and only the magnitude charge is separately identified
(MarketConfig.magnitude_cost); in the NK task there is no default to depart from, and its existing `cost` is a
magnitude charge per changed component, now joined by a distinct NKEnv.switch_cost. These are not three
parameterizations of one construct. The study code changed, so its plan hash changed, 6fbc88a80332fc38 →
a98b971b49fe5e3e, and the whole study was rerun; the numbers below and in docs/learnability_results/ are from that
rerun. Results under 6fbc88a80332fc38 are kept in the README's revision notes and REQUIRE REPLICATION. No other plan
hash or finding fingerprint is affected.

Revision note (8 October 2026, five defect repairs): five defects were confirmed against the code and repaired.
(1) The workbench built its hyperparameters without a "gate" entry, so the confidence-sensitive gate silently used
learnability.GATE_HP0 and neither its control nor a tuned value reached gated(); the gate is now exposed as controls,
forwarded, and tuned with the same four candidates as the forecast gain and the band. (2) BOCPD advanced its
run-length posterior only when a demand was observed, so a gap in the data froze regime uncertainty; calendar time
and conditioning are now separate (BOCPD.advance()), exactly one hazard transition per period. (3) The NC3 negative
control passed when the *upper* confidence limit exceeded −0.25, which only fails when the gate is confidently
much worse; it is now a noninferiority test on the *lower* limit at a prespecified margin (learnability.NC3_MARGIN).
(4) NK landscape_path redrew max(1, round(frac * N)) components, so change_frac = 0 still changed the landscape;
zero now means no modification. (5) learnability_market.ratio counted feedback in the regime of the decision rather
than of its release, crediting a regime with feedback that arrived only after it ended; feedback is now counted by
release period and decisions whose judgement window crosses a regime boundary are excluded from both the regime mean
and its count. Repairs (2), (3) and (5) change the learnability study's code, so its plan hash changed,
8f43bedc7dae10bc → 6fbc88a80332fc38, and the whole study was rerun; the numbers below and in
docs/learnability_results/ are from that rerun. Results under 8f43bedc7dae10bc are kept in the README's revision
notes and REQUIRE REPLICATION. Repairs (1) and (4) touch no registered hash: the tournament plan, the mechanism and
rule-choice plans and all seven finding fingerprints are unchanged.

Revision note (8 October 2026, equal search density in the tournament): every tournament design was tuned with the
same *number* of candidate settings, but designs carry one to six free parameters, so the same budget searched a
six-dimensional space six times less thoroughly than a one-dimensional one — and the two reliability-condition
designs have the most free parameters of any theory, so the rule searched the focal theory's space least thoroughly.
Prereg now gives every design `budget_per_parameter` candidates *per free parameter* (24, so 24–144 per design; no
design is searched less than under the old flat budget of 24). `Prereg.budget_rule = "per_design"` restores the old
rule. arena.code_digest() changed, so the tournament, the mechanism study and the rule-choice study were re-registered
and rerun, together with the signature tests and the field patterns, which use the tuned agents:
tournament 110b3146bb072c2c → 22393384781bed84, mechanism study 11a279e507f246e2 → bbfaed9e70b900d6, rule choice
f15f62149d08e800 → d2f88699dc6e6eb6, signature-test fingerprint 71d28315d7da1800 → 1ca69a156d8a3b3a, field-pattern
fingerprint c18e9f0d5c24e544 → 899d4e263e8e7c5e. Design selection was unchanged for every theory; the tuned
parameters, the ranks and several effects were not. Two conclusions moved, in opposite directions: reliability-based
selection on the model-based target now raises profit significantly in all four runs (+23, +57, +109, +46) where it
was mixed before, while the reliability-condition agent's head-to-head profit rank got worse (4.29 → 7.19 in the main
run) because the mid-dimensional rivals gained more from the larger search than it did. The ecology signature test
flipped from supported to not supported. All earlier values under the superseded plans are kept in the README's
revision notes and REQUIRE REPLICATION.

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

TOURNAMENT_PLAN = "22393384781bed84"
STUDY_PLAN = "bbfaed9e70b900d6"
CHOICE_PLAN = "d2f88699dc6e6eb6"       # endogenous rule choice (Rule choice page)
TASK_PLAN = "b46c1f64a8250547"         # generalization tasks (Generalization page)
TRACK_PLAN = "f582721595727105"        # single-firm tracking benchmark (Solvable benchmark page)
# Human experiment protocol 2.0 (Play the market / Human experiments: analysis). It replaced protocol 1.0
# ("01f956595e90e5ab") on 7 October 2026, before any data were collected under either; not preregistered externally.
EXPERIMENT_PLAN = "e39ff2f6582cc591"
# Learnability study (heiner_abm.learnability): frozen in the repository on 7 October 2026, before its registered run;
# not preregistered with any external registry. Re-registered the same day when every policy was given the same tuning
# budget (see the revision note above); the run under the previous plan "ec9b781e125e289d" requires replication.
LEARN_PLAN = "a98b971b49fe5e3e"

# Agent tournament: mean profit rank and aggregate rank over six criteria (1 = best of 10), main run and three
# replications with fresh seeds, and the design selected on training data in the main run.
TOURNAMENT: Dict[str, Dict] = {
    "heuristic": dict(design="Target-margin rule", profit=(2.01, 3.08, 2.0, 3.67), aggregate=(4.0, 4.33, 4.33, 4.5)),
    "cobweb": dict(design="Adaptive price expectations", profit=(3.91, 2.8, 5.01, 1.82), aggregate=(4.0, 2.5, 2.83, 2.33)),
    "optimiser": dict(design="Rational expectations (Cournot–Nash)", profit=(5.11, 6.11, 6.58, 5.5), aggregate=(3.5, 5.0, 7.5, 4.5)),
    "ecology": dict(design="Reorganize under threat of failure (scheduled reorganization in one replication)", profit=(5.3, 4.03, 3.77, 5.74), aggregate=(6.0, 4.83, 4.5, 7.17)),
    "options": dict(design="Inaction-band heuristic · price-based target", profit=(5.31, 4.19, 3.67, 2.98), aggregate=(6.17, 2.67, 4.0, 3.83)),
    "ruleb": dict(design="Rule B (rigid)", profit=(5.55, 6.49, 6.11, 6.84), aggregate=(3.83, 5.83, 5.0, 5.83)),
    "imitation": dict(design="Imitate the best (imitate the average in two replications)", profit=(5.89, 6.08, 5.9, 6.05), aggregate=(8.33, 8.83, 6.67, 6.0)),
    "rl": dict(design="Softmax value learner", profit=(6.34, 7.05, 7.01, 7.42), aggregate=(5.83, 6.0, 6.0, 6.67)),
    "heiner": dict(design="Reliability condition · price-based target", profit=(7.19, 5.9, 5.86, 5.8), aggregate=(7.67, 7.0, 7.17, 7.33)),
    "satisficing": dict(design="Aspiration search · price-based target", profit=(8.39, 9.29, 9.07, 9.18), aggregate=(5.5, 7.83, 6.33, 6.83)),
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
        'With known reliability the reliability condition beats always adjusting toward the error-prone model-based '
        'target by +57 per period (95% CI 31–84) and breaks even near an error-to-signal ratio K ≈ 0.98. An agent '
        'that must learn its own reliability keeps little of that gain (+10 [5, 15] on the model-based target; +10 '
        '[−20, 44], not distinguishable from zero, on the price-based target): the total cost of learning is +48 '
        '[21, 74] on the model-based target, almost all of it from judging past decisions with a misspecified model '
        '(+49 [7, 97]) rather than from estimation noise (−1 [−49, +43]).'
    ),
    "optimiser": (
        'Always adjusting toward the filtered best reply is the reference rule. On the model-based target it is '
        'beaten by a reliability-based restriction that knows the true reliability (+57 per period), because a '
        'misspecified demand model makes the best reply unreliable after unannounced shifts.'
    ),
    "options": (
        'An inaction band on the model-based target cut downside risk in every run of the tournament; in the '
        'mechanism study it raised profit by +165 per period (95% CI 110–219) and broke even near K ≈ 0.98. On the '
        'reliable price-based target it also raised profit (+92 [79, 107]).'
    ),
    "cobweb": (
        'The adaptive price-expectations design needs no demand model; it ranked second on profit in the main '
        'tournament run, behind the target-margin rule, and first in two of the three replications. In the mechanism '
        'study, restricting the price-based target with an inaction band raised profit (+92 [79, 107]); restricting '
        'it with the learned reliability condition did not change it detectably (+10 [−20, 44]).'
    ),
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
TUNED_PARAMS: Dict[str, Dict[str, float]] = {'heiner_m': {'a_cost': 0.226082,
              'a_rival': 0.078607,
              'phi': 0.063212,
              'theta': 13.961262,
              'memory': 0.922407,
              'horizon': 15.0},
 'heiner_p': {'lam': 0.987565, 'phi': 0.874157, 'theta': 8.889382, 'memory': 0.973606, 'horizon': 3.0},
 'opt_br': {'a_cost': 0.112079, 'a_rival': 0.071136, 'phi': 0.067446},
 'opt_nash': {'a_cost': 0.105048, 'phi': 0.955373},
 'options_m': {'a_cost': 0.090808, 'a_rival': 0.228507, 'phi': 0.11664, 'k': 2.781461},
 'options_p': {'lam': 0.932418, 'phi': 0.947886, 'k': 0.244807},
 'cobweb_p': {'lam': 0.942435, 'phi': 0.580972},
 'cobweb_q': {'a_rival': 0.139733, 'phi': 0.052618},
 'heur_wsls': {'step': 5.803435},
 'heur_markup': {'phi': 1.820927, 'm': 3.738067},
 'satis_m': {'a_cost': 0.592298, 'a_rival': 0.081083, 'phi': 0.087846, 'alpha': 0.163209},
 'satis_p': {'lam': 0.088779, 'phi': 0.079298, 'alpha': 0.255155},
 'rl_softmax': {'step': 2.157201, 'eta': 0.29541, 'temp': 1.760235},
 'rl_erevroth': {'step': 1.058066, 'forget': 0.001552, 'gain': 0.196501},
 'imit_best': {'p': 0.237141, 'noise': 3.455233},
 'imit_avg': {'p': 0.094798, 'noise': 3.334032},
 'ecol_periodic': {'lam': 0.061428, 'phi': 0.319639, 'interval': 18.0},
 'ecol_crisis': {'lam': 0.932537, 'phi': 0.761047, 'floor': 0.349556, 'alpha': 0.07261},
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
            'n = 3: theory 1.00, simulated 0.95; n = 4: theory 0.80, simulated 0.80; n = 6: theory 0.57, '
            'simulated 0.55; n = 8: theory 0.44, simulated 0.45'),
 'ecology': ('not supported',
             'Δ = 2: +2 [+2, +3]; Δ = 10: +62 [+45, +77]; Δ = 20: +194 [+176, +215]; Δ = 30: +255 [+241, '
             '+269]'),
 'heuristic': ('not supported',
               "heuristic's advantage changes by -29.3 per unit of noise (p = 1.23e-10); at the highest noise "
               '+500 [+410, +605]'),
 'imitation': ('supported', 'imitate-the-best markets produce 1.25 [1.24, 1.27] × the Cournot–Nash output'),
 'optimiser': ('not supported',
               'profit slope on foresight -35.9 per unit (p = 0.000258); with full information the rational '
               "market's price is within 0.13% of Cournot–Nash on average"),
 'options': ('not supported', 'value of flexibility changes by -3.9 per unit of volatility (p = 0.00043)'),
 'rl': ('not supported',
        'relative profit improves by -72 [-247, +104] from the first to the last third in a stationary market, '
        'and by -202 [-386, -38] with unannounced shifts'),
 'satisficing': ('not supported', 'change rate changes by -0.0017 per unit of volatility (p = 5.49e-21)')}


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
    "special": "1ca69a156d8a3b3a",
    "patterns": "899d4e263e8e7c5e",
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
