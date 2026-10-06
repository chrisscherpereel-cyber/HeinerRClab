# Decision making under uncertainty: an agent-based laboratory

A Streamlit agent-based simulation laboratory that compares eight theories of when a decision maker should adapt
and when it should stick to a rule: Heiner's reliability condition, neoclassical optimization, real options, cobweb
theory and adaptive expectations, simple heuristics (bias–variance), satisficing, reinforcement learning, and
imitation and evolutionary selection. All are implemented in the same cobweb oligopoly and tested on equal terms.
Every hypothesis, and every alternative to it, is grounded in published research (see *Research basis* below).

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Tests (agent/engine equivalence, RC accounting identity, literature-registry integrity, headless smoke test of every page):

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository and push the contents of this folder to it, with `app.py` at the repository root.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, and choose **Deploy a public app from GitHub**.
3. Select the repository and branch, and set **Main file path** to `app.py`.
4. Optional: under **Advanced settings**, choose Python 3.11 or 3.12.
5. Click **Deploy**. Dependencies are installed from `requirements.txt`. The repository has no
   `.streamlit/config.toml`, so Streamlit's default theme is used.

No secrets or data files are needed. The repository has no continuous-integration workflow; run the test suite
locally (see above) before pushing.

**Resource note:** Community Cloud apps have about 1 GB of memory. The default experiments use well under that. Very
large runs can hit the limit: for example, hundreds of environments at H = 100, or tens of thousands of periods with
many replications. For heavy research runs, use a local installation.

## Pages

The left panel lists the eight **theories** first. Directly beneath them is the **theory under test**: a selector that
chooses which theory is highlighted. Every hypothesis card then shows that theory's prediction first (from the
registry, the directional tournament, a statement derived from the theory's core claim, or, where it makes none, its
general stance, labeled as such), with the competing predictions beside it and any theory-specific reasoning under its
own prediction. The research panels, the overview and research tables, the competing-theories verdicts and the agent
tournament's head-to-head follow the same choice. Heiner's reliability condition is the default, not a privileged
position. The other sections (Simulate, Special tests, Validate & generalize, Reference) and the base-scenario
settings are collapsed by default; the section holding the current page opens.

**Theories**

| Page | What it does |
|---|---|
| Overview | The eight theories side by side (core claim, view of uncertainty, what triggers change, effect of uncertainty on the value of flexibility), the common testbed, how each fared, every hypothesis with its research, and the contribution to the literature |
| One page per theory | Heiner, optimization, real options, cobweb, simple heuristics, satisficing, reinforcement learning, imitation. Same sections for each: origins, formal core, view of flexibility, an interactive illustration, how the laboratory implements it, how it fared in the registered runs, strengths and limits, references |

**Simulate** (experiments that test every theory on equal terms; hypotheses are stated as neutral questions)

| Page | What it does |
|---|---|
| Market lab | One market with editable heterogeneous firm agents. Shows price/cost dynamics, a per-firm reliability scoreboard, and a period-by-period decision inspector (correct deviations, type I and type II errors) |
| Hypothesis tests | H1 free flexibility · H2 profitability switch · H3 volatility · H4 fixed costs · H5 competition intensity · H6 regimes and equilibrium · H7 competence · H8 perception noise · H9 selection rules · H10 predictable behavior · H11 number of rivals · H12 model-updating lag. Each shows the RC prediction next to the alternative, the research behind both, the contribution, and a verdict |
| Risk vs Knightian uncertainty | Cost-volatility risk versus unannounced demand-regime shifts at matched unpredictability, for model-based (Cournot) and model-free (Bertrand) firms. Event study of punctuated slow–quick–slow adjustment |
| Endogenous flexibility | Firms imitate the most profitable rival's φ (plus mutation). Does volatility breed rigidity? |
| Competing theories | Heiner's RC against neoclassical optimization, real options, cobweb stability, bias–variance / ecological rationality, satisficing and structural inertia. A tournament of nine discriminating experiments scores each theory's directional predictions; an out-of-sample horse race scores each theory's forecast of which firms benefit from flexibility, plus an encompassing test of whether the RC adds information beyond all rivals |
| Agent tournament | Every rival theory implemented as two agent designs competing in the same market (17 designs, including target × selection-rule composites). Equal tuning budget per design on training environments, design selection on training data, held-out test environments, a frozen hashed plan with six pre-registered hypotheses, six performance criteria (profit, downside risk, survival, volatility, regret, worst case), a selection-rule experiment, invasion tests, global sensitivity analysis and replication across seeds |
| Rule choice (emergence) | Firms switch between six rules (three restricted, three flexible) by recent performance with logit choice and an adjustable intensity of choice β (Brock & Hommes 1997). Rule shares, change rates, price volatility and distance from Cournot–Nash across uncertainty levels. Frozen plan with four pre-registered hypotheses |
| Experiment designer | Your own *what if* question: sweep one or two settings of the base market and plot any outcome, with common random numbers across conditions. The page explains its purpose, the steps, a worked example and what each outcome means; CSV export |

**Special tests** (each theory's signature prediction)

| Page | What it does |
|---|---|
| Signature tests by theory | One test per rival theory of the prediction that characterizes it, with a criterion fixed in advance (results below); Heiner's tab links to his five special-test pages |
| Heiner: does the RC predict performance? | Random environments; each firm is compared with its own rigid twin (same shocks). The RC is estimated in the first part of each run and predicts the second part (**out-of-sample**), scored by AUC with environment-clustered bootstrap CIs |
| Heiner: dynamic RC (1989) | Decomposes each decision's value into immediate, persistence and strategic-feedback parts. Tests Heiner's (1989) partial-adjustment bound β₀ = 1/((1+K)(1−f′)) against the profit-maximizing flexibility. Signal-detection ROC of each firm's decisions |
| Heiner: CD-gap explorer | Difficulty (Δ or noise) × competence (foresight κ) heatmaps of r, w, π, RC margin and the payoff to flexibility |
| Heiner: mechanisms (oracle vs learned) | Principle versus implementation: focal-firm variants on a shared target (always, inaction band, learned reliability condition, the same learner judging with the true model, an oracle with true reliability, a memory grid), with the cost of applying the principle decomposed into estimation and model bias; a boundary test against the measured error-to-signal ratio K; standardized effects of each source of uncertainty; cross-validated metamodel maps of which theory does best where. Frozen study plan with five pre-registered hypotheses |
| Heiner vs optimal filtering (Muth–Kalman) | A single firm tracks a random-walk target observed with noise, where the best adjustment speed is the Kalman gain. The simulation reproduces the exact loss curve and optimum; then lopsided stakes test the prediction that is uniquely Heiner's against optimal filtering's certainty equivalence. Frozen plan with four pre-registered hypotheses |

**Validate & generalize**

| Page | What it does |
|---|---|
| Field patterns | Pattern-oriented validation (Grimm et al. 2005): cobweb cycles, damping by adaptive adjustment, sticky and lumpy adjustment, imitation beyond Cournot–Nash, excess volatility around equilibrium and positive markups, each with a criterion fixed in advance and its sources |
| Calibration to experiments | Fits every theory's decision rule per subject to learning-to-forecast cobweb data (Hommes et al. 2007 design) or Cournot data (Huck et al. 1999 design) on the first half of periods and scores it on the second half; classifies subjects by best-predicting rule. Upload data or check recovery on synthetic subjects |
| Empirical validation (public data) | The five public datasets that can validate the simulation, their access, licenses and caveats; a protocol fixed in advance (seven hypotheses); loaders that read each repository's files as distributed; per-dataset analyses (out-of-sample rule comparison, generative check of simulated Cournot markets, newsvendor patterns, structural changes, time pressure); registered results |
| Play the market | A person runs one firm against three agent rivals in three counterbalanced blocks (low, medium, high uncertainty) of 25 periods; every agent design records in shadow mode what it would have chosen. Download the decisions as CSV |
| Experiment analysis | Pools participants' files, classifies each person by the best-predicting design, and tests X1 (fewer changes under high uncertainty) and X2 (restraint pays under high uncertainty). Synthetic demonstration clearly labeled |
| Generalization | The same selection layers and boundary test in three other decision tasks with a default, a flexible alternative and a difficulty–competence gap: an inventory (newsvendor) task with shifting demand, a learning task with shifting payoffs and an irreversible investment task. Frozen plan with four pre-registered hypotheses |

**Reference**

| Page | What it does |
|---|---|
| Agents as implemented | Every agent in the laboratory as the code implements it: the shared decision cycle, the market-lab firms (production and selection rules, adaptive learning, endogenous flexibility), the seventeen tournament designs with their equations, parameters, registered tuned values and sources, the mechanism variants, the rule-choosing firms, the task agents, the tracking rules, the experiment's rivals and shadows, and the rules fitted to human data |
| Research & contribution | The simulation's contributions to the literature, the evidence matrix (supporting and alternative research for every hypothesis), the research behind each rival theory and method, and the full bibliography with BibTeX, APA and CSV export |
| Model & methods | Equations, schedule, measurement, statistics, baseline calibration, a full description of every preset scenario (setup, why to run it, the hypotheses it serves and a typical result), what a pre-registered plan is and why it is used, and verification |

**Pre-registered plans.** Before a confirmatory study is run, its plan (hypotheses, tests and decision rules,
environments, sample sizes, tuning budgets and seeds) is fixed and, together with the agent and analysis code, turned
into a short hash. This prevents choosing tests or settings after seeing the results, makes the comparison between
theories fair, and lets anyone verify that a reported result came from exactly that plan and code. Any change produces
a new hash and the run is labeled exploratory. Every page with a plan explains this in an expandable note.

*What this does not establish.* "Pre-registered" here means frozen in the code and identified by a hash. The
repository contains no record of any plan being deposited with an external, time-stamped registry (such as OSF or
AsPredicted), so the hash shows which plan and code produced a result but not that the plan was fixed before the
results were seen. Some plans were revised during development before they were frozen (see the mechanism study).

**Findings without a frozen plan.** Some reported results were produced without a frozen plan: the directional
tournament and the out-of-sample forecasts on the *Competing theories* page, the typical results of the presets, the
signature tests, the field patterns and the calibration recovery rates. Each is tied to a fingerprint of the source
files that produce it (`FINDING_SOURCES` and `FINDING_FINGERPRINTS` in `heiner_abm/registered.py`). When one of those
files changes, the test suite fails until the finding is either rerun (and re-fingerprinted with a revision note) or
listed in `REPLICATION_REQUIRED`. A listed finding keeps its originally reported numbers and is flagged *requires
replication* here, on every page that reports it, and in a status table on the *Model & methods* page. The baseline
fingerprints were taken from the code of 6 October 2026 (commit `469619d`); the findings were not rerun at that time.

## Signature tests by theory

Each rival theory's distinctive prediction, tested with its own agents in the shared market (Full scale; deterministic
given the code):

| Theory | Signature prediction | Result | Verdict |
|---|---|---|---|
| Neoclassical optimization | Better information raises an optimizer's profit; full information yields Cournot–Nash | profit slope on foresight +40.6 per unit (p = 0.0116); with full information the rational market's price is within 0.13% of Cournot–Nash on average | **supported** |
| Real options | The value of flexibility rises with volatility | value of flexibility changes by -11.2 per unit of volatility (p = 1.03e-15) | not supported |
| Cobweb theory | Partial best-reply adjustment is stable only below φ = 4/(n + 1) | n = 3: theory 1.00, simulated 0.95; n = 4: theory 0.80, simulated 0.80; n = 6: theory 0.57, simulated 0.55; n = 8: theory 0.44, simulated 0.45 | **supported** |
| Simple heuristics | A simple rule beats the optimizer by more as estimation noise rises | heuristic's advantage changes by -26.0 per unit of noise (p = 8.1e-06); at the highest noise +684 [+564, +826] | not supported |
| Satisficing | Harder (more volatile) conditions trigger more search | change rate changes by -0.0002 per unit of volatility (p = 0.386) | not supported |
| Reinforcement learning | Performance improves with experience in a stationary environment | relative profit improves by -138 [-488, +230] from the first to the last third in a stationary market, and by -190 [-487, +112] with unannounced shifts | not supported |
| Imitation | Imitating the best drives output above Cournot–Nash | imitate-the-best markets produce 1.24 [1.22, 1.26] × the Cournot–Nash output | **supported** |

Three of seven signatures hold in this market. The cobweb stability boundary is reproduced almost exactly (0.95, 0.80,
0.55 and 0.45 against 1.00, 0.80, 0.57 and 0.44), and imitation of the best pushes output 24% above Cournot–Nash, as
Vega-Redondo (1997) predicts. Real options' signature fails: always adjusting loses more to its rigid twin as
volatility rises. The heuristic beats the optimizer at every noise level, but its advantage shrinks rather than grows
with noise; the aspiration searcher does not change more often in more volatile markets; and the tuned reinforcement
learner does not improve with experience over 1,500 periods.

## Competing theories: reference results

> **Revision note (6 October 2026).** This section was rerun after the Adaptive selection rule was changed to learn
> only from information available to the agent (it previously learned from the researcher's look-ahead
> counterfactual). The protocol was first checked on the old code, which reproduced every number reported here
> before. Only experiment E7 (the Adaptive agents' deviation rate) uses that rule; every other experiment gave
> bit-identical results. Under the new rule E7's deviation rate falls significantly with volatility in every run
> (Bertrand slope about −0.008 instead of −0.011; Cournot about −0.019 instead of −0.002, which had not been
> significant in the first Cournot seed). As a result the first Cournot run changed: Heiner 5/1/3 → 6/1/2,
> neoclassical 3/1/3 → 3/2/2, real options 1/3/3 → 2/3/2, satisficing 0/0/1 → 0/1/0. The forecasting results moved
> by at most 0.02 AUC and their ranking and conclusions are unchanged. Earlier values: Cournot records in the table
> as given above; AUCs 0.61 (dynamic RC), 0.58, 0.56, 0.56, 0.54, 0.47, track record 0.86, encompassing gain −0.001,
> one-shot RC 0.55.

From the *Competing theories* page with 1,000 periods, 20 replications per condition and H = 20, for the default
Bertrand market and a Cournot market (φ = 0.1–0.4), each with two seeds (1 and 3 for Bertrand, 1 and 42 for Cournot;
the second seeds were not recorded originally and were identified as those that reproduce the earlier records on the
old code). Across eight seeds per market (1, 3, 21, 42, 100, 1000, 12345, 2026) Heiner has the best net record in
every run:

| Theory | Tournament record, Bertrand (✅ / ❌ / ➖) | Tournament record, Cournot (✅ / ❌ / ➖) |
|---|---|---|
| Heiner: reliability condition | 6/1/2 and 5/1/3 | 6/1/2 (both seeds) |
| Bias–variance / ecological rationality | 4/2/0 and 3/2/1 | 3/2/1 (both seeds) |
| Cobweb stability theory | 1/5/1 and 2/4/1 | 4/3/0 (both seeds) |
| Real options | 2/4/1 and 2/3/2 | 2/3/2 (both seeds) |
| Neoclassical optimization | 2/4/1 and 1/4/2 | 3/2/2 (both seeds) |

Heiner had the best net record in all four runs. Its one contradiction differs by market: in Bertrand markets
perception noise *raised* the payoff to flexibility, and in Cournot markets free flexibility did not hurt at the
tested low-profit level. Evolved flexibility showed no volatility gradient in any run. The best adjustment speed
fell with noise, as Heiner predicts, but optimal filtering (Muth 1960; Kalman 1960) predicts the same, so that test
does not discriminate between them.

Out-of-sample forecasting (100 random environments × 2 replications, 800 firms, both production rules): the dynamic
RC was the best theory-based forecast of which firms beat their rigid twin (AUC 0.60, 95% CI 0.55–0.65), ahead of
cobweb stability (0.55), stakes only (0.55), Heiner's K (0.55), accuracy only (0.54) and real options (0.47). A firm's
own track record in the first half of the run was far better (AUC 0.85), and adding the RC to all rival forecasts
combined changed cross-validated AUC by +0.000 (95% CI −0.004 to +0.005). Heiner's one-shot (one-period) RC scored
only 0.54. (Sampling seed 12345, base-scenario seed 1, H = 20, full measure.)

## Agent tournament: registered results

> **Revision note (2 October 2026).** The text of the agents and hypotheses was converted to American spelling. Plan hashes include that text, so the tournament, mechanism-study, rule-choice, generalization and tracking hashes changed (from `15193603c3f8359b`, `fafb771393aab7fe`, `e6df56611a66c23f`, `ee7faf1cff253ace` and `e6c4f642f296c19b`). The computation did not change: every registered study was rerun under the new hashes and reproduced every number reported here exactly.

Registered plan `9699f86a6a899cf7`: two designs per theory, each tuned with a budget of 24 settings × 2 rounds on 24
training environments; each theory enters with the design that scored higher on training data; evaluation on 40 × 2
held-out environments (800 periods); six criteria; invasion with 4 residents + 1 mutant; three replications with
fresh seeds.

| Theory (selected design) | Profit rank: main / reps | Aggregate rank, six criteria: main / reps |
|---|---|---|
| Cobweb: adaptive price expectations | 1.89 / 2.91, 2.74, 2.64 | 3.50 / 2.67, 2.33, 2.83 |
| Simple heuristics: target-margin rule | 2.79 / 3.30, 1.91, 1.65 | 4.00 / 4.17, 4.00, 3.83 |
| Real options: inaction band, price-based target | 3.41 / 2.99, 3.24, 4.21 | 3.33 / 3.00, 3.83, 3.67 |
| Heiner: reliability condition, price-based target | 3.98 / 4.89, 6.00, 5.42 | 4.17 / 5.00, 3.33, 6.00 |
| Optimization: rational expectations (Cournot–Nash) | 5.84 / 5.62, 5.90, 5.36 | 4.83 / 5.50, 6.83, 4.67 |
| Reinforcement learning: softmax value learner | 6.20 / 6.65, 5.85, 6.20 | 8.17 / 6.33, 5.17, 6.67 |
| Imitation (best or average) | 6.24 / 4.35, 5.32, 5.29 | 6.00 / 6.83, 7.00, 5.00 |
| Rule B (rigid benchmark) | 6.28 / 6.41, 5.72, 5.62 | 5.50 / 5.83, 5.33, 5.67 |
| Satisficing: aspiration search | 8.39 / 7.88, 8.31, 8.60 | 5.50 / 5.33, 7.17, 6.00 |

Selection rule versus always adjusting toward the same target (difference in profit and in CVaR 5%; * = 95% CI
excludes 0; main run / three replications):

| Selection rule · target | Profit | Downside (CVaR 5%) |
|---|---|---|
| Inaction band · model-based | +119* / +23, +70, +6 | +3349* / +1074*, +2093*, +1528* |
| Reliability condition · model-based | +31 / +30*, −45*, +13 | +599* / −102, −308*, −42 |
| Reliability condition · price-based | −339* / −265*, −364*, −435* | −359* / −818*, −708*, −992* |
| Aspiration · price-based | −983* / −756*, −808*, −912* | +699* / −106, −275, +169 |

None of the six pre-registered hypotheses was supported in the main run; PR2 was supported in one replication.

**What the tournament shows**

* *Model-free beats model-based.* In every theory that had the choice, the design that needs no demand model was
  selected on training data, and the top three in every run are model-free. With unannounced regime shifts, a
  misspecified model is the dominant source of decision error.
* *Restricting an unreliable flexible rule pays, as Heiner argued.* An inaction band on the error-prone model-based
  target cut downside risk in every run without costing profit. Restricting the already reliable price-based target
  hurt.
* *Heiner's own learned selection rule does not deliver that benefit robustly.* Estimating π, r, w, G and D
  case by case is itself a hard inference problem. The learned reliability condition is a flexible rule about when to
  be flexible, and it is noisy; simple fixed restrictions (an inaction band, a target margin) capture the gains
  without the estimation error. This is consistent with Heiner's deeper argument that reliable behavior comes from
  rules rather than case-by-case assessment, and it qualifies the use of the condition as an agent's decision rule.
* *Criteria matter.* Ranking by profit alone, by downside risk or by survival gives different orders; seven of nine
  theories are Pareto-efficient in the main run, so claims of superiority must name the criterion.

Adding the instrumentation used by the mechanism study changed the plan's hash (it covers the code) but not a single
result: the registered run reproduces the earlier one exactly.

## Mechanism study: registered results

Study plan `e9b5cbda8a4c4ab7` (120 environments drawn by Latin hypercube over separate sources of uncertainty, 800
periods, oracle values from 3,200-period independent runs, background rivals tuned under tournament plan
`9699f86a6a899cf7`). A focal firm aims at the same target as the always-adjusting rule and differs only in when it
moves. Profit per period relative to always adjusting (95% CI):

| Selection rule | Model-based target | Price-based target |
|---|---|---|
| Reliability condition, oracle (true reliability) | **+158 [91, 233]** | −29 [−58, −4] |
| Reliability condition, learned with the true model | +18 [−39, 81] | −35 [−58, −10] |
| Reliability condition, learned (own model) | −14 [−32, 2] | −132 [−173, −91] |
| Inaction band | +28 [−5, 62] | −144 [−165, −122] |

Cost of applying the principle (oracle − learned): model-based target +172 [102, 246], of which estimation from
limited experience +140 [42, 238] and judging with a misspecified model +32 [−28, 94]; price-based target +103
[64, 140], of which estimation +5 [−22, 30] and model bias +97 [61, 131]. Pooling more experience (memory up to
0.999) moved the learned agent only from −14 to +1, far from the oracle's +158.

Boundary: on the model-based target the oracle's gain rises with the flexible rule's error-to-signal ratio K (slope
+1,401 per unit of ln K, p = 0.011) and breaks even at K ≈ 0.97; the inaction band's gain breaks even at K ≈ 1.06
(p < 0.001). Restriction pays once the flexible rule's error is about as large as the adjustment it should make. The
gain is driven by misspecification (shift hazard +0.22 [0.05, 0.39], model lag +0.29 [0.12, 0.44], standardized)
rather than by risk (volatility +0.06 [−0.14, 0.23]).

Pre-registered verdicts: M1 (the principle pays with known reliability) **supported**; M2 (applying it is costly on
both targets) **supported**; M3 (model bias on the model-based target) not supported; M4 (pooled boundary slope)
not supported, p = 0.07, although the model-based slope alone is significant; M5 (misspecification minus risk
coefficient) not supported, CI [−0.13, 0.46].

Maps: the metamodels' cross-validated R² ranges from 0.0 (simple heuristics, reinforcement learning) to 0.6
(adaptive expectations), so the winner maps are only partly predictable from the environment; the page reports the
cross-validated R² with every map.

**What the mechanism study adds.** Heiner's principle is sound: knowing when deviations are reliable is worth a
large gain exactly where the flexible rule is error-prone, with a break-even near K = 1, and it is worth nothing (or
slightly negative) where the flexible rule is already reliable. The tournament results are therefore not a failure
of the principle but of its implementation: an agent that must learn its own reliability loses the whole gain, mostly
because limited and non-stationary experience makes the estimates noisy (model-based target), or because it judges
its past decisions with the wrong model (price-based target). More experience does not fix this when the environment
keeps shifting. Fixed restrictions such as an inaction band recover part of the gain without estimation.

The plan was revised during development, before it was frozen: a first version represented each theory by one
design (and, earlier still, the reliability-condition agent used an unfiltered target). Both earlier versions also
placed the reliability-condition agent in the lower half.

## Rule choice: registered results

Plan `0f7bb98f8f7e8e44` (uncertainty levels u = 0, 0.25, 0.5, 0.75, 1, where cost volatility, perception error and
the demand-shift hazard rise together; intensities of choice β = 0, 1, 4, 16; 8 replications; 12 firms; 1,500
periods; rules tuned under tournament plan `9699f86a6a899cf7`).

| | E1 restricted share rises with uncertainty | E2 change rate falls with uncertainty | E3 selection raises the restricted share at u = 1 | E4 restricted share widens the distance from Nash |
|---|---|---|---|---|
| Result | slope +0.003 (p = 0.95) | slope −0.087 (p = 0.009) | −0.005 [−0.18, 0.15] | +4.9 per unit of share (p = 0.002) |
| Verdict | not supported | **supported** | not supported | **supported** |

With moderate selection (β = 4) the restricted share rises from 51% at u = 0 to 60% at u = 1 and the change rate
falls from 57% to 42%. Under strong selection (β = 16) the flexible optimizer (filtered best reply) is driven out as
uncertainty rises (32% of firms at u = 0, under 1% at u = 1), but its place is taken by the target-margin heuristic
(0.5% to 47%), a simple rule that ignores most market information but is not one of the three restricted rules.
Populations become more predictable and move away from sophisticated optimization under uncertainty, as Heiner
argued, but performance-based selection favors a simple heuristic as much as an explicit restriction. Markets
with more rule-governed firms stay further from the Cournot–Nash price at every uncertainty level.

## Validation against data

**Field patterns** (registered tuned parameters, 12 markets per pattern, 600 periods): 5 of 6 documented patterns
reproduced. Naive price expectations produce cobweb cycles (lag-1 autocorrelation −0.80); slow adaptive
expectations dampen them (price s.d. 18.4 vs 37.4); imitate-the-best firms produce 1.19 × the Cournot–Nash output;
a mixed market's mean price lies within 4.3% of the Nash price with 1.06 × its volatility; prices lie between cost
and the monopoly price. **Not reproduced:** lumpy adjustment. Inaction-band firms change output in only 34% of
periods (sticky), but their changes are no larger on average than those of always-adjusting firms (67.0 vs 69.2),
so the "large steps" half of the pattern fails.

**Calibration to laboratory experiments.** The pipeline fits each theory's rule per subject on the first half of the
periods and predicts the second half. On synthetic subjects with known rules it recovers the generating rule for
about 70% of subjects in both formats (chance about 10%). Recovery is uneven: naive, keep, anchoring, trend and
win-stay/lose-shift subjects are recovered almost perfectly, but synthetic reliability-condition forecasters are
never recovered (they are classified as inaction band, keep or imitation, which make similar predictions over a
short series), and imitate-the-best subjects in the Cournot format mostly look like keep. Classifying a person as a
reliability-condition user therefore needs longer series than these designs provide. The published datasets are not bundled:
the Cournot data of Huck, Normann & Oechssler (1999) are archived in heiDATA,
[doi:10.11588/data/10012](https://doi.org/10.11588/data/10012) (access on request), and the cobweb data of Hommes,
Sonnemans, Tuinstra & van de Velden (2007) are available from the authors. Results on five public datasets are in
the next section.

**Human experiment.** Protocol `01f956595e90e5ab`: three blocks of 25 periods (low, medium, high uncertainty),
counterbalanced order (all six orders, assigned from the participant ID), the same three tuned agent rivals and the
same random draws per block for everyone, 15 agent designs recorded in shadow mode. Pre-registered hypotheses: X1
participants change output less often under high than low uncertainty (paired bootstrap); X2 under high
uncertainty, participants who change less often earn more relative to their rivals (OLS). On simulated participants
the classification recovers the generating design for every participant. **No human data have been collected yet**;
ethics approval and informed consent are needed before collecting data.

## Empirical validation on public data: registered results

Five public datasets were analyzed under a protocol fixed in advance (plan `96b15ef4d9b78d56`): every rule is fitted on
each participant's first half of periods and scored on the second half; every treatment is analyzed separately with
the rules its information set allows; uncertainty is a participant-cluster bootstrap; a hypothesis tested on several
datasets or treatments counts as supported only if it holds in every one. The data are not bundled (their licenses
govern redistribution); the Empirical validation page reads the files exactly as distributed.

| Data | Participants | What the files contain |
|---|---|---|
| Gomez-Martinez, Onderstal & Sonnemans (2016), Mendeley Data | 144 (36 markets of 4) | SQL dumps per session. Each firm's price is P_i = 150 − q_i − (2/3)·Q_−i, unit cost 2 (reproduces every recorded price; Nash output 37). Part 1, periods 1–25, without communication; aggregate vs individual information about rivals |
| Evans, Gibbs & McGough (2025), openICPSR 198204 | 372 (62 markets of 6) | One-period-ahead price forecasts, 50 periods; negative feedback (treatments 1–6) and positive feedback (7–8); announced structural changes around periods 20 and 45 |
| Anufriev & Hommes (2012) replication package, openICPSR 114401 | 120 (20 markets of 6) | Asset-pricing experiments of Hommes et al. (2005, 2008); two-period-ahead forecasts (verified with the pricing equation); fundamental price 60 (40 in three groups) |
| Brokesova, Deck & Peliova (2022), PLOS ONE S2 | 52 newsvendors | 100 orders each, uniform demand 0–100, price 100, unit cost 25 (optimal order 75) or 75 (optimal order 25) |
| Time-pressure learning-to-forecast experiments, University of Amsterdam figshare | 198, each under high and low time pressure | oTree exports; asset-pricing design, two-period-ahead forecasts |

| | Hypothesis | Result | Verdict |
|---|---|---|---|
| V1 | Cournot: the reliability condition predicts later quantities better than the best reply it restricts | Aggregate information −0.061 [−0.152, +0.028]; individual information −0.133 [−0.277, +0.010] (RMSE gain) | not supported |
| V2 | Cournot: markets re-run with each participant's fitted rule reproduce the human markets | Output relative to Nash reproduced (human 0.98 and 1.06, inside the simulated intervals); change frequency **not** reproduced: humans change output in 62% and 67% of periods, the fitted rules plus noise in 88–93% | not supported |
| V3 | Forecasting: the reliability condition predicts later forecasts better than adaptive expectations | EGM negative feedback −0.33 [−0.64, −0.10]; positive feedback −0.32 [−0.60, −0.11]; asset markets −1.6 [−4.2, +1.0] | not supported |
| V4 | Forecasting: no single rule is best for a majority | Most common best rule fits 22% (imitation, EGM negative), 24% (naive, EGM positive), 37% (trend following, asset markets) | **supported** |
| V5 | Newsvendor: pull-to-center and demand chasing | Pull-to-center ratio 0.16 [0.01, 0.31] and 0.29 [0.14, 0.46]; demand-chasing slope +0.34 and +0.30 | **supported** |
| V6 | Newsvendor: the reliability condition predicts later orders better than demand chasing | −1.41 [−2.29, −0.65] and −1.02 [−1.99, −0.14] | not supported |
| V7 | Time pressure changes reliance on simple or restricted rules | Share best predicted by them: 46% under high, 35% under low time pressure, +0.11 [+0.01, +0.21], paired by participant | **supported** |

Best out-of-sample rules: aspiration-based adjustment toward the best reply (Cournot, aggregate information),
imitating the most profitable firm (Cournot, individual information, consistent with the original finding that
individual information makes markets more competitive), imitating the most accurate forecaster (EGM, negative
feedback), naive expectations (EGM, positive feedback), trend following (asset markets) and anchoring on a
pulled-to-center order with adjustment toward last demand (newsvendor). In five of the six EGM treatments with an
announced change, forecast errors rise in the five periods after it and fall in the next five (treatment 5 is the
exception); the control treatment without a change also shows an error rise around period 20, so not all of the rise
can be attributed to the announcement.

**What the data say about the simulation.** The learned reliability condition, as implemented here, is not a better
description of individual choices than the flexible rules it restricts, in any of the five datasets; fixed
restrictions (an inaction band) do no better. The data do support the population-level claims: behavior is
heterogeneous, the documented newsvendor patterns are present, simulated Cournot markets reach the human level of
competition, and lower competence (time pressure) shifts people toward simple or restricted rules. The generative
check also shows a specific failure of the simulated rules: people keep their decision unchanged far more often than
the fitted rules plus decision noise imply, so human behavior is more inert than any of the flexible rules. As an
exploratory, not pre-registered observation, restricted rules are the best description for more participants where
information is poorer: 31% under aggregate and 11% under individual information in the Cournot data.

## Generalization: registered results

Plan `b46c1f64a8250547` (three tasks, 120 environments each by Latin hypercube over outcome noise, shift hazard,
observation error and the flexible rule's gain; 3,000 periods; band width tuned on 30 separate environments; oracle
values from 30,000-period independent runs). Each task has a default (slow, long-memory estimate), a flexible
alternative (fast re-estimate) and no cost of deviating. In the investment task a project arrives every period, mean
project quality shifts without warning between booms and busts, and a bad investment also costs a write-off of half
its loss. Gain over always deviating (95% CI):

| | Inventory (cost per period ≈ 41) | Learning with shifting payoffs (payoff ≈ 0.16) | Irreversible investment (payoff ≈ 1.3) |
|---|---|---|---|
| Boundary slope of the oracle's gain on ln K | +5.5 (p < 0.001), K* ≈ 0.68 | +0.18 (p < 0.001), K* ≈ 0.88 | +0.041 (p < 0.001), linear K* ≈ 0.15 (poor fit) |
| Oracle, top third of K | **+2.45 [1.32, 3.73]** | **+0.061 [0.041, 0.082]** | **+0.191 [0.140, 0.242]** |
| Oracle, bottom third of K | −0.01 [−0.08, 0.06] | −0.001 [−0.005, 0.002] | −0.004 [−0.012, 0.005] |
| Learned RC, top third of K | **+2.42 [1.42, 3.61]** | **+0.047 [0.026, 0.068]** | **+0.186 [0.137, 0.236]** |
| Rule B, bottom third of K | −10.6 [−11.7, −9.6] | −0.095 [−0.113, −0.078] | −0.070 [−0.093, −0.048] |
| Oracle − learned (all environments) | +0.04 [−0.09, 0.17] | +0.011 [0.004, 0.016] | −0.002 [−0.006, 0.001] |

Pre-registered verdicts: G1 (boundary in every task) **supported**; G2 (restriction pays only where K is high)
**supported**; G3 (learning the reliability is costly in every task) not supported, because in the inventory and
investment tasks the learned rule does as well as the oracle; G4 (the learned RC beats always deviating where K is
high) **supported**. With binary actions K is 0 or infinite when one rule is never wrong over a run, so K is clipped
to [0.05, 20] in the boundary analysis; in the investment task the linear fit on ln K is poor (R² 0.16), and the gain
by tercile turns positive between K ≈ 0.9 and 1.6.

**What generalization adds.** In all three tasks restriction pays where the flexible rule is unreliable and is
worthless where it is reliable, with a break-even error-to-signal ratio near or below 1, as in the market (K ≈ 0.97).
The boundary is therefore a property of decisions with a default, a flexible alternative and a difficulty–competence
gap, not of cobweb markets. What differs is the cost of applying the principle: in these tasks the gain of a
deviation can be judged exactly from what the agent observes, so a learner captures the oracle's gain; in the
market, where outcomes depend on rivals and on a misspecified model, it does not.

## Solvable benchmark: registered results

Plan `f582721595727105`. A single firm tracks a random walk (variance q per period) observed with noise (variance r)
and loses the squared tracking error. With partial adjustment at speed φ the expected loss is
E(φ) = [(1 − φ)²q + φ²r] / [φ(2 − φ)], minimized at the steady-state Kalman gain k = P/(P + r),
P = (q + √(q² + 4qr))/2 (Muth 1960; Kalman 1960).

**Exactness (T1, supported).** Over 96 speeds and q/r = 0.01, 0.1, 1 and 10 (8 paths × 50,000 periods), the
simulated loss is within 0.9% of the formula everywhere, and the simulated best speed equals the Kalman gain to the
grid step (0.095 → 0.10, 0.270 → 0.27, 0.618 → 0.62, 0.916 → 0.92).

**Lopsided stakes.** Overshooting now costs ρ times as much as undershooting (weights normalized to average 1, so
the filter's loss stays 0.271 at every ρ). The information (q/r = 0.1) is unchanged. Rules are tuned on 8 training
paths and evaluated on 16 test paths (20,000 periods each). Loss reduction relative to the Kalman filter (95% CI):

| ρ | Restricted filter (thresholds up/down) | Speeds up/down | Kalman + optimal offset (Bayes) | Upward moves of the restricted rule |
|---|---|---|---|---|
| 1 | 0.000 (threshold 0 = the filter) | −0.001 | 0.000 | 50% |
| 2 | +0.005 [0.004, 0.005] | +0.016 | +0.020 | 25% |
| 4 | +0.025 [0.023, 0.027] | +0.063 | +0.068 | 10% |
| 8 | +0.058 [0.056, 0.060] | +0.113 | +0.124 | 6% |
| 16 | +0.092 [0.090, 0.095] | +0.162 | +0.172 | 3% |

Pre-registered verdicts: T1 **supported**; T2 (with symmetric stakes nothing beats the filter) **supported**; T3
(restricting the filter's moves pays with lopsided stakes, more so as ρ rises; slope +0.043 per unit of ln ρ,
p < 0.001) **supported**; T4 (restriction never beats the filter with the optimal offset) **supported**.

**What the benchmark pins down.** Optimal filtering and Heiner agree whenever the stakes are symmetric: the best
response to news is the Kalman gain and no restriction helps. With lopsided stakes they part: certainty equivalence
says the stakes should shift the level of the action but not the response to news, whereas the reliability
condition says moves in the costly direction should be restricted. With the same information, the restricted rule
cuts the loss by up to a third, and upward moves fall from half of all periods to 3%. That is Heiner's unique
prediction, and it holds. Its limit is equally clear: a Bayes-optimal agent that builds the stakes into its estimate
(the filter plus the optimal offset) does better still, so restriction is the reliability condition's answer for
agents whose flexible rule does not encode the stakes. That is the situation Heiner describes, but not the only one.

## Code layout

```
heiner_abm/params.py       scenario dataclasses and the baseline calibration
heiner_abm/agents.py       readable agent implementation: Firm, Market, Industry
heiner_abm/engine.py       vectorized batch engine (same model, same random streams) for Monte Carlo
heiner_abm/analysis.py     reliability metrics, market statistics, regressions
heiner_abm/experiments.py  sweeps with common random numbers, rigid-twin RC validation, evolution, horse race
heiner_abm/theories.py     rival theories, their predictions, the tournament experiments and scoring
heiner_abm/literature.py   references, research behind every hypothesis and alternative, contributions
heiner_abm/arena.py        rival agents, shared-market simulator, equal-budget tuning, pre-registered tournament
heiner_abm/mechanisms.py   oracle vs learned reliability condition, boundary test, uncertainty-type maps
heiner_abm/rulechoice.py   endogenous rule choice with logit switching (Brock & Hommes 1997)
heiner_abm/patterns.py     pattern-oriented validation against documented field patterns
heiner_abm/calibration.py  per-subject out-of-sample fitting of every theory's rule to laboratory data
heiner_abm/stepper.py      a market that advances one period at a time, for human participants
heiner_abm/experiment.py   human experiment protocol, classification and pre-registered tests
heiner_abm/empirical.py    validation against public experimental data: protocol, adapters, generative check
heiner_abm/datasets.py     loaders for the five public datasets, from the files as distributed
heiner_abm/tasks.py        generalization tasks: inventory, learning with shifting payoffs, irreversible investment
heiner_abm/tracking.py     single-firm tracking benchmark: exact Muth–Kalman solution and lopsided stakes
heiner_abm/focal.py        the theory under test: every theory's prediction for every hypothesis
heiner_abm/special.py      signature tests: each rival theory's distinctive prediction in the shared market
heiner_abm/theory_content.py  the eight theories, described with the same structure
heiner_abm/registered.py   registered results shown on the theory pages, tied to the plan hashes
ui/theory_page.py, ui/illustrations.py  theory page renderer and one interactive illustration per theory
ui/common.py               sidebar base scenario, presets, caching, chart helpers
app_pages/*.py             the Streamlit pages
```

## Established theory, reduced form and proposed extensions

`heiner_abm/claim_status.py` classifies every part of the laboratory, and the *Model & methods* page shows the table.

* **Established theory** (used or reproduced as published): the reliability condition (Heiner 1983); the
  partial-adjustment bound β₀ (Heiner 1989); optimal filtering of a random walk (Muth 1960; Kalman 1960); cobweb
  cycles and their damping (Ezekiel 1938; Nerlove 1958); logit rule choice (Brock & Hommes 1997); imitation of the
  best (Vega-Redondo 1997).
* **Reduced-form implementations** (modeling choices, not claims of the cited work): the stylized market; the
  model-based and model-free production rules; difficulty and competence as Δ, rivals, demand shifts, κ and σ;
  Knightian uncertainty as unannounced regime shifts; each rival theory as one or two agent designs with stylized
  directional predictions.
* **Proposed extensions** (the laboratory's own constructs): the H-period forked measurement of the reliability
  condition and its decomposition; the reliability condition as a decision rule learned from experience; the oracle
  versus learned decomposition; the error-to-signal boundary K ≈ 1; the lopsided-stakes test against certainty
  equivalence. Results about these test the laboratory's operationalization, not Heiner's theory as published.

## Modeling notes

* The core model is a cobweb oligopoly: firms commit to output before a random raw-material cost is realized, using
  a Cournot best-reply rule (model-based) or a Bertrand margin-feedback rule (model-free), and a selection rule that
  decides when to deviate from rule B (keep last period's output).
* **Features:**
  * per-firm, per-decision reliability measurement (π, r, w, G, D);
  * competence (cost foresight κ, perception noise σ), which makes the CD-gap explicit and measurable;
  * an Adaptive selection rule that learns when deviating pays;
  * evolution of flexibility;
  * 2–12 firms;
  * out-of-sample RC tests (estimation and evaluation windows);
  * three measures of decision value (one-period, persistence, full strategic feedback) with discounting;
  * Heiner's (1989) error-to-signal ratio K and partial-adjustment bound;
  * unannounced demand-regime shifts with a model-updating lag (structural / Knightian uncertainty).
* **Counterfactual horizon H.** Heiner's condition treats each deviation from rule B as a one-shot bet, judged here
  by one period of profit.
  Because production changes persist and rivals react, that measure says "deviate" almost always and does not
  predict flexible-vs-rigid performance. The model therefore forks the market at each decision and compares the
  two branches over H periods. By default the firm returns to rule B afterwards, which is Heiner's "deviate at this
  instance, otherwise follow B". H = 1 gives the one-shot measure. The default is H = 20. H is a researcher's
  measurement: the forks use future periods, the true demand curve and rivals' true rules, so agents never see them.
* **What agents learn from.** Agents decide and learn only from information available to them at the time. The
  Adaptive selection rule judges each past deviation once the W periods it covers (default 20, sidebar *judgement
  window*) have passed: holding the recommended output against holding the old output, with rivals' actual output,
  realized costs and prices on the firm's own believed demand curve (the same judgement as the tournament's
  reliability-condition agents). Until 6 October 2026 it learned instead from the researcher's look-ahead
  counterfactual over H. That rule is kept as the explicitly researcher-only *look-ahead* feedback, which reproduces
  the earlier trajectories exactly (`tests/test_information.py`). The findings that involved Adaptive agents (the
  directional tournament, the out-of-sample forecasts and the Adaptive preset) were rerun under the new rule on
  6 October 2026; see the revision note under *Competing theories*.

## Research basis

`heiner_abm/literature.py` is the single source for the research behind the simulation. Each hypothesis records its
reliability-condition prediction, its alternative prediction and the rival theory it comes from, the studies that
support each side (with a note on how each bears on the hypothesis), and the contribution the test makes. The app
shows this on every hypothesis card and on the *Research & contribution* page, which also exports the bibliography
as BibTeX and APA and the evidence matrix as CSV. A test checks that every cited work exists in the registry, that
every hypothesis has research on both sides, and that the app text cites research only through the registry.
