# Decision making under uncertainty: an agent-based laboratory

A Streamlit agent-based simulation laboratory that compares eight theories of when a decision maker should adapt
and when it should stick to a rule: Heiner's reliability condition, neoclassical optimisation, real options, cobweb
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
5. Click **Deploy**. Dependencies are installed from `requirements.txt`, and the theme comes from
   `.streamlit/config.toml`.

No secrets or data files are needed. The GitHub Actions workflow in `.github/workflows/tests.yml` runs the test
suite on every push.

**Resource note:** Community Cloud apps have about 1 GB of memory. The default experiments use well under that. Very
large runs can hit the limit: for example, hundreds of environments at H = 100, or tens of thousands of periods with
many replications. For heavy research runs, use a local installation.

## Pages

| Page | What it does |
|---|---|
| Theories · Overview | The eight theories side by side (core claim, view of uncertainty, what triggers change, effect of uncertainty on the value of flexibility), the common testbed, how each fared, every hypothesis with its research, and the contribution to the literature |
| Theories · one page per theory | Heiner, optimisation, real options, cobweb, simple heuristics, satisficing, reinforcement learning, imitation. Same sections for each: origins, formal core, view of flexibility, an interactive illustration, how the laboratory implements it, how it fared in the registered runs, strengths and limits, references |
| Market lab | One market with editable heterogeneous firm agents. Shows price/cost dynamics, a per-firm reliability scoreboard, and a period-by-period decision inspector (correct deviations, type I and type II errors) |
| Hypothesis tests | H1 free flexibility · H2 profitability switch · H3 volatility · H4 fixed costs · H5 competition intensity · H6 regimes and equilibrium · H7 competence · H8 perception noise · H9 selection rules · H10 predictable behaviour · H11 number of rivals · H12 model-updating lag. Each shows the RC prediction next to the alternative, the research behind both, the contribution, and a verdict |
| Does the RC predict performance? | Random environments; each firm is compared with its own rigid twin (same shocks). The RC is estimated in the first part of each run and predicts the second part (**out-of-sample**), scored by AUC with environment-clustered bootstrap CIs |
| Dynamic RC (Heiner 1989) | Decomposes each decision's value into immediate, persistence and strategic-feedback parts. Tests Heiner's (1989) partial-adjustment bound β₀ = 1/((1+K)(1−f′)) against the profit-maximising flexibility. Signal-detection ROC of each firm's decisions |
| Risk vs Knightian uncertainty | Cost-volatility risk versus unannounced demand-regime shifts at matched unpredictability, for model-based (Cournot) and model-free (Bertrand) firms. Event study of punctuated slow–quick–slow adjustment |
| CD-gap explorer | Difficulty (Δ or noise) × competence (foresight κ) heatmaps of r, w, π, RC margin and the payoff to flexibility |
| Endogenous flexibility | Firms imitate the most profitable rival's φ (plus mutation). Does volatility breed rigidity? |
| Competing theories | Heiner's RC against neoclassical optimisation, real options, cobweb stability, bias–variance / ecological rationality, satisficing and structural inertia. A tournament of nine discriminating experiments scores each theory's directional predictions; an out-of-sample horse race scores each theory's forecast of which firms benefit from flexibility, plus an encompassing test of whether the RC adds information beyond all rivals |
| Agent tournament | Every rival theory implemented as two agent designs competing in the same market (17 designs, including target × selection-rule composites). Equal tuning budget per design on training environments, design selection on training data, held-out test environments, a frozen hashed plan with six pre-registered hypotheses, six performance criteria (profit, downside risk, survival, volatility, regret, worst case), a selection-rule experiment, invasion tests, global sensitivity analysis and replication across seeds |
| Mechanisms | Principle versus implementation: focal-firm variants on a shared target (always, inaction band, learned reliability condition, the same learner judging with the true model, an oracle with true reliability, a memory grid), with the cost of applying the principle decomposed into estimation and model bias; a boundary test against the measured error-to-signal ratio K; standardised effects of each source of uncertainty; cross-validated metamodel maps of which theory does best where. Frozen study plan with five pre-registered hypotheses |
| Rule choice (emergence) | Firms switch between six rules (three restricted, three flexible) by recent performance with logit choice and an adjustable intensity of choice β (Brock & Hommes 1997). Rule shares, change rates, price volatility and distance from Cournot–Nash across uncertainty levels. Frozen plan with four pre-registered hypotheses |
| Experiment designer | Generic one- or two-parameter sweeps with CSV export |
| Field patterns | Pattern-oriented validation (Grimm et al. 2005): cobweb cycles, damping by adaptive adjustment, sticky and lumpy adjustment, imitation beyond Cournot–Nash, excess volatility around equilibrium and positive markups, each with a criterion fixed in advance and its sources |
| Calibration to experiments | Fits every theory's decision rule per subject to learning-to-forecast cobweb data (Hommes et al. 2007 design) or Cournot data (Huck et al. 1999 design) on the first half of periods and scores it on the second half; classifies subjects by best-predicting rule. Upload data or check recovery on synthetic subjects |
| Play the market | A person runs one firm against three agent rivals in three counterbalanced blocks (low, medium, high uncertainty) of 25 periods; every agent design records in shadow mode what it would have chosen. Download the decisions as CSV |
| Experiment analysis | Pools participants' files, classifies each person by the best-predicting design, and tests X1 (fewer changes under high uncertainty) and X2 (restraint pays under high uncertainty). Synthetic demonstration clearly labelled |
| Generalisation | The same selection layers and boundary test in two other decision tasks with a default, a flexible alternative and a difficulty–competence gap: an inventory (newsvendor) task with shifting demand and a learning task with shifting payoffs. Frozen plan with four pre-registered hypotheses |
| Research & contribution | The simulation's contributions to the literature, the evidence matrix (supporting and alternative research for every hypothesis), the research behind each rival theory and method, and the full bibliography with BibTeX, APA and CSV export |
| Model & methods | Equations, schedule, measurement, statistics, baseline calibration, verification |

## Competing theories: reference results

From the *Competing theories* page with 1,000 periods, 20 replications per condition and H = 20, for the default
Bertrand market and a Cournot market (φ = 0.1–0.4), each with two seeds:

| Theory | Tournament record, Bertrand (✅ / ❌ / ➖) | Tournament record, Cournot (✅ / ❌ / ➖) |
|---|---|---|
| Heiner: reliability condition | 6/1/2 and 5/1/3 | 5/1/3 and 6/1/2 |
| Bias–variance / ecological rationality | 4/2/0 and 3/2/1 | 3/2/1 (both seeds) |
| Cobweb stability theory | 1/5/1 and 2/4/1 | 4/3/0 (both seeds) |
| Real options | 2/4/1 and 2/3/2 | 1/3/3 and 2/3/2 |
| Neoclassical optimisation | 2/4/1 and 1/4/2 | 3/1/3 and 3/2/2 |

Heiner had the best net record in all four runs. Its one contradiction differs by market: in Bertrand markets
perception noise *raised* the payoff to flexibility, and in Cournot markets free flexibility did not hurt at the
tested low-profit level. Evolved flexibility showed no volatility gradient in any run. The best adjustment speed
fell with noise, as Heiner predicts, but optimal filtering (Muth 1960; Kalman 1960) predicts the same, so that test
does not discriminate between them.

Out-of-sample forecasting (100 random environments × 2 replications, 800 firms, both production rules): the dynamic
RC was the best theory-based forecast of which firms beat their rigid twin (AUC 0.61, 95% CI 0.56–0.66), ahead of
cobweb stability (0.58), stakes only (0.56), Heiner's K (0.56), accuracy only (0.54) and real options (0.47). A firm's
own track record in the first half of the run was far better (AUC 0.86), and adding the RC to all rival forecasts
combined changed cross-validated AUC by −0.001 (95% CI −0.005 to +0.002). Heiner's one-shot (one-period) RC scored only 0.55.

## Agent tournament: registered results

Registered plan `15193603c3f8359b`: two designs per theory, each tuned with a budget of 24 settings × 2 rounds on 24
training environments; each theory enters with the design that scored higher on training data; evaluation on 40 × 2
held-out environments (800 periods); six criteria; invasion with 4 residents + 1 mutant; three replications with
fresh seeds.

| Theory (selected design) | Profit rank: main / reps | Aggregate rank, six criteria: main / reps |
|---|---|---|
| Cobweb: adaptive price expectations | 1.89 / 2.91, 2.74, 2.64 | 3.50 / 2.67, 2.33, 2.83 |
| Simple heuristics: target-margin rule | 2.79 / 3.30, 1.91, 1.65 | 4.00 / 4.17, 4.00, 3.83 |
| Real options: inaction band, price-based target | 3.41 / 2.99, 3.24, 4.21 | 3.33 / 3.00, 3.83, 3.67 |
| Heiner: reliability condition, price-based target | 3.98 / 4.89, 6.00, 5.42 | 4.17 / 5.00, 3.33, 6.00 |
| Optimisation: rational expectations (Cournot–Nash) | 5.84 / 5.62, 5.90, 5.36 | 4.83 / 5.50, 6.83, 4.67 |
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
  without the estimation error. This is consistent with Heiner's deeper argument that reliable behaviour comes from
  rules rather than case-by-case assessment, and it qualifies the use of the condition as an agent's decision rule.
* *Criteria matter.* Ranking by profit alone, by downside risk or by survival gives different orders; seven of nine
  theories are Pareto-efficient in the main run, so claims of superiority must name the criterion.

Adding the instrumentation used by the mechanism study changed the plan's hash (it covers the code) but not a single
result: the registered run reproduces the earlier one exactly.

## Mechanism study: registered results

Study plan `fafb771393aab7fe` (120 environments drawn by Latin hypercube over separate sources of uncertainty, 800
periods, oracle values from 3,200-period independent runs, background rivals tuned under tournament plan
`15193603c3f8359b`). A focal firm aims at the same target as the always-adjusting rule and differs only in when it
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
gain is driven by misspecification (shift hazard +0.22 [0.05, 0.39], model lag +0.29 [0.12, 0.44], standardised)
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

Plan `e6df56611a66c23f` (uncertainty levels u = 0, 0.25, 0.5, 0.75, 1, where cost volatility, perception error and
the demand-shift hazard rise together; intensities of choice β = 0, 1, 4, 16; 8 replications; 12 firms; 1,500
periods; rules tuned under tournament plan `15193603c3f8359b`).

| | E1 restricted share rises with uncertainty | E2 change rate falls with uncertainty | E3 selection raises the restricted share at u = 1 | E4 restricted share widens the distance from Nash |
|---|---|---|---|---|
| Result | slope +0.003 (p = 0.95) | slope −0.087 (p = 0.009) | −0.005 [−0.18, 0.15] | +4.9 per unit of share (p = 0.002) |
| Verdict | not supported | **supported** | not supported | **supported** |

With moderate selection (β = 4) the restricted share rises from 51% at u = 0 to 60% at u = 1 and the change rate
falls from 57% to 42%. Under strong selection (β = 16) the flexible optimiser (filtered best reply) is driven out as
uncertainty rises (32% of firms at u = 0, under 1% at u = 1), but its place is taken by the target-margin heuristic
(0.5% to 47%), a simple rule that ignores most market information but is not one of the three restricted rules.
Populations become more predictable and move away from sophisticated optimisation under uncertainty, as Heiner
argued, but performance-based selection favours a simple heuristic as much as an explicit restriction. Markets
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
Sonnemans, Tuinstra & van de Velden (2007) are available from the authors. **No results on human data are reported
yet.**

**Human experiment.** Protocol `01f956595e90e5ab`: three blocks of 25 periods (low, medium, high uncertainty),
counterbalanced order (all six orders, assigned from the participant ID), the same three tuned agent rivals and the
same random draws per block for everyone, 15 agent designs recorded in shadow mode. Pre-registered hypotheses: X1
participants change output less often under high than low uncertainty (paired bootstrap); X2 under high
uncertainty, participants who change less often earn more relative to their rivals (OLS). On simulated participants
the classification recovers the generating design for every participant. **No human data have been collected yet**;
ethics approval and informed consent are needed before collecting data.

## Generalisation: registered results

Plan `ffaa66523475d466` (two tasks, 120 environments each by Latin hypercube over outcome noise, shift hazard,
observation error and the flexible rule's gain; 3,000 periods; band width tuned on 30 separate environments; oracle
values from 30,000-period independent runs). Each task has a default (slow, long-memory estimate), a flexible
alternative (fast re-estimate) and no cost of deviating. Gain over always deviating (95% CI):

| | Inventory (cost per period ≈ 41) | Learning with shifting payoffs (payoff per period ≈ 0.16) |
|---|---|---|
| Boundary slope of the oracle's gain on ln K | +5.5 (p < 0.001), break-even K* ≈ 0.68 | +0.18 (p < 0.001), K* ≈ 0.88 |
| Oracle, top third of K | **+2.45 [1.32, 3.73]** | **+0.061 [0.041, 0.082]** |
| Oracle, bottom third of K | −0.01 [−0.08, 0.06] | −0.001 [−0.005, 0.002] |
| Learned RC, top third of K | **+2.42 [1.42, 3.61]** | **+0.047 [0.026, 0.068]** |
| Rule B, bottom third of K | −10.6 [−11.7, −9.6] | −0.095 [−0.113, −0.078] |
| Oracle − learned (all environments) | +0.04 [−0.09, 0.17] | +0.011 [0.004, 0.016] |

Pre-registered verdicts: G1 (boundary in every task) **supported**; G2 (restriction pays only where K is high)
**supported**; G3 (learning the reliability is costly in every task) not supported, because in the inventory task
the learned rule does as well as the oracle; G4 (the learned RC beats always deviating where K is high) **supported**.

**What generalisation adds.** In both tasks restriction pays where the flexible rule is unreliable and is worthless
where it is reliable, with a break-even error-to-signal ratio below or near 1, as in the market (K ≈ 0.97). The
boundary is therefore a property of decisions with a default, a flexible alternative and a difficulty–competence
gap, not of cobweb markets. What differs is the cost of applying the principle: in these tasks the gain of a
deviation can be judged exactly from what the agent observes, so a learner captures most of the oracle's gain; in
the market, where outcomes depend on rivals and on a misspecified model, it does not.

## Code layout

```
heiner_abm/params.py       scenario dataclasses and the baseline calibration
heiner_abm/agents.py       readable agent implementation: Firm, Market, Industry
heiner_abm/engine.py       vectorised batch engine (same model, same random streams) for Monte Carlo
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
heiner_abm/tasks.py        generalisation tasks: inventory and learning with shifting payoffs
heiner_abm/theory_content.py  the eight theories, described with the same structure
heiner_abm/registered.py   registered results shown on the theory pages, tied to the plan hashes
ui/theory_page.py, ui/illustrations.py  theory page renderer and one interactive illustration per theory
ui/common.py               sidebar base scenario, presets, caching, chart helpers
app_pages/*.py             the Streamlit pages
```

## Modelling notes

* The core model is a cobweb oligopoly: firms commit to output before a random raw-material cost is realised, using
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
  instance, otherwise follow B". H = 1 gives the one-shot measure. The default is H = 20.

## Research basis

`heiner_abm/literature.py` is the single source for the research behind the simulation. Each hypothesis records its
reliability-condition prediction, its alternative prediction and the rival theory it comes from, the studies that
support each side (with a note on how each bears on the hypothesis), and the contribution the test makes. The app
shows this on every hypothesis card and on the *Research & contribution* page, which also exports the bibliography
as BibTeX and APA and the evidence matrix as CSV. A test checks that every cited work exists in the registry, that
every hypothesis has research on both sides, and that the app text cites research only through the registry.
