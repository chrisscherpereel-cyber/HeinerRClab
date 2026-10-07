# Decision making under uncertainty: an agent-based laboratory

A Streamlit agent-based simulation laboratory that compares nine theories of when a decision maker should adapt
and when it should stick to a rule: Heiner's reliability condition, neoclassical optimization, real options, cobweb
theory and adaptive expectations, simple heuristics (bias–variance), satisficing, reinforcement learning, and
imitation and evolutionary selection. All are implemented in the same cobweb oligopoly and tested on equal terms.
Every hypothesis, and every alternative to it, is grounded in published research (see *Research basis* below).
Read *Reading the results* first: it defines the terms, the kinds of uncertainty and the kinds of evidence, and the
scope of every conclusion.

## Reading the results

**Terms.** The model-free production rule is the *margin-feedback quantity rule*: firms choose quantities and move
output in proportion to how far the observed margin (last price minus cost estimate) is from a desired margin. Its
configuration value is still `"Bertrand"` so that saved scenarios and registered results load, but it is not
conventional price-setting Bertrand competition. The model-based rule is the *Cournot best-reply quantity rule*
(partial adjustment toward the best reply on the believed demand curve). The real-options designs are an
*inaction-band heuristic inspired by real options*: they move only when the gap to the target exceeds a band that
widens with measured volatility; there is no adjustment cost, no irreversibility and no option valuation, so they are
not a dynamic real-options model. Definitions live in `heiner_abm/terminology.py`.

**Five kinds of uncertainty.** *Known stochastic risk* (cost shocks from a fully specified distribution);
*parameter uncertainty* (the form of the process is known, its parameters are not: agents never see the parameters
of their environment); *distributional ambiguity* (no single distribution can be assigned: **not modeled** here);
*structural model misspecification* (the agent's model has the wrong structure: believed demand curves lag the true
one after a shift, and every rule simplifies the market); *unannounced environmental change* (demand-regime shifts
the agents are not told about). A simulator must specify every process it runs, so the regime shifts are unknown to
the agents but not unknowable: a long-lived agent could learn their hazard and size distribution. Unannounced shifts
alone therefore do not establish Knightian uncertainty in the unrestricted sense; where the app says "Knightian" it
means this restricted, agent-relative sense, and the page formerly titled *Risk vs Knightian uncertainty* is now *Risk
vs structural change*.

**Four kinds of evidence.** Each results section below is labeled with one or more of:

| Evidence type | What it establishes | What it does not | Results here |
|---|---|---|---|
| Analytical verification | The code reproduces a result derived mathematically | Anything about real behavior or rival theories | Agent/engine equivalence, the RC accounting identity, the exact Muth–Kalman loss curve (T1), the cobweb stability boundary |
| Simulation comparison | Within the specified model, one implementation does better or worse than another under stated settings | Anything about real firms or people; other operationalizations of the same theories | Hypothesis tests, competing theories, agent tournament, mechanism study, rule choice, generalization, field patterns, the other signature tests, the benchmark's lopsided stakes, calibration recovery on synthetic subjects |
| Out-of-sample behavioral prediction | Rules fitted to part of a person's choices predict the rest better or worse, in published experimental data | Causation; the data were collected by others | Calibration and empirical validation V1–V7 |
| Causal experimental evidence | A manipulated condition's effect on behavior | — | **None yet.** The human experiment (Play the market) is a protocol; no data have been collected |

The forecasts on the *Competing theories* page are out of sample *within the simulation*; they are simulation
comparison, not behavioral prediction. V7 compares fitted-rule shares across a condition the original authors
manipulated within subjects; the shares themselves are model-based classifications.

**Scope of conclusions.** Every verdict concerns a specific implementation of a theory in a specific simulated
environment, under the stated settings, criteria and sample sizes. "Not supported" means the tested implementation
did not show the predicted pattern there; it does not refute the theory, and the rival theories' predictions are
stylized readings of their literatures. The historical results below are reported as they were obtained; the
interpretations have been narrowed to what they test.

**Verification versus validation.** The test suite checks the RC accounting identity (realized deviation gains equal
correct-deviation gains minus wrong-deviation losses). That verifies internal consistency and is true by
construction. Predictive validation needs estimates fixed before the outcomes they predict; the RC validation page
estimates the condition in the first part of each run and predicts the second. Because the H-period gains of a
decision cover periods t … t + H − 1, the estimation window for those measures ends H − 1 periods before the
evaluation window starts, so no estimate uses an outcome it is meant to predict (`tests/test_oos_gap.py`). Until
6 October 2026 the windows overlapped by H − 1 periods (19 of about 490 estimation decisions at H = 20).

**Frozen plans versus preregistration.** A *frozen hashed specification* fixes a plan in the code; its hash
identifies the plan and code that produced a result. *Externally timestamped prospective preregistration* deposits
the plan with an independent registry before the data are generated, so others can verify the timing. The plans here
are frozen hashed specifications; the repository has no record of external preregistration. "Registered" and
"pre-registered" below refer to the former.

**Model specification.** Every agent's objective, information, actions, feedback, assumptions and limits are tabulated
in full (one row per agent) on the *Agents as implemented* and *Model & methods* pages (`heiner_abm/model_spec.py`).
In summary:

| Agents | Objective | Information at decision time | Actions | Feedback (and when) | Key assumptions | Limits |
|---|---|---|---|---|---|---|
| Market lab: margin-feedback quantity rule | Keep the margin near m* | Last price, own output, cost estimate; no demand model | q* = q + φ(P − ĉ − m*) | None | Firms set quantities; one clearing price | Not price-setting Bertrand; ignores rivals and the demand slope |
| Market lab: Cournot best-reply quantity rule | Move toward the static best reply | Last market output, cost estimate, believed demand curve (lags shifts) | Partial adjustment φ | None | Rivals keep last output | Myopic; misspecified after shifts |
| Market lab: selection rules (Always, Never, Small, Large) | When to adopt q* | Size of the recommended change | Adopt or keep q | None | Fixed threshold θ | No learning |
| Market lab: Adaptive selection rule (estimated-gain gate, default) | Deviate only where deviating has paid | Learned gain per size bin | Adopt q* if the learned gain ≥ c (c = 0 by default) | Observed gain, released W periods after the decision; oracle treatment only if chosen | Memory λ, window W | Judges with a possibly misspecified model |
| Market lab: Adaptive rule, confidence-sensitive gate (extension) | Adopt only where the advantage is reliably above the adjustment cost | Mean, s.e. and effective evidence per size bin | Adopt q* if n_eff ≥ minimum and the lower bound > c | As the default gate | Bound nominal only for independent feedback | Overlapping windows and drift make the bound optimistic; coverage is measured, not assumed |
| Market lab: Adaptive rule, exploration-enabled gate (extension) | Learn when adopting pays without counterfactual feedback | Its own payoffs after randomized trials, per size bin | Adopt / keep when the interval excludes c; otherwise a randomized trial with the exploration rate, else keep | Own realized payoff over W periods minus a pre-decision baseline, released W periods later | Trials are randomized (separate stream) | Trials cost payoff; stops exploring once confident (optional stopping) |
| Tournament designs (17) and mechanism variants (4) | A target (model-based best reply or price-based) and a selection rule, or a hand-written rule | Last price, market output, own profit, cost estimate, believed demand; imitators also see rivals' output and profit | Move toward the target when the selection rule says so | Own realized profits; the reliability-condition agent's h-period judgement is released after h periods | Parameters tuned on training environments | True-model and oracle variants are researcher-only and never in tournament lineups |
| Rule-choosing firms | Use the rule that recently earned most | Recent profits of each rule's users | Logit choice of rule | Realized profits after clearing | Intensity of choice β | Choice ignores why a rule did well |
| Generalization task agents | Default, flexible, band, learned RC, oracle RC | Observed outcomes so far | Default or flexible action | Observed gain after each period's outcome; oracle values are researcher-only | No cost of deviating | Short histories make learned bins noisy |
| Human participant | Earn as much as possible | The experiment screen | Output each period | Own outcomes | Counterbalanced blocks | No data collected |
| Rules fitted to human data | Predict a person's next choice | That person's past choices and market information | One prediction per period | Fitted on the first half only | Grid search per person | Predictive fit, not a causal model |

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
| Overview | The nine theories side by side (core claim, view of uncertainty, what triggers change, effect of uncertainty on the value of flexibility), the common testbed, how each fared, every hypothesis with its research, and the contribution to the literature |
| One page per theory | Heiner, optimization, real options, cobweb, simple heuristics, satisficing, reinforcement learning, imitation, organizational ecology. Same sections for each: origins, formal core, view of flexibility, an interactive illustration, how the laboratory implements it, how it fared in the registered runs, strengths and limits, references |

**Simulate** (experiments that test every theory on equal terms; hypotheses are stated as neutral questions)

| Page | What it does |
|---|---|
| Market lab | One market with editable heterogeneous firm agents. Shows price/cost dynamics, a per-firm reliability scoreboard, and a period-by-period decision inspector (correct deviations, type I and type II errors) |
| Hypothesis tests | H1 free flexibility · H2 profitability switch · H3 volatility · H4 fixed costs · H5 competition intensity · H6 regimes and equilibrium · H7 competence · H8 perception noise · H9 selection rules · H10 predictable behavior · H11 number of rivals · H12 model-updating lag. Each shows the RC prediction next to the alternative, the research behind both, the contribution, and a verdict |
| Risk vs structural change | Cost-volatility risk versus unannounced demand-regime shifts at matched unpredictability, for model-based (Cournot best reply) and model-free (margin-feedback) firms. The shifts are specified by the simulator and unknown to the firms; this is not Knightian uncertainty in the unrestricted sense. Event study of punctuated slow–quick–slow adjustment |
| Endogenous flexibility | Firms imitate the most profitable rival's φ (plus mutation). Does volatility breed rigidity? |
| Competing theories | Heiner's RC against neoclassical optimization, real options (as operationalized here), cobweb stability, bias–variance / ecological rationality, satisficing and structural inertia. A tournament of nine discriminating experiments scores each theory's directional predictions; an out-of-sample horse race scores each theory's forecast of which firms benefit from flexibility, plus an encompassing test of whether the RC adds information beyond all rivals |
| Agent tournament | Every rival theory implemented as two agent designs competing in the same market (17 designs, including target × selection-rule composites). Equal tuning budget per design on training environments, design selection on training data, held-out test environments, a frozen hashed plan with six hypotheses fixed in advance, six performance criteria (profit, downside risk, survival, volatility, regret, worst case), a selection-rule experiment, invasion tests, global sensitivity analysis and replication across seeds |
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
| Reliability gates under uncertainty (extension) | **Proposed extension.** The Adaptive rule's decision to adopt a recommendation is made by one of three gates with identical recommendations: the existing estimated-gain gate, a confidence-sensitive gate (adopt only with enough evidence and a lower confidence bound above the adjustment cost) and an exploration-enabled gate (learns from its own payoffs in randomized trials, without counterfactual feedback), plus an ORACLE benchmark estimated from independent runs. Reports false adaptations, missed opportunities, net payoff, calibration and lower-bound coverage, learning delay and performance after regime changes |
| Heiner vs optimal filtering (Muth–Kalman) | A single firm tracks a random-walk target observed with noise, where the best adjustment speed is the Kalman gain. The simulation reproduces the exact loss curve and optimum; then lopsided stakes test the prediction that is uniquely Heiner's against optimal filtering's certainty equivalence. Frozen plan with four pre-registered hypotheses |

**Validate & generalize**

| Page | What it does |
|---|---|
| Field patterns | Pattern-oriented validation (Grimm et al. 2005): cobweb cycles, damping by adaptive adjustment, sticky and lumpy adjustment, imitation beyond Cournot–Nash, excess volatility around equilibrium and positive markups, each with a criterion fixed in advance and its sources |
| Calibration to experiments | Fits every theory's decision rule per subject to learning-to-forecast cobweb data (Hommes et al. 2007 design) or Cournot data (Huck et al. 1999 design) on the first half of periods and scores it on the second half; classifies subjects by best-predicting rule. Upload data or check recovery on synthetic subjects |
| Decision benchmarks (Bayes, robust, bandit) | Established decision methods under one protocol (training, validation and test environments; tuning performance against the evaluation budget): Bayesian change detection, correctly specified and misspecified, and distributionally robust versus empirical optimization in the inventory task; sliding-window and discounted UCB in a chosen-action-feedback version of the learning task. Correctness is checked on analytic and enumerated cases first; benchmarks, oracles and the full-feedback reference are kept out of the rankings |
| When can reliability be learned? (extension) | **Proposed extension.** Tests whether a learnability ratio (observations needed to learn the sign of the advantage of adapting, relative to the informative observations available within a regime) explains when the confidence-sensitive gate beats the better fixed rule, beyond volatility and observation noise; seven policies on shared paths, training/pilot/test separation, new process families, negative controls, market replication. Frozen in the repository, not externally preregistered |
| NK landscapes (complexity) | Search on NK landscapes with K_NK interacting components: hill climbing, stochastic search, satisficing, imitation with stated observability and (extension) reliability-gated search, under separately controlled observation noise and landscape change. Exact benchmarks by enumeration for small N, best-known otherwise; performance against interdependence, noise and change with landscape-clustered intervals |
| Empirical validation (public data) | The five public datasets that can validate the simulation, their access, licenses and caveats; a protocol fixed in advance (seven hypotheses); loaders that read each repository's files as distributed; per-dataset analyses (out-of-sample rule comparison, generative check of simulated Cournot markets, newsvendor patterns, structural changes, time pressure); registered results |
| Play the market | A person runs one firm against three agent rivals in three counterbalanced blocks (low, medium, high uncertainty) of 25 periods; every agent design records in shadow mode what it would have chosen. Download the decisions as CSV |
| Experiment analysis | Pools participants' files, classifies each person by the best-predicting design, and tests X1 (fewer changes under high uncertainty) and X2 (restraint pays under high uncertainty). Synthetic demonstration clearly labeled |
| Generalization | The same selection layers and boundary test in three other decision tasks with a default, a flexible alternative and a difficulty–competence gap: an inventory (newsvendor) task with shifting demand, a learning task with shifting payoffs and an irreversible investment task. Frozen plan with four pre-registered hypotheses |

**Reference**

| Page | What it does |
|---|---|
| Agents as implemented | Every agent in the laboratory as the code implements it: the shared decision cycle, the market-lab firms (production and selection rules, adaptive learning, endogenous flexibility), the seventeen tournament designs with their equations, parameters, registered tuned values and sources, the mechanism variants, the rule-choosing firms, the task agents, the tracking rules, the experiment's rivals and shadows, and the rules fitted to human data |
| Research & contribution | The simulation's contributions to the literature, the evidence matrix (supporting and alternative research for every hypothesis), the research behind each rival theory and method, and the full bibliography with BibTeX, APA and CSV export |
| Model & methods | Equations, schedule, measurement, statistics, baseline calibration, a full description of every preset scenario (setup, why to run it, the hypotheses it serves and a typical result), what a frozen plan is (and how it differs from external preregistration), the model specification table, the kinds of uncertainty and evidence, and verification |

**Frozen plans (called "pre-registered" in the app).** Before a confirmatory study is run, its plan (hypotheses, tests and decision rules,
environments, sample sizes, tuning budgets and seeds) is fixed and, together with the agent and analysis code, turned
into a short hash. This prevents choosing tests or settings after seeing the results, makes the comparison between
theories fair, and lets anyone verify that a reported result came from exactly that plan and code. Any change produces
a new hash and the run is labeled exploratory. Every page with a plan explains this in an expandable note.

*What this does not establish.* A frozen hashed specification is not externally timestamped prospective
preregistration. "Pre-registered" here means frozen in the code and identified by a hash. The
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

*Evidence type: analytical verification (the cobweb boundary) and simulation comparison (the others).*

Each rival theory's distinctive prediction, tested with its own agents in the shared market (Full scale; deterministic
given the code). Each test operationalizes the prediction in one specific way; a verdict is about that test in this
market.

| Theory | Signature prediction | Result | Verdict |
|---|---|---|---|
| Neoclassical optimization | Better information raises an optimizer's profit; full information yields Cournot–Nash | profit slope on foresight -10.6 per unit (p = 0.0807); with full information the rational market's price is within 0.13% of Cournot–Nash on average | not supported |
| Real options | The value of flexibility rises with volatility | value of flexibility changes by -3.7 per unit of volatility (p = 0.0289) | not supported |
| Cobweb theory | Partial best-reply adjustment is stable only below φ = 4/(n + 1) | n = 3: theory 1.00, simulated 0.95; n = 4: theory 0.80, simulated 0.80; n = 6: theory 0.57, simulated 0.55; n = 8: theory 0.44, simulated 0.45 | **supported** |
| Simple heuristics | A simple rule beats the optimizer by more as estimation noise rises | heuristic's advantage changes by -8.8 per unit of noise (p = 0.00777); at the highest noise +409 [+311, +525] | not supported |
| Satisficing | Harder (more volatile) conditions trigger more search | change rate changes by -0.0011 per unit of volatility (p = 0.00213) | not supported |
| Reinforcement learning | Performance improves with experience in a stationary environment | relative profit improves by -73 [-283, +156] from the first to the last third in a stationary market, and by +148 [-163, +424] with unannounced shifts | not supported |
| Imitation | Imitating the best drives output above Cournot–Nash | imitate-the-best markets produce 1.24 [1.22, 1.26] × the Cournot–Nash output | **supported** |
| Organizational ecology | Inert organizations perform more reliably than flexible ones at every volatility | Δ = 2: -6 [-9, -3]; Δ = 10: -94 [-101, -87]; Δ = 20: -310 [-326, -293]; Δ = 30: -753 [-778, -728] | **supported** |

Three of eight signature tests came out as predicted in this market. The cobweb stability boundary is reproduced almost
exactly (0.95, 0.80, 0.55 and 0.45 against 1.00, 0.80, 0.57 and 0.44), and imitation of the best pushes output 24%
above Cournot–Nash, as Vega-Redondo (1997) predicts. The firm that reorganizes only under threat of failure has less
variable profit than an always-adjusting firm with the same price-based target at every volatility; the two designs
are tuned separately and the inert one also moves in smaller steps (φ = 0.15 against 0.88), so part of the
difference comes from the size of its moves, not from inertia alone. With the rivals retuned for the
ten-agent tournament, the optimizer no longer gains significantly from better information (−10.6 per unit of
foresight, p = 0.08). In the real-options test, an always-adjusting firm without adjustment costs lost more to its
rigid twin as volatility rose; that test has no irreversibility or option to wait, so it says that flexibility of
this kind lost value with volatility here, not that real-options theory is wrong. The heuristic beats the optimizer
at every noise level, but its advantage shrinks rather than grows with noise; the aspiration searcher changes
slightly less, not more, often in more volatile markets; and the tuned reinforcement learner does not improve
significantly with experience over 1,500 periods. Each of these is a statement about the implementation tested.

> **Revision note (6 October 2026, organizational ecology).** The signature tests use the registered tuned agents,
> which were retuned in the ten-agent tournament, and the ecology test is new. Earlier results: optimization
> **supported** (profit slope on foresight +40.6, p = 0.012); real options −11.2 per unit of volatility (p < 0.001);
> heuristics −26.0 per unit of noise, +684 at the highest noise; satisficing −0.0002 (p = 0.39); reinforcement
> learning −138 [−488, +230] and −190 [−487, +112]; cobweb and imitation unchanged. Three of seven were supported
> before; three of eight now (the optimizer's no longer, ecology's yes).

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

> **Revision note (6 October 2026, organizational ecology).** Ecology now states a prediction for every experiment
> (before, it predicted E1 and E9 only), from its core claim that inertia is favored whatever the environment. The
> four reference runs were repeated: every experiment's result was bit-identical, and only ecology's record changed,
> from 2/0/0 and 2/0/0 (Bertrand) and 1/1/0 (Cournot, both seeds) to the values in the table.

*Evidence type: simulation comparison.* A ✅ means the observed sign matched a theory's stylized directional
prediction in this market and a ❌ that it was opposite; neither confirms or refutes the theory beyond this test.

From the *Competing theories* page with 1,000 periods, 20 replications per condition and H = 20, for the default
margin-feedback market (configuration value "Bertrand", labeled Bertrand below) and a Cournot best-reply market
(φ = 0.1–0.4), each with two seeds (1 and 3 for Bertrand, 1 and 42 for Cournot;
the second seeds were not recorded originally and were identified as those that reproduce the earlier records on the
old code). Across eight seeds per market (1, 3, 21, 42, 100, 1000, 12345, 2026) Heiner had the best net record in
every run before ecology's predictions were completed; in the four runs reported here it still does (the other four
seeds have not been rerun since):

| Theory | Tournament record, Bertrand (✅ / ❌ / ➖) | Tournament record, Cournot (✅ / ❌ / ➖) |
|---|---|---|
| Heiner: reliability condition | 6/1/2 and 5/1/3 | 6/1/2 (both seeds) |
| Bias–variance / ecological rationality | 4/2/0 and 3/2/1 | 3/2/1 (both seeds) |
| Cobweb stability theory | 1/5/1 and 2/4/1 | 4/3/0 (both seeds) |
| Real options | 2/4/1 and 2/3/2 | 2/3/2 (both seeds) |
| Neoclassical optimization | 2/4/1 and 1/4/2 | 3/2/2 (both seeds) |
| Structural inertia (organizational ecology) | 4/5/0 and 5/4/0 | 3/6/0 (both seeds) |

Heiner had the best net record in all four runs (a record of predicted signs in these experiments, not a ranking of
the theories in general). Its one contradiction differs by market: in margin-feedback ("Bertrand") markets
perception noise *raised* the payoff to flexibility, and in Cournot markets free flexibility did not hurt at the
tested low-profit level. Evolved flexibility showed no volatility gradient in any run. The best adjustment speed
fell with noise, as Heiner predicts, but optimal filtering (Muth 1960; Kalman 1960) predicts the same, so that test
does not discriminate between them.

Out-of-sample forecasting within the simulation (100 random environments × 2 replications, 800 firms, both
production rules; simulation comparison, not behavioral prediction): the dynamic
RC was the best theory-based forecast of which firms beat their rigid twin (AUC 0.60, 95% CI 0.55–0.65), ahead of
cobweb stability (0.55), stakes only (0.55), Heiner's K (0.55), accuracy only (0.54) and real options (0.47). A firm's
own track record in the first half of the run was far better (AUC 0.85), and adding the RC to all rival forecasts
combined changed cross-validated AUC by −0.001 (95% CI −0.003 to +0.000). Heiner's one-shot (one-period) RC scored
only 0.54. (Sampling seed 12345, base-scenario seed 1, H = 20, full measure.)

*Revision note (6 October 2026, estimation gap).* The estimation window of the H-period measures now ends H − 1
periods before the evaluation window, so that no estimate uses an outcome it predicts. Rerun: dynamic RC 0.602 →
0.601 [0.554, 0.651], stakes only 0.550 → 0.548, accuracy only 0.536 → 0.539; every other forecast, the track record
and the one-shot RC are unchanged (they need no gap), and the encompassing gain stays +0.000 (CI −0.0044/+0.0048 →
−0.0045/+0.0045). The directional tournament does not use the windows and was unchanged.

*Revision note (6 October 2026, encompassing test).* The encompassing test now maps each forecast onto its training
fold's ranks before fitting, and scores a single forecast directly in the direction learned on the training folds.
Before, a logit on the raw, heavy-tailed and clipped forecasts could flatten a slope to about zero and flip its sign
between folds: "RC margin alone" scored 0.515 although the margin's own AUC is 0.60, and leaving out one environment
moved it by up to 0.10 (now 0.014). Rerun: rivals combined 0.851 → 0.866, rivals + tolerance limit 0.851 → 0.865, RC
margin alone 0.515 → 0.610, gain +0.000 [−0.0045, +0.0045] → −0.001 [−0.003, +0.000]. The forecast AUCs above are
unchanged, and the conclusion is the same: the RC adds no significant information once the rival forecasts are
combined.

## Agent tournament: results under the frozen plan

*Evidence type: simulation comparison.*

> **Revision note (6 October 2026, organizational ecology).** Organizational ecology became the ninth theory, with two
> tournament designs (scheduled reorganization; reorganization under threat of failure). The tournament's code hash
> changed from `9699f86a6a899cf7` to `110b3146bb072c2c`, so the registered protocol (tuning, main run, three
> replications) was rerun; every theory was retuned in markets that now include ecology's agent, so every number
> moved. Earlier values (nine agents): profit ranks, main run, cobweb 1.89, heuristic 2.79, real options 3.41, Heiner
> 3.98, optimization 5.84, reinforcement learning 6.20, imitation 6.24, rule B 6.28, satisficing 8.39; selection-rule
> effects, profit / CVaR 5%, main run: band · model +119* / +3349*, reliability condition · model +31 / +599*,
> reliability condition · price −339* / −359*, aspiration · price −983* / +699*; seven of nine theories
> Pareto-efficient; PR2 supported in one replication.
>
> **Revision note (2 October 2026).** The text of the agents and hypotheses was converted to American spelling. Plan hashes include that text, so the tournament, mechanism-study, rule-choice, generalization and tracking hashes changed (from `15193603c3f8359b`, `fafb771393aab7fe`, `e6df56611a66c23f`, `ee7faf1cff253ace` and `e6c4f642f296c19b`). The computation did not change: every registered study was rerun under the new hashes and reproduced every number reported here exactly.

Registered plan `110b3146bb072c2c`: two designs per theory, each tuned with a budget of 24 settings × 2 rounds on 24
training environments; each theory enters with the design that scored higher on training data; evaluation on 40 × 2
held-out environments (800 periods); six criteria; invasion with 4 residents + 1 mutant; three replications with
fresh seeds.

| Theory (selected design) | Profit rank: main / reps | Aggregate rank, six criteria: main / reps |
|---|---|---|
| Cobweb: adaptive price expectations | 2.10 / 3.53, 3.11, 3.24 | 3.50 / 4.17, 3.00, 3.50 |
| Simple heuristics: target-margin rule | 2.96 / 2.41, 2.29, 2.79 | 4.50 / 5.00, 3.83, 4.67 |
| Real options: inaction-band heuristic, price-based target | 3.50 / 2.85, 4.10, 6.16 | 3.67 / 2.83, 4.00, 6.33 |
| Heiner: reliability condition, price-based target | 4.29 / 5.00, 6.10, 3.98 | 4.17 / 6.17, 4.50, 3.17 |
| Organizational ecology: reorganize under threat of failure (scheduled in one replication) | 5.30 / 4.64, 6.25, 3.60 | 3.33 / 5.00, 6.83, 5.00 |
| Optimization: rational expectations (Cournot–Nash) | 6.33 / 6.53, 5.80, 5.85 | 5.83 / 6.17, 6.67, 6.00 |
| Imitation: imitate the best (the average in two replications) | 6.35 / 6.00, 5.73, 6.00 | 9.00 / 6.33, 7.67, 6.33 |
| Rule B (rigid benchmark) | 6.71 / 7.49, 5.74, 6.89 | 6.00 / 6.67, 4.17, 6.17 |
| Reinforcement learning: softmax value learner | 8.71 / 7.68, 6.98, 7.69 | 9.17 / 6.83, 5.67, 7.33 |
| Satisficing: aspiration search | 8.75 / 8.89, 8.91, 8.81 | 5.33 / 5.50, 8.50, 6.17 |

Selection rule versus always adjusting toward the same target (difference in profit and in CVaR 5%; * = 95% CI
excludes 0; main run / three replications):

| Selection rule · target | Profit | Downside (CVaR 5%) |
|---|---|---|
| Inaction-band heuristic · model-based | +27 / +192*, +63, +7 | +1825* / +3449*, +2007*, +1595* |
| Reliability condition · model-based | +5 / +201*, −44*, +20 | −1115* / +2391*, −305*, +16 |
| Reliability condition · price-based | −341* / −214*, −363*, −203* | −663* / −534*, −1078*, −450* |
| Aspiration · price-based | −1084* / −734*, −786*, −954* | +673* / +1480*, −190, +84 |

None of the six hypotheses fixed in the frozen plan was supported in the main run; PR2 was supported in two of the
three replications.

**What the tournament shows** (for these designs, tuning budgets and held-out environments)

* *Model-free beats model-based here.* The top three in every run are designs that need no demand model, and in the
  main run every theory that had the choice selected its model-free design. In these environments, with unannounced
  regime shifts, a misspecified model is the dominant source of decision error.
* *Restricting an unreliable flexible rule paid, as Heiner argued.* The inaction-band heuristic on the error-prone
  model-based target cut downside risk in every run without costing profit. Restricting the already reliable
  price-based target mostly hurt.
* *The learned reliability-condition rule, as implemented here, did not deliver that benefit robustly.* On the
  model-based target its profit effect was +5 (not significant) in the main run and its downside risk was worse; the
  replications disagree in sign. Estimating π, r, w, G and D case by case is itself a hard inference problem, and simple
  fixed restrictions (an inaction band, a target margin) capture the gains without the estimation error. This is
  consistent with Heiner's deeper argument that reliable behavior comes from rules rather than case-by-case
  assessment, and it qualifies the use of the condition as an agent's decision rule.
* *Inertia is rewarded on the downside.* Organizational ecology's crisis-driven agent is mid-field on profit but has
  the best aggregate rank over six criteria in the main run (3.33), through low downside risk and high survival; it
  does not keep that place in the replications (5.00, 6.83, 5.00).
* *Criteria matter.* Ranking by profit alone, by downside risk or by survival gives different orders; six of ten
  theories are Pareto-efficient in the main run, so claims of superiority must name the criterion.

## Mechanism study: results under the frozen plan

*Evidence type: simulation comparison.*

> **Revision note (6 October 2026, organizational ecology).** The study builds on the tournament (its rivals are the
> tournament's tuned agents, now including ecology's), so its plan changed from `e9b5cbda8a4c4ab7` to
> `11a279e507f246e2` and it was rerun. Earlier values: oracle +158 [91, 233] (model) and −29 (price); learned with the
> true model +18 and −35; learned −14 and −132; inaction band +28 and −144; cost of applying the principle +172
> (model: estimation +140, model bias +32) and +103 (price: estimation +5, model bias +97); memory up to 0.999 moved
> the learned agent from −14 to +1; boundary slope +1,401 per ln K (p = 0.011), K* ≈ 0.97 (oracle) and 1.06 (band);
> standardized effects hazard +0.22, lag +0.29, volatility +0.06; verdicts M1, M2 supported, M3, M4 (p = 0.07), M5 not
> supported; map R² 0.0–0.6. Two conclusions changed: model bias (M3) and the pooled boundary (M4) are now supported,
> and on the model-based target the cost of learning is now mainly model bias rather than estimation.

Study plan `11a279e507f246e2` (120 environments drawn by Latin hypercube over separate sources of uncertainty, 800
periods, oracle values from 3,200-period independent runs, background rivals tuned under tournament plan
`110b3146bb072c2c`). A focal firm aims at the same target as the always-adjusting rule and differs only in when it
moves. Profit per period relative to always adjusting (95% CI):

| Selection rule | Model-based target | Price-based target |
|---|---|---|
| Reliability condition, oracle (true reliability) | **+119 [72, 175]** | −73 [−129, −28] |
| Reliability condition, learned with the true model | +95 [45, 150] | −66 [−94, −42] |
| Reliability condition, learned (own model) | +13 [2, 24] | −170 [−206, −135] |
| Inaction-band heuristic | **+216 [141, 291]** | −137 [−160, −115] |

Cost of applying the principle (oracle − learned): model-based target +106 [58, 161], of which estimation from
limited experience +24 [−20, 66] and judging with a misspecified model +82 [34, 137]; price-based target +97
[38, 147], of which estimation −7 [−54, 31] and model bias +104 [75, 134]. Pooling more experience (memory 0.9 to
0.999) left the learned agent between +3 and +10, far from the oracle's +119.

Boundary: on the model-based target the oracle's gain rises with the flexible rule's error-to-signal ratio K (slope
+530 per unit of ln K, p = 0.011) and breaks even at K ≈ 0.98; the inaction band's gain breaks even at K ≈ 1.03
(p < 0.001); pooled over both targets the slope is +289 (p = 0.031). Restriction pays once the flexible rule's error
is about as large as the adjustment it should make. Among the sources of uncertainty, the oracle's gain is driven
by the model-updating lag (+0.42 [0.32, 0.53], standardized) rather than by the shift hazard (+0.05 [−0.07, 0.24])
or risk (volatility +0.12 [−0.12, 0.30]).

Verdicts under the frozen plan: M1 (the principle pays with known reliability) **supported**; M2 (applying it is
costly on both targets) **supported**; M3 (model bias on the model-based target) **supported**; M4 (pooled boundary
slope) **supported**, p = 0.031; M5 (shift hazard minus volatility coefficient) not supported, CI [−0.30, 0.29]
(the misspecification that matters here is the updating lag, not the hazard).

Maps: the metamodels' cross-validated R² ranges from about 0 (satisficing, reinforcement learning, ecology, rule B)
to 0.45 (adaptive expectations), so the winner maps are only partly predictable from the environment; the page
reports the cross-validated R² with every map.

**What the mechanism study adds.** In this market, knowing when deviations are reliable is worth a large gain
exactly where the flexible rule is error-prone, with a break-even near K = 1, and it costs profit where the flexible
rule is already reliable. The oracle that delivers this gain is a researcher construct. An agent that must learn its
own reliability keeps little of it, mainly because it judges its past decisions with a misspecified model; judging
with the true model recovers most of the gain on the model-based target. More experience does not fix this when the
environment keeps shifting. A fixed inaction band captured more than the oracle's gain on the model-based target
(+216 against +119), without any estimation.

The plan was revised during development, before it was frozen: a first version represented each theory by one
design (and, earlier still, the reliability-condition agent used an unfiltered target). Both earlier versions also
placed the reliability-condition agent in the lower half.

## Rule choice: results under the frozen plan

*Evidence type: simulation comparison.*

> **Revision note (6 October 2026, organizational ecology).** The rules are tuned under the tournament plan, which
> changed (see above), so the study's plan changed from `0f7bb98f8f7e8e44` to `f15f62149d08e800` and it was rerun.
> Earlier results: E1 slope +0.003 (p = 0.95) not supported, E2 slope −0.087 (p = 0.009) supported, E3 −0.005
> [−0.18, 0.15], E4 +4.9 (p = 0.002) supported; at β = 4 the restricted share rose from 51% to 60% and the change
> rate fell from 57% to 42%; at β = 16 the filtered best reply fell from 32% to under 1% and the target-margin
> heuristic rose from 0.5% to 47%. E1 and E2 swapped verdicts.

Plan `f15f62149d08e800` (uncertainty levels u = 0, 0.25, 0.5, 0.75, 1, where cost volatility, perception error and
the demand-shift hazard rise together; intensities of choice β = 0, 1, 4, 16; 8 replications; 12 firms; 1,500
periods; rules tuned under tournament plan `110b3146bb072c2c`).

| | E1 restricted share rises with uncertainty | E2 change rate falls with uncertainty | E3 selection raises the restricted share at u = 1 | E4 restricted share widens the distance from Nash |
|---|---|---|---|---|
| Result | slope +0.158 (p = 0.0004) | slope −0.044 (p = 0.17) | +0.031 [−0.083, 0.131] | +2.8 per unit of share (p = 0.004) |
| Verdict | **supported** | not supported | not supported | **supported** |

With moderate selection (β = 4) the restricted share rises from 46% at u = 0 to 59% at u = 1, but the change rate does
not fall (45% and 46%). Under strong selection (β = 16) the flexible optimizer (filtered best reply) falls from 29% of
firms at u = 0 to 5% at u = 1 and the restricted share rises from 33% to 55%, while the target-margin heuristic, a
simple rule that is not one of the three restricted rules, holds between 21% and 44% of firms at
every level (32% at u = 0, 34% at u = 1). Populations move toward restricted rules and away from sophisticated optimization under
uncertainty, as Heiner argued, but they do not change output less often, and performance-based selection favors a
simple heuristic as much as an explicit restriction. Markets with more rule-governed firms stay further from the
Cournot–Nash price.

## Validation against data

*Evidence type: simulation comparison (field patterns, calibration recovery on synthetic subjects).*

**Field patterns** (registered tuned parameters, 12 markets per pattern, 600 periods): 4 of 6 documented patterns
reproduced. Naive price expectations produce cobweb cycles (lag-1 autocorrelation −0.80); slow adaptive
expectations dampen them (price s.d. 18.4 vs 37.4); imitate-the-best firms produce 1.19 × the Cournot–Nash output;
prices lie between cost and the monopoly price (mean price 46.0, cost 42.1). **Not reproduced:** excess volatility (a
mixed market's mean price lies within 9.2% of the Nash price but fluctuates only 0.90 × as much; with the agents
retuned for the ten-agent tournament this pattern, reproduced before at 4.3% and 1.06 ×, no longer holds) and lumpy
adjustment. Inaction-band firms change output in only 34% of
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
same random draws per block for everyone, 15 agent designs recorded in shadow mode. Hypotheses fixed in the frozen protocol: X1
participants change output less often under high than low uncertainty (paired bootstrap); X2 under high
uncertainty, participants who change less often earn more relative to their rivals (OLS). On simulated participants
the classification recovers the generating design for every participant. **No human data have been collected yet**;
ethics approval and informed consent are needed before collecting data.

## Empirical validation on public data: results under the frozen plan

*Evidence type: out-of-sample behavioral prediction (observational fits to published experimental data; not causal).*

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
restrictions (an inaction band) do no better. The data are consistent with the population-level claims: behavior is
heterogeneous, the documented newsvendor patterns are present, simulated Cournot markets reach the human level of
competition, and lower competence (time pressure) shifts people toward simple or restricted rules. The generative
check also shows a specific failure of the simulated rules: people keep their decision unchanged far more often than
the fitted rules plus decision noise imply, so human behavior is more inert than any of the flexible rules. As an
exploratory observation outside the frozen plan, restricted rules are the best description for more participants where
information is poorer: 31% under aggregate and 11% under individual information in the Cournot data.

## Generalization: results under the frozen plan

*Evidence type: simulation comparison.*

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

Verdicts under the frozen plan: G1 (boundary in every task) **supported**; G2 (restriction pays only where K is high)
**supported**; G3 (learning the reliability is costly in every task) not supported, because in the inventory and
investment tasks the learned rule does as well as the oracle; G4 (the learned RC beats always deviating where K is
high) **supported**. With binary actions K is 0 or infinite when one rule is never wrong over a run, so K is clipped
to [0.05, 20] in the boundary analysis; in the investment task the linear fit on ln K is poor (R² 0.16), and the gain
by tercile turns positive between K ≈ 0.9 and 1.6.

**What generalization adds.** In all three tasks restriction pays where the flexible rule is unreliable and is
worthless where it is reliable, with a break-even error-to-signal ratio near or below 1, as in the market (K ≈ 0.98).
In the three tasks tested, the boundary is therefore not specific to cobweb markets; it appears in decisions with a
default, a flexible alternative and a difficulty–competence gap of the kinds implemented here. What differs is the cost of applying the principle: in these tasks the gain of a
deviation can be judged exactly from what the agent observes, so a learner captures the oracle's gain; in the
market, where outcomes depend on rivals and on a misspecified model, it does not.

## Solvable benchmark: results under the frozen plan

*Evidence type: analytical verification (T1) and simulation comparison (T2–T4).*

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

Verdicts under the frozen plan: T1 **supported**; T2 (with symmetric stakes nothing beats the filter) **supported**; T3
(restricting the filter's moves pays with lopsided stakes, more so as ρ rises; slope +0.043 per unit of ln ρ,
p < 0.001) **supported**; T4 (restriction never beats the filter with the optimal offset) **supported**.

**What the benchmark pins down.** Optimal filtering and Heiner agree whenever the stakes are symmetric: the best
response to news is the Kalman gain and no restriction helps. With lopsided stakes they part: certainty equivalence
says the stakes should shift the level of the action but not the response to news, whereas the reliability
condition says moves in the costly direction should be restricted. With the same information, the restricted rule
cuts the loss by up to a third, and upward moves fall from half of all periods to 3%. That prediction separates the
reliability condition from certainty-equivalent filtering, and it holds in this benchmark. Its limit is equally clear: a Bayes-optimal agent that builds the stakes into its estimate
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
heiner_abm/arena.py        rival agents, shared-market simulator, equal-budget tuning, frozen-plan tournament
heiner_abm/mechanisms.py   oracle vs learned reliability condition, boundary test, uncertainty-type maps
heiner_abm/rulechoice.py   endogenous rule choice with logit switching (Brock & Hommes 1997)
heiner_abm/patterns.py     pattern-oriented validation against documented field patterns
heiner_abm/calibration.py  per-subject out-of-sample fitting of every theory's rule to laboratory data
heiner_abm/stepper.py      a market that advances one period at a time, for human participants
heiner_abm/experiment.py   human experiment protocol, classification and frozen-plan tests
heiner_abm/empirical.py    validation against public experimental data: protocol, adapters, generative check
heiner_abm/datasets.py     loaders for the five public datasets, from the files as distributed
heiner_abm/tasks.py        generalization tasks: inventory, learning with shifting payoffs, irreversible investment
heiner_abm/tracking.py     single-firm tracking benchmark: exact Muth–Kalman solution and lopsided stakes
heiner_abm/focal.py        the theory under test: every theory's prediction for every hypothesis
heiner_abm/special.py      signature tests: each rival theory's distinctive prediction in the shared market
heiner_abm/theory_content.py  the nine theories, described with the same structure
heiner_abm/registered.py   registered results shown on the theory pages, tied to the plan hashes
heiner_abm/terminology.py  terms, kinds of uncertainty and evidence, scope of conclusions
heiner_abm/information.py  information-and-feedback specification, Observation, engine support, agents' needs
heiner_abm/bench_inventory.py, bench_bandit.py, bench_tuning.py, bench_checks.py
                           decision benchmarks: Bayes, robust optimization, bandits; tuning protocol; correctness checks
heiner_abm/gates.py        selection gates of the Adaptive rule: evidence statistics, uncertainty bounds, exploration
heiner_abm/gate_study.py   comparison of the gates with an ORACLE benchmark from independent runs
heiner_abm/learnability.py, learnability_market.py
                           learnability study: when reliability can be learned before change; market replication
heiner_abm/nk.py           NK-landscape environment: landscapes, searchers, gates, exact or best-known benchmarks
heiner_abm/model_spec.py   model specification: every agent's objective, information, actions, feedback, limits
heiner_abm/claim_status.py established theory, reduced form and proposed extensions
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
  unannounced structural change as regime shifts (an agent-relative stand-in for Knightian uncertainty); each rival
  theory as one or two agent designs (real options as an inaction-band heuristic) with stylized
  directional predictions.
* **Proposed extensions** (the laboratory's own constructs): the H-period forked measurement of the reliability
  condition and its decomposition; the reliability condition as a decision rule learned from experience; the oracle
  versus learned decomposition; the error-to-signal boundary K ≈ 1; the lopsided-stakes test against certainty
  equivalence; the confidence-sensitive and exploration-enabled selection gates. Results about these test the laboratory's operationalization, not Heiner's theory as published.

## Modeling notes

* The core model is a cobweb oligopoly: firms commit to output before a random raw-material cost is realized, using
  a Cournot best-reply quantity rule (model-based) or a margin-feedback quantity rule (model-free; configuration value
  "Bertrand", not price-setting Bertrand competition), and a selection rule that
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
  * unannounced demand-regime shifts with a model-updating lag (structural change unknown to the agents; not
    Knightian uncertainty in the unrestricted sense).
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
  counterfactual over H, released immediately. The findings that involved Adaptive agents (the directional
  tournament, the out-of-sample forecasts and the Adaptive preset) were rerun under the new rule on 6 October 2026;
  see the revision note under *Competing theories*.
* **Decision schedule** (`DECISION_SCHEDULE` in `heiner_abm/agents.py`, shown on the *Model & methods* page). In
  period t: (1) firms decide with information up to t − 1 (plus a share κ of the coming cost change for firms with
  cost foresight, a competence parameter; κ = 0 in the baseline); (2) the market clears and cost c[t] is realized;
  (3) the researcher evaluates the counterfactual over H, never shown to ordinary agents; (4) every decision enters a
  pending-feedback queue with its maturity, t + W − 1 (estimated) or t + H − 1 (oracle); (5) items are released at
  maturity plus the observation delay and first used in the next period's decisions; items that would be released
  after the last period are never released (no partial feedback); (6) optional evolution. The vectorized engine follows it
  exactly, the tournament agents and the interactive market follow the same order (act, clear, update), and
  `tests/test_information.py` checks that changing shocks after a period changes no earlier decision or learned
  state in any engine.
* **Oracle treatment.** The researcher's counterfactual can be given to Adaptive agents only by choosing feedback
  `"oracle"` explicitly; it is then released at maturity (t + H − 1) and results must be labeled as oracle
  benchmarks. The former `"lookahead"` option released it immediately, so with H > 1 agents learned from periods that
  had not yet occurred; it was removed on 6 October 2026 (with H = 1 it is identical to `"oracle"`). The mechanism
  study's oracle and true-model variants are separate, labeled designs that never enter ordinary tournament
  lineups.

## Information and feedback

Every scenario carries an information specification (`Scenario.info`, `heiner_abm.information.InfoSpec`), shared by
all engines:

* **Feedback treatments:** *chosen-action* (only the payoff of the agent's own action); *full* (payoffs of alternative
  actions, only where the design shows them: the generalization tasks do, the cobweb market does not, so it is
  rejected there); *estimated counterfactual* (the default for Adaptive firms: estimated from the firm's own believed
  demand curve and its observations); *oracle* (true counterfactual values, a clearly labeled diagnostic benchmark).
* **Information controls:** observation noise on observed prices and market output (its own random stream);
  observation delay (outcomes and feedback arrive d periods late); demand knowledge (none, believed, or the true
  curve as ORACLE information); announced regime shifts; visibility of rivals' individual actions and payoffs; and
  observability of censored outcomes such as unmet demand (not applicable in the cobweb market, which clears).
* **Observation object.** Firms decide from an `Observation` built only from observed periods; it carries no
  researcher-only quantity and no future state (apart from the share κ of the coming cost change that cost foresight,
  a competence parameter, grants). The interactive market builds the participant's screen from one as well.
* **Engines and agents.** Each engine declares the specifications it can run: the cobweb-market engines run any
  supported one; the tournament, task and tracking engines are covered by frozen plans and run only their native
  specification. Each agent declares what it needs (feedback, a demand model, visible rivals). An agent that needs
  information the specification withholds is reported as unsupported and is not run; experiments that would need it
  are shown as *not run* (and excluded from scoring) rather than run with extra information. Oracle and true-model
  variants are labeled as such.
* **UI and export.** The sidebar's *Information & feedback* section sets the specification; the *Information &
  feedback* page shows every agent's access, what each engine supports and a compatibility checker. Every results
  download comes with the specification that produced it (JSON). The default specification reproduces every
  reported result exactly.

## Reliability gates under uncertainty (proposed extension)

The Adaptive selection rule decides whether to adopt its production rule's recommendation x* instead of the default
x_B (rule B). `heiner_abm/gates.py` makes that decision explicit as a **gate**; the recommendations are identical in
every gate, so differences come from selection. Heiner (1983) did not propose these gates; they are the laboratory's
operationalizations of acting on uncertain estimates of one's own reliability.

* **Target.** A = E[payoff over the next W periods with x* − the same with x_B | information at decision time],
  estimated per size bin of the recommended change, gross of the adjustment cost c, which is a separate setting
  (`AdaptiveParams.adjust_cost`) used only as the decision threshold and in net-payoff accounting.
* **Gates.** *Estimated gain* (existing, the default): adopt if the learned gain ≥ c; with no evidence it adopts (the
  existing optimistic start); with c = 0 it reproduces the earlier rule exactly. *Confidence-sensitive*: adopt only
  when the bin has at least `min_evidence` effective feedback items and the lower confidence bound exceeds c.
  *Exploration-enabled*: learns only from its own realized payoffs (chosen-action feedback, so it also runs where
  counterfactual feedback is unavailable); per bin it compares randomized trials of adopting and keeping, acts when
  the interval excludes c and otherwise runs a randomized trial with probability `explore_rate` (separate random
  stream), else keeps. *ORACLE benchmark*: adopts where the true bin advantage, estimated by the researcher from
  independent runs, exceeds c; the table is accepted by no other gate.
* **Estimator.** Exponentially weighted mean (memory λ), Kish effective sample size, unbiased weighted variance and a
  Student-t lower bound (Welch–Satterthwaite for the two arms). A bin without evidence has no estimate; missing
  outcomes are never counted as zero gains. The bound's nominal coverage assumes independent feedback with a common
  mean, which overlapping judgement windows, drift, demand shifts and adaptive sampling violate, so no nominal
  coverage is claimed; the comparison measures it against the ORACLE benchmark.
* **Environmental change.** `on_change = "forget"` relies on exponential memory; `"reset"` discards evidence and
  pending feedback when the firms' information shows a change (an announced shift, or an update of the demand curve
  they are given).
* **Comparison** (`heiner_abm/gate_study.py`, page *Reliability gates under uncertainty*): every gate on the same
  markets (common random numbers), with false adaptations and missed opportunities judged by the realized
  counterfactual and by the ORACLE table, net payoff, calibration, lower-bound coverage, learning delay and net payoff
  around regime shifts. The ORACLE benchmark decides per bin, so it is a benchmark, not an upper bound. No results of
  this comparison are reported here; they are produced on the page from the chosen settings.
* **Tests** (`tests/test_gates.py`): agreement of the two engines for every gate; future shocks change no earlier
  decision, prediction or evidence; feedback is released at maturity plus delay and used only afterwards;
  initialization and sparse-data behavior; limiting cases (unreachable evidence or no exploration equal rule B;
  extreme oracle tables equal Always and Never; confidence 0.5 gives the point estimate); evidence accumulation and
  forgetting; nominal coverage under independent draws only; separation of the oracle benchmark.

## Decision benchmarks

Three established decision methods are run in two tasks of the generalization study, whose environment generators
(`heiner_abm/tasks.py`, part of a frozen plan) are imported unchanged.

| Benchmark | Task and design | Agents (role) | Sources |
|---|---|---|---|
| A. Bayesian learning with change detection | Inventory (newsvendor): mean demand jumps by N(0, 30²) with the environment's hazard, clipped to [30, 170]; demand max(0, mean + σ·e), observed with noise τ; order = 0.8 critical fractile of the predictive | Exact grid filter of this process with the true parameters, ordering by posterior expected payoff (**benchmark**, not ranked); Bayesian online change-point detection with a misspecified reset model, tuned (competitor) | Adams & MacKay (2007); Fearnhead & Liu (2007); Arrow, Harris & Marschak (1951) |
| B. Distributionally robust optimization | The same newsvendor; the last N observed demands | Robust order against every distribution within modified-χ² distance ρ of their empirical distribution, ρ and N calibrated on training environments; empirical (SAA) order as comparator; ρ = 0 reduces to SAA | Ben-Tal et al. (2013) |
| C. Nonstationary bandit | The learning task with **chosen-action feedback**: only the chosen option's payoff is observed | Sliding-window and discounted UCB, UCB (stationary baseline), ε-greedy with constant step size (competitors); oracle (not ranked); the task's full-feedback learner as a separate reference, not a bandit | Garivier & Moulines (2011); Auer, Cesa-Bianchi & Fischer (2002); Sutton & Barto (2018) |

* **Protocol** (`heiner_abm/bench_tuning.py`): disjoint training, validation and test environments; random search where
  budget b means the first b sampled configurations; the configuration chosen at each budget is scored on validation
  environments (the curve of tuning performance against the evaluation budget, with the training runs spent); the
  final configuration is the best of these on validation; only it is run on the test environments. Correctly
  specified benchmarks, oracles and references are reported beside the ranking of competitors, never in it.
* **Correctness first** (`heiner_abm/bench_checks.py`, also shown on the page): grid filter = enumeration of all paths;
  BOCPD = enumeration of all change-point configurations, and = the conjugate posterior with hazard 0; the documented
  process reproduces the task's generator draw for draw; ρ = 0 gives the SAA objective and optimum; the worst case
  equals mean + √(ρ·variance) when no weight hits zero, a numerical optimum otherwise, and the largest loss for
  ρ ≥ N − 1 (minimax order); SW-UCB with τ ≥ T and D-UCB with γ = 1 equal UCB; bandit agents ignore payoffs they did
  not observe; noise-free cases give the analytic choices.
* **Assumptions and computation** of every agent are listed on the page and in the model specification table (for
  example, BOCPD is O(R) per period for R kept run lengths; the robust order is an exact O(N log N) worst case inside
  a golden-section search; the grid filter is O(G²) per period).
* No results of these comparisons are reported here; they are produced on the page (Quick check or Full) with the
  plan, environments and information specification exported alongside.

## When can reliability be learned before the environment changes? (proposed extension)

`heiner_abm/learnability.py` (inventory task) and `heiner_abm/learnability_market.py` (market replication). The
question, the **learnability ratio** and the operationalizations are this laboratory's own proposal; the ratio is a
construct whose measurement and usefulness the study tests, not an established quantity.

* **Construct.** On independent pilot paths, per regime: the observations needed to determine the sign of the
  advantage of adapting with 90% one-sided confidence, n_needed = (z₀.₉ s / |μ_r|)², divided by the informative
  observations the agent actually receives in that regime; R is the median ratio over regimes. μ_r is the regime's
  true mean advantage (a researcher-only counterfactual used for measurement, never shown to the agents).
* **Manipulated separately:** outcome volatility, observation error, regime-change frequency, feedback availability
  (unobserved periods give no evidence), the gates' memory, and default quality (slow forecast, a fixed default that
  deteriorates after the first change, a biased and a dominated default).
* **Policies** on shared exogenous paths (common random numbers): always adapt, retain the default, inaction band,
  the existing estimated-gain gate, the confidence-sensitive gate, Bayesian change detection (BOCPD) and the
  distributionally robust order. The last two have no counterpart in the market's adoption decision and are reported
  there as not applicable.
* **Data sets.** Disjoint seed blocks for training (all tuning), pilot (R and the better fixed rule) and untouched test
  paths; tests on two process families never used for tuning (switching and drifting means), not only new seeds.
* **Outcomes.** Primary: net payoff per period (payoff minus the adaptation cost c in every period whose order departs
  from the default). Secondary: regret against perfect information, downside loss (CVaR 5%), calibration of the gates'
  predicted advantage, adaptation rate, missed opportunities and recovery delay after changes.
* **Hypotheses** H1–H4 (the advantage of the confidence-sensitive gate over the better fixed rule falls with log R;
  R adds cross-validated explanatory power beyond volatility and noise; the relation transfers to new families; it
  replicates in the market) and **negative controls** NC1–NC3 (perfect information with costless adaptation; a
  dominated default; a stable environment with abundant feedback), each with its decision rule. Effects are paired
  differences with 95% bootstrap intervals; rank differences are not interpreted as one theory's superiority.
* **Registration.** Plan `ec9b781e125e289d` is frozen in the repository (`registered.LEARN_PLAN`); the exported
  document is [`docs/learnability_registration.md`](docs/learnability_registration.md) and can be downloaded from the
  page. It has **not** been preregistered with an external registry.

### Results under the frozen plan `ec9b781e125e289d`

One run of the registered plan on commit `b748da3` (the specification commit), about 11 minutes. Tuned on training
configurations only: forecast gain 0.1, band b = 0.5, BOCPD (hazard 0.005, prior sd 30, obs sd 25), robust order
(N = 20, ρ = 0.01); market band threshold 50. Tables: [`docs/learnability_results/`](docs/learnability_results/).

| | Decision rule | Result | Verdict |
|---|---|---|---|
| H1 | slope of the gate's advantage on log10 R, CI below 0 | −1.44 [−2.36, −0.59], 30 jump configurations | supported |
| H2 | cross-validated R² gain of log10 R over volatility and noise, CI above 0 | +0.23 [−0.46, +0.84] | **not supported** |
| H3 | H1 slope in the new families (switching, drifting) | −1.75 [−2.79, −1.01], 30 configurations | supported |
| H4 | H1 slope in the market | −1625 [−2922, −478], 16 configurations | supported |
| NC1 | perfect information, costless adaptation: nothing beats always adapting | every policy − always ≤ −0.16 per period | passed |
| NC2 | dominated default: gates adapt ≥ 90%, lose ≤ 5% | adaptation 1.000 and 0.995; −31.80 and −32.85 vs −31.78 | passed |
| NC3 | stable, abundant feedback: R < 1, gate not worse | R = 0.076; gate − default 0.00 [−0.00, +0.00] | passed |

How to read this:
* **The construct's usefulness is not established.** R ordered the gate's advantage in the inventory task, the new
  families and the market (H1, H3, H4), but it did not add cross-validated explanatory power beyond volatility and
  observation noise (H2): the interval is wide and includes zero. In the 30 test configurations log10 R correlates
  with log σ (r = 0.37) and with feedback availability (r = −0.55), so H1 and H3 alone cannot separate R from the
  factors it is built from.
* **The confidence-sensitive gate rarely paid.** In the inventory task the better fixed rule was always adapting in
  every configuration; the gate's advantage was below zero with its interval excluding zero in 22 of 30 jump and 29 of
  30 new-family configurations, and above zero in none. The slope says how *much* it lost, more where R is large; it
  does not show a region where waiting for evidence wins. Its losses are largest with scarce feedback (avail 0.2:
  −3.40 [−5.51, −1.26] against always adapting).
* **The market replication is weak evidence.** Retaining the default was the better fixed rule in 15 of 16 market
  configurations, per-configuration intervals are wide (only 5 of 16 exclude zero, 4 below and 1 above), and the
  slope comes from 16 points. It is a direction, not a precise effect.
* **Other policies (secondary, not hypotheses).** Against the better fixed rule, Bayesian change detection's
  interval was above zero in 20 of 30 jump configurations and below in 4; the robust order 9 above and 10 below; the
  estimated-gain gate 3 and 7; the inaction band 0 and 11. These are per-configuration counts with intervals, not a
  ranking, and the tuned settings come from one training family; they are not evidence that one theory is superior.
* **Secondary outcomes** (jump-family test paths, means with 95% intervals; `secondary.csv`): regret per period
  always 9.1 [8.4, 9.9], estimated-gain gate 9.1 [8.4, 9.9], confidence gate 11.0 [10.2, 11.9], BOCPD 7.3 [6.8, 7.8],
  robust 8.3 [7.8, 8.9], default 52.5 [48.1, 57.3]; the gates adapt in 81% and 63% of periods and miss 14% and 29%
  of opportunities; recovery delay after a change 13.9 periods for always adapting, 17.3 and 21.3 for the gates.
  Both gates share one learned estimate, so their calibration is identical (slope 0.76 [0.72, 0.79], bias −0.66).

## NK landscapes: interaction complexity, reliability learning and adaptation

`heiner_abm/nk.py`, page *NK landscapes (complexity)*. An NK landscape (Kauffman & Levin 1987; Kauffman 1993), as used
in organization science (Levinthal 1997; Rivkin 2000), represents **complexity**: N binary decisions whose payoff
contributions each depend on K_NK other decisions (0 ≤ K_NK ≤ N − 1; named K_NK to avoid confusion with the
error-to-signal ratio K). Complexity is not uncertainty. Two **separate uncertainty mechanisms** are added, each with its
own control and random stream: **observation noise** (every evaluation is the true payoff plus noise) and
**environmental change** (unannounced redraws of a share of the contribution tables). Interdependence, noise and change
can therefore be varied one at a time.

* **Environment.** Task interface as in the generalization tasks: a default (keep the current configuration), a
  flexible alternative (a proposed candidate) and a switch decision each period. Configurable interaction topology
  (adjacent, random, block or an explicit partner list), seeded contribution tables and change paths, an explicit
  evaluation budget, and a switching cost per changed component.
* **Searchers.** Local hill climbing; broader stochastic search (multi-flip proposals with annealing acceptance;
  Kirkpatrick et al. 1983); satisficing with an adaptive aspiration (Simon 1955; Cyert & March 1963); imitation of a
  leader observed on a stated share of its components only (Rivkin 2000); and, as a **proposed extension**,
  reliability-gated search: apparent improvements are verified once, independently, and the existing estimated-gain gate
  or the confidence-sensitive gate (`heiner_abm/gates.py`) learns per signal-strength bin whether acting on such
  improvements pays. The gates and an ungated reference use identical proposals, draws and evaluation budgets.
* **Benchmarks.** For N ≤ 16 every configuration is enumerated, so the maximum and every local peak are exact. For
  larger N the benchmark is the best-known payoff (multi-start full-information hill climbing and every configuration an
  agent held), labelled "best-known"; optimality is not claimed.
* **Outcomes.** Attained payoff, regret, evaluations used, false and missed improvements, escapes from local peaks and
  recovery after change (definitions in the module docstring), over multiple independent landscapes and starting
  configurations. Intervals are cluster bootstraps over landscapes, because runs on one landscape are dependent.
* **Checks** (`tests/test_nk.py`): K_NK = 0 is separable, with a single peak at the per-component optimum; seeded
  reproducibility; exact payoffs in a hand-computed small case; changes redraw only the stated share; information
  restrictions (decisions replay exactly from the recorded observations alone, without the landscape; no lookahead;
  verification released after the decision; imitators unaffected by a leader's unobservable components).
* **Settings.** N = 8 and K_NK = 0, 2, 4, 7 are configurable pilot settings, not optimal design values. No results are
  reported here; they are produced on the page with their settings.

## Research basis

`heiner_abm/literature.py` is the single source for the research behind the simulation. Each hypothesis records its
reliability-condition prediction, its alternative prediction and the rival theory it comes from, the studies that
support each side (with a note on how each bears on the hypothesis), and the contribution the test makes. The app
shows this on every hypothesis card and on the *Research & contribution* page, which also exports the bibliography
as BibTeX and APA and the evidence matrix as CSV. A test checks that every cited work exists in the registry, that
every hypothesis has research on both sides, and that the app text cites research only through the registry.
