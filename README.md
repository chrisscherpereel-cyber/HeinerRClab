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
python -m pytest -q
```

`pytest.ini` runs the suite across worker processes, and that is required rather than optional. The headless page
tests start about 300 full Streamlit app sessions; run in a single interpreter, the process corrupts its own memory
after roughly 60–250 of them and then either reports an impossible array shape or dies with a Windows access
violation, in a different module each time. Measured on 8–9 October 2026: one process failed or crashed in 4 of 7
runs; the same tests split across processes passed 7 of 7, and the whole suite under `-n auto` passed twice and ran
about six minutes faster. Capping OpenBLAS threads did not help, so it is not BLAS threading. The underlying fault is
in the Streamlit/numpy stack rather than in this repository; `pytest.ini` keeps each worker's share small enough to
avoid it. Forcing one process (`-n 0`) brings the flakiness back.

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

The app guides a study through **choose a question → configure the experiment → check validity → run → interpret → export**. The left panel has five destinations (Start, Experiment, Results, Validation, Reference), an interface **mode** (Explore for teaching and a first investigation, with simplified uncertainty presets that show the settings they change and quick previews; Research for controlled experiments, with every setting, validity checks and background runs; both use the same engines), the pages within the current section, every other page under *All pages*, the **theory under test** (the theory highlighted on hypothesis cards and theory pages; Heiner's reliability condition is the default, not a privileged position) the **agents** of every theory (one folded entry per theory, plus rule B, giving each agent design's decision rule and tuned parameters, read from the code) and the base market scenario used by the specialized market pages. Every earlier page remains available within these sections.

Within each section the simulation pages are split by fairness. **General simulations** compare the theories on equal terms: every theory takes part as agents (or, in the directional experiments, as a stated prediction for every experiment), with the same information, random draws and tuning density: Competing theories, Agent tournament, Rule choice and Signature tests by theory. **Special simulations** are built around Heiner's framework (rule B, the market model's flexibility φ, the CD-gap and the reliability-condition bookkeeping): Market lab, Experiment designer, Endogenous flexibility, Generalization tasks, When can reliability be learned?, NK landscapes, Mechanisms, Reliability gates, Dynamic RC, CD-gap explorer, Does the RC predict performance? and Heiner vs optimal filtering. Rival predictions appear there where they apply, but the models themselves are Heiner's.

**Start** (choose a question)

| Page | What it does |
|---|---|
| Start | Guided questions (each loads a documented, editable preset with its primary outcome and comparison), your own question from any environment, example (registered) studies, and saved experiments (open, duplicate, delete) |
| Hypothesis tests | H1 free flexibility · H2 profitability switch · H3 volatility · H4 fixed costs · H5 competition intensity · H6 regimes and equilibrium · H7 competence · H8 perception noise · H9 selection rules · H10 predictable behavior · H11 number of rivals · H12 model-updating lag. Each shows the RC prediction next to the alternative, the research behind both, the contribution, and a verdict |
| Risk vs structural change | Cost-volatility risk versus unannounced demand-regime shifts at matched unpredictability, for model-based (Cournot best reply) and model-free (margin-feedback) firms. The shifts are specified by the simulator and unknown to the firms; this is not Knightian uncertainty in the unrestricted sense. Event study of punctuated slow–quick–slow adjustment |

**Experiment** (configure and run a comparison; the specialized and registered study pages follow)

| Page | What it does |
|---|---|
| Experiment | The central workspace: six revisitable steps (question, environment, agents, information, design, run) editing one experiment specification. Environment, candidate generation, selection policy and information are kept separate; complete policies show their assumptions; one uncertainty panel (outcome variation, observation quality, environmental change, model knowledge, decision complexity) with advanced settings disclosed on demand; an agent table, an access matrix, a design table with validity checks, a workload estimate, preview runs in the page and research runs in a background process with progress and cancellation |
| Market lab | One market with editable heterogeneous firm agents. Shows price/cost dynamics, a per-firm reliability scoreboard, and a period-by-period decision inspector (correct deviations, type I and type II errors) |
| Experiment designer | Your own *what if* question: sweep one or two settings of the base market and plot any outcome, with common random numbers across conditions. The page explains its purpose, the steps, a worked example and what each outcome means; CSV export |
| Agent tournament | Every rival theory implemented as two agent designs competing in the same market (19 designs, including target × selection-rule composites). Equal tuning density (candidates per free parameter) on training environments, design selection on training data, held-out test environments, a frozen hashed plan with six hypotheses fixed in advance, six performance criteria (profit, downside risk, survival, volatility, regret, worst case), a selection-rule experiment, invasion tests, global sensitivity analysis and replication across seeds |
| Rule choice (emergence) | Firms switch between six rules (three restricted, three flexible) by recent performance with logit choice and an adjustable intensity of choice β (Brock & Hommes 1997). Rule shares, change rates, price volatility and distance from Cournot–Nash across uncertainty levels. Frozen plan with four pre-registered hypotheses |
| Endogenous flexibility | Firms imitate the most profitable rival's φ (plus mutation). Does volatility breed rigidity? |
| Competing theories | Heiner's RC against neoclassical optimization, real options (as operationalized here), cobweb stability, bias–variance / ecological rationality, satisficing, reinforcement learning, imitation and structural inertia. A tournament of nine discriminating experiments scores each theory's directional predictions, with every theory predicting every experiment; an agent track lets every theory's own agent (registered tuned design) compete against all others and rule B in every condition of six experiments (profitability, volatility, competence, noise, number of rivals, demand shifts); an out-of-sample horse race scores each theory's forecast of which firms benefit from flexibility, plus an encompassing test of whether the RC adds information beyond all rivals |
| Generalization | The same selection layers and boundary test in three other decision tasks with a default, a flexible alternative and a difficulty–competence gap: an inventory (newsvendor) task with shifting demand, a learning task with shifting payoffs and an irreversible investment task. Frozen plan with four pre-registered hypotheses |
| When can reliability be learned? (extension) | **Proposed extension.** Tests whether a learnability ratio (observations needed to learn the sign of the advantage of adapting, relative to the informative observations available within a regime) explains when the confidence-sensitive gate beats the better fixed rule, beyond volatility and observation noise; seven policies on shared paths, training/pilot/test separation, new process families, negative controls, market replication. Frozen in the repository, not externally preregistered |
| NK landscapes (complexity) | Search on NK landscapes with K_NK interacting components: hill climbing, stochastic search, satisficing, imitation with stated observability and (extension) reliability-gated search, under separately controlled observation noise and landscape change. Exact benchmarks by enumeration for small N, best-known otherwise; performance against interdependence, noise and change with landscape-clustered intervals |

**Results** (interpret a completed experiment; specialized mechanism analyses follow)

| Page | What it does |
|---|---|
| Results | Opens with the question, the comparison, the primary outcome and the kind of evidence, then the paired effect with its 95% interval and a plain interpretation against the smallest effect of interest; tabs for performance, behavior, mechanisms, robustness (split replications, per-replication spread), a decision inspector (what the agent observed, believed and proposed, why it acted, when feedback arrived; researcher-only values marked) and details & export (configuration, provenance, trial data, a reproducible bundle) |
| Heiner: mechanisms (oracle vs learned) | Principle versus implementation: focal-firm variants on a shared target (always, inaction band, learned reliability condition, the same learner judging with the true model, an oracle with true reliability, a memory grid), with the cost of applying the principle decomposed into estimation and model bias; a boundary test against the measured error-to-signal ratio K; standardized effects of each source of uncertainty; cross-validated metamodel maps of which theory does best where. Frozen study plan with five pre-registered hypotheses |
| Reliability gates under uncertainty (extension) | **Proposed extension.** The Adaptive rule's decision to adopt a recommendation is made by one of three gates with identical recommendations: the existing estimated-gain gate, a confidence-sensitive gate (adopt only with enough evidence and a lower confidence bound above the adjustment cost) and an exploration-enabled gate (learns from its own payoffs in randomized trials, without counterfactual feedback), plus an ORACLE benchmark estimated from independent runs. Reports false adaptations, missed opportunities, net payoff, calibration and lower-bound coverage, learning delay and performance after regime changes |
| Heiner: dynamic RC (1989) | Decomposes each decision's value into immediate, persistence and strategic-feedback parts. Tests Heiner's (1989) partial-adjustment bound β₀ = 1/((1+K)(1−f′)) against the profit-maximizing flexibility. Signal-detection ROC of each firm's decisions |
| Heiner: CD-gap explorer | Difficulty (Δ or noise) × competence (foresight κ) heatmaps of r, w, π, RC margin and the payoff to flexibility |

**Validation** (establish what the findings can support)

| Page | What it does |
|---|---|
| Validation | What the findings can support: analytical benchmarks, empirical data, human experiments, signature tests and predictive checks, each linked |
| Heiner vs optimal filtering (Muth–Kalman) | A single firm tracks a random-walk target observed with noise, where the best adjustment speed is the Kalman gain. The simulation reproduces the exact loss curve and optimum; then lopsided stakes test the prediction that is uniquely Heiner's against optimal filtering's certainty equivalence. Frozen plan with four pre-registered hypotheses |
| Decision benchmarks (Bayes, robust, bandit) | Established decision methods under one protocol (training, validation and test environments; tuning performance against the evaluation budget): Bayesian change detection, correctly specified and misspecified, and distributionally robust versus empirical optimization in the inventory task; sliding-window and discounted UCB in a chosen-action-feedback version of the learning task. Correctness is checked on analytic and enumerated cases first; benchmarks, oracles and the full-feedback reference are kept out of the rankings |
| Signature tests by theory | One test per rival theory of the prediction that characterizes it, with a criterion fixed in advance (results below); Heiner's tab links to his five special-test pages |
| Heiner: does the RC predict performance? | Random environments; each firm is compared with its own rigid twin (same shocks). The RC is estimated in the first part of each run and predicts the second part (**out-of-sample**), scored by AUC with environment-clustered bootstrap CIs |
| Field patterns | Pattern-oriented validation (Grimm et al. 2005): cobweb cycles, damping by adaptive adjustment, sticky and lumpy adjustment, imitation beyond Cournot–Nash, excess volatility around equilibrium and positive markups, each with a criterion fixed in advance and its sources |
| Calibration to experiments | Fits every theory's decision rule per subject to learning-to-forecast cobweb data (Hommes et al. 2007 design) or Cournot data (Huck et al. 1999 design) on the first half of periods and scores it on the second half; classifies subjects by best-predicting rule. Upload data or check recovery on synthetic subjects |
| Empirical validation (public data) | The five public datasets that can validate the simulation, their access, licenses and caveats; a protocol fixed in advance (seven hypotheses); loaders that read each repository's files as distributed; per-dataset analyses (out-of-sample rule comparison, generative check of simulated Cournot markets, newsvendor patterns, structural changes, time pressure); registered results |
| Play the market | Protocol 2.0: consent and allocation slot, comprehension checks, a practice block, then four blocks that each change one uncertainty mechanism (volatility, observation noise, regime change) relative to a baseline, in a Williams-square order on rotated exogenous paths; randomized decision aid (or the unaided control) and separately timed belief questions; records response times, information access, recommendations, choices and feedback. Download the decisions as CSV |
| Experiment analysis | Allocation schedule and protocol export; data quality and exclusions; randomized condition, aid and elicitation effects on adjustment probability, magnitude and relative profit with dependence-aware intervals; hurdle mechanism models with held-out prediction; parameter and model recovery (confusion matrix); the adjustment–profit association, labeled as such; simulation-based power from pilot data; the prospective analysis plan. Synthetic pilots are labeled and never presented as human evidence |

**Reference** (supporting detail without interrupting the workflow)

| Page | What it does |
|---|---|
| Reference | Theory, equations and methods, agent specifications, information access and the bibliography |
| Overview | The nine theories side by side (core claim, view of uncertainty, what triggers change, effect of uncertainty on the value of flexibility), the common testbed, how each fared, every hypothesis with its research, and the contribution to the literature |
| One page per theory | Heiner, optimization, real options, cobweb, simple heuristics, satisficing, reinforcement learning, imitation, organizational ecology. Same sections for each: origins, formal core, view of flexibility, an interactive illustration, how the laboratory implements it, how it fared in the registered runs, strengths and limits, references |
| Agents as implemented | Every agent in the laboratory as the code implements it: the shared decision cycle, the market-lab firms (production and selection rules, adaptive learning, endogenous flexibility), the nineteen tournament designs with their equations, parameters, registered tuned values and sources, the mechanism variants, the rule-choosing firms, the task agents, the tracking rules, the experiment's rivals and shadows, and the rules fitted to human data |
| Model & methods | Equations, schedule, measurement, statistics, baseline calibration, a full description of every preset scenario (setup, why to run it, the hypotheses it serves and a typical result), what a frozen plan is (and how it differs from external preregistration), the model specification table, the kinds of uncertainty and evidence, and verification |
| Research & contribution | The simulation's contributions to the literature, the evidence matrix (supporting and alternative research for every hypothesis), the research behind each rival theory and method, and the full bibliography with BibTeX, APA and CSV export |

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
| Neoclassical optimization | Better information raises an optimizer's profit; full information yields Cournot–Nash | profit slope on foresight -35.9 per unit (p = 0.000258); with full information the rational market's price is within 0.13% of Cournot–Nash on average | not supported |
| Real options | The value of flexibility rises with volatility | value of flexibility changes by -3.9 per unit of volatility (p = 0.00043) | not supported |
| Cobweb theory | Partial best-reply adjustment is stable only below φ = 4/(n + 1) | n = 3: theory 1.00, simulated 0.95; n = 4: theory 0.80, simulated 0.80; n = 6: theory 0.57, simulated 0.55; n = 8: theory 0.44, simulated 0.45 | **supported** |
| Simple heuristics | A simple rule beats the optimizer by more as estimation noise rises | heuristic's advantage changes by -29.3 per unit of noise (p = 1.23e-10); at the highest noise +500 [+410, +605] | not supported |
| Satisficing | Harder (more volatile) conditions trigger more search | change rate changes by -0.0017 per unit of volatility (p = 5.49e-21) | not supported |
| Reinforcement learning | Performance improves with experience in a stationary environment | relative profit improves by -72 [-247, +104] from the first to the last third in a stationary market, and by -202 [-386, -38] with unannounced shifts | not supported |
| Imitation | Imitating the best drives output above Cournot–Nash | imitate-the-best markets produce 1.25 [1.24, 1.27] × the Cournot–Nash output | **supported** |
| Organizational ecology | Inert organizations perform more reliably than flexible ones at every volatility | Δ = 2: +2 [+2, +3]; Δ = 10: +62 [+45, +77]; Δ = 20: +194 [+176, +215]; Δ = 30: +255 [+241, +269] | not supported |

Two of eight signature tests came out as predicted in this market. The cobweb stability boundary is reproduced almost
exactly (0.95, 0.80, 0.55 and 0.45 against 1.00, 0.80, 0.57 and 0.44), and imitation of the best pushes output 25%
above Cournot–Nash, as Vega-Redondo (1997) predicts. The optimizer's profit still falls with better foresight
(−35.9 per unit, p = 0.0003) rather than rising. In the real-options test, an always-adjusting firm without
adjustment costs lost more to its rigid twin as volatility rose; that test has no irreversibility or option to wait,
so it says that flexibility of this kind lost value with volatility here, not that real-options theory is wrong. The
heuristic beats the optimizer at every noise level, but its advantage shrinks rather than grows with noise; the
aspiration searcher changes slightly less, not more, often in more volatile markets; and the tuned reinforcement
learner does not improve significantly with experience over 1,500 periods. Each of these is a statement about the
implementation tested.

> **Revision note (8 October 2026, equal search density).** The signature tests use the tournament's tuned agents, so
> they were rerun and their source fingerprint changed from `71d28315d7da1800` to `1ca69a156d8a3b3a`. **The earlier
> numbers require replication.** One verdict flipped: organizational ecology was **supported** (the firm that
> reorganizes only under threat of failure had *less* variable profit than its always-adjusting twin at every
> volatility: Δ = 2 −6 [−9, −3]; Δ = 10 −94 [−101, −87]; Δ = 20 −310 [−326, −293]; Δ = 30 −753 [−778, −728]) and is
> now **not supported**, with the sign reversed at every volatility. The test compares the crisis-driven design with
> an always-adjusting firm on the same price-based target, and the two are tuned separately, so their adjustment
> speeds confound the comparison: under the old flat budget the inert design moved in much smaller steps than its
> twin (φ = 0.15 against 0.88), and under equal-density tuning it moves in *larger* steps (φ = 0.76 against 0.58).
> The earlier result therefore rested on step size rather than on inertia, and the prediction does not survive once
> the step size moves the other way. The test as written cannot separate inertia from step size; that is a limitation
> of the test, not evidence either way about structural-inertia theory. Earlier values for the other tests: optimization −10.6 per unit of
> foresight (p = 0.08); real options −3.7 per unit of volatility (p = 0.029); simple heuristics −8.8 per unit of
> noise (p = 0.0078), at the highest noise +409 [+311, +525]; satisficing −0.0011 per unit of volatility
> (p = 0.0021); reinforcement learning −73 [−283, +156] and +148 [−163, +424]; imitation 1.24 [1.22, 1.26] ×
> Cournot–Nash; cobweb unchanged.

> **Revision note (7 October 2026, every theory in every experiment).** Reinforcement learning and imitation joined
> the directional tournament, and every theory now states a prediction for every experiment (before, neoclassical
> optimization, real options, cobweb stability, bias–variance and satisficing made no prediction for some
> experiments). The four reference runs were repeated on the old and the new code: the old code reproduced every
> record, and every experiment's result was bit-identical on the new code, so only the scoring changed. Earlier
> records (Bertrand seeds 1 and 3; Cournot, both seeds): bias–variance 4/2/0 and 3/2/1; 3/2/1. Satisficing 0/1/0
> everywhere. Neoclassical 2/4/1 and 1/4/2; 3/2/2. Real options 2/4/1 and 2/3/2; 2/3/2. Cobweb 1/5/1 and 2/4/1;
> 4/3/0. Heiner and ecology are unchanged, and Heiner keeps the best net record in all four runs.

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
| Bias–variance / ecological rationality | 5/2/2 and 4/2/3 | 5/2/2 (both seeds) |
| Satisficing / aspiration-level search | 5/3/1 and 4/3/2 | 4/3/2 (both seeds) |
| Reinforcement learning | 3/4/2 and 2/4/3 | 4/3/2 (both seeds) |
| Neoclassical optimization | 3/5/1 and 2/5/2 | 4/3/2 (both seeds) |
| Real options | 3/5/1 and 3/4/2 | 2/5/2 (both seeds) |
| Structural inertia (organizational ecology) | 4/5/0 and 5/4/0 | 3/6/0 (both seeds) |
| Cobweb stability theory | 1/7/1 and 2/6/1 | 4/5/0 (both seeds) |
| Imitation / evolutionary selection | 0/7/2 and 1/6/2 | 2/5/2 (both seeds) |

Every theory states a prediction for each of the nine experiments (read from its core mechanism where its literature
does not address an experiment; the reasoning is shown on the page), so no theory is scored on a convenient subset.
The *agent track* on the same page complements these records: every theory's own agent competes in every condition
of six experiments (an exploratory analysis run on the page; no registered numbers).

Heiner had the best net record in all four runs (a record of predicted signs in these experiments, not a ranking of
the theories in general). Its one contradiction differs by market: in margin-feedback ("Bertrand") markets
perception noise *raised* the payoff to flexibility, and in Cournot markets free flexibility did not hurt at the
tested low-profit level. Evolved flexibility showed no volatility gradient in any run. The best adjustment speed
fell with noise, as Heiner predicts, but optimal filtering (Muth 1960; Kalman 1960), bias–variance reasoning,
satisficing and reinforcement learning predict the same, so that test does not discriminate between them.

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

Registered plan `22393384781bed84`: two designs per theory, each tuned at 24 candidate settings **per free
parameter** (24–144 candidates depending on the design) × 2 rounds on 24 training environments; each theory enters
with the design that scored higher on training data; evaluation on 40 × 2 held-out environments (800 periods); six
criteria; invasion with 4 residents + 1 mutant; three replications with fresh seeds.
Reproduce with `python tools/rerun_tournament_findings.py` (about 10 minutes).

| Theory (selected design) | Profit rank: main / reps | Aggregate rank, six criteria: main / reps |
|---|---|---|
| Simple heuristics: target-margin rule | 2.01 / 3.08, 2.00, 3.67 | 4.00 / 4.33, 4.33, 4.50 |
| Cobweb: adaptive price expectations | 3.91 / 2.80, 5.01, 1.82 | 4.00 / 2.50, 2.83, 2.33 |
| Optimization: rational expectations (Cournot–Nash) | 5.11 / 6.11, 6.58, 5.50 | 3.50 / 5.00, 7.50, 4.50 |
| Organizational ecology: reorganize under threat of failure (scheduled in one replication) | 5.30 / 4.03, 3.77, 5.74 | 6.00 / 4.83, 4.50, 7.17 |
| Real options: inaction-band heuristic, price-based target | 5.31 / 4.19, 3.67, 2.98 | 6.17 / 2.67, 4.00, 3.83 |
| Rule B (rigid benchmark) | 5.55 / 6.49, 6.11, 6.84 | 3.83 / 5.83, 5.00, 5.83 |
| Imitation: imitate the best (the average in two replications) | 5.89 / 6.08, 5.90, 6.05 | 8.33 / 8.83, 6.67, 6.00 |
| Reinforcement learning: softmax value learner | 6.34 / 7.05, 7.01, 7.42 | 5.83 / 6.00, 6.00, 6.67 |
| Heiner: reliability condition, price-based target | 7.19 / 5.90, 5.86, 5.80 | 7.67 / 7.00, 7.17, 7.33 |
| Satisficing: aspiration search, price-based target | 8.39 / 9.29, 9.07, 9.18 | 5.50 / 7.83, 6.33, 6.83 |

Selection rule versus always adjusting toward the same target (difference in profit and in CVaR 5%; * = 95% CI
excludes 0; main run / three replications):

| Selection rule · target | Profit | Downside (CVaR 5%) |
|---|---|---|
| Inaction-band heuristic · model-based | +14 / +29, +239*, +77* | +1728* / +1727*, +2635*, +1936* |
| Reliability condition · model-based | +23* / +57*, +109*, +46* | +29 / +218*, +536*, +556* |
| Reliability condition · price-based | -248* / -505*, -395*, -520* | -1226* / -2556*, -2862*, -2034* |
| Aspiration · price-based | -946* / -980*, -679*, -937* | +620* / -1363*, -302, -304 |

None of the six hypotheses fixed in the frozen plan was supported in the main run; PR4 (the
reliability-condition agent invades every rival population) was supported in one of the three replications.

*What "equal tuning budget" means here.* Every design is searched at the same **density**: `Prereg.budget_per_parameter`
candidate settings for each of its free parameters (24, Latin-hypercube, defaults included as candidate 0), on training
environments only. Designs carry one to six free parameters, so budgets run from 24 to 144 candidates; the *Agent
tournament* page reports free parameters, candidates and candidates per parameter next to each design.
`Prereg.budget_rule = "per_design"` restores the legacy rule of a flat 24 candidates for every design.

> **Revision note (8 October 2026, equal search density).** Until 7 October 2026 every design got the same *number* of
> candidates whatever its dimension, so a six-parameter design was searched six times less thoroughly than a
> one-parameter one — and the two reliability-condition designs have the most free parameters of any theory, so the
> rule searched the focal theory's space least thoroughly. The plan hash changed from `110b3146bb072c2c` to
> `22393384781bed84`, the mechanism study from `11a279e507f246e2` to `bbfaed9e70b900d6` and rule choice from
> `f15f62149d08e800` to `d2f88699dc6e6eb6`; the tournament (main run and three replications), both studies, the
> signature tests and the field patterns were rerun. **Everything reported under the superseded hashes requires
> replication and is not comparable one for one with the tables above.** Design selection was unchanged for every
> theory. Earlier values: profit ranks, main run / three replications — cobweb 2.10 / 3.53, 3.11, 3.24; heuristic
> 2.96 / 2.41, 2.29, 2.79; real options 3.50 / 2.85, 4.10, 6.16; Heiner 4.29 / 5.00, 6.10, 3.98; ecology 5.30 / 4.64,
> 6.25, 3.60; optimization 6.33 / 6.53, 5.80, 5.85; imitation 6.35 / 6.00, 5.73, 6.00; rule B 6.71 / 7.49, 5.74, 6.89;
> reinforcement learning 8.71 / 7.68, 6.98, 7.69; satisficing 8.75 / 8.89, 8.91, 8.81. Earlier selection-rule effects
> (profit / CVaR 5%, main run): band · model +27 / +1825*, reliability condition · model +5 / −1115*, reliability
> condition · price −341* / −663*, aspiration · price −1084* / +673*. PR2 was supported in two of the three
> replications; it is now supported in none, and PR4 in one.

**What the tournament shows** (for these designs, tuning budgets and held-out environments)

* *Model-free beats model-based here.* The top three in every run are designs that need no demand model, and in every
  run each theory that had the choice selected its model-free design. In these environments, with unannounced regime
  shifts, a misspecified model is the dominant source of decision error.
* *Restricting an unreliable flexible rule paid, as Heiner argued.* On the error-prone model-based target the
  inaction-band heuristic cut downside risk in all four runs (+1728* to +2635*) without costing profit, and
  reliability-condition selection raised profit in all four (+23*, +57*, +109*, +46*) without raising downside risk.
  Restricting the already reliable price-based target hurt on both criteria in all four runs.
* *The reliability condition works as a selection rule, but not as this agent.* The point above is the strongest
  support for Heiner's claim in the laboratory, and it got stronger once every design was searched at the same
  density: under the old flat budget the six-parameter reliability-condition design was the most under-searched, and
  its profit effect on the model-based target was +5 (not significant) with worse downside risk. Yet the
  reliability-condition *agent* still ranks ninth of ten on profit in the main run (7.19), and worse than under the old
  budget, because the mid-dimensional rivals gained more from the larger search than it did. What pays is restricting
  an unreliable target; what does not pay is carrying the estimation of π, r, w, G and D as an agent's own decision
  rule, which is itself a hard inference problem that simple fixed restrictions avoid. This is consistent with
  Heiner's deeper argument that reliable behavior comes from rules rather than case-by-case assessment.
* *Inertia is mid-field, not rewarded, in this run.* Organizational ecology's crisis-driven agent sits mid-field on
  profit (5.30) and on the six-criterion aggregate (6.00; 4.83, 4.50, 7.17 in the
  replications). Under the superseded flat budget it held the best aggregate rank in the main run (3.33), so that
  earlier reading does not survive equal-density tuning.
* *Criteria matter.* Ranking by profit alone, by downside risk or by survival gives different orders: rational
  expectations has the best aggregate rank in the main run (3.50) while ranking third on profit (5.11), and five of
  ten theories are Pareto-efficient over the six criteria (cobweb, simple heuristics, optimization, rule B,
  satisficing), so claims of superiority must name the criterion.

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

Study plan `bbfaed9e70b900d6` (120 environments drawn by Latin hypercube over separate sources of uncertainty, 800
periods, oracle values from 3,200-period independent runs, background rivals tuned under tournament plan
`22393384781bed84`). A focal firm aims at the same target as the always-adjusting rule and differs only in when it
moves. Profit per period relative to always adjusting (95% CI):

| Selection rule | Model-based target | Price-based target |
|---|---|---|
| Reliability condition, oracle (true reliability) | **+57 [31, 84]** | +21 [−13, 56] |
| Reliability condition, learned with the true model | +58 [17, 107] | +46 [21, 77] |
| Reliability condition, learned (own model) | +10 [5, 15] | +10 [−20, 44] |
| Inaction-band heuristic | **+165 [110, 219]** | +92 [79, 107] |

Cost of applying the principle (oracle − learned): model-based target +48 [21, 74], of which estimation from limited
experience −1 [−49, +43] and judging with a misspecified model +49 [7, 97]; price-based target +10 [−28, 44], of
which estimation −25 [−58, +4] and model bias +36 [12, 59]. Pooling more experience (memory 0.9 to 0.999) left the
learned agent between +4 and +7 on the model-based target, far from the oracle's +57.

Boundary: on the model-based target the oracle's gain rises with the flexible rule's error-to-signal ratio K (slope
+221 per unit of ln K, p = 0.033) and breaks even at K ≈ 0.98; the inaction band's gain breaks even at K ≈ 0.98
(slope +633, p = 0.003); pooled over both targets the slope is +263 (p = 0.002). Restriction pays once the flexible
rule's error is about as large as the adjustment it should make. Among the sources of uncertainty, the oracle's gain
on the model-based target is driven by the model-updating lag (+0.27 [0.12, 0.40], standardized) and the demand
quantity range (+0.27 [0.12, 0.41]) rather than by the shift hazard (+0.10 [−0.05, 0.23]); cost volatility also
matters (+0.24 [0.04, 0.47]), so misspecification does not raise the gain more than risk does and M5 is not
supported.

> **Revision note (8 October 2026, equal search density).** The study's rivals are the tournament's tuned agents and
> its plan hash folds in `arena.code_digest()`, so the tuning change moved its plan from `11a279e507f246e2` to
> `bbfaed9e70b900d6` and it was rerun. **The earlier values require replication.** They were: oracle +119 [72, 175]
> (model) and −73 [−129, −28] (price); learned with the true model +95 and −66; learned +13 and −170; inaction band
> +216 and −137; cost of applying the principle +106 (model: estimation +24, model bias +82) and +97 (price:
> estimation −7, model bias +104); memory up to 0.999 left the learned agent between +3 and +10; boundary slope +530
> per ln K on the model target (p = 0.011), K* ≈ 0.98 (oracle) and 1.03 (band), pooled +289 (p = 0.031);
> model-updating lag +0.42, hazard +0.05, volatility +0.12; verdicts M1, M3, M4 supported, M2 and M5 not. The
> headline sign flipped on the price-based target: restriction there used to cost profit and now does not, and the
> oracle's advantage on the model-based target is about half its earlier size. M1, M3 and M4 remain supported and
> M2 and M5 remain unsupported.

## Rule choice: results under the frozen plan

*Evidence type: simulation comparison.*

> **Revision note (6 October 2026, organizational ecology).** The rules are tuned under the tournament plan, which
> changed (see above), so the study's plan changed from `0f7bb98f8f7e8e44` to `f15f62149d08e800` and it was rerun.
> Earlier results: E1 slope +0.003 (p = 0.95) not supported, E2 slope −0.087 (p = 0.009) supported, E3 −0.005
> [−0.18, 0.15], E4 +4.9 (p = 0.002) supported; at β = 4 the restricted share rose from 51% to 60% and the change
> rate fell from 57% to 42%; at β = 16 the filtered best reply fell from 32% to under 1% and the target-margin
> heuristic rose from 0.5% to 47%. E1 and E2 swapped verdicts.

Plan `d2f88699dc6e6eb6` (uncertainty levels u = 0, 0.25, 0.5, 0.75, 1, where cost volatility, perception error and
the demand-shift hazard rise together; intensities of choice β = 0, 1, 4, 16; 8 replications; 12 firms; 1,500
periods; rules tuned under tournament plan `22393384781bed84`).

| | E1 restricted share rises with uncertainty | E2 change rate falls with uncertainty | E3 selection raises the restricted share at u = 1 | E4 restricted share widens the distance from Nash |
|---|---|---|---|---|
| Result | slope −0.096 (p = 0.035) | slope +0.100 (p = 0.005) | −0.256 [−0.409, −0.118] | +2.61 per unit of share (p = 0.004) |
| Verdict | not supported | not supported | not supported | **supported** |

With moderate selection (β = 4) the restricted share *falls* from 42% at u = 0 to 26% at u = 1 and the change rate
rises from 49% to 67%. Under strong selection (β = 16) the pattern is the same (restricted share 59% to 26%, change
rate 37% to 66%): the flexible optimizer (filtered best reply) falls from 26% of firms to 12%, but what replaces it
is the target-margin heuristic, which rises from 6% to 52%. Under equal-density tuning, performance-based selection
under uncertainty moves populations toward a *simple flexible* rule rather than toward an explicit restriction, so
three of the four emergent-restriction hypotheses fail in the direction opposite to Heiner's argument. What survives
is E4: markets with more rule-governed firms still stay further from the Cournot–Nash price.

> **Revision note (8 October 2026, equal search density).** The rules are the tournament's tuned agents and the plan
> hash folds in `arena.code_digest()`, so the plan moved from `f15f62149d08e800` to `d2f88699dc6e6eb6` and the study
> was rerun. **The earlier values require replication.** They were: E1 slope +0.158 (p = 0.0004) supported, E2 slope
> −0.044 (p = 0.17) not supported, E3 +0.031 [−0.083, 0.131] not supported, E4 +2.8 (p = 0.004) supported; at β = 4
> the restricted share rose from 46% to 59% while the change rate held near 45%; at β = 16 the filtered best reply
> fell from 29% to 5%, the restricted share rose from 33% to 55% and the target-margin heuristic held between 21% and
> 44%. E1 reversed sign and E2 reversed sign: the conclusion that populations move toward restricted rules under
> uncertainty does not survive retuning, and should be treated as not established.

## Validation against data

*Evidence type: simulation comparison (field patterns, calibration recovery on synthetic subjects).*

**Field patterns** (registered tuned parameters, 12 markets per pattern, 600 periods): 5 of 6 documented patterns
reproduced. Naive price expectations produce cobweb cycles (lag-1 autocorrelation −0.80, in 100% of markets); slow
adaptive expectations dampen them (price s.d. 18.4 vs 37.4, in 100% of markets); imitate-the-best firms produce
1.18 × the Cournot–Nash output; a mixed market's mean price lies within 8.1% of the Nash price and fluctuates
1.05 × as much (excess volatility, in 58% of markets); prices lie between cost and the monopoly price (mean price
46.5, cost 42.1, monopoly price 71.1, in 100% of markets). **Not reproduced:** lumpy adjustment. Inaction-band firms
change output in 91% of periods against the always-adjusting rule's 99%, so they are barely stickier, even though
their changes are much larger when they come (mean 257.5 vs 42.1) — the "rare" half of the pattern fails.

> **Revision note (8 October 2026, equal search density).** The patterns use the tournament's tuned agents, so they
> were rerun and their source fingerprint changed from `c18e9f0d5c24e544` to `899d4e263e8e7c5e`. **The earlier
> numbers require replication.** Previously 4 of 6 patterns were reproduced: excess volatility failed (mean price
> within 9.2% of Nash, fluctuating only 0.90 × as much) and lumpy adjustment failed for the opposite reason to now
> (inaction-band firms changed output in only 34% of periods, but their changes were no larger than the
> always-adjusting rule's, 67.0 vs 69.2). Retuning restored excess volatility and moved the inaction band from
> "sticky but not lumpy" to "lumpy but not sticky".

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

**Human experiment.** Protocol 2.0 (`e39ff2f6582cc591`, `heiner_abm/experiment.py`) replaced protocol 1.0
(`01f956595e90e5ab`) on 8 October 2026, before any data were collected under either. Version 1.0 mixed the sources of
uncertainty in its low/medium/high blocks and tested only the correlation between restraint and profit. Version 2.0 is
designed to identify mechanisms:

* **Separated treatments** within participants: a baseline (B) and three blocks that each change one mechanism, cost
  volatility (V), noise in the cost estimate (N) or unannounced demand shifts (R); 30 periods each, after an 8-period
  practice block and comprehension checks (all correct within three attempts).
* **Counterbalancing and paths**: block order from a balanced 4 × 4 Williams square (order and position recorded);
  four exogenous paths rotated across conditions so every path meets every condition equally often; conditions share
  a path's random draws.
* **Randomized between participants** in permuted blocks of 16 (allocation schedule exported from the analysis page):
  a reliability-gated decision aid versus a clearly specified **unaided control** (same screens and information
  options; the recommendation is recorded but never shown), and separately timed belief elicitation on or off (to
  measure its effect on behavior).
* **Recorded** per decision: decision time, belief time, information panels opened, the aid's recommendation and
  whether it was shown, the chosen output, the feedback shown, beliefs, and every design's shadow choice.
* **Analysis** (`heiner_abm/human_analysis.py`, `heiner_abm/human_models.py`): adjustment probability, adjustment
  magnitude and relative profit per participant and block; condition contrasts with participant, path and position
  fixed effects and participant- (or session-) clustered intervals; randomized aid and elicitation effects; hurdle
  models of adjustment probability and magnitude with lapse, inertia, rounding, aid following and partially pooled
  individual parameters, compared on held-out blocks by log score, Brier score and RMSE; parameter recovery and model
  recovery with an exported confusion matrix; simulation-based power from pilot effect sizes and variance components
  using the same allocation and estimators; an exportable prospective analysis plan. The relation between adjustment
  frequency and profit is reported as **associational** (frequency is chosen, not assigned).
* **Synthetic pilots** (simulated participants generated by an assumed mechanism model in the real market) are
  labeled as such everywhere and are used only to check the design, the recovery of parameters and models, and
  planning. Parameter recovery on synthetic pilots of 24 participants (about 120 decisions each; seeds 1 and 11)
  recovers the adjustment intercept, magnitude noise, rounding and aid following consistently (correlations 0.67–0.91
  between true and estimated values); sensitivity to the reference signal is not recovered (−0.23 and 0.19), and the
  condition shifts in the model and partial adjustment vary from 0 to 0.61 between seeds. These parameters should not
  be interpreted in human data without a design shown to recover them; the confirmatory treatment effects (E1–E5)
  are therefore model-free. **No human data have been
  collected**; ethics approval and informed consent are needed first, and the plan should be deposited with an
  external registry to make it a preregistration.

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
heiner_abm/experiment.py   human experiment protocol 2.0: treatments, allocation, recording, synthetic pilots
heiner_abm/human_analysis.py, human_models.py
                           human experiment inference, power, analysis plan; hurdle models and recovery studies
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
heiner_abm/workbench/      experiment specification, environment adapters, execution, analysis, run store, CLI
heiner_abm/model_spec.py   model specification: every agent's objective, information, actions, feedback, limits
heiner_abm/claim_status.py established theory, reduced form and proposed extensions
ui/theory_page.py, ui/illustrations.py  theory page renderer and one interactive illustration per theory
ui/common.py               navigation, sidebar base scenario, presets, caching, chart helpers
ui/workbench_ui.py         workspace widgets bound to the experiment specification, result charts
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
* **Equal tuning budget.** Every policy with free hyperparameters gets the same number of candidate settings (four in
  the inventory task, three in the market), scored on the same training configurations and paths: the flexible
  forecast's gain, the band's width, the confidence-sensitive gate's confidence and minimum evidence, change
  detection's three settings and the robust order's two. Candidate 0 of each is the registered starting value, so the
  search can only help. *Retain the default* and the *estimated-gain gate* have no free hyperparameters and get none;
  the ORACLE is a bound, not a competitor. The run's tuning log (one row per policy with the number of candidates
  evaluated) is shown on the page and exported as `docs/learnability_results/tuning.csv`. Until 6 October 2026 the two
  gates received no tuning while their four comparators received four candidates each, and the plan's
  `gate_confidence` / `gate_min_evidence` never reached the gate; `LearnPlan.tune_gate = False` and
  `MarketPlan.tune_gate = False` restore that behavior.
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
* **Registration.** Plan `6fbc88a80332fc38` is frozen in the repository (`registered.LEARN_PLAN`); the exported
  document is [`docs/learnability_registration.md`](docs/learnability_registration.md) and can be downloaded from the
  page. It has **not** been preregistered with an external registry.

### Results under the frozen plan `6fbc88a80332fc38`

One run of the registered plan, 18.4 minutes (`python tools/rerun_learnability_study.py` reproduces it; every seed is
derived from the plan). Tuned on training configurations only, four candidates each: forecast gain 0.1, band b = 0.5,
confidence-sensitive gate (confidence 0.8, minimum evidence 3), BOCPD (hazard 0.005, prior sd 30, obs sd 25), robust
order (N = 20, ρ = 0.01); market band threshold 50 and market gate (0.8, 3), three candidates each. Tables:
[`docs/learnability_results/`](docs/learnability_results/), tuning log in `tuning.csv`.

| | Decision rule | Result | Verdict |
|---|---|---|---|
| H1 | slope of the gate's advantage on log10 R, CI below 0 | −0.95 [−1.50, −0.44], 30 jump configurations | supported |
| H2 | cross-validated R² gain of log10 R over volatility and noise, CI above 0 | +0.25 [−0.32, +0.85] | **not supported** |
| H3 | H1 slope in the new families (switching, drifting) | −1.13 [−1.74, −0.65], 30 configurations | supported |
| H4 | H1 slope in the market | −1722 [−2721, −703], 16 configurations | supported |
| NC1 | perfect information, costless adaptation: nothing beats always adapting | every policy − always ≤ −0.16 per period | passed |
| NC2 | dominated default: gates adapt ≥ 90%, lose ≤ 5% | adaptation 1.000 and 0.997; −31.80 and −32.48 vs −31.78 | passed |
| NC3 | stable, abundant feedback: R < 1, gate noninferior at a margin of 0.25 | R = 0.076; gate − default −0.00 [−0.00, +0.00], lower limit clears −0.25 | passed |

*Revision note (7 October 2026): the run under the superseded plan `ec9b781e125e289d` **requires replication and is
not comparable one for one with the table above.** It gave the two reliability gates no tuning budget while their four
comparators each received four candidates, and the plan's `gate_confidence` / `gate_min_evidence` never reached the
gate. Under the equal budget the gate chose confidence 0.8 and minimum evidence 3 instead of 0.9 and 5, worth +0.65
net payoff per period on the training configurations, so the earlier run understated the confidence-sensitive gate.
Its numbers were: H1 −1.44 [−2.36, −0.59] supported; H2 +0.23 [−0.46, +0.84] not supported; H3 −1.75 [−2.79, −1.01]
supported; H4 −1625 [−2922, −478] supported; NC1–NC3 passed (NC1 confidence gate − always −1.09, NC2 adaptation 0.995
and net payoff −32.85). Every verdict is unchanged; the H1 and H3 slopes are flatter and the gate's shortfall in NC1
and NC2 is smaller, which is what tuning the gate would be expected to do.*

> **Revision note (8 October 2026, five defect repairs).** Five defects were confirmed against the code and repaired;
> three of them change this study, so its plan moved from `8f43bedc7dae10bc` to `6fbc88a80332fc38` and it was rerun.
> **The earlier values require replication.**
> * *BOCPD froze on missing demands.* The change-point model advanced its run-length posterior only when a demand was
>   observed, so a gap in the data stopped regime uncertainty accumulating and the model acted as if no time had
>   passed. Calendar time and conditioning are now separate, exactly one hazard transition per period. This changes
>   change detection only where demand is unobserved: its tuning score moved from −39.288 to −39.472, its chosen
>   settings did not, and NC1 (every period observed) is unchanged at −2.21.
> * *NC3 tested the wrong tail.* It passed when the **upper** confidence limit exceeded −0.25, which fails only when
>   the gate is confidently much worse, so it established nothing. It is now a noninferiority test on the **lower**
>   limit at a prespecified margin of 0.25 per period. The control still passes, now on evidence: −0.00 [−0.00, +0.00].
> * *The market ratio counted feedback too early.* It attributed a decision's feedback to the regime the decision was
>   made in, even when the judgement window matured only after that regime had ended, so n_avail was overstated and R
>   understated. Feedback is now counted by **release** period, and decisions whose window crosses a regime boundary
>   are excluded from both the regime's mean advantage and its count. H4 moved from −1612 [−2777, −513] to
>   −1722 [−2721, −703] and stays supported; H1, H2 and H3 are unchanged (−0.95, +0.251, −1.13).
>
> Two further repairs touch no registered number: the experiment workbench never forwarded the confidence gate's
> hyperparameters to it (silent fallback to the defaults, and no tuning budget), and the NK landscape redrew one
> component even at `change_frac = 0`, so the no-change control changed the landscape.

How to read this:
* **The construct's usefulness is not established.** R ordered the gate's advantage in the inventory task, the new
  families and the market (H1, H3, H4), but it did not add cross-validated explanatory power beyond volatility and
  observation noise (H2): the interval is wide and includes zero. In the 30 test configurations log10 R correlates
  with log σ (r = 0.37) and with feedback availability (r = −0.55), so H1 and H3 alone cannot separate R from the
  factors it is built from.
* **The confidence-sensitive gate rarely paid, even when tuned.** In the inventory task the better fixed rule was
  always adapting in all 30 jump and all 30 new-family configurations; the gate's advantage was below zero with its
  interval excluding zero in 22 of 30 jump and 28 of 30 new-family configurations, and above zero in 1 of 30 jump and
  none of the new-family ones. The slope says how *much* it lost, more where R is large; it does not establish a
  region where waiting for evidence wins. Its losses are largest with scarce feedback (avail 0.2: −2.22
  [−3.96, −0.43] against always adapting).
* **The market replication is weak evidence.** Retaining the default was the better fixed rule in 15 of 16 market
  configurations, per-configuration intervals are wide (only 4 of 16 exclude zero, all below), and the slope comes
  from 16 points. It is a direction, not a precise effect.
* **Other policies (secondary, not hypotheses).** Against the better fixed rule, Bayesian change detection's
  interval was above zero in 20 of 30 jump configurations and below in 4; the robust order 9 above and 10 below; the
  estimated-gain gate 3 and 7; the inaction band 0 and 11. These are per-configuration counts with intervals, not a
  ranking, and the tuned settings come from one training family; they are not evidence that one theory is superior.
* **Secondary outcomes** (jump-family test paths, means with 95% intervals; `secondary.csv`): regret per period
  always 9.1 [8.4, 9.9], estimated-gain gate 9.1 [8.4, 9.9], confidence gate 10.2 [9.4, 11.0], BOCPD 7.4 [6.9, 7.9],
  robust 8.3 [7.8, 8.9], default 52.5 [48.1, 57.3]; the gates adapt in 81% and 69% of periods and miss 14% and 24%
  of opportunities; recovery delay after a change 13.9 periods for always adapting, 17.3 and 19.8 for the gates.
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

## Experiment workbench: one specification and one execution system

The Start, Experiment and Results pages are thin: they edit and display one **experiment specification** and call
services in `heiner_abm/workbench/`. The specialized pages keep their own implementations and remain available.

| Component | Module | Responsibility |
|---|---|---|
| Experiment specification | `spec.py` | Question (text, primary outcome, comparison, smallest effect of interest), environment and its settings, candidate generation, policies, information, design (treatment, replications, periods, seed, tuning, outcomes); JSON round trip; a digest of everything that determines results |
| Guided questions | `presets.py` | Five documented, editable presets, including one human-experiment question that is routed to Validation; Explore-mode uncertainty levels that list the settings they change |
| Validity checks | `validate.py` | Per-step errors (block a run) and warnings: a stated comparison of two non-benchmark policies, defined outcomes, enough replications, treatment levels within range, environment-specific checks, information compatibility of every market agent |
| Environment interface | `environments.py` | Adapters over the existing engines (market, inventory, learning, investment, NK): controls grouped by the uncertainty panel, candidate generators, selection and complete policies with their assumptions, researcher benchmarks, outcomes, tuning and decision traces |
| Execution service | `execution.py` | Disjoint training and test seed blocks, common random numbers across policies, scheduling in blocks with progress and cancellation, workload estimates, provenance, and background runs in a separate process |
| Analysis service | `analysis.py` | Paired effects with bootstrap intervals by replication, a plain interpretation against the smallest effect of interest, summaries, split-replication robustness |
| Run store | `store.py` | Saved and duplicated experiments, completed runs keyed by configuration, kind (preview or research) and code version (so cached results are reused only for identical configuration and code), progress files, and exports built from stored outputs |

The command line runs the same code as the interface:

```bash
python -m heiner_abm.workbench preset learnability > spec.json   # a guided question as a specification
python -m heiner_abm.workbench estimate spec.json                 # workload before running
python -m heiner_abm.workbench run spec.json --kind research --out results/
```

Runs are stored under `$HEINER_LAB_DIR` (default `~/.heiner_lab`). A preview reduces the design (at most three
replications and 300 periods) and is labeled as a preview wherever it is shown. Changing the specification after a run
marks the displayed results as outdated and names the changed settings. `tests/test_workbench.py` checks the
specification round trip, the validity rules, reproducible and paired execution, identical results from the command
line and the interface, the run store and its cache, cancellation, background runs and every workspace step.

## Research basis

`heiner_abm/literature.py` is the single source for the research behind the simulation. Each hypothesis records its
reliability-condition prediction, its alternative prediction and the rival theory it comes from, the studies that
support each side (with a note on how each bears on the hypothesis), and the contribution the test makes. The app
shows this on every hypothesis card and on the *Research & contribution* page, which also exports the bibliography
as BibTeX and APA and the evidence matrix as CSV. A test checks that every cited work exists in the registry, that
every hypothesis has research on both sides, and that the app text cites research only through the registry.
