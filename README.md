# Heiner_RC_Lab: agent-based tests of Heiner's reliability condition

A Streamlit agent-based simulation laboratory that tests Heiner's (1983, 1989) reliability condition in a cobweb
oligopoly and confronts it with rival theories of flexibility under uncertainty: neoclassical optimisation, real
options, cobweb stability, bias–variance / ecological rationality, satisficing and structural inertia. Every
hypothesis, and every alternative to it, is grounded in published research (see *Research basis* below).

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
| Heiner's reliability condition | The theory (CD-gap, r, w, π, G, D, tolerance limit), an interactive RC calculator, every hypothesis with its alternative and research, and the contribution to the literature |
| Market lab | One market with editable heterogeneous firm agents. Shows price/cost dynamics, a per-firm reliability scoreboard, and a period-by-period decision inspector (correct deviations, type I and type II errors) |
| Hypothesis tests | H1 free flexibility · H2 profitability switch · H3 volatility · H4 fixed costs · H5 competition intensity · H6 regimes and equilibrium · H7 competence · H8 perception noise · H9 selection rules · H10 predictable behaviour · H11 number of rivals · H12 model-updating lag. Each shows the RC prediction next to the alternative, the research behind both, the contribution, and a verdict |
| Does the RC predict performance? | Random environments; each firm is compared with its own rigid twin (same shocks). The RC is estimated in the first part of each run and predicts the second part (**out-of-sample**), scored by AUC with environment-clustered bootstrap CIs |
| Dynamic RC (Heiner 1989) | Decomposes each decision's value into immediate, persistence and strategic-feedback parts. Tests Heiner's (1989) partial-adjustment bound β₀ = 1/((1+K)(1−f′)) against the profit-maximising flexibility. Signal-detection ROC of each firm's decisions |
| Risk vs Knightian uncertainty | Cost-volatility risk versus unannounced demand-regime shifts at matched unpredictability, for model-based (Cournot) and model-free (Bertrand) firms. Event study of punctuated slow–quick–slow adjustment |
| CD-gap explorer | Difficulty (Δ or noise) × competence (foresight κ) heatmaps of r, w, π, RC margin and the payoff to flexibility |
| Endogenous flexibility | Firms imitate the most profitable rival's φ (plus mutation). Does volatility breed rigidity? |
| Competing theories | Heiner's RC against neoclassical optimisation, real options, cobweb stability, bias–variance / ecological rationality, satisficing and structural inertia. A tournament of nine discriminating experiments scores each theory's directional predictions; an out-of-sample horse race scores each theory's forecast of which firms benefit from flexibility, plus an encompassing test of whether the RC adds information beyond all rivals |
| Agent tournament | Every rival theory implemented as two agent designs competing in the same market (17 designs, including target × selection-rule composites). Equal tuning budget per design on training environments, design selection on training data, held-out test environments, a frozen hashed plan with six pre-registered hypotheses, six performance criteria (profit, downside risk, survival, volatility, regret, worst case), a selection-rule experiment, invasion tests, global sensitivity analysis and replication across seeds |
| Experiment designer | Generic one- or two-parameter sweeps with CSV export |
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

Registered plan `2e7bc47b726b5ba3`: two designs per theory, each tuned with a budget of 24 settings × 2 rounds on 24
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

The plan was revised during development, before it was frozen: a first version represented each theory by one
design (and, earlier still, the reliability-condition agent used an unfiltered target). Both earlier versions also
placed the reliability-condition agent in the lower half.

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
