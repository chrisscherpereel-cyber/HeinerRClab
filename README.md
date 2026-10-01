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
| Agent tournament | Every rival theory implemented as a decision rule competing in the same market: reliability-condition agent, filtered best reply, inaction band, adaptive expectations, win-stay/lose-shift, aspiration-level search, reinforcement learner, imitator and rule B. Equal tuning budget on training environments, held-out test environments, a frozen hashed plan with four pre-registered hypotheses, head-to-head profit, invasion tests, global sensitivity analysis and replication across seeds |
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

Registered plan `cc1a20c7ceb0484f` (24 training and 40 × 2 held-out environments, 800 periods, tuning budget 24 per
agent × 2 rounds, invasion with 4 residents + 1 mutant). Mean rank among nine agents (1 = best) in the main run and
three replications with fresh seeds:

| Agent (theory) | Main | Rep 1 | Rep 2 | Rep 3 |
|---|---|---|---|---|
| Adaptive expectations (cobweb theory) | 1.72 | 2.12 | 2.10 | 2.33 |
| Imitate the best (evolutionary selection) | 3.56 | 3.08 | 3.82 | 3.54 |
| Rule B (rigid benchmark) | 3.96 | 5.44 | 4.86 | 5.54 |
| Win-stay, lose-shift (simple heuristics) | 4.05 | 4.72 | 4.78 | 5.39 |
| Reinforcement learner | 4.49 | 6.89 | 5.30 | 5.21 |
| Filtered best reply (optimisation) | 6.06 | 5.08 | 4.82 | 5.62 |
| Reliability-condition agent (Heiner) | 6.24 | 4.76 | 5.72 | 4.45 |
| Inaction band (real options) | 7.00 | 4.90 | 5.68 | 4.85 |
| Aspiration-level search (satisficing) | 7.91 | 8.01 | 7.91 | 8.07 |

None of the four pre-registered hypotheses was supported in the main run (PR2, a growing advantage over the
optimiser with difficulty, was supported in one of three replications). The reliability-condition agent did not
differ significantly from the optimiser or the inaction band, which chase the same target, and was beaten by the
rigid rule B. No population was uninvadable. The robust results are that model-free adaptive expectations wins in
every run, especially when demand regimes shift, and that aspiration-level search comes last. Model-based agents
(optimiser, inaction band, reliability-condition agent) lose most where their demand model goes out of date.

The plan was revised once during development, before it was frozen: in the first version the reliability-condition
agent built its candidate from an unfiltered best reply, while the optimiser used filters, which handicapped it. It
now shares the optimiser's filtered target and differs only in when it deviates. That version also ranked in the
lower half.

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
