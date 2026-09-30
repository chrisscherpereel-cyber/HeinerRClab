# Heiner_RC_Lab: agent-based tests of Heiner's reliability condition

A Streamlit application that turns *The influence of mistakes in Cournot and Bertrand competition: a test of
Heiner's reliability condition* (Scherpereel & Summers) and its 2006 VBA workbook (`HeinerIOExp.xls`) into an
agent-based simulation laboratory.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Tests (agent/engine equivalence, VBA parity checks, RC accounting identity, headless smoke test of every page):

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
| Heiner's reliability condition | The theory (CD-gap, r, w, π, G, D, tolerance limit), an interactive RC calculator, and the competing hypotheses |
| Market lab | One market with editable heterogeneous firm agents. Shows price/cost dynamics, a per-firm reliability scoreboard, and a period-by-period decision inspector (correct deviations, type I and type II errors) |
| Hypothesis tests | H1 free flexibility · H2 profitability switch · H3 volatility · H4 fixed costs · H5 competition intensity · H6 regimes and equilibrium. Each shows the RC prediction next to the traditional prediction and gives a verdict |
| Does the RC predict performance? | Random environments; each firm is compared with its own rigid twin (same shocks). The RC is estimated in the first part of each run and predicts the second part (**out-of-sample**), scored by AUC with environment-clustered bootstrap CIs |
| Dynamic RC (Heiner 1989) | Decomposes each decision's value into immediate, persistence and strategic-feedback parts. Tests Heiner's (1989) partial-adjustment bound β₀ = 1/((1+K)(1−f′)) against the profit-maximising flexibility. Signal-detection ROC of each firm's decisions |
| Risk vs Knightian uncertainty | Cost-volatility risk versus unannounced demand-regime shifts at matched unpredictability, for model-based (Cournot) and model-free (Bertrand) firms. Event study of punctuated slow–quick–slow adjustment |
| CD-gap explorer | Difficulty (Δ or noise) × competence (foresight κ) heatmaps of r, w, π, RC margin and the payoff to flexibility |
| Endogenous flexibility | Firms imitate the most profitable rival's φ (plus mutation). Does volatility breed rigidity? |
| Experiment designer | Generic one- or two-parameter sweeps with CSV export |
| Model & methods | Equations, schedule, measurement, statistics, parity table against the VBA code |

## Code layout

```
heiner_abm/params.py       scenario dataclasses (defaults = VBA SetUpForm)
heiner_abm/agents.py       readable agent implementation: Firm, Market, Industry
heiner_abm/engine.py       vectorised batch engine (same model, same random streams) for Monte Carlo
heiner_abm/analysis.py     reliability metrics, market statistics, regressions
heiner_abm/experiments.py  sweeps with common random numbers, rigid-twin RC validation, evolution
ui/common.py               sidebar base scenario, presets, caching, chart helpers
app_pages/*.py             the Streamlit pages
```

## Modelling notes

* The market, cost process, Cournot/Bertrand rules, selection rules, rounding and profit accounting match the VBA
  exactly (see the parity table in the app).
* **Extensions beyond the VBA:**
  * per-firm reliability measurement (the paper's planned extension);
  * competence (cost foresight κ, perception noise σ), which makes the CD-gap explicit and measurable;
  * an Adaptive selection rule that learns when deviating pays;
  * evolution of flexibility;
  * 2–12 firms;
  * out-of-sample RC tests (estimation and evaluation windows);
  * three measures of decision value (one-period, persistence, full strategic feedback) with discounting;
  * Heiner's (1989) error-to-signal ratio K and partial-adjustment bound;
  * unannounced demand-regime shifts with a model-updating lag (structural / Knightian uncertainty).
* **Counterfactual horizon H.** The paper proposed judging each deviation from rule B by one period of profit.
  Because production changes persist and rivals react, that measure says "deviate" almost always and does not
  predict flexible-vs-rigid performance. The model therefore forks the market at each decision and compares the
  two branches over H periods. By default the firm returns to rule B afterwards, which is Heiner's "deviate at this
  instance, otherwise follow B". H = 1 reproduces the paper's measure. The default is H = 20.
