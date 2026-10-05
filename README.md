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

The app has two tabs of simulations.

**General simulations**: every theory is an agent; fair comparisons in a shared market

| Page | What it does |
|---|---|
| Theory scorecard | Tests each theory's own falsifiable claim in one common battery: tournament, debiased control, single-theory industries and ecological selection, in calm, risky and Knightian environments |
| Theory tournament | All theories compete in the same market across a grid of cost-volatility and regime-shift environments; relative profit, activity vs performance, failures |
| Single-theory industries | The industry each theory produces: profit, margins, distance from Cournot–Nash, price volatility |
| Ecological selection | Exit when capital runs out; entrants' forms drawn from survivors; population shares over time |
| Multi-agent market lab | One market with any line-up of agents |

Agents (sidebar → **Agents (theories)**, each with its description, decision rule, tested prediction, literature
and parameters): neoclassical optimizer, real-options flexibility seeker, Bayesian (Kalman) learner, Heiner
reliability agent, satisficer, heuristics-and-biases agent, fast-and-frugal imitator, organizational-ecology inert
firm, and the paper's own Cournot/Bertrand rule.

**Special simulations (Heiner)**: the paper's model and Heiner-specific analyses

| Page | What it does |
|---|---|
| Paper-model market lab | One market with editable paper firms; reliability scoreboard and decision inspector |
| Paper hypotheses H1–H6 | Free flexibility, profitability switch, volatility, fixed costs, competition intensity, regimes |
| Does the RC predict performance? | Rigid-twin comparisons; out-of-sample AUC of the RC |
| Dynamic RC (Heiner 1989) | Immediate / persistence / strategic-feedback decomposition; partial-adjustment bound; signal-detection ROC |
| Risk vs Knightian uncertainty | Cost risk vs demand-regime shifts for model-based and model-free firms; punctuated adjustment |
| CD-gap explorer, Endogenous flexibility, Experiment designer | Difficulty × competence grids, evolution of φ, generic sweeps |

## Code layout

```
heiner_abm/params.py       scenario dataclasses (defaults = VBA SetUpForm)
heiner_abm/agents.py       readable agent implementation: Firm, Market, Industry
heiner_abm/engine.py       vectorised batch engine (same model, same random streams) for Monte Carlo
heiner_abm/analysis.py     reliability metrics, market statistics, regressions
heiner_abm/experiments.py  sweeps with common random numbers, rigid-twin RC validation, evolution
heiner_abm/theories.py     competing theories as agent classes + the shared multi-agent market (with selection)
heiner_abm/general.py      fair-comparison designs: tournament, single-theory industries, selection, scorecard
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
