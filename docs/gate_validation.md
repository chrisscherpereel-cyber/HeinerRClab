# Validating the reliability gates

`heiner_abm/gate_study.py` compares the gates on outcomes. This note is about the prior question: the gates report
a predicted advantage, and the confidence-sensitive gate a lower bound on it — predictions of *what*, and how good?

Module: `heiner_abm/gate_validation.py`. Tests: `tests/test_gate_validation.py` (41 cases).
Run it: `python tools/run_gate_validation.py`. Results below: `docs/gate_validation_results/`
(`summary.csv`, `calibration.csv`, `variants.csv`; the per-state table `states.csv` is written locally by the tool and not committed, being large and fully regenerable).

**Proposed extension, and researcher-only.** Nothing here is visible to any agent, and no registered finding
depends on it.

---

## 1. Three quantities, not one

| | Quantity | What it is |
|---|---|---|
| 1 | **Observed-feedback advantage** `g_t` | What the gate actually learns from: its own estimated counterfactual on its **believed** demand curve with rivals held, or (for `explore`) its own realized payoff in a randomized trial. The firm's misspecification is in it. |
| 2 | **Realized advantage** `A_t^realized` | The researcher's ex-post fork on the single shock path that occurred. True demand and costs — but one draw, so it carries that path's continuation noise. This is what `gate_study` uses. |
| 3 | **Conditional expected true net advantage** `A*_t` | `E[ Σ_h γ^h (U_h(x*) − U_h(x_B)) | state_t ] − c`, under a declared continuation policy and horizon, estimated from independent continuations. **The estimand.** |

A gate predicting (1) but judged against (2) looks noisier than it is. Judged against (3), the comparison is
between a prediction and the thing the prediction is for. `decompose()` splits the gap:

```
prediction − A*  =  (prediction − realized)  +  (realized − A*)
                     misspecification + luck    continuation noise
```

## 2. The continuation

- **Policy** `rivals_hold` — the evaluated firm holds the action under test; every rival holds its period-t output.
  This is deliberately the assumption the firm's **own** counterfactual makes (`gates.py`: it "holds rivals' output
  at its observed path"), so `A*` is the expectation of the quantity the gate is trying to estimate rather than of
  a different game. It is **not** the engine's `default` continuation, in which rivals re-decide.
- **Horizon** the judgement window `W`, discounted by `discount`.
- **Draws** independent resampled cost paths per state. Both arms of a state share their draws (common random
  numbers), so the difference has far less variance than either arm.
- **Regime** `frozen`: the demand regime at `t` persists through the horizon.
- Only the raw-material cost path is resampled — it is the exogenous process the firm faces. The reported
  Monte-Carlo standard error is of `A*` itself, so a prediction is never called biased on fewer draws than the
  claim needs.

**The continuation is pinned to the engine, not merely believed.** With a one-period judgement window the engine's
own realized fork *is* this estimand, and the test asserts they agree to `atol=1e-9` (observed: identically `0.0`).
A second test asserts the `H = 1` case equals a closed-form one-period advantage bit-for-bit. If these dynamics
ever drift from `engine.py`, those tests fail.

## 3. What is not claimed

The bound is **not** asserted to have nominal coverage, and `gates.py` already listed why: overlapping judgement
windows, nonstationarity, optional stopping, and misspecification each break a condition it would need.
`coverage()` reports the **measured** share with a Wilson interval and states its target explicitly. A gap from the
nominal level is a finding, not a contradiction of a claim nobody made.

No confidence-sequence method is introduced here. If one is added later it must state its own assumptions and
target; anytime-validity under optional stopping does **not** by itself handle a moving mean, which is the binding
problem in this market.

## 4. Results

Six markets, 400 periods, 200 sampled decision states per cell, 64 continuations per state.
Plan `0abb8bb9d36d628a`. Full tables in `docs/gate_validation_results/`.

### Coverage: the 90% bound covers 45–62%

| Regime | `gain` | `lcb` |
|---|---|---|
| clean | 0.470 | 0.450 |
| overlapping | 0.510 | 0.503 |
| delayed | 0.603 | 0.574 |
| noisy | 0.548 | 0.467 |
| missing | 0.617 | 0.538 |
| correlated | 0.492 | 0.475 |

Nominal is 0.90 everywhere. **Undercoverage is not an artifact of the continuation policy**: it is just as large in
`clean`, where `A*` equals the engine's realized fork exactly. It comes from the estimator and the firm's
misspecification, not from how the truth was computed.

### Calibration degrades as soon as windows overlap

With a one-period window the predictions are nearly unbiased (`gain` +2.6, `lcb` +5.0) with slopes near 1 (0.84,
0.76). With `W = 20` the bias jumps to +105 and +179 and the slope leaves 1 entirely (−0.39, +0.46). Overlapping
judgement windows are the single largest source of damage among the regimes tested.

### Caution is not free

`lcb` does what it is designed to do — it adapts falsely less often than `gain` in every regime (e.g. overlapping
0.215 vs 0.405) — but it misses beneficial adaptations more often (0.255 vs 0.135). Priced in payoff:

| Regime | regret, `gain` | regret, `lcb` |
|---|---|---|
| clean | 9.0 | 10.6 |
| overlapping | 947 | 972 |
| delayed | 1914 | 2313 |
| noisy | 898 | 1065 |
| missing | 1073 | 1755 |
| correlated | 1148 | **1078** |

**In five of six regimes the confidence-sensitive gate's total regret is higher than the existing gate's.** Only
under `correlated` feedback does the caution pay for itself. That is a result against the proposed extension in
this market, and it is reported as such.

### Memory variants

Compared as **estimators** on the same recorded evidence streams, scored one step ahead against the next released
value in the same bin (an observable, not a counterfactual). Separate variants, never combined.

| Variant | RMSE | coverage vs next value | decided share | exploration propensity | exploration cost |
|---|---|---|---|---|---|
| per-update forgetting (existing) | 4911 | 0.549 | 0.604 | 0.079 | 633 |
| calendar-time aging | 4645 | 0.582 | 0.573 | 0.085 | 812 |
| change-detection reset | **4456** | 0.572 | 0.425 | 0.115 | 2192 |
| continuing exploration (p = 0.1) | 4911 | 0.549 | 0.604 | 0.140 | 1242 |

Calendar-time aging improves on the existing rule on every axis shown at a small extra exploration cost.
Change-detection reset has the lowest error but classifies far less often and explores three times as much.
Continuing exploration leaves the estimates untouched by construction — it changes only the propensity and what
that propensity costs, both logged.

**Limitation, stated plainly.** This is an estimator-level comparison. It does not capture how a different memory
rule would change the states the firm later visits; a policy-level comparison would require changing `gates.py`
and `engine.py`, which are fingerprinted by four registered findings and would mark them for replication.

## 5. Leakage

Continuations are researcher-only and cannot reach an agent:

- they draw from `CONTINUATION_SEED_BASE = 11_000_000`, above `gate_study.ORACLE_SEED_BASE` and far above any
  evaluation seed;
- they run in this module, never through an agent's feedback queue;
- every function is pure in its inputs — it copies what it reads and returns new frames;
- researcher-only columns carry the `researcher_` prefix, and the one researcher quantity a scenario can hold (the
  ORACLE table) is rejected by `Scenario.validate` for every gate but the benchmark.

The decisive test: **run the market, run a full validation, run the same market again, and assert every array is
bit-identical.** If a continuation could touch an agent, that test fails.

## 6. Reproducing

```bash
python tools/run_gate_validation.py                       # defaults: 6 markets, 200 states, 64 continuations
python tools/run_gate_validation.py --draws 128 --states 400
```

`experiment_spec()` writes `spec.json`: the frozen plan and its hash, the estimands, the continuation policies, the
regime definitions, the variant notes, the scenario, and the three disjoint seed blocks. The plan is hashed, so a
reported number identifies the run behind it.

No registered fingerprint or plan hash depends on this module; verified after the change, all seven finding
fingerprints and all four plan hashes are unchanged and `REPLICATION_REQUIRED` is empty.
