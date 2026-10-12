# Validating the reliability gates

`heiner_abm/gate_study.py` compares the gates on outcomes. This note is about the prior question: the gates report
a predicted advantage, and the confidence-sensitive gate a lower bound on it — predictions of *what*, and how good?

Module: `heiner_abm/gate_validation.py`. Tests: `tests/test_gate_validation.py` (56 cases).
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
Plan `139c10aef80b3d60` — the hash of the committed defaults, recorded in `docs/gate_validation_results/spec.json`, which is the authority for it. Full tables in the same folder.

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

This is an **estimator-level** comparison: the same evidence replayed through different memory rules. It cannot
show how a rule changes the states the firm later visits, because the stream is held fixed. Section 4b runs the
same four rules as live policies, and the ranking does not survive.

### 4b. The same variants as live policies

Each variant now drives the gate inside the market, so its decisions feed back into the states it then faces.
`run_policy_comparison`; results in `docs/gate_validation_results/policy_variants.csv`.

| Variant | gate | net payoff | adaptation rate | evidence | coverage | false | missed | regret |
|---|---|---|---|---|---|---|---|---|
| per-update forgetting (existing) | lcb | 1371 | 0.533 | 44.7 | 0.538 | 0.245 | 0.275 | 1755 |
| calendar-time aging | lcb | 1306 | 0.443 | 24.5 | 0.578 | 0.135 | 0.295 | **1580** |
| change-detection reset | lcb | **724** | 0.230 | 6.8 | 0.505 | 0.095 | 0.560 | **4233** |
| continuing exploration (p = 0.1) | explore | **1973** | 0.139 | 5.4 | **0.921** | 0.035 | 0.650 | 4257 |
| exploration baseline | explore | 1925 | 0.157 | 5.1 | **0.862** | 0.045 | 0.610 | 3856 |

Net payoff is **paired and comparable**: every variant sees the same seeds. The state-level columns are not a
common set — each policy is scored at the states *it* visits, which is the whole point of running them live, but
it means those columns describe how well a rule is calibrated where it actually ends up, not on a shared sample.

**The estimator-level ranking does not survive.** Change-detection reset had the *lowest* one-step-ahead error of
the four as an estimator; as a policy it is the worst by a wide margin — net payoff 724 against 1371, regret 4233
against 1755. Resetting on every detected change leaves it with almost no evidence (6.8 effective items against
44.7), so it adapts in 23% of opportunities and misses **56%** of the beneficial ones. A rule can track a fixed
stream well and still starve itself of the stream when it is allowed to choose.

**Calendar-time aging survives both levels**, and is the only variant that does. It roughly halves false
adaptations (0.135 against 0.245) and has the lowest regret of the counterfactual gates, at a modest cost in net
payoff (1306 against 1371). It is the one change here worth considering on the evidence.

**The strongest result is about randomization, not memory.** The exploration-enabled gate's bound covers 86–92%
against a nominal 90%, while the counterfactual gates sit at 50–58% in the same regime. Its evidence comes from
randomized trials, which are unbiased for the advantage; the counterfactual gates' evidence is formed on a
believed demand curve and is not. This only appears at policy level, because a gate has to run live to generate
trials at all. It comes at a price: the exploration gate adapts rarely (14–16% of opportunities) and misses most
beneficial adaptations, yet still earns the highest net payoff of any variant here.

**What this does not establish.** One market, one regime, one parameterization, and no tuning of the variants
against each other — the comparison is between rules at their stated settings, not between tuned policies on an
equal budget. Nothing here is a registered finding.

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

No registered fingerprint or plan hash depends on **this module**. The policy-level variants in section 4b did
change `gates.py`, `params.py` and `engine.py`, which six findings and the learnability plan fingerprint. Every
default reproduces the previous rule exactly: that was checked first on a 28-case battery of output hashes, then
by rerunning every affected finding in full, and every number reproduced exactly. The fingerprints and
`LEARN_PLAN` were re-registered on that evidence, with a revision note in `heiner_abm/registered.py`, and
`REPLICATION_REQUIRED` stayed empty because the computation did not change.
