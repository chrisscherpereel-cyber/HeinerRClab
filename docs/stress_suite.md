# A registered stress-test suite

Module: `heiner_abm/stress.py`. Tests: `tests/test_stress.py` (67). Results: `docs/stress_results/`.
Frozen plan **`7a5c48cca6ce2817`**. Reproduce: `python tools/run_stress_suite.py`.

**Nothing here is a registered finding.** This is its own frozen plan. No registered fingerprint, plan hash or
result is touched; the module is a new file that no registered code imports.

---

## The question

`bench_inventory` compares its decision benchmarks on one process: a piecewise-constant mean with Gaussian demand
around it. On that process exactly one model is correctly specified, and the ranking it produces largely restates
that fact. The more useful question is what happens when the process is not the one anybody's model assumes.

So: the same agents, the same information restrictions, the same cost definitions, on five deliberately hostile
processes — plus the original as a **control**, so the stressed numbers have a reference point rather than being
read in isolation.

## The six families

Full equations are in the module docstring and exported to `processes.csv`. In outline:

| | Family | What it breaks |
|---|---|---|
| — | `base` (control) | nothing: the process the correct model assumes |
| **A** | `reversal` | constant hazard — a long quiet spell is not evidence of stability |
| **B** | `heavy` | Gaussian outcomes — the likelihood and the sample mean are both non-robust |
| **C** | `outside_support` | empirical support — the best order is outside everything ever observed |
| **D** | `multimodal` | unimodality — the predictive mean is a demand that almost never occurs |
| **E** | `decision_dependent` | exogeneity — the agent's own orders move the process |

Two of these needed a second attempt before they tested anything, and both failures are the same failure:

> **Family C** originally truncated the history to ±2σ and added rare excursions. But the newsvendor fractile is
> 0.8, so the optimal order sat at μ + 0.84σ — *inside* the observed band — and a 1% excursion is below a 20%
> tail, so even the oracle ignored it. Every agent scored a regret near zero. The band is now ±0.5σ, which puts
> the optimal order itself outside the observed support: an empirical model cannot reach it at any confidence
> level, however much data it has.
>
> **Family B** scaled the contamination by `magnitude / sigma`. In the cells where the shock is no larger than the
> noise — half of its own factorial — the "contamination" was *smaller* than ordinary noise and the realized
> kurtosis was 0.1. The scale is now in units of σ (5.3× to 14.7× across the design), and all four corners of
> the factorial are tested for heaviness rather than one.

The lesson in both cases is the one this laboratory keeps relearning: a condition named "heavy-tailed" or
"outside support" is not heavy or outside anything until it has been measured.

### Family E and the two oracles

Decision-dependence makes the path depend on the agent, so the agents cannot be run on a precomputed path. Every
agent in this suite is therefore **online** — it is handed `y_t` only after it has ordered — which is what lets
all six families run under identical information restrictions. A test asserts the engine does exactly this, for
the decision-dependent families too, where the order and the outcome are produced in the same loop.

The `level` variant needed a correction of its own. With goodwill lost on every stockout and never recovering,
the mean demand of *every* agent — the oracle included — fell to the lower clip, and the family measured the clip
rather than the mechanism. Goodwill is now a stock that decays back (`RECOVER = 0.10`), so stocking out often
costs a persistently lower level without being absorbing.

## Regret against what

A regret needs a benchmark that is stated, not implied. Two are defined:

- **`oracle_path`** knows the state on the path the agent actually generated and orders the 0.8 quantile of the
  **true conditional law**. It knows the law of the noise, never the draw — the best *feasible* order, not
  hindsight. A test confirms it takes shortages on about 20% of periods; an oracle with none would be hindsight.
- **`oracle_free`** acts on its own trajectory. Under A–D the path is exogenous and the two coincide (tested).
  Under E they do not, and the difference is reported separately as `endogenous_gap` — it measures what the agent
  *caused*, not what it got wrong, and adding it to regret would double-count.

The conditional law differs by family, so the oracle is written per family and each one is checked empirically:
order it on a 4000-period path and realized demand should fall at or below it 80% of the time. Using `μ + zσ`
everywhere would have quietly handed the benchmark a Gaussian assumption that four of these families violate.

## Design

A full **2⁵ factorial** — outcome noise, change frequency, shock magnitude, observation precision, adjustment
cost — at two levels each, 32 cells per variant, every policy on identical cells and seeds. Factor effects are
reported as **matched contrasts**: the paired difference over the 16 cells that differ in that factor alone, not
a regression coefficient.

**Reserved variants.** Every stress family has two process variants. Tuning sees one; the other is never used for
tuning and is the one every reported number comes from. A test asserts the two sets are disjoint for all five
stress families. Seed blocks are disjoint too, and a test checks that no evaluation path can coincide with a
tuning path.

| | tuned on | reserved for evaluation |
|---|---|---|
| A | `single` | `double` |
| B | `contaminated` | `student_t` |
| C | `upper` | `both_tails` |
| D | `two_mode` | `three_mode` |
| E | `level` | `regime` |

**Budget.** Every policy with parameters draws the same number of configurations and scores each on the same
cells, paths and horizon; `budget.csv` states it. A policy with no parameters gets one configuration and the
table says so, rather than padding it with duplicate draws to make the column look equal. Tuning is pooled across
families — a policy is tuned once for the whole suite, because tuning per family would tell it which stress it
was about to face.

## Recovery

Recovery is a survival time and is estimated like one.

- **Both windows are aligned.** The post-change window must lie entirely after the change, so the earliest
  possible recovery is one full window later. The baseline window `[c−W, c)` must lie entirely in the previous
  regime.
- **The baseline must be settled.** If the agent had not recovered from the *previous* change, the level just
  before this one is "still failing", and returning to it is satisfied by never recovering at all — an agent that
  is permanently behind would post the fastest recovery times in the suite. Such an episode has no definable
  target, so it is **excluded and counted** (`unmeasurable`), not censored: censoring asserts a lower bound on a
  duration, and here there is no duration to bound.
- **Subsequent changes and the horizon right-censor.** An episode cut short by the next change, or still running
  at the end, is a right-censored observation. Averaging only the episodes that finished would report the fast
  ones and silently drop the slow, biasing every comparison towards whichever agent fails most often.
- **Kaplan–Meier**, with the censoring rate always reported beside the median. If survival never reaches 0.5 the
  median is **not estimable and is reported as NaN**, rather than as the largest duration observed — which would
  read as a fast recovery precisely when the agent never recovered.

## What is measured

Cumulative net payoff; regret against `oracle_path`; tail loss as the mean of the worst 5% of per-period net
payoffs, with the single worst period beside it; calibration as the realized service level against the nominal
0.8 fractile; and recovery as above. Adjustment costs are charged to the agent *and* to the oracle, so the
comparison is like for like.

## Results

Plan **`7a5c48cca6ce2817`** (code digest `6d77b359def5f03f`): 6 families × 32 cells × 7 policies × 4 paths ×
600 periods, 35.3 minutes, run once. Regret is against `oracle_path`, lower is better. No selected parameter sat
at the edge of its search range.

> An earlier run under plan `41646b8bbca17af2` was discarded, not corrected. Its seeds derived from Python's
> builtin `hash()` of the family name, which is randomized per process, so that run was not reproducible and its
> numbers described nothing anyone could repeat. Every figure below comes from the run after the fix. All of the
> findings survived it, which is reassuring but is not why they are reported — they are reported because this
> run is the one that can be reproduced.

### Mean regret, by family

| policy | base (control) | reversal | heavy | outside support | multimodal | decision dependent |
|---|---|---|---|---|---|---|
| bayes_true | **4.85** | **5.11** | **6.43** | 6.32 | 5.59 | **6.07** |
| saa | 7.03 | 6.88 | 7.96 | **4.57** | **3.42** | 10.44 |
| dro_emp | 7.02 | 7.11 | 7.88 | 5.06 | 4.01 | 9.89 |
| ewma | 7.97 | 7.83 | 9.81 | 6.72 | 5.17 | 9.89 |
| bayes_est | 9.08 | 9.11 | 7.67 | 7.05 | 8.90 | 11.60 |
| dro_moment | 10.51 | 10.00 | 10.53 | 6.07 | 6.07 | 17.47 |
| rule_b | 15.98 | 10.87 | 18.28 | 5.88 | 4.84 | 20.65 |

Standard errors run 0.17–1.19; `summary.csv` carries each one. On the control, `saa` and `dro_emp` are a tie, not
a ranking: paired over all 128 runs the difference is +0.004 ± 0.075.

**The control behaves.** The correctly specified model wins on `base`, which is the sanity check the control
exists to provide.

**Knowing the parameters beats having the right shape.** `bayes_true` is misspecified on all five stress families
and still finishes first on three of them. It is handed the true hazard, σ, τ and jump scale; every other model
must find them. On these processes that nuisance-parameter advantage outweighs having the wrong functional form —
worth stating plainly, because it means a "correctly specified benchmark" in a comparison like this is partly
measuring privileged information rather than modelling skill.

**Where shape is the problem, shape-free models win.** On `multimodal` the two window models take the top two
places (3.42, 4.01) and every unimodal predictive model trails. On `outside_support` the same two lead. Both are
the predicted direction.

### A prediction of mine that the data contradicts

The specification map calls `dro_moment` the model suited to family C: its worst case is a two-point distribution
that need not sit on observed values, so it can price outcomes never seen, which `dro_emp` provably cannot.
**It finished fourth, behind both models that lack the capability.**

| paired contrast on family C | difference in regret | worse in |
|---|---|---|
| dro_moment − dro_emp | **+1.007 ± 0.347** | 62% of 128 paired runs |
| dro_moment − saa | **+1.500 ± 0.349** | 66% of 128 paired runs |

The mechanism is visible in the orders. On family C `dro_moment` orders 109.6 and serves 72.0% of periods against
a nominal 80%, while `dro_emp` and `saa` order about 117 and serve 84.6%. It is not failing to price the tail; it
is ordering too little, because its tuned `sd_scale` is **0.63** — it shrinks the spread it estimates.

That value came from tuning pooled across all six families, which is deliberate: tuning per family would tell a
policy which stress it was about to face. The cost of that choice is visible here. A shrunken margin pays in the
quiet families and is punished in the one family where the margin was the whole point.

The capability claim in the specification map is unchanged and still true — `dro_emp` cannot represent an unseen
outcome at any confidence level. What is refuted is the step from capability to performance. Being able to
represent a stress does not help if the parameter governing that representation is chosen against an objective
the stress does not dominate.

### Factor effects

Matched contrasts, paired over the 16 cells that differ in one factor alone, averaged over policies:

| family | hazard | magnitude | sigma | tau | switch cost |
|---|---|---|---|---|---|
| base | **+5.06** | +4.13 | −1.21 | +1.18 | +4.60 |
| reversal | +2.80 | **+4.71** | −0.87 | +1.72 | +4.91 |
| heavy | **+6.50** | +4.67 | −1.58 | +2.62 | +5.27 |
| outside_support | +0.21 | +0.05 | +0.40 | +2.82 | **+4.94** |
| multimodal | +0.08 | +1.30 | −0.08 | +0.81 | **+5.05** |
| decision_dependent | **+9.29** | +7.04 | −1.99 | −0.46 | +5.32 |

Each family is hurt most by the factor it was built around: change frequency where the mean moves, shock
magnitude under reversal, and neither in family C, where the stress is the support rather than the level.

Two of these need a caveat rather than an interpretation.

- **Outcome noise lowers regret** in five families. This is not "noise helps". Regret is measured against an
  oracle that suffers the same noise, and a noisier environment raises the oracle's own loss faster than the
  agent's, so the gap narrows. It is a property of the comparison, not of the agents.
- **The adjustment-cost effect is largely mechanical.** The oracle's order is piecewise constant, so it is
  charged only at changes, while every estimating agent re-orders against its own noise and pays in almost every
  period. The effect is real — jitter has a price — but it is not a subtle finding, and at these levels it is
  close to the charge itself.

### Recovery

Only family A gives a clean comparison, and there it is sharp:

| policy on `reversal` | median recovery | censored |
|---|---|---|
| bayes_est, bayes_true | 24.0 | 34%, 31% |
| ewma | 24.5 | 35% |
| dro_emp | 25.0 | 32% |
| saa | 28.0 | 35% |
| dro_moment | 34.0 | 37% |
| **rule_b** | **57.0** | **58%** |

The slow default rule takes more than twice as long to recover from an unannounced reversal and fails to recover
at all in 58% of episodes against 31% for the best model.

Everywhere else the estimator mostly reports that it cannot measure, which is the point of counting what is
excluded rather than dropping it:

| family | episodes measured | excluded as unmeasurable |
|---|---|---|
| outside_support | 896 | 0% |
| reversal | 1365 | 75% |
| heavy | 1042 | 87% |
| base | 957 | **89%** |
| multimodal | 909 | 89% |
| decision_dependent | 866 | **95%** |

Where changes arrive closer together than two recovery windows, the baseline is rarely a settled level, so most
candidate episodes have no definable target. Recovery is estimable only in the slow-hazard part of the design,
and in family E — where the agent's own stockouts raise the hazard — barely at all. Had these been dropped
silently, the surviving medians would have looked like a clean comparison across all 32 cells.

Family C is the opposite failure: every policy recovers at exactly the window minimum with zero censoring. The
single transition does not lift regret far enough above the run's own scale for the 25% tolerance to register any
delay, so the metric carries no information there. It is reported as uninformative rather than as a seven-way
tie.

### What this does and does not show

Seven agents, one newsvendor payoff, one frozen plan. The ranking is a statement about these models on these
processes, not about model classes in general. `bayes_true` is a reference and not a competitor: it is given
parameters nobody else gets, which is exactly why its results are interesting and exactly why it should not be
read as "the Bayesian model wins".

## Reproducing



```bash
python tools/run_stress_suite.py
```

Seed streams are disjoint from the registered studies: process `[seed, 818181]`, outcome `[seed, 828282]`,
observation `[seed, 838383]`, mode `[seed, 848484]`, contamination `[seed, 858585]`.
