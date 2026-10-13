# Censored demand and endogenous information acquisition

Modules: `heiner_abm/information_contract.py`, `heiner_abm/information_env.py`.
Tests: `tests/test_information_env.py` (36). Results: `docs/information_results/`. Frozen plan **`bc23bb881461aa28`**.
Reproduce: `python tools/run_information_experiment.py`.

**Nothing here is a registered finding.** These are two new information environments with their own frozen plan.
No registered fingerprint, plan hash or result was touched; the modules are new files that no registered code
imports, and the suite confirms zero drift.

---

## Why a contract, not a convention

Every study in this laboratory states which quantities an agent may see. Until now that was kept true by
convention and by review, which is adequate when every policy sees the same thing.

Censoring breaks that. The point of a censored experiment is that total demand is **unobservable**: the firm sees
what it sold, and when the shelf emptied it does not see how many more customers it turned away. A policy written
for full feedback will reach for `demand` — and if `demand` is still sitting in the dictionary it was handed, it
will get it, keep its privileged access, and the experiment will measure nothing while looking fine. That failure
is silent, which is exactly what makes it worth machinery.

`InfoContract` names, for one policy, the fields it may read. `InfoView` hands a policy its period record
**through** the contract. Three properties make the enforcement real rather than decorative:

1. **Indexing a field the contract does not grant raises `InformationViolation`**, naming the contract, the field
   and what it *would* have allowed — because the usual cause is a policy written against a richer environment.
2. **Attribute access, iteration, membership and `dict()` conversion give the same answer.** A refusal that only
   covers `view["demand"]` and leaves `view.demand` open is cosmetic.
3. **The view carries only the granted fields.** Hiding an attribute is not enforceable in Python — a determined
   policy can reach any slot by name — so the view holds a filtered copy and there is nothing privileged inside
   it to reach. The refusals make the *mistake* loud; the filtering makes the *circumvention* empty.

`RESEARCHER_ONLY` lists fields no contract may grant at all, whatever it asks for: `demand`, `mu`, `lost_sales`,
`regime`, `true_gain`, `oracle_order`. A contract requesting one fails to construct. The researcher keeps all of
them in the path record for evaluation; they are simply unreachable from the agent side.

The view also records which granted fields were **actually read**, so an experiment can report what a policy used
rather than what it was entitled to use.

### The test the requirement names

`test_a_full_feedback_policy_cannot_keep_privileged_access_in_a_censored_run` substitutes a learner whose
`observe` does what a full-feedback policy does — reads `view["demand"]` — and asserts the censored run raises
rather than quietly serving it.

## Environment 1: censored demand

Each period the agent orders `S`, demand `d` arrives, and it observes `sales = min(d, S)` and whether it stocked
out. `unmet_access ∈ [0, 1]` is the probability that a stocked-out period's unmet demand is **recorded** and made
observable — a configurable spectrum from a till that counts only sales to a system that logs turned-away
customers. The researcher retains `d`, `mu`, `lost_sales` and the oracle order throughout, for evaluation only.

Demand paths come from the registered `heiner_abm.learnability` configuration, so the environment is the one the
rest of the laboratory already uses, not a new one invented here.

**The censoring correction.** A stocked-out period is a *lower bound* on demand, not a reading of it. Treating
sales as demand there biases the estimate down, which makes the next stockout likelier, which biases it down
further — the classic death spiral of the censored newsvendor. Where unmet demand is not recorded, `CensoredLearner`
updates towards `E[d | d ≥ S]` under its own current belief (inverse Mills ratio), which is the simplest
correction that is not simply wrong. `correct_censoring=False` preserves the naive behaviour for comparison, and
a test pins the gap.

## Environment 2: endogenous information acquisition

Four instruments, each with an explicit price, precision, delay, and effect on what becomes available:

| Instrument | Price | s.d. | Delay | Reveals |
|---|---|---|---|---|
| none | 0.0 | — | — | — |
| forecast | 1.0 | 12.0 | 0 | the mean |
| audit | 6.0 | 3.0 | 0 | the mean |
| pilot | 3.0 | 1.0 | 3 | demand |

The pilot is the interesting one: it is the most precise instrument and the cheapest precise one, and it is
**useless for the decision it was bought for**, because it arrives three periods late. A test asserts it delivers
no same-period signal and still arrives later.

Only a contract with `may_acquire=True` may buy. Handing an acquisition rule to a policy whose contract forbids
buying raises, rather than silently ignoring the rule — a policy that may not choose must not be measured as
though it could. Acquisition rules see only the agent's own belief: `buy_when_uncertain` triggers on the agent's
**believed** standard deviation, a quantity it has, never on the truth.

## Separating the value of information from the value of choosing

A buyer that beats a no-information agent might be doing neither thing well — it might simply have been handed a
better environment. The design includes an **exogenous free-information control**: the same instrument, handed
over free at the rate the buyer actually bought it, with no choice and no charge. Three arms on one demand path:

```
value of the information   =  free control        −  no information
value of choosing when     =  buyer (before cost) −  free control
price paid                 =  the buyer's information cost
net of buying              =  value of information + value of choosing − price paid
```

The second line is the part the control exists for. `net_before_info` is reported separately from `net_payoff` so
that timing skill is not conflated with the price paid; the decomposition is exactly additive and a test asserts
it (residual ≤ 1e-9; observed ~6e-15).

### Adaptation has to be a decision

An early version found adjustment costs had no effect whatever. The reason was not subtle: the learner re-ordered
every single period, so there was no adaptation *decision*, the adjustment charge was a constant, and it cancelled
out of every comparison. An inaction `band` fixes this — the agent re-orders only when its target moves beyond a
fraction of its believed spread. `band=0.0` is the previous behaviour and remains the default for `simulate`; the
plan uses 0.35. A test pins both ends.

## The value-of-information benchmark

A benchmark needs a case where the answer is known.

**Exact (`voi_exact_two_point`).** Two-point demand with one normal signal. The prior-optimal order, the
perfect-information value and the value of the imperfect signal are all closed form: the posterior on the high
demand is monotone in the signal, so the decision rule is a threshold, and the value integrates through the normal
c.d.f. Two limits must hold and are tested: a perfect signal is worth exactly EVPI, and an infinitely noisy one is
worth zero.

**Approximation (`voi_monte_carlo`).** The same quantity by simulation, using the estimator that must be used once
demand is not two points and the belief is not conjugate. Checked against the exact answer at signal s.d. ∈
{0, 5, 10, 25, 40, 100}; it agrees within the paired standard error at every precision.

> A correction worth recording: the two terms of the estimator share demand draws, so the standard error of the
> *signal term alone* is the wrong yardstick and made the comparison look inconsistent by −0.072. The reported
> quantity is the s.e. of the **paired difference**.

**Approximation (`voi_in_environment`).** In the full environment there is no exact answer: the belief is not
conjugate, the regime moves, and the agent's own orders determine what it observes. What is reported is the
free-information control's advantage, which is the value of the information with the cost and the choosing
removed. It is returned with `exact: False` and a note saying explicitly that it is **not an EVSI**, and a test
asserts that label is present.

## Exports

Four frames, kept apart deliberately — `cells`, `decisions`, `availability`, `truth`. Researcher-only ground
truth lives in its own file with a `researcher_` prefix on every column, so it cannot be mistaken for agent
information or joined in by accident. A test asserts the prefix appears in `truth` and in none of the others.

## Results

Plan **`bc23bb881461aa28`**: 27 cells (3 access levels x 3 adjustment costs x 3 hazards) x 12 paths x 600
periods, three arms each, run once. Decomposition residual 1.5e-14. All intervals below are paired standard
errors over the cells pooled.

### Buying the audit usually does not pay

| hazard | value of information | value of choosing | price paid | net of buying |
|---|---|---|---|---|
| 0.002 | +0.082 +- 0.035 | +0.229 +- 0.032 | 0.523 | **-0.213 +- 0.036** |
| 0.010 | +0.303 +- 0.072 | +0.182 +- 0.039 | 0.530 | -0.045 +- 0.054 |
| 0.050 | +0.364 +- 0.058 | +0.871 +- 0.072 | 0.587 | **+0.648 +- 0.062** |

Buying pays in **12 of 27 cells**; overall +0.130 +- 0.036. In a persistent environment the audit is a clear
loss: the information is worth +0.08 and costs 0.52. The agent can learn the level from its own sales when the
level holds still, and paying for a noisy reading of it is money spent on something it was going to find out.

**The value of the information rises with the change hazard** (+0.082 to +0.364), which is the expected
direction: a signal is worth more when the thing it is about keeps moving.

### A sign split that did not replicate

A four-cell probe run during development suggested that the value of *choosing when* to buy falls as the
environment changes faster, opposite to the value of the information itself. **It did not reproduce.** On the
full design the value of choosing is +0.229, +0.182, +0.871 across the three hazards — not monotone, and
dominated by the fastest-change cell. The probe had 4 paths against the standard errors above. Recorded here
because it was the more interesting result and it is not supported.

### Bought information substitutes for recorded unmet demand

| unmet access | value of information | value of choosing |
|---|---|---|
| 0.0 | **+0.359 +- 0.083** | +0.259 +- 0.071 |
| 0.5 | +0.238 +- 0.041 | +0.500 +- 0.049 |
| 1.0 | **+0.152 +- 0.037** | +0.523 +- 0.051 |

The more of its own unmet demand the firm already records, the less a bought signal adds. The two are
substitutes, which is what a signal about the mean should be when censoring is the thing that was hiding it.

The second column moves the other way: when base information is richer, what is left to gain is in the *timing*
of acquisition rather than in the information. Three cells have a point estimate at or below zero, all at
`unmet_access = 1.0` and the slowest hazard, and none is distinguishable from zero (the lowest is
-0.063 +- 0.097).

### The adjustment-cost interaction is not established

| adjustment cost | value of information | value of choosing |
|---|---|---|
| 0.0 | +0.268 +- 0.057 | +0.425 +- 0.058 |
| 2.0 | +0.257 +- 0.058 | +0.427 +- 0.059 |
| 8.0 | +0.223 +- 0.060 | +0.430 +- 0.060 |

The value of information declines as adapting gets more expensive, which is the sign theory would predict — if
you act on a signal less often, it is worth less. But the decline is 0.045 against standard errors near 0.06.
**This design cannot distinguish it from zero**, and the value of choosing is flat to three decimals. Reported
as a direction, not a finding. Separating it would need more paths or a wider cost range; the inaction band is
0.35 here, and the interaction should grow with it.

### What this does and does not show

The environments and the benchmark are the deliverable; these numbers characterize them. One instrument
(`audit`), one acquisition rule, one learner and one frozen plan are not a general claim about the value of
information. The exact-case benchmark is exact; everything measured in the environment is an approximation and
is labelled one.

## Reproducing

```bash
python tools/run_information_experiment.py
```

Seeds are drawn from streams disjoint from the registered studies: record access `[seed, 515151]`, signal noise
`[seed, 626262]`, free-information draws `[seed, 737373]`.
