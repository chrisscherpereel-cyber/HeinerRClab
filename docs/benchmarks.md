# Competing-model benchmarks, and one tuning protocol

Five new comparators and a shared tuning protocol. Implemented in stages, each completed and tested before the
next, with the protocol brought forward ahead of the last stage for the reason given in §6.

Modules: `heiner_abm/bench_inventory.py`, `bench_bandit.py`, `bench_tuning.py`, `nk.py`.
Tests: `tests/test_benchmarks.py`, `tests/test_tuning_protocol.py`, `tests/test_nk.py`.

**No registered result is affected by any of it.** All seven finding fingerprints and all four plan hashes were
verified unchanged after every stage, and `REPLICATION_REQUIRED` is empty. That is not luck: the learnability
plan digests six *named objects* from `bench_inventory`, so the work was kept clear of them deliberately.

---

## A. A Bayesian demand model that estimates its own parameters

The inventory suite had a correctly specified benchmark handed the true hazard, latent demand variation σ and
measurement error τ, and a misspecified BOCPD competitor whose single "observation s.d." **conflates** demand
variation with measurement error. Nothing in between.

`bayes_est` runs the *same grid filter* as the benchmark — so the comparison isolates knowing the parameters from
having to estimate them — with σ, τ and the hazard tuned on training environments only. The filter was lifted out
of `bayes_true_orders` into `state_space_orders`; `bayes_true_orders` is now a thin wrapper that supplies
researcher parameters, which puts the distinction in the code rather than only in a role field.

**Why separating the two sources matters for the decision**, stated where it is used: the likelihood uses σ *and*
τ, because an observation carries both, but the order is about **demand**, so the predictive quantile uses σ
alone. Folding τ into the order inflates the fractile margin by measurement error the agent never pays for. A
test holds total observation noise fixed and shifts it from demand to the instrument; the margin must fall.

**Missing observations** advance the state through the transition without updating it. A period whose demand was
not observed is not evidence that demand was average, and a NaN likelihood would destroy the posterior. Tested on
no gaps, alternating gaps, a run of thirty, and a path with nothing observed — and that filling a gap with the
prior mean gives a *different* answer.

## B. Thompson sampling for the switching bandit

Two competitors, kept as separate variants: `ts_sw` (sliding window, Trovò, Paladino, Restelli & Gatti 2020) and
`ts_disc` (discounted). Both use Gaussian posterior sampling (Thompson 1933; the Gaussian version analysed by
Agrawal & Goyal 2013), playing the argmax of one independent posterior draw per arm.

- **Action space**: exactly one of the two options per period, over the same periods as every other agent.
- **Observable feedback**: `obs[t, a_t]` alone. Tested by corrupting the **unchosen** option's payoff by +1000 in
  every period and asserting the action sequence is byte-identical.

The working reward s.d. and prior s.d. are **tuned**, not the environment's noise, so the posterior is the
agent's own and not the correct one.

**The two variants are not equal in provenance.** The sliding window implements a refereed paper and cites it.
The discount device does not: the usual citation is an arXiv preprint with no confirmed refereed venue, so it is
documented as this laboratory's adaptation of Garivier & Moulines' UCB discount to the Bayesian update, and
attributed to no one. A test asserts the family says so.

## C. A change-detecting UCB, verified against its source

`cusum_ucb` implements Liu, Lee & Shroff (2018, AAAI-18) as their own equations, with `cusum_step` split out so
the recursion is checked **against the published equations** rather than inferred from a policy's actions:

```
s+ = y − u₀ − ε,  g+ = max(0, g+ + s+)
s− = u₀ − y − ε,  g− = max(0, g− + s−)
```

u₀ is the mean of the first M samples after a reset; an alarm at `g ≥ h` either side resets that arm; choice is
their CD-UCB rule (uniform exploration with probability α, else `argmax x̄ + c√(log n / N)`).

**Adaptations to this task**, on the family and asserted by a test: rewards are assumed bounded in [0,1] by the
source and are not here, so the exploration constant absorbs the scale; ε is called a *known* detectability
parameter there and is tuned here; and the reference mean is a sample mean, which is the weak point under
contamination, so `cusum_ucb_robust` substitutes a median of block means — attributed to no one.

**A heavy-tailed evaluation condition** exists so that "robust estimation where heavy-tailed conditions are
evaluated" refers to something real. It contaminates what agents *observe* and leaves the regime and the true
payoffs untouched, so agents are still scored on the truth; every family faces the same condition.

> A first attempt simply drew observation noise from a Student t. It did **not** produce a heavy tail — kurtosis
> 3.1 against the Gaussian 2.8 — because with `obs_noise` 0.5 against payoff noise 1.0 the payoff's own Gaussian
> noise dominates what is observed. Contaminating at the payoff-noise scale gives kurtosis **141**, and a test now
> asserts the heavy condition is heavy rather than assuming it.

**A result that does not flatter the new family.** At one shared untuned configuration, averaged over six
environments under contamination, robust location does **worse** than the sample mean (0.090 against 0.131). A
test records that rather than the comfortable direction. Resistance to outliers is a property of the estimator
and is tested directly; that it *pays* here is not established, and the tuned comparison is where it should be
settled, since both families have the same parameters and may want different values.

## D. Robustness beyond the observed support

Two distributionally robust models, named for **what they are robust to**:

| Family | Ambiguity set | Can it price an unobserved outcome? |
|---|---|---|
| `dro` — *empirical support only* (modified χ² ball) | reweightings of the observed window | **No.** Its order always lies inside the observed range. |
| `dro_moment` — *moment ambiguity* (Scarf rule) | every distribution with the estimated mean and s.d., any shape, any support | **Yes.** The worst case is a two-point distribution that need not sit on observed values. |

Scarf (1958), with Gallego & Moon (1993) for the simpler proof and the non-negativity treatment. The closed form
`S* = μ + (σ/2)(√(p/h) − √(h/p))` is **checked, not trusted**: a test grid-searches the worst-case objective and
asserts the closed form is its minimizer, at three mean/s.d. pairs.

The difference is tested rather than described: the empirical order always lies within `[min, max]` of its
window; the moment order on the window `[90, 110]` comes out **above 110**; and two windows with identical mean
and s.d. but different shapes give the moment model the *same* order and the empirical model *different* ones.

**Adaptation.** Scarf takes μ and σ as given; here they are estimated from the last N observations, which carry
measurement error on top of demand variation, so the estimated s.d. overstates the variation the agent is exposed
to. Rather than hand it the true measurement error, it tunes a multiplier on the estimated s.d.

> Recorded because the direction is not obvious: at this 0.8 critical fractile the distribution-free order is
> **less** than the Gaussian quantile, because the worst case it guards against is a two-point distribution rather
> than a normal tail. Distribution-free is not the same as conservative in every direction.

## E. An NK searcher that verifies before moving

The `Observer` had reserved slots named *"verification of the candidate"* and *"re-observation of the incumbent
for verification"* since it was written, and the gated searchers do take that pair — but **after** the decision,
to feed what they learn for later moves. Nothing used a verification reading to decide the move it was about.

`verify` buys `verify_reps` independent paired readings of candidate and incumbent, averages them with the first
evaluation, and moves only if the averaged advantage clears the move cost by `verify_margin`. Readings are bought
from the **same evaluation budget as search**, so verifying harder means proposing less.

Tested: every accepted move follows the verified gain; verification sometimes overturns the first reading;
measured against the researcher's true payoffs the RMSE of the decision variable falls from one pair to eight;
four times the verification yields strictly fewer proposals; the budget is never exceeded; a demanded margin
reduces moves.

**Nothing that already existed moved**, checked rather than argued: verification draws come from a *fifth* child
of the run's `SeedSequence`, children are deterministic by index, and every existing searcher over three
environments and three runs is byte-identical before and after — 21 cases, 0 changed.

## 6. One tuning protocol

Brought forward ahead of stage E, because in **three of the four preceding stages** a first-draft test asserted a
performance ordering at hand-picked untuned settings, and in two of them the ordering was false. Those are the
comparisons this protocol exists to make properly, so the last comparator should arrive into it.

**"The same budget" is a stated choice, not an assumption.**

| Rule | Meaning | Trade |
|---|---|---|
| `equal_candidates` | the same *number* of candidates for every family | equal effort; a three-parameter family searches a volume with the draws a one-parameter family spends on a line |
| `per_parameter` | `base` candidates per free parameter | equal density, unequal effort |

Neither is more correct. Both are implemented, every run records which was used, and the per-parameter density is
reported under either rule. An unnamed rule is refused rather than defaulted.

- **Documented budgets** — `BudgetRecord` carries the rule, base, parameter count, candidates, density, and the
  training, validation and test runs spent, with a total. Recorded, not inferred.
- **Boundary selections** — `boundary_report` says where each chosen value sits in the range it was offered and
  whether it is pinned at an end, measured on the scale the parameter is sampled on, so a log range is judged in
  logs. *A pinned value is a statement about the range rather than about the agent.*
- **Sensitivity** — `sensitivity_report` gives the spread of validation scores across repetitions at each budget
  and what the largest budget bought over the smallest. A family whose score swings across repetitions has been
  sampled, not tuned.

**The workbench now uses the same protocol.** What is shared is the protocol; what is deliberately *not* shared is
the search. The benchmarks sample from ranges; the workbench enumerates short hand-picked lists that span a
rule's behaviour, because four chosen values beat four random draws at this budget. Declaring them as `Choice`
spaces also makes a smaller budget a strict prefix of a larger one. An environment declares `tuning_plan()` and
`score_policy()`; the base class does the rest once.

**Verified:** the shared protocol picks *exactly* what the hand-rolled grids picked — gain 0.1, band_b 0.25, gate
(0.8, 3.0). A test reconstructs the old grids and asserts the equality. The unification changed what the
workbench tells you, not what it chooses.

### Two things the reporting found immediately

**A bug, within minutes of the reporting existing.** The confidence gate is tuned over four *pairs*; the first
version passed the pair through as a single key the adapter does not read, so every candidate scored identically
and the search looked decisive while doing nothing. It showed as a gain of **+0.000** and a pinned selection.
`TuningStep.expand` now maps paired candidates into the keys the adapter reads, with a regression test asserting
the candidates actually reach the gate.

**A finding about the laboratory's own grids.** The inaction band is chosen at the **smallest value it is
offered**, at two of three hazard levels, with a score spread of 5 to 36 payoff units across candidates. Its
candidate list is too narrow at the low end, so the reported band width is partly an artefact of the grid. It now
appears in the run manifest under `tuning.pinned`.

---

## What is not established

No ranking of these comparators is reported anywhere, here or in the app, because none has been produced under
the protocol at a research budget. The families are implemented, documented, verified against their sources where
they have one, and tested on the properties they claim; which of them *wins* is the next question, and it should
be answered by a tuned run rather than by the untuned spot checks that appear in the tests as recorded
observations.

Correctly specified models (`bayes_true`, the oracles) remain reported beside rankings and never in them, and no
agent receives a hidden true parameter: every new family's working parameters are tuned on training environments.
