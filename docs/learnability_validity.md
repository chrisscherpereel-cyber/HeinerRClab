# Incremental construct validity of the learnability index

Module: `heiner_abm/learnability_validity.py`. Tests: `tests/test_learnability_validity.py` (27).
Results: `docs/learnability_validity_results/`. Frozen plan **`1164c1fedc632908`**.

**Nothing here is a registered finding.** This is a separate frozen plan with its own confirmatory arm, analysing
an index the registered study proposes. The registered learnability results are unchanged.

---

## The question

`heiner_abm.learnability` proposes an index R and reports that a gate's advantage falls as R rises (H1), and that
log10 R adds cross-validated R² beyond log σ and log(1 + τ) (H2). Two covariates is a low bar. A negative slope
on R is consistent with R naming a distinct mechanism, and equally consistent with R being a convenient summary
of quantities already known to matter. This separates those.

## What was prespecified, and when

The index is **imported unchanged** and is never redefined here. The plan — covariates, interactions, both
held-out protocols, the matched designs, the robustness probes, the resampling scheme — was frozen and hashed
before the confirmatory configurations were run. The plan hash includes the index definition, so the index cannot
be altered without changing the hash, and a test asserts that.

Exploratory and confirmatory data are separate by construction: seed blocks 5 and 6, disjoint from each other and
from the study's blocks 1–4 (tested), with different configuration draws. Everything was built and debugged on
the exploratory arm. **The confirmatory arm was run once.**

Two choices were made on the exploratory arm and fixed before that run: standardizing covariates within each
fold, which turned a meaningless comparison model (CV R² ≈ −1.2) into a real one, and the ridge penalty, chosen
on the **comparison model's own fit** — never on the contrast with the index, since selecting the penalty on the
quantity under test would bias it.

## The comparison model

Ten covariates, each computable without the index — advantage-to-noise, regime duration, feedback availability,
feedback delay, learner memory, effective evidence, default quality, and the three adjustment charges — plus
three prespecified interactions: advantage-to-noise × regime duration, availability × memory, departure cost ×
advantage-to-noise.

Three of these could not be expressed before: the task had no observation delay, and no way to make feedback
biased or correlated. `Config.obs_delay`, `obs_bias` and `obs_corr` were added, all inert at zero, verified
bit-identical on an 18-case battery and then by a full rerun of the registered study (see
`heiner_abm/registered.py`, revision note of 12 October 2026).

## The confirmatory result

180 configurations, 60 per process family, 8 pilot and 8 evaluation paths each, 1000 outer resamples, 16 inner.

| | Test | Δ cross-validated R² | Supported |
|---|---|---|---|
| **V1** | held-out configurations | **+0.0027** [−0.0062, +0.0258] | No |
| **V2** | held-out process families | **−0.0349** [−0.2635, +0.0433] | No |
| **V3** | with pilot uncertainty propagated | **+0.0032** [−0.0064, +0.0254] | No |
| **D1** | matched on simple uncertainty, differing index | +0.382 [−0.400, +1.236] | No |
| **D2** | matched on index, differing process | 1.627 against an outcome s.d. of 1.712 | sufficiency not supported |

The comparison model reaches **CV R² = +0.399** on held-out configurations. Adding the index moves it to
**+0.4015**. On held-out families it makes transfer *worse*, from 0.076 to 0.041.

### The slope is real, and the index is redundant

The null is not an artefact of a flat or broken index. Across the 180 confirmatory configurations the index
varies properly (range −1.98 to +0.59, s.d. 0.50, all finite), correlates with the outcome at **r = −0.369**, and
regressing the advantage on it alone gives a slope of **−1.276 per decade**.

So the registered study's negative slope **replicates cleanly on independent confirmatory data** — and the index
still adds nothing once the comparison model is present. Both statements are true together, and together they are
the finding:

> The negative slope is not evidence of a distinct mechanism. It is the sign the construct was built to have.
> Everything the index predicts about this gate's advantage is already carried by quantities a reader can compute
> without it.

The strongest single correlate of the advantage is **learner memory** (r = +0.386) — stronger than the index —
and the index is defined without reference to learner memory at all.

### D2: not a sufficient summary

Configurations matched on the index but drawn from different process families differ in outcome by **95% of the
overall spread**. Whatever the index summarizes, it is not enough to say how well a gate will do without also
knowing which process generated the data. Reported as evidence against *sufficiency*, which is a different claim
from *usefulness*, and never as a pass or fail.

## What the robustness probes show

- **Correlated feedback** — the index *falls* as the observation noise becomes correlated, i.e. it says learning
  gets easier, when correlated observations carry less information per observation. The index counts
  observations without discounting for dependence.
- **Learner memory** — performance moves substantially across λ while the index barely moves. The index cannot
  see memory by construction.
- **Biased observations** — the index is built from the spread of observed gains, not their location, so a bias
  is invisible to it. Stated in the module rather than discovered.
- **Clipping** — on a near-zero-advantage configuration the index moves from 1.100 to 0.959 when the upper clip
  tightens; on a healthy configuration it does not move at all. In exactly the regime the index was designed for,
  part of what it reports is where the clip was put.

## What this does and does not establish

**Does:** on this task, with this gate, the proposed index adds no measurable predictive accuracy beyond a
prespecified comparison model, does not transfer to unseen process families, and is not a sufficient summary
across processes.

**Does not:** refute H1. The slope is real and replicates. It also does not establish that *no* index of this
kind could add validity — only that this one, measured this way, does not here.

**Allowed conclusion the design preserves:** had V1 been supported, it would have shown the index is a *useful
summary index*, not that it names a distinct mechanism. The verdict text says so, and a test asserts the word
"mechanism" never appears in that reading.

## How the test was checked before being trusted

A prediction contest is worthless if it cannot tell a good covariate from a useless one, so both controls are
tested: a column of pure noise must not buy a supported verdict (Δ < 0.05), and a genuinely informative column
must be detected (Δ > 0.2). The hierarchical resampler is tested to *widen* the interval rather than narrow it,
since propagating estimation uncertainty cannot make an answer more certain.

## Reproducing

```bash
python -c "import heiner_abm.learnability_validity as V; print(V.ValidityPlan().digest)"   # 1164c1fedc632908
```

`run_validity(plan, hp, "exploratory")` may be run freely. `run_validity(plan, hp, "confirmatory")` is intended to
be run once; it was, and `docs/learnability_validity_results/confirmatory.json` records the plan hash, the index
definition, the covariates, the interactions, the claims and their decision rules alongside the numbers.
