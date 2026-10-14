# Replication on independent seeds: which claims survive

Module: `heiner_abm/adjudication.py`. Tests: `tests/test_adjudication.py`.
Results: `docs/adjudication_results/`. Frozen plan **`eedc8c4ac843a18a`** (code `0328bfad98dfa197`).
Reproduce: `python tools/run_adjudication.py`.

Seventeen claims, six studies, every seed moved by a fixed offset and nothing else changed. 89 minutes.

| outcome | n | claims |
|---|---|---|
| rejection confirmed | 9 | PR1, PR3, PR4, PR5, PR6, V1, V2, G1, S3 |
| **retained** | 5 | H1, S1, S2, S4, I2 |
| **rejected** | 2 | **H2, I1** |
| **overturned** | 1 | **PR2** |

---

## What makes this a test rather than a re-run

Re-running a frozen plan on its own seeds reproduces its numbers exactly. That is reproduction, and it contains
no information the first run did not, so it cannot move a claim. Every plan here moves its seed by **+770,000**
and changes nothing else: the design, the estimator, the decision rule and the analysis are untouched.

**No new decision rules were written.** Each study already carries its own prespecified rule — the
`decision_rule` column of its verdict table, fixed before its first run. Those were re-applied unchanged. The
only rule frozen *here* maps a pair of verdicts onto an outcome, and it was hashed before anything was run:

> retained (supported twice, at ≥ 50% of the original magnitude) · narrowed (supported twice but weaker) ·
> rejected (supported, then not) · rejection confirmed (not supported twice) · overturned (not supported, then
> supported)

## The three claims that moved

### H2 — rejected

> *log₁₀ R explains the gate's advantage beyond outcome volatility and observation noise.*

Supported originally; not supported on independent seeds (gain +0.053). Read with V1 and V2 — which confirm,
again, that the index adds nothing beyond a prespecified ten-covariate comparison model — the learnability
construct narrows to something precise:

| test | question | outcome |
|---|---|---|
| H1 | does the gate's advantage fall as R rises? | **retained** (slope −0.72 against −0.95) |
| H2 | does R add accuracy beyond volatility and noise? | **rejected** |
| V1 | does R add accuracy beyond the full comparison model? | rejection confirmed (+0.0094 [−0.0185, +0.0746]) |
| V2 | does R transfer to unseen process families? | rejection confirmed (−0.0584 [−0.3394, +0.0687]) |

The slope is real and replicates. The *incremental* content does not, by two independent routes. R is a useful
summary of quantities already known to matter, and is not evidence of a distinct mechanism. That was the
conclusion the construct-validity study reached on one arm; it now has a second, independent confirmation, and
H2 — which the original study had recorded as supported — does not survive.

### I1 — rejected, and the original was never significant

> *Bought information and recorded unmet demand are substitutes.*

The original found the value of information falling monotonically with access (0.359 → 0.238 → 0.152, gap
+0.206). The replication found it **rising** monotonically (0.045 → 0.160 → 0.239, gap −0.194).

Two tight, opposite results cannot both be sampling noise, so the flip was worth chasing. It is not the
acquisition threshold, which is near-identical across runs (22.96 against 23.81, purchase rates 0.091 and
0.089). It is that **all nine design cells share the same twelve path seeds**, so 108 rows were treated as 108
independent observations when they are closer to 12:

| | gap | naive s.e. | path-clustered s.e. | t |
|---|---|---|---|---|
| original | +0.206 | ±0.081 | **±0.157** | +1.32 |
| replication | −0.194 | ±0.046 | **±0.095** | −2.04 |

**The original claim was never significant** once the shared stimulus is accounted for; the naive standard error
understated by about 2×. The honest verdict is not "fragile to sampling" but "asserted with a precision the
design never had".

> This clustering diagnosis is **post-hoc**. It was run after the sign flip appeared, to explain it, and it is
> not part of the frozen plan. It is reported as a diagnosis, not as a test. It is also the third appearance in
> this repository of the same defect — a shared stimulus treated as independent replication — after the
> human-study power analysis and the stress suite's recovery estimator.

### PR2 — overturned, and not a vindication

> *The reliability-condition agent's advantage grows with environmental uncertainty.*

Failed on the registered seeds, supported on these. The frozen rule calls this *overturned* and says a third run
is needed before anybody relies on it, which is the right reading: a prediction that lands on opposite sides of
its threshold on two draws is sitting near the boundary, and the first verdict was as fragile as the second.
Nothing here licenses reporting PR2 as supported. The other five tournament predictions fail again.

## The claims that held

- **S1** the correctly specified model wins on the control process (rank 1 again).
- **S2** shape-free window models win where distributional shape is the problem (saa rank 1 again on
  multimodal demand).
- **S4** knowing the true nuisance parameters beats having the right functional form — `bayes_true` ranks first
  on 3 of the 5 stress families again, all of which it is misspecified for.
- **I2** buying information pays only when the environment changes fast enough (net −0.221 against −0.213,
  104% of the original).
- **H1** the gate's advantage falls as the learnability ratio rises (−0.72 against −0.95, 76%).

## The rejections that were confirmed

- **G1** the gate's nominal 90% lower bound covers **51.8%** of the time (range 45.0–56.3% across six feedback
  regimes), against 45–62% originally. It is not a confidence bound, twice.
- **S3** the moment-ambiguity model — the only one that *can* price outcomes never observed — ranked 4th
  originally and **5th** now, behind both models that provably cannot represent the stress. Capability does not
  imply performance, and the second run says so more loudly than the first.
- **PR1, PR3–PR6** five of the six reliability-condition predictions fail again.
- **V1, V2** as above.

## What this cannot do

A single replication of a simulation study is a weaker instrument than it looks. Both runs share the generating
code, the environment definitions, the policy implementations and every modelling assumption; what differs is
the draw. **A claim that survives here has survived sampling variation and nothing else** — not a different
operationalisation, not a different environment family, not a different research group. A claim rejected here
was fragile to sampling alone, which is the stronger statement.

Three adapter faults in this module, not in the studies, prevented V1, V2 and G1 from running on the first pass
(a wrong `build_scenario` signature, a `Tuned` object passed where a hyperparameter dict was wanted, and a
`GateValidation` object indexed as a frame). All three were fixed and the studies re-run. The plan digest is
unchanged throughout, which is correct: a broken adapter is not a change to the specification or the rule.

Nothing here concerns the human experiment, which has never been run on people.
