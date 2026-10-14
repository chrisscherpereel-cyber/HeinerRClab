# Methods, for a reviewer

A short account of how this laboratory produces a number and what that number may be taken to mean. It is
written for somebody deciding how much weight to put on a result, not for somebody using the software.

**This application is not publication-ready, and passing its tests would not make it so.** What the tests and
the integration audit establish is that the software is wired the way this document says. Whether any
particular claim is supported is a separate question, answered per study, and several of the studies here
answer it in the negative. Section 7 lists what remains.

---

## 1. What a run is

A run executes one **specification**: a research question, an environment, a candidate-generation rule, a set
of policies, an information treatment, and a design. The same specification produces the same run from the
interface, the command line (`python -m heiner_abm.workbench`) or a saved file, because all three execute the
same object. The specification's content hash — everything except its name and the interface mode — is part of
the cache key, so a renamed experiment reuses its results and an edited one does not.

Work is organised in seven stages: **research question → environment and information → policies and tuning →
experimental design → execution → results → validation and export.** The ordering is not cosmetic. Each stage
can block the next, and the disclosure below is computed at the point between design and execution, which is
the last moment at which anything can still be changed.

## 2. What you are told before a run

Seven items, computed from the specification rather than entered by hand. An item that cannot be determined
says so; none is filled with a plausible default.

| | item | what it answers |
|---|---|---|
| 1 | Primary estimand and outcome | what is being estimated, on what scale, against which comparison |
| 2 | Cost semantics | which frictions are charged, at what rate, when, and whether any is inactive |
| 3 | Agent information restrictions | what each policy may read; whether any is privileged |
| 4 | Training and evaluation separation | which seed blocks serve which purpose, and whether they are disjoint |
| 5 | Policy tuning status | what was tuned, on which data, under what budget; who was *not* tuned |
| 6 | Replication requirements | whether a registered finding this run touches is awaiting a re-run |
| 7 | Calibration limitations | where an interval or bound is not known to mean what it appears to |

Items 1 and 4 can **block** a research run: an estimand that does not exist, or seed blocks that overlap. The
others inform rather than block, because a free-adaptation run and an untuned comparison are legitimate choices
that a reader simply has to know about.

Item 7 is never "ok". A reader told nothing about calibration will assume the default meaning, and in this
repository the default meaning is wrong in at least two measured places: the reliability gate's nominal 90%
bound covers 45–62% of the time, and a *t*-test over per-participant estimates from a pooled empirical-Bayes
fit rejects at 0.50–0.75 against a nominal 0.05.

## 3. What a result is allowed to claim

Five statuses. The weakest is offered first and is not embarrassing to use.

| status | means | may not claim |
|---|---|---|
| **Exploratory** | the analysis was chosen after seeing these data | a tested effect, or a p-value that means what it usually means |
| **Confirmatory** | quantity, comparison and analysis were fixed in a registered plan whose hash matches the code that produced the data | anything the plan did not name — a post-hoc subgroup is exploratory inside a confirmatory run |
| **Not supported** | a prespecified prediction was tested and not borne out | that the opposite is true |
| **Requires replication** | code this result depends on has changed since it was produced | the result as current |
| **Descriptive** | a summary with no comparison claimed | that any difference between conditions is an effect |

The workbench does not register plans. Unless a run executes one of the frozen plans in
`heiner_abm/registered.py`, the honest status is **exploratory**, and the interface says so when confirmatory is
selected.

## 4. Separation of training and evaluation

Four disjoint seed blocks — `training`, `pilot`, `evaluation`, `validation` — allocated by
`heiner_abm/workbench/provenance.py`. Tuning draws only from `training`; reported results draw only from
`evaluation`; the pilot never enters a reported result; validation re-runs a comparison on fresh environments
with the tuned parameters frozen. Overlap is detected and blocks the run rather than warning.

All policies within a condition see the **same** evaluation environments, so differences are paired and the
bootstrap resamples whole replications.

## 5. Information and cost parity

Charges are applied by the environment, not by the policy, so no policy can be exempted from a cost another
pays. The audit verifies that the set of charges in force does not depend on which policies are selected.

Every policy declares what it reads. A policy given researcher-only information is marked as a **benchmark**
and reported beside the ranking rather than inside it — `bayes_true` in the stress suite is the clearest case,
and the stress results say plainly that its advantage is partly privileged information rather than modelling
skill.

Where a comparison must be enforced rather than documented, it is:
`heiner_abm/information_contract.py` hands a policy its period record through a contract that refuses any field
the contract does not grant, by indexing, by attribute, by iteration and by copying, and carries only the
granted fields so there is nothing privileged inside to reach.

## 6. Provenance

Every stored run carries a manifest with the specification, the code fingerprint (an AST-closure hash over the
modules a run actually depends on), the software environment including package versions, the seed plan, the
cost definitions in force, the information assumptions, the tuning log, and the run status.

Registered findings are pinned by fingerprint in `heiner_abm/registered.py`. A change to code a registered
finding depends on marks that finding as requiring replication; fingerprints are never updated to make a test
pass. Frozen plans additionally hash the **code** that generates paths and scores them, not only the
parameters, so changing an update equation invalidates the plan that describes it. That rule was added after a
plan hash failed to notice a changed seed derivation.

## 7. What is not established

A reviewer should read these as the honest limits of the current state.

- **The human experiment has not been run.** Everything in `docs/human_learning.md` is synthetic participants
  generated by the models being fitted. It says whether the design *could* identify these mechanisms, not
  whether anybody behaves this way.
- **Several model comparisons are not identified at the lengths used.** In the trial-by-trial human models,
  only aspiration satisficing is cleanly recovered (0.90); the four action-value models are confused with each
  other, and data generated by one is assigned to another more often than to itself.
- **Confidence sensitivity is not separable from slow learning** through the parameter the hypothesis is about.
- **The reliability gate's bound is not a confidence bound** and is not described as one.
- **Registered tournament findings predate several refactors.** Where that is true the finding is marked, and
  the marking is the mechanism, not a note in prose.
- **Calibration of the corrected human-study test is consistent with nominal and not demonstrated.** Twelve
  replications cannot distinguish 0.08 from 0.05.
- **The learnability index is redundant** against a prespecified comparison model (+0.003 [−0.006, +0.026]),
  which bounds what its replicated negative slope can mean.

## 8. Reproducing

```bash
python tools/run_integration_audit.py        # the wiring audit (~5 min)
python -m pytest -q                          # the full suite
python -m heiner_abm.workbench <saved.json>  # one specification, from the command line
```

Study-level reruns are listed in `docs/implementation_report.md` with their commands and runtimes, including
which have been executed in this repository and which have not.
