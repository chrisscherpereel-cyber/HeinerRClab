# Implementation report: integrating the improvements into the workflow

Modules: `heiner_abm/workbench/disclosure.py`, `heiner_abm/integration_audit.py`.
Tests: `tests/test_disclosure_audit.py` (31). Audit results: `docs/audit_results/`.
Reproduce: `python tools/run_integration_audit.py`.

> **The application is not publication-ready, and the audit passing does not make it so.** What follows
> establishes that the software is wired the way the documentation says. That is a precondition for a result,
> not a result. Section 5 lists what is still missing, and several entries there are substantive.

---

## 1. What changed in the workflow

**Seven stages instead of six.** The workbench now runs research question → environment and information →
policies and tuning → experimental design → execution → results → validation and export. Information moved in
with the environment, because what an agent may see is a property of the world it is in rather than a separate
decision. Results and validation/export are new: a run previously ended at execution and left the reader to
find the numbers and decide, unaided, what they meant.

The existing navigation is unchanged. The top-level sections (Start, Experiment, Results, Validation,
Reference) are as they were; the restructuring is inside the Experiment workspace, which is where the research
work actually happens.

Validation-issue keys were **not** renamed. `information` is kept as an alias onto the environment stage, so
environment code raising an issue against it keeps working, and an issue raised against an unknown stage is
routed rather than silently dropped. Both are tested.

**A pre-run disclosure panel**, between design and execution — the last point at which anything can still be
changed. Seven items, computed from the specification: primary estimand and outcome, cost semantics, agent
information restrictions, training/evaluation separation, policy tuning status, replication requirements and
calibration limitations. Two can block a research run (a non-existent estimand, overlapping seed blocks); the
rest inform. An item that cannot be determined says so rather than filling in a default.

**Explicit result statuses** — exploratory, confirmatory, not supported, requires replication, descriptive —
each with its meaning, what it permits and what it forbids. The weakest is offered first. Choosing
*confirmatory* raises a warning that the workbench does not register plans, so unless the run executes one of
the frozen plans the honest status is exploratory.

**A run record** exportable as JSON: the specification, the seven disclosure items as they stood, the chosen
status, the reproducibility checklist, the code fingerprint and the software environment.

## 2. The integration audit

Six checks, 106 findings, run in 3.8 minutes.

| check | pass | fail | warn | info | verdict |
|---|---|---|---|---|---|
| reachability | 48 | 0 | 0 | 0 | pass |
| parity | 33 | 0 | 0 | 3 | pass |
| provenance | 7 | 0 | 0 | 0 | pass |
| tests | 4 | 0 | 0 | 0 | pass |
| end_to_end | 3 | 0 | 0 | 0 | pass |
| outstanding | 7 | 0 | 0 | 1 | pass |

**1. Configuration parameters reach the intended components.** Every one of the **48 treatable controls across
all five environments** was set to a different value and the run re-executed: all 48 moved at least one
reported outcome. This is the check that would have caught the tuning step that scored the same configuration
four times, and the sampling-history manipulation that moved nothing.

**2. Policy information and cost parity.** All 33 policies declare what they read. Three are privileged
benchmarks, correctly marked and reported beside the ranking rather than inside it. In every environment the
set of charges in force is the same whichever policies are compared, so no policy can be exempted from a cost
another pays.

**3. Caching and provenance.** The run key is stable for an unchanged specification, changes when the
specification changes, and does **not** change when only the name changes. The code fingerprint covers the
modules a run depends on; the software environment records Python, platform and ten package versions; the four
seed blocks are disjoint at a representative size.

**4. Engine and interface tests.** Engine 58 passed; workbench 84 passed, 8 skipped; interface 311 passed;
registered 4 passed.

**5. Small reproducible end-to-end experiments.** A complete preview run of the `restriction` preset,
executed twice: **45 rows × 10 substantive columns identical**. `compute_ms` is excluded by name and the
exclusion is reported — it is wall-clock time, and comparing it would report a reproducibility failure on
every machine. The manifest carries 15 sections including code, configuration, costs, information, policies
and seeds.

**6. Full studies still requiring execution.** Seven of eight have results in the repository (tournament,
learnability, construct validity, gate validation, information environments, stress suite, human-model
simulation). The eighth — **the human experiment itself — requires participants and has not been run.** The
audit reports it as such rather than as complete.

### Three defects the audit found in itself

Worth recording, because each would have produced a false verdict:

1. **An invalid probe reported as a wiring failure.** NK declares interactions up to 15 but requires
   K ≤ N − 1 = 7 at the default N. The environment validates that and blocks the run; the audit's probe
   bypassed validation and reported the guard as a defect. Probes are now validation-aware and try
   progressively milder values.
2. **Wall-clock time compared as a result.** The first end-to-end check reported "two runs disagreed" on every
   invocation. The only differing column was `compute_ms`.
3. **A provenance warning from reading the wrong key.** numpy and pandas are recorded under `packages`, not at
   the top level of the software environment, which the audit reported as missing.

## 3. Test outcomes

| suite | result |
|---|---|
| `tests/test_disclosure_audit.py` (new) | 31 passed |
| `tests/test_workbench.py` | 79 passed |
| full suite | see below |

Two existing tests failed during the restructuring and were fixed rather than skipped:
`test_workspace_steps_and_results_pages_run` iterated a hard-coded list of the old six stages — it now reads
`STEPS` from the module, so it cannot drift again — and then clicked the preview button while the page had been
left on the last stage.

## 4. Reproducibility checklist

For any result produced with this workbench:

- [ ] The specification is saved and named.
- [ ] The primary outcome and the comparison are named in the question stage, not inherited from a default.
- [ ] A smallest effect of practical interest is declared, so "no difference" can be distinguished from "too
      small a sample to tell".
- [ ] Seed blocks are disjoint, and tuning (if any) draws only from `training`.
- [ ] Evaluation replications are at or above the research minimum of 8.
- [ ] Decision logs are retained at `illustrative` or `all`, not `summary`.
- [ ] The seven disclosure items were read before the run, not after.
- [ ] A result status is chosen, and it is `exploratory` unless the run executes a registered frozen plan.
- [ ] The run record JSON is exported and kept with the result.
- [ ] `python tools/run_integration_audit.py` passes on the code that produced the result.

The first eight are checked in the interface at stage 7 and shown as met or unmet.

## 5. Remaining limitations

**Of the integration work:**

- The disclosure reads the specification, not the run. A run whose environment behaves differently from its
  declared cost definitions would not be caught by the disclosure; the audit's parity check covers the
  declaration, not the execution.
- Reachability tests that a control moves *something*. It does not test that it moves the *right* thing in the
  right direction. A control wired to the wrong component would pass.
- The audit's end-to-end check runs one preset at preview scale. It establishes determinism, not correctness.
- The result status is chosen by the user and stored in the session; nothing prevents choosing `confirmatory`
  for an unregistered run beyond a warning.

**Of the laboratory, carried forward and unresolved:**

- The human experiment has not been run. All human-model results are synthetic.
- Only aspiration satisficing is cleanly recovered among the trial-by-trial models; the action-value family is
  not identified at the lengths used.
- Confidence sensitivity is not separable from slow learning through its own parameter.
- The reliability gate's nominal 90% bound covers 45–62%.
- The corrected human-study test is consistent with nominal calibration but not shown to achieve it.
- The learnability index is redundant against its prespecified comparison model.

## 6. Study inventory

| study | results | command | scale |
|---|---|---|---|
| Registered agent tournament | present (16 files) | `python tools/rerun_tournament_findings.py` | hours |
| Learnability study | present (9) | `python tools/rerun_learnability_study.py` | ~16 min |
| Incremental construct validity | present (4) | module `run_validity` | ~20 min |
| Reliability-gate validation | present (6) | `python tools/run_gate_validation.py` | ~25 min |
| Censored demand and acquisition | present (6) | `python tools/run_information_experiment.py` | ~12 s |
| Registered stress-test suite | present (11) | `python tools/run_stress_suite.py` | ~35 min |
| Trial-by-trial human models | present (13) | `python tools/run_human_learning.py` | ~83 min |
| **The human experiment** | **none** | requires participants | not executable here |
