# Provenance and auditability of workbench runs

A stored run has to answer five questions on its own, without anyone re-reading the code that produced it:

1. Which source code produced it?
2. In what software environment?
3. Which random streams served which purpose?
4. What did each agent know at the moment it decided, and what was it told afterwards?
5. Which recorded fields did no agent ever see?

This note states how each is answered, and what to do when a change invalidates a registered result.

Implementation: `heiner_abm/workbench/provenance.py`, with the run assembled in
`heiner_abm/workbench/execution.py` and stored by `heiner_abm/workbench/store.py`. Tests: `tests/test_provenance.py`.

---

## 1. The code fingerprint

`provenance.code_fingerprint()` hashes the **transitive closure of first-party imports** reachable from the
workbench's entry points (`FINGERPRINT_ROOTS`). The closure is computed by parsing the source with `ast`, not by
walking imported module objects, because the environment adapters import their engines *inside* `run_cell`
(`from .. import learnability as L`), and an import-object walk would miss them.

The fingerprint currently covers 23 modules, among them `heiner_abm.agents`, `heiner_abm.arena`,
`heiner_abm.analysis` and `heiner_abm.workbench.analysis`. **The previous fingerprint hashed a hand-written list of
eleven modules and left all four of those out**, so a change to an agent, to the tournament code or to the analysis
left stored runs looking current.

Inspect it:

```bash
python -m heiner_abm.workbench fingerprint --modules
```

A run's key hashes the specification content, the kind (preview or research) **and** the fingerprint, so a change to
any covered module changes every run key and no stale run is reused. `tests/test_provenance.py` asserts this for
`heiner_abm.agents`, `heiner_abm.workbench.analysis` and `heiner_abm.nk`, and asserts the converse: with the code
unchanged, a rerun reproduces the same key and the same numbers (wall-clock `compute_ms` excepted).

## 2. The software environment

`provenance.software_environment()` records the Python version and implementation, the platform, and the installed
versions of every declared dependency, plus the BLAS thread-limit variables (which do not change results here but do
change timings). It deliberately records **no absolute paths**: a stored run may be published, and an interpreter
path carries the user's account name. A test asserts that.

## 3. Seed namespaces

Four blocks, disjoint by construction:

| Namespace | Offset | Purpose |
|---|---|---|
| `training` | `base*1000 + 100_000_000 + r` | Tuning only. Never reported as a result. |
| `pilot` | `base*1000 + 200_000_000 + r` | Pilot estimates: effect size, variance, workload. Never reported. |
| `evaluation` | `base*1000 + 300_000_000 + r` | The reported results. Common random numbers across policies. |
| `validation` | `base*1000 + 400_000_000 + r` | Confirmation on environments the result was not estimated on. |

`training` and `evaluation` keep the offsets the workbench has always used, so runs stored before this change
reproduce exactly.

`provenance.disjoint()` states the two conditions under which the blocks cannot collide — at most 1000 replications
per base seed, and a base seed below 100000 — and both the validator and `execute()` refuse a design that violates
them, rather than silently wrapping one namespace into the next.

Pilot and validation are separate passes over the same cells with the same tuned parameters. Their rows never enter
`trials`: they are returned and stored as `pilot.csv` and `validation.csv`, tagged with a `stage` column.

## 4. Decision logs

Every column of a decision log has a declared role (`provenance.DECISION_LOG_ROLES`):

- **predecision** — what the agent had when it chose. Everything it could have used.
- **action** — what it chose, and the stated reason.
- **postdecision** — consequences, known only afterwards.
- **feedback** — what it was later told, where that came from, when the outcome matured and when it reached the agent.
- **researcher_only** — recorded for analysis, never available to any agent.

Feedback carries three columns, so "feedback" is never ambiguous:

- `feedback_origin` — one of `own_outcome`, `own_counterfactual`, `observed_world`, `none`.
- `feedback_matured_period` — when the outcome existed.
- `feedback_release_period` — when the agent could use it; `-1` means **never**.

(The old single `feedback_available` column conflated maturation with release and had no origin.)

Per environment:

| Environment | Origin | Matures | Released |
|---|---|---|---|
| Inventory | `own_counterfactual` | same period | same period, and only where demand is observed |
| Market | `own_counterfactual` | decision period + H | when the engine validates it; never if a regime change invalidates it first |
| Generalization tasks | `own_counterfactual` | same period | same period (no delay, no missing feedback) |
| NK | `own_outcome` | same period | same period, where a second reading exists |

`provenance.check_decision_log()` rejects an unclassified column, feedback recorded without its timing, and an
undeclared origin. A test runs it over every environment.

## 5. Researcher-only fields

Columns prefixed `researcher_` were never available to any agent at any time. They are listed explicitly in the
manifest (`researcher_only_fields`) and in the export README, and a test asserts they never appear under the
`predecision` role.

The market records `researcher_profit_rule` and `researcher_profit_default`: `profit_rule` and `profit_default` are
researcher-only by `heiner_abm.information.RESEARCHER_ONLY`, because a firm sees the gain the engine releases to it,
never both branches of the counterfactual. A test asserts they are recorded only under the prefix.

## 6. Trace retention

`Design.trace_retention`, exposed in the design step and recorded in the manifest:

- `summary` — no per-period decision log is kept.
- `illustrative` — one replication per condition (the first). The default, and what the workbench did before.
- `all` — every replication of every condition.

Logs are stored column-wise and compressed: Parquet when `pyarrow` is installed (it is **not** a declared
dependency), gzipped pickle otherwise. Both round-trip the same frames; a test asserts an uncompressed layout is
never written.

## 7. The manifest

`runs/<key>/manifest.json`, also in the export bundle, holds: the question; the configuration and its parameters per
condition; the policies with what each reads and assumes; the candidate rule; the tuning mode, budget and chosen
values; **the cost definitions in force, one entry per distinct friction**; the information assumptions; the code
fingerprint with its module list; the software environment; the git commit; the seed plan; the decision-log schema;
the researcher-only fields; and the run status.

Cost definitions are stated separately rather than summed, and each environment states its own honestly:

- **Inventory** — three charges: default-departure overhead, fixed switching cost, magnitude cost.
- **Market** — one charge, with the note that the default *is* the previous output, so departure and switching are
  the same event there and cannot be separated. That is a property of this market's default, not evidence that the
  two frictions are one construct.
- **NK** — a magnitude cost per flipped bit and a fixed switching cost per move; the departure overhead is recorded
  as *not defined*, because an NK searcher has no default to depart from.
- **Generalization tasks** — nothing is charged for deviating, so these are the laboratory's zero-cost control.

---

## Registered findings: what this change did and did not touch

**No registered result is affected.** These changes are confined to `heiner_abm/workbench/*` and the interface pages.
Verified after the change: all seven finding fingerprints and all four registered plan hashes are unchanged
(tournament `22393384781bed84`, learnability `a98b971b49fe5e3e`, mechanisms `bbfaed9e70b900d6`, rule choice
`d2f88699dc6e6eb6`), and `REPLICATION_REQUIRED` is empty. No fingerprint was updated, because none moved.

The workbench's **cached runs** are invalidated, by design: the fingerprint now covers modules it previously missed,
so every run key changes. Stored runs from before this change remain readable; they will simply be recomputed rather
than reused.

### If a later change does move a registered hash

Do **not** re-register a hash to make a test pass. The procedure:

1. Confirm what moved:

   ```bash
   python -c "from heiner_abm import registered as R; print({k: R.source_fingerprint(p) for k,(_,p) in R.FINDING_SOURCES.items()})"
   ```

2. Add each affected finding to `REPLICATION_REQUIRED` in `heiner_abm/registered.py`, with the model change and its
   date as the reason. The app and the README then show it as requiring replication, and the reported numbers stay
   as they are.

3. Rerun what the change touched, each stage in its own process:

   ```bash
   python tools/rerun_tournament_findings.py        # tournament, 3 replications, mechanisms, rule choice
   python tools/rerun_special_and_patterns.py       # signature tests and field patterns (after the tournament)
   python tools/rerun_learnability_study.py         # the learnability study
   ```

4. Compare every plan hash printed by those tools with `heiner_abm/registered.py` **before** reporting any number as
   a registered result.

5. Update the registered numbers and fingerprints from the rerun, record a revision note saying what changed and
   which earlier results it supersedes, and remove the finding from `REPLICATION_REQUIRED` only once it has actually
   been rerun.
