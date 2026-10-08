# Model & Assessment Audit — Juicetification: Capacity Crush

This audit documents the simulation's current assumptions and separates **verified defects**
(reproduced in code/tests) from **modeling assumptions** and **proposed changes that would move
tuned numbers** (left for a sign-off step so lab thresholds can be re-tuned deliberately). It
reflects the code as reviewed on 2026-10-07. Tests live in `tests/test_model.py`
(`python tests/test_model.py` → 11/11 passing).

Guiding rule followed here: *do not silently change simulation outputs or the supply model.*
Every change shipped in this pass is either additive instrumentation, a label/wording
clarification, or UI/assessment/reporting — none of them alter the random stream or the
tuned lab numbers (verified: the Operations lab back-compat sequence
`[6965, 6869, 4129, 4197, 4119, 3071, 7062, 7082]` is unchanged).

## 1. Current model (as implemented)

- **Line.** Up to 9 serial stations; a station is *active* with dice>0 and faces>0. Each hour a
  station's potential output is the sum of `dice` draws of `uniform(1..faces)` — mean
  `dice·(faces+1)/2` bottles/hr. A station passes the **min** of (its roll, the inventory in
  front of it, and downstream free space under any WIP cap). Moves resolve **downstream-first**.
- **Time/units.** 1 tick = 1 bottling hour; `HOURS_PER_DAY` per day, `DAYS_PER_YEAR` days/yr,
  `HOURS_PER_YEAR` hours/yr. All inventories are whole bottles.
- **Event order each hour:** (1) roll potentials; (2) **supplier reorder check** on the raw
  buffer; (3) resolve moves downstream→upstream, applying **scrap** to the units each station
  works; (4) finished goods meet **demand** (sold / lost / held as FGI); (5) accumulate stats.
- **Supply model (precise).** A **reorder policy**, *not* a lead-time model: when the raw buffer
  in front of Op 1 falls below the reorder point, the supplier ships `ceil(deficit/order_size)`
  lots of `order_size` **with probability = reliability that hour**; a missed hour simply retries
  next hour. "Supplier reliability" is therefore the per-hour chance a replenishment *opportunity*
  succeeds — there is no explicit transit lead time. Default reorder point ≈ one hour of Op 1
  capacity.
- **Scrap/yield.** Applied deterministically via a fractional carry (draws **no** RNG, so
  `scrap=None` is a byte-for-byte no-op). Scrapped units leave the system at the station.
- **Starvation / "service level".** `service_level = 1 − starved_hours/hours` = share of hours Op 1
  had material = **production availability** (supplier/line side). A separate **customer fill
  rate** `= sold/demand` exists and is only meaningful when a fluctuating market is on.
- **Financials.** Revenue = units sold × price. Production cost = Σ(good units × per-unit cost).
  Fixed (dice) cost allocated per active die. Holding = avg WIP (unit-days) × rate, plus FGI
  holding. Raw cost = Op 1 good output × unit cost. Ordering = `ceil(consumption/order_size) ×
  order_cost`.

## 2. Verified defects

| # | Area | Finding (reproduced) | Status |
|---|------|----------------------|--------|
| D0 | Deploy | The GitHub repo is **missing `.streamlit/config.toml`** (the web "upload files" flow drops hidden folders). Without it a dark-mode viewer sees instruction text blend into the background — the earlier dark-mode report, re-introduced by the upload. | **Fixed** — file restored; also a CSS light-theme safety net is in place. |
| D1 | Service label (spec 3G) | The glossary called `service_level` a "fill rate", and the results panel labeled it "Service level (line fed)". It is **production availability**, not a customer service level; a true customer fill rate also exists but wasn't surfaced. | **Fixed** — relabeled to "Raw-material availability"; customer fill rate now shown separately when a market is on; glossary distinguishes the two. |
| D2 | Assessment (spec 5) | Running **out of tries** marks a challenge step "done" (student may continue) — correct — but the sidebar progress tracker showed the same green ✓ as a pass, so "attempted" and "passed" looked identical on screen. (The PDF report already distinguished them.) | **Fixed** — progress card now shows "Challenges passed: X of Y" and flags attempted-but-not-passed separately. |
| D3 | Save status (spec 6) | The caption always read "progress saved automatically" even if a save failed or no storage was configured. | **Fixed** — status now reports enabled/last-saved/failed honestly, and a storage-independent **backup/restore** (JSON) is provided. |

## 3. Verified modeling inconsistencies — **proposed** (would move tuned numbers; not yet applied)

These are real and reproduced, but fixing them shifts financial outputs and the tuned
EOQ/economics lab thresholds and distractors. They are **documented, instrumented, and left for a
deliberate re-tuning pass** rather than changed silently (per the spec).

- **P1 — Ordering cost is inferred, not counted (spec 3C).** `orders = ceil(consumption/order_size)`
  ignores the actual replenishment events. The engine now **records the real counts**
  (`delivery_events`, `orders_placed`); e.g. a balanced JIT line consumes ≈6965 units but actually
  ships 7112 lots over 2096 delivery hours — so the proxy understates orders. *Proposed:* cost
  ordering from `orders_placed`. Impact: EOQ-lab dollar figures and the EOQ curve shift; EOQ-driver
  thresholds/distractors need re-checking.
- **P2 — Raw & processing cost exclude scrapped units (spec 3B).** Raw cost uses Op 1 **good**
  output and processing cost uses **good** units, so bottles a station worked and then scrapped are
  effectively free. *Proposed:* charge raw on units **started** at Op 1 and processing on units
  **worked** (`mv`) rather than good. Impact: economics/quality-lab costs rise where scrap is on.
- **P3 — Little's Law boundary on non-stationary lines (spec 3F).** For a **stationary** line
  (balanced or WIP-capped) derived `W=L/λ` matches the measured sojourn within ~1% (test
  `test_littles_law_compatible_when_stationary`). For an **uncapped bottleneck** WIP never settles,
  so units still queued at run end are excluded from measured flow while still counted in average
  WIP — a legitimate finite-run/censoring divergence (~58% in the test case). *Proposed, optional:*
  a warm-up/observation window and an on-screen "line hasn't reached steady state" note. This is a
  limitation to **explain**, not a bug to hide.

## 4. Assumptions worth stating (not defects)

- Starting inventory is placed in front of **every** station (a deliberate UI choice, labeled as
  such), not only at Op 1.
- The reported financial result is a **simplified operating result**, not accrual accounting or
  cash flow; initial inventory is not costed, unsold FGI is held at the holding rate, capital is
  allocated per die. This should be stated in-app wherever "profit" appears (minor doc task).
- "Stability" is never asserted from two close estimates; the app reports distributions/efficiency
  rather than a binary stable/unstable verdict.

## 5. What this pass changed intentionally (deliverable #6)

- **Simulation outputs: no change.** The RNG stream and all existing return values are identical
  (verified by the back-compat sequence and `test_deterministic_reproducibility`). New return
  fields are **additive** (`raw_received`, `delivery_events`, `orders_placed`, `initial_material`,
  `ending_material`, `ending_wip_downstream`, `material_residual`).
- **Labels/UI/assessment/reporting:** the D0–D3 fixes above, plus the test suite.

## 6. Deferred to later phases (plan)

Phase 2: consolidate the guided-lab workspace (one primary action; task controls in the main
panel) and surface the line-diagram instrumentation (nominal vs processed vs good vs starved vs
blocked). Phase 3: matched exogenous random draws for A/B (thread an explicit `random.Random`
per stream so a configuration change doesn't desync draws), CSV export of comparisons, and the
three targeted scenarios (moving constraint, quality location, capacity investment under
uncertainty). Phase 4: split engine / financials / lab definitions / assessment / reports /
presentation into modules behind the current entry point. Each of P1/P2 above belongs with a
re-tuning of the affected lab thresholds.
