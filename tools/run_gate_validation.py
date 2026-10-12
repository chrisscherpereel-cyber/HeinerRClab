"""Run the reliability-gate validation and write its results.

Usage:  python tools/run_gate_validation.py [<output directory>] [--draws N] [--states N] [--markets N]

Writes, into docs/gate_validation_results/ by default:
    summary.csv      one row per gate and feedback regime: calibration, measured coverage, both error rates and
                     what each costs
    calibration.csv  the plot data: predicted-advantage decile against the conditional expectation A*
    states.csv       every sampled decision state with A*, its Monte-Carlo standard error, and the researcher-only
                     truths it was computed from
    variants.csv     the four memory variants scored on the same recorded evidence streams
    spec.json        the frozen plan, its hash, the estimands, the continuation policy and the seed blocks

Nothing here is a registered finding: the validation plan is its own frozen plan, separate from the registered
studies, and no registered fingerprint or plan hash depends on this module.
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heiner_abm import gate_validation as V          # noqa: E402
from heiner_abm.scenarios import DEFAULTS, build_scenario   # noqa: E402

args = [a for a in sys.argv[1:]]
opts = {}
for flag in ("--draws", "--states", "--markets"):
    if flag in args:
        i = args.index(flag)
        opts[flag.lstrip("-")] = int(args[i + 1])
        del args[i:i + 2]
out_dir = Path(args[0]) if args else ROOT / "docs/gate_validation_results"
out_dir.mkdir(parents=True, exist_ok=True)
LF = chr(10)


def show(obj):
    enc = sys.stdout.encoding or "utf-8"
    print(str(obj).encode(enc, "replace").decode(enc), flush=True)


base = build_scenario(dict(DEFAULTS))
base = dataclasses.replace(base, periods=400, burn_in=50)
for f in base.firms:
    f.selection = "Adaptive"

plan = V.ValidationPlan(
    markets=opts.get("markets", 6),
    states_per_cell=opts.get("states", 200),
    continuation=V.ContinuationSpec(draws=opts.get("draws", 64)),
)
show(f"plan {plan.digest}  |  {len(plan.gates)} gates x {len(plan.regimes)} regimes, "
     f"{plan.states_per_cell} states per cell, {plan.continuation.draws} continuations per state")

t0 = time.time()
res = V.run_validation(plan, base, progress=lambda f, m: show(f"  {100 * f:5.1f}%  {m}"))
show(f"{LF}TOTAL {(time.time() - t0) / 60:.1f} min")

res.summary.to_csv(out_dir / "summary.csv", index=False, lineterminator=LF)
res.calibration.to_csv(out_dir / "calibration.csv", index=False, lineterminator=LF)
res.states.to_csv(out_dir / "states.csv", index=False, lineterminator=LF)
res.variants.to_csv(out_dir / "variants.csv", index=False, lineterminator=LF)
json.dump({**res.settings, "not_run": res.not_run}, open(out_dir / "spec.json", "w"), indent=1, default=str)

cols = ["gate", "regime", "states", "cal_bias", "cal_slope", "cov_nominal", "cov_empirical",
        "false_adaptation_rate", "missed_adaptation_rate", "cost_of_false_adaptation",
        "cost_of_missed_adaptation", "regret_total"]
show(LF + res.summary[[c for c in cols if c in res.summary]].round(3).to_string(index=False))
if not res.variants.empty:
    vcols = ["variant", "n", "bias", "rmse", "coverage_vs_next_value", "decided_share",
             "exploration_propensity", "exploration_cost"]
    show(LF + res.variants[[c for c in vcols if c in res.variants]].round(3).to_string(index=False))
for key, why in (res.not_run or {}).items():
    show(f"not run: {key}: {why}")
show(f"{LF}written to {out_dir}")
