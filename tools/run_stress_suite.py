"""Run the registered stress-test suite and write its results.

Usage:  python tools/run_stress_suite.py [<output directory>] [--quick] [--paths N] [--periods N]

Writes, into docs/stress_results/ by default, one CSV per export (see heiner_abm.stress.EXPORTS) plus
spec.json with the frozen plan, its hash, the process equations and the specification map.

Two halves run in order. TUNING uses only the tuning variant of each family, a subset of the design cells, a
shorter horizon and seed block 1. EVALUATION freezes the selected configurations and runs the reserved variant
of every stress family on the full factorial, at full length, on seed block 2. The two never share a variant of
a stress family or a seed.

Nothing here is a registered finding of the laboratory: this is its own frozen plan, and no registered
fingerprint or plan hash depends on this module.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heiner_abm import stress as S          # noqa: E402

args = list(sys.argv[1:])
opts = {}
if "--quick" in args:
    args.remove("--quick")
    opts.update(S.QUICK)
for flag in ("--paths", "--periods"):
    if flag in args:
        i = args.index(flag)
        opts[flag.lstrip("-")] = int(args[i + 1])
        del args[i:i + 2]

out = Path(args[0]) if args else ROOT / "docs" / "stress_results"
out.mkdir(parents=True, exist_ok=True)

plan = S.StressPlan(**opts)
print(f"plan {plan.digest}: {len(plan.families)} families x 32 cells x {len(S.POLICIES)} policies "
      f"x {plan.paths} paths x {plan.periods} periods", flush=True)

t0 = time.time()
last = [0.0]


def note(fraction: float, message: str) -> None:
    if fraction - last[0] >= 0.02 or fraction >= 1.0:
        last[0] = fraction
        print(f"  {fraction * 100:5.1f}%  {message}   [{(time.time() - t0) / 60:5.1f} min]", flush=True)


frames = S.run_suite(plan, progress=note)
for name, df in frames.items():
    df.to_csv(out / f"{name}.csv", index=False)

(out / "spec.json").write_text(json.dumps({
    "plan_digest": plan.digest,
    "plan": json.loads(plan.to_json()),
    "exports": S.EXPORTS,
    "policies": {k: {"role": S.ROLE[k], "parameters": sorted(S.SPACES[k])} for k in S.POLICIES},
}, indent=1, default=str), encoding="utf-8")

print(f"\nelapsed {(time.time() - t0) / 60:.1f} min")
summary = frames["summary"]
for family in plan.families:
    g = summary[summary["family"] == family]
    print(f"\n=== {family} ({S.PROCESSES[family].name}) ===")
    print(g[["policy", "specification", "regret", "regret_se", "tail_loss_cvar5",
             "calibration_gap"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nwrote {out}")
