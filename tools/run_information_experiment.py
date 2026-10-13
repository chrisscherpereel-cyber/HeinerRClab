"""Run the information-environment experiment and write its results.

Usage:  python tools/run_information_experiment.py [<output directory>] [--paths N] [--periods N]

Writes, into docs/information_results/ by default:
    cells.csv         one row per cell and path: the three arms and the decomposition of what buying is worth
    decisions.csv     acquisition decisions per arm and what they cost
    availability.csv  what was observable: stockouts, recorded unmet demand, signals received
    truth.csv         researcher-only ground truth, in its own file so it cannot be mistaken for agent information
    interactions.csv  the value of information and the value of choosing against adjustment cost and persistence
    spec.json         the frozen plan and its hash, the instruments, and every contract with its granted fields

The four frames are written separately on purpose. `truth.csv` prefixes every researcher-only column with
`researcher_`, and a test asserts that prefix appears there and in none of the others.

Nothing here is a registered finding: this is its own frozen plan, separate from the registered studies, and no
registered fingerprint or plan hash depends on these modules.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heiner_abm import information_env as E          # noqa: E402

args = [a for a in sys.argv[1:]]
opts = {}
for flag in ("--paths", "--periods"):
    if flag in args:
        i = args.index(flag)
        opts[flag.lstrip("-")] = int(args[i + 1])
        del args[i:i + 2]

out = Path(args[0]) if args else ROOT / "docs" / "information_results"
out.mkdir(parents=True, exist_ok=True)

plan = E.InfoPlan(**opts)
print(f"plan {plan.digest}  "
      f"{len(plan.unmet_access) * len(plan.adjustment_costs) * len(plan.hazards)} cells "
      f"x {plan.paths} paths x {plan.periods} periods", flush=True)

t0 = time.time()
last = [0.0]


def note(fraction: float, message: str) -> None:
    if fraction - last[0] >= 0.05 or fraction >= 1.0:
        last[0] = fraction
        print(f"  {fraction * 100:5.1f}%  {message}   [{time.time() - t0:6.1f}s]", flush=True)


frames = E.run_experiment(plan, progress=note)
for name, df in frames.items():
    df.to_csv(out / f"{name}.csv", index=False)

table = E.interaction_table(frames["cells"])
table.to_csv(out / "interactions.csv", index=False)

cells = frames["cells"]
residual = float(np.abs(cells["net_of_buying"]
                        - (cells["value_of_information"] + cells["value_of_choosing"]
                           - cells["price_paid"])).max())

(out / "spec.json").write_text(json.dumps({
    "plan_digest": plan.digest,
    "plan": json.loads(plan.to_json()),
    "exports": E.EXPORTS,
    "instruments": {k: {"price": v.price, "sd": v.sd, "delay": v.delay, "reveals": v.reveals}
                    for k, v in E.INSTRUMENTS.items()},
    "contracts": {k: v.to_dict() for k, v in E.CONTRACTS.items()},
    "decomposition_residual_max": residual,
}, indent=1), encoding="utf-8")

print(f"\nelapsed {(time.time() - t0) / 60:.1f} min")
print(f"decomposition residual (max |.|): {residual:.3e}")
print(f"\n{table.to_string(index=False, float_format=lambda v: f'{v:.3f}')}")
print(f"\nwrote {out}")
