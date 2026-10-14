"""Run the integration audit and write its findings.

Usage:  python tools/run_integration_audit.py [<output directory>] [--quick] [--checks a,b,c]

Writes findings.csv, summary.csv and checks.csv into docs/audit_results/ by default.

The audit reports whether the software is wired the way the documentation says. It is not a scientific result
and passing it does not make anything publication-ready: it is a precondition, not a conclusion.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heiner_abm import integration_audit as A          # noqa: E402

args = list(sys.argv[1:])
quick = "--quick" in args
if quick:
    args.remove("--quick")
checks = tuple(k for k, _ in A.CHECKS)
if "--checks" in args:
    i = args.index("--checks")
    checks = tuple(args[i + 1].split(","))
    del args[i:i + 2]

out = Path(args[0]) if args else ROOT / "docs" / "audit_results"
out.mkdir(parents=True, exist_ok=True)

t0 = time.time()
last = [0.0]


def note(fraction: float, message: str) -> None:
    if fraction - last[0] >= 0.03 or fraction >= 1.0:
        last[0] = fraction
        print(f"  {fraction * 100:5.1f}%  {message}   [{time.time() - t0:5.0f}s]", flush=True)


frames = A.run_audit(checks, quick=quick, progress=note)
for name, df in frames.items():
    df.to_csv(out / f"{name}.csv", index=False)

print(f"\nelapsed {(time.time() - t0) / 60:.1f} min\n")
print(frames["summary"].to_string(index=False))
bad = frames["findings"][frames["findings"]["status"].isin(("fail", "warn"))]
if len(bad):
    print(f"\n=== {len(bad)} finding(s) needing attention ===")
    print(bad[["check", "item", "status", "detail"]].to_string(index=False))
else:
    print("\nNo failing or warning findings.")
print(f"\nwrote {out}")
