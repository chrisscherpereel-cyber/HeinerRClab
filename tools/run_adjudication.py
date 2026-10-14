"""Replicate the research studies on independent seeds and adjudicate the claims.

Usage:  python tools/run_adjudication.py [<output directory>] [--studies a,b,c]

Every study is re-run with its seed moved by `adjudication.SEED_OFFSET` and nothing else changed. Each study's
own prespecified decision rule is re-applied to the new data, and the frozen rule in `heiner_abm/adjudication.py`
maps the pair of verdicts onto retained / narrowed / rejected / rejection_confirmed / overturned.

The plan, the claim table and the decision rule were frozen and hashed before any of this was run. Writes
verdicts.csv, claims.csv, outcomes.csv, not_run.csv, errors.csv and spec.json into docs/adjudication_results/.

A single replication tests a claim against SAMPLING variation only. Both runs share the generating code, the
environments and every modelling assumption; what differs is the draw. A claim that survives here has survived
that and nothing more.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heiner_abm import adjudication as A          # noqa: E402

args = list(sys.argv[1:])
studies = A.STUDIES
if "--studies" in args:
    i = args.index("--studies")
    studies = tuple(args[i + 1].split(","))
    del args[i:i + 2]

out = Path(args[0]) if args else ROOT / "docs" / "adjudication_results"
out.mkdir(parents=True, exist_ok=True)

plan = A.AdjudicationPlan()
print(f"plan {plan.digest} (code {A.adjudication_code_digest()})", flush=True)
print(f"seed offset {A.SEED_OFFSET}, retain fraction {A.RETAIN_FRACTION}", flush=True)
print(f"{len(A.CLAIMS)} claims across {len(studies)} studies: {', '.join(studies)}\n", flush=True)

t0 = time.time()
last = [0.0]


def note(fraction: float, message: str) -> None:
    if fraction - last[0] >= 0.02 or fraction >= 1.0:
        last[0] = fraction
        print(f"  {fraction * 100:5.1f}%  {message}   [{(time.time() - t0) / 60:5.1f} min]", flush=True)


frames = A.run_adjudication(studies, progress=note)
for name, df in frames.items():
    df.to_csv(out / f"{name}.csv", index=False)

(out / "spec.json").write_text(json.dumps({
    "plan_digest": plan.digest,
    "code_digest": A.adjudication_code_digest(),
    "plan": json.loads(plan.to_json()),
    "exports": A.EXPORTS,
    "caveat": "A single replication tests a claim against sampling variation only. Both runs share the "
              "generating code, the environments and every modelling assumption.",
}, indent=1, default=str), encoding="utf-8")

v = frames["verdicts"]
print(f"\nelapsed {(time.time() - t0) / 60:.1f} min\n")
if len(v):
    print(v["outcome"].value_counts().to_string())
    print()
    print(v[["claim", "study", "original_verdict", "replication_verdict", "outcome"]].to_string(index=False))
    moved = v[v["outcome"].isin(("rejected", "narrowed", "overturned"))]
    if len(moved):
        print(f"\n=== {len(moved)} claim(s) moved ===")
        for _, r in moved.iterrows():
            print(f"  {r['claim']} [{r['outcome']}] {r['statement'][:80]}")
            print(f"      {r['reason']}")
if len(frames["not_run"]):
    print(f"\n{len(frames['not_run'])} claim(s) could not be adjudicated:")
    print(frames["not_run"][["claim", "detail"]].to_string(index=False))
print(f"\nwrote {out}")
