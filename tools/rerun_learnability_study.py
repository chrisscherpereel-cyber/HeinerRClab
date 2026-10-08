"""Rerun the registered learnability study and write the tables in docs/learnability_results/.

Usage:  python tools/rerun_learnability_study.py [<output directory>]

Runs heiner_abm.learnability.run_study under the frozen plan (heiner_abm.learnability.LEARN_PREREG) together with
its market replication, and writes verdicts.csv, test_configurations.csv, new_families.csv, sweeps.csv,
negative_controls.csv, secondary.csv, market.csv, tuning.csv and run.json. run.json records the plan hash the run was
produced under; compare it with heiner_abm.registered.LEARN_PLAN before reporting the numbers as registered results.

The study takes under 10 minutes on one core (7.4 minutes on the machine that produced the registered
tables). Nothing is cached: rerunning reproduces the same numbers because every seed is derived from the plan.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heiner_abm import learnability as L
from heiner_abm import learnability_market as LM
from heiner_abm import registered

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs/learnability_results")
out_dir.mkdir(parents=True, exist_ok=True)
LF = chr(10)      # the repository stores these tables with LF endings on every platform

plan = L.LEARN_PREREG
t0 = time.time()
res = L.run_study(plan, progress=lambda f, m: print(f"[{f:5.1%}] {m}", flush=True), market_plan=LM.MarketPlan())
elapsed = time.time() - t0

res.verdicts.to_csv(out_dir / "verdicts.csv", index=False, lineterminator=LF)
res.tests.to_csv(out_dir / "test_configurations.csv", index=False, lineterminator=LF)
res.new_families.to_csv(out_dir / "new_families.csv", index=False, lineterminator=LF)
res.sweeps.to_csv(out_dir / "sweeps.csv", index=False, lineterminator=LF)
res.controls.to_csv(out_dir / "negative_controls.csv", index=False, lineterminator=LF)
res.tuning.to_csv(out_dir / "tuning.csv", index=False, lineterminator=LF)
L.secondary_table(res.paths, "test", n_boot=plan.n_boot).to_csv(out_dir / "secondary.csv", index=False, lineterminator=LF)
res.market["tests"].to_csv(out_dir / "market.csv", index=False, lineterminator=LF)
json.dump({"plan_hash": res.plan_hash, "registered_plan_hash": registered.LEARN_PLAN,
           "matches_registered_plan": res.plan_hash == registered.LEARN_PLAN,
           "hyperparameters": res.hyperparameters, "market_theta": res.market["theta"],
           "market_gate": list(res.market["gate_hp"]), "minutes": round(elapsed / 60, 1)},
          open(out_dir / "run.json", "w"), indent=1)


def show(obj):
    """Windows consoles default to cp1252; the tables contain minus signs and superscripts."""
    enc = sys.stdout.encoding or "utf-8"
    print(str(obj).encode(enc, "replace").decode(enc))


show(f"{chr(10)}plan {res.plan_hash} (registered {registered.LEARN_PLAN}) in {elapsed / 60:.1f} min")
show(res.tuning.to_string(index=False))
show(res.verdicts[["id", "verdict", "result"]].to_string(index=False))
show(res.controls.to_string(index=False))
