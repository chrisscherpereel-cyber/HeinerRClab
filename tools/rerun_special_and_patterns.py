"""Rerun the signature tests and the field patterns, the two registered findings that use the tournament's agents.

Usage:  python tools/rerun_special_and_patterns.py [<output directory>]

Both read the tuned designs and parameters from heiner_abm/registered.py (registered_tuned), so run this *after*
TUNED_DESIGN / TUNED_PARAMS have been updated from a tournament rerun. Settings are the ones the registered numbers
were produced at: signature tests at the Full scale (special.SCALES), field patterns at 12 environments and 600
periods, the defaults of the Field patterns page.

Writes special.json and patterns.json, and prints the source fingerprints so FINDING_FINGERPRINTS can be updated.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heiner_abm import registered as R
from heiner_abm import special as S
from heiner_abm.patterns import PATTERNS, run_patterns

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs/tournament_results")
out_dir.mkdir(parents=True, exist_ok=True)
LF = chr(10)


def show(obj):
    enc = sys.stdout.encoding or "utf-8"
    print(str(obj).encode(enc, "replace").decode(enc), flush=True)


t0 = time.time()
show("signature tests (Full scale)…")
special = {}
for key in sorted(S.SPECIAL_BY_KEY):
    res = S.run_special(key, "Full")
    special[key] = {"verdict": res["verdict"], "result": res.get("result", ""), "ok": bool(res["ok"])}
    show(f"  {key:12s} {res['verdict']:14s} {str(res.get('result', ''))[:120]}")
json.dump(special, open(out_dir / "special.json", "w"), indent=1)

show(LF + "field patterns (12 environments, 600 periods)…")
pat = run_patterns(R.registered_tuned(), n_envs=12, periods=600)
pat.to_csv(out_dir / "patterns.csv", index=False, lineterminator=LF)
cols = [c for c in ("pattern", "key", "population", "passed", "result", "detail") if c in pat.columns]
json.dump(json.loads(pat[cols].to_json(orient="records")), open(out_dir / "patterns.json", "w"), indent=1)
show(pat[cols].to_string(index=False))

show(LF + f"TOTAL {(time.time() - t0) / 60:.1f} min")
for key in ("special", "patterns"):
    show(f"{key}: fingerprint {R.source_fingerprint(R.FINDING_SOURCES[key][1])} "
         f"(registered {R.FINDING_FINGERPRINTS.get(key)})")
