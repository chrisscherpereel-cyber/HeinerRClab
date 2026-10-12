"""Command line: identical execution to the interface.

    python -m heiner_abm.workbench run spec.json [--kind research|preview] [--out results_dir]
    python -m heiner_abm.workbench run-dir <run directory>        (used by background runs)
    python -m heiner_abm.workbench preset <key> > spec.json       (write a guided-question preset)
    python -m heiner_abm.workbench estimate spec.json
    python -m heiner_abm.workbench fingerprint                    (the code fingerprint and what it covers)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import execution, presets, store
from .spec import ExperimentSpec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m heiner_abm.workbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("spec")
    r.add_argument("--kind", default="research", choices=execution.KINDS)
    r.add_argument("--out", default=None, help="also copy the stored outputs to this directory")
    d = sub.add_parser("run-dir")
    d.add_argument("dir")
    p = sub.add_parser("preset")
    p.add_argument("key", choices=[k for k, v in presets.PRESETS.items() if v.spec is not None])
    e = sub.add_parser("estimate")
    e.add_argument("spec")
    e.add_argument("--kind", default="research", choices=execution.KINDS)
    f = sub.add_parser("fingerprint")
    f.add_argument("--modules", action="store_true", help="list every module the fingerprint covers")
    a = ap.parse_args(argv)
    if a.cmd == "fingerprint":
        from . import provenance
        detail = provenance.code_fingerprint_detail()
        if not a.modules:
            detail = {k: v for k, v in detail.items() if k != "modules"}
        print(json.dumps({"code": detail, "software": provenance.software_environment()}, indent=1))
        return 0
    if a.cmd == "preset":
        print(presets.load(a.key, "research").to_json())
        return 0
    if a.cmd == "run-dir":
        execution.run_from_dir(a.dir)
        return 0
    spec = ExperimentSpec.from_json(open(a.spec).read())
    if a.cmd == "estimate":
        print(json.dumps(execution.estimate(spec, a.kind), indent=1))
        return 0
    out = store.cached(spec, a.kind)
    if out is None:
        out = execution.execute(spec, a.kind, progress=lambda f, m: print(f"{100 * f:5.1f}%  {m}", file=sys.stderr))
        store.save_run(out)
    print(json.dumps({"key": out.key, "status": out.provenance["status"], "dir": store.run_dir(out.key),
                      "code_fingerprint": out.provenance.get("code_version"),
                      "manifest": os.path.join(store.run_dir(out.key), "manifest.json")}))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"run_{out.key}.zip"), "wb") as f:
            f.write(store.export_zip(out.key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
