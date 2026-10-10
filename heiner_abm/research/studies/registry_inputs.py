"""Default cross-run inputs: the recorded runs that the reported results come from (evidence/index.json)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from ..store import EVIDENCE_DIR, file_ref, find_run


def reported_run(study: str) -> Optional[str]:
    """Run id of the recorded run behind the reported results of a study, or None."""
    p = os.path.join(EVIDENCE_DIR, "index.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("reported", {}).get(study)


def tournament_input() -> Optional[Dict[str, Any]]:
    """The tuned designs and parameters of the reported tournament run, as a checksummed input reference."""
    rid = reported_run("tournament")
    if rid is None or find_run(rid) is None:
        return None
    return file_ref(rid, "tuning", "tuned.json")
