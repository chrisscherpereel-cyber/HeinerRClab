"""The studies the research runner can execute (heiner_abm.research.store.Study), keyed by name."""
from __future__ import annotations

from typing import Dict

from ..store import Study
from . import choice, data, frozen, market, mechanisms, patterns, signature, tournament

STUDIES: Dict[str, Study] = {s.key: s for s in (
    tournament.STUDY, mechanisms.STUDY, choice.STUDY, signature.STUDY, patterns.STUDY, market.DIRECTIONAL,
    market.HORSE_RACE, market.PRESETS, *frozen.STUDIES, data.EMPIRICAL, data.HUMAN)}
