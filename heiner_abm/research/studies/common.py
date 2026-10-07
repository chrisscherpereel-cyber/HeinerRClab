"""Helpers shared by the study adapters."""
from __future__ import annotations

import contextlib
import dataclasses
import os
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from ..provenance import module_closure

HERE = os.path.relpath(os.path.dirname(os.path.abspath(__file__)),
                       os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace(os.sep, "/")


def sources(adapter_file: str, *modules: str) -> Tuple[str, ...]:
    """Fingerprinted files of a stage: the adapter module itself, the listed heiner_abm modules and everything they
    import (heiner_abm.research.provenance.module_closure)."""
    adapter = f"{HERE}/{os.path.basename(adapter_file)}"
    return tuple(sorted(set(module_closure([adapter, *modules])) | {adapter}))


def tuned_from(obj: Dict[str, Any]):
    from ...arena import Tuned
    return Tuned(design=dict(obj["design"]), params={k: dict(v) for k, v in obj["params"].items()})


@contextlib.contextmanager
def tuned_override(params: Dict[str, Dict[str, float]]):
    """Make modules that read the registered tuned parameters (special.TUNED_PARAMS) use the parameters recorded by a
    tournament run instead, for the duration of a stage."""
    from ... import special
    old = special.TUNED_PARAMS
    special.TUNED_PARAMS = params
    try:
        yield
    finally:
        special.TUNED_PARAMS = old


def plain(obj: Any) -> Any:
    """JSON-ready form of dataclasses, numpy values and nested containers (DataFrames are left as they are)."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def split_outputs(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Stage outputs from a result object: every DataFrame becomes its own table; everything else is collected in
    one JSON object named `<prefix>summary` (or `summary`)."""
    out, rest = {}, {}
    if dataclasses.is_dataclass(obj):
        items = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
    elif isinstance(obj, dict):
        items = list(obj.items())
    else:
        items = [("value", obj)]
    for k, v in items:
        if isinstance(v, pd.DataFrame):
            out[f"{prefix}{k}"] = v
        elif isinstance(v, dict) and v and all(isinstance(x, pd.DataFrame) for x in v.values()):
            for kk, df in v.items():
                out[f"{prefix}{k}__{kk}"] = df
        else:
            rest[k] = plain(v)
    if rest:
        out[f"{prefix}summary"] = rest
    return out


def env_table(envs: Iterable, role: str) -> pd.DataFrame:
    return pd.DataFrame([dict(role=role, index=i, **dataclasses.asdict(e)) for i, e in enumerate(envs)])


def _tupled(v: Any) -> Any:
    return tuple(_tupled(x) for x in v) if isinstance(v, list) else v


def plan_to_dict(plan: Any) -> Dict[str, Any]:
    return plain(plan)


def plan_from_dict(cls: type, d: Dict[str, Any]) -> Any:
    """Rebuild a frozen plan dataclass from its JSON form (lists back to tuples, nested dataclasses rebuilt); the
    caller checks the digest."""
    import typing
    hints = typing.get_type_hints(cls)
    kw = {}
    for f in dataclasses.fields(cls):
        if not f.init or f.name not in d:
            continue
        t, v = hints.get(f.name), d[f.name]
        kw[f.name] = plan_from_dict(t, v) if dataclasses.is_dataclass(t) and isinstance(v, dict) else _tupled(v)
    return cls(**kw)
