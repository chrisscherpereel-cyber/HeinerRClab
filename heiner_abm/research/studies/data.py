"""Studies on recorded human data: empirical validation (published datasets) and the laboratory's own experiment.

empirical   out-of-sample behavioral prediction. The data files are inputs: their paths, sizes and SHA-256 checksums
            are recorded, and every participant-period row used is saved, so the analysis is auditable without the
            original download. Rules are fitted on the first share of each participant's periods (EMPIRICAL_PLAN.split)
            and scored on the rest; the assignment is saved per participant.
human       causal experimental evidence: participants' files from Play the market under protocol 2.0, data quality
            and exclusions fixed in advance, randomized treatment effects and mechanism models.

Synthetic demonstration data can be used to exercise either study; such runs are labeled synthetic and the claim
registry never reports them as evidence about people.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..provenance import sha256_file
from ..store import Stage, Study
from .common import plan_to_dict, sources, split_outputs


def _read_files(ctx, paths: List[str]):
    out, recs = [], []
    for p in paths:
        with open(p, "rb") as f:
            data = f.read()
        out.append((os.path.basename(p), data))
        recs.append(dict(file=os.path.basename(p), bytes=len(data), sha256=sha256_file(p)))
    ctx.note("data_files", recs)
    return out


# ================================================================================================ empirical
def empirical_config() -> Dict[str, Any]:
    """Each dataset entry: label (as reported), kind (cournot | forecast | newsvendor | time), loader (datasets.LOADERS
    key) and files (paths to the files as downloaded), or synthetic: true for demonstration data; plus the selection
    and settings of that analysis. The defaults reproduce the reported analyses once the file paths are filled in."""
    from ...datasets import GOS_MARKET
    return dict(datasets=[
        dict(label="Cournot, aggregate information", kind="cournot", loader="cournot_gos", files=[],
             select=dict(part=1, treatment="aggregate"), market=dict(GOS_MARKET), individual_info=False,
             generative=True),
        dict(label="Cournot, individual information", kind="cournot", loader="cournot_gos", files=[],
             select=dict(part=1, treatment="individual"), market=dict(GOS_MARKET), individual_info=True,
             generative=True),
        dict(label="EGM, negative feedback", kind="forecast", loader="ltf_egm", files=[],
             select=dict(feedback="negative"), horizon=1),
        dict(label="EGM, positive feedback", kind="forecast", loader="ltf_egm", files=[],
             select=dict(feedback="positive"), horizon=1),
        dict(label="Anufriev–Hommes asset markets", kind="forecast", loader="ltf_ah", files=[], select={}, horizon=2),
        dict(label="Newsvendor, low cost (optimal 75)", kind="newsvendor", loader="newsvendor_bdp", files=[],
             select=dict(unit_cost=25.0), spec=dict(dist="uniform", low=0, high=100, co=25.0, cu=75.0)),
        dict(label="Newsvendor, high cost (optimal 25)", kind="newsvendor", loader="newsvendor_bdp", files=[],
             select=dict(unit_cost=75.0), spec=dict(dist="uniform", low=0, high=100, co=75.0, cu=25.0)),
        dict(label="Time pressure (high vs low)", kind="time", loader="ltf_time", files=[], select={}, horizon=2,
             high="high", low="low"),
    ])


def _empirical_stages(cfg):
    from ... import empirical as E
    from ...calibration import synthetic_cournot_data, synthetic_forecast_data
    from ...datasets import LOADERS
    SRC = sources(__file__, "empirical.py", "datasets.py")

    def _load(ctx, d):
        if d.get("synthetic"):
            if d["kind"] == "cournot":
                df, truth = synthetic_cournot_data(n_groups=int(d.get("n_groups", 6)))
            elif d["kind"] in ("forecast", "time"):
                df, truth = synthetic_forecast_data(n_groups=int(d.get("n_groups", 4)))
                if d["kind"] == "time":                     # as the page's synthetic demonstration
                    noise = np.random.default_rng(1).normal(0, 0.3, len(df))
                    df = pd.concat([df.assign(pressure="high"), df.assign(pressure="low", forecast=df["forecast"] + noise)],
                                   ignore_index=True)
            else:
                df, truth = E.synthetic_newsvendor_data(E.NewsvendorSpec(**d["spec"]),
                                                        n_subjects=int(d.get("n_subjects", 12)))
            return df
        if not d["files"]:
            raise FileNotFoundError(f"{d['label']}: no data files given (config datasets[].files)")
        raw = LOADERS[d["loader"]](_read_files(ctx, d["files"]))
        for k, v in d.get("select", {}).items():
            raw = raw[raw[k] == v]
        return raw

    def one(i, d):
        def inputs(ctx):
            df = _load(ctx, d)
            if "subject" not in df:
                df = df.assign(subject=0)
            ctx.note("synthetic", bool(d.get("synthetic")))
            per = df.groupby("subject")["period"].agg(["min", "max", "count"]).reset_index()
            plan = E.EMPIRICAL_PLAN
            per["fit_until_period"] = per["min"] + np.floor(plan.split * per["count"]).astype(int) - 1
            return dict(data=df.reset_index(drop=True), assignment=per)

        def compute(ctx):
            df = ctx.load(f"data_{i}", "data")
            if d["kind"] == "cournot":
                m = d["market"]
                res = E.run_cournot(df, m["a"], m["b"], m["c"], int(m["n"]), bool(d["individual_info"]),
                                    bool(d["generative"]), gamma=m.get("gamma", 1.0))
            elif d["kind"] == "forecast":
                res = E.run_forecast(df, int(d["horizon"]), d.get("fundamental"))
            elif d["kind"] == "newsvendor":
                res = E.run_newsvendor(df, E.NewsvendorSpec(**d["spec"]))
            else:
                res = E.run_time_pressure(df, int(d["horizon"]), "pressure", d["high"], d["low"])
            return split_outputs(res)
        return inputs, compute

    out = [Stage("plan", lambda ctx: dict(plan=plan_to_dict(E.EMPIRICAL_PLAN), digest=E.EMPIRICAL_PLAN.digest),
                 SRC, role="inputs", description="the frozen protocol")]
    for i, d in enumerate(cfg["datasets"]):
        inp, comp = one(i, d)
        out.append(Stage(f"data_{i}", inp, SRC, ("plan",), "splits",
                         f"{d['label']}: data, checksums and fit/test assignment (decision-level observations)"))
        out.append(Stage(f"fit_{i}", comp, SRC, (f"data_{i}",), description=f"{d['label']}: rules fitted and scored"))

    def analysis(ctx):
        store = {}
        for i, d in enumerate(ctx.config["datasets"]):
            res = {}
            rec = ctx.run.stage_record(f"fit_{i}")
            for fn in rec["outputs"]:
                if fn.endswith(".schema.json"):
                    continue
                name = fn.split(".")[0]
                val = ctx.load(f"fit_{i}", name)
                if name == "summary":
                    res.update(val)
                else:
                    res[name] = val
            res["synthetic"] = bool(d.get("synthetic"))
            store.setdefault(d["kind"] if d["kind"] != "time" else "time", {})[d["label"]] = res
        return dict(verdicts=E.evaluate_empirical(store), details=E.evaluate_details(store))

    out.append(Stage("analysis", analysis, SRC, tuple(f"fit_{i}" for i in range(len(cfg["datasets"]))), "analysis",
                     "frozen hypotheses V1-V7"))
    return out


EMPIRICAL = Study("empirical", "Empirical validation on public data", "behavioral",
                  "Rules fitted to part of each participant's choices predict the rest (published data).",
                  empirical_config, _empirical_stages)


# ================================================================================================ human experiment
def human_config() -> Dict[str, Any]:
    from ...experiment import PLAN
    return dict(files=[], synthetic=None, plan_digest=PLAN.digest)


def _human_stages(cfg):
    from ... import human_analysis as HA
    from ... import human_models as HM
    from ...experiment import PLAN, synthetic_pilot
    SRC = sources(__file__, "human_analysis.py", "human_models.py", "experiment.py")

    def data(ctx):
        if ctx.config.get("synthetic"):
            s = ctx.config["synthetic"]
            df, truth = synthetic_pilot(int(s["n"]), seed=int(s.get("seed", 1)))
            ctx.note("synthetic", True)
        else:
            if not ctx.config["files"]:
                raise FileNotFoundError("no participant files given (config files); no human data have been collected")
            frames = [pd.read_csv(p) for p in ctx.config["files"]]
            _read_files(ctx, ctx.config["files"])
            df = pd.concat(frames, ignore_index=True)
            if set(df["plan"].astype(str)) != {PLAN.digest}:
                raise ValueError("files recorded under a different protocol hash")
            ctx.note("synthetic", bool(df.get("synthetic", pd.Series([False])).astype(bool).any()))
        q = HA.quality(df)
        return dict(decisions=df, quality=q)

    def analysis(ctx):
        df = ctx.load("data", "decisions")
        q = ctx.load("data", "quality")
        keep = set(q.loc[~q["excluded"].astype(bool), "participant"])
        df = df[df["participant"].isin(keep)]
        summ = HA.block_summary(df)
        w = pd.concat([HA.within_effects(summ, o).assign(outcome=o) for o in HA.OUTCOMES], ignore_index=True)
        b = pd.concat([HA.between_effects(summ, o).assign(outcome=o) for o in HA.OUTCOMES], ignore_index=True)
        data_ = HM.prepare(df)
        comp, fits, _ = HM.compare_models(data_)
        return dict(block_summary=summ, within_effects=w, between_effects=b, associational=HA.associational(summ),
                    model_comparison=comp, mechanism_effects=HA.mechanism_effects(fits["full"]))

    return [Stage("data", data, SRC, role="splits", description="participants' decisions, checksums, exclusions"),
            Stage("analysis", analysis, SRC, ("data",), "analysis", "randomized effects and mechanism models")]


HUMAN = Study("human", "Human experiment (Play the market, protocol 2.0)", "causal",
              "Randomized conditions and their effect on participants' adjustment.", human_config, _human_stages)
