"""Empirical validation: loaders for the five public datasets, adapters and the evaluation rules.

The datasets themselves are not in the repository (their licenses govern redistribution). The loader tests build small
files in each original format; set HEINER_DATA to a folder with the downloaded archives to also test the real files.
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from heiner_abm import registered
from heiner_abm.calibration import prepare_quantities, synthetic_cournot_data
from heiner_abm.datasets import (GOS_MARKET, load_ah, load_bdp_newsvendor, load_egm, load_gos_cournot,
                                 load_time_pressure)
from heiner_abm.empirical import (EMPIRICAL_PLAN, NewsvendorSpec, cournot_moments, evaluate_empirical, nash_quantity,
                                  run_cournot, run_newsvendor, run_time_pressure, synthetic_newsvendor_data)


# ------------------------------------------------------------------------------------------------ adapters
def test_newsvendor_adapter_recovers_rules():
    spec = NewsvendorSpec(low=0, high=100, co=25, cu=75)
    assert spec.optimal == pytest.approx(75.0)
    df, truth = synthetic_newsvendor_data(spec, n_subjects=12, periods=60)
    res = run_newsvendor(df, spec)
    acc = (res["best"].merge(truth)["rule"] == res["best"].merge(truth)["true_rule"]).mean()
    assert acc > 0.5
    p, ch = res["stats"]["ptc"], res["stats"]["chase"]
    assert np.isfinite(p[0]) and np.isfinite(ch[0])


def test_differentiated_cournot():
    m = GOS_MARKET
    assert nash_quantity(m["a"], m["b"], m["c"], m["n"], m["gamma"]) == pytest.approx(37.0)
    df = pd.DataFrame(dict(group=0, subject=[0, 1, 2, 3] * 3, period=np.repeat([1, 2, 3], 4),
                           quantity=[30, 25, 40, 25] * 3))
    d = prepare_quantities(df, m["a"], m["b"], m["c"], m["gamma"])
    # P_i = 150 - q_i - (2/3) * rivals' output: for q = 30 with rivals 90, P = 60
    assert d.loc[d["quantity"] == 30, "price"].iloc[0] == pytest.approx(60.0)
    mom = cournot_moments(df, m["a"], m["b"], m["c"], m["n"], m["gamma"])
    assert mom["output_vs_nash"] == pytest.approx(120 / 148)


def test_generative_check_respects_choice_grid():
    df, _ = synthetic_cournot_data(n_groups=2, seed=3)
    df["quantity"] = np.round(df["quantity"])
    res = run_cournot(df, 100.0, 1.0, 1.0, 4, True, generative=True)
    assert len(res["sim"]) == EMPIRICAL_PLAN.gen_reps
    assert res["sim"]["change_rate"].between(0, 1).all()


def test_paired_time_pressure():
    from heiner_abm.calibration import synthetic_forecast_data
    d, _ = synthetic_forecast_data(n_groups=2)
    df = pd.concat([d.assign(pressure="high"), d.assign(pressure="low")], ignore_index=True)
    res = run_time_pressure(df, 1, "pressure", "high", "low")
    assert res["paired"] and res["diff"][0] == pytest.approx(0.0)


def test_evaluation_is_conjunctive_and_marks_missing_data():
    ok = dict(fits=pd.DataFrame(), gain_rc=dict(mean=1.0, lo=0.5, hi=1.5, n=10, share=0.8),
              modal=dict(rule="naive", share=0.3, lo=0.2, hi=0.4))
    bad = dict(ok, gain_rc=dict(mean=-1.0, lo=-1.5, hi=-0.5, n=10, share=0.2))
    v = evaluate_empirical({"forecast": {"a": ok, "b": bad}}).set_index("id")["verdict"]
    assert v["V3"] == "not supported" and v["V4"] == "supported" and v["V1"] == "not tested"
    v = evaluate_empirical({"forecast": {"a": ok}}).set_index("id")["verdict"]
    assert v["V3"] == "supported"


def test_empirical_plan_registered():
    assert EMPIRICAL_PLAN.digest == registered.EMPIRICAL_PLAN
    assert set(registered.EMPIRICAL_VERDICTS) == {h[0] for h in EMPIRICAL_PLAN.hypotheses}


# ------------------------------------------------------------------------------------------------ loaders (formats)
def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_gos_loader_reads_sql_dumps():
    rows, idx = [], 1
    q = {1: 30, 2: 25, 3: 40, 4: 25}
    for p in q:
        others = sum(q.values()) - q[p]
        price = 150 - q[p] - 2 / 3 * others
        rows.append(f"INSERT INTO `results` VALUES ({idx}, {p}, 1, 1, 1, {q[p]}, 10, '', 0, 0, 0, {others}, 0, 0, 0, "
                    f"{price:.2f}, {round(q[p] * (price - 2))});\n")
        idx += 1
    sql = ("CREATE TABLE `commonparameters` (\n  `index` int(11),\n  `name` text,\n  `value` text,\n  `comment` text\n"
           ") ENGINE=MyISAM;\nINSERT INTO `commonparameters` VALUES (1, 'treatment', '2', '');\n"
           "INSERT INTO `commonparameters` VALUES (2, 'session', '7', '');\n"
           "CREATE TABLE `results` (\n  `index` int(11),\n  `ppnr` int(11),\n  `groep` int(11),\n  `part` int(11),\n"
           "  `period` int(11),\n  `decision` int(11),\n  `time` int(11),\n  `calculator` text,\n  `prod1` int(11),\n"
           "  `prod2` int(11),\n  `prod3` int(11),\n  `productionothers` int(11),\n  `prof1` int(11),\n"
           "  `prof2` int(11),\n  `prof3` int(11),\n  `price` float,\n  `profit` int(11)\n) ENGINE=MyISAM;\n"
           + "".join(rows))
    df = load_gos_cournot([("data.zip", _zip({"sql files /fgmsession7.sql": sql}))])
    assert len(df) == 4 and df["demand_check"].all()
    assert set(df["treatment"]) == {"individual"} and df["subject"].iloc[0].startswith("s7-")


def test_egm_loader_reshapes_wide_data(tmp_path):
    d = pd.DataFrame(dict(participantcode=["a", "b"], market_id=[1, 1], treatment=[2, 2],
                          ind_fcast_1=[50.0, 60.0], mk_price_1=[55.0, 55.0],
                          ind_fcast_2=[52.0, 58.0], mk_price_2=[56.0, 56.0]))
    path = tmp_path / "combined_data.dta"
    d.to_stata(path, write_index=False)
    long = load_egm([("combined_data.dta", path.read_bytes())])
    assert len(long) == 4 and set(long["feedback"]) == {"negative"}
    assert long.loc[(long["subject"] == "b") & (long["period"] == 2), "forecast"].iloc[0] == 58.0


def test_ah_loader_reads_mat_groups(tmp_path):
    from scipy.io import savemat
    a = np.column_stack([np.linspace(50, 60, 10)] + [np.linspace(49, 61, 10)] * 6)
    path = tmp_path / "hstv05.mat"
    savemat(path, {"gr09": a, "fout": np.zeros((10, 1))})
    long = load_ah([("hstv05.mat", path.read_bytes())])
    assert long["subject"].nunique() == 6 and (long["fundamental"] == 40.0).all()


def test_bdp_loader_reads_workbook(tmp_path):
    d = pd.DataFrame({"Subject ": [1, 2], "Game": ["N", "P"], "Loss": [25, 75], "Periods": [2, 2],
                      "Choice1": [50, 40], "Choice2": [60, 45], "Realization1": [30, 70], "Realization2": [80, 20]})
    path = tmp_path / "s2.xlsx"
    d.to_excel(path, index=False)
    long = load_bdp_newsvendor([("s2.xlsx", path.read_bytes())])
    assert len(long) == 2 and (long["optimal"] == 75.0).all() and list(long["order"]) == [50.0, 60.0]


def test_time_pressure_loader_aligns_forecasts(tmp_path):
    row = {"participant.code": "x1"}
    for t in range(1, 5):
        row.update({f"long_run_high.{t}.group.id_in_subsession": 1, f"long_run_high.{t}.player.prediction": 50.0 + t,
                    f"long_run_high.{t}.group.spot_price": 60.0 + t})
    path = tmp_path / "rawdata_1908_01.xlsx"
    pd.DataFrame([row]).to_excel(path, index=False)
    long = load_time_pressure([("rawdata_1908_01.xlsx", path.read_bytes())])
    # the prediction made in round t targets the price of round t + 1
    assert long.loc[long["period"] == 3, "forecast"].iloc[0] == 52.0
    assert long.loc[long["period"] == 3, "price"].iloc[0] == 63.0


# ------------------------------------------------------------------------------------------------ real files
DATA = os.environ.get("HEINER_DATA")


@pytest.mark.skipif(not DATA, reason="set HEINER_DATA to a folder with the downloaded archives")
def test_real_datasets_load():
    import glob
    files = [(os.path.basename(p), open(p, "rb").read()) for p in glob.glob(os.path.join(DATA, "*.zip"))]
    c = load_gos_cournot(files)
    assert c["subject"].nunique() == 144 and c["demand_check"].all()
    assert load_egm(files)["subject"].nunique() == 372
    assert load_ah(files)["subject"].nunique() == 120
    assert load_bdp_newsvendor(files)["subject"].nunique() == 52
    assert load_time_pressure(files)["subject"].nunique() == 198
