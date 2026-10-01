"""Loaders for the five public datasets, from the files as distributed by their repositories.

Each loader returns a long table with one row per participant and period in the format the validation adapters expect
(heiner_abm.empirical), plus the market parameters needed to analyse it. The mappings were derived from the files
themselves and are checked when loading:

* Gomez-Martinez, Onderstal & Sonnemans (2016), Mendeley Data: phpMyAdmin SQL dumps, one per session ("sql files
  .zip" inside the archive). Four firms per market. Each firm's price is P_i = 150 − q_i − (2/3)·Q_−i and the unit cost
  is 2 (differentiated Cournot: Nash output 37, joint-profit output ≈ 25, competitive output ≈ 49.3); this reproduces
  every price and profit in the files. Part 1 (periods 1–25) has no communication. Treatment 1 = aggregate
  information about rivals, 2 = individual information.
* Evans, Gibbs & McGough (2025), openICPSR 198204: combined_data.dta, one row per participant with ind_fcast_t (the
  forecast of period t's price) and mk_price_t for t = 1..50. Markets of six (market_id). Treatments 7 and 8 have
  positive feedback, 1–6 negative feedback; announced structural changes take effect around period 20–21
  (treatments 2, 3, 5, 6, 7, 8) and again around period 45 (3, 6, 8).
* Anufriev & Hommes (2012), openICPSR 114401: hstv05.mat (groups gr01–gr14, 51 periods) and hstv08.mat (norobgr1–6,
  50 periods). Column 0 is the price, columns 1–6 the six participants' forecasts. Row t holds the forecasts of p_t,
  made when p_{t−2} was the latest known price (the pricing equation fits only with this alignment), so the forecast
  horizon is 2. Fundamental price 60 (40 in groups 8–10 of HSTV05).
* Brokesova, Deck & Peliova (2022), PLOS ONE S2: one row per participant, Choice1..100 (order), Realization1..100
  (demand), Game (N = newsvendor, P = price gouging), Loss (unit cost 25 or 75 at a price of 100; demand uniform on
  0..100, so the optimal order is 75 or 25).
* Time-pressure learning-to-forecast experiments (University of Amsterdam figshare 13948409): oTree exports, one per
  session, with a high and a low time-pressure part for the same participants. In round t, player.prediction is the
  forecast of the next round's price (group.spot_price), made when the latest known price was this round's
  predecessor, so the horizon is 2. Rounds without a prediction (time ran out) are dropped.
"""
from __future__ import annotations

import ast
import io
import re
import zipfile
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# ================================================================================================ Cournot (GOS 2016)
GOS_MARKET = dict(a=150.0, b=1.0, gamma=2.0 / 3.0, c=2.0, n=4)


def _sql_table(text: str, name: str) -> pd.DataFrame:
    cols = re.search(r"CREATE TABLE `%s` \((.*?)\) ENGINE" % name, text, re.S)
    if cols is None:
        return pd.DataFrame()
    names = re.findall(r"^\s*`(\w+)`", cols.group(1), re.M)
    rows = [ast.literal_eval("(" + m + ")") for m in re.findall(r"INSERT INTO `%s` VALUES \((.*?)\);\n" % name, text)]
    return pd.DataFrame(rows, columns=names)


def _iter_files(data: bytes, name: str) -> Iterable[Tuple[str, bytes]]:
    """Files inside an upload (recursively through zip archives), skipping macOS metadata."""
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                if info.is_dir() or "__MACOSX" in info.filename or info.filename.split("/")[-1].startswith("."):
                    continue
                yield from _iter_files(z.read(info), info.filename)
    else:
        yield name, data


def load_gos_cournot(files: List[Tuple[str, bytes]]) -> pd.DataFrame:
    """Session SQL dumps (or the archive containing them) → group, subject, period, quantity, treatment, part."""
    frames = []
    for name, data in files:
        for fname, raw in _iter_files(data, name):
            if not fname.lower().endswith(".sql"):
                continue
            text = raw.decode("latin1")
            res, par = _sql_table(text, "results"), _sql_table(text, "commonparameters")
            if res.empty:
                continue
            p = dict(zip(par["name"], par["value"])) if not par.empty else {}
            ses = int(p.get("session", len(frames) + 1))
            treat = {"1": "aggregate", "2": "individual"}.get(str(p.get("treatment")), str(p.get("treatment")))
            frames.append(pd.DataFrame(dict(
                group=[f"s{ses}-m{g}" for g in res["groep"]], subject=[f"s{ses}-p{s}" for s in res["ppnr"]],
                period=res["period"].astype(int), part=res["part"].astype(int),
                quantity=res["decision"].astype(float), treatment=treat,
                price_reported=res["price"].astype(float), profit_reported=res["profit"].astype(float),
                others=res["productionothers"].astype(float))))
    if not frames:
        raise ValueError("No session SQL dumps with a `results` table were found.")
    df = pd.concat(frames, ignore_index=True)
    m = GOS_MARKET
    model_price = np.maximum(0.0, m["a"] - m["b"] * (df["quantity"] + m["gamma"] * df["others"]))
    df["demand_check"] = np.abs(model_price - df["price_reported"]) < 0.02
    return df


# ================================================================================================ EGM (2025)
EGM_BREAKS = {2: (20,), 3: (20, 45), 5: (21,), 6: (21, 45), 7: (20,), 8: (20, 45)}


def load_egm(files: List[Tuple[str, bytes]]) -> pd.DataFrame:
    """combined_data.dta → group, subject, period, price, forecast, treatment, feedback."""
    for name, data in files:
        for fname, raw in _iter_files(data, name):
            if fname.split("/")[-1].lower() == "combined_data.dta":
                d = pd.read_stata(io.BytesIO(raw), convert_categoricals=False)
                periods = sorted(int(c.rsplit("_", 1)[1]) for c in d.columns if c.startswith("ind_fcast_"))
                long = pd.concat([pd.DataFrame(dict(
                    group=d["market_id"].astype(int).astype(str), subject=d["participantcode"].astype(str), period=t,
                    forecast=d[f"ind_fcast_{t}"].astype(float), price=d[f"mk_price_{t}"].astype(float),
                    treatment=d["treatment"].astype(int))) for t in periods], ignore_index=True)
                long["feedback"] = np.where(long["treatment"] >= 7, "positive", "negative")
                return long
    raise ValueError("combined_data.dta was not found.")


# ================================================================================================ Anufriev & Hommes
def load_ah(files: List[Tuple[str, bytes]]) -> pd.DataFrame:
    """hstv05.mat and hstv08.mat → group, subject, period, price, forecast, experiment, fundamental."""
    from scipy.io import loadmat
    frames = []
    for name, data in files:
        for fname, raw in _iter_files(data, name):
            base = fname.split("/")[-1].lower()
            if not base.endswith(".mat") or not base.startswith("hstv"):
                continue
            m = loadmat(io.BytesIO(raw))
            for k, v in m.items():
                if not re.match(r"(gr\d+|norobgr\d+)$", k):
                    continue
                exp = "HSTV05" if base.startswith("hstv05") else "HSTV08"
                grp = int(re.findall(r"\d+", k)[0])
                fund = 40.0 if exp == "HSTV05" and 8 <= grp <= 10 else 60.0
                T = v.shape[0]
                for j in range(1, v.shape[1]):
                    frames.append(pd.DataFrame(dict(group=f"{exp}-{k}", subject=f"{exp}-{k}-{j}",
                                                    period=np.arange(1, T + 1), price=v[:, 0], forecast=v[:, j],
                                                    experiment=exp, fundamental=fund)))
    if not frames:
        raise ValueError("No hstv05.mat / hstv08.mat group arrays were found.")
    return pd.concat(frames, ignore_index=True)


# ================================================================================================ newsvendor (BDP 2022)
def load_bdp_newsvendor(files: List[Tuple[str, bytes]], newsvendor_only: bool = True) -> pd.DataFrame:
    """pone.0264183.s002.xlsx → subject, period, order, demand, game, unit_cost, optimal."""
    for name, data in files:
        for fname, raw in _iter_files(data, name):
            if not fname.lower().endswith((".xlsx", ".xls")):
                continue
            d = pd.read_excel(io.BytesIO(raw))
            d.columns = [str(c).strip() for c in d.columns]
            if "Choice1" not in d or "Realization1" not in d:
                continue
            T = max(int(c[6:]) for c in d.columns if re.match(r"Choice\d+$", c))
            rows = []
            for _, r in d.iterrows():
                for t in range(1, T + 1):
                    rows.append(dict(subject=str(r["Subject"]), period=t, order=float(r[f"Choice{t}"]),
                                     demand=float(r[f"Realization{t}"]), game=str(r["Game"]),
                                     unit_cost=float(r["Loss"])))
            long = pd.DataFrame(rows)
            long["optimal"] = 100.0 * (100.0 - long["unit_cost"]) / 100.0
            if newsvendor_only:
                long = long[long["game"] == "N"]
            return long.reset_index(drop=True)
    raise ValueError("The raw-data spreadsheet (Choice1..., Realization1...) was not found.")


# ================================================================================================ time pressure
def load_time_pressure(files: List[Tuple[str, bytes]]) -> pd.DataFrame:
    """oTree session exports → group, subject, period, price, forecast, pressure ('high'/'low')."""
    frames = []
    for name, data in files:
        for fname, raw in _iter_files(data, name):
            if not fname.lower().endswith((".xlsx", ".xls", ".csv")):
                continue
            d = (pd.read_csv(io.BytesIO(raw)) if fname.lower().endswith(".csv")
                 else pd.read_excel(io.BytesIO(raw)))
            ses = re.sub(r"\.\w+$", "", fname.split("/")[-1])
            for part in ("high", "low"):
                app = f"long_run_{part}"
                cols = [c for c in d.columns if str(c).startswith(app + ".")]
                if not cols:
                    continue
                N = max(int(str(c).split(".")[1]) for c in cols)
                for _, r in d.iterrows():
                    rows = []
                    for t in range(1, N + 1):
                        g = r.get(f"{app}.{t}.group.id_in_subsession")
                        if pd.isna(g):
                            continue
                        rows.append(dict(round=t, group=f"{ses}-{part}-{int(g)}",
                                         pred=r.get(f"{app}.{t}.player.prediction"),
                                         spot=r.get(f"{app}.{t}.group.spot_price")))
                    if not rows:
                        continue
                    x = pd.DataFrame(rows).sort_values("round")
                    # the forecast submitted in round t is for the price of round t + 1
                    x["forecast"] = x["pred"].shift(1)
                    frames.append(pd.DataFrame(dict(group=x["group"], subject=f"{ses}-{r['participant.code']}",
                                                    period=x["round"], price=x["spot"].astype(float),
                                                    forecast=pd.to_numeric(x["forecast"], errors="coerce"),
                                                    pressure=part)))
    if not frames:
        raise ValueError("No oTree exports with long_run_high / long_run_low parts were found.")
    return pd.concat(frames, ignore_index=True)


LOADERS = {"cournot_gos": load_gos_cournot, "ltf_egm": load_egm, "ltf_ah": load_ah,
           "newsvendor_bdp": load_bdp_newsvendor, "ltf_time": load_time_pressure}
