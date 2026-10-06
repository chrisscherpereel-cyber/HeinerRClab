"""Comparison of the Adaptive rule's selection gates (heiner_abm.gates), with an ORACLE benchmark. PROPOSED EXTENSION.

Design
    Every gate runs on the same scenarios (all firms use the Adaptive rule, the same seeds: common random numbers), so
    production rules, recommendations and random shocks are identical and differences come from the selection
    decision and the evidence it acts on.

Researcher evaluation (never visible to agents)
    For every opportunity (x* != x) the researcher forks the market (engine.run_batch with horizon H = the judgement
    window W by default) and measures the realized advantage A = U(x*) - U(x_B) over H periods (continuation "default":
    the firm then holds its output while rivals follow their rules). This is the ex-post counterpart of the target
    A_t = E[U(x*) - U(x_B) | I_t].

ORACLE benchmark
    oracle_table() estimates the true mean advantage per firm and size bin from INDEPENDENT runs (seeds from
    ORACLE_SEED_BASE on, disjoint from the evaluation seeds, which must be below it). Pass 1 generates decisions with
    every firm adopting every recommendation ("Always"); pass 2 (default) re-estimates the table under the benchmark
    gate that uses the pass-1 table, so the table reflects the states the benchmark itself visits (one step of policy
    iteration; it is not shown to be optimal). Bins never observed in the independent runs get -inf: the benchmark keeps
    x_B there (no evidence is not treated as a zero gain). The table enters only the oracle_table gate's scenario;
    Scenario.validate rejects it for every other gate.

Metrics (Adaptive firms, recorded periods t >= burn-in; c = adjustment cost)
    adoption rate       adoptions / opportunities
    false adaptations   share of adoptions with realized A <= c (ex post; includes bad luck) and with oracle-table
                        value <= c (ex ante, by the benchmark's standard)
    missed opportunities share of opportunities with realized A > c that were not adopted, and the same with the
                        oracle table
    net payoff          mean market profit per period minus c times adoptions per period
    calibration         over decisions with a prediction: mean (prediction - realized A), slope of realized A on the
                        prediction, and the empirical share of decisions whose lower bound lies at or below the oracle
                        value of their bin (one-sided coverage; measured, not assumed)
    learning delay      periods until the share of the last `delay_window` opportunities on which the gate agrees with
                        the oracle benchmark's decision first reaches `delay_level` (NaN if never)
    after regime change with demand shifts: net payoff per period in the `shift_window` periods after each shift minus
                        the same before it, and the false-adaptation rate after shifts
Summaries are means over markets with normal 95% intervals (markets are independent; firms within a market are not).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .engine import run_batch
from .gates import GATE_LABELS, GATES
from .information import UnsupportedInformation
from .params import Scenario

ORACLE_SEED_BASE = 7_000_000          # independent seeds for the oracle benchmark; evaluation seeds must be below


def gate_scenario(base: Scenario, gate: str, seed: int, table=None, **settings) -> Scenario:
    """The base scenario with every firm using the Adaptive rule with `gate` (other settings: AdaptiveParams fields)."""
    s = base.copy(seed=int(seed))
    for f in s.firms:
        f.selection = "Adaptive"
    s.adaptive.gate = gate
    for k, v in settings.items():
        setattr(s.adaptive, k, v)
    s.adaptive.oracle_table = tuple(tuple(float(x) for x in row) for row in table) if gate == "oracle_table" else ()
    return s


def _bins(res, edges):
    q = res.firm_hist["q"]
    q_prev = np.concatenate([q[:, :1], q[:, :-1]], axis=1)
    return np.searchsorted(edges, np.abs(res.firm_hist["rec"] - q_prev), side="right")


def _bin_means(res, edges, burn):
    """Mean realized advantage per firm and bin over the opportunities of every market in `res`."""
    h = res.firm_hist
    opp = h["opportunity"].copy()
    opp[:, :burn] = False
    b = _bins(res, edges)
    N, nb = opp.shape[2], len(edges) + 1
    tab, cnt = np.full((N, nb), -np.inf), np.zeros((N, nb), int)
    for i in range(N):
        for k in range(nb):
            m = opp[:, :, i] & (b[:, :, i] == k)
            cnt[i, k] = int(m.sum())
            if cnt[i, k]:
                tab[i, k] = float(h["gain_full"][:, :, i][m].mean())
    return tab, cnt


def oracle_table(base: Scenario, runs: int = 8, horizon: Optional[int] = None, continuation: str = "default",
                 passes: int = 2, **settings) -> Dict:
    """ORACLE: true mean advantage per firm and size bin from `runs` independent markets (see the module docstring)."""
    H = int(horizon or base.adaptive.window)
    edges = np.asarray(base.adaptive.bin_edges, float)
    seeds = [ORACLE_SEED_BASE + k for k in range(runs)]
    always = []
    for sd in seeds:
        s = base.copy(seed=sd)
        for f in s.firms:
            f.selection = "Always"
        always.append(s)
    res = run_batch(always, record_firm_history=True, horizon=H, continuation=continuation)
    tab, cnt = _bin_means(res, edges, base.burn_in)
    for _ in range(passes - 1):
        scns = [gate_scenario(base, "oracle_table", sd, tab, **settings) for sd in seeds]
        res = run_batch(scns, record_firm_history=True, horizon=H, continuation=continuation)
        new, cnt2 = _bin_means(res, edges, base.burn_in)
        tab, cnt = np.where(cnt2 > 0, new, tab), np.where(cnt2 > 0, cnt2, cnt)     # keep pass-1 values where unseen
    return dict(table=tab, counts=cnt, seeds=seeds, horizon=H, continuation=continuation, passes=passes)


@dataclass
class GateStudy:
    summary: pd.DataFrame                 # one row per gate
    markets: pd.DataFrame                 # one row per gate x market (means over its Adaptive firms)
    calibration: pd.DataFrame             # binned predictions vs realized advantage, per gate
    after_shift: pd.DataFrame             # per gate: net payoff relative to the pre-shift window, by period after shift
    oracle: Dict
    not_run: Dict[str, str] = field(default_factory=dict)
    settings: Dict = field(default_factory=dict)


def _rolling_delay(agree: np.ndarray, t_idx: np.ndarray, window: int, level: float) -> float:
    if len(agree) < window:
        return float("nan")
    roll = np.convolve(agree.astype(float), np.ones(window) / window, mode="valid")
    hit = np.flatnonzero(roll >= level)
    return float(t_idx[hit[0] + window - 1]) if len(hit) else float("nan")


def _finite_mean(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def evaluate(res, scns: Sequence[Scenario], table: np.ndarray, cost: float, delay_window: int = 50,
             delay_level: float = 0.8, shift_window: int = 50):
    """Per-market metrics and pooled calibration data for one gate's batch (see the module docstring)."""
    h, burn, T = res.firm_hist, res.burn_in, res.price.shape[1]
    edges = np.asarray(scns[0].adaptive.bin_edges, float)
    b = _bins(res, edges)
    rows, calib = [], []
    for m, s in enumerate(scns):
        firm_rows = []
        for i in range(s.n_firms):
            sl = slice(burn, T)
            opp, dev = h["opportunity"][m, sl, i], h["deviate"][m, sl, i]
            adopt = opp & dev
            A = h["gain_full"][m, sl, i]
            tb = table[i, b[m, sl, i]]
            n_opp, n_ad = max(int(opp.sum()), 1), int(adopt.sum())
            good = opp & (A > cost)
            pred, bound = h["adv_pred"][m, sl, i], h["adv_bound"][m, sl, i]
            has = opp & np.isfinite(pred)
            t_opp = np.arange(burn, T)[opp]
            firm_rows.append(dict(
                opportunities=int(opp.sum()), adoption_rate=n_ad / n_opp,
                trial_share=float(h["explored"][m, sl, i][opp].mean()) if opp.any() else np.nan,
                false_adapt_realized=float((A[adopt] <= cost).mean()) if n_ad else np.nan,
                false_adapt_oracle=float((tb[adopt] <= cost).mean()) if n_ad else np.nan,
                missed_realized=float((~dev[good]).mean()) if good.any() else np.nan,
                missed_oracle=float((~dev[opp & (tb > cost)]).mean()) if (opp & (tb > cost)).any() else np.nan,
                net_payoff=float(h["profit"][m, sl, i].mean() - cost * adopt.sum() / (T - burn)),
                evidence=_finite_mean(h["adv_neff"][m, sl, i][opp]),
                share_predicted=float(has.sum() / n_opp),
                bias=float((pred[has] - A[has]).mean()) if has.any() else np.nan,
                coverage_lower=float((bound[has & np.isfinite(bound)] <= tb[has & np.isfinite(bound)]).mean())
                if (has & np.isfinite(bound)).any() else np.nan,
                learning_delay=_rolling_delay(dev[opp] == (tb[opp] > cost), t_opp, delay_window, delay_level),
            ))
            calib.append(pd.DataFrame({"pred": pred[has], "realized": A[has], "market": m, "firm": i}))
            shifts = [tau for tau in np.flatnonzero(res.shifts[m]) if tau - shift_window >= burn
                      and tau + shift_window <= T]
            if shifts:
                prof = h["profit"][m, :, i] - cost * (h["opportunity"][m, :, i] & h["deviate"][m, :, i])
                pre = np.mean([prof[tau - shift_window:tau].mean() for tau in shifts])
                post = np.mean([prof[tau:tau + shift_window].mean() for tau in shifts])
                fa = [((h["gain_full"][m, tau:tau + shift_window, i] <= cost)
                       & h["opportunity"][m, tau:tau + shift_window, i] & h["deviate"][m, tau:tau + shift_window, i]
                       ).sum() / max(1, (h["opportunity"][m, tau:tau + shift_window, i]
                                         & h["deviate"][m, tau:tau + shift_window, i]).sum()) for tau in shifts]
                firm_rows[-1].update(post_shift_change=float(post - pre), post_shift_false_adapt=float(np.mean(fa)),
                                     n_shifts=len(shifts))
        df = pd.DataFrame(firm_rows)
        rows.append({"market": m, "seed": s.seed, **df.mean(numeric_only=True).to_dict()})
    return pd.DataFrame(rows), pd.concat(calib, ignore_index=True) if calib else pd.DataFrame()


def _shift_profile(res, scns, cost, shift_window):
    """Net payoff by period relative to each shift (k = -shift_window .. shift_window - 1), minus the pre-shift mean."""
    h, burn, T = res.firm_hist, res.burn_in, res.price.shape[1]
    out = []
    for m in range(len(scns)):
        prof = (h["profit"][m] - cost * (h["opportunity"][m] & h["deviate"][m])).mean(axis=1)
        for tau in np.flatnonzero(res.shifts[m]):
            if tau - shift_window >= burn and tau + shift_window <= T:
                seg = prof[tau - shift_window:tau + shift_window]
                out.append(seg - seg[:shift_window].mean())
    if not out:
        return pd.DataFrame(columns=["k", "net_payoff_vs_pre", "n"])
    arr = np.array(out)
    return pd.DataFrame({"k": np.arange(-shift_window, shift_window), "net_payoff_vs_pre": arr.mean(axis=0),
                         "n": len(arr)})


METRICS = ("adoption_rate", "trial_share", "false_adapt_realized", "false_adapt_oracle", "missed_realized",
           "missed_oracle", "net_payoff", "evidence", "share_predicted", "bias", "coverage_lower", "learning_delay",
           "post_shift_change", "post_shift_false_adapt")


def compare_gates(base: Scenario, gates: Sequence[str] = GATES, reps: int = 8, seed0: int = 1,
                  oracle_runs: int = 8, oracle_passes: int = 2, horizon: Optional[int] = None,
                  continuation: str = "default", delay_window: int = 50, delay_level: float = 0.8,
                  shift_window: int = 50, progress=None, **settings) -> GateStudy:
    """Run every gate on the same `reps` markets (seeds seed0 ...) and the ORACLE benchmark on independent markets.
    `settings` are AdaptiveParams fields applied to every gate (adjust_cost, confidence, min_evidence, explore_rate,
    on_change, window, memory)."""
    seeds = list(range(int(seed0), int(seed0) + int(reps)))
    if max(seeds) >= ORACLE_SEED_BASE:
        raise ValueError(f"Evaluation seeds must be below {ORACLE_SEED_BASE} (reserved for the oracle benchmark).")
    probe = gate_scenario(base, "gain", seeds[0], **settings)
    H = int(horizon or probe.adaptive.window)
    cost = float(probe.adaptive.adjust_cost)
    note = progress or (lambda f, msg: None)
    note(0.0, "ORACLE benchmark (independent runs)")
    oracle = oracle_table(probe, oracle_runs, H, continuation, oracle_passes, **settings)
    markets, calib, shifts, not_run = [], [], [], {}
    for j, g in enumerate(gates):
        note((j + 1) / (len(gates) + 1), GATE_LABELS[g])
        scns = [gate_scenario(base, g, sd, oracle["table"], **settings) for sd in seeds]
        try:
            res = run_batch(scns, record_firm_history=True, horizon=H, continuation=continuation)
        except UnsupportedInformation as err:
            not_run[g] = str(err)
            continue
        mk, cal = evaluate(res, scns, oracle["table"], cost, delay_window, delay_level, shift_window)
        markets.append(mk.assign(gate=g))
        calib.append(cal.assign(gate=g))
        shifts.append(_shift_profile(res, scns, cost, shift_window).assign(gate=g))
    note(1.0, "Summarizing")
    mk = pd.concat(markets, ignore_index=True) if markets else pd.DataFrame()
    cal = pd.concat(calib, ignore_index=True) if calib else pd.DataFrame()
    summary = _summarize(mk, cal)
    return GateStudy(summary=summary, markets=mk, calibration=calibration_curve(cal), after_shift=pd.concat(
        shifts, ignore_index=True) if shifts else pd.DataFrame(), oracle=oracle, not_run=not_run,
        settings=dict(seeds=seeds, horizon=H, continuation=continuation, adjust_cost=cost, **{
            k: getattr(probe.adaptive, k) for k in ("window", "memory", "confidence", "min_evidence",
                                                    "explore_rate", "on_change")}, info=probe.info.to_dict()))


def _summarize(mk: pd.DataFrame, cal: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if mk.empty:
        return pd.DataFrame()
    rows = []
    ref = mk[mk["gate"] == "oracle_table"].set_index("seed")["net_payoff"] if "oracle_table" in set(mk["gate"]) else None
    for g, d in mk.groupby("gate", sort=False):
        r = {"gate": g, "label": GATE_LABELS[g], "markets": len(d)}
        for k in METRICS:
            if k in d:
                x = d[k].to_numpy(float)
                x = x[np.isfinite(x)]
                r[k] = float(x.mean()) if len(x) else np.nan
                r[k + "_ci"] = float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
        if ref is not None:
            diff = (d.set_index("seed")["net_payoff"] - ref).dropna().to_numpy(float)
            r["net_payoff_vs_oracle"] = float(diff.mean())
            r["net_payoff_vs_oracle_ci"] = float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 \
                else np.nan
        c = cal[cal["gate"] == g] if cal is not None and not cal.empty else None
        r["calibration_slope"] = (float(np.polyfit(c["pred"], c["realized"], 1)[0])
                                  if c is not None and len(c) > 2 and c["pred"].nunique() > 1 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def calibration_curve(cal: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Mean realized advantage by decile of the predicted advantage, per gate."""
    if cal.empty:
        return pd.DataFrame(columns=["gate", "bin", "pred", "realized", "n"])
    out = []
    for g, d in cal.groupby("gate", sort=False):
        if d["pred"].nunique() < 2:
            continue
        q = pd.qcut(d["pred"].rank(method="first"), min(n_bins, len(d)), labels=False)
        agg = d.groupby(q).agg(pred=("pred", "mean"), realized=("realized", "mean"), n=("pred", "size"))
        out.append(agg.reset_index(names="bin").assign(gate=g))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["gate", "bin", "pred", "realized", "n"])
