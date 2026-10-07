"""Inference, data quality, simulation-based power analysis and the prospective analysis plan for the human experiment.

Dependence structure
    Decisions within a block are summarized per participant and block (adjustment rate, adjustment magnitude, relative
    profit), so serial dependence within a block does not enter the standard errors. Each participant contributes four
    blocks (participant dependence: participant fixed effects in within-participant contrasts, clustering by
    participant). Each block uses one of four exogenous paths, shared across participants (path dependence: path fixed
    effects; four paths are too few clusters to cluster on). Participants play against agent rivals in their own
    market and never interact with each other; when sessions are recorded, standard errors are clustered by session
    instead of participant (sessions nest participants). Between-participant (randomized) contrasts use one value per
    participant with heteroskedasticity-robust (HC1) or session-clustered errors.

Causal status
    E1-E3 (condition contrasts) are randomized within participants (order counterbalanced, paths rotated): causal
    effects of the treatment conditions on the measured behavior. E4-E5 are randomized between participants: causal
    effects of the decision aid and of belief elicitation. A1 (adjustment frequency and profit) is associational:
    adjustment frequency is chosen by participants, not assigned. The aid assignment shifts adjustment frequency, but it
    also changes timing and magnitude, so it is not a valid instrument for the effect of frequency alone; no
    instrumental-variable estimate is reported.

Power
    Effect sizes and variance components come from a pilot (human pilot data, or a SYNTHETIC pilot labeled as such).
    Datasets are simulated with the same allocation schedule, paths and dependence structure and analyzed with the same
    estimators as the main analysis; power is the share of simulated datasets whose 95% interval excludes zero.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .experiment import (CONDITION_LABELS, CONDITIONS, EXCLUSIONS, MECHANISM, PLAN, WILLIAMS, ExperimentPlan,
                         allocation_schedule)

CONTRASTS = ("V", "N", "R")
OUTCOMES = {"adjust_rate": "Adjustment probability (share of periods with a change)",
            "magnitude": "Adjustment magnitude (mean |change| when adjusting)",
            "relative_profit": "Relative profit (own − rivals' mean, per period)"}


# ================================================================================================ data quality
def quality(df: pd.DataFrame, plan: ExperimentPlan = PLAN) -> pd.DataFrame:
    """Per participant: comprehension, completion and response-time checks, and the exclusion decision."""
    main = df[~df["practice"].astype(bool)]
    rows = []
    for pid, g in main.groupby("participant"):
        comp = bool(g["comprehension_passed"].iloc[0]) if "comprehension_passed" in g else True
        per_block = g.groupby("block_index").size()
        complete = bool(len(per_block) == len(plan.conditions) and per_block.min() >= 0.75 * plan.periods_per_block)
        med_rt = g.groupby("block_index")["rt_decision"].median() if "rt_decision" in g else pd.Series([np.nan])
        fast = bool((med_rt < 1.0).any())
        reasons = [r for r, bad in zip(EXCLUSIONS, (not comp, not complete, fast)) if bad]
        rows.append(dict(participant=pid, comprehension_passed=comp,
                         comprehension_attempts=g["comprehension_attempts"].iloc[0]
                         if "comprehension_attempts" in g else np.nan, blocks=len(per_block),
                         min_decisions_per_block=int(per_block.min()), median_rt=float(g["rt_decision"].median())
                         if "rt_decision" in g else np.nan, excluded=bool(reasons), reasons="; ".join(reasons),
                         synthetic=bool(g["synthetic"].any()) if "synthetic" in g else False))
    return pd.DataFrame(rows)


def block_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per participant and main block."""
    main = df[~df["practice"].astype(bool)].copy()
    main["absdq"] = (main["q"] - main["q_prev"]).abs()
    main["rel"] = main["profit"] - main["rivals_mean_profit"]
    if "session" not in main:
        main["session"] = ""
    g = main.groupby(["participant", "block_index"]).agg(
        condition=("condition", "first"), path=("path", "first"), aid=("aid", "first"), elicit=("elicit", "first"),
        session=("session", "first"), williams_row=("williams_row", "first"), adjust_rate=("deviated", "mean"),
        relative_profit=("rel", "mean"), profit=("profit", "mean"), decisions=("q", "size")).reset_index()
    mag = main[main["deviated"].astype(bool)].groupby(["participant", "block_index"])["absdq"].mean()
    g = g.merge(mag.rename("magnitude").reset_index(), on=["participant", "block_index"], how="left")
    return g


# ================================================================================================ estimators
def _ols_cluster(y: np.ndarray, X: np.ndarray, groups: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    """OLS with cluster-robust (CR1) or HC1 covariance. Returns coefficients, covariance and degrees of freedom."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n, k = X.shape
    if groups is None:
        meat = (X * e[:, None] ** 2).T @ X
        V = XtX_inv @ meat @ XtX_inv * n / max(n - k, 1)
        return b, V, max(n - k, 1)
    G = pd.factorize(groups)[0]
    nG = G.max() + 1
    S = np.zeros((nG, k))
    np.add.at(S, G, X * e[:, None])
    V = XtX_inv @ (S.T @ S) @ XtX_inv * (nG / max(nG - 1, 1)) * ((n - 1) / max(n - k, 1))
    return b, V, max(nG - 1, 1)


def _ci(b, se, dof, alpha=0.05):
    t = stats.t.ppf(1 - alpha / 2, dof)
    return b - t * se, b + t * se, float(2 * stats.t.sf(abs(b / se), dof)) if se > 0 else np.nan


def within_effects(summ: pd.DataFrame, outcome: str, alpha: float = 0.05) -> pd.DataFrame:
    """Condition contrasts (V, N, R versus B) with participant, path and position fixed effects; clustered by session
    when sessions are recorded, else by participant."""
    d = summ.dropna(subset=[outcome]).copy()
    if d.empty or d["condition"].nunique() < 2:
        return pd.DataFrame()
    y = d[outcome].to_numpy(float)
    cols, names = [], []
    for c in CONTRASTS:
        if (d["condition"] == c).any():
            cols.append((d["condition"] == c).to_numpy(float))
            names.append(c)
    fe = pd.get_dummies(d["participant"], drop_first=False).to_numpy(float)
    for key in ("path", "block_index"):
        dm = pd.get_dummies(d[key].astype(str), drop_first=True).to_numpy(float)
        if dm.shape[1]:
            fe = np.column_stack([fe, dm])
    X = np.column_stack(cols + [fe])
    C = np.column_stack(cols)
    if np.linalg.matrix_rank(X) < np.linalg.matrix_rank(fe) + C.shape[1]:
        raise ValueError("Condition contrasts are not identified: conditions are confounded with paths, positions or "
                         "participants in these data (check the allocation).")
    sess = d["session"].astype(str)
    groups = sess.to_numpy() if (sess != "").all() and sess.nunique() > 1 else d["participant"].to_numpy()
    b, V, dof = _ols_cluster(y, X, groups)
    rows = []
    for i, c in enumerate(names):
        se = float(np.sqrt(max(V[i, i], 0.0)))
        lo, hi, p = _ci(b[i], se, dof, alpha)
        rows.append(dict(contrast=f"{c} − B", mechanism=MECHANISM[c], outcome=outcome, effect=float(b[i]), se=se,
                         lo=lo, hi=hi, p=p, participants=d["participant"].nunique(),
                         clusters=len(np.unique(groups)), causal="causal (randomized within participants)"))
    return pd.DataFrame(rows)


def participant_means(summ: pd.DataFrame) -> pd.DataFrame:
    return summ.groupby("participant").agg(aid=("aid", "first"), elicit=("elicit", "first"),
                                           session=("session", "first"), williams_row=("williams_row", "first"),
                                           adjust_rate=("adjust_rate", "mean"), magnitude=("magnitude", "mean"),
                                           relative_profit=("relative_profit", "mean")).reset_index()


def between_effects(summ: pd.DataFrame, outcome: str, alpha: float = 0.05) -> pd.DataFrame:
    """Randomized between-participant effects of the aid and of belief elicitation (participant means)."""
    pm = participant_means(summ).dropna(subset=[outcome])
    if len(pm) < 4 or pm["aid"].nunique() < 2 and pm["elicit"].nunique() < 2:
        return pd.DataFrame()
    y = pm[outcome].to_numpy(float)
    X = np.column_stack([np.ones(len(pm)), pm["aid"].astype(float), pm["elicit"].astype(float)])
    sess = pm["session"].astype(str)
    groups = sess.to_numpy() if (sess != "").all() and sess.nunique() > 3 else None
    b, V, dof = _ols_cluster(y, X, groups)
    rows = []
    for i, (name, lab) in enumerate((("aid", "Decision aid (vs unaided control)"),
                                     ("elicit", "Belief elicitation (vs none)")), start=1):
        se = float(np.sqrt(max(V[i, i], 0.0)))
        lo, hi, p = _ci(b[i], se, dof, alpha)
        rows.append(dict(contrast=lab, outcome=outcome, effect=float(b[i]), se=se, lo=lo, hi=hi, p=p,
                         participants=len(pm), causal="causal (randomized between participants)"))
    return pd.DataFrame(rows)


def associational(summ: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """A1: across participants, relative profit on adjustment rate, overall and per condition. ASSOCIATIONAL."""
    rows = []
    for cond in (None,) + tuple(CONDITIONS):
        d = summ if cond is None else summ[summ["condition"] == cond]
        pm = d.groupby("participant")[["adjust_rate", "relative_profit"]].mean().dropna()
        if len(pm) < 4 or pm["adjust_rate"].std() == 0:
            continue
        X = np.column_stack([np.ones(len(pm)), pm["adjust_rate"]])
        b, V, dof = _ols_cluster(pm["relative_profit"].to_numpy(float), X, None)
        se = float(np.sqrt(V[1, 1]))
        lo, hi, p = _ci(b[1], se, dof, alpha)
        rows.append(dict(condition="all" if cond is None else cond, slope=float(b[1]), lo=lo, hi=hi, p=p,
                         participants=len(pm), status="ASSOCIATIONAL (adjustment frequency is not randomized)"))
    return pd.DataFrame(rows)


def mechanism_effects(fit, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Population means of the fitted condition shifts on adjustment probability (logit scale) and of the other
    parameters, with participant-bootstrap 95% intervals (estimation noise of individual fits not propagated)."""
    rng = np.random.default_rng(seed)
    p = fit.params
    rows = []
    for name in [c for c in p.columns if c not in ("participant", "aid")]:
        v = p[name].dropna().to_numpy(float)
        if name == "follow":
            v = p.loc[p["aid"], name].dropna().to_numpy(float)
        if len(v) < 2:
            continue
        bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
        rows.append(dict(parameter=name, mean=float(v.mean()), lo=float(np.percentile(bs, 2.5)),
                         hi=float(np.percentile(bs, 97.5)), sd_across_participants=float(v.std(ddof=1)), n=len(v)))
    return pd.DataFrame(rows)


# ================================================================================================ power
def pilot_components(summ: pd.DataFrame, outcome: str) -> Dict[str, float]:
    """Effect sizes and variance components from pilot block summaries (method of moments on a fixed-effects fit)."""
    d = summ.dropna(subset=[outcome]).copy()
    y = d[outcome].to_numpy(float)
    P = pd.get_dummies(d["participant"]).to_numpy(float)
    C = np.column_stack([(d["condition"] == c).to_numpy(float) for c in CONTRASTS])
    Pa = pd.get_dummies(d["path"].astype(str), drop_first=True).to_numpy(float)
    X = np.column_stack([C, P, Pa])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k = X.shape
    s2e = float(resid @ resid / max(n - np.linalg.matrix_rank(X), 1))
    nb = d.groupby("participant").size().mean()
    pfx = b[3:3 + P.shape[1]]
    s2p = max(float(np.var(pfx, ddof=1)) - s2e / nb, 0.0) if len(pfx) > 1 else 0.0
    pafx = np.concatenate([[0.0], b[3 + P.shape[1]:]])
    s2path = max(float(np.var(pafx, ddof=1)) - s2e / max(len(d) / max(len(pafx), 1), 1), 0.0) if len(pafx) > 1 \
        else 0.0
    pm = participant_means(d).dropna(subset=[outcome])
    aid_eff = float(pm.loc[pm["aid"], outcome].mean() - pm.loc[~pm["aid"], outcome].mean()) \
        if pm["aid"].nunique() == 2 else 0.0
    el_eff = float(pm.loc[pm["elicit"], outcome].mean() - pm.loc[~pm["elicit"], outcome].mean()) \
        if pm["elicit"].nunique() == 2 else 0.0
    return dict(outcome=outcome, mean=float(d.loc[d["condition"] == "B", outcome].mean()),
                V=float(b[0]), N=float(b[1]), R=float(b[2]), aid=aid_eff, elicit=el_eff,
                sd_participant=float(np.sqrt(s2p)), sd_path=float(np.sqrt(s2path)), sd_residual=float(np.sqrt(s2e)),
                pilot_participants=int(d["participant"].nunique()))


def simulate_summary(comp: Dict[str, float], n: int, rng: np.random.Generator,
                     plan: ExperimentPlan = PLAN) -> pd.DataFrame:
    """Block summaries for n participants under the allocation schedule, with the pilot's dependence structure."""
    paths = rng.normal(0, comp["sd_path"], plan.n_paths)
    rows = []
    for a in allocation_schedule(n, plan):
        u = rng.normal(0, comp["sd_participant"])
        for bi, c in enumerate(a.order(plan)):
            pth = a.path(c, plan)
            y = (comp["mean"] + comp.get(c, 0.0) + comp["aid"] * a.aid + comp["elicit"] * a.elicit + u + paths[pth]
                 + rng.normal(0, comp["sd_residual"]))
            rows.append(dict(participant=f"p{a.slot}", block_index=bi, condition=c, path=pth, aid=a.aid,
                             elicit=a.elicit, session="", williams_row=a.williams_row, **{comp["outcome"]: y},
                             **{k: np.nan for k in OUTCOMES if k != comp["outcome"]}))
    return pd.DataFrame(rows)


def power_analysis(comp: Dict[str, float], n_grid: Sequence[int] = (16, 32, 48, 64, 96, 128, 192, 256),
                   reps: int = 200, seed: int = 20261008, alpha: float = 0.05, plan: ExperimentPlan = PLAN
                   ) -> pd.DataFrame:
    """Share of simulated datasets whose 95% interval excludes zero, per effect and sample size (reproducible)."""
    rng = np.random.default_rng(seed)
    out = comp["outcome"]
    rows = []
    for n in n_grid:
        hits = {k: 0 for k in ("V − B", "N − B", "R − B", "Decision aid (vs unaided control)",
                                "Belief elicitation (vs none)")}
        for _ in range(reps):
            s = simulate_summary(comp, n, rng, plan)
            w = within_effects(s, out, alpha)
            bt = between_effects(s, out, alpha)
            for _, r in pd.concat([w, bt], ignore_index=True).iterrows():
                key = r["contrast"].replace("−", "−")
                if key in hits:
                    hits[key] += int(r["lo"] > 0 or r["hi"] < 0)
        for k, h in hits.items():
            rows.append(dict(outcome=out, effect=k, n=n, power=h / reps, reps=reps))
    return pd.DataFrame(rows)


def required_n(power: pd.DataFrame, target: float = 0.8) -> pd.DataFrame:
    rows = []
    for (o, e), g in power.groupby(["outcome", "effect"], sort=False):
        ok = g[g["power"] >= target]
        rows.append(dict(outcome=o, effect=e, required_n=int(ok["n"].min()) if len(ok) else None,
                         max_power=float(g["power"].max()), largest_n_tried=int(g["n"].max())))
    return pd.DataFrame(rows)


# ================================================================================================ analysis plan
def analysis_plan(plan: ExperimentPlan = PLAN, components: Optional[List[Dict]] = None,
                  power: Optional[pd.DataFrame] = None, pilot_label: str = "", registered_hash: str = "") -> str:
    """The prospective analysis plan as Markdown (exportable before data collection)."""
    L = []
    L.append("# Prospective analysis plan: mechanisms of adaptation under uncertainty (human experiment)")
    L.append("")
    L.append(f"* Protocol hash: `{plan.digest}`" + (f" (registered in the repository: `{registered_hash}`)"
                                                    if registered_hash else ""))
    L.append("* Status: frozen in the repository with its hash. It has **not** been preregistered with an external "
             "registry; deposit it (for example on OSF or AsPredicted) before data collection to make it a "
             "time-stamped preregistration. No human data have been collected under this protocol.")
    L.append("")
    L.append("## Design")
    L.append(f"* Within participants: four main blocks of {plan.periods_per_block} periods, one per condition:")
    for c, v in CONDITIONS.items():
        L.append(f"  * **{c}** {CONDITION_LABELS[c]}: cost volatility {v['delta']:g}, cost-estimate noise "
                 f"{v['noise']:g}, demand-shift hazard {v['hazard']:g}")
    L.append("  Each non-baseline condition differs from B in exactly one mechanism.")
    L.append(f"* Order: balanced Williams square {WILLIAMS} (rows assigned by the allocation schedule); order and "
             "block position recorded.")
    L.append(f"* Paths: {plan.n_paths} exogenous paths (seeds {list(plan.path_seeds)}), path = (condition index + "
             "offset) mod 4, so each path appears equally often in each condition.")
    L.append("* Between participants (randomized; nested permuted blocks of 16 and 64): decision aid shown "
             f"(recommendation of design `{plan.aid_design}` with its gate's change/keep verdict) versus the UNAIDED "
             "CONTROL (identical screens and information options; the recommendation is computed and recorded but "
             f"never displayed); belief elicitation on (every {plan.elicit_every}rd decision, separate timed screen: "
             "price forecast and confidence that the chosen output beats keeping) versus off.")
    L.append(f"* Practice block of {plan.practice_periods} baseline periods (separate path, not analyzed); "
             f"comprehension checks (all correct within {plan.max_attempts} attempts).")
    L.append("")
    L.append("## Measures")
    L.append("Per decision: condition, block position, path, decision time, belief time (separately), information "
             "panels opened, the aid's recommendation and whether it was shown, chosen output, feedback shown "
             "(price, profit), belief responses, every design's shadow choice.")
    L.append("")
    L.append("## Exclusions (fixed in advance)")
    for e in plan.exclusions:
        L.append(f"* {e}")
    L.append("")
    L.append("## Hypotheses and decision rules")
    for h in plan.hypotheses:
        L.append(f"* **{h[0]}** {h[1]} *Rule:* {h[2]}")
    L.append("")
    L.append("## Analysis")
    L.append("* Outcomes per participant and block: adjustment probability (share of periods with a change), "
             "adjustment magnitude (mean absolute change when changing), relative profit (own minus rivals' mean). "
             "Probability and magnitude are analyzed separately.")
    L.append("* Within-participant contrasts (E1–E3): OLS with participant, path and block-position fixed effects; "
             "cluster-robust intervals by participant, or by session when sessions are recorded.")
    L.append("* Between-participant effects (E4–E5): participant means on aid and elicitation indicators, HC1 (or "
             "session-clustered) intervals.")
    L.append("* Mechanism models (M1, secondary): hurdle models of adjustment probability (band, logit, logit + "
             "inertia, + rounding; all with lapse) and magnitude (partial adjustment toward a reference target, "
             "rounding, aid following), partially pooled across participants. Compared on held-out blocks by log "
             "score, Brier score and RMSE; parameter recovery and model recovery (confusion matrix) reported before "
             "interpreting parameters.")
    L.append("* A1 is reported as an association only.")
    L.append("")
    if components:
        L.append("## Sample size")
        L.append(f"Pilot used: **{pilot_label}**. Effect sizes and variance components (method of moments):")
        L.append("")
        L.append("| outcome | B mean | V−B | N−B | R−B | aid | elicitation | sd participant | sd path | sd residual | "
                 "pilot n |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in components:
            L.append(f"| {c['outcome']} | {c['mean']:.4g} | {c['V']:.4g} | {c['N']:.4g} | {c['R']:.4g} | "
                     f"{c['aid']:.4g} | {c['elicit']:.4g} | {c['sd_participant']:.4g} | {c['sd_path']:.4g} | "
                     f"{c['sd_residual']:.4g} | {c['pilot_participants']} |")
        if power is not None and len(power):
            rq = required_n(power)
            L.append("")
            L.append("Simulation-based power (same allocation, paths, dependence structure and estimators; "
                     f"{int(power['reps'].iloc[0])} simulated datasets per size):")
            L.append("")
            L.append("| outcome | effect | required n for 80% power | power at largest n |")
            L.append("|---|---|---|---|")
            for _, r in rq.iterrows():
                L.append(f"| {r['outcome']} | {r['effect']} | {r['required_n'] if r['required_n'] else '> ' + str(r['largest_n_tried'])} | "
                         f"{r['max_power']:.2f} |")
        if "SYNTHETIC" in pilot_label.upper():
            L.append("")
            L.append("**The pilot is synthetic.** These sample sizes check the design under assumed effects; they "
                     "must be recomputed from a human pilot before the main study.")
    return "\n".join(L) + "\n"
