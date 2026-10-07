"""Mechanism models of human adjustment decisions (hurdle models), fitting, held-out prediction and recovery studies.

Each decision is split into two parts that are modeled separately:
    adjustment probability   whether output changes (d = 1 if q_t != q_{t-1})
    adjustment magnitude     how much it changes, given that it changes (Delta = q_t - q_{t-1})
The reference signal s_t is the change toward the researcher's reference target (the filtered best reply computed from
what the participant saw, experiment.SIGNAL_DESIGN; never shown to the participant); x_t = |s_t| / 50.

Models (all include a lapse rate epsilon: with probability epsilon a decision is random, i.e. adjust with probability
1/2 and, if adjusting, a uniform change in [-400, 400]):
    band     adjust if x exceeds a threshold theta (steep logistic, slope 10), plus condition shifts
    logit    P(adjust) = logistic(a + b x + c_condition)
    inertia  logit + stickiness: - g * 1[kept output last period]
    full     inertia + rounding: with probability rho the intended change is rounded to a multiple of 25
Magnitude in every model: Delta ~ phi * s + N(0, sd^2), observed on whole units and conditioned on Delta != 0.
In the aid arm, a participant follows the shown recommendation with probability f (adjusting exactly to it when it
says change); f is estimated only for aided participants and absent for the unaided control.

Individual heterogeneity: every parameter is estimated per participant with partial pooling (empirical Bayes): a first
pass with a weak prior, then the population mean and spread of the first-pass estimates become the prior of a second
pass. Held-out prediction leaves out one block at a time and uses a prior computed without the participant.

Scores on held-out decisions: log score (log probability of the observed adjustment decision and, if adjusted, of the
observed change on whole units), Brier score of the adjustment probability, and RMSE of the predicted output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, ndtr

MODELS = ("band", "logit", "inertia", "full")
MODEL_LABELS = {"band": "Inaction band (threshold)", "logit": "Probabilistic adjustment",
                "inertia": "Probabilistic adjustment + inertia", "full": "Inertia + rounding"}
COND_INDEX = {"B": 0, "V": 1, "N": 2, "R": 3}
SCALE = 50.0
LAPSE_RANGE = 801.0
ROUND = 25.0

# parameter -> (transform, weak prior mean, weak prior sd) on the unconstrained scale
PARAMS = {
    "a": ("id", 0.0, 3.0), "b": ("exp", 0.0, 1.5), "theta": ("exp", 0.0, 1.5), "g": ("id", 0.0, 3.0),
    "cV": ("id", 0.0, 2.0), "cN": ("id", 0.0, 2.0), "cR": ("id", 0.0, 2.0), "lapse": ("logit", -3.0, 1.5),
    "phi": ("phi", 0.0, 2.0), "sd": ("exp", np.log(15.0), 1.5), "rho": ("logit", -1.0, 2.0),
    "follow": ("logit", -1.0, 2.0),
}
MODEL_PARAMS = {
    "band": ("theta", "cV", "cN", "cR", "lapse", "phi", "sd"),
    "logit": ("a", "b", "cV", "cN", "cR", "lapse", "phi", "sd"),
    "inertia": ("a", "b", "g", "cV", "cN", "cR", "lapse", "phi", "sd"),
    "full": ("a", "b", "g", "cV", "cN", "cR", "lapse", "phi", "sd", "rho"),
}

# Assumed population for SYNTHETIC PILOTS (natural scale means and unconstrained-scale spreads). These are design
# assumptions for checking the pipeline and planning, not estimates of human behavior.
DEFAULT_POPULATION = dict(a=-0.3, a_sd=0.8, b=1.5, b_sd=0.4, theta=0.6, theta_sd=0.3, g=1.0, g_sd=0.5,
                          cV=0.6, cN=-0.3, cR=0.4, c_sd=0.3, lapse=0.03, lapse_sd=0.5, phi=0.5, phi_sd=0.4,
                          sd=15.0, sd_sd=0.3, rho=0.3, rho_sd=0.7, follow=0.35, follow_sd=0.6, elicit_shift=-0.2)


def _to_nat(name, u):
    t = PARAMS[name][0]
    if t == "id":
        return u
    if t == "exp":
        return np.exp(u)
    if t == "logit":
        return expit(u)
    return 1.5 * expit(u)                     # phi in (0, 1.5)


def _to_unc(name, v):
    t = PARAMS[name][0]
    if t == "id":
        return v
    if t == "exp":
        return np.log(v)
    if t == "logit":
        return np.log(v / (1 - v))
    p = v / 1.5
    return np.log(p / (1 - p))


def param_names(model: str, aid: bool) -> Tuple[str, ...]:
    return MODEL_PARAMS[model] + (("follow",) if aid else ())


# ================================================================================================ data
@dataclass
class PData:
    participant: str
    aid: bool
    d: np.ndarray            # adjusted (0/1)
    dq: np.ndarray           # observed change
    s: np.ndarray            # reference signal
    cond: np.ndarray         # condition index
    kept: np.ndarray         # kept output in the previous decision
    rec_dq: np.ndarray       # change the aid recommended (NaN if not computed)
    q_prev: np.ndarray
    block: np.ndarray

    def subset(self, mask) -> "PData":
        return PData(self.participant, self.aid, *(x[mask] for x in (self.d, self.dq, self.s, self.cond, self.kept,
                                                                     self.rec_dq, self.q_prev, self.block)))


def prepare(df: pd.DataFrame, signal: str = "opt_br", aid_design: str = "heiner_m") -> List[PData]:
    """Per-participant arrays from recorded decisions (practice excluded), in decision order."""
    d0 = df[~df["practice"].astype(bool)] if "practice" in df else df
    out = []
    for pid, g in d0.sort_values(["participant", "block_index", "period"]).groupby("participant", sort=True):
        q_prev = g["q_prev"].to_numpy(float)
        dq = g["q"].to_numpy(float) - q_prev
        d = (dq != 0).astype(float)
        kept = np.concatenate([[0.0], 1.0 - d[:-1]])
        s = g[f"shadow:{signal}"].to_numpy(float) - q_prev
        rec = g[f"shadow:{aid_design}"].to_numpy(float) if f"shadow:{aid_design}" in g else np.full(len(g), np.nan)
        rec_dq = np.maximum(5.0, np.rint(rec)) - q_prev
        out.append(PData(str(pid), bool(g["aid"].iloc[0]) if "aid" in g else False, d, dq, np.nan_to_num(s),
                         g["condition"].map(COND_INDEX).to_numpy(int), kept, rec_dq, q_prev,
                         g["block_index"].to_numpy(int)))
    return out


# ================================================================================================ likelihood
def _nat(model, aid, u):
    names = param_names(model, aid)
    return {n: _to_nat(n, u[i]) for i, n in enumerate(names)}


def components(model: str, p: Dict[str, float], x: PData):
    """Per decision: P(adjust), log P(observed change | adjust), expected change given adjust."""
    xs = np.abs(x.s) / SCALE
    c = np.array([0.0, p["cV"], p["cN"], p["cR"]])[x.cond]
    if model == "band":
        core = expit(10.0 * (xs - p["theta"]) + c)
    else:
        z = p["a"] + p["b"] * xs + c
        if model in ("inertia", "full"):
            z = z - p["g"] * x.kept
        core = expit(z)
    eps = p["lapse"]
    f = p.get("follow", 0.0) if x.aid else 0.0
    rc = (np.nan_to_num(x.rec_dq) != 0).astype(float)
    p_adj = eps / 2 + (1 - eps) * (f * rc + (1 - f) * core)
    p_adj = np.clip(p_adj, 1e-9, 1 - 1e-9)
    mu, sd = p["phi"] * x.s, p["sd"]
    D = x.dq
    gauss = ndtr((D + 0.5 - mu) / sd) - ndtr((D - 0.5 - mu) / sd)
    m0 = ndtr((0.5 - mu) / sd) - ndtr((-0.5 - mu) / sd)
    gauss = gauss / np.maximum(1 - m0, 1e-12)
    rho = p.get("rho", 0.0) if model == "full" else 0.0
    if rho > 0:
        on = (np.mod(D, ROUND) == 0) & (D != 0)
        rnd = np.where(on, ndtr((D + ROUND / 2 - mu) / sd) - ndtr((D - ROUND / 2 - mu) / sd), 0.0)
        r0 = ndtr((ROUND / 2 - mu) / sd) - ndtr((-ROUND / 2 - mu) / sd)
        rnd = rnd / np.maximum(1 - r0, 1e-12)
        body = (1 - rho) * gauss + rho * rnd
    else:
        body = gauss
    fol = (D == np.nan_to_num(x.rec_dq, nan=np.inf)).astype(float) * rc
    p_mag = eps / LAPSE_RANGE + (1 - eps) * ((1 - f) * body + f * fol)
    e_mag = (1 - eps) * ((1 - f) * mu + f * np.nan_to_num(x.rec_dq) * rc)
    return p_adj, np.log(np.maximum(p_mag, 1e-300)), e_mag


def loglik(model: str, u: np.ndarray, x: PData) -> float:
    p_adj, lmag, _ = components(model, _nat(model, x.aid, u), x)
    return float(np.sum(x.d * np.log(p_adj) + (1 - x.d) * np.log(1 - p_adj)) + np.sum(x.d * lmag))


def fit_one(model: str, x: PData, prior_mean: np.ndarray, prior_sd: np.ndarray) -> np.ndarray:
    def nlp(u):
        return -loglik(model, u, x) + 0.5 * np.sum(((u - prior_mean) / prior_sd) ** 2)
    r = minimize(nlp, prior_mean.copy(), method="L-BFGS-B", options=dict(maxiter=400))
    return r.x


def _weak_prior(model, aid):
    names = param_names(model, aid)
    return np.array([PARAMS[n][1] for n in names]), np.array([PARAMS[n][2] for n in names])


def _group_prior(model, est: Dict[str, np.ndarray], aids: Dict[str, bool], exclude: Optional[str] = None):
    """Population prior from first-pass estimates (optionally leaving one participant out), per parameter."""
    names = MODEL_PARAMS[model] + ("follow",)
    vals = {n: [] for n in names}
    for pid, u in est.items():
        if pid == exclude:
            continue
        for i, n in enumerate(param_names(model, aids[pid])):
            vals[n].append(u[i])
    prior = {}
    for n in names:
        v = np.array(vals[n])
        m0, s0 = PARAMS[n][1], PARAMS[n][2]
        if len(v) >= 3:
            prior[n] = (float(v.mean()), float(np.clip(v.std(ddof=1), 0.15, s0)))
        else:
            prior[n] = (m0, s0)
    return prior


def _prior_vec(model, aid, prior):
    names = param_names(model, aid)
    return np.array([prior[n][0] for n in names]), np.array([prior[n][1] for n in names])


@dataclass
class Fit:
    model: str
    params: pd.DataFrame          # participant x natural-scale parameters
    unconstrained: Dict[str, np.ndarray]
    prior: Dict[str, Tuple[float, float]]
    loglik: float
    n_params: int
    n_obs: int

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.n_params * np.log(max(self.n_obs, 1))


def fit_model(model: str, data: Sequence[PData]) -> Fit:
    """Two-pass empirical-Bayes fit with partial pooling across participants."""
    aids = {x.participant: x.aid for x in data}
    first = {x.participant: fit_one(model, x, *_weak_prior(model, x.aid)) for x in data}
    prior = _group_prior(model, first, aids)
    est = {x.participant: fit_one(model, x, *_prior_vec(model, x.aid, prior)) for x in data}
    rows, ll, k, n = [], 0.0, 0, 0
    for x in data:
        u = est[x.participant]
        ll += loglik(model, u, x)
        k += len(u)
        n += len(x.d)
        rows.append(dict(participant=x.participant, aid=x.aid, **_nat(model, x.aid, u)))
    return Fit(model, pd.DataFrame(rows), est, prior, ll, k, n)


# ================================================================================================ held-out prediction
def heldout_scores(model: str, data: Sequence[PData], full: Optional[Fit] = None) -> pd.DataFrame:
    """Leave-one-block-out per participant; the prior comes from the other participants' fits."""
    full = full or fit_model(model, data)
    aids = {x.participant: x.aid for x in data}
    rows = []
    for x in data:
        prior = _group_prior(model, full.unconstrained, aids, exclude=x.participant)
        pm, ps = _prior_vec(model, x.aid, prior)
        for b in np.unique(x.block):
            tr, te = x.subset(x.block != b), x.subset(x.block == b)
            if len(tr.d) < 5 or len(te.d) == 0:
                continue
            u = fit_one(model, tr, pm, ps)
            p_adj, lmag, e_mag = components(model, _nat(model, x.aid, u), te)
            ls = te.d * np.log(p_adj) + (1 - te.d) * np.log(1 - p_adj) + te.d * lmag
            pred_q = te.q_prev + p_adj * e_mag
            rows.append(dict(model=model, participant=x.participant, block=int(b), n=len(te.d),
                             log_score=float(ls.sum()), log_score_adjust=float(
                                 np.sum(te.d * np.log(p_adj) + (1 - te.d) * np.log(1 - p_adj))),
                             brier=float(np.sum((p_adj - te.d) ** 2)),
                             sse_q=float(np.sum((pred_q - (te.q_prev + te.dq)) ** 2))))
    return pd.DataFrame(rows)


def compare_models(data: Sequence[PData], models: Sequence[str] = MODELS):
    fits, frames = {}, []
    for m in models:
        fits[m] = fit_model(m, data)
        frames.append(heldout_scores(m, data, fits[m]))
    h = pd.concat(frames, ignore_index=True)
    g = h.groupby("model").agg(n=("n", "sum"), log_score=("log_score", "sum"), log_adj=("log_score_adjust", "sum"),
                               brier=("brier", "sum"), sse=("sse_q", "sum")).reset_index()
    g["log_score_per_decision"] = g["log_score"] / g["n"]
    g["adjustment_log_score_per_decision"] = g["log_adj"] / g["n"]
    g["brier"] = g["brier"] / g["n"]
    g["rmse_output"] = np.sqrt(g["sse"] / g["n"])
    g["bic_full_data"] = g["model"].map(lambda m: fits[m].bic)
    g["label"] = g["model"].map(MODEL_LABELS)
    return g[["model", "label", "n", "log_score_per_decision", "adjustment_log_score_per_decision", "brier",
              "rmse_output", "bic_full_data"]], fits, h


# ================================================================================================ synthetic agents
def draw_participant(model: str, pop: Dict, rng: np.random.Generator, aid: bool, elicit: bool) -> Dict[str, float]:
    """Natural-scale parameters of one SYNTHETIC participant from the assumed population."""
    def draw(name, mean, sd):
        return float(_to_nat(name, _to_unc(name, mean) + sd * rng.standard_normal()))
    th = dict(a=pop["a"] + pop["a_sd"] * rng.standard_normal() + (pop["elicit_shift"] if elicit else 0.0),
              b=draw("b", pop["b"], pop["b_sd"]), theta=draw("theta", pop["theta"], pop["theta_sd"]),
              g=pop["g"] + pop["g_sd"] * rng.standard_normal(),
              cV=pop["cV"] + pop["c_sd"] * rng.standard_normal(), cN=pop["cN"] + pop["c_sd"] * rng.standard_normal(),
              cR=pop["cR"] + pop["c_sd"] * rng.standard_normal(), lapse=draw("lapse", pop["lapse"], pop["lapse_sd"]),
              phi=draw("phi", pop["phi"], pop["phi_sd"]), sd=draw("sd", pop["sd"], pop["sd_sd"]),
              rho=draw("rho", pop["rho"], pop["rho_sd"]),
              follow=draw("follow", pop["follow"], pop["follow_sd"]) if aid else 0.0)
    if model == "band":
        th.update(g=0.0, rho=0.0)
    elif model == "logit":
        th.update(g=0.0, rho=0.0)
    elif model == "inertia":
        th.update(rho=0.0)
    return {k: th[k] for k in param_names(model, aid)}


def simulate_choice(model: str, theta: Dict[str, float], cond: str, s: float, q_prev: float, kept_last: float,
                    rng: np.random.Generator, aid_rec: Optional[float] = None) -> Tuple[float, bool]:
    """One decision of a SYNTHETIC participant under the model (same structure as the likelihood)."""
    p = {**dict(cV=0.0, cN=0.0, cR=0.0, g=0.0, rho=0.0, follow=0.0, a=0.0, b=0.0, theta=1.0), **theta}
    eps = p["lapse"]
    if rng.random() < eps:                                   # lapse
        if rng.random() < 0.5:
            return q_prev + float(rng.integers(-400, 401)), True
        return q_prev, False
    if aid_rec is not None and rng.random() < p["follow"]:
        rec = float(max(5.0, np.rint(aid_rec)))
        return rec, rec != q_prev
    c = {"B": 0.0, "V": p["cV"], "N": p["cN"], "R": p["cR"]}[cond]
    x = abs(s) / SCALE
    if model == "band":
        pa = expit(10.0 * (x - p["theta"]) + c)
    else:
        pa = expit(p["a"] + p["b"] * x + c - (p["g"] * kept_last if model in ("inertia", "full") else 0.0))
    if rng.random() >= pa:
        return q_prev, False
    for _ in range(50):
        delta = p["phi"] * s + p["sd"] * rng.standard_normal()
        if model == "full" and rng.random() < p["rho"]:
            delta = ROUND * np.round(delta / ROUND)
        delta = float(np.rint(delta))
        if delta != 0:
            return q_prev + delta, True
    return q_prev + 1.0, True


# ================================================================================================ recovery studies
def parameter_recovery(n: int = 24, seed: int = 1, model: str = "full") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the generating model to SYNTHETIC participants and compare estimated with true parameters."""
    from .experiment import synthetic_pilot
    df, truth = synthetic_pilot(n, seed=seed, model=model)
    fit = fit_model(model, prepare(df))
    m = fit.params.merge(truth, on="participant")
    rows = []
    for name in MODEL_PARAMS[model] + ("follow",):
        if f"true:{name}" not in m or name not in m:
            continue
        sub = m[[name, f"true:{name}"]].dropna()
        if name == "follow":
            sub = sub[m.loc[sub.index, "aid"]]
        if len(sub) < 3:
            continue
        t, e = sub[f"true:{name}"].to_numpy(float), sub[name].to_numpy(float)
        rows.append(dict(parameter=name, n=len(sub), correlation=float(np.corrcoef(t, e)[0, 1]) if np.std(t) > 0
                         and np.std(e) > 0 else np.nan, bias=float(np.mean(e - t)), rmse=float(np.sqrt(np.mean(
                             (e - t) ** 2))), true_mean=float(t.mean()), est_mean=float(e.mean())))
    return pd.DataFrame(rows), m


def model_recovery(n: int = 8, seed: int = 2, models: Sequence[str] = MODELS, criterion: str = "heldout"
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """For every generating model, simulate SYNTHETIC participants, fit every candidate model and record which one each
    participant is assigned to (best held-out log score, or BIC on the full data). Returns the confusion matrix
    (rows: generating model; columns: selected model; shares) and the per-participant table."""
    from .experiment import synthetic_pilot
    rows = []
    for j, gen in enumerate(models):
        df, _ = synthetic_pilot(n, seed=seed * 100 + j, model=gen)
        data = prepare(df)
        scores = {}
        for m in models:
            fit = fit_model(m, data)
            if criterion == "heldout":
                h = heldout_scores(m, data, fit).groupby("participant")["log_score"].sum()
                scores[m] = h
            else:
                ll = {x.participant: loglik(m, fit.unconstrained[x.participant], x)
                      - 0.5 * len(fit.unconstrained[x.participant]) * np.log(len(x.d)) for x in data}
                scores[m] = pd.Series(ll)
        tab = pd.DataFrame(scores)
        for pid, r in tab.iterrows():
            rows.append(dict(generating=gen, participant=pid, selected=r.idxmax(), **{f"score:{m}": r[m]
                                                                                       for m in models}))
    per = pd.DataFrame(rows)
    conf = pd.crosstab(per["generating"], per["selected"], normalize="index").reindex(index=list(models),
                                                                                     columns=list(models),
                                                                                     fill_value=0.0)
    return conf, per
