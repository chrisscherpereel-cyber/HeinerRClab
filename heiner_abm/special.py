"""Signature tests: each theory's own, distinctive prediction, tested in the shared market.

The hypothesis tests and tournaments ask how the theories compare on common questions. These tests ask a different
question: does the theory's signature claim, the prediction that characterizes it, hold when its own agents act in the
simulated market? Each test fixes its criterion in advance and reports whether it is met.

    optimization   better information raises an optimizer's profit, and with full information a market of rational
                   (Cournot–Nash) players settles at the equilibrium (Muth 1961)
    real options   the value of flexibility over a rigid twin rises with volatility (Dixit & Pindyck 1994)
    cobweb         partial best-reply dynamics are stable only for adjustment speeds below 4/(n+1) (Theocharis 1960)
    heuristics     a simple rule beats the model-based optimizer by more as estimation noise rises (Gigerenzer &
                   Brighton 2009)
    satisficing    worse or more volatile conditions trigger more search, i.e. more frequent change (Simon 1955;
                   Cyert & March 1963)
    learning       a reinforcement learner improves with experience in a stationary environment (Sutton & Barto 2018)
    imitation      imitating the most profitable firm drives output above the Cournot–Nash level (Vega-Redondo 1997)
    ecology        inert organizations perform more reliably than flexible ones in every environment (Hannan &
                   Freeman 1984)

Heiner's reliability condition has its own set of special tests on separate pages (listed in HEINER_TESTS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .analysis import ols
from .arena import Env, nash_output, simulate
from .registered import TUNED_PARAMS
from .rulechoice import simulate_choice

BACKGROUND = ("cobweb_p", "heur_markup", "options_p")      # tuned rivals around a focal firm (as in the experiment)
SCALES = {"Quick": dict(n_envs=6, periods=500, burn=100), "Full": dict(n_envs=20, periods=1500, burn=200)}


@dataclass(frozen=True)
class SpecialTest:
    key: str                  # theory key (heiner_abm.theory_content)
    title: str
    claim: str                # the signature prediction
    setup: str                # what is simulated
    criterion: str            # the decision rule, fixed in advance
    sources: Tuple[str, ...]
    run: Callable[[Dict], Dict]


def _env(seed: int, delta: float = 10.0, noise: float = 0.0, foresight: float = 0.0, hazard: float = 0.0,
         belief_lag: int = 20, q_range: float = 1500.0) -> Env:
    return Env(delta=delta, c_max=80.0, q_range=q_range, noise=noise, foresight=foresight, hazard=hazard,
               belief_lag=belief_lag, seed=seed)


def _focal_gain(focal_a: str, focal_b: str, envs: List[Env], periods: int, burn: int, params=None) -> np.ndarray:
    """Per environment: profit of a focal firm using design a minus using design b, against the same tuned
    background rivals and the same random draws (common random numbers)."""
    params = params or TUNED_PARAMS
    out = []
    for d in (focal_a, focal_b):
        lineup = np.array([[d, *BACKGROUND]] * len(envs), dtype=object)
        out.append(simulate(envs, lineup, params, periods, burn)["profit"][:, 0])
    return out[0] - out[1]


def _slope(x, y) -> Tuple[float, float]:
    tab, _ = ols(np.asarray(y, float), [np.asarray(x, float)], ["x"])
    return float(tab.loc[1, "coef"]), float(tab.loc[1, "p"])


def _ci(x, n_boot: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    b = np.random.default_rng(seed).choice(x, (n_boot, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ================================================================================================ the tests
def run_optimization(sc: Dict) -> Dict:
    n, T, burn = sc["n_envs"], sc["periods"], sc["burn"]
    rows = []
    for kappa in (0.0, 0.25, 0.5, 0.75, 1.0):
        envs = [_env(9100 + j, delta=15.0, foresight=kappa) for j in range(n)]
        lineup = np.array([["opt_br", *BACKGROUND]] * n, dtype=object)
        # perception noise is zero here, so the optimizer uses its cost estimate unfiltered (a_cost = 1)
        params = {**TUNED_PARAMS, "opt_br": {**TUNED_PARAMS["opt_br"], "a_cost": 1.0}}
        prof = simulate(envs, lineup, params, T, burn)["profit"]
        for j in range(n):
            rows.append(dict(foresight=kappa, env=j, relative_profit=prof[j, 0] - prof[j].mean()))
    d = pd.DataFrame(rows)
    b, p = _slope(d["foresight"], d["relative_profit"])
    # full information, rational players: does the market settle at Cournot–Nash?
    envs = [_env(9200 + j, delta=2.0, foresight=1.0) for j in range(n)]
    out = simulate_choice(envs, {"opt_nash": {"a_cost": 1.0, "phi": 1.0}}, 0.0, 4, T, burn, revision=0.0,
                          rules=("opt_nash",), keep_market=True)
    Pn = np.maximum(10.0, (out["p_max"] + 4 * out["cost"]) / 5)[:, burn:]
    gap = float(np.mean(np.abs(out["price"][:, burn:] / Pn - 1)))
    ok = bool(b > 0 and p < 0.05 and gap < 0.02)
    return dict(table=d.groupby("foresight")["relative_profit"].agg(["mean", "std"]).reset_index()
                .rename(columns={"mean": "optimizer profit − market mean", "std": "s.d. across markets"}),
                x="foresight", y="optimizer profit − market mean", x_title="Foresight κ (information quality)",
                ok=ok, result=f"profit slope on foresight {b:+.1f} per unit (p = {p:.3g}); with full information the "
                              f"rational market's price is within {gap:.2%} of Cournot–Nash on average")


def run_options(sc: Dict) -> Dict:
    n, T, burn = sc["n_envs"], sc["periods"], sc["burn"]
    rows = []
    for delta in (2.0, 10.0, 20.0, 30.0):
        envs = [_env(9300 + j, delta=delta) for j in range(n)]
        g = _focal_gain("cobweb_p", "ruleb", envs, T, burn)
        rows += [dict(volatility=delta, gain=x) for x in g]
    d = pd.DataFrame(rows)
    b, p = _slope(d["volatility"], d["gain"])
    return dict(table=d.groupby("volatility")["gain"].agg(["mean", "std"]).reset_index()
                .rename(columns={"mean": "flexible − rigid twin", "std": "s.d. across markets"}),
                x="volatility", y="flexible − rigid twin", x_title="Cost volatility Δ",
                ok=bool(b > 0 and p < 0.05),
                result=f"value of flexibility changes by {b:+.1f} per unit of volatility (p = {p:.3g})")


def run_cobweb(sc: Dict) -> Dict:
    T, burn = max(400, sc["periods"] // 2), 100
    phis = np.round(np.arange(0.3, 1.31, 0.05), 2)
    rows, ok = [], True
    for n_f in (3, 4, 6, 8):
        theory = 4.0 / (n_f + 1)
        envs = [_env(9400 + j, delta=1.0, foresight=1.0) for j in range(3)]
        sd = []
        for phi in phis:
            out = simulate_choice(envs, {"cobweb_q": {"phi": float(phi), "a_rival": 1.0}}, 0.0, n_f, T, burn,
                                  revision=0.0, rules=("cobweb_q",), keep_market=True)
            Q = out["quantity"][:, -100:]
            qn = n_f * nash_output(1500.0, n_f)
            sd.append(float(np.mean(Q.std(1) / qn)))
        sd = np.array(sd)
        unstable = phis[sd > 0.05]
        emp = float(unstable.min()) if len(unstable) else np.nan
        if theory <= phis.max():
            ok &= bool(np.isfinite(emp) and abs(emp - theory) <= 0.1)
        rows.append(dict(firms=n_f, theoretical_boundary=round(theory, 3), simulated_boundary=emp))
    d = pd.DataFrame(rows)
    return dict(table=d, x="firms", y="simulated_boundary", y2="theoretical_boundary", x_title="Number of firms n",
                ok=ok, result="; ".join(f"n = {r.firms}: theory {r.theoretical_boundary:.2f}, simulated "
                                        f"{r.simulated_boundary:.2f}" for r in d.itertuples()))


def run_heuristics(sc: Dict) -> Dict:
    n, T, burn = sc["n_envs"], sc["periods"], sc["burn"]
    rows = []
    for sigma in (0.0, 5.0, 10.0, 20.0):
        envs = [_env(9500 + j, delta=10.0, noise=sigma, hazard=0.02) for j in range(n)]
        g = _focal_gain("heur_markup", "opt_br", envs, T, burn)
        rows += [dict(noise=sigma, gain=x) for x in g]
    d = pd.DataFrame(rows)
    b, p = _slope(d["noise"], d["gain"])
    m, lo, hi = _ci(d[d["noise"] == d["noise"].max()]["gain"])
    return dict(table=d.groupby("noise")["gain"].agg(["mean", "std"]).reset_index()
                .rename(columns={"mean": "heuristic − optimizer", "std": "s.d. across markets"}),
                x="noise", y="heuristic − optimizer", x_title="Perception noise σ",
                ok=bool(b > 0 and p < 0.05 and lo > 0),
                result=f"heuristic's advantage changes by {b:+.1f} per unit of noise (p = {p:.3g}); at the highest "
                       f"noise {m:+.0f} [{lo:+.0f}, {hi:+.0f}]")


def run_satisficing(sc: Dict) -> Dict:
    n, T, burn = sc["n_envs"], sc["periods"], sc["burn"]
    rows = []
    for delta in (2.0, 10.0, 20.0, 30.0):
        envs = [_env(9600 + j, delta=delta) for j in range(n)]
        lineup = np.array([["satis_p", *BACKGROUND]] * n, dtype=object)
        cr = simulate(envs, lineup, TUNED_PARAMS, T, burn)["change_rate"][:, 0]
        rows += [dict(volatility=delta, change_rate=x) for x in cr]
    d = pd.DataFrame(rows)
    b, p = _slope(d["volatility"], d["change_rate"])
    return dict(table=d.groupby("volatility")["change_rate"].agg(["mean", "std"]).reset_index()
                .rename(columns={"mean": "share of periods with a change", "std": "s.d. across markets"}),
                x="volatility", y="share of periods with a change", x_title="Cost volatility Δ",
                ok=bool(b > 0 and p < 0.05),
                result=f"change rate changes by {b:+.4f} per unit of volatility (p = {p:.3g})")


def run_learning(sc: Dict) -> Dict:
    n, T = sc["n_envs"], max(900, sc["periods"])
    rows = []
    for lab, hazard in (("stationary", 0.0), ("unannounced shifts", 0.03)):
        envs = [_env(9700 + j, delta=5.0, hazard=hazard) for j in range(n)]
        lineup = np.array([["rl_softmax", *BACKGROUND]] * n, dtype=object)
        path = simulate(envs, lineup, TUNED_PARAMS, T, 1, keep_path=True)["path"]
        rel = path[:, :, 0] - path.mean(2)
        third = rel.shape[1] // 3
        imp = rel[:, -third:].mean(1) - rel[:, :third].mean(1)
        rows += [dict(environment=lab, improvement=x) for x in imp]
    d = pd.DataFrame(rows)
    m, lo, hi = _ci(d[d["environment"] == "stationary"]["improvement"])
    m2, lo2, hi2 = _ci(d[d["environment"] != "stationary"]["improvement"])
    return dict(table=d.groupby("environment")["improvement"].agg(["mean", "std"]).reset_index()
                .rename(columns={"mean": "last third − first third", "std": "s.d. across markets"}),
                x="environment", y="last third − first third", x_title="Environment", categorical=True,
                ok=bool(lo > 0),
                result=f"relative profit improves by {m:+.0f} [{lo:+.0f}, {hi:+.0f}] from the first to the last third "
                       f"in a stationary market, and by {m2:+.0f} [{lo2:+.0f}, {hi2:+.0f}] with unannounced shifts")


def run_imitation(sc: Dict) -> Dict:
    n, T, burn = sc["n_envs"], sc["periods"], sc["burn"]
    rows = []
    for key in ("imit_best", "imit_avg"):
        envs = [_env(9800 + j, delta=10.0) for j in range(n)]
        out = simulate_choice(envs, TUNED_PARAMS, 0.0, 4, T, burn, revision=0.0, rules=(key,), keep_market=True)
        qn = 4 * np.maximum(0, out["p_max"] - out["cost"]) / (out["slope"] * 5)
        ratio = out["quantity"][:, burn:].mean(1) / qn[:, burn:].mean(1)
        rows += [dict(rule={"imit_best": "imitate the best", "imit_avg": "imitate the average"}[key], ratio=x)
                 for x in ratio]
    d = pd.DataFrame(rows)
    m, lo, hi = _ci(d[d["rule"] == "imitate the best"]["ratio"])
    return dict(table=d.groupby("rule")["ratio"].agg(["mean", "std"]).reset_index()
                .rename(columns={"mean": "market output ÷ Cournot–Nash", "std": "s.d. across markets"}),
                x="rule", y="market output ÷ Cournot–Nash", x_title="Imitation rule", categorical=True,
                ok=bool(lo > 1),
                result=f"imitate-the-best markets produce {m:.2f} [{lo:.2f}, {hi:.2f}] × the Cournot–Nash output")


def run_ecology(sc: Dict) -> Dict:
    n, T, burn = sc["n_envs"], sc["periods"], sc["burn"]
    rows = []
    for delta in (2.0, 10.0, 20.0, 30.0):
        envs = [_env(9900 + j, delta=delta) for j in range(n)]
        sd = {}
        for d in ("ecol_crisis", "cobweb_p"):
            lineup = np.array([[d, *BACKGROUND]] * n, dtype=object)
            sd[d] = simulate(envs, lineup, TUNED_PARAMS, T, burn, keep_path=True)["path"][:, :, 0].std(1)
        rows += [dict(volatility=delta, diff=x) for x in sd["ecol_crisis"] - sd["cobweb_p"]]
    d = pd.DataFrame(rows)
    cis = {v: _ci(g["diff"]) for v, g in d.groupby("volatility")}
    ok = all(hi < 0 for _, _, hi in cis.values())
    return dict(table=d.groupby("volatility")["diff"].agg(["mean", "std"]).reset_index()
                .rename(columns={"mean": "inert − flexible twin (s.d. of profit)", "std": "s.d. across markets"}),
                x="volatility", y="inert − flexible twin (s.d. of profit)", x_title="Cost volatility Δ", ok=ok,
                result="; ".join(f"Δ = {v:g}: {m:+.0f} [{lo:+.0f}, {hi:+.0f}]" for v, (m, lo, hi) in cis.items()))


SPECIAL_TESTS: List[SpecialTest] = [
    SpecialTest("optimiser", "Information is valuable, and full information yields equilibrium",
                "An optimizer gains from better information, and when every player is rational and fully informed "
                "the market settles at the Cournot–Nash equilibrium.",
                "A focal best-reply firm among three tuned rivals, with cost foresight κ from 0 to 1 (cost volatility "
                "15, no perception noise, so the firm uses its cost estimate unfiltered). Separately, a market of four "
                "rational-expectations players with full foresight.",
                "Supported if the optimizer's profit relative to the market rises with foresight (p < 0.05) and the "
                "fully informed rational market's price is within 2% of the Cournot–Nash price.",
                ("muth1961", "theocharis1960"), run_optimization),
    SpecialTest("options", "The value of flexibility rises with volatility",
                "Flexibility is an option, and options are worth more when the underlying is more volatile.",
                "A focal firm that always adjusts (adaptive price expectations) against its rigid twin (rule B), among "
                "the same three tuned rivals and the same random draws, at cost volatility Δ = 2, 10, 20 and 30.",
                "Supported if the flexible-minus-rigid profit difference rises with volatility (OLS slope > 0, "
                "p < 0.05).", ("dixit1994", "marschak1962"), run_options),
    SpecialTest("cobweb", "Adjustment is stable only below 4/(n + 1)",
                "With linear demand, simultaneous partial adjustment toward the best reply is stable only if the "
                "adjustment speed φ is below 4/(n + 1); faster adjustment makes output oscillate.",
                "Markets of n = 3, 4, 6 and 8 identical firms adjusting a share φ toward the best reply (full cost "
                "foresight, almost no cost volatility), for φ from 0.30 to 1.30.",
                "Supported if, for every n whose boundary lies in the range tested, the smallest φ at which output "
                "oscillates (s.d. above 5% of Nash output) is within 0.10 of 4/(n + 1).",
                ("theocharis1960", "hommes1994"), run_cobweb),
    SpecialTest("heuristic", "Less is more under estimation noise",
                "A simple rule that ignores most information beats a model-based optimizer, and by more the noisier "
                "the information (the bias–variance trade-off).",
                "A focal firm using the target-margin heuristic against its twin using the filtered best reply, among "
                "the same rivals and draws, at perception noise σ = 0, 5, 10 and 20 (with unannounced demand shifts).",
                "Supported if the heuristic's advantage rises with noise (p < 0.05) and is positive at the highest "
                "noise (95% CI above 0).", ("gigerenzer2009", "geman1992"), run_heuristics),
    SpecialTest("satisficing", "Harder conditions trigger more search",
                "Firms change behavior when results fall short of aspirations, so worse or more volatile conditions "
                "trigger more search and more frequent change.",
                "A focal aspiration-search firm among three tuned rivals at cost volatility Δ = 2, 10, 20 and 30; the "
                "share of periods in which it changes output.",
                "Supported if the change rate rises with volatility (OLS slope > 0, p < 0.05).",
                ("simon1955", "cyert1963"), run_satisficing),
    SpecialTest("rl", "Learning improves performance in a stationary world",
                "A reinforcement learner improves with experience when payoffs are stationary; unannounced changes "
                "erode what it has learned.",
                "A focal softmax value learner among three tuned rivals for at least 900 periods, without and with "
                "unannounced demand shifts; its profit relative to the market in the last third minus the first third.",
                "Supported if relative profit improves in the stationary market (95% CI above 0).",
                ("sutton2018", "erev1998"), run_learning),
    SpecialTest("imitation", "Imitating the best makes markets more competitive",
                "When firms copy the most profitable firm, the firm producing most (and earning most at high prices) "
                "is copied, which pushes output beyond the Cournot–Nash level toward the competitive outcome.",
                "Markets of four firms that all imitate the most profitable firm (and, as a contrast, the average "
                "firm); market output relative to the Cournot–Nash output.",
                "Supported if imitate-the-best markets produce more than the Cournot–Nash output (95% CI above 1).",
                ("vegaredondo1997", "huck1999"), run_imitation),
    SpecialTest("ecology", "Inert organizations are more reliable in every environment",
                "Selection favors organizations that perform reliably, and reliable performance requires structural "
                "inertia; the advantage does not depend on how volatile the environment is.",
                "A focal firm that reorganizes only under threat of failure against its flexible twin that adjusts "
                "toward the same price-based target every period, among the same three tuned rivals and the same "
                "random draws, at cost volatility Δ = 2, 10, 20 and 30; the standard deviation of its per-period "
                "profit.",
                "Supported if the inert firm's profit is less variable than its flexible twin's at every volatility "
                "(95% CI of the difference below 0).", ("hannan1984", "amburgey1993"), run_ecology),
]
SPECIAL_BY_KEY = {t.key: t for t in SPECIAL_TESTS}

HEINER_TESTS = (
    ("app_pages/rc_validation.py", "Does the reliability condition predict performance?",
     "Estimates each firm's reliability condition in one part of a run and predicts, in the next, whether its "
     "flexibility beats its own rigid twin (out of sample, scored by AUC)."),
    ("app_pages/dynamic_rc.py", "Dynamic reliability condition (Heiner 1989)",
     "Values each decision over a horizon and tests the partial-adjustment bound β₀ = 1/((1 + K)(1 − f′))."),
    ("app_pages/cd_gap.py", "CD-gap explorer",
     "Maps how difficulty and competence shape r, w, π and the payoff to flexibility."),
    ("app_pages/mechanisms.py", "Mechanisms: oracle versus learned reliability",
     "Separates the principle (restriction with known reliability) from the cost of learning it, and tests the "
     "error-to-signal boundary."),
    ("app_pages/tracking.py", "Heiner versus optimal filtering (Muth–Kalman)",
     "A case solvable on paper: the simulation reproduces the Kalman gain, then lopsided stakes test the prediction "
     "that is uniquely Heiner's."),
)


def run_special(key: str, scale: str = "Quick") -> Dict:
    t = SPECIAL_BY_KEY[key]
    out = t.run(SCALES[scale])
    out["verdict"] = "supported" if out["ok"] else "not supported"
    return out
