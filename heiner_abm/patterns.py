"""Pattern-oriented validation (Grimm et al. 2005): does the market reproduce documented empirical patterns?

Each pattern is stated as it appears in the literature, with a qualitative criterion fixed in advance and the
population of agents that should produce it. A model that reproduces several independent patterns at once is harder
to dismiss as an artefact of its assumptions. The criteria are deliberately qualitative: the market is stylised and is
not calibrated to any particular commodity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .arena import Env, Tuned
from .rulechoice import simulate_choice

BASE = dict(delta=10.0, c_max=80.0, q_range=1500.0, noise=0.0, foresight=0.0, hazard=0.0, belief_lag=20)


@dataclass(frozen=True)
class Pattern:
    key: str
    name: str
    evidence: str          # what the literature documents
    sources: Tuple[str, ...]
    criterion: str         # the qualitative test, fixed in advance
    population: str        # which agents should produce it


PATTERNS: List[Pattern] = [
    Pattern("cycles", "Cobweb cycles", "Markets in which producers commit output on last period's price show "
            "alternating price cycles, as in the classic hog and potato cycles.", ("ezekiel1938", "nerlove1958"),
            "Lag-1 autocorrelation of price below −0.3 when every firm uses naive price expectations with full "
            "adjustment.", "Naive expectations (adaptive price expectations with λ = 1, φ = 1)"),
    Pattern("damping", "Adaptive adjustment dampens cycles", "Adaptive expectations and partial adjustment stabilise "
            "cobweb markets.", ("nerlove1958", "carlson1968"),
            "With slow adaptive expectations (λ = 0.3, φ = 0.3), price volatility is lower and lag-1 autocorrelation "
            "higher than with naive expectations.", "Adaptive expectations (λ = 0.3, φ = 0.3)"),
    Pattern("sticky", "Sticky, lumpy adjustment", "Many transaction prices and quantities stay unchanged for long "
            "spells and then change by large steps.", ("carlton1986", "caballero1999"),
            "With inaction-band firms, output changes in fewer than half of the periods, and changes are larger on "
            "average than those of always-adjusting firms.", "Inaction band, price-based target (tuned)"),
    Pattern("imitation", "Imitation makes oligopolies more competitive", "In Cournot experiments with information "
            "about rivals' profits, output exceeds the Cournot–Nash level.", ("huck1999", "vegaredondo1997"),
            "With imitate-the-best firms, mean market output exceeds the Cournot–Nash output computed from the true "
            "demand and cost.", "Imitate the best (tuned)"),
    Pattern("excess", "Prices fluctuate around equilibrium with excess volatility", "In learning-to-forecast and "
            "cobweb experiments, prices stay near the equilibrium on average but fluctuate more than fundamentals "
            "imply.", ("hommes2007", "sonnemans2004"),
            "In a market of heterogeneous learners, the mean price is within 10% of the mean Cournot–Nash price, and "
            "its standard deviation exceeds that of the Cournot–Nash price.", "Mixed tuned designs (one of each "
            "flexible and restricted rule)"),
    Pattern("markups", "Positive, persistent markups", "Oligopolies price above marginal cost but below the "
            "monopoly price.", ("stigler1964", "carlton1986"),
            "Mean price lies above mean cost and below the mean monopoly price.", "Mixed tuned designs"),
]


def _run(rules: Sequence[str], params: Dict, n_firms: int, envs: Sequence[Env], periods: int, burn: int,
         mixed: bool = False):
    """Homogeneous market of one rule, or a fixed mixed population (firms assigned to rules in turn)."""
    if not mixed:
        return simulate_choice(envs, params, 0.0, n_firms, periods, burn, revision=0.0, rules=tuple(rules),
                               keep_market=True)
    # fixed mixture: run with all rules and no revision, starting from a deterministic assignment
    out = simulate_choice(envs, params, 0.0, n_firms, periods, burn, revision=0.0, rules=tuple(rules),
                          keep_market=True, rng_seed=12345)
    return out


def _ac1(x: np.ndarray) -> np.ndarray:
    a, b = x[:, :-1] - x[:, :-1].mean(1, keepdims=True), x[:, 1:] - x[:, 1:].mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def _nash(out, n):
    pm, sl, c = out["p_max"], out["slope"], out["cost"]
    Qn = np.maximum(0, n * (pm - c) / (sl * (n + 1)))
    return Qn, np.maximum(10.0, pm - sl * Qn)


def run_patterns(tuned: Tuned, n_envs: int = 12, periods: int = 600, burn: int = 100, n_firms: int = 6,
                 seed: int = 5005, progress: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
    envs = [Env(**BASE, seed=seed + j) for j in range(n_envs)]
    steep = [Env(**{**BASE, "q_range": 700.0}, seed=seed + j) for j in range(n_envs)]   # steep demand: unstable
    sl = slice(burn, None)
    rows = []

    def add(p: Pattern, value: str, passed: bool, detail: str):
        rows.append(dict(pattern=p.name, key=p.key, evidence=p.evidence, criterion=p.criterion,
                         population=p.population, result=value, passed=bool(passed), detail=detail,
                         sources=p.sources))

    P = {p.key: p for p in PATTERNS}
    step = lambda k, msg: progress(k / 6, msg) if progress else None
    step(0, "Cobweb cycles")
    naive = _run(["cobweb_p"], {"cobweb_p": {"lam": 1.0, "phi": 1.0}}, n_firms, steep, periods, burn)
    ac_n = _ac1(naive["price"][:, sl])
    sd_n = naive["price"][:, sl].std(1)
    add(P["cycles"], f"lag-1 autocorrelation {ac_n.mean():+.2f}", ac_n.mean() < -0.3,
        f"{(ac_n < -0.3).mean():.0%} of markets below −0.3")
    step(1, "Damping")
    slow = _run(["cobweb_p"], {"cobweb_p": {"lam": 0.3, "phi": 0.3}}, n_firms, steep, periods, burn)
    ac_s, sd_s = _ac1(slow["price"][:, sl]), slow["price"][:, sl].std(1)
    add(P["damping"], f"s.d. {sd_s.mean():.1f} vs {sd_n.mean():.1f}; autocorrelation {ac_s.mean():+.2f} vs "
        f"{ac_n.mean():+.2f}", sd_s.mean() < sd_n.mean() and ac_s.mean() > ac_n.mean(),
        f"lower volatility in {(sd_s < sd_n).mean():.0%} of markets")
    step(2, "Stickiness")
    band = _run(["options_p"], tuned.params, n_firms, envs, periods, burn)
    always = _run(["cobweb_p"], tuned.params, n_firms, envs, periods, burn)

    def lumpy(o):
        dq = np.abs(np.diff(o["q"][:, burn:], axis=1))
        rate = (dq > 0).mean()
        return rate, dq[dq > 0].mean() if (dq > 0).any() else 0.0
    rb, mb = lumpy(band)
    ra, ma = lumpy(always)
    add(P["sticky"], f"output changes in {rb:.0%} of periods (always-adjusting: {ra:.0%}); mean change {mb:.1f} vs "
        f"{ma:.1f}", rb < 0.5 and mb > ma, "inaction band vs adaptive price expectations, same environments")
    step(3, "Imitation")
    imit = _run(["imit_best"], tuned.params, n_firms, envs, periods, burn)
    Qn, _ = _nash(imit, n_firms)
    ratio = imit["quantity"][:, sl].mean(1) / Qn[:, sl].mean(1)
    add(P["imitation"], f"mean output {ratio.mean():.2f} × Cournot–Nash", ratio.mean() > 1,
        f"above Nash in {(ratio > 1).mean():.0%} of markets")
    step(4, "Excess volatility")
    mix_rules = ("cobweb_p", "opt_br", "heur_markup", "options_p", "heiner_p", "ruleb")
    mix = _run(mix_rules, tuned.params, n_firms, envs, periods, burn, mixed=True)
    _, Pn = _nash(mix, n_firms)
    pr = mix["price"][:, sl]
    dev = np.abs(pr.mean(1) / Pn[:, sl].mean(1) - 1)
    exc = pr.std(1) / Pn[:, sl].std(1)
    add(P["excess"], f"mean price within {dev.mean():.1%} of Nash; s.d. {exc.mean():.2f} × Nash s.d.",
        dev.mean() < 0.10 and exc.mean() > 1, f"excess volatility in {(exc > 1).mean():.0%} of markets")
    step(5, "Markups")
    c = mix["cost"][:, sl]
    mono = (mix["p_max"][:, sl] + c) / 2
    ok = (pr.mean(1) > c.mean(1)) & (pr.mean(1) < mono.mean(1))
    add(P["markups"], f"mean price {pr.mean():.1f}, mean cost {c.mean():.1f}, monopoly price {mono.mean():.1f}",
        ok.mean() > 0.5, f"between cost and monopoly price in {ok.mean():.0%} of markets")
    if progress:
        progress(1.0, "Done")
    return pd.DataFrame(rows)
