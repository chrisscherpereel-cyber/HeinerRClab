"""Rerun the findings that involve Adaptive agents (see heiner_abm/registered.py, revision notes of 6 October 2026).

Usage:  python tools/rerun_adaptive_findings.py <code root> directional <Bertrand|Cournot> <seed>
        python tools/rerun_adaptive_findings.py <code root> horse
        python tools/rerun_adaptive_findings.py <code root> preset

<code root> is a checkout containing heiner_abm/ (use "." for this one, or an export of an older commit to
reproduce the earlier numbers). Prints one JSON line. Settings match the README: the sidebar defaults (seed 1),
1,000 periods, 20 replications and H = 20 for the directional tournament (Cournot with phi = 0.1-0.4); 100
environments x 2 replications, sampling seed 12345, H = 20 for the forecasts; 6 replications (seeds 1-6) for the
presets, H = 25 and a 25-period judgement window for the Adaptive preset. Reported seeds for the directional
tournament: 1 and 3 (Bertrand), 1 and 42 (Cournot)."""
import sys, time, json
sys.path.insert(0, sys.argv[1])
import numpy as np
from heiner_abm.params import (AdaptiveParams, GlobalFirmParams, MarketParams, Scenario, StructuralParams,
                               linear_flex_firms)

NEW = hasattr(AdaptiveParams(), "feedback")


def base(rule="Bertrand", flex_slope=0.25, seed=1, selection="Always", window=None):
    """ui.common.base_scenario() with the sidebar DEFAULTS (seed 1, structural parameters as in DEFAULTS)."""
    firms = linear_flex_firms(n=4, slope=flex_slope, intercept=0.0, rule=rule, selection=selection, threshold=25.0,
                              desired_margin=5.0, foresight=0.0, noise=0.0)
    if rule == "Cournot":
        for f in firms:
            f.flex = min(f.flex, 1.0)
    ad = AdaptiveParams(memory=0.97, window=window or 20) if NEW else AdaptiveParams(memory=0.97)
    return Scenario(market=MarketParams(100.0, 10.0, 1500.0, 45.0, 80.0, 10.0), firms=firms,
                    firm_globals=GlobalFirmParams(200.0, 15.0, 0.0, 0.0, False), adaptive=ad,
                    structural=StructuralParams(False, 0.02, 15.0, 0.4, 20), periods=1000, burn_in=25, seed=seed)


which = sys.argv[2]
t0 = time.time()
if which == "directional":
    from heiner_abm.theories import EXPERIMENTS, run_tournament, scoreboard
    rule, seed = sys.argv[3], int(sys.argv[4])
    b = base(rule, 0.1 if rule == "Cournot" else 0.25, seed)
    out = run_tournament(b, 20, horizon=20)
    sb = scoreboard(out)
    rec = {r.theory: f"{r.matches}/{r.contradicted}/{r.inconclusive}" for r in sb.itertuples()}
    exps = {k: (o.observed, round(float(o.stat), 5), float(o.p), o.summary) for k, o in out.items()}
    print(json.dumps(dict(rule=rule, seed=seed, record=rec, experiments=exps)))
elif which == "horse":
    from heiner_abm.experiments import (EnvRanges, encompassing_test, horse_race, random_environments,
                                        rc_validation)
    envs = random_environments(base(), 100, EnvRanges(rules=("Bertrand", "Cournot")), seed=12345)
    df = rc_validation(envs, 2, horizon=20, continuation="default", discount=1.0, oos_split=0.5)
    res = {}
    for meas in ("full", "static"):
        hr = horse_race(df, meas)
        enc, gain = encompassing_test(df, meas)
        res[meas] = dict(auc={r.key: [round(r.auc, 3), round(r.lo, 3), round(r.hi, 3)] for r in hr.itertuples()},
                         enc={r.model: round(r.cv_auc, 3) for r in enc.itertuples()},
                         gain={k: round(float(v), 4) for k, v in gain.items()})
    print(json.dumps(dict(n=len(df), share=round(float((df["dyn_adv_eval"] > 0).mean()), 3),
                          adaptive_firms=int((df["selection"] == "Adaptive").sum()), **res)))
elif which == "preset":
    from heiner_abm.engine import run_batch
    from heiner_abm.analysis import reliability_table, market_table
    rows = []
    for sel, H, w in (("Always", 1, None), ("Adaptive", 25, 25)):
        scns = [base(selection=sel, seed=1 + r, window=w) for r in range(6)]
        res = run_batch(scns, horizon=H)
        mt, ft = market_table(res), reliability_table(res)
        slopes = [np.polyfit([f.flex for f in s.firms], ft[ft["market"] == i]["avg_profit"].to_numpy(), 1)[0] * 0.25
                  for i, s in enumerate(scns)]
        rows.append(dict(selection=sel, avg_price=round(float(mt["avg_price"].mean()), 2),
                         avg_cost=round(float(mt["avg_cost"].mean()), 2), sd_price=round(float(mt["sd_price"].mean()), 2),
                         slope=round(float(np.mean(slopes)), 1),
                         dev_rate=round(float((ft["deviations"] / ft["opportunities"]).mean()), 3),
                         dev_rate_by_flex=[round(float(x), 3) for x in
                                           (ft["deviations"] / ft["opportunities"]).groupby(ft["firm"]).mean()]))
    print(json.dumps(rows))
print(f"[{which} {sys.argv[3:] } {'new' if NEW else 'old'} code, {time.time() - t0:.0f}s]", file=sys.stderr)
