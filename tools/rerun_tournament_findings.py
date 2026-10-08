"""Rerun the agent tournament and everything registered that is built on it.

Usage:  python tools/rerun_tournament_findings.py            # every stage, each in its own process
        python tools/rerun_tournament_findings.py <stage>    # one stage: main, rep1, rep2, rep3, mech, choice
        python tools/rerun_tournament_findings.py <stage> --out <directory>

The tournament's plan hash covers the agent, simulator and analysis code (arena.code_digest), and the mechanism
study and the rule-choice study fold that hash into their own, so a change to the tuning rule invalidates all three
plus the signature-test and field-pattern fingerprints. This script reruns them and writes the numbers that
heiner_abm/registered.py and the README report, into docs/tournament_results/:

    main.json / rep1..3.json   plan hash, selected designs, tuned parameters, mean-profit and aggregate ranks,
                               PR1-PR6 verdicts. rep1-3 use arena.replicate's fresh training and test seeds.
    tuning.csv                 the main run's tuning log (free parameters, candidates, candidates per parameter)
    mechanisms.json            mechanism study: plan hash and the effects the theory pages quote
    rule_choice.json           endogenous rule choice: plan hash and verdicts

Each stage runs in a fresh interpreter: four tournament protocols in one process exhaust memory on this machine and
the run dies with a segmentation fault partway through. Stages are resumable - rerun one by name. Compare every plan
hash with heiner_abm/registered.py before reporting the numbers as registered results. About 10 minutes in total.
"""
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGES = ("main", "rep1", "rep2", "rep3", "mech", "choice")
LF = chr(10)


def show(obj):
    enc = sys.stdout.encoding or "utf-8"
    print(str(obj).encode(enc, "replace").decode(enc), flush=True)


def protocol_record(pr, A, R, label):
    """Run one tournament protocol and reduce it to the numbers the registry reports."""
    res = A.run_protocol(pr)
    rank = res.ranking.set_index("key")
    return res, {
        "label": label,
        "plan_hash": res.prereg_hash,
        "registered_plan_hash": R.TOURNAMENT_PLAN,
        "matches_registered_plan": res.prereg_hash == R.TOURNAMENT_PLAN,
        "exploratory": bool(res.exploratory),
        "train_seed": pr.train_seed,
        "test_seed": pr.test_seed,
        "budget_rule": pr.budget_rule,
        "budget_per_parameter": pr.budget_per_parameter,
        "design_budgets": {d: A.design_budget(pr, d) for d in A.DESIGNS},
        "selected_design": dict(res.tuned.design),
        "selected_design_name": {k: str(rank.loc[k, "design_name"]) for k in rank.index},
        "tuned_params": {d: {k: round(float(v), 6) for k, v in p.items()} for d, p in res.tuned.params.items()},
        "verdicts": {r["id"]: r["verdict"] for _, r in res.verdicts.iterrows()},
        "verdict_results": {r["id"]: r["result"] for _, r in res.verdicts.iterrows()},
        "mean_rank": {k: round(float(rank.loc[k, "mean_rank"]), 2) for k in rank.index},
        "aggregate_rank": {k: round(float(rank.loc[k, "aggregate_rank"]), 2) for k in rank.index},
        # selection rule vs always adjusting toward the same target (the README's second tournament table)
        "pareto": sorted(res.ranking.loc[res.ranking["pareto"].astype(bool), "key"].tolist())
                  if "pareto" in res.ranking else [],
        "best_aggregate": str(res.ranking.sort_values("aggregate_rank")["key"].iloc[0]),
        "selection": [{"target": r["target"], "select": r["select"], "design": r["design"],
                       "criterion": r["criterion"], "diff": round(float(r["diff"]), 1),
                       "lo": round(float(r["lo"]), 1), "hi": round(float(r["hi"]), 1),
                       "significant": bool(r["lo"] > 0 or r["hi"] < 0)}
                      for _, r in res.selection.iterrows()],
    }


def run_stage(stage: str, out: Path):
    from heiner_abm import arena as A
    from heiner_abm import registered as R

    out.mkdir(parents=True, exist_ok=True)
    pr = A.PREREG
    t0 = time.time()

    if stage == "main":
        show(f"plan {pr.digest} (registered {R.TOURNAMENT_PLAN}) - {pr.budget_per_parameter} candidates per free "
             f"parameter, {min(A.design_budget(pr, d) for d in A.DESIGNS)}-"
             f"{max(A.design_budget(pr, d) for d in A.DESIGNS)} per design")
        res, rec = protocol_record(pr, A, R, "main")
        res.tuning_log.to_csv(out / "tuning.csv", index=False, lineterminator=LF)
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        json.dump(rec, open(out / "main.json", "w"), indent=1)
        show(res.ranking[["agent", "design_name", "mean_rank", "aggregate_rank"]].to_string(index=False))
        show(res.verdicts[["id", "verdict", "result"]].to_string(index=False))

    elif stage.startswith("rep"):
        r = int(stage[3:])                       # arena.replicate's seed shift for replication r
        p = replace(pr, train_seed=pr.train_seed + 100 * r, test_seed=pr.test_seed + 100 * r)
        _, rec = protocol_record(p, A, R, stage)
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        json.dump(rec, open(out / f"{stage}.json", "w"), indent=1)
        show(f"{stage}: mean ranks " + json.dumps(rec["mean_rank"]))
        show(f"{stage}: verdicts " + json.dumps(rec["verdicts"]))

    elif stage in ("mech", "choice"):
        main = json.load(open(out / "main.json"))
        tuned = A.Tuned(design=main["selected_design"], params=main["tuned_params"])
        if stage == "mech":
            from heiner_abm import mechanisms as M
            plan = M.StudyPlan(tournament_plan=main["plan_hash"])
            res = M.run_study(plan, tuned)
            rec = {"plan_hash": plan.digest, "registered_plan_hash": R.STUDY_PLAN,
                   "matches_registered_plan": plan.digest == R.STUDY_PLAN}
            names = ("effects", "decomposition", "types", "verdicts")
            rec["boundary"] = {" ".join(map(str, k)) if isinstance(k, tuple) else str(k):
                               (float(v) if isinstance(v, (int, float)) else str(v))
                               for k, v in (res.boundary or {}).items()}
            prefix, path = "mechanism_", out / "mechanisms.json"
        else:
            from heiner_abm import rulechoice as RC
            plan = RC.ChoicePlan(tournament_plan=main["plan_hash"])
            res = RC.run_choice_study(plan, tuned)
            rec = {"plan_hash": plan.digest, "registered_plan_hash": R.CHOICE_PLAN,
                   "matches_registered_plan": plan.digest == R.CHOICE_PLAN}
            names = ("runs", "verdicts")
            prefix, path = "choice_", out / "rule_choice.json"
        for name in names:
            df = getattr(res, name, None)
            if df is not None and hasattr(df, "to_csv"):
                df.to_csv(out / f"{prefix}{name}.csv", index=False, lineterminator=LF)
                rec[name] = json.loads(df.to_json(orient="records"))
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        json.dump(rec, open(path, "w"), indent=1)
        show(f"{stage} plan {plan.digest} (registered {rec['registered_plan_hash']})")
        if hasattr(res, "verdicts") and hasattr(res.verdicts, "to_string"):
            show(res.verdicts.to_string(index=False))
    else:
        raise SystemExit(f"unknown stage {stage!r} (one of {', '.join(STAGES)})")
    show(f"[{stage}] {(time.time() - t0) / 60:.1f} min")


args = list(sys.argv[1:])
out = ROOT / "docs/tournament_results"
if "--out" in args:
    i = args.index("--out")
    out = Path(args[i + 1])
    del args[i:i + 2]

if args:
    run_stage(args[0], out)
else:
    t0 = time.time()
    for stage in STAGES:
        show(f"{LF}===== {stage} =====")
        rc = subprocess.call([sys.executable, "-u", str(Path(__file__).resolve()), stage, "--out", str(out)],
                             cwd=str(ROOT))
        if rc != 0:
            raise SystemExit(f"stage {stage} failed with exit code {rc}")
    show(f"{LF}TOTAL {(time.time() - t0) / 60:.1f} min")
    from heiner_abm import registered as R
    show(f"fingerprints now: special={R.source_fingerprint(R.FINDING_SOURCES['special'][1])} "
         f"patterns={R.source_fingerprint(R.FINDING_SOURCES['patterns'][1])}")
