import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.workbench import analysis, store
from heiner_abm.workbench.environments import ENVIRONMENTS
from heiner_abm.workbench.provenance import FEEDBACK_ORIGINS, researcher_only_columns
from ui.common import CAT, set_engine, style
from ui.workbench_ui import ci_chart, config_table, effect_chart, get_spec, outdated_notice

st.title("Results")
runs = store.list_runs()
current = st.session_state.get("wb_run_key")
keys = [r["key"] for r in runs if r["status"] == "complete"]
if current and current not in keys and store.load_run(current) is not None:
    keys.insert(0, current)
if not keys:
    st.info("No completed runs yet. Configure and run an experiment first.")
    st.page_link("app_pages/experiment.py", label="Go to the Experiment workspace", icon="🧪")
    st.stop()
label = {r["key"]: f"{r['name']} · {r['kind']} · {r['key']}" for r in runs}
pick = st.selectbox("Run", keys, index=keys.index(current) if current in keys else 0, format_func=lambda x: label.get(x, x),
                    key="res_pick")
run = store.load_run(pick)
spec = run.spec
env = ENVIRONMENTS[spec.environment]
set_engine(env.engine)
if pick == current:
    outdated_notice(get_spec())
if run.kind == "preview":
    st.warning("**Preview run**: reduced replications and periods, for orientation only. Use a research run for "
               "conclusions.", icon="🔍")

ov = analysis.overview(run)
a, b = spec.question.comparison
bench = [p for p in spec.policies if env.policy(p).kind == "benchmark"]
st.markdown(f"**Question:** {ov['question']}  \n**Comparison:** {ov['comparison']}  \n**Primary outcome:** "
            f"{ov['outcome']} ({'higher is better' if ov['higher_better'] else 'lower is better' if ov['higher_better'] is False else 'descriptive'})"
            f"  \n**Evidence:** {ov['evidence']}")

tabs = st.tabs(["Overview", "Performance", "Behavior", "Mechanisms", "Robustness", "Decision inspector",
                "Details & export"])

with tabs[0]:
    eff = ov["effects"]
    if eff.empty:
        st.info("The comparison policies have no results in this run.")
    else:
        st.plotly_chart(effect_chart(eff, ov["smallest"], f"{ov['comparison']}: paired difference with 95% interval",
                                     ov["outcome"] + " (difference)"))
        for _, r in eff.iterrows():
            st.markdown(f"- **{r['level']}**: {r['effect']:+.4g} [{r['lo']:.4g}, {r['hi']:.4g}] over {int(r['n'])} "
                        f"replications ({r['share_positive']:.0%} of replications favor the treatment policy). "
                        + analysis.interpret(r["effect"], r["lo"], r["hi"], ov["smallest"], ov["higher_better"]))
        st.caption("Paired by replication: both policies faced the same environments. This is the stated comparison "
                   "only; other policies are shown in the other tabs without a ranking.")


def _outcome_tab(tab: str):
    outs = [o for o in env.outcomes if o.tab == tab and o.key in run.trials]
    if not outs:
        st.caption("No outcomes of this kind are defined for this environment.")
        return
    for o in outs:
        s = analysis.summary(run.trials, o.key, min(spec.design.n_boot, 1000))
        if s.empty or s["mean"].isna().all():
            continue
        st.plotly_chart(ci_chart(s, env, o.label, o.label, bench))


with tabs[1]:
    _outcome_tab("performance")
with tabs[2]:
    _outcome_tab("behavior")
with tabs[3]:
    _outcome_tab("mechanisms")
    st.markdown("**Specialized mechanism analyses** (market):")
    for path, lab in (("app_pages/mechanisms.py", "Oracle versus learned reliability"),
                      ("app_pages/reliability_gates.py", "Reliability gates: calibration and coverage"),
                      ("app_pages/dynamic_rc.py", "Dynamic reliability condition"),
                      ("app_pages/cd_gap.py", "Difficulty–competence gap")):
        st.page_link(path, label=lab)

with tabs[4]:
    rb = analysis.robustness(run, 500)
    if "halves" in rb:
        h = rb["halves"]
        st.markdown("**Split replications.** The primary comparison in two disjoint halves of the replications; "
                    "agreement between halves is a minimal check of stability.")
        st.dataframe(h[["block", "level", "effect", "lo", "hi", "n"]].round(4), hide_index=True, width="stretch")
    pr = rb["per_replication"]
    if len(pr):
        f = go.Figure()
        for j, (lvl, g) in enumerate(pr.groupby("level", sort=False)):
            f.add_trace(go.Box(y=g["difference"], name=str(lvl), boxpoints="all", marker_color=CAT[j % len(CAT)]))
        f.add_hline(y=0, line=dict(color="gray", dash="dot"))
        st.plotly_chart(style(f, 320, title="Difference per replication (treatment − reference)"))
    st.markdown("**Alternative environments and parameters:** change the treatment or environment in the Experiment "
                "workspace, or see the registered robustness studies:")
    st.page_link("app_pages/generalisation.py", label="Generalization across tasks")
    st.page_link("app_pages/nk.py", label="NK landscapes: interdependence, noise and change sweeps")

with tabs[5]:
    st.caption("What an agent observed, believed and proposed in one period, why it acted or not, what followed, and "
               "where its feedback came from, when that outcome matured and when it reached the agent. Columns "
               "starting with 'researcher' were never available to any agent at any time. How many replications are "
               "kept is set by 'Decision logs kept' in the design step.")
    if not run.traces:
        st.info("This run holds no decision traces.")
    else:
        c1, c2 = st.columns(2)
        lvl = c1.selectbox("Condition", list(run.traces), key="res_tr_level")
        pol = c2.selectbox("Policy", list(run.traces[lvl]), format_func=lambda p: env.policy(p).label,
                           key="res_tr_pol")
        tr = run.traces[lvl][pol]
        if "replication" in tr and tr["replication"].nunique() > 1:
            reps = sorted(int(r) for r in tr["replication"].unique())
            rep = st.selectbox("Replication", reps, key="res_tr_rep",
                               help="Every replication is kept for this run (decision logs: all).")
            tr = tr[tr["replication"] == rep].reset_index(drop=True)
        t = st.slider("Period", 0, len(tr) - 1, min(spec.design.burn_in, len(tr) - 1), key="res_tr_t")
        row = tr.iloc[t]
        groups = {"Before deciding": ["observed", "observed_price", "observed_candidate", "signal", "configuration",
                                      "belief_sd", "belief_cost", "belief_best_reply", "predicted_gain",
                                      "lower_bound", "bin", "default_order", "default_action", "current_output",
                                      "proposed", "opportunity"],
                  "Decision": ["acted", "chosen", "reason"],
                  "After deciding": ["realized_payoff", "realized_cost"],
                  "Feedback": ["feedback", "feedback_origin", "feedback_matured_period",
                               "feedback_release_period"]}
        def _fmt(kk, v):
            if kk in ("feedback_release_period", "feedback_available"):
                return "never released" if v is None or (isinstance(v, (int, float, np.integer)) and v < 0) \
                    else f"end of period {int(v)}"
            if kk == "feedback_matured_period":
                return "—" if v is None else f"end of period {int(v)}"
            if kk == "feedback_origin":
                return FEEDBACK_ORIGINS.get(str(v), str(v))
            if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
                return "—"
            if isinstance(v, (float, np.floating)):
                return f"{v:.4g}"
            return str(v)

        cols = st.columns(len(groups))
        for (g, keys_), c in zip(groups.items(), cols):
            c.markdown(f"**{g}**")
            for kk in keys_:
                if kk in tr:
                    c.markdown(f"{kk.replace('_', ' ').capitalize()}: **{_fmt(kk, row[kk])}**")
        res_cols = researcher_only_columns(tr.columns)
        if res_cols:
            st.caption("Researcher-only: " + ", ".join(f"{c.replace('researcher_', '')} = {row[c]:.4g}"
                                                       for c in res_cols if isinstance(row[c], (int, float, np.floating))))
        lo, hi = max(0, t - 40), min(len(tr), t + 40)
        win = tr.iloc[lo:hi]
        ycols = [c for c in ("proposed", "chosen", "default_order", "current_output") if c in win and
                 pd.api.types.is_numeric_dtype(win[c])]
        if ycols:
            f = go.Figure([go.Scatter(x=win["period"], y=win[c], name=c.replace("_", " "), mode="lines",
                                      line=dict(color=CAT[j])) for j, c in enumerate(ycols)])
            acted = win[win["acted"].astype(bool)]
            f.add_trace(go.Scatter(x=acted["period"], y=acted[ycols[0]], mode="markers", name="acted",
                                   marker=dict(color="#e34948", size=6)))
            f.add_vline(x=t, line=dict(color="gray", dash="dot"))
            st.plotly_chart(style(f, 300, title="Around the selected period"))
        st.dataframe(win, hide_index=True, width="stretch", height=240)

with tabs[6]:
    st.dataframe(config_table(spec), hide_index=True, width="stretch")
    man = getattr(run, "manifest", {}) or {}
    if man:
        st.markdown("**Manifest** — configuration, policies, tuning, costs, information assumptions, code "
                    "fingerprint, software versions, seed plan and the decision-log schema.")
        man_code = man.get("code", {})
        man_soft = man.get("software", {})
        man_pkgs = man_soft.get("packages") or {}
        st.caption(f"Code fingerprint {man_code.get('fingerprint')} over {man_code.get('n_modules')} first-party "
                   f"modules. Python {man_soft.get('python')}, numpy {man_pkgs.get('numpy')}, "
                   f"pandas {man_pkgs.get('pandas')} on {man_soft.get('platform')}.")
        for cost_level, cost_rows in (man.get("costs") or {}).items():
            charged = [c for c in cost_rows if c.get("in_force")]
            st.caption(f"Charges in force ({cost_level}): "
                       + ("; ".join(f"{c['name']} = {c['rate']} ({c['units']})" for c in charged)
                          if charged else "none - this is a zero-cost control."))
        st.json(man, expanded=False)
    for pass_name, pass_df, pass_note in (
            ("Pilot pass", getattr(run, "pilot", None),
             "Pilot seeds. Not a reported result: used to size effects and workload."),
            ("Validation pass", getattr(run, "validation", None),
             "Validation seeds: the same comparison on environments it was not estimated on.")):
        if pass_df is not None and len(pass_df):
            st.markdown(f"**{pass_name}**")
            st.caption(pass_note)
            st.dataframe(pass_df, hide_index=True, width="stretch", height=200)
    c1, c2 = st.columns(2)
    c1.markdown("**Provenance**")
    c1.json(run.provenance, expanded=False)
    c2.markdown("**Specification**")
    c2.json(spec.to_dict(), expanded=False)
    st.markdown("**Trial data** (one row per policy, replication and condition)")
    st.dataframe(run.trials, hide_index=True, width="stretch", height=240)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Reproducible bundle (ZIP)", store.export_zip(run.key), file_name=f"run_{run.key}.zip",
                       mime="application/zip", width="stretch",
                       help="Specification, provenance, trial data and traces from the stored run, with the command "
                            "that reproduces it.")
    c2.download_button("Trial data (CSV)", run.trials.to_csv(index=False).encode(), file_name=f"trials_{run.key}.csv",
                       mime="text/csv", width="stretch")
    c3.download_button("Specification (JSON)", spec.to_json().encode(), file_name=f"spec_{run.key}.json",
                       mime="application/json", width="stretch")
    st.code(f"python -m heiner_abm.workbench run spec_{run.key}.json --kind {run.kind}", language="bash")
