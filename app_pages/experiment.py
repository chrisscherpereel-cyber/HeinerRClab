import time

import pandas as pd
import streamlit as st

from heiner_abm.workbench import execution, presets, store
from heiner_abm.workbench.environments import ENVIRONMENTS
from heiner_abm.workbench.spec import GROUP_LABELS, STEPS, Treatment
from ui.common import set_engine
from ui.workbench_ui import (MODE_LABELS, config_table, controls_block, get_spec, k, mode, outdated_notice, set_spec,
                             show_issues, statuses, step_label, uncertainty_table)

st.title("Experiment")
spec = get_spec()
env = ENVIRONMENTS[spec.environment]
set_engine(env.engine)
st.caption(f"Mode: **{MODE_LABELS[mode()]}** (switch in the sidebar). One specification is edited across the six "
           "steps; you can return to any step. ⛔ marks a step that blocks a run, ⚠️ one with warnings.")

issues, status = statuses(spec)
st.session_state.setdefault("wb_step", "question")


def _go(step):
    st.session_state["wb_step"] = step


step = st.radio("Step", STEPS, key="wb_step", horizontal=True, format_func=lambda s: step_label(s, status),
                label_visibility="collapsed")
st.divider()
pols = {p.key: p for p in env.policies}

# ============================================================================================ 1. Question
if step == "question":
    st.subheader("1. Question")
    st.markdown("Pick a guided question (it loads a documented, editable preset) or write your own.")
    names = {"custom": "Custom question (keep the current settings)",
             **{key: p.title for key, p in presets.PRESETS.items() if p.spec is not None}}

    def _pick(key):
        choice = st.session_state.get(key)
        if choice and choice != "custom":
            set_spec(presets.load(choice, mode()))
        elif choice == "custom":
            spec.question.preset = "custom"

    pick_key = k("preset")
    current = spec.question.preset if spec.question.preset in names else "custom"
    st.selectbox("Guided questions", list(names), index=list(names).index(current), format_func=names.get,
                 key=pick_key, on_change=_pick, args=(pick_key,),
                 help="Choosing a guided question loads its documented preset at once (question, environment, "
                      "policies, comparison and design); everything stays editable.")
    if current != "custom":
        p = presets.PRESETS[current]
        st.caption(p.summary + (f" Related registered study: *{p.registered_page[1]}*." if p.registered_page else ""))
    spec.name = st.text_input("Experiment name", spec.name, key=k("name"))
    spec.question.text = st.text_area("Research question", spec.question.text, key=k("q_text"), height=80)
    spec.question.rationale = st.text_area("Why this comparison answers it (optional)", spec.question.rationale,
                                           key=k("q_rat"), height=80)
    outs = {o.key: o.label for o in env.outcomes}
    c1, c2, c3 = st.columns(3)
    spec.question.primary_outcome = c1.selectbox(
        "Primary outcome", list(outs), format_func=outs.get, key=k("q_out"),
        index=list(outs).index(spec.question.primary_outcome) if spec.question.primary_outcome in outs else 0)
    comparable = [x for x in spec.policies if x in pols and pols[x].kind != "benchmark"] or \
        [x.key for x in env.policies if x.kind != "benchmark"]
    a0, b0 = spec.question.comparison
    a = c2.selectbox("Treatment policy", comparable, format_func=lambda x: pols[x].label, key=k("q_a"),
                     index=comparable.index(a0) if a0 in comparable else 0)
    b = c3.selectbox("Reference policy", comparable, format_func=lambda x: pols[x].label, key=k("q_b"),
                     index=comparable.index(b0) if b0 in comparable else min(1, len(comparable) - 1))
    spec.question.comparison = (a, b)
    spec.question.smallest_effect = st.number_input(
        "Smallest effect of practical interest (in outcome units; 0 = any difference)", 0.0, 1e6,
        float(spec.question.smallest_effect), key=k("q_small"),
        help="Used to interpret the interval: a difference smaller than this is not treated as practically important.")
    show_issues(issues, "question")

# ============================================================================================ 2. Environment
elif step == "environment":
    st.subheader("2. Environment")
    keys = list(ENVIRONMENTS)
    new = st.radio("Task environment", keys, format_func=lambda e: ENVIRONMENTS[e].label, key=k("env"),
                   index=keys.index(spec.environment), horizontal=True)
    st.caption(ENVIRONMENTS[new].description)
    if new != spec.environment:
        if st.button(f"Switch to {ENVIRONMENTS[new].label} (resets policies and settings)", key="wb_switch_env"):
            q = spec.question.text
            ns = presets.blank(new, mode())
            ns.question.text, ns.name = q, spec.name
            set_spec(ns)
            st.rerun()
        st.stop()
    cand_keys = [c.key for c in env.candidates]
    spec.candidate = st.radio("Candidate generation (held constant across selection policies)", cand_keys,
                              format_func=lambda c: next(x.label for x in env.candidates if x.key == c),
                              key=k("cand"), index=cand_keys.index(spec.candidate) if spec.candidate in cand_keys
                              else 0)
    st.caption(env.candidate_for(spec).description)
    st.markdown("#### Uncertainty")
    if mode() == "explore":
        levels = presets.EXPLORE_LEVELS.get(env.key, {})
        c1, c2 = st.columns([1, 3])
        lvl = c1.radio("Simplified level", list(levels), key=k("unc_level"), horizontal=False)
        changes = levels[lvl]
        c2.dataframe(pd.DataFrame([(env.control(kk).label, v) for kk, v in changes.items()],
                                  columns=["Underlying setting it changes", "Value"]), hide_index=True,
                     width="stretch")
        if c2.button(f"Apply '{lvl}'", key="wb_apply_level"):
            for kk, v in changes.items():
                if kk != spec.design.treatment.control:
                    spec.env_params[kk] = v
            set_spec(spec)
            st.rerun()
        st.caption("Explore mode shows the full settings below; the simplified level only fills them in.")
    for g, lab in GROUP_LABELS.items():
        cs = [c for c in env.controls if c.group == g]
        with st.expander(lab, expanded=bool(cs) and g in ("outcome", "observation", "change")):
            if not cs:
                st.caption("Not applicable here. " + env.not_applicable.get(g, ""))
                continue
            controls_block(spec, cs, f"env_{g}")
    show_issues(issues, "environment")

# ============================================================================================ 3. Agents
elif step == "agents":
    st.subheader("3. Agents")
    st.markdown(f"Candidates come from **{env.candidate_for(spec).label}** for every selection policy, so selection "
                "policies differ only in when they act on the same candidate. Complete policies generate their own "
                "actions; their assumptions are listed. Researcher benchmarks use information agents do not have and "
                "are shown beside the comparison, never in it.")
    kind_label = {"selection": "Selection policy", "complete": "Complete policy", "benchmark": "Researcher benchmark"}
    df = pd.DataFrame([dict(Include=p.key in spec.policies, Policy=p.label, Type=kind_label[p.kind],
                            Status="proposed extension" if p.extension else "", Behavior=p.description, Uses=p.uses,
                            Assumptions=p.assumptions, key=p.key) for p in env.policies])
    ed = st.data_editor(df, hide_index=True, width="stretch", key=k("agents_table"),
                        disabled=[c for c in df.columns if c != "Include"],
                        column_config={"key": None,
                                       "Include": st.column_config.CheckboxColumn("Include", width="small")})
    spec.policies = tuple(ed.loc[ed["Include"], "key"])
    agent_controls = [c for c in env.all_controls() if c.group == "agents"]
    if agent_controls:
        st.markdown("#### Policy settings")
        controls_block(spec, agent_controls, "agents")
    show_issues(issues, "agents")

# ============================================================================================ 4. Information
elif step == "information":
    st.subheader("4. Information and feedback")
    from heiner_abm.information import ENGINES
    eng = ENGINES[env.engine]
    if env.key == "market":
        c1, c2 = st.columns(2)
        fb = c1.radio("Feedback the Adaptive gates learn from", ["estimated", "chosen"], key=k("info_fb"),
                      index=["estimated", "chosen"].index(spec.information.get("feedback", "estimated"))
                      if spec.information.get("feedback", "estimated") in ("estimated", "chosen") else 0,
                      format_func={"estimated": "Estimated counterfactual (own model, observed data)",
                                   "chosen": "Chosen-action feedback only"}.get)
        rv = c2.checkbox("Rivals' individual outputs visible", bool(spec.information.get("rivals_visible", True)),
                         key=k("info_rivals"))
        spec.information.update(feedback=fb, rivals_visible=rv)
        st.caption("Observation noise, delay, announcement of shifts and the demand model are set in step 2 "
                   "(uncertainty panel); they appear in the matrix below.")
    else:
        st.caption(f"This environment has a fixed information design: {eng.name}. Noise, missing observations and "
                   "change are set in step 2.")
    info = env.information(spec)
    st.markdown("#### Access matrix")
    rows = []
    for p in spec.policies:
        if p not in pols:
            continue
        pi = pols[p]
        rows.append({"Policy": pi.label, "Reads": pi.uses,
                     "Feedback": info.get("feedback", "—"),
                     "Observation noise": info.get("obs_noise", "see step 2"),
                     "Delay": info.get("obs_delay", 0),
                     "Researcher-only information": "yes (benchmark)" if pi.kind == "benchmark" else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("#### What this environment can deliver")
    st.dataframe(pd.DataFrame([(kk, v) for kk, v in eng.controls.items()], columns=["Information control", "Support"]),
                 hide_index=True, width="stretch")
    show_issues(issues, "information")

# ============================================================================================ 5. Design
elif step == "design":
    st.subheader("5. Design")
    d = spec.design
    treatable = [c for c in env.all_controls() if c.treatable]
    opts = [""] + [c.key for c in treatable]
    lab = {c.key: c.label for c in treatable}
    c1, c2 = st.columns([1, 2])
    tc = c1.selectbox("Treatment (manipulated setting)", opts, format_func=lambda x: lab.get(x, "none (one "
                                                                                                   "condition)"),
                      key=k("tr_control"), index=opts.index(d.treatment.control) if d.treatment.control in opts else 0)
    if tc:
        ctrl = env.control(tc)
        default_levels = ", ".join(map(str, d.treatment.levels)) if tc == d.treatment.control else ""
        txt = c2.text_input("Levels (comma-separated)" + (f"; options: {', '.join(ctrl.options)}" if ctrl.options
                                                            else f" in [{ctrl.lo}, {ctrl.hi}]"), default_levels,
                            key=k(f"tr_levels_{tc}"))
        vals = []
        for v in [x.strip() for x in txt.split(",") if x.strip()]:
            try:
                vals.append(int(v) if ctrl.kind == "int" else float(v) if ctrl.kind == "float" else v)
            except ValueError:
                vals.append(v)
        d.treatment = Treatment(tc, tuple(vals))
    else:
        d.treatment = Treatment()
    c1, c2, c3, c4 = st.columns(4)
    d.replications = int(c1.number_input("Replications per condition", 1, 500, int(d.replications), key=k("reps"),
                                         help="Independent environments (paths, markets or landscapes)."))
    d.periods = int(c2.number_input("Periods", 50, 20000, int(d.periods), 50, key=k("periods")))
    d.burn_in = int(c3.number_input("Burn-in", 0, 5000, int(d.burn_in), 10, key=k("burn")))
    d.seed = int(c4.number_input("Seed", 0, 10_000_000, int(d.seed), key=k("seed")))
    c1, c2 = st.columns(2)
    d.tuning = c1.radio("Tuning", ["none", "grid"], key=k("tuning"), index=["none", "grid"].index(d.tuning),
                        format_func={"none": "None (documented parameters)",
                                     "grid": "Grid search on separate training seeds"}.get, horizontal=True)
    c2.caption(env.tuning_note)
    if d.tuning == "grid":
        d.train_replications = int(st.number_input("Training replications (disjoint seeds)", 1, 50,
                                                   int(d.train_replications), key=k("train")))
    outs = {o.key: o.label for o in env.outcomes if o.key != spec.question.primary_outcome}
    d.outcomes = tuple(st.multiselect("Secondary outcomes", list(outs), [o for o in d.outcomes if o in outs],
                                      format_func=outs.get, key=k("outs")))
    if mode() == "research":
        d.n_boot = int(st.number_input("Bootstrap resamples", 200, 20000, int(d.n_boot), 100, key=k("boot")))
    st.markdown("#### Validity checks")
    st.caption("Seeds: training (tuning) and test (reported) blocks never overlap, and all policies in a condition "
               "see the same test environments, so differences are paired. " + env.cluster_note)
    if issues:
        show_issues(issues)
    else:
        st.success("No problems found.", icon="✅")

# ============================================================================================ 6. Run
else:
    st.subheader("6. Run")
    st.markdown("**Configuration**")
    st.dataframe(config_table(spec), hide_index=True, width="stretch")
    st.markdown("**Uncertainty settings**")
    st.dataframe(uncertainty_table(spec), hide_index=True, width="stretch")
    errors = [i for i in issues if i.level == "error"]
    if errors:
        st.error("Fix the marked steps before running.", icon="⛔")
        show_issues(errors)
    outdated_notice(spec)
    est_p, est_r = execution.estimate(spec, "preview"), execution.estimate(spec, "research")
    st.caption(f"Workload: preview ≈ {est_p['policy_periods']:,} policy-periods (~{est_p['seconds']:.0f} s); "
               f"research ≈ {est_r['policy_periods']:,} policy-periods (~{est_r['seconds']:.0f} s, rough estimate). "
               "Results are cached by the complete configuration and code version.")
    c1, c2, c3, c4 = st.columns(4)
    if c3.button("Save experiment", width="stretch", key="wb_save"):
        store.save_experiment(spec)
        st.toast(f"Saved '{spec.name}'.")
    if c4.button("Duplicate", width="stretch", key="wb_dup"):
        dup = spec.copy()
        dup.name = spec.name + " (copy)"
        store.save_experiment(dup)
        set_spec(dup)
        st.toast(f"Saved and opened '{dup.name}'.")
    if c1.button("Preview run", type="primary", width="stretch", disabled=bool(errors), key="wb_preview",
                 help="A reduced run (≤ 3 replications, ≤ 300 periods) in this page, for orientation only."):
        hit = store.cached(spec, "preview")
        if hit is None:
            bar = st.progress(0.0, "Starting")
            hit = execution.execute(spec, "preview", progress=lambda f, m: bar.progress(min(f, 1.0), m))
            store.save_run(hit)
            bar.empty()
        else:
            st.info("Loaded from the run store (same configuration and code version).")
        st.session_state["wb_run_key"], st.session_state["wb_run_spec"] = hit.key, spec.copy()
        st.success("Preview complete. Open **Results** in the sidebar to interpret it.", icon="✅")
    if mode() == "research":
        if c2.button("Research run (background)", width="stretch", disabled=bool(errors), key="wb_research",
                     help="The full design, in a separate process; this page shows its progress and can cancel it."):
            hit = store.cached(spec, "research")
            if hit is not None:
                st.session_state["wb_run_key"], st.session_state["wb_run_spec"] = hit.key, spec.copy()
                st.info("Loaded from the run store (same configuration and code version).")
            else:
                st.session_state["wb_bg_key"] = execution.start_background(spec, "research")
                st.session_state["wb_bg_spec"] = spec.copy()
    else:
        c2.caption("Research runs (full design, background process) are available in Research mode.")

    bg = st.session_state.get("wb_bg_key")
    if bg:
        @st.fragment(run_every=2)
        def _poll():
            p = store.progress(bg) or {"fraction": 0.0, "message": "Starting", "status": "running"}
            st.progress(min(float(p["fraction"]), 1.0), f"Research run {bg}: {p['message']}")
            if p["status"] == "running":
                if st.button("Cancel research run", key="wb_cancel"):
                    store.request_cancel(bg)
                    st.toast("Cancellation requested; the run stops after the current block.")
            else:
                st.session_state.pop("wb_bg_key", None)
                if p["status"] == "complete":
                    st.session_state["wb_run_key"] = bg
                    st.session_state["wb_run_spec"] = st.session_state.pop("wb_bg_spec", spec.copy())
                    st.success("Research run complete. Open **Results** to interpret it.", icon="✅")
                else:
                    st.warning(f"Research run ended: {p['message']}.")
        _poll()
    if st.session_state.get("wb_run_key"):
        st.page_link("app_pages/results.py", label="Open the results of the latest run", icon="📊")

st.divider()
i = STEPS.index(step)
c1, _, c3 = st.columns([1, 4, 1])
if i > 0:
    c1.button("← Back", on_click=_go, args=(STEPS[i - 1],), width="stretch", key="wb_back")
if i < len(STEPS) - 1:
    c3.button("Next →", on_click=_go, args=(STEPS[i + 1],), width="stretch", key="wb_next")
