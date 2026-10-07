"""Interface helpers for the experiment workspace: the one specification in session state, step status, widgets that
edit the specification, and shared result charts. Pages stay thin: simulation, statistics and storage live in
heiner_abm.workbench."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.workbench import presets
from heiner_abm.workbench.environments import ENVIRONMENTS, Control
from heiner_abm.workbench.spec import GROUP_LABELS, STEP_LABELS, STEPS, ExperimentSpec, changed_fields, step_status
from heiner_abm.workbench.validate import validate
from ui.common import CAT, style

STATUS_ICON = {"ok": "✅", "incomplete": "⚠️", "invalid": "⛔"}
MODE_LABELS = {"explore": "Explore (teaching and first look)", "research": "Research (controlled experiments)"}


# ------------------------------------------------------------------------------------------------ state
def mode() -> str:
    return st.session_state.get("wb_mode", "explore")


def get_spec() -> ExperimentSpec:
    if "wb_spec" not in st.session_state:
        st.session_state["wb_spec"] = presets.load("learnability", mode())
        st.session_state.setdefault("wb_rev", 0)
    spec = st.session_state["wb_spec"]
    spec.mode = mode()
    return spec


def set_spec(spec: ExperimentSpec) -> None:
    """Replace the specification (preset, saved experiment, environment change): widgets reinitialize from it."""
    spec.mode = mode()
    st.session_state["wb_spec"] = spec
    st.session_state["wb_rev"] = st.session_state.get("wb_rev", 0) + 1


def k(name: str) -> str:
    """Widget key tied to the current specification revision."""
    return f"wb{st.session_state.get('wb_rev', 0)}_{name}"


def statuses(spec: ExperimentSpec):
    issues = validate(spec)
    return issues, step_status(issues)


def step_label(step: str, status: Dict[str, str]) -> str:
    icon = STATUS_ICON[status[step]] if step != "run" else ("⛔" if "invalid" in status.values() else "▶️")
    return f"{icon} {STEP_LABELS[step]}"


def show_issues(issues, step: Optional[str] = None) -> None:
    for i in issues:
        if step is None or i.step == step:
            (st.error if i.level == "error" else st.warning)(
                (f"**{STEP_LABELS[i.step]}** · " if step is None else "") + i.message, icon="⛔" if i.level == "error"
                else "⚠️")


def outdated_notice(current: ExperimentSpec) -> bool:
    """Flag displayed results whose configuration differs from the current specification."""
    ran = st.session_state.get("wb_run_spec")
    if ran is None:
        return False
    diff = changed_fields(ran, current)
    if diff:
        st.warning("**Results are outdated:** the configuration changed after this run ("
                   + ", ".join(f"`{d}`" for d in diff[:8]) + (" …" if len(diff) > 8 else "")
                   + "). Run again to update them.", icon="🕒")
        return True
    return False


# ------------------------------------------------------------------------------------------------ widgets
def control_widget(c: Control, value: Any, key: str, disabled: bool = False, container=st):
    help_ = c.help or None
    if c.kind == "float":
        return container.number_input(c.label, float(c.lo), float(c.hi), float(value), float(c.step or 0.01),
                                       key=key, help=help_, disabled=disabled,
                                       format="%.4g" if (c.step or 1) < 0.01 else None)
    if c.kind == "int":
        return container.number_input(c.label, int(c.lo), int(c.hi), int(value), int(c.step or 1), key=key,
                                       help=help_, disabled=disabled)
    if c.kind == "bool":
        return container.checkbox(c.label, bool(value), key=key, help=help_, disabled=disabled)
    opts = list(c.options)
    return container.selectbox(c.label, opts, opts.index(value) if value in opts else 0, key=key, help=help_,
                                disabled=disabled)


def env_value(spec: ExperimentSpec, c: Control):
    env = ENVIRONMENTS[spec.environment]
    if c.key in {p.key for cand in env.candidates for p in cand.params}:
        return spec.candidate_params.get(c.key, c.default)
    return spec.env_params.get(c.key, c.default)


def write_env_value(spec: ExperimentSpec, c: Control, v: Any) -> None:
    env = ENVIRONMENTS[spec.environment]
    if c.key in {p.key for cand in env.candidates for p in cand.params}:
        spec.candidate_params[c.key] = v
    else:
        spec.env_params[c.key] = v


def controls_block(spec: ExperimentSpec, controls: List[Control], tag: str, container=st) -> None:
    """Render controls (essential first, advanced under an expander); the treatment variable is shown as such."""
    treated = spec.design.treatment.control
    basic = [c for c in controls if not c.advanced]
    adv = [c for c in controls if c.advanced]
    for c in basic:
        _one(spec, c, tag, treated, container)
    if adv:
        with container.expander("Advanced settings", expanded=False):
            for c in adv:
                _one(spec, c, tag, treated, st)


def _one(spec, c, tag, treated, container):
    if c.key == treated:
        container.caption(f"**{c.label}**: manipulated as the treatment (levels set in step 5: "
                          f"{', '.join(map(str, spec.design.treatment.levels))}).")
        return
    v = control_widget(c, env_value(spec, c), k(f"{tag}_{c.key}"), container=container)
    write_env_value(spec, c, v)


# ------------------------------------------------------------------------------------------------ summaries
def config_table(spec: ExperimentSpec) -> pd.DataFrame:
    env = ENVIRONMENTS[spec.environment]
    cand = env.candidate_for(spec)
    sel = [env.policy(p).label for p in spec.policies if p in {x.key for x in env.policies}
           and env.policy(p).kind == "selection"]
    comp = [env.policy(p).label for p in spec.policies if p in {x.key for x in env.policies}
            and env.policy(p).kind == "complete"]
    bench = [env.policy(p).label for p in spec.policies if p in {x.key for x in env.policies}
             and env.policy(p).kind == "benchmark"]
    info = env.information(spec)
    tr = spec.design.treatment
    rows = [("Question", spec.question.text or "—"),
            ("Primary outcome", env.outcome(spec.question.primary_outcome).label
             if spec.question.primary_outcome in {o.key for o in env.outcomes} else "—"),
            ("Comparison", " versus ".join(env.policy(p).label for p in spec.question.comparison
                                           if p in {x.key for x in env.policies}) or "—"),
            ("Environment", env.label),
            ("Candidate generation (held constant)", cand.label),
            ("Selection policies", ", ".join(sel) or "—"),
            ("Complete policies", ", ".join(comp) or "—"),
            ("Researcher benchmarks (not compared)", ", ".join(bench) or "—"),
            ("Information and learning", ", ".join(f"{a}: {b}" for a, b in info.items())),
            ("Treatment", f"{tr.control} ∈ {list(tr.levels)}" if tr.control else "none (one condition)"),
            ("Replications × periods", f"{spec.design.replications} × {spec.design.periods} (burn-in "
                                       f"{spec.design.burn_in})"),
            ("Tuning", "grid on training seeds" if spec.design.tuning == "grid" else "none (documented parameters)"),
            ("Seed", str(spec.design.seed))]
    return pd.DataFrame(rows, columns=["Element", "Setting"])


def uncertainty_table(spec: ExperimentSpec) -> pd.DataFrame:
    env = ENVIRONMENTS[spec.environment]
    rows = []
    for g in GROUP_LABELS:
        cs = [c for c in env.controls if c.group == g]
        if not cs:
            rows.append((GROUP_LABELS[g], "not applicable", env.not_applicable.get(g, "")))
        for c in cs:
            rows.append((GROUP_LABELS[g], c.label, str(env_value(spec, c))))
    return pd.DataFrame(rows, columns=["Group", "Setting", "Value"])


def ci_chart(s: pd.DataFrame, env, title: str, y_title: str, benchmarks=()) -> go.Figure:
    """Means with 95% intervals by treatment level, one trace per policy; benchmarks drawn dashed."""
    f = go.Figure()
    for j, (pol, g) in enumerate(s.groupby("policy", sort=False)):
        lab = env.policy(pol).label + (" (benchmark, not compared)" if pol in benchmarks else "")
        f.add_trace(go.Scatter(x=g["level"], y=g["mean"], mode="lines+markers", name=lab,
                               line=dict(color=CAT[j % len(CAT)], dash="dot" if pol in benchmarks else "solid"),
                               error_y=dict(type="data", symmetric=False, array=g["hi"] - g["mean"],
                                            arrayminus=g["mean"] - g["lo"], thickness=1)))
    f.update_yaxes(title=y_title)
    return style(f, 340, title=title)


def effect_chart(eff: pd.DataFrame, smallest: float, title: str, y_title: str) -> go.Figure:
    f = go.Figure(go.Scatter(x=eff["level"], y=eff["effect"], mode="markers", marker=dict(size=10, color=CAT[0]),
                             error_y=dict(type="data", symmetric=False, array=eff["hi"] - eff["effect"],
                                          arrayminus=eff["effect"] - eff["lo"], thickness=2), name="effect"))
    f.add_hline(y=0, line=dict(color="gray", dash="dot"))
    if smallest:
        for y in (smallest, -smallest):
            f.add_hline(y=y, line=dict(color="#eda100", dash="dash"))
    f.update_yaxes(title=y_title)
    return style(f, 320, title=title)
