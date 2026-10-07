import time

import streamlit as st

from heiner_abm.workbench import presets, store
from heiner_abm.workbench.environments import ENVIRONMENTS
from ui.workbench_ui import MODE_LABELS, mode, set_spec

st.title("Decision under uncertainty lab")
st.markdown("**Choose a question → configure the experiment → check validity → run → interpret → export.** "
            "Every study in the laboratory follows these steps; the specialized pages remain available from the "
            "questions and results that lead to them.")
st.caption(f"Current mode: **{MODE_LABELS[mode()]}**. Explore offers simplified uncertainty presets and quick previews; "
           "Research exposes every setting, design validity checks and full background runs. Both use the same "
           "simulation engines. Switch in the sidebar.")


def _load(key):
    set_spec(presets.load(key, mode()))
    st.session_state["wb_step"] = "question"
    st.session_state["start_loaded"] = key


def _blank(env):
    set_spec(presets.blank(env, mode()))
    st.session_state["wb_step"] = "question"
    st.session_state["start_loaded"] = "custom"


st.header("Guided questions", divider="gray")
cols = st.columns(2)
for j, (key, p) in enumerate(presets.PRESETS.items()):
    with cols[j % 2].container(border=True):
        st.markdown(f"**{p.title}**")
        st.caption(p.summary)
        if p.spec is not None:
            env = ENVIRONMENTS[p.spec.environment]
            a, b = p.spec.question.comparison
            st.markdown(f"Primary outcome: *{env.outcome(p.spec.question.primary_outcome).label}* · Comparison: "
                        f"*{env.policy(a).label}* vs *{env.policy(b).label}* · Environment: *{env.label}*")
            if st.button("Use this question", key=f"start_use_{key}", type="primary"):
                _load(key)
                st.switch_page("app_pages/experiment.py")
        else:
            st.info(p.note, icon="👥")
            if st.button("Go to human experiments", key=f"start_use_{key}"):
                st.switch_page("app_pages/play_market.py")
        if p.registered_page:
            st.page_link(p.registered_page[0], label=p.registered_page[1])
if st.session_state.get("start_loaded"):
    st.page_link("app_pages/experiment.py", label="Continue with the experiment you loaded", icon="🧪")

st.header("Your own question", divider="gray")
c1, c2 = st.columns([3, 1])
env_key = c1.selectbox("Start from an environment", list(ENVIRONMENTS), format_func=lambda e: ENVIRONMENTS[e].label,
                       key="start_env")
if c2.button("Start blank", key="start_blank", width="stretch"):
    _blank(env_key)
    st.switch_page("app_pages/experiment.py")
st.caption(ENVIRONMENTS[env_key].description)

st.header("Example studies", divider="gray")
st.caption("Registered studies with frozen plans and reported results, each on its own page.")
for path, lab in (("app_pages/hypotheses.py", "Hypothesis tests in the shared market"),
                  ("app_pages/uncertainty.py", "Risk versus structural change"),
                  ("app_pages/arena.py", "Agent tournament (registered)"),
                  ("app_pages/rule_choice.py", "Rule choice: which rules emerge"),
                  ("app_pages/mechanisms.py", "Mechanisms: oracle versus learned reliability"),
                  ("app_pages/learnability.py", "When can reliability be learned? (registered)"),
                  ("app_pages/generalisation.py", "Generalization to other tasks")):
    st.page_link(path, label=lab)

st.header("Saved experiments", divider="gray")
saved = store.list_experiments()
if not saved:
    st.caption("None yet. Save an experiment from the Run step of the workspace.")


def _open(slug):
    set_spec(store.load_experiment(slug))
    st.session_state["start_loaded"] = slug


def _duplicate(slug, name):
    set_spec(store.duplicate_experiment(slug, name + " (copy)"))
    st.session_state["start_loaded"] = slug


for e in saved:
    c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
    c1.markdown(f"**{e['name']}** · {ENVIRONMENTS[e['environment']].label if e['environment'] in ENVIRONMENTS else e['environment']}"
                f"  \n{e['question'] or ''} · saved {time.strftime('%Y-%m-%d %H:%M', time.localtime(e['modified']))}")
    if c2.button("Open", key=f"start_open_{e['slug']}", width="stretch"):
        _open(e["slug"])
        st.switch_page("app_pages/experiment.py")
    c3.button("Duplicate", key=f"start_dup_{e['slug']}", on_click=_duplicate, args=(e["slug"], e["name"]),
              width="stretch")
    c4.button("Delete", key=f"start_del_{e['slug']}", on_click=store.delete_experiment, args=(e["slug"],),
              width="stretch")
