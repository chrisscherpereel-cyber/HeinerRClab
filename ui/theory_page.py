"""Renders one theory page with the same sections for every theory."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from heiner_abm.arena import DESIGNS, THEORY_DESIGNS
from heiner_abm.registered import DIRECTIONAL, MECHANISMS, STUDY_PLAN, TOURNAMENT, TOURNAMENT_PLAN
from heiner_abm.theories import EXPERIMENTS
from heiner_abm.theory_content import THEORY_BY_KEY
from ui.illustrations import ILLUSTRATIONS

SYM = {"+": "↑ rises", "-": "↓ falls", "0": "no effect", ">=0": "≥ 0 (never harmful)", None: "—"}


def _lit():
    import heiner_abm.literature as lit       # looked up at call time (see ui.common._literature)
    return lit


def render(key: str):
    t = THEORY_BY_KEY[key]
    lit = _lit()
    st.title(f"{t.icon} {t.title}")
    st.markdown(f"*{t.tagline}*")
    st.caption("One of nine theories of decision making under uncertainty tested in this laboratory. Every theory "
               "page has the same sections; the *Overview* compares them side by side.")

    st.header("1 · Origins and core idea", divider="gray")
    st.markdown(t.origins)

    st.header("2 · Formal core", divider="gray")
    st.markdown(t.formal)

    st.header("3 · What it says about flexibility under uncertainty", divider="gray")
    with st.container(border=True):
        st.markdown(t.flexibility)
        c = st.columns(3)
        c[0].markdown(f"**Uncertainty is…**  \n{t.uncertainty_view}")
        c[1].markdown(f"**Behavior changes…**  \n{t.when_to_change}")
        c[2].markdown(f"**More uncertainty makes flexibility…**  \n{t.more_uncertainty}")

    st.header("4 · Explore", divider="gray")
    ILLUSTRATIONS[key]()

    st.header("5 · In this laboratory", divider="gray")
    if t.lab_notes:
        st.markdown(t.lab_notes)
    st.markdown("**Agent designs in the tournament** (each theory enters with whichever design scores higher on "
                "training environments)")
    st.dataframe(pd.DataFrame([dict(Design=DESIGNS[d].name, Rule=DESIGNS[d].rule,
                                    Parameters=", ".join(f"{n} ∈ [{s.lo:g}, {s.hi:g}]"
                                                         for n, s in DESIGNS[d].SPACE.items()) or "–",
                                    Sources=lit.cite(*DESIGNS[d].sources)) for d in THEORY_DESIGNS[key]]),
                 hide_index=True, width="stretch")
    hyps = [lit.HYPOTHESIS_BY_ID[h] for h in t.hypotheses if h in lit.HYPOTHESIS_BY_ID]
    if hyps:
        st.markdown("**Hypotheses this theory informs** (each card in the app shows the research on both sides)")
        st.dataframe(pd.DataFrame([dict(ID=h.hid, Hypothesis=h.title, Alternative=h.alt_label, Tested=h.where)
                                   for h in hyps]), hide_index=True, width="stretch")
    if t.tournament_key:
        st.markdown("**Directional predictions** in the tournament of experiments (*Competing theories* page)")
        st.dataframe(pd.DataFrame([dict(Experiment=e.title, Prediction=SYM[e.predictions.get(t.tournament_key)],
                                        Reasoning=e.why.get(t.tournament_key, "No clear prediction."))
                                   for e in EXPERIMENTS]), hide_index=True, width="stretch")

    st.header("6 · How it fared", divider="gray")
    r = TOURNAMENT.get(key)
    if r:
        c = st.columns(3)
        c[0].metric("Profit rank, main run (1 = best of 10)", f"{r['profit'][0]:.2f}",
                    help="Mean rank by profit in held-out mixed markets.")
        c[1].metric("Profit rank, three replications", f"{min(r['profit'][1:]):.2f}–{max(r['profit'][1:]):.2f}",
                    help="Range of the mean profit rank when the whole protocol is repeated with fresh seeds.")
        c[2].metric("Aggregate rank, six criteria", f"{min(r['aggregate']):.2f}–{max(r['aggregate']):.2f}",
                    help="Range over the main run and replications of the mean rank on profit, downside risk, "
                         "survival, volatility, regret and worst case.")
        st.caption(f"Agent tournament, registered plan `{TOURNAMENT_PLAN}`. Design entering the tournament: "
                   f"{r['design']}.")
    if t.tournament_key and t.tournament_key in DIRECTIONAL:
        d = DIRECTIONAL[t.tournament_key]
        st.markdown("**Tournament of experiments** (matches / contradictions / inconclusive, two seeds each): "
                    + "; ".join(f"{m} market {a} and {b}" for m, (a, b) in d.items()))
    if key in MECHANISMS:
        st.markdown(f"**Mechanism study** (registered plan `{STUDY_PLAN}`): {MECHANISMS[key]}")
    st.caption("Rerun any of these on the *Competing theories*, *Agent tournament* and *Mechanisms* pages.")

    st.header("7 · Strengths and limits", divider="gray")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("\n".join(f"* {x}" for x in t.strengths))
    with c2:
        st.markdown("**Limits**")
        st.markdown("\n".join(f"* {x}" for x in t.limits))

    st.header("8 · Key references", divider="gray")
    st.markdown("\n".join(f"* {ref.apa}" for ref in lit.bibliography(list(t.refs))))
