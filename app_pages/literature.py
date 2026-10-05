import pandas as pd
import streamlit as st

from heiner_abm.literature import (CONTRIBUTIONS, HYPOTHESES, HYPOTHESIS_BY_ID, METHOD_REFS, REFERENCES,
                                   THEORY_SOURCES, bibliography, cite, cited_keys)
from heiner_abm.theories import THEORIES
from heiner_abm.focal import FOCAL_KEYS, prediction, title as theory_title
from ui.common import focal_theory, focal_title, research_panel

st.title("Research basis and contribution")
st.caption("Every hypothesis in this laboratory is grounded in published research, and so is its alternative. "
           "This page collects the contribution the simulation makes, the evidence behind each hypothesis and each "
           "rival theory, and the complete bibliography, ready to export for a literature review.")

# ------------------------------------------------------------------------------------------------ contribution
st.header("1 · Contribution to the literature", divider="gray")
st.markdown(
    "The laboratory tests nine theories of decision making under uncertainty on equal terms; the sidebar chooses "
    "which one is highlighted. The question that unites them is the one Heiner's reliability condition answers "
    "most directly. Heiner's reliability condition explains rule-governed behavior as a response to uncertainty: when the gap "
    "between the difficulty of a problem and an agent's competence widens, restricting behavior to simple rules "
    "can beat flexible, 'optimizing' behavior. The theory is widely cited but hard to test, because its quantities "
    "(how often deviations are right or wrong, and what they gain or lose) are rarely observable. The simulation "
    "makes them observable and puts the theory in competition with its rivals.")
for k, (title, text, keys) in enumerate(CONTRIBUTIONS, 1):
    with st.container(border=True):
        st.markdown(f"**C{k} · {title}**")
        st.markdown(text)
        st.caption("Builds on: " + cite(*keys))

# ------------------------------------------------------------------------------------------------ hypotheses
st.header("2 · Hypotheses and the research behind them", divider="gray")
focal = focal_theory()
st.caption(f"Each row is a hypothesis tested in the app. The first prediction column is the theory under test "
           f"(⭐ {focal_title()}, chosen in the sidebar). 'Supporting research' backs the reliability-condition "
           "prediction; 'Alternative research' backs the registered alternative.")
matrix = pd.DataFrame([{
    "ID": h.hid, "Hypothesis": h.title, "Tested on": h.where,
    f"⭐ {focal_title()} predicts": prediction(h.hid, focal)[0],
    "Reliability condition predicts": h.rc_prediction, "Alternative": f"{h.alt_label}: {h.alt_prediction}",
    "Supporting research": cite(*[k for k, _ in h.support]),
    "Alternative research": cite(*[k for k, _ in h.alternative]),
    "Contribution": h.contribution} for h in HYPOTHESES])
st.dataframe(matrix, hide_index=True, width="stretch", height=420)
hid = st.selectbox("Show the full research basis for", [h.hid for h in HYPOTHESES],
                   format_func=lambda k: f"{k} · {HYPOTHESIS_BY_ID[k].title}",
                   help="Pick a hypothesis to see how each study bears on it and the full references.")
h = HYPOTHESIS_BY_ID[hid]
st.dataframe(pd.DataFrame([dict(Theory=("⭐ " if k == focal else "") + theory_title(k),
                                Prediction=prediction(hid, k)[0], Basis=prediction(hid, k)[1]) for k in FOCAL_KEYS]),
             hide_index=True, width="stretch")
research_panel(h, nested=True)

# ------------------------------------------------------------------------------------------------ theories
st.header("3 · Rival theories", divider="gray")
for t in THEORIES:
    with st.expander(f"{t.name} · {t.sources}"):
        st.markdown(t.claim)
        st.markdown("\n".join(f"* {r.apa}" for r in bibliography(THEORY_SOURCES[t.key])))

st.header("4 · Methods", divider="gray")
st.markdown(
    "Agent-based simulation as a theory-building method; ROC analysis (AUC) for scoring forecasts; out-of-sample "
    "validation; cluster bootstrap confidence intervals; and forecast-comparison and encompassing tests.")
st.markdown("\n".join(f"* {r.apa}" for r in bibliography(METHOD_REFS)))

# ------------------------------------------------------------------------------------------------ bibliography
st.header("5 · Bibliography", divider="gray")
refs = bibliography(cited_keys())
st.caption(f"{len(refs)} works cited across the laboratory.")
st.markdown("\n".join(f"* {r.apa}" for r in refs))
c1, c2, c3 = st.columns(3)
c1.download_button("Download BibTeX (.bib)", "\n\n".join(r.bibtex for r in refs).encode(), file_name="references.bib",
                   mime="text/plain", width="stretch")
c2.download_button("Download APA list (.md)", "\n\n".join(r.apa for r in refs).encode(), file_name="references.md",
                   mime="text/markdown", width="stretch")
evidence = pd.DataFrame([dict(hypothesis=h.hid, title=h.title, side=side, reference=REFERENCES[k].cite, key=k,
                              how_it_bears=note)
                         for h in HYPOTHESES for side, items in (("supports RC", h.support),
                                                                  ("supports alternative", h.alternative))
                         for k, note in items])
c3.download_button("Download evidence matrix (.csv)", evidence.to_csv(index=False).encode(),
                   file_name="hypothesis_evidence.csv", mime="text/csv", width="stretch")
