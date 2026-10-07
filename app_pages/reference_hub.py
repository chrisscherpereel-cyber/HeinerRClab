import streamlit as st

st.title("Reference")
st.markdown("Supporting detail for the workflow: the theories, the equations and methods, how each agent is "
            "specified, and the bibliography.")

THEORIES = [("app_pages/theory_overview.py", "Overview of the nine theories"),
            ("app_pages/theory_heiner.py", "Heiner: reliability condition"),
            ("app_pages/theory_optimiser.py", "Neoclassical optimization"),
            ("app_pages/theory_options.py", "Real options"),
            ("app_pages/theory_cobweb.py", "Cobweb & adaptive expectations"),
            ("app_pages/theory_heuristics.py", "Simple heuristics"),
            ("app_pages/theory_satisficing.py", "Satisficing"),
            ("app_pages/theory_rl.py", "Reinforcement learning"),
            ("app_pages/theory_imitation.py", "Imitation & selection"),
            ("app_pages/theory_ecology.py", "Organizational ecology")]
c1, c2 = st.columns(2)
with c1.container(border=True):
    st.subheader("Theory")
    for path, lab in THEORIES:
        st.page_link(path, label=lab)
with c2.container(border=True):
    st.subheader("Equations and methods")
    st.page_link("app_pages/model_docs.py", label="Model & methods (equations, presets, measurement)")
    st.subheader("Agent specifications")
    st.page_link("app_pages/agents_reference.py", label="Agents as implemented")
    st.page_link("app_pages/information.py", label="Information & feedback (what each agent may observe)")
    st.subheader("Bibliography")
    st.page_link("app_pages/literature.py", label="Research & contribution (references, BibTeX)")
