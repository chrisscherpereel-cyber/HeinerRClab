import streamlit as st

st.title("Validation")
st.markdown("What can the findings support? Simulation results hold within the specified model. These checks test the "
            "implementation against exact solutions, compare simulated behavior with data, and collect human "
            "decisions under the same conditions.")

SECTIONS = {
    "Analytical benchmarks": (
        "Cases with known answers: the implementation must reproduce them before its comparisons are trusted.",
        [("app_pages/tracking.py", "Heiner versus optimal filtering (exact Muth–Kalman solution)"),
         ("app_pages/benchmarks.py", "Decision benchmarks: Bayes, robust optimization, bandits (enumeration checks)"),
         ("app_pages/nk.py", "NK landscapes: exact optima by enumeration for N ≤ 16")]),
    "Empirical data": (
        "Patterns and data from the field and from laboratory experiments; limits of each comparison are stated on "
        "its page.",
        [("app_pages/field_patterns.py", "Field patterns (criteria fixed in advance)"),
         ("app_pages/calibration.py", "Calibration to laboratory experiments"),
         ("app_pages/empirical.py", "Empirical validation on public data (registered protocol)")]),
    "Human experiments": (
        "People decide in the same market; the agents' shadow decisions are recorded for comparison. A question such "
        "as 'does a decision aid improve human performance?' is answered here, not by simulation.",
        [("app_pages/play_market.py", "Play the market (collect decisions)"),
         ("app_pages/experiment_analysis.py", "Experiment analysis (classify and test)")]),
    "Signature tests and predictive checks": (
        "Each theory's distinctive prediction, and whether the reliability condition predicts performance "
        "out of sample.",
        [("app_pages/special_tests.py", "Signature tests by theory"),
         ("app_pages/rc_validation.py", "Does the reliability condition predict performance?")]),
}
for name, (note, pages) in SECTIONS.items():
    with st.container(border=True):
        st.subheader(name)
        st.caption(note)
        for path, lab in pages:
            st.page_link(path, label=lab)
