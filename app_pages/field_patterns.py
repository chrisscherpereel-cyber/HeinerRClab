import pandas as pd
import streamlit as st

from heiner_abm.literature import REFERENCES
from heiner_abm.patterns import PATTERNS, run_patterns
from heiner_abm.registered import TOURNAMENT_PLAN, registered_tuned
from ui.common import download, hypothesis_card, replication_notice, verdict
from ui.common import evidence_note, set_engine

st.title("Field patterns: does the market look like real markets?")
evidence_note("simulation")
set_engine("arena")
st.caption("Pattern-oriented validation (Grimm et al. 2005). The market is stylized and not calibrated to any "
           "commodity, so it is judged qualitatively, by whether it reproduces several documented empirical "
           "patterns at once. Each criterion was fixed before the run.")

hypothesis_card(
    "PATTERN",
    "Each pattern is produced by the population of agents that the literature associates with it (for example, "
    "naive price expectations for cobweb cycles), using the parameters tuned in the registered agent tournament "
    f"(plan `{TOURNAMENT_PLAN}`). A model that reproduces several independent patterns at once is much harder to "
    "dismiss as an artifact of its assumptions than one fitted to a single pattern.")
replication_notice("patterns")

st.header("1 · Patterns and criteria", divider="gray")
st.dataframe(pd.DataFrame([dict(Pattern=p.name, Evidence=p.evidence, Criterion=p.criterion, Population=p.population,
                                Sources="; ".join(REFERENCES[k].cite for k in p.sources)) for p in PATTERNS]),
             hide_index=True, width="stretch")

c1, c2 = st.columns(2)
n_envs = c1.slider("Environments per pattern", 4, 24, 12, 4, key="pat_envs",
                   help="Independent markets (different random draws) per pattern. 12 is the registered setting.")
periods = c2.slider("Periods per market", 200, 1200, 600, 100, key="pat_periods",
                    help="Length of each run; the first 100 periods are discarded as burn-in. 600 is the registered "
                         "setting.")

store = st.session_state.setdefault("pattern_results", {})
key = (n_envs, periods)
if st.button("Run pattern checks", type="primary", key="pat_run"):
    bar = st.progress(0.0, "Starting")
    store[key] = run_patterns(registered_tuned(), n_envs=n_envs, periods=periods,
                              progress=lambda f, m: bar.progress(min(f, 1.0), m))
    bar.empty()
res = store.get(key)
if res is None:
    st.info("Run the checks to see results (about 10 seconds at the registered settings).", icon="ℹ️")
    st.stop()

st.header("2 · Results", divider="gray")
n_pass = int(res["passed"].sum())
st.metric("Patterns reproduced", f"{n_pass} of {len(res)}",
          help="A pattern counts as reproduced only if its criterion, fixed in advance, is met on average across "
               "environments.")
for _, r in res.iterrows():
    verdict("support" if r["passed"] else "reject",
            f"**{r['pattern']}** · {r['result']}  \n{r['detail']} · criterion: {r['criterion']}")
st.caption("A failed pattern is reported as it is: it marks where the model departs from the evidence, not something "
           "to be tuned away after the fact.")
download(res.drop(columns=["sources"]), "field_patterns.csv", "Download results (CSV)")
