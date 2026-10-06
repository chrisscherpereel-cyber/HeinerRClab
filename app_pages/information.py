import json

import pandas as pd
import streamlit as st

from heiner_abm.information import (AGENTS, DEMAND_LABELS, ENGINES, FEEDBACK_LABELS, InfoSpec, access_table,
                                    compatibility, export_record, market_agents)
from ui.common import base_scenario

st.title("Information & feedback")
st.caption("What every agent may observe when it decides, and what feedback it learns from. One specification "
           "(heiner_abm.information.InfoSpec) is shared by every engine; each engine declares which specifications it "
           "can run, and each agent declares what it needs. An agent that needs information the specification "
           "withholds is reported as unsupported and is not run: it never receives extra information silently.")

with st.expander("The four feedback treatments and six information controls", expanded=False):
    st.markdown(
        "* **Chosen-action feedback:** only the payoff of the agent's own action.\n"
        "* **Full feedback:** payoffs of alternative actions, only where the design shows them (the generalization "
        "tasks show both outcomes; the cobweb market does not).\n"
        "* **Estimated counterfactual:** the agent estimates the alternative's payoff from its own believed model and "
        "its observations.\n"
        "* **Oracle counterfactual:** true counterfactual values computed by the researcher, a clearly labeled "
        "diagnostic benchmark.\n\n"
        "Controls: observation noise, observation delay, demand knowledge (none / believed / true), announced regime "
        "shifts, visibility of rivals' individual actions and payoffs, and observability of censored outcomes such "
        "as unmet demand. Every agent always knows its own action and observes its own payoff.")

# ------------------------------------------------------------------------------------------------ current market spec
st.header("1 · The current market scenario", divider="gray")
scn = base_scenario()
spec = scn.info
st.markdown(f"**Specification** (sidebar *Information & feedback*): {FEEDBACK_LABELS[spec.feedback]} · noise s.d. "
            f"{spec.obs_noise:g} · delay {spec.obs_delay} · {DEMAND_LABELS[spec.demand_knowledge].lower()} · shifts "
            f"{'announced' if spec.regime_announced else 'not announced'} · rivals "
            f"{'visible' if spec.rivals_visible else 'not visible'}")
errs = scn.validate()
if errs:
    st.error("This scenario cannot run under its information specification:\n\n" + "\n".join(f"* {e}" for e in errs),
             icon="⛔")
elif spec.is_oracle:
    st.warning("ORACLE information is switched on: results are a diagnostic benchmark.", icon="🔬")
else:
    st.success("Every agent in this scenario is supported by its information specification.", icon="✅")
st.dataframe(access_table(market_agents(scn), spec, "market"), hide_index=True, width="stretch")
st.download_button("Download this specification (JSON)", json.dumps(export_record("market", spec, market_agents(scn)),
                   indent=1).encode(), file_name="market_information_spec.json", mime="application/json")

# ------------------------------------------------------------------------------------------------ engines
st.header("2 · What each engine supports", divider="gray")
st.dataframe(pd.DataFrame([{"Engine": e.name, "Feedback it can deliver": ", ".join(FEEDBACK_LABELS[f] for f in e.feedback),
                            "Runs": "its native specification only" if e.fixed else "any supported specification",
                            **{k.replace("_", " "): v for k, v in e.controls.items() if k != "feedback"}}
                           for e in ENGINES.values()]), hide_index=True, width="stretch")
st.caption("The tournament, task and tracking engines are covered by frozen plans, so they run only the "
           "specification their registered results used; other specifications are reported as not implemented "
           "there rather than approximated.")

# ------------------------------------------------------------------------------------------------ every agent
st.header("3 · Every agent under its engine's own specification", divider="gray")
eng_key = st.selectbox("Engine", list(ENGINES), format_func=lambda k: ENGINES[k].name, key="info_engine")
st.dataframe(access_table([a for a in AGENTS if a.engine == eng_key], ENGINES[eng_key].native),
             hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------------------ what-if
st.header("4 · Check a specification", divider="gray")
st.caption("Pick any specification to see which agents could run under it, in each engine, and why the others "
           "cannot.")
c = st.columns(3)
w_fb = c[0].selectbox("Feedback", list(FEEDBACK_LABELS), format_func=FEEDBACK_LABELS.get, index=2, key="wi_fb")
w_dm = c[1].selectbox("Demand knowledge", list(DEMAND_LABELS), format_func=DEMAND_LABELS.get, index=1, key="wi_dm")
w_delay = c[2].number_input("Delay", 0, 20, 0, key="wi_delay")
c = st.columns(3)
w_noise = c[0].number_input("Noise s.d.", 0.0, 20.0, 0.0, key="wi_noise")
w_riv = c[1].checkbox("Rivals visible", True, key="wi_riv")
w_unmet = c[2].checkbox("Unmet demand observed", True, key="wi_unmet")
w = InfoSpec(feedback=w_fb, obs_noise=float(w_noise), obs_delay=int(w_delay), demand_knowledge=w_dm,
             rivals_visible=w_riv, unmet_demand_observed=w_unmet)
rows = []
for a in AGENTS:
    comp = compatibility(a, w)
    rows.append({"Agent": a.name, "Engine": ENGINES[a.engine].name.split(" (")[0],
                 "Status": {"supported": "✅ supported", "oracle variant": "🔬 ORACLE variant",
                            "unsupported": "⛔ unsupported"}[comp.status], "Why": " ".join(comp.reasons)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
