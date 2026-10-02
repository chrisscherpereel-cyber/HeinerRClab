import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm import registered
from heiner_abm.literature import REFERENCES
from heiner_abm.special import HEINER_TESTS, SCALES, SPECIAL_BY_KEY, SPECIAL_TESTS, run_special
from heiner_abm.theory_content import THEORIES
from ui.common import CAT, download, style, verdict

st.title("Signature tests by theory")
st.caption("The hypothesis tests and tournaments compare the theories on common questions. Here each theory is tested "
           "on its own terms: does its signature prediction, the claim that characterizes it, hold when its own "
           "agents act in the simulated market? Each test's criterion is fixed in advance.")

st.dataframe(pd.DataFrame(
    [dict(Theory="Heiner: reliability condition", Test="Five dedicated pages (see the first tab)",
          **{"Reference result": "see each page"})] +
    [dict(Theory=next(x.title for x in THEORIES if x.key == t.key), Test=t.title,
          **{"Reference result": registered.SPECIAL_RESULTS.get(t.key, ("not run", ""))[0]})
     for t in SPECIAL_TESTS]), hide_index=True, width="stretch")
st.caption("Reference results from the Full scale. Rerun any test below; at the Full scale it reproduces them.")

tabs = st.tabs(["Heiner"] + [next(x.title for x in THEORIES if x.key == t.key).split(":")[0] for t in SPECIAL_TESTS])
with tabs[0]:
    st.markdown("Heiner's reliability condition makes several predictions that no rival makes, so it has its own "
                "special tests, each on a dedicated page:")
    for page, title, text in HEINER_TESTS:
        with st.container(border=True):
            st.page_link(page, label=title, icon="🔎")
            st.caption(text)

for tab, t in zip(tabs[1:], SPECIAL_TESTS):
    with tab:
        st.markdown(f"#### {t.title}")
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.markdown(f"**Signature prediction.** {t.claim}")
            st.caption("Sources: " + "; ".join(REFERENCES[s].cite for s in t.sources if s in REFERENCES))
        with c2.container(border=True):
            st.markdown(f"**Setup.** {t.setup}")
            st.markdown(f"**Criterion (fixed in advance).** {t.criterion}")
        ref = registered.SPECIAL_RESULTS.get(t.key)
        if ref:
            st.caption(f"Reference run (Full scale): **{ref[0]}**. {ref[1]}")
        sc = st.radio("Scale", list(SCALES), horizontal=True, key=f"special_scale_{t.key}",
                      help="Quick: a few markets, for exploring. Full: the reference scale.")
        store = st.session_state.setdefault("special_results", {})
        if st.button("Run test", type="primary", key=f"special_run_{t.key}"):
            with st.spinner("Simulating"):
                store[(t.key, sc)] = run_special(t.key, sc)
        res = store.get((t.key, sc))
        if res is None:
            continue
        verdict("support" if res["ok"] else "reject", f"**{res['verdict'].capitalize()}.** {res['result']}")
        d = res["table"]
        fig = go.Figure()
        if res.get("categorical"):
            fig.add_trace(go.Bar(x=d[res["x"]].astype(str), y=d[res["y"]], marker_color=CAT[0],
                                 error_y=dict(type="data", array=d.get("s.d. across markets"), thickness=1, width=3)))
        else:
            fig.add_trace(go.Scatter(x=d[res["x"]], y=d[res["y"]], mode="lines+markers", name=res["y"],
                                     line=dict(color=CAT[0])))
            if res.get("y2"):
                fig.add_trace(go.Scatter(x=d[res["x"]], y=d[res["y2"]], mode="lines+markers", name=res["y2"],
                                         line=dict(color=CAT[1], dash="dash")))
        fig.update_xaxes(title=res["x_title"])
        fig.update_yaxes(title=res["y"].replace("_", " "))
        st.plotly_chart(style(fig, height=340), width="stretch")
        st.dataframe(d, hide_index=True, width="stretch")
        download(d, f"special_{t.key}.csv", "Download (CSV)")
