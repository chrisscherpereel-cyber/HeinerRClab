import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.experiment import PLAN, block_market, block_order
from ui.common import CAT, hypothesis_card, style
from ui.common import evidence_note

st.title("Play the market")
evidence_note("causal")
st.caption("Run one firm in the same market as the agents, over three blocks of 25 periods. The data feed the "
           "Experiment analysis page, which classifies each person by the decision rule that predicts them best "
           "and tests whether people change their output less often when the market is harder to read.")

S = st.session_state.setdefault("play", {})

if not S.get("started"):
    hypothesis_card(
        "EXPER",
        "Each participant plays three blocks, one for each level of uncertainty (low, medium and high cost "
        "volatility, perception error and unannounced demand shifts), in an order counterbalanced across "
        "participants. Everyone faces the same three agent rivals and the same random draws in a given block, so "
        "choices are comparable across people. Alongside every choice, each agent design records what it would have "
        "chosen in the participant's place.")
    st.header("Before you start", divider="gray")
    st.markdown(
        "* You run **one of four firms** that sell the same product. Each period you choose how many units to "
        "produce. Keeping last period's output is always an option.\n"
        "* The market price falls as total output rises. Your profit is (price − unit cost) × your output.\n"
        "* Your unit cost changes from period to period. You see an **estimate** of the coming cost, which may be "
        "off. In some blocks, demand can also shift without warning.\n"
        "* There are three blocks of 25 periods. Try to earn as much as you can in each block.")
    st.info("For research use: obtain ethics approval and informed consent before collecting data from participants. "
            "Nothing is stored on a server; the data stay in this browser session until you download them.",
            icon="🔒")
    pid = st.text_input("Participant ID", key="play_pid",
                        help="An anonymous code (no names or e-mail addresses). It also sets the order of the "
                             "three blocks.")
    consent = st.checkbox("I have read the information above and agree to take part.", key="play_consent",
                          help="Required before the first block starts.")
    if st.button("Start", type="primary", disabled=not (pid.strip() and consent), key="play_start"):
        order = block_order(pid.strip())
        S.update(started=True, pid=pid.strip(), order=order, block=0, market=block_market(order[0]), frames=[])
        st.rerun()
    st.stop()

order, b = S["order"], S["block"]
if b >= len(order):
    st.success("All three blocks are complete. Thank you!", icon="🎉")
    data = pd.concat(S["frames"], ignore_index=True).assign(participant=S["pid"], plan=PLAN.digest)
    data["block_index"] = data["block"].map({c: i for i, c in enumerate(order)})
    s = data.groupby("block_index").agg(periods=("period", "count"), changes=("deviated", "mean"),
                                        profit=("profit", "mean"), rivals=("rivals_mean_profit", "mean"))
    s.index = [f"Block {i + 1}" for i in s.index]
    s.columns = ["Periods", "Share of periods you changed output", "Your mean profit", "Rivals' mean profit"]
    st.dataframe(s, width="stretch")
    st.download_button("Download your data (CSV)", data.to_csv(index=False).encode(),
                       file_name=f"market_{S['pid']}.csv", mime="text/csv", type="primary",
                       help="Upload this file on the Experiment analysis page, together with other participants' "
                            "files.")
    if st.button("Start a new participant", key="play_reset", help="Clears this session's data."):
        st.session_state["play"] = {}
        st.rerun()
    st.stop()

m = S["market"]
info = m.info()
mk = m.mk
st.progress((b * PLAN.periods_per_block + info["period"] - 1) / (len(order) * PLAN.periods_per_block),
            f"Block {b + 1} of {len(order)} · period {info['period']} of {PLAN.periods_per_block}")

c = st.columns(4)
c[0].metric("Last price", f"{info['last_price']:.1f}", help="The market price last period.")
c[1].metric("Your output last period", f"{info['own_q']:.0f}",
            help="Keeping this output is always an option.")
c[2].metric("Your profit last period", f"{info['own_profit']:,.0f}", help="(Price − unit cost) × your output.")
c[3].metric("Estimated unit cost this period", f"{info['cost_estimate']:.1f}",
            help="An estimate of the cost you will pay per unit this period. It may be off.")
st.caption(f"Total market output last period: {info['last_market_q']:.0f} units (you and three rivals). At the "
           f"start of the block, the price fell by about {mk.SL[0, 0]:.3f} for each extra unit sold, from "
           f"{mk.PM[0, 0]:.0f} at zero output. Demand may shift without warning.")

with st.form(f"play_form_{b}_{info['period']}"):
    q = st.number_input("Your output this period", min_value=5, max_value=2000, value=int(round(info["own_q"])),
                        step=5, key=f"play_q_{b}_{info['period']}",
                        help="Units to produce. Leave it unchanged to keep last period's output.")
    c1, c2 = st.columns(2)
    keep = c1.form_submit_button(f"Keep {info['own_q']:.0f}", width="stretch",
                                 help="Produce the same as last period.")
    go_ = c2.form_submit_button("Submit output", type="primary", width="stretch")
if keep or go_:
    m.step(float(info["own_q"]) if keep else float(q))
    if m.done:
        S["frames"].append(m.frame())
        S["block"] = b + 1
        if b + 1 < len(order):
            S["market"] = block_market(order[b + 1])
    st.rerun()

log = m.frame()
if len(log):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=log["period"], y=log["q"], name="Your output", mode="lines+markers",
                             line=dict(color=CAT[0])))
    fig.add_trace(go.Scatter(x=log["period"], y=log["price"], name="Price", mode="lines+markers", yaxis="y2",
                             line=dict(color=CAT[1])))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Price", showgrid=False))
    fig.update_xaxes(title="Period")
    fig.update_yaxes(title="Output")
    st.plotly_chart(style(fig, height=320), width="stretch")
    st.caption(f"This block so far: mean profit {log['profit'].mean():,.0f} per period (rivals "
               f"{log['rivals_mean_profit'].mean():,.0f}); you changed output in "
               f"{log['deviated'].mean():.0%} of periods.")
