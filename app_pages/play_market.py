import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.experiment import (AID_QUESTION, COMPREHENSION, CONDITION_LABELS, PLAN, assignment_for, block_market,
                                   decision_row, elicit_now, slot_from_id)
from ui.common import CAT, evidence_note, set_engine, style

st.title("Human experiments: play the market")
evidence_note("causal")
set_engine("arena")
st.caption(f"Protocol {PLAN.version} (`{PLAN.digest}`): a practice block, then four blocks that each change one "
           "source of uncertainty (cost volatility, noise in the cost estimate, unannounced demand shifts) relative "
           "to a baseline. The order, the exogenous paths, access to a decision aid and belief questions are assigned "
           "by an allocation schedule (Validation → Human experiments: analysis).")

S = st.session_state.setdefault("play", {})


def _now():
    return time.time()


# ============================================================================================ consent and assignment
if not S.get("stage"):
    st.header("Before you start", divider="gray")
    st.markdown(
        "* You run **one of four firms** that sell the same product. Each period you choose how many units to "
        "produce. Keeping last period's output is always an option.\n"
        "* The market price falls as total output rises. Your profit is (price − unit cost) × your output.\n"
        "* Before deciding you see an **estimate** of this period's unit cost, which may be off. In some blocks, "
        "demand can shift without warning.\n"
        f"* First a short practice block ({PLAN.practice_periods} periods), then four blocks of "
        f"{PLAN.periods_per_block} periods. Try to earn as much as you can in each block.")
    st.info("For research use: obtain ethics approval and informed consent before collecting data. Nothing is stored "
            "on a server; the data stay in this browser session until downloaded.", icon="🔒")
    c1, c2, c3 = st.columns(3)
    pid = c1.text_input("Participant ID", key="play_pid", help="An anonymous code (no names or e-mail addresses).")
    slot_txt = c2.text_input("Allocation slot (from the schedule)", key="play_slot",
                             help="The next unused slot of the allocation schedule. Leave empty only for a "
                                  "demonstration: the slot is then derived from the ID and assignments are balanced "
                                  "only in expectation.")
    session = c3.text_input("Session (optional)", key="play_session",
                            help="Lab session code; analyses cluster by session when it is recorded.")
    consent = st.checkbox("I have read the information above and agree to take part.", key="play_consent")
    if st.button("Start", type="primary", disabled=not (pid.strip() and consent), key="play_start"):
        try:
            slot = int(slot_txt) if slot_txt.strip() else slot_from_id(pid.strip())
        except ValueError:
            st.error("The allocation slot must be a whole number.")
            st.stop()
        a = assignment_for(slot)
        S.update(stage="comprehension", pid=pid.strip(), session=session.strip(), assignment=a, attempts=0,
                 slot_from_schedule=bool(slot_txt.strip()), records=[], block=-1, passed=False)
        st.rerun()
    st.stop()

a = S["assignment"]


# ============================================================================================ comprehension
if S["stage"] == "comprehension":
    st.header("Check your understanding", divider="gray")
    if S.get("comp_error"):
        st.error(S["comp_error"])
    qs = COMPREHENSION + ((AID_QUESTION,) if a.aid else ())
    with st.form(f"comp_{S['attempts']}"):
        answers = {k: st.radio(text, opts, index=None, key=f"play_comp_{k}_{S['attempts']}")
                   for k, text, opts, _ in qs}
        ok = st.form_submit_button("Check answers", type="primary")
    if ok:
        S["attempts"] += 1
        wrong = [text for k, text, opts, right in qs if answers[k] != opts[right]]
        if not wrong:
            S.update(stage="practice", passed=True)
            S["market"] = block_market("B", -1, practice=True)
            st.rerun()
        elif S["attempts"] >= PLAN.max_attempts:
            S.update(stage="done", passed=False)
            st.rerun()
        else:
            S["comp_error"] = (f"Not quite: please reread the instructions and try again ({len(wrong)} answer(s) to "
                               f"correct; attempt {S['attempts']} of {PLAN.max_attempts}).")
            st.rerun()
    st.stop()


# ============================================================================================ done
if S["stage"] == "done" and S.get("belief_for") is None:
    recs = pd.DataFrame(S["records"])
    if not S["passed"]:
        st.warning("Thank you. The comprehension questions were not all answered correctly, so the session ends here "
                   "(recorded as an exclusion, as fixed in the protocol).", icon="ℹ️")
        recs = pd.DataFrame([dict(participant=S["pid"], session=S["session"], slot=a.slot, plan=PLAN.digest,
                                  comprehension_passed=False, comprehension_attempts=S["attempts"], practice=True,
                                  synthetic=False)])
    else:
        st.success("All blocks are complete. Thank you!", icon="🎉")
        main = recs[~recs["practice"]]
        s = main.groupby("block_index").agg(periods=("period", "count"), changes=("deviated", "mean"),
                                            profit=("profit", "mean"), rivals=("rivals_mean_profit", "mean"))
        s.index = [f"Block {i + 1}" for i in s.index]
        s.columns = ["Periods", "Share of periods you changed output", "Your mean profit", "Rivals' mean profit"]
        st.dataframe(s, width="stretch")
    meta = dict(participant=S["pid"], session=S["session"], plan=PLAN.digest, **a.to_dict(),
                slot_from_schedule=S["slot_from_schedule"], comprehension_attempts=S["attempts"],
                comprehension_passed=S["passed"])
    c1, c2 = st.columns(2)
    c1.download_button("Download the data (CSV)", recs.to_csv(index=False).encode(),
                       file_name=f"market_{S['pid']}.csv", mime="text/csv", type="primary", width="stretch")
    c2.download_button("Download the session record (JSON)", json.dumps(meta, indent=1).encode(),
                       file_name=f"market_{S['pid']}_session.json", mime="application/json", width="stretch")
    if st.button("Start a new participant", key="play_reset", help="Clears this session's data."):
        st.session_state["play"] = {}
        st.rerun()
    st.stop()


# ============================================================================================ belief screen
def _record(m, row, extra):
    practice = S["stage"] == "practice"
    cond = "practice" if practice else a.order()[S["block"]]
    path = -1 if practice else a.path(cond)
    rec = decision_row(m, row, a, -1 if practice else S["block"], cond, path,
                       dict(participant=S["pid"], session=S["session"], comprehension_passed=True,
                            comprehension_attempts=S["attempts"], synthetic=False, elicited=False,
                            belief_price=np.nan, belief_confidence=np.nan, rt_belief=np.nan, **extra))
    S["records"].append(rec)
    return rec


if S.get("belief_for") is not None:
    st.header("Before you see the result", divider="gray")
    key = f"belief_{len(S['records'])}"
    if S.get("belief_key") != key:
        S["belief_key"], S["belief_shown"] = key, _now()
    st.caption("A separate question about the period you just decided. It does not change your profit.")
    with st.form(key):
        bp = st.number_input("What do you expect the price to be this period?", 0.0, 500.0, step=0.5, value=None,
                             key=f"play_bp_{key}")
        bc = st.slider("How confident are you that your chosen output will earn more than keeping last period's "
                       "output? (0 = not at all, 100 = certain)", 0, 100, 50, key=f"play_bc_{key}")
        sent = st.form_submit_button("Continue", type="primary")
    if sent and bp is None:
        st.error("Please enter a price forecast.")
    elif sent:
        r = S["records"][S["belief_for"]]
        r.update(belief_price=float(bp), belief_confidence=float(bc), rt_belief=_now() - S["belief_shown"],
                 elicited=True)
        S["belief_for"] = None
        st.rerun()
    st.stop()


# ============================================================================================ decision screen
if S["stage"] == "practice":
    title = "Practice block (not scored)"
    total = PLAN.practice_periods
else:
    title = f"Block {S['block'] + 1} of 4"
    total = PLAN.periods_per_block
m = S["market"]
info = m.info()
screen = f"{S['stage']}_{S['block']}_{info['period']}"
if S.get("shown_key") != screen:
    S.update(shown_key=screen, shown_at=_now(), opened=set())
st.progress((info["period"] - 1) / total, f"{title} · period {info['period']} of {total}")

c = st.columns(4)
c[0].metric("Last price", f"{info['last_price']:.1f}")
c[1].metric("Your output last period", f"{info['own_q']:.0f}", help="Keeping this output is always an option.")
c[2].metric("Your profit last period", f"{info['own_profit']:,.0f}")
c[3].metric("Estimated unit cost this period", f"{info['cost_estimate']:.1f}", help="An estimate; it may be off.")


def _open(name):
    S["opened"].add(name)


c1, c2 = st.columns(2)
show_hist = c1.toggle("Show the history of price and your output", key=f"play_hist_{screen}",
                      on_change=_open, args=("history",))
show_cost = c2.toggle("Show the history of unit costs", key=f"play_cost_{screen}", on_change=_open, args=("cost",))
log = m.frame()
if show_hist and len(log):
    f = go.Figure([go.Scatter(x=log["period"], y=log["q"], name="Your output", line=dict(color=CAT[0])),
                   go.Scatter(x=log["period"], y=log["price"], name="Price", yaxis="y2", line=dict(color=CAT[1]))])
    f.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
    st.plotly_chart(style(f, 260), width="stretch")
if show_cost and len(log):
    f = go.Figure([go.Scatter(x=log["period"], y=log["cost"], name="Unit cost", line=dict(color=CAT[2])),
                   go.Scatter(x=log["period"], y=log["cost_estimate"], name="Your estimate", line=dict(color=CAT[3]))])
    st.plotly_chart(style(f, 240), width="stretch")

if a.aid:
    rec = m.pending["shadows"][PLAN.aid_design]
    rec_q = float(max(5.0, np.rint(rec)))
    verdict = "change" if rec_q != info["own_q"] else "keep"
    st.info(f"**Decision aid:** produce **{rec_q:.0f}** units ({'change' if verdict == 'change' else 'keep'} your "
            "output). The aid only changes output when its record suggests that changing has paid off; it can be "
            "wrong. You may follow or ignore it.", icon="🧭")

with st.form(f"play_form_{screen}"):
    q = st.number_input("Your output this period", min_value=5, max_value=2000, value=int(round(info["own_q"])),
                        step=5, key=f"play_q_{screen}")
    k1, k2 = st.columns(2)
    keep = k1.form_submit_button(f"Keep {info['own_q']:.0f}", width="stretch")
    go_ = k2.form_submit_button("Submit output", type="primary", width="stretch")
if keep or go_:
    rt = _now() - S["shown_at"]
    row = m.step(float(info["own_q"]) if keep else float(q))
    rec = _record(m, row, dict(rt_decision=rt, info_history="history" in S["opened"] or bool(show_hist),
                               info_cost="cost" in S["opened"] or bool(show_cost)))
    if a.elicit and S["stage"] == "main" and elicit_now(row["period"]):
        S["belief_for"] = len(S["records"]) - 1
    if m.done:
        if S["stage"] == "practice":
            S.update(stage="main", block=0)
        else:
            S["block"] += 1
        if S["block"] >= 4:
            S["stage"] = "done"
        else:
            nxt = a.order()[S["block"]]
            S["market"] = block_market(nxt, a.path(nxt))
    st.rerun()

if len(log):
    st.caption(f"This block so far: your mean profit {log['profit'].mean():,.0f} per period (rivals "
               f"{log['rivals_mean_profit'].mean():,.0f}).")
