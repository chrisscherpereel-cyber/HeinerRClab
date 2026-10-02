import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.experiments import MEASURE_LABELS, EnvRanges
from heiner_abm.literature import HYPOTHESIS_BY_ID, THEORY_SOURCES, bibliography
from heiner_abm.theories import EXPERIMENTS, THEORIES, THEORY_NAMES, score, scoreboard
from heiner_abm.focal import tournament_key
from ui.common import (CAT, base_scenario, cached_horse_race, cached_tournament, download, focal_theory, focal_title,
                       fmt_p, measure_opts, measurement, reps, research_panel, show_errors, style, to_json, verdict)

FOCAL = focal_theory()
FKEY = tournament_key(FOCAL)                      # the focal theory's key in the directional tournament, if any
FNAME = THEORY_NAMES.get(FKEY) if FKEY else None
FORECASTS = {"heiner": ("rc", "K"), "optimiser": ("neoclassical",), "options": ("options",), "cobweb": ("cobweb",),
             "heuristic": ("accuracy",), "rl": ("past",)}.get(FOCAL, ())

st.title("Competing theories: a tournament of predictions")
st.caption("Seven theories make directional predictions about when behavioral flexibility pays under uncertainty. "
           "This page states what each predicts in this market, runs the experiments that tell them apart, and "
           "scores every theory against the results. A second test asks which theory best *forecasts*, out of "
           "sample, whether a firm's flexibility will beat its own rigid twin. Each theory has its own page under "
           "*Theories*.")

base = base_scenario()
if not show_errors(base):
    st.stop()
H, cont = measurement()
disc, split = measure_opts()

SYM = {"+": "↑ rises", "-": "↓ falls", "0": "no effect", ">=0": "≥ 0 (never harmful)", None: "—"}
OBS = {"+": "↑ rises", "-": "↓ falls", "0": "no significant effect"}
MARK = {"match": "✅", "contradicted": "❌", "inconclusive": "➖", None: ""}

# ------------------------------------------------------------------------------------------------ contenders
st.header("1 · The contenders", divider="gray")
st.dataframe(pd.DataFrame([dict(Theory=t.name, Sources=t.sources, **{"Core claim": t.claim}) for t in THEORIES]),
             hide_index=True, width="stretch")
with st.expander("📚 Research behind each theory"):
    for t in THEORIES:
        st.markdown(f"**{t.name}**")
        st.markdown("\n".join(f"* {r.apa}" for r in bibliography(THEORY_SOURCES[t.key])))

st.header("2 · What each theory predicts", divider="gray")
st.caption("Each experiment reports one statistic; a theory predicts its sign. For E1 the statistic is the payoff "
           "to flexibility itself (a negative value means rigid firms earn more); for the others it is how that "
           "payoff, or behavior, changes with the manipulated variable. '—' means the theory makes no clear "
           "prediction. These are stylized readings of each literature, open to debate.")
pred = pd.DataFrame([{"Experiment": e.title, "Statistic": e.statistic,
                      **{t.name: SYM[e.predictions.get(t.key)] for t in THEORIES}} for e in EXPERIMENTS])
st.dataframe(pred, hide_index=True, width="stretch")
with st.expander("Why each theory predicts what it does"):
    for e in EXPERIMENTS:
        st.markdown(f"**{e.title}** · {e.manipulation} Tests hypothesis **{e.hid}** "
                    f"({HYPOTHESIS_BY_ID[e.hid].where}).")
        st.markdown("\n".join(f"* *{THEORY_NAMES[k]}* ({SYM[e.predictions.get(k)]}): {txt}" for k, txt in e.why.items()))

# ------------------------------------------------------------------------------------------------ tournament
st.header("3 · Tournament of experiments", divider="gray")
st.caption(f"Every experiment starts from the sidebar's base scenario (**{base.firms[0].rule}**, {base.n_firms} firms, "
           f"{base.periods} periods) and changes only what it manipulates. An effect counts when p < 0.05. A theory "
           "scores ✅ when the observed sign matches its prediction, ❌ when it contradicts it, and ➖ when it "
           "predicted an effect that did not reach significance. Theories are ranked by net score (✅ − ❌); ties go "
           "to the theory with more ✅, which risked and passed more tests. Switch the sidebar to Cournot to rerun the "
           "tournament in a model-based market.")
c1, c2 = st.columns([1, 3])
t_reps = c1.number_input("Replications per condition", 4, 100, max(10, min(reps(), 20)), key="tour_reps",
                         help="Markets per condition in every experiment. More replications detect smaller effects "
                              "but take longer (about 20 s at 20).")
t_H = max(2, H)
tkey = (to_json(base), int(t_reps), t_H)
if c2.button("Run tournament", type="primary", key="tour_btn"):
    st.session_state["tour_key"] = tkey
outcomes = None
if st.session_state.get("tour_key") == tkey:
    with st.spinner(f"Running {len(EXPERIMENTS)} experiments…"):
        outcomes = cached_tournament(to_json(base), int(t_reps), t_H)

if outcomes:
    res = pd.DataFrame([dict(Experiment=e.title, Observed=OBS[outcomes[e.key].observed],
                             Statistic=outcomes[e.key].stat, p=outcomes[e.key].p, Result=outcomes[e.key].summary)
                        for e in EXPERIMENTS])
    st.dataframe(res, hide_index=True, width="stretch", column_config={
        "Statistic": st.column_config.NumberColumn(format="%.4g"),
        "p": st.column_config.NumberColumn("p", format="%.3g")})
    grid = pd.DataFrame([{"Experiment": e.title,
                          **{t.name: MARK[score(e.predictions.get(t.key), outcomes[e.key].observed)] for t in THEORIES}}
                         for e in EXPERIMENTS])
    st.markdown("**Each theory's record** (✅ match · ❌ contradicted · ➖ predicted effect not found · blank = no prediction)")
    st.dataframe(grid, hide_index=True, width="stretch")
    sb = scoreboard(outcomes)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("**Scoreboard**")
        st.dataframe(sb, hide_index=True, width="stretch", column_config={
            "theory": "Theory", "predictions": "Predictions", "matches": "✅", "contradicted": "❌",
            "inconclusive": "➖", "net": st.column_config.NumberColumn("Net (✅ − ❌)"),
            "hit_rate": st.column_config.NumberColumn("Hit rate", format="%.0%")})
    with c2:
        fig = go.Figure(go.Bar(y=sb["theory"][::-1], x=sb["net"][::-1], orientation="h",
                               marker_color=[CAT[0] if t == FNAME else "#9aa0a6" for t in sb["theory"][::-1]],
                               hovertemplate="%{y}<br>net score %{x}<extra></extra>"))
        fig.update_xaxes(title="Net score (matches − contradictions)")
        st.plotly_chart(style(fig, 300, "Net score"))
    for e in EXPERIMENTS:
        with st.expander(f"Details · {e.title}"):
            st.markdown(f"*{e.manipulation}* Statistic: {e.statistic.lower()}. {outcomes[e.key].summary}.")
            st.dataframe(outcomes[e.key].detail, hide_index=True, width="stretch")
            h = HYPOTHESIS_BY_ID[e.hid]
            st.markdown(f"**Hypothesis {h.hid}: {h.title}**")
            research_panel(h, nested=True)
    top = sb.iloc[0]
    if FNAME is None or FNAME not in set(sb["theory"]):
        verdict("neutral", f"**{focal_title()}** is not part of the directional tournament (it makes no directional "
                f"predictions about these experiments). **{top['theory']}** has the best record.")
    else:
        h = sb[sb["theory"] == FNAME].iloc[0]
        tied = sb[(sb["net"] == top["net"]) & (sb["matches"] == top["matches"])]["theory"].tolist()
        misses = [e.title for e in EXPERIMENTS
                  if score(e.predictions.get(FKEY), outcomes[e.key].observed) == "contradicted"]
        if h["theory"] == top["theory"] and len(tied) == 1:
            verdict("support", f"**{FNAME} has the best record** ({int(h['matches'])} of {int(h['predictions'])} "
                    f"predictions confirmed, {int(h['contradicted'])} contradicted).")
        elif h["theory"] in tied:
            verdict("neutral", f"{FNAME} **ties for the best record** with "
                    f"{', '.join(t for t in tied if t != FNAME)}.")
        else:
            verdict("reject", f"**{top['theory']}** has a better record than {FNAME} in this market.")
        if misses:
            st.caption(f"Predictions of {FNAME} contradicted here: " + "; ".join(misses) + ".")
    download(res, "theory_tournament.csv")

# ------------------------------------------------------------------------------------------------ horse race
st.header("4 · Forecasting horse race (out of sample)", divider="gray")
st.markdown(
    "Directional tests reward theories for getting the average effect right. A sharper test is whether a theory "
    "tells you **which firm** will benefit from flexibility. Random environments are drawn as on the *Does the RC "
    "predict performance?* page, and every firm is compared with its own rigid twin on the same shocks. Each theory "
    "supplies a forecast built only from information available in the **first** part of the run, and is scored by "
    "AUC on the **second** part (0.5 = coin flip, 1 = perfect).\n\n"
    "* **Heiner (1983):** the RC margin ln(r/w) − ln(tolerance limit), estimated in the first window.\n"
    "* **Heiner (1989):** a low error-to-signal ratio K.\n"
    "* **Bias–variance / ecological rationality:** accuracy alone, ln(r/w), without the stakes.\n"
    "* **Stakes only:** a low tolerance limit, without accuracy.\n"
    "* **Real options:** high volatility Δ.\n"
    "* **Cobweb stability:** a stable market (small spectral radius of the linearized dynamics).\n"
    "* **Reinforcement learning (atheoretical benchmark):** flexibility paid off in the first window.\n"
    "* **Neoclassical:** flexibility always pays, the same forecast for everyone (AUC = 0.5 by construction).")
if H < 2:
    st.warning("Set the counterfactual horizon H ≥ 2 in the sidebar. With H = 1 the reliability quantities use the "
               "one-shot, one-period measure, which is uninformative here.", icon="⚠️")
with st.form("race"):
    c = st.columns(4)
    n_env = c[0].number_input("Environments", 10, 400, 60, step=10,
                              help="Random market environments; each draws its parameters from the default ranges.")
    n_reps = c[1].number_input("Replications per environment", 1, 10, 2,
                               help="Independent runs (different cost shocks) of each environment.")
    rules = c[2].multiselect("Production rules", ["Bertrand", "Cournot"], default=["Bertrand", "Cournot"],
                             help="Production rules the environments may use; each firm draws one at random.")
    meas = c[3].radio("Reliability measure", ["full", "static"], format_func=MEASURE_LABELS.get,
                      help="Full dynamic RC (H periods, rivals react) or the one-shot, one-period RC.")
    env_seed = st.number_input("Sampling seed", 0, 10**6, 12345, help="Seed for drawing the environments.")
    st.caption(f"Markets simulated ≈ environments × replications × (firms + 1) = "
               f"{int(n_env) * int(n_reps) * (base.n_firms + 1):,} at H = {H}.")
    go_race = st.form_submit_button("Run horse race", type="primary")
ranges = EnvRanges(rules=tuple(rules) or ("Bertrand",))
rkey = (to_json(base), int(n_env), int(n_reps), json.dumps(asdict(ranges)), H, cont, int(env_seed), disc, split, meas)
if go_race:
    st.session_state["race_key"] = rkey
race = None
if st.session_state.get("race_key") == rkey:
    with st.spinner("Simulating environments and rigid twins…"):
        race = cached_horse_race(*rkey)

if race:
    hr, enc, gain, share, n_rows = race
    st.caption(f"{n_rows:,} firms; flexibility beat the rigid twin for {share:.0%} of them in the evaluation window.")
    fig = go.Figure(go.Bar(
        y=hr["theory"][::-1], x=hr["auc"][::-1], orientation="h",
        marker_color=[CAT[0] if k in ("rc", "K") else CAT[3] if k == "past" else "#9aa0a6" for k in hr["key"][::-1]],
        error_x=dict(type="data", symmetric=False, array=(hr["hi"] - hr["auc"])[::-1].fillna(0),
                     arrayminus=(hr["auc"] - hr["lo"])[::-1].fillna(0), thickness=1.2, width=4),
        hovertemplate="%{y}<br>AUC %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"))
    fig.update_xaxes(title="Out-of-sample AUC (95% CI, environments resampled)", range=[0.3, 1])
    st.plotly_chart(style(fig, 380, "Which theory forecasts who benefits from flexibility?"))
    st.dataframe(hr.drop(columns="key"), hide_index=True, width="stretch", column_config={
        "theory": "Theory (forecast)", "auc": st.column_config.NumberColumn("AUC", format="%.3f"),
        "lo": st.column_config.NumberColumn("95% lo", format="%.3f"),
        "hi": st.column_config.NumberColumn("95% hi", format="%.3f"), "n": "Firms"})
    st.markdown("**Encompassing test.** Does the reliability condition add information once every rival forecast is "
                "known? Logistic models are fitted on some environments and scored on others (5-fold, grouped by "
                "environment).")
    if not enc.empty:
        st.dataframe(enc, hide_index=True, width="stretch", column_config={
            "model": "Model", "features": "Forecasts combined",
            "cv_auc": st.column_config.NumberColumn("Cross-validated AUC", format="%.3f")})
        st.markdown(f"Gain from adding the RC's tolerance limit to all rivals: **{gain['gain']:+.3f}** AUC "
                    f"(95% CI {gain['lo']:+.3f} to {gain['hi']:+.3f}).")
    best = hr.iloc[0]
    rc = hr[hr["key"] == "rc"]
    mine = hr[hr["key"].isin(FORECASTS)]
    if mine.empty:
        verdict("neutral", f"**{focal_title()}** makes no firm-level forecast in this horse race. The most accurate "
                f"forecast is **{best['theory'].split(':')[0]}** (AUC {best['auc']:.3f}).")
    elif best["key"] in FORECASTS:
        verdict("support", f"The forecast of the theory under test, **{best['theory'].split(':')[0]}**, is the most "
                f"accurate (AUC {best['auc']:.3f}).")
    else:
        verdict("reject", f"**{best['theory'].split(':')[0]}** forecasts better than the theory under test "
                f"(best: {mine.iloc[0]['theory'].split(':')[0]}, AUC {mine.iloc[0]['auc']:.3f}).")
    past = hr[hr["key"] == "past"]
    if len(past) and len(rc) and past["auc"].iloc[0] > rc["auc"].iloc[0]:
        st.caption(f"The atheoretical benchmark (did flexibility pay in the first window?) reaches AUC "
                   f"{past['auc'].iloc[0]:.3f}, above the RC's {rc['auc'].iloc[0]:.3f}. Past performance is the "
                   "stronger forecaster; the RC's value lies in explaining *why* flexibility pays, not in beating "
                   "a track record.")
    if gain and gain["lo"] > 0:
        verdict("support", "The RC adds significant information beyond every rival forecast combined.")
    elif gain:
        verdict("neutral", "Once the rival forecasts are combined, the RC adds **no significant** further "
                "information.")
    download(hr, "theory_horse_race.csv")

# ------------------------------------------------------------------------------------------------ verdict
st.header(f"5 · Where does {focal_title()} stand among its rivals?", divider="gray")
if not (outcomes and race):
    st.info("Run the tournament (section 3) and the horse race (section 4) to fill in this verdict with this "
            "session's results.", icon="ℹ️")
else:
    sb = scoreboard(outcomes)
    hr = race[0].reset_index(drop=True)
    lines = []
    if FNAME in set(sb["theory"]):
        h = sb[sb["theory"] == FNAME].iloc[0]
        rank = int(sb.index[sb["theory"] == FNAME][0]) + 1
        lines.append(f"* **Directional tests:** {FNAME} ranks **#{rank} of {len(sb)}** by net score, with "
                     f"{int(h['matches'])} matches, {int(h['contradicted'])} contradictions and "
                     f"{int(h['inconclusive'])} inconclusive tests out of {int(h['predictions'])} predictions.")
    else:
        lines.append(f"* **Directional tests:** {focal_title()} makes no directional predictions here.")
    if hr["key"].isin(FORECASTS).any():
        rrank = int(hr.index[hr["key"].isin(FORECASTS)][0]) + 1
        lines.append(f"* **Forecasting:** its best forecast ranks **#{rrank} of {len(hr)}** forecasts.")
    else:
        lines.append("* **Forecasting:** it makes no firm-level forecast in the horse race.")
    lines.append(f"* **Added value of the reliability condition:** combining all rival forecasts and then adding the "
                 f"RC changes the cross-validated AUC by {race[2].get('gain', np.nan):+.3f}.")
    st.markdown("\n".join(lines))
st.markdown("**Reference results for the reliability condition.** The notes below summarize how Heiner's theory "
            "fared in the reference runs; the rows above follow the theory chosen in the sidebar.")
st.markdown(
    """
**How to read this.** Heiner's theory makes more, and more specific, predictions than its rivals: it says when
flexibility helps *and* when it hurts, through the CD-gap and the stakes. Neoclassical optimization and real
options cannot explain free flexibility being harmful, and cobweb stability theory cannot explain effects of
volatility or noise that leave the market stable. Bias–variance reasoning comes closest, and differs mainly in
ignoring the stakes. In the reference runs (Bertrand and Cournot markets, two seeds each, 20 replications, H = 20;
see the README), "superior" holds in one sense and not the other:

* **As an explanation**, the RC had the best record in all four tournaments (5–6 of 9 predictions confirmed, 1
  contradicted). Its miss was perception noise in Bertrand markets, which *raised* the payoff to flexibility.
* **As a forecasting tool**, the dynamic RC was the best theory-based forecast (AUC ≈ 0.61) but was far behind a
  firm's own track record (AUC ≈ 0.86), and added nothing once the rival forecasts were combined. The one-shot,
  one-period RC did no better than cobweb stability or accuracy alone.

**Caveats.** The simulation was built to test Heiner's theory: rule B, the CD-gap and the counterfactual
bookkeeping follow his framework. The rival predictions are stylized, and several rivals (satisficing,
organizational ecology) are not implemented as agents, so they are tested only on the few predictions they make.
Results depend on the base scenario (Bertrand versus Cournot in particular), so rerun the tournament under both.
A simulated market can show that a theory is internally coherent and discriminating. It cannot show that real
firms behave this way.
""")
