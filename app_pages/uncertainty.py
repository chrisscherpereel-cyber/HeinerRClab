import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.experiments import RISK, STRUCT, slope_summary
from heiner_abm.params import FirmSpec, linear_flex_firms
from ui.common import (CAT, base_scenario, cached_event, cached_uncertainty, download, hypothesis_card,
                       measure_opts, measurement, reps, show_errors, slope_chart, style, to_json, verdict)

st.title("Risk versus genuine (Knightian) uncertainty")
hypothesis_card(
    "KNIGHT",
    "Knight separated **risk** (known probabilities) from **uncertainty** (the structure itself is unknown). This "
    "page compares two families of environments with rising unpredictability:\n\n"
    "* **Risk:** raw-material cost volatility Δ rises. Its distribution is stationary and bounded; firms' models "
    "are correct, only the draws are unknown.\n"
    "* **Structural uncertainty:** unannounced **demand-regime shifts** (intercept and slope jump at random). "
    "*Model-based* firms (Cournot best replies) keep using an outdated demand curve until they update it, "
    "L periods later. *Model-free* firms (Bertrand margin feedback) react only to observed prices.\n\n"
    "The two families are compared at **matched unpredictability** (the RMS period-to-period price change, or the "
    "Cournot target error ξ).",
    notes={"heiner": "The reliability condition applies whatever the source of the CD-gap, so both families should "
                     "lower the payoff to flexibility once firms' decisions become unreliable.",
           "options": "Arrow and Lucas argued that economic reasoning fails under structural uncertainty, while option "
                      "theory treats more uncertainty of either kind as raising the value of flexibility."})

base = base_scenario()
if not show_errors(base):
    st.stop()
H, cont = measurement()
disc, _ = measure_opts()

st.subheader("① Payoff to flexibility under risk vs structural uncertainty")
with st.form("unc"):
    c = st.columns(4)
    kinds = c[0].multiselect("Firm types", ["Cournot (model-based)", "Bertrand (model-free)"],
                             default=["Cournot (model-based)", "Bertrand (model-free)"],
                             help="Industries to compare. Cournot firms use a demand model that can become outdated; "
                                  "Bertrand firms react only to observed prices.")
    deltas = [float(x) for x in c[1].text_input("Risk family: Δ levels", "2, 5, 10, 15, 20, 30",
                                                help="Comma-separated cost volatilities for the risk family (no regime shifts).").split(",") if x.strip()]
    hazards = [float(x) for x in c[2].text_input("Structural family: hazard λ levels", "0, 0.01, 0.02, 0.04, 0.07, 0.1",
                                                 help="Comma-separated per-period probabilities of an unannounced "
                                                      "demand-regime shift for the structural family.")
               .split(",") if x.strip()]
    n_reps = c[3].number_input("Replications", 2, 50, max(4, min(reps(), 10)),
                               help="Independent markets per level and firm type.")
    c = st.columns(4)
    base_delta = c[0].number_input("Structural family: base Δ", 0.0, 40.0, 5.0, 0.5,
                                    help="Cost volatility held fixed while the hazard λ varies.")
    isd = c[1].number_input("Shift size: intercept s.d.", 0.0, 40.0, float(base.structural.intercept_sd), 1.0,
                             help="S.d. of the jump in max price (demand intercept) at each regime shift.")
    ssd = c[2].number_input("Shift size: log-slope s.d.", 0.0, 1.0, float(base.structural.slope_sd), 0.05,
                             help="S.d. of the log change in demand slope at each regime shift.")
    lag = c[3].number_input("Model-updating lag L (−1 = never)", -1, 500, int(base.structural.belief_lag),
                             help="Periods Cournot firms keep using the old demand curve after a shift (−1 = never update).")
    c = st.columns(2)
    cphi = c[0].slider("Cournot φ range (φᵢ spread evenly)", 0.05, 1.0, (0.1, 0.4), 0.05,
                       help="Keep Cournot φ below the Heiner / Theocharis stability limit (see Dynamic RC page).")
    bphi = c[1].slider("Bertrand φ range", 0.05, 3.0, (0.25, 1.0), 0.05,
                       help="Lowest and highest Bertrand flexibility; firms' φᵢ are spread evenly across it.")
    st.caption(f"Measurement horizon H = {H}, γ = {disc:g} (sidebar). Firms use the sidebar selection rule "
               f"(**{base.firms[0].selection}**) and number of firms ({base.n_firms}).")
    run = st.form_submit_button("Run comparison", type="primary")

key = (tuple(kinds), tuple(deltas), tuple(hazards), int(n_reps), base_delta, isd, ssd, int(lag), cphi, bphi, H, cont,
       disc, to_json(base))
if run:
    st.session_state["unc_key"] = key
if st.session_state.get("unc_key") == key and kinds:
    xopt = st.radio("Match unpredictability by", ["price_change_rms", "xi_rms", "intensity"], horizontal=True,
                    help="Horizontal axis used to line up the two families: a common measure of how unpredictable "
                         "the market is, or each family's own intensity (Δ or λ).",
                    format_func=lambda k: {"price_change_rms": "RMS price change (agent-independent)",
                                           "xi_rms": "Cournot target error ξ (Heiner 1989)",
                                           "intensity": "Raw intensity (Δ or λ, separate scales)"}[k])
    all_summ = []
    for kind in kinds:
        rule = "Cournot" if kind.startswith("Cournot") else "Bertrand"
        lo_hi = cphi if rule == "Cournot" else bphi
        scn = base.copy()
        n = scn.n_firms
        flexes = np.linspace(lo_hi[0], lo_hi[1], n) if n > 1 else [lo_hi[1]]
        scn.firms = [FirmSpec(rule=rule, flex=float(x), selection=f.selection, threshold=f.threshold,
                              desired_margin=f.desired_margin, foresight=f.foresight, noise=f.noise)
                     for x, f in zip(flexes, scn.firms)]
        scn.market.delta = float(base_delta)
        scn.structural.enabled = True
        scn.structural.intercept_sd, scn.structural.slope_sd, scn.structural.belief_lag = float(isd), float(ssd), int(lag)
        with st.spinner(f"Simulating {kind}…"):
            firms, mk = cached_uncertainty(to_json(scn), tuple(deltas), tuple(hazards), int(n_reps), H, cont, disc)
        summ = slope_summary(firms, mk, ["family", "intensity"])
        summ["firm_type"] = kind
        all_summ.append(summ)
        st.markdown(f"#### {kind}")
        c1, c2 = st.columns(2)
        xt = {"price_change_rms": "RMS period-to-period price change", "xi_rms": "RMS Cournot target error ξ",
              "intensity": "Intensity (Δ for risk, λ for structural)"}[xopt]
        c1.plotly_chart(slope_chart(summ, xopt, xt, group="family", title="Payoff to flexibility (profit slope on φ)"))
        fr = go.Figure()
        for k, (fam, g) in enumerate(summ.groupby("family")):
            g = g.sort_values(xopt)
            fr.add_trace(go.Scatter(x=g[xopt], y=g["rc_share"], mode="lines+markers", name=f"{fam}: full dynamic RC",
                                    line=dict(color=CAT[k], width=2)))
            fr.add_trace(go.Scatter(x=g[xopt], y=g["rc_share_static"], mode="lines+markers", name=f"{fam}: one-period RC",
                                    line=dict(color=CAT[k], width=2, dash="dot"), marker=dict(symbol="circle-open")))
        fr.update_xaxes(title=xt); fr.update_yaxes(title="Share of firms satisfying the RC", range=[-0.02, 1.02])
        c2.plotly_chart(style(fr, 380, "Does the RC register the change?"))
        # matched comparison: interpolate the risk curve at the structural family's unpredictability
        rk = summ[summ["family"] == RISK].sort_values(xopt)
        sk = summ[(summ["family"] == STRUCT) & (summ["intensity"] > 0)].sort_values(xopt)
        if xopt != "intensity" and len(rk) > 1 and len(sk):
            inside = sk[(sk[xopt] >= rk[xopt].min()) & (sk[xopt] <= rk[xopt].max())]
            if len(inside):
                gap = (inside["slope"] - np.interp(inside[xopt], rk[xopt], rk["slope"])).mean()
                txt = (f"At matched unpredictability, flexibility pays **{abs(gap):,.0f}** per unit of φ "
                       f"{'less' if gap < 0 else 'more'} under structural uncertainty than under risk.")
                verdict("support" if (gap < 0) == (rule == "Cournot") else "neutral", txt)
            else:
                st.caption("The two families don't overlap in unpredictability; widen the Δ levels to match them.")
        struct_hi = sk.tail(1)
        if len(struct_hi):
            s_full, s_static = float(struct_hi["rc_share"].iloc[0]), float(struct_hi["rc_share_static"].iloc[0])
            slope_hi = float(struct_hi["slope"].iloc[0])
            if slope_hi < 0 and s_full < s_static - 0.2:
                verdict("support", f"At the highest hazard, flexibility hurts (slope {slope_hi:,.0f}) and the **dynamic RC "
                        f"detects it** ({s_full:.0%} satisfy it) while the one-period RC does not ({s_static:.0%}).")
            elif slope_hi > 0:
                verdict("neutral", f"For {kind.split(' ')[0]} firms, structural shifts leave flexibility valuable "
                        f"(slope {slope_hi:,.0f}). Model-free adjustment isn't misled by an outdated model.")
    if all_summ:
        out = pd.concat(all_summ)
        st.dataframe(out[["firm_type", "family", "intensity", "price_change_rms", "xi_rms", "K", "avg_firm_profit",
                          "slope", "lo", "hi", "rc_share", "rc_share_static", "log_rc_margin"]],
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                    ("price_change_rms", "xi_rms", "K", "rc_share", "rc_share_static", "log_rc_margin")})
        download(out, "risk_vs_uncertainty.csv")

# -------------------------------------------------------------------------------------------- event study
st.subheader("② Punctuated adjustment around regime shifts (Heiner 1989, §6)")
st.markdown(
    "Heiner argued that when the equilibrium itself shifts unpredictably, imperfect agents first keep adjusting "
    "slowly until they can locate the new equilibrium reliably, then **jump**, then return to slow adjustment. "
    "Below, every regime shift is aligned at k = 0 and each firm's behavior is averaged, relative to its own "
    "normal level (1.0 = normal).")
with st.form("evt"):
    c = st.columns(5)
    ev_rule = c[0].radio("Firms", ["Cournot", "Bertrand"], horizontal=True,
                         help="Production rule of the firms in the event study.")
    mixed = c[1].toggle("One firm per selection rule", True,
                        help="On: four firms (Always, Small, Large, Adaptive) side by side. Off: the sidebar's firms, "
                             "switched to the chosen rule (Cournot φ capped at 0.4).")
    ev_hazard = c[2].number_input("Hazard λ", 0.001, 0.1, 0.01, 0.001, format="%.3f",
                                   help="Per-period probability of a demand-regime shift. Low values keep shifts apart.")
    ev_lag = c[3].number_input("Model-updating lag L", 0, 200, int(max(base.structural.belief_lag, 0)),
                                help="Periods Cournot firms keep using the old demand curve after a shift.")
    ev_reps = c[4].number_input("Replications (runs of 3,000 periods)", 2, 40, 10,
                                 help="Independent runs; every shift in every run is one event in the average.")
    post = st.slider("Periods shown after the shift", 10, 150, max(40, int(ev_lag) + 20),
                     help="Length of the window after each shift (k = 0) in the event-study charts.")
    run_ev = st.form_submit_button("Run event study", type="primary")
ev = base.copy(periods=3000)
ev.market.delta = min(ev.market.delta, 5.0)
if mixed:
    phi = 0.3 if ev_rule == "Cournot" else 0.75
    ev.firms = [FirmSpec(ev_rule, phi, s, threshold=15 if ev_rule == "Cournot" else 5) for s in ("Always", "Small", "Large", "Adaptive")]
else:
    for f in ev.firms:
        f.rule = ev_rule
        f.flex = min(f.flex, 0.4) if ev_rule == "Cournot" else f.flex
ev.structural.enabled = True
ev.structural.hazard, ev.structural.belief_lag = float(ev_hazard), int(ev_lag)
ekey = (to_json(ev), int(ev_reps), H, cont, post)
if run_ev:
    st.session_state["evt_key"] = ekey
if st.session_state.get("evt_key") == ekey:
    with st.spinner("Collecting regime shifts…"):
        es = cached_event(to_json(ev), int(ev_reps), 1 if not any(f.selection == "Adaptive" for f in ev.firms) else H,
                          cont, 10, int(post))
    if es.empty:
        st.info("No complete shift windows; increase replications or the hazard.")
    else:
        c1, c2 = st.columns(2)
        f1, f2 = go.Figure(), go.Figure()
        for k, (sel, g) in enumerate(es.groupby("selection")):
            f1.add_trace(go.Scatter(x=g["k"], y=g["rel_absdq"].rolling(3, center=True, min_periods=1).mean(), name=sel,
                                    line=dict(color=CAT[k], width=2)))
            f2.add_trace(go.Scatter(x=g["k"], y=g["rel_xi"], name=sel, line=dict(color=CAT[k], width=2)))
        for f in (f1, f2):
            f.add_vline(x=0, line=dict(color="rgba(128,128,128,0.9)", width=1))
            f.add_hline(y=1, line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dot"))
            if ev_rule == "Cournot":
                f.add_vline(x=int(ev_lag) + 1, line=dict(color=CAT[1], width=1, dash="dash"))
            f.update_xaxes(title="Periods since the regime shift (k)")
            f.update_layout(hovermode="x unified")
        f1.update_yaxes(title="|Δq| relative to normal (3-period smoothed)")
        f2.update_yaxes(title="Decision error |ξ| relative to normal")
        c1.plotly_chart(style(f1, 380, "How much do firms adjust?"))
        c2.plotly_chart(style(f2, 380, "How wrong is their target? (dashed = model update)"))
        agg = es.groupby("k")[["rel_absdq", "rel_xi"]].mean()
        L = int(ev_lag)
        if ev_rule == "Cournot" and L >= 3:
            slow = agg.loc[1:L, "rel_absdq"].mean()
            quick = agg.loc[L + 1:L + 4, "rel_absdq"].max()
            err = agg.loc[1:L, "rel_xi"].mean()
            if slow < 1.0 and quick > 1.15:
                verdict("support", f"**Slow–quick–slow.** While the target error is {err:.1f}× normal, firms adjust "
                        f"*less* than usual ({slow:.2f}×). When the model is updated they jump ({quick:.2f}×) and then "
                        "settle back, as Heiner (1989, §6) describes.")
            else:
                verdict("neutral", f"No clear punctuated pattern (pre-update adjustment {slow:.2f}×, post-update peak "
                        f"{quick:.2f}×).")
        download(es, "event_study.csv")
