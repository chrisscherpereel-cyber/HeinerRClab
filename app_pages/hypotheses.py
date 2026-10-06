import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.analysis import flex_profit_by_market, ols
from heiner_abm.experiments import PARAMS, slope_summary, switch_point
from heiner_abm.params import FirmSpec
from ui.common import (CAT, base_scenario, behaviour_horizon, download, firm_profit_bars, fmt_p, hypothesis_card, measurement, reps,
                       run_sweep_ui, show_errors, slope_chart, style, verdict)

st.title("Hypothesis tests")
st.caption("Each experiment runs many replicated markets from the sidebar's base scenario. Every condition uses the "
           "same raw-material cost shocks (common random numbers). The key statistic is the **within-market slope "
           "of average profit on flexibility φ**: positive means flexible firms earn more, negative means rigid "
           "firms earn more. Error bars are 95% confidence intervals across replications.")

base = base_scenario()
if not show_errors(base):
    st.stop()
H, cont = measurement()
H = behaviour_horizon(base)      # these tests report profits only; forks matter only for Adaptive agents
R = reps()
st.info(f"Base: **{base.firms[0].rule}**, {base.n_firms} firms, φ = "
        f"{', '.join(f'{f.flex:.2f}' for f in base.firms)}, selection **{base.firms[0].selection}**, "
        f"Δ = {base.market.delta:g}, m* = {base.firms[0].desired_margin:g} · {R} replications × {base.periods} periods. "
        "Change these in the sidebar.", icon="ℹ️")


def parse(txt, default):
    try:
        vals = [float(x) for x in str(txt).replace(";", ",").split(",") if x.strip()]
        return vals or default
    except ValueError:
        st.error("Could not parse the list of values; using defaults.")
        return default


def lever_choices(rule):
    return ["desired_margin", "c_max", "delta"] if rule == "Bertrand" else ["c_max", "delta", "q_range"]


LEVER_HELP = ("Parameter that moves industry profitability. Bertrand: desired margin m*, max raw-material cost or "
              "volatility Δ. Cournot: max raw-material cost, volatility Δ or demand quantity range.")
STEPS_HELP = "Number of evenly spaced values tested within the range. More steps = finer curve, longer run."


def range_help(p):
    return f"Lowest and highest {p.label} to sweep. {p.help}".strip()


def trend_test(key, scn, param, x_title, rng, lo_hi_default, steps_default, expect, horizon=None, rng_help="",
               log_x=False):
    """Sweep one parameter, chart the payoff to flexibility and regress each market's slope on the parameter.
    expect: '+' or '-', the sign the reliability condition predicts."""
    c1, c2 = st.columns(2)
    lo_hi = c1.slider("Range", *rng, lo_hi_default, key=f"{key}_rng", help=rng_help)
    steps = c2.slider("Steps", 3, 12, steps_default, key=f"{key}_steps", help=STEPS_HELP)
    vals = list(np.round(np.geomspace(max(lo_hi[0], 1e-3), lo_hi[1], steps) if log_x else
                         np.linspace(*lo_hi, steps), 3))
    if param == "n_firms" or param == "belief_lag":
        vals = sorted(set(int(round(x)) for x in vals))
    Hk = H if horizon is None else horizon
    if not run_state(key, (tuple(vals), R, Hk, cont, str(scn))):
        return None
    firms, mk = run_sweep_ui(scn, param, vals, R, horizon=Hk, continuation=cont)
    summ = slope_summary(firms, mk, [param])
    fig = slope_chart(summ, param, x_title)
    if log_x:
        fig.update_xaxes(type="log")
    st.plotly_chart(fig)
    per = pd.concat([flex_profit_by_market(g).assign(**{param: v}) for v, g in firms.groupby(param)])
    per = per.dropna(subset=["slope"])
    tab, r2 = ols(per["slope"].to_numpy(float), [per[param].to_numpy(float)], [param])
    coef, pv = tab.loc[1, "coef"], tab.loc[1, "p"]
    st.markdown(f"**Regression of market-level slope on {x_title}** ({len(per)} markets): coefficient = {coef:.3g}, "
                f"p = {fmt_p(pv)}, R² = {r2:.3f}")
    sign = "+" if coef > 0 else "-"
    if pv < 0.05 and sign == expect:
        verdict("support", f"The payoff to flexibility **{'rises' if sign == '+' else 'falls'} significantly** with "
                f"{x_title}, as the reliability condition predicts.")
    elif pv < 0.05:
        verdict("reject", f"The payoff to flexibility **{'rises' if sign == '+' else 'falls'} significantly** with "
                f"{x_title}, the opposite of the reliability condition's prediction.")
    else:
        verdict("neutral", f"No significant relationship between {x_title} and the payoff to flexibility here.")
    download(summ, f"{key}_summary.csv")
    return summ


def run_state(key, params):
    """Run button pattern: results stay visible across reruns until parameters change."""
    if st.button("Run experiment", key=f"btn_{key}", type="primary"):
        st.session_state[key] = params
    return st.session_state.get(key) == params


tabs = st.tabs(["H1 · Free flexibility", "H2 · Profitability switch", "H3 · Volatility", "H4 · Fixed costs",
                "H5 · Competition intensity", "H6 · Regimes & equilibrium", "H7 · Competence",
                "H8 · Perception noise", "H9 · Selection rules", "H10 · Predictable behavior",
                "H11 · Number of rivals", "H12 · Model-updating lag"])
rule = base.firms[0].rule

# ------------------------------------------------------------------------------------------------ H1
with tabs[0]:
    hypothesis_card(
        "H1",
        "The decisive test sets the cost of flexibility to zero (a = b = 0), so any difference in profit between more "
        "and less flexible firms comes from how they use their flexibility. The statistic is the within-market slope "
        "of profit on φ: negative means rigid firms out-earn flexible ones.",
        notes={"optimiser": "Relaxing a constraint on behavior can only help an optimizer, so a market in which rigid "
                            "firms out-earn flexible ones would contradict the optimizing view.",
               "heiner": "When decision errors are costly, flexibility used imperfectly can lower profit even when it "
                         "costs nothing."})
    c1, c2 = st.columns(2)
    lever = c1.selectbox("Profitability lever", lever_choices(rule), format_func=lambda k: PARAMS[k].label, key="h1_lever",
                         help=LEVER_HELP)
    dflt = {"desired_margin": "0, 2, 5, 10, 16", "c_max": "60, 80, 95", "delta": "3, 10, 25", "q_range": "1000, 1500, 3000"}[lever]
    levels = parse(c2.text_input("Levels to test (comma-separated)", dflt, key=f"h1_levels_{lever}",
                                help=f"Values of {PARAMS[lever].label} to simulate, e.g. `0, 5, 10`. Each level is one "
                                     "market condition; flexibility costs a and b are set to 0."), [0, 5, 16])
    scn = base.copy()
    scn.firm_globals.flex_cost_slope = 0.0
    scn.firm_globals.fixed_cost = 0.0
    if run_state("h1", (lever, tuple(levels), R, H, cont, str(scn))):
        firms, mk = run_sweep_ui(scn, lever, levels, R, horizon=H, continuation=cont)
        summ = slope_summary(firms, mk, [lever])
        st.plotly_chart(firm_profit_bars(firms, lever, PARAMS[lever].label))
        t = summ[[lever, "avg_firm_profit", "avg_margin", "slope", "lo", "hi", "p", "spearman", "flex_best", "rigid_best"]]
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            lever: PARAMS[lever].label, "avg_firm_profit": st.column_config.NumberColumn("Avg firm profit", format="%.0f"),
            "avg_margin": st.column_config.NumberColumn("Avg P − c", format="%.2f"),
            "slope": st.column_config.NumberColumn("Slope", format="%.1f"),
            "lo": st.column_config.NumberColumn("95% lo", format="%.1f"),
            "hi": st.column_config.NumberColumn("95% hi", format="%.1f"),
            "p": st.column_config.NumberColumn("p (slope = 0)", format="%.4f"),
            "spearman": st.column_config.NumberColumn("Rank corr. φ~profit", format="%.2f"),
            "flex_best": st.column_config.NumberColumn("Most flexible best", format="%.2f"),
            "rigid_best": st.column_config.NumberColumn("Most rigid best", format="%.2f")})
        neg = summ[summ["hi"] < 0]
        pos = summ[summ["lo"] > 0]
        if len(neg):
            verdict("support", "**Traditional hypothesis rejected.** With free flexibility, rigid firms earned significantly "
                    f"more at {PARAMS[lever].label} = {', '.join(f'{x:g}' for x in neg[lever])} "
                    f"(slope CI entirely below 0). This matches Heiner's prediction.")
        else:
            verdict("neutral", "No condition shows a significantly negative slope, so these levels do not reject the traditional "
                    "hypothesis. Try a less profitable market (lower m*, higher max cost) or more volatility.")
        if len(pos) and len(neg):
            st.caption("Both signs appear: the profit ranking reverses with profitability.")
        download(firms, "H1_firms.csv")

# ------------------------------------------------------------------------------------------------ H2
with tabs[1]:
    hypothesis_card(
        "H2",
        "Industry profitability is varied with one lever (desired margin, maximum raw-material cost, volatility or demand "
        "slope). The experiment locates the profitability at which the profit–flexibility slope changes sign, the "
        "**switch point**. H1 and H4 discriminate between explanations of the switch.")
    c1, c2, c3 = st.columns(3)
    lever = c1.selectbox("Profitability lever", lever_choices(rule), format_func=lambda k: PARAMS[k].label, key="h2_lever",
                         help=LEVER_HELP)
    p = PARAMS[lever]
    rng_default = {"desired_margin": (0.0, 16.0), "c_max": (55.0, 98.0), "delta": (2.0, 35.0), "q_range": (600.0, 4000.0)}[lever]
    lo_hi = c2.slider("Range", float(p.lo), float(p.hi), rng_default, key=f"h2_rng_{lever}", help=range_help(p))
    steps = c3.slider("Steps", 3, 20, 9, key="h2_steps", help=STEPS_HELP)
    xaxis = st.radio("Plot against", ["avg_firm_profit", "avg_margin", lever], horizontal=True, key="h2_x",
                     help="Horizontal axis of the chart: measured industry profitability (average firm profit or "
                          "price − cost margin) or the lever value itself.",
                     format_func=lambda k: {"avg_firm_profit": "Average firm profit", "avg_margin": "Average margin P − c"}.get(k, p.label))
    vals = list(np.round(np.linspace(*lo_hi, steps), 3))
    if run_state("h2", (lever, tuple(vals), R, H, cont, str(base))):
        firms, mk = run_sweep_ui(base, lever, vals, R, horizon=H, continuation=cont)
        summ = slope_summary(firms, mk, [lever])
        xt = {"avg_firm_profit": "Average firm profit per period", "avg_margin": "Average margin P − c"}.get(xaxis, p.label)
        fig = slope_chart(summ, xaxis, xt)
        sp = switch_point(summ, xaxis)
        if sp is not None:
            fig.add_vline(x=sp, line=dict(color=CAT[1], width=1.5, dash="dash"))
            fig.add_annotation(x=sp, y=1, yref="paper", text=f"switch ≈ {sp:.3g}", showarrow=False, yanchor="bottom")
        st.plotly_chart(fig)
        c1, c2 = st.columns(2)
        fr = go.Figure()
        fr.add_trace(go.Scatter(x=summ[xaxis], y=summ["flex_best"], name="Most flexible firm is best", mode="lines+markers",
                                line=dict(color=CAT[0], width=2)))
        fr.add_trace(go.Scatter(x=summ[xaxis], y=summ["rigid_best"], name="Most rigid firm is best", mode="lines+markers",
                                line=dict(color=CAT[1], width=2)))
        fr.update_xaxes(title=xt); fr.update_yaxes(title="Share of replications", range=[0, 1])
        c1.plotly_chart(style(fr, 320, "Who wins?"))
        fs = go.Figure()
        fs.add_trace(go.Scatter(x=summ[xaxis], y=summ["spearman"], mode="lines+markers", name="Spearman ρ",
                                line=dict(color=CAT[2], width=2)))
        fs.add_hline(y=0, line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"))
        fs.update_xaxes(title=xt); fs.update_yaxes(title="Mean rank correlation φ ~ profit", range=[-1, 1])
        c2.plotly_chart(style(fs, 320, "Is profit ordered by flexibility?"))
        if sp is not None:
            verdict("support", f"Switch point found at **{xt} ≈ {sp:.3g}**. Below it rigid firms earn more; above it flexible "
                    "firms do. The RC explains this as mistakes becoming cheaper in profitable markets.")
        else:
            s = "positive" if (summ["slope"] > 0).all() else "negative" if (summ["slope"] < 0).all() else "mixed"
            verdict("neutral", f"No sign change in this range (slopes are {s}). Widen the range or change the lever.")
        download(summ, "H2_summary.csv")

# ------------------------------------------------------------------------------------------------ H3
with tabs[2]:
    hypothesis_card(
        "H3",
        "The volatility of the raw-material cost makes the environment harder to predict. It is also reported as the "
        "serial correlation of cost (lower = less predictable), a common empirical proxy for predictability.",
        notes={"heiner": "Volatility is the *difficulty* side of the CD-gap (the gap between the difficulty of a "
                         "problem and the agent's competence)."})
    c1, c2 = st.columns(2)
    lo_hi = c1.slider("Volatility Δ range", 0.5, 40.0, (2.0, 30.0), key="h3_rng",
                      help="Lowest and highest cost volatility Δ (max raw-material cost change per period) to sweep.")
    steps = c2.slider("Steps", 3, 20, 8, key="h3_steps", help=STEPS_HELP)
    xaxis = st.radio("Plot against", ["delta", "sc_cost", "cd_gap"], horizontal=True, key="h3_x",
                     help="Horizontal axis: Δ itself, an empirical predictability proxy (serial correlation of cost; "
                          "lower = more uncertain), or the measured CD-gap (RMSE of firms' cost perception).",
                     format_func=lambda k: {"delta": "Δ", "sc_cost": "Serial corr. of cost (predictability proxy)",
                                            "cd_gap": "Measured CD-gap (cost-perception RMSE)"}[k])
    vals = list(np.round(np.linspace(*lo_hi, steps), 3))
    if run_state("h3", (tuple(vals), R, H, cont, str(base))):
        firms, mk = run_sweep_ui(base, "delta", vals, R, horizon=H, continuation=cont)
        summ = slope_summary(firms, mk, ["delta"])
        xt = {"delta": "Cost volatility Δ", "sc_cost": "Serial correlation of cost", "cd_gap": "Measured CD-gap"}[xaxis]
        fig = slope_chart(summ, xaxis, xt)
        if xaxis == "sc_cost":
            fig.update_xaxes(autorange="reversed", title=xt + " (reversed: right = more uncertain)")
        st.plotly_chart(fig)
        per = pd.concat([flex_profit_by_market(g).assign(delta=d) for d, g in firms.groupby("delta")])
        per = per.dropna(subset=["slope"])
        tab, r2 = ols(per["slope"].to_numpy(float), [per["delta"].to_numpy(float)], ["Δ"])
        coef, pv = tab.loc[1, "coef"], tab.loc[1, "p"]
        st.markdown(f"**Regression of market-level slope on Δ** ({len(per)} markets): coefficient = {coef:.2f}, "
                    f"p = {fmt_p(pv)}, R² = {r2:.3f}")
        if coef < 0 and pv < 0.05:
            verdict("support", "Flexibility's payoff **falls significantly** as volatility rises. This supports the RC "
                    "and contradicts the traditional prediction.")
        elif coef > 0 and pv < 0.05:
            verdict("reject", "Flexibility's payoff **rises significantly** with volatility, as the traditional view predicts.")
        else:
            verdict("neutral", "No significant relationship between volatility and the payoff to flexibility here.")
        download(summ, "H3_summary.csv")

# ------------------------------------------------------------------------------------------------ H4
with tabs[3]:
    hypothesis_card(
        "H4",
        "A fixed cost per period, identical for all firms, is added; marginal costs are unchanged. **Control:** if the "
        "firms' decision margin ignores fixed costs (margin = P − c), decisions and therefore slopes cannot change. The "
        "substantive test uses a margin that includes fixed cost per unit (P − c − F/q), so fixed costs raise the stakes "
        "of each decision.")
    c1, c2, c3 = st.columns(3)
    lever = c1.selectbox("Profitability lever", lever_choices(rule), format_func=lambda k: PARAMS[k].label, key="h4_lever",
                         help=LEVER_HELP)
    p = PARAMS[lever]
    rng_default = {"desired_margin": (0.0, 16.0), "c_max": (55.0, 98.0), "delta": (2.0, 35.0), "q_range": (600.0, 4000.0)}[lever]
    lo_hi = c2.slider("Lever range", float(p.lo), float(p.hi), rng_default, key=f"h4_rng_{lever}", help=range_help(p))
    steps = c3.slider("Steps", 3, 15, 7, key="h4_steps", help=STEPS_HELP)
    c1, c2 = st.columns(2)
    fcs = parse(c1.text_input("Fixed-cost levels b", "0, 500, 1500", key="h4_fc",
                                help="Comma-separated fixed costs per period, identical for every firm. One line "
                                     "(and one switch point) per level."), [0, 500, 1500])
    incl = c2.toggle("Bertrand margin includes F/q", value=True, key="h4_incl",
                     help="On = margin P − c − F/q, so fixed costs change decisions. Off = baseline "
                          "margin P − c: the control, where fixed costs cannot change behavior.")
    vals = list(np.round(np.linspace(*lo_hi, steps), 3))
    scn = base.copy()
    scn.firm_globals.margin_includes_fixed = incl
    if run_state("h4", (lever, tuple(vals), tuple(fcs), incl, R, H, cont, str(base))):
        firms, mk = run_sweep_ui(scn, lever, vals, R, "fixed_cost", fcs, horizon=H, continuation=cont)
        summ = slope_summary(firms, mk, [lever, "fixed_cost"])
        st.plotly_chart(slope_chart(summ, lever, p.label, group="fixed_cost", group_title="b"))
        sps = []
        for b, g in summ.groupby("fixed_cost"):
            sps.append(dict(fixed_cost=b, switch_lever=switch_point(g, lever), switch_margin=switch_point(g, "avg_margin")))
        spd = pd.DataFrame(sps)
        st.dataframe(spd, hide_index=True, column_config={
            "fixed_cost": "Fixed cost b", "switch_lever": st.column_config.NumberColumn(f"Switch point ({p.label})", format="%.3g"),
            "switch_margin": st.column_config.NumberColumn("Switch point (avg P − c)", format="%.3g")})
        ok = spd.dropna(subset=["switch_lever"])
        if not incl:
            verdict("neutral", "Control condition: firms ignore fixed costs when deciding, so the lines coincide. A fixed "
                    "cost is a pure level shift that changes no one's relative performance.")
        elif len(ok) >= 2 and np.all(np.diff(ok.sort_values("fixed_cost")["switch_lever"]) > 0):
            verdict("support", "The switch point **rises with fixed costs**, as the RC predicts. The traditional hypothesis "
                    "predicts no shift because marginal costs are unchanged.")
        elif len(ok) >= 2:
            verdict("warn", "Switch points were found but do not rise monotonically with fixed costs.")
        else:
            verdict("neutral", "Fewer than two switch points were found in this range. Widen the lever range.")
        download(summ, "H4_summary.csv")

# ------------------------------------------------------------------------------------------------ H5
with tabs[4]:
    hypothesis_card(
        "H5",
        "Exploratory. In more competitive markets (smaller desired margin m*), does the switch to favoring flexible "
        "firms happen at a *lower* level of industry profit? A profitability lever is swept at several competition "
        "intensities.")
    if rule != "Bertrand":
        st.warning("Desired margin m* only affects Bertrand firms. Switch the base scenario to Bertrand.")
    else:
        c1, c2, c3 = st.columns(3)
        lever = c1.selectbox("Profitability lever", ["c_max", "delta", "q_range"], format_func=lambda k: PARAMS[k].label,
                             key="h5_lever", help="Parameter swept to move industry profitability at each m*.")
        p = PARAMS[lever]
        rng_default = {"c_max": (50.0, 98.0), "delta": (2.0, 35.0), "q_range": (600.0, 4000.0)}[lever]
        lo_hi = c2.slider("Lever range", float(p.lo), float(p.hi), rng_default, key=f"h5_rng_{lever}", help=range_help(p))
        steps = c3.slider("Steps", 3, 15, 8, key="h5_steps", help=STEPS_HELP)
        ms = parse(st.text_input("Desired margins m* (competition intensities)", "1, 4, 8", key="h5_ms",
                                     help="Comma-separated Bertrand desired margins. Small m* = fierce competition, "
                                          "large m* = gentlemanly. One line per value."), [1, 4, 8])
        vals = list(np.round(np.linspace(*lo_hi, steps), 3))
        if run_state("h5", (lever, tuple(vals), tuple(ms), R, H, cont, str(base))):
            firms, mk = run_sweep_ui(base, lever, vals, R, "desired_margin", ms, horizon=H, continuation=cont)
            summ = slope_summary(firms, mk, [lever, "desired_margin"])
            st.plotly_chart(slope_chart(summ, "avg_firm_profit", "Average firm profit per period", group="desired_margin",
                                        group_title="m*"))
            spd = pd.DataFrame([dict(m=m, switch_profit=switch_point(g, "avg_firm_profit"))
                                for m, g in summ.groupby("desired_margin")])
            st.dataframe(spd, hide_index=True, column_config={"m": "m*", "switch_profit": st.column_config.NumberColumn(
                "Switch point (avg firm profit)", format="%.0f")})
            ok = spd.dropna()
            if len(ok) >= 2 and np.all(np.diff(ok.sort_values("m")["switch_profit"]) > 0):
                verdict("support", "More intense competition (lower m*) switches to favoring flexibility at **lower** profit "
                        "levels, as H5 predicts.")
            elif len(ok) >= 2:
                verdict("warn", "Switch points found, but the ordering differs from H5.")
            else:
                verdict("neutral", "Not enough switch points in range. Try another lever or range.")
            download(summ, "H5_summary.csv")

# ------------------------------------------------------------------------------------------------ H6
with tabs[5]:
    hypothesis_card(
        "H6",
        "Scaling every firm's φ moves the industry between three regimes: **sluggish** (prices smoother than cost, "
        "serial corr. of price > cost), **tracking** (prices follow cost), and **oscillating** (prices overshoot). "
        "The question is whether inflexible industries stay away from the competitive equilibrium and earn higher "
        "profits without any coordination.")
    c1, c2 = st.columns(2)
    lo_hi = c1.slider("Flexibility scale range (× every φ)", 0.05, 4.0, (0.1, 2.5), key="h6_rng",
                      help="Multiplier applied to every firm's φ. Low = sluggish industry, high = oscillating. "
                           "Values are spaced geometrically.")
    steps = c2.slider("Steps", 3, 20, 10, key="h6_steps", help=STEPS_HELP)
    vals = list(np.round(np.geomspace(*lo_hi, steps), 3))
    if run_state("h6", (tuple(vals), R, H, cont, str(base))):
        firms, mk = run_sweep_ui(base, "flex_scale", vals, R, horizon=H, continuation=cont)
        g = mk.groupby("flex_scale").agg(sc_price=("sc_price", "mean"), sc_cost=("sc_cost", "mean"),
                                         margin=("avg_margin", "mean"), gap=("avg_abs_gap", "mean"),
                                         industry_profit=("industry_profit", "mean"),
                                         sd_price=("sd_price", "mean")).reset_index()
        c1, c2 = st.columns(2)
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=g["flex_scale"], y=g["sc_price"], name="Serial corr. of price", mode="lines+markers",
                                line=dict(color=CAT[0], width=2)))
        f1.add_trace(go.Scatter(x=g["flex_scale"], y=g["sc_cost"], name="Serial corr. of cost", mode="lines+markers",
                                line=dict(color=CAT[1], width=2, dash="dash")))
        f1.update_xaxes(title="Flexibility scale", type="log"); f1.update_yaxes(title="Lag-1 serial correlation")
        c1.plotly_chart(style(f1, 340, "Market regime"))
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=g["flex_scale"], y=g["industry_profit"], mode="lines+markers", name="Industry profit",
                                line=dict(color=CAT[2], width=2), showlegend=False))
        f2.update_xaxes(title="Flexibility scale", type="log"); f2.update_yaxes(title="Industry profit per period")
        c2.plotly_chart(style(f2, 340, "Industry profit"))
        f3 = go.Figure()
        f3.add_trace(go.Scatter(x=g["flex_scale"], y=g["gap"], mode="lines+markers", name="avg |P − c|",
                                line=dict(color=CAT[6], width=2), showlegend=False))
        f3.update_xaxes(title="Flexibility scale", type="log"); f3.update_yaxes(title="Average |P − c|")
        st.plotly_chart(style(f3, 300, "Distance from the price = cost line"))
        rho = np.corrcoef(np.log(g["flex_scale"]), g["industry_profit"])[0, 1]
        if rho < -0.5:
            verdict("support", f"Industry profit **falls** as the whole industry becomes more flexible (ρ = {rho:.2f}). "
                    "Rigid industries stay further from equilibrium and earn more, as H6 predicts.")
        elif rho > 0.5:
            verdict("reject", f"Industry profit **rises** with industry flexibility here (ρ = {rho:.2f}).")
        else:
            verdict("neutral", f"No clear monotone relationship between industry flexibility and profit (ρ = {rho:.2f}).")
        download(g, "H6_regimes.csv")

# ------------------------------------------------------------------------------------------------ H7
with tabs[6]:
    hypothesis_card(
        "H7",
        "Raising cost foresight κ lets firms anticipate part of the coming cost change, so their recommendations are "
        "right more often.",
        notes={"heiner": "Competence is the other side of the CD-gap: more competence narrows the gap."})
    trend_test("h7", base, "foresight", "competence κ", (0.0, 1.0), (0.0, 1.0), 5, "+",
               rng_help="Lowest and highest cost foresight κ: the share of the coming cost change firms anticipate.")

# ------------------------------------------------------------------------------------------------ H8
with tabs[7]:
    hypothesis_card(
        "H8",
        "Perception noise σ adds a random error to every firm's estimate of the coming cost without changing the "
        "environment (Δ stays fixed).",
        notes={"heiner": "Perception noise widens the CD-gap from the agent's side (lower competence)."})
    trend_test("h8", base, "noise", "perception noise σ", (0.0, 20.0), (0.0, 20.0), 5, "-",
               rng_help="Lowest and highest standard deviation of the error in firms' cost estimates.")

# ------------------------------------------------------------------------------------------------ H9
with tabs[8]:
    hypothesis_card(
        "H9",
        "Every firm has the same flexibility φ but a different **selection rule**: Always (fully flexible), "
        "Never (rule B, rigid), Small (SR1, deviate only on changes below θ) and Large (SR2, deviate only on changes "
        "above θ, 'big imbalances send clear signals'). They compete in the same market on the same shocks.")
    c1, c2, c3 = st.columns(3)
    lo_hi = c1.slider("Volatility Δ range", 0.5, 40.0, (2.0, 32.0), key="h9_rng",
                      help="Lowest and highest cost volatility Δ to sweep.")
    steps = c2.slider("Steps", 3, 12, 6, key="h9_steps", help=STEPS_HELP)
    add_adapt = c3.toggle("Add an Adaptive firm", False, key="h9_adapt",
                          help="Adds a fifth firm that learns, per size of change, whether deviating pays, from what it "
                               "has observed (see the sidebar's Adaptive settings).")
    phi = float(np.mean([f.flex for f in base.firms]))
    thr = float(base.firms[0].threshold)
    sels = ["Always", "Never", "Small", "Large"] + (["Adaptive"] if add_adapt else [])
    scn = base.copy()
    t0 = base.firms[0]
    scn.firms = [FirmSpec(rule=t0.rule, flex=min(phi, 1.0) if t0.rule == "Cournot" else phi, selection=sname,
                          threshold=thr, desired_margin=t0.desired_margin, foresight=t0.foresight, noise=t0.noise,
                          label=sname) for sname in sels]
    scn.firm_globals.q0 = round(base.firm_globals.q0 * base.n_firms / len(sels))
    st.caption(f"{len(sels)} firms, each with φ = {phi:.2f} (mean of the sidebar ladder) and θ = {thr:g}.")
    vals = list(np.round(np.linspace(*lo_hi, steps), 3))
    Hs = max(H, 2) if add_adapt else 1
    if run_state("h9", (tuple(vals), R, Hs, cont, str(scn))):
        firms, mk = run_sweep_ui(scn, "delta", vals, R, horizon=Hs, continuation=cont)
        agg = firms.groupby(["delta", "selection"])["avg_profit"].agg(["mean", "std", "count"]).reset_index()
        agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
        fig = go.Figure()
        for k, sname in enumerate(sels):
            d = agg[agg["selection"] == sname]
            fig.add_trace(go.Scatter(x=d["delta"], y=d["mean"], name=sname, mode="lines+markers",
                                     line=dict(color=CAT[k % len(CAT)], width=2),
                                     error_y=dict(type="data", array=d["ci"], thickness=1, width=3)))
        fig.update_xaxes(title="Cost volatility Δ"); fig.update_yaxes(title="Average profit per period")
        st.plotly_chart(style(fig, title="Profit by selection rule"))
        wide = firms.pivot_table(index=["market", "delta"], columns="selection", values="avg_profit").reset_index()
        wide["always_minus_never"] = wide["Always"] - wide["Never"]
        wide["large_minus_small"] = wide["Large"] - wide["Small"]
        tab, r2 = ols(wide["always_minus_never"].to_numpy(float), [wide["delta"].to_numpy(float)], ["Δ"])
        coef, pv = tab.loc[1, "coef"], tab.loc[1, "p"]
        st.markdown(f"**Regression of (Always − Never) profit on Δ** ({len(wide)} markets): coefficient = {coef:.3g}, "
                    f"p = {fmt_p(pv)}. Mean (Large − Small) = {wide['large_minus_small'].mean():,.1f}.")
        worse = agg.pivot(index="delta", columns="selection", values="mean")
        beaten = (worse.drop(columns="Always").max(axis=1) > worse["Always"]).any()
        if coef < 0 and pv < 0.05:
            verdict("support", "The fully flexible rule **loses ground** to rule B as volatility rises, as the "
                    "reliability condition predicts.")
        elif coef > 0 and pv < 0.05:
            verdict("reject", "The fully flexible rule **gains** on rule B as volatility rises.")
        else:
            verdict("neutral", "No significant trend in the flexible rule's advantage over rule B.")
        if beaten:
            st.caption("At some Δ a less flexible rule earned more than **Always** on average: evidence against "
                       "'flexibility is never harmful'.")
        download(agg, "H9_selection_rules.csv")

# ------------------------------------------------------------------------------------------------ H10
with tabs[9]:
    hypothesis_card(
        "H10",
        "Every firm uses the **Adaptive** selection rule, which learns from its own observed experience when deviating "
        "from keeping its output pays; no rule is imposed on how often to deviate. Cost volatility is swept and the share "
        "of periods in which firms change their output is measured.",
        notes={"heiner": "This is Heiner's (1983) central claim, 'The origin of predictable behavior': imperfect "
                         "agents facing more uncertainty restrict themselves to fewer, more rule-governed actions.",
               "satisficing": "More volatile results fall below aspiration more often, which triggers more search and "
                              "change."})
    c1, c2 = st.columns(2)
    lo_hi = c1.slider("Volatility Δ range", 0.5, 40.0, (2.0, 32.0), key="h10_rng",
                      help="Lowest and highest cost volatility Δ to sweep.")
    steps = c2.slider("Steps", 3, 12, 6, key="h10_steps", help=STEPS_HELP)
    scn = base.copy()
    for f in scn.firms:
        f.selection = "Adaptive"
    H10 = max(H, 2)
    st.caption(f"Adaptive firms judge each deviation over {scn.adaptive.window} periods from what they have observed "
               f"({'researcher-only look-ahead feedback' if scn.adaptive.feedback == 'lookahead' else 'no future information'}); "
               f"r and w are measured by the researcher over H = {H10} periods (sidebar H, at least 2).")
    vals = list(np.round(np.linspace(*lo_hi, steps), 3))
    if run_state("h10", (tuple(vals), R, H10, cont, str(scn))):
        firms, mk = run_sweep_ui(scn, "delta", vals, R, horizon=H10, continuation=cont)
        firms = firms[firms["opportunities"] > 0].copy()
        firms["dev_rate"] = firms["deviations"] / firms["opportunities"]
        per = firms.groupby(["market", "delta"]).agg(dev_rate=("dev_rate", "mean"), r=("r", "mean"),
                                                     w=("w", "mean")).reset_index()
        g = per.groupby("delta").agg(dev=("dev_rate", "mean"), dev_sd=("dev_rate", "std"), n=("dev_rate", "count"),
                                     r=("r", "mean"), w=("w", "mean")).reset_index()
        g["ci"] = 1.96 * g["dev_sd"] / np.sqrt(g["n"])
        c1, c2 = st.columns(2)
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=g["delta"], y=g["dev"], mode="lines+markers", name="Deviation rate",
                                line=dict(color=CAT[0], width=2), error_y=dict(type="data", array=g["ci"], thickness=1,
                                                                               width=3)))
        f1.update_xaxes(title="Cost volatility Δ"); f1.update_yaxes(title="Share of opportunities with a deviation",
                                                                    range=[0, 1])
        c1.plotly_chart(style(f1, 340, "How often do firms deviate from rule B?"))
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=g["delta"], y=g["r"], mode="lines+markers", name="r (deviate | exception)",
                                line=dict(color=CAT[2], width=2)))
        f2.add_trace(go.Scatter(x=g["delta"], y=g["w"], mode="lines+markers", name="w (deviate | no exception)",
                                line=dict(color=CAT[1], width=2)))
        f2.update_xaxes(title="Cost volatility Δ"); f2.update_yaxes(title="Rate", range=[0, 1])
        c2.plotly_chart(style(f2, 340, "Learned reliability"))
        tab, r2 = ols(per["dev_rate"].to_numpy(float), [per["delta"].to_numpy(float)], ["Δ"])
        coef, pv = tab.loc[1, "coef"], tab.loc[1, "p"]
        st.markdown(f"**Regression of the deviation rate on Δ** ({len(per)} markets): coefficient = {coef:.4f}, "
                    f"p = {fmt_p(pv)}, R² = {r2:.3f}")
        if coef < 0 and pv < 0.05:
            verdict("support", "Reliability-learning firms **deviate less** as uncertainty rises: behavior becomes "
                    "more predictable, as Heiner (1983) argued.")
        elif coef > 0 and pv < 0.05:
            verdict("reject", "Firms **deviate more** as uncertainty rises, as the optimizing and satisficing views "
                    "predict.")
        else:
            verdict("neutral", "No significant trend in how often firms deviate.")
        download(g, "H10_predictability.csv")

# ------------------------------------------------------------------------------------------------ H11
with tabs[10]:
    hypothesis_card(
        "H11",
        "Markets with more firms are compared, with the same range of flexibility and comparable total initial output. "
        "With more firms, the output each firm should aim at moves more with its rivals' choices.",
        notes={"heiner": "In Heiner (1989) the slope of the target map is f′ = −(n−1)/2 in symmetric Cournot, so the "
                         "reliable adjustment speed β₀ = 1/((1+K)(1−f′)) falls as n rises.",
               "cobweb": "More firms shrink the stability region of simultaneous adjustment (φ < 4/(n + 1))."})
    trend_test("h11", base, "n_firms", "number of firms n", (2.0, 12.0), (2.0, 12.0), 6, "-",
               rng_help="Smallest and largest number of firms; each market spreads φ over the sidebar's range.")

# ------------------------------------------------------------------------------------------------ H12
with tabs[11]:
    hypothesis_card(
        "H12",
        "Unannounced demand-regime shifts (hazard λ from the sidebar, or 0.02 if off) make the demand model that "
        "Cournot firms use for their best replies wrong until they update it, L periods later. The test sweeps L for "
        "a Cournot industry (sidebar φ if already Cournot, otherwise φ from 0.1 to 0.4, below the stability limit).")
    scn = base.copy()
    if base.firms[0].rule != "Cournot":
        for f, phi in zip(scn.firms, np.linspace(0.1, 0.4, scn.n_firms)):
            f.rule, f.flex = "Cournot", float(phi)
    scn.structural.enabled = True
    if not base.structural.enabled:
        scn.structural.hazard = 0.02
    trend_test("h12", scn, "belief_lag", "model-updating lag L", (0.0, 200.0), (0.0, 100.0), 5, "-",
               rng_help="Shortest and longest lag, in periods, before Cournot firms learn a new demand regime.")
