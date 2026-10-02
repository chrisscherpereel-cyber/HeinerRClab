import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from heiner_abm.analysis import auc, sdt_roc
from heiner_abm.experiments import MEASURE_LABELS, PARAMS
from ui.common import (CAT, base_scenario, cached_bound, cached_single, download, firm_colors, hypothesis_card,
                       measure_opts, measurement, reps, run_sweep_ui, show_errors, style, to_json, verdict)

st.title("The dynamic reliability condition")
st.markdown(
    "Heiner's 1983 condition treats every decision as a separate, one-shot bet. In a market, a production change "
    "**persists** (rule B keeps the new level) and **rivals react** to it. This page extends the RC to that setting, "
    "along three lines:\n"
    "1. **Decomposition.** The value of each decision is split into its *immediate* effect (Heiner's one-shot, one-period "
    "comparison), a *persistence* effect, and a *strategic feedback* effect.\n"
    "2. **Heiner (1989).** A firm partially adjusting toward an imperfectly perceived target has a maximal "
    "reliable adjustment speed. The bound is computed from measured decision errors and compared with the "
    "profit-maximizing flexibility.\n"
    "3. **Signal detection.** Heiner's r and w are hit and false-alarm rates. The ROC curve shows how well a firm "
    "can tell preferred exceptions apart, and where its best deviation threshold lies.")

base = base_scenario()
if not show_errors(base):
    st.stop()
H, cont = measurement()
disc, split = measure_opts()
tabs = st.tabs(["① Decomposing a decision's value", "② Heiner (1989) partial-adjustment bound",
                "③ Signal-detection view"])

# ================================================================================================ 1
with tabs[0]:
    hypothesis_card(
        "DRC",
        r"For every opportunity, the gain of following the rule instead of B is measured three ways over H periods "
        r"(discount γ):"
        "\n\n* **Immediate** (H = 1, the one-shot RC): this period only, rivals' choices fixed.\n"
        "* **+ Persistence**: the firm keeps the new level for H periods; rivals follow their *actual* path and do "
        "not react.\n"
        "* **+ Strategic feedback**: the market is forked, so rivals *react* to the deviation.\n\n"
        "Decomposition: full = immediate + (persist − immediate) + (full − persist).")
    if H < 2:
        st.warning("Set the counterfactual horizon H ≥ 2 in the sidebar; with H = 1 all three measures coincide.", icon="⚠️")
    c = st.columns(3)
    keys = ["delta", "desired_margin", "hazard", "q_range", "n_firms", "foresight"]
    lever = c[0].selectbox("Vary", keys, format_func=lambda k: PARAMS[k].label, key="drc_lever",
                           help="Parameter swept to see how the three components of a decision's value change.")
    P = PARAMS[lever]
    dflt = {"delta": (2.0, 30.0), "desired_margin": (0.0, 16.0), "hazard": (0.0, 0.1), "q_range": (600.0, 4000.0),
            "n_firms": (2.0, 10.0), "foresight": (0.0, 1.0)}[lever]
    rng = c[1].slider("Range", float(P.lo), float(P.hi), dflt, key=f"drc_rng_{lever}",
                      help=f"Lowest and highest {P.label} to sweep. {P.help}".strip())
    steps = c[2].slider("Steps", 3, 12, 6, key="drc_steps",
                        help="Number of evenly spaced values within the range.")
    vals = np.round(np.linspace(*rng, steps), 3)
    if lever == "n_firms":
        vals = sorted(set(int(round(x)) for x in vals))
    R = max(3, min(reps(), 10))
    st.caption(f"{len(vals) * R} markets at H = {H}, γ = {disc:g} ({R} replications per point; forks make this slower "
               "than profit-only sweeps).")
    key = ("drc", lever, tuple(vals), R, H, cont, disc, str(base))
    if st.button("Run decomposition", type="primary", key="drc_btn"):
        st.session_state["drc_key"] = key
    if st.session_state.get("drc_key") == key:
        firms, mk = run_sweep_ui(base, lever, vals, R, horizon=H, continuation=cont, discount=disc, oos_split=split)
        g = firms.groupby(lever).agg(
            immediate=("gain_per_opp_static", "mean"), persist=("gain_per_opp_persist", "mean"),
            full=("gain_per_opp", "mean"), rc_static=("rc_holds_static", "mean"),
            rc_persist=("rc_holds_persist", "mean"), rc_full=("rc_holds", "mean"),
            pi_static=("pi_static", "mean"), pi_persist=("pi_persist", "mean"), pi_full=("pi", "mean")).reset_index()
        g["persistence"] = g["persist"] - g["immediate"]
        g["feedback"] = g["full"] - g["persist"]
        c1, c2 = st.columns(2)
        fb = go.Figure()
        for k, (col, name) in enumerate((("immediate", "Immediate (one period)"), ("persistence", "Persistence effect"),
                                         ("feedback", "Strategic feedback effect"))):
            fb.add_trace(go.Bar(x=[f"{x:g}" for x in g[lever]], y=g[col], name=name, marker_color=CAT[k],
                                hovertemplate=f"{P.label} %{{x}}<br>{name}: %{{y:,.1f}}<extra></extra>"))
        fb.add_trace(go.Scatter(x=[f"{x:g}" for x in g[lever]], y=g["full"], name="Total (full dynamic)",
                                mode="lines+markers", line=dict(color="#0b0b0b", width=2), marker=dict(size=8)))
        fb.update_layout(barmode="relative", bargap=0.3)
        fb.update_xaxes(title=P.label, type="category")
        fb.update_yaxes(title="Mean value of following the rule vs B, per opportunity")
        c1.plotly_chart(style(fb, 420, "Where does a decision's value come from?"))
        fr = go.Figure()
        for k, (col, m) in enumerate((("rc_static", "static"), ("rc_persist", "persist"), ("rc_full", "full"))):
            fr.add_trace(go.Scatter(x=g[lever], y=g[col], name=MEASURE_LABELS[m], mode="lines+markers",
                                    line=dict(color=CAT[k], width=2, dash=[None, "dash", None][k])))
        fr.update_xaxes(title=P.label); fr.update_yaxes(title="Share of firms satisfying the RC", range=[-0.02, 1.02])
        c2.plotly_chart(style(fr, 420, "Does the RC hold? It depends on the measure"))
        share_fb = (g["feedback"].abs() / (g["immediate"].abs() + g["persistence"].abs() + g["feedback"].abs())).mean()
        disagree = (np.abs(g["rc_static"] - g["rc_full"])).mean()
        verdict("support" if share_fb > 0.34 or disagree > 0.2 else "neutral",
                f"On average **{share_fb:.0%}** of the absolute value of a decision comes from strategic feedback, and the "
                f"one-period and full RC classify firms differently **{disagree:.0%}** of the time. "
                + ("The static RC omits a first-order part of decision value." if share_fb > 0.34 or disagree > 0.2
                   else "In these conditions the static RC is a reasonable approximation."))
        st.dataframe(g, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in g.columns if c != lever})
        download(g, "dynamic_rc_decomposition.csv")

# ================================================================================================ 2
with tabs[1]:
    hypothesis_card(
        "BOUND",
        r"The model's Cournot rule is Heiner's (1989) partial-adjustment model: the firm moves a fraction φ (his β) of the way from last "
        r"period's output toward its **perceived** best reply $\hat x_t = x^*_t + \xi_t$. The gap to the true target is "
        r"$d_t = (1-\beta)\Delta^*_t - \beta\,\xi_t$: adjusting slowly lags the target, adjusting fast imports the "
        r"error. Theorem 2 bounds the speed that still converges despite errors: "
        r"$\beta \le \beta_0 = \dfrac{1}{(1+K)(1-f')}$, where $K$ bounds the error-to-signal ratio $|\xi_t/\Delta^*_t|$ "
        r"and $f'$ is the slope of the target map. With $n$ symmetric Cournot firms, $f' = -(n-1)/2$, so "
        r"$\beta_0 = 2/((1+K)(n+1))$. Without errors the market is stable only for $\beta < 4/(n+1)$ (Theocharis 1960).")
    c = st.columns(4)
    diff = c[0].selectbox("Difficulty", ["delta", "hazard", "noise"], format_func=lambda k: PARAMS[k].label, key="hb_diff",
                          help="Source of the error ξ in firms' perceived best reply: cost volatility, demand-regime "
                               "shifts, or cost-perception noise.")
    lv_default = {"delta": "2, 10, 20, 30", "hazard": "0, 0.02, 0.05, 0.1", "noise": "0, 5, 10, 20"}[diff]
    levels = [float(x) for x in c[1].text_input("Levels", lv_default, key=f"hb_lv_{diff}",
                                                      help="Comma-separated difficulty values; one profit-vs-φ curve "
                                                           "per level.").split(",") if x.strip()]
    phi_hi = c[2].slider("φ grid up to", 0.2, 1.0, 1.0, 0.05, key="hb_phimax",
                         help="Largest common adjustment speed φ (Heiner's β) tested; the grid starts at 0.05.")
    npts = c[3].slider("Grid points", 8, 40, 20, key="hb_n",
                       help="Number of φ values between 0.05 and the upper limit.")
    phis = tuple(np.round(np.linspace(0.05, phi_hi, npts), 3))
    R2 = max(3, min(reps(), 15))
    scn = base.copy()
    if diff == "hazard":
        scn.structural.enabled = True
    n = scn.n_firms
    key = ("hb", diff, tuple(levels), phis, R2, to_json(scn))
    st.caption(f"Symmetric Cournot industry with {n} firms, every firm uses the same φ. "
               f"{len(levels) * len(phis) * R2:,} markets.")
    if st.button("Run partial-adjustment test", type="primary", key="hb_btn"):
        st.session_state["hb_key"] = key
    if st.session_state.get("hb_key") == key:
        with st.spinner("Simulating…"):
            g = cached_bound(to_json(scn), phis, diff, tuple(levels), R2)
        lim = 4.0 / (n + 1)
        rows = []
        fig = go.Figure()
        fk = go.Figure()
        for k, (lv, gg) in enumerate(g.groupby(diff)):
            gg = gg.sort_values("common_flex")
            col = CAT[k % len(CAT)]
            best = gg.loc[gg["avg_profit"].idxmax()]
            ok = gg[gg["common_flex"] <= gg["bound"]]
            phi_hat = float(ok["common_flex"].max()) if len(ok) else np.nan
            rows.append(dict(level=lv, phi_star=best.common_flex, heiner_bound=phi_hat,
                             K_at_bound=float(ok.loc[ok["common_flex"].idxmax(), "K"]) if len(ok) else np.nan,
                             stability_limit=lim, profit_at_phi_star=best.avg_profit))
            fig.add_trace(go.Scatter(x=gg["common_flex"], y=gg["avg_profit"], mode="lines", name=f"{PARAMS[diff].label.split(' (')[0]} = {lv:g}",
                                     line=dict(color=col, width=2)))
            fig.add_trace(go.Scatter(x=[best.common_flex], y=[best.avg_profit], mode="markers", showlegend=False,
                                     marker=dict(color=col, size=12, symbol="star", line=dict(color="white", width=1.5)),
                                     hovertemplate="φ* = %{x:.2f}<extra></extra>"))
            if not np.isnan(phi_hat):
                yb = float(np.interp(phi_hat, gg["common_flex"], gg["avg_profit"]))
                fig.add_trace(go.Scatter(x=[phi_hat], y=[yb], mode="markers", showlegend=False,
                                         marker=dict(color=col, size=11, symbol="line-ns-open", line=dict(width=3)),
                                         hovertemplate="Heiner bound φ̂ = %{x:.2f}<extra></extra>"))
            fk.add_trace(go.Scatter(x=gg["common_flex"], y=gg["bound"], mode="lines", name=f"β₀(K(φ)), level {lv:g}",
                                    line=dict(color=col, width=2)))
        fk.add_trace(go.Scatter(x=list(phis), y=list(phis), mode="lines", name="φ = β₀ (45°)",
                                line=dict(color="rgba(128,128,128,0.9)", width=1.5, dash="dot")))
        fig.add_vline(x=lim, line=dict(color=CAT[7], width=1.5, dash="dash"))
        fig.add_annotation(x=lim, y=1, yref="paper", text=f"instability 4/(n+1) = {lim:.2f}", showarrow=False,
                           yanchor="bottom", font=dict(color=CAT[7]))
        fig.update_xaxes(title="Common partial-adjustment weight φ (β)")
        fig.update_yaxes(title="Average firm profit per period")
        c1, c2 = st.columns([1.4, 1])
        c1.plotly_chart(style(fig, 440, "Profit vs adjustment speed (★ = optimum, | = Heiner bound)"))
        fk.update_xaxes(title="φ"); fk.update_yaxes(title="Heiner bound β₀ from measured K", range=[0, max(0.35, float(g["bound"].max()) * 1.1)])
        c2.plotly_chart(style(fk, 440, "Self-consistent bound: largest φ below the curve"))
        tb = pd.DataFrame(rows)
        st.dataframe(tb, hide_index=True, width="stretch", column_config={
            "level": PARAMS[diff].label, "phi_star": st.column_config.NumberColumn("Profit-maximizing φ*", format="%.2f"),
            "heiner_bound": st.column_config.NumberColumn("Heiner bound φ̂ (largest φ ≤ β₀(K(φ)))", format="%.2f"),
            "K_at_bound": st.column_config.NumberColumn("Measured K at φ̂", format="%.2f"),
            "stability_limit": st.column_config.NumberColumn("Stability limit 4/(n+1)", format="%.2f"),
            "profit_at_phi_star": st.column_config.NumberColumn("Profit at φ*", format="%.0f")})
        step = float(np.diff(phis).mean()) if len(phis) > 1 else 0.05
        within = (tb["phi_star"] <= tb["heiner_bound"] + 2 * step).mean()
        rho = stats.spearmanr(tb["level"], tb["phi_star"])[0] if tb["phi_star"].nunique() > 1 else 0.0
        if within >= 0.75:
            verdict("support", f"The profit-maximizing φ* lies at or below Heiner's bound (within two grid steps) in "
                    f"**{within:.0%}** of conditions. Full adjustment (φ = 1) is never optimal; beyond 4/(n+1) the market "
                    "becomes unstable, as the theory implies.")
        else:
            verdict("warn", f"φ* exceeds Heiner's bound in {1 - within:.0%} of conditions. The bound is a sufficient "
                    "condition (Theorem 2), so profitable speeds slightly above it are possible.")
        st.caption(f"Rank correlation between difficulty and φ*: ρ = {rho:.2f}. Heiner predicts ρ < 0 (more difficulty → "
                   "slower adjustment). With few levels and flat profit curves near the optimum, φ* can be noisy; add "
                   "replications to sharpen it. K is measured as RMS(ξ)/RMS(Δ*), a robust version of the theorem's bound "
                   "on |ξ/Δ*|.")
        download(g, "heiner1989_bound.csv")

# ================================================================================================ 3
with tabs[2]:
    hypothesis_card(
        "SDT",
        "Treat the recommended change |q* − q| as a *signal* that a preferred exception exists. A selection rule "
        "that deviates when the signal exceeds θ (SR2, 'Large') has hit rate r(θ) and false-alarm rate w(θ). Sweeping θ "
        "traces the firm's **ROC curve**. Its area (AUC) is the firm's *discriminability*, a competence measure. "
        "Heiner's tolerance limit plays the role of the signal-detection optimal criterion: the best threshold "
        "is where the curve's slope matches the value-weighted odds (1−π)D / (πG).")
    c = st.columns(3)
    meas = c[0].radio("Measure of a preferred exception", ["full", "persist", "static"], format_func=MEASURE_LABELS.get,
                      key="sdt_meas",
                      help="How a deviation's payoff is judged when deciding whether it was a preferred exception: "
                           "this period only, over H periods with rivals fixed, or over H periods with rivals reacting.")
    scope = c[1].radio("Firms", ["pool", "one"], format_func=lambda x: {"pool": "Pool all firms", "one": "One firm"}[x],
                       key="sdt_scope", help="Trace the ROC curve from every firm's decisions together, or from one firm.")
    fi = c[2].number_input("Firm (when one)", 1, base.n_firms, 1, key="sdt_firm",
                           help="Firm number (1 = most rigid) used when Firms = One firm.") - 1
    out = cached_single(to_json(base), H, cont, disc, split)
    h = out["hist"]
    burn = base.burn_in
    q_prev = np.vstack([h["q"][:1], h["q"][:-1]])
    signal = np.abs(h["rec"] - q_prev)[burn:]
    gain = h[{"full": "gain_full", "persist": "gain_persist", "static": "gain_static"}[meas]][burn:]
    opp = h["opportunity"][burn:]
    if scope == "one":
        sig, gn = signal[:, fi][opp[:, fi]], gain[:, fi][opp[:, fi]]
    else:
        sig, gn = signal[opp], gain[opp]
    roc = sdt_roc(sig, gn)
    if roc.empty:
        st.info("Not enough opportunities with both outcomes to trace an ROC curve.")
    else:
        disc_auc = auc(sig, gn > 0)
        best = roc.loc[roc["net_per_opp"].idxmax()]
        m = st.columns(4)
        m[0].metric("Discriminability (AUC of signal)", f"{disc_auc:.3f}", help="0.5 = the size of the recommended change "
                    "says nothing about whether deviating pays; 1 = perfectly diagnostic.")
        m[1].metric("π (share of opportunities that are exceptions)", f"{(gn > 0).mean():.3f}")
        m[2].metric("Value-maximizing threshold θ*", "deviate always" if np.isinf(best.theta) else f"{best.theta:.1f}")
        m[3].metric("Net value per opportunity at θ*", f"{best.net_per_opp:,.1f}")
        c1, c2 = st.columns(2)
        fr = go.Figure()
        fr.add_trace(go.Scatter(x=roc["w"], y=roc["r"], mode="lines+markers", name="ROC: deviate if |q*−q| > θ",
                                line=dict(color=CAT[0], width=2), marker=dict(size=5),
                                customdata=np.stack([roc["theta"].replace(-np.inf, 0), roc["net_per_opp"]], axis=1),
                                hovertemplate="θ = %{customdata[0]:.1f}<br>w = %{x:.2f}, r = %{y:.2f}<br>"
                                              "net/opp %{customdata[1]:.1f}<extra></extra>"))
        fr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="rgba(128,128,128,0.8)", dash="dot", width=1)))
        fr.add_trace(go.Scatter(x=[best.w], y=[best.r], mode="markers", name="Value-maximizing θ*",
                                marker=dict(size=13, color=CAT[1], symbol="star", line=dict(color="white", width=1.5))))
        fr.update_xaxes(title="w(θ) = false-alarm rate (type II errors)", range=[0, 1.02])
        fr.update_yaxes(title="r(θ) = hit rate (1 − type I)", range=[0, 1.02])
        c1.plotly_chart(style(fr, 400, "ROC of the firm's decision signal"))
        fv = go.Figure()
        th = roc[np.isfinite(roc["theta"])]
        fv.add_trace(go.Scatter(x=th["theta"], y=th["net_per_opp"], mode="lines", name="Net value per opportunity",
                                line=dict(color=CAT[2], width=2)))
        fv.add_hline(y=0, line=dict(color="rgba(128,128,128,0.8)", width=1))
        fv.update_xaxes(title="Threshold θ on |q* − q|"); fv.update_yaxes(title="Net value of the rule vs B per opportunity")
        c2.plotly_chart(style(fv, 400, "Choosing how selective to be"))
        st.caption("Based on one run of the sidebar scenario (Market lab settings). Positive net value at θ* > 0 with "
                   "negative value at θ = −∞ (deviate always) is Heiner's point: a constrained rule beats full "
                   "flexibility.")
