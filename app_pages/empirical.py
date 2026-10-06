import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm import registered
from heiner_abm.calibration import synthetic_cournot_data, synthetic_forecast_data
from heiner_abm.datasets import EGM_BREAKS, GOS_MARKET, LOADERS
from heiner_abm.empirical import (DATASETS, EMPIRICAL_PLAN, NewsvendorSpec, apply_mapping, break_analysis,
                                  evaluate_details, evaluate_empirical, read_table, run_cournot, run_forecast,
                                  run_newsvendor, run_time_pressure, synthetic_newsvendor_data)
from heiner_abm.literature import REFERENCES
from ui.common import CAT, download, hypothesis_card, style, verdict, prereg_explainer
from ui.common import evidence_note, set_engine

st.title("Empirical validation: public experimental datasets")
evidence_note("behavioral")
set_engine("none")
st.caption("Five public datasets can test whether the simulated decision rules describe human choices. This page "
           "documents each source, fixes the validation protocol in advance, and runs the analyses once the data "
           "are uploaded.")

hypothesis_card(
    "EMPVAL",
    "The protocol follows the recommendations for these datasets: one primary Cournot validation, one independent "
    "forecasting validation and the newsvendor data as an extension, with the time-pressure data as supplementary "
    "evidence. Rules are fitted on each participant's earlier periods and evaluated on later ones, every treatment "
    "is analyzed with the rules its information set allows, and uncertainty is reported at the participant level. "
    "A rule that predicts people well shows which decision rule describes their behavior, not that they consciously "
    "apply Heiner's reliability condition.")

# ------------------------------------------------------------------------------------------------ sources
st.header("1 · The datasets", divider="gray")
st.dataframe(pd.DataFrame([dict(Priority=d.priority, Dataset=d.title, Study=REFERENCES[d.ref].cite,
                                Adapter=d.adapter, Validates=d.validates, Qualification=d.qualification)
                           for d in DATASETS]), hide_index=True, width="stretch")
for d in DATASETS:
    with st.expander(f"{d.priority} · {d.title}"):
        st.markdown(f"**Source:** [{d.url}]({d.url})  \n**Access:** {d.access}  \n**License:** {d.licence}  \n"
                    f"**Files:** {d.files}")
        st.markdown(f"**What it can validate:** {d.validates}")
        st.markdown(f"**Qualification:** {d.qualification}")
        st.markdown(f"**Recommended use:** {d.recommended}")
        st.caption(f"Study: {REFERENCES[d.ref].apa}")
st.info("The data are not bundled with the app: each repository's license governs reuse and redistribution, and "
        "two require a login. Download the files from the sources above and upload them below **as downloaded** "
        "(the zip archives work): the app recognizes each format and applies the market parameters derived from the "
        "files. Other data can be uploaded with a column mapping.", icon="📂")

# ------------------------------------------------------------------------------------------------ protocol
st.header("2 · Validation protocol", divider="gray")
prereg_explainer()
c1, c2 = st.columns([2, 1])
c1.markdown(f"**Plan hash:** `{EMPIRICAL_PLAN.digest}` · fit on the first {EMPIRICAL_PLAN.split:.0%} of each "
            f"participant's periods, evaluate on the rest · participant-cluster bootstrap ({EMPIRICAL_PLAN.n_boot} "
            f"draws) · generative check with {EMPIRICAL_PLAN.gen_reps} simulated replications")
c2.download_button("Download protocol (.json)", EMPIRICAL_PLAN.to_json().encode(),
                   file_name=f"empirical_{EMPIRICAL_PLAN.digest}.json", mime="application/json", width="stretch",
                   help="Register the protocol before analyzing the data.")
st.dataframe(pd.DataFrame(EMPIRICAL_PLAN.hypotheses, columns=["ID", "Hypothesis", "Decision rule"]),
             hide_index=True, width="stretch")

store = st.session_state.setdefault("emp_results", {})

# ------------------------------------------------------------------------------------------------ registered results
st.header("3 · Registered results on the five datasets", divider="gray")
if registered.EMPIRICAL_PLAN == EMPIRICAL_PLAN.digest:
    st.caption(f"The protocol above (`{EMPIRICAL_PLAN.digest}`) applied to the files as distributed by the five "
               "repositories. Each dataset and treatment is tested separately; a hypothesis counts as supported only "
               "if it holds in every one.")
    for hid, hyp, _ in EMPIRICAL_PLAN.hypotheses:
        v = registered.EMPIRICAL_VERDICTS.get(hid, "not tested")
        lines = [f"* **{d}:** {r}" for i, d, r in registered.EMPIRICAL_DETAILS if i == hid]
        verdict({"supported": "support", "not supported": "reject"}.get(v, "neutral"),
                f"**{hid}** · {hyp}  \n{v.capitalize()}.  \n" + "\n".join(lines))
    st.markdown("**Exploratory (not pre-registered): which rules describe people best?**")
    st.dataframe(pd.DataFrame([dict(Data=k, Participants=v["participants"],
                                    **{"Best predicted by a restricted rule": v["restricted"],
                                       "Rule with the lowest out-of-sample error": v["top_rule"],
                                       "Its RMSE": v["top_rmse"]})
                               for k, v in registered.EMPIRICAL_RULES.items()]), hide_index=True, width="stretch",
                 column_config={"Best predicted by a restricted rule": st.column_config.ProgressColumn(
                     format="percent", min_value=0, max_value=1,
                     help="Keep the previous decision, inaction band or reliability condition.")})
    st.caption("Restricted rules describe more participants where information is poorer (aggregate rather than "
               "individual information about rivals) or competence is lower (high time pressure, V7). This pattern "
               "was not pre-registered and needs confirmation.")
else:
    st.warning("The protocol has changed since the registered run; rerun the analyses below.", icon="⚠️")


def _guess(cols, *names):
    low = {c.lower(): c for c in cols}
    for n in names:
        for c_low, c in low.items():
            if n in c_low:
                return c
    return None


SOURCES = ["Original files (as downloaded)", "Other data file (map columns)", "Synthetic demonstration"]


def choose_source(key: str) -> str:
    return st.radio("Source", SOURCES, horizontal=True, key=f"emp_{key}_src",
                    help="Original files: the archive or files exactly as downloaded from the repository. Other "
                         "data: any table with one row per participant and period, mapped to the required columns. "
                         "Synthetic demonstration: simulated participants with known rules (not evidence about people).")


def load_original(key: str, loader: str):
    ups = st.file_uploader("Files as downloaded (zip archives are fine)", accept_multiple_files=True,
                           key=f"emp_{key}_orig", help="Upload the repository's archive, or the data files inside it.")
    if not ups:
        return None
    try:
        return LOADERS[loader]([(u.name, u.getvalue()) for u in ups])
    except Exception as e:      # a wrong file should not take the page down
        st.error(f"Could not read these files: {e}")
        return None


def load_mapped(key: str, required: dict):
    up = st.file_uploader("Data file", type=["csv", "txt", "xlsx", "xls", "dta", "mat"], key=f"emp_{key}_file",
                          help="CSV, Excel (every sheet), Stata (.dta) or MATLAB (.mat) files. One row per "
                               "participant and period is needed; reshape wide tables before uploading.")
    if up is None:
        return None
    try:
        tables = read_table(up.name, up.getvalue())
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None
    if not tables:
        st.error("No usable table found in the file.")
        return None
    name = st.selectbox("Table", list(tables), key=f"emp_{key}_table", help="Sheet or variable to analyze.")
    raw = tables[name]
    st.caption(f"{len(raw)} rows · columns: {', '.join(map(str, raw.columns[:30]))}")
    cols = [None] + list(raw.columns)
    mapping = {}
    cs = st.columns(min(4, len(required)))
    for j, (target, (label, hints)) in enumerate(required.items()):
        g = _guess(raw.columns, *hints)
        mapping[target] = cs[j % len(cs)].selectbox(label, cols, index=cols.index(g) if g in cols else 0,
                                                    key=f"emp_{key}_map_{target}", help=f"Column holding {label.lower()}.")
    missing = [required[k][0] for k in required if not mapping.get(k)]
    if missing:
        st.info("Map the required columns: " + ", ".join(missing) + ".", icon="ℹ️")
        return None
    treat = st.selectbox("Treatment column (optional)", cols, key=f"emp_{key}_treat",
                         help="Analyze one treatment at a time so that rules match its information set.")
    df = apply_mapping(raw, mapping, [treat] if treat else [])
    if treat:
        val = st.selectbox("Treatment to analyze", sorted(df[treat].astype(str).unique()), key=f"emp_{key}_treatval",
                           help="Only this treatment is analyzed.")
        df = df[df[treat].astype(str) == val]
    for col in ("group", "subject"):
        if col not in df:
            df[col] = 0
    return df


def show_ranking(res: dict, truth=None):
    rank = res["rank"]
    r = rank.sort_values("test_rmse", ascending=False)
    fig = go.Figure(go.Bar(y=r["name"], x=r["test_rmse"], orientation="h", marker_color=CAT[0],
                           name="Out of sample"))
    fig.add_trace(go.Scatter(y=r["name"], x=r["train_rmse"], mode="markers", name="In sample",
                             marker=dict(color=CAT[1], size=9, symbol="diamond")))
    fig.update_xaxes(title="Prediction error (RMSE, mean over participants)")
    st.plotly_chart(style(fig, height=80 + 32 * len(r)), width="stretch")
    c1, c2 = st.columns(2)
    by = res["best"].groupby("theory").size().rename("participants").reset_index().sort_values("participants",
                                                                                               ascending=False)
    c1.dataframe(by.rename(columns={"theory": "Theory", "participants": "Participants best predicted"}),
                 hide_index=True, width="stretch")
    g = res["gain_rc"]
    c2.metric("RC vs the flexible rule it restricts", f"{g['mean']:+.3f}",
              help="Mean reduction in out-of-sample error from restricting the flexible rule with the reliability "
                   f"condition (positive = RC predicts better); 95% CI {g['lo']:+.3f} to {g['hi']:+.3f}; RC better "
                   f"for {g['share']:.0%} of participants.")
    gb = res.get("gain_band")
    if gb:
        c2.metric("Inaction band vs the flexible rule", f"{gb['mean']:+.3f}", help=f"95% CI {gb['lo']:+.3f} to "
                                                                                    f"{gb['hi']:+.3f}.")
    if truth is not None:
        m = res["best"].merge(truth)
        st.caption(f"Synthetic check: the generating rule is recovered for {(m['rule'] == m['true_rule']).mean():.0%} "
                   "of simulated participants.")


def keep(kind: str, label: str, res: dict):
    store.setdefault(kind, {})[label] = res


def results_of(kind: str) -> dict:
    return store.get(kind, {})


# ------------------------------------------------------------------------------------------------ analyses
st.header("4 · Run the analyses", divider="gray")
tabs = st.tabs(["Cournot (primary)", "Forecasting (independent)", "Newsvendor (extension)",
                "Time pressure (supplementary)"])

with tabs[0]:
    st.markdown("**Data:** Gomez-Martinez, Onderstal & Sonnemans (2016). Only part 1 (periods 1–25, **no "
                "communication**) is used; each information treatment is analyzed separately.")
    src = choose_source("cournot")
    df, truth, label, info, mkt = None, None, "", True, dict(GOS_MARKET)
    if src == SOURCES[0]:
        raw = load_original("cournot", "cournot_gos")
        if raw is not None:
            st.caption(f"{raw['subject'].nunique()} participants in {raw['group'].nunique()} markets; the demand model "
                       f"P_i = 150 − q_i − (2/3)·Q_−i with unit cost 2 reproduces {raw['demand_check'].mean():.0%} of the "
                       "recorded prices.")
            tr = st.radio("Information treatment", ["aggregate", "individual"], horizontal=True,
                          key="emp_cournot_tr", help="Aggregate: participants saw rivals' total output only, so the "
                                                     "imitate-the-best rule is excluded. Individual: they saw each "
                                                     "rival's output and profit.")
            df = raw[(raw["part"] == 1) & (raw["treatment"] == tr)][["group", "subject", "period", "quantity"]]
            label, info = f"Cournot, {tr} information", tr == "individual"
    else:
        if src == SOURCES[1]:
            df = load_mapped("cournot", {"group": ("Market / group", ("group", "market", "matching")),
                                         "subject": ("Participant", ("subject", "participant", "id", "player")),
                                         "period": ("Period", ("period", "round")),
                                         "quantity": ("Quantity", ("quantity", "output", "decision"))})
            label = "Cournot (uploaded)"
        else:
            df, truth = synthetic_cournot_data(n_groups=st.session_state.get("emp_cournot_groups", 6))
            label = "Cournot (synthetic)"
            mkt = dict(a=100.0, b=1.0, gamma=1.0, c=1.0, n=4)
            st.warning("Synthetic demonstration: not evidence about people.", icon="🧪")
        c = st.columns(5)
        mkt["a"] = c[0].number_input("Demand intercept a", value=float(mkt["a"]), key="emp_cournot_a",
                                     help="Each firm's inverse demand P_i = a − b(q_i + γ·Q_−i).")
        mkt["b"] = c[1].number_input("Demand slope b", value=float(mkt["b"]), min_value=1e-6, key="emp_cournot_b",
                                     help="Own-quantity slope.")
        mkt["gamma"] = c[2].number_input("Substitutability γ", value=float(mkt["gamma"]), min_value=0.0,
                                         max_value=1.0, key="emp_cournot_g", help="1 = homogeneous product.")
        mkt["c"] = c[3].number_input("Unit cost c", value=float(mkt["c"]), key="emp_cournot_c", help="Marginal cost.")
        mkt["n"] = int(c[4].number_input("Firms per market", value=int(mkt["n"]), min_value=2, key="emp_cournot_n",
                                         help="For the Cournot–Nash benchmark."))
        info = st.checkbox("Individual information", value=True, key="emp_cournot_info",
                           help="Did participants see each rival's quantity and profit? If not, the imitate-the-best "
                                "rule is excluded.")
    gen = st.checkbox("Run the generative check (V2)", value=True, key="emp_cournot_gen",
                      help="Re-run every market with each participant replaced by its best-fitting rule.")
    if df is not None and len(df) and st.button("Analyze", type="primary", key="emp_cournot_run"):
        with st.spinner("Fitting rules and simulating markets"):
            keep("cournot", label, dict(run_cournot(df, mkt["a"], mkt["b"], mkt["c"], mkt["n"], info, gen,
                                                    gamma=mkt["gamma"]), synthetic=src == SOURCES[2], truth=truth))
    for lab, res in results_of("cournot").items():
        with st.expander(f"Results · {lab}", expanded=True):
            show_ranking(res, res.get("truth"))
            hm = res["moments"]
            rows = [dict(Statistic=k.replace("_", " "), Human=hm[k]) for k in hm]
            if res.get("sim") is not None and len(res["sim"]):
                for r_ in rows:
                    k = r_["Statistic"].replace(" ", "_")
                    if k in res["sim"]:
                        r_["Simulated 2.5%"], r_["Simulated 97.5%"] = np.percentile(res["sim"][k].dropna(), [2.5, 97.5])
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            download(res["fits"], "cournot_fits.csv", "Download fits (CSV)")

with tabs[1]:
    st.markdown("**Data:** Evans, Gibbs & McGough (2025) (one-period-ahead forecasts, announced structural changes) "
                "and the Anufriev & Hommes (2012) replication package (asset-pricing experiments, two-period-ahead "
                "forecasts).")
    src = choose_source("forecast")
    df, truth, label, h, brk = None, None, "", 1, 0
    fund = None
    if src == SOURCES[0]:
        which = st.radio("Dataset", ["Evans, Gibbs & McGough", "Anufriev & Hommes"], horizontal=True,
                         key="emp_forecast_ds", help="Which repository's files are uploaded.")
        raw = load_original("forecast", "ltf_egm" if which.startswith("Evans") else "ltf_ah")
        if raw is not None and which.startswith("Evans"):
            fb = st.radio("Feedback", ["negative", "positive"], horizontal=True, key="emp_forecast_fb",
                          help="Negative feedback (cobweb-like, treatments 1–6) or positive feedback (7–8).")
            df = raw[raw["feedback"] == fb]
            label = f"EGM, {fb} feedback"
            tr_break = st.selectbox("Structural-change analysis for treatment", sorted(EGM_BREAKS),
                                    key="emp_forecast_trb", help="Treatments with an announced structural change.")
            brk = EGM_BREAKS[tr_break][0]
            brk_df = raw[raw["treatment"] == tr_break]
        elif raw is not None:
            df, h, label = raw, 2, "Anufriev–Hommes asset markets"
            st.caption("Forecast horizon 2 and the fundamental price of each group (60, or 40 in groups 8–10 of HSTV05) "
                       "are applied automatically.")
    else:
        if src == SOURCES[1]:
            df = load_mapped("forecast", {"group": ("Market / group", ("group", "market")),
                                          "subject": ("Participant", ("subject", "participant", "id")),
                                          "period": ("Period", ("period", "round")),
                                          "price": ("Realized price", ("price",)),
                                          "forecast": ("Forecast of that period's price", ("forecast", "expect"))})
            label = "Forecasting (uploaded)"
        else:
            df, truth = synthetic_forecast_data(n_groups=st.session_state.get("emp_forecast_groups", 4))
            label = "Forecasting (synthetic)"
            st.warning("Synthetic demonstration: not evidence about people.", icon="🧪")
        c = st.columns(3)
        h = c[0].selectbox("Forecast horizon", [1, 2], key="emp_forecast_h",
                           help="1 for cobweb and production-lag markets; 2 for asset-pricing experiments.")
        f_ = c[1].number_input("Fundamental price (0 = unknown)", value=0.0, key="emp_forecast_fund",
                               help="The rational-expectations benchmark; if unknown, the training-window mean.")
        fund = f_ or None
        brk = c[2].number_input("Structural change in period (0 = none)", value=0, min_value=0,
                                key="emp_forecast_break", help="Compares errors and forecast changes around it.")
        brk_df = df
    if df is not None and len(df) and st.button("Analyze", type="primary", key="emp_forecast_run"):
        with st.spinner("Fitting rules"):
            out = run_forecast(df, int(h), fund)
            out["break"] = break_analysis(brk_df, int(brk)) if brk else None
            keep("forecast", label, dict(out, synthetic=src == SOURCES[2], truth=truth))
    for lab, res in results_of("forecast").items():
        with st.expander(f"Results · {lab}", expanded=True):
            show_ranking(res, res.get("truth"))
            ms = res["modal"]
            st.caption(f"Most common best rule: **{ms['rule']}**, best for {ms['share']:.0%} of participants "
                       f"(95% CI {ms['lo']:.0%}–{ms['hi']:.0%}).")
            if res.get("break") is not None:
                st.markdown("**Around the (announced) structural change**")
                st.dataframe(res["break"], hide_index=True, width="stretch")
            download(res["fits"], "forecast_fits.csv", "Download fits (CSV)")

with tabs[2]:
    st.markdown("**Data:** Brokesova, Deck & Peliova (2022), newsvendor treatments (uniform demand 0–100, price "
                "100, unit cost 25 or 75).")
    src = choose_source("nv")
    df, truth, label = None, None, ""
    if src == SOURCES[0]:
        raw = load_original("nv", "newsvendor_bdp")
        if raw is not None:
            cost = st.radio("Unit cost", [25.0, 75.0], horizontal=True, key="emp_nv_cost",
                            format_func=lambda x: f"{x:g} (optimal order {100 - x:g})",
                            help="Low cost: critical fractile 0.75; high cost: 0.25.")
            spec = NewsvendorSpec(dist="uniform", low=0, high=100, co=cost, cu=100 - cost)
            df = raw[raw["unit_cost"] == cost][["subject", "period", "order", "demand"]]
            label = f"Newsvendor, {'low' if cost == 25 else 'high'} cost (optimal {100 - cost:g})"
    else:
        c = st.columns(4)
        dist = c[0].selectbox("Demand distribution", ["uniform", "normal"], key="emp_nv_dist", help="As in the data.")
        if dist == "uniform":
            kw = dict(dist="uniform", low=c[1].number_input("Lowest demand", value=0.0, key="emp_nv_low",
                                                             help="Uniform lower bound."),
                      high=c[2].number_input("Highest demand", value=100.0, key="emp_nv_high",
                                             help="Uniform upper bound."))
        else:
            kw = dict(dist="normal", mean=c[1].number_input("Mean demand", value=50.0, key="emp_nv_mean",
                                                             help="Normal mean."),
                      sd=c[2].number_input("S.d. of demand", value=20.0, min_value=0.01, key="emp_nv_sd",
                                           help="Normal s.d."))
        co = c[3].number_input("Overage cost (unit cost)", value=25.0, min_value=0.0, key="emp_nv_co",
                               help="Cost of a unit left over.")
        cu = st.number_input("Underage cost (price − unit cost)", value=75.0, min_value=0.0, key="emp_nv_cu",
                             help="Profit lost per unit of unmet demand.")
        spec = NewsvendorSpec(co=co, cu=cu, **kw)
        if src == SOURCES[1]:
            df = load_mapped("nv", {"subject": ("Participant", ("subject", "participant", "id")),
                                    "period": ("Period", ("period", "round")),
                                    "order": ("Order quantity", ("order", "choice", "quantity")),
                                    "demand": ("Realized demand", ("demand", "realization"))})
            label = "Newsvendor (uploaded)"
        else:
            df, truth = synthetic_newsvendor_data(spec, n_subjects=st.session_state.get("emp_nv_subjects", 12))
            label = "Newsvendor (synthetic)"
            st.warning("Synthetic demonstration: not evidence about people.", icon="🧪")
    if df is not None and len(df):
        st.caption(f"Optimal order {spec.optimal:.1f}; mean demand {spec.mu:.1f}; critical fractile "
                   f"{spec.fractile:.2f}.")
        if st.button("Analyze", type="primary", key="emp_nv_run"):
            with st.spinner("Fitting rules"):
                keep("newsvendor", label, dict(run_newsvendor(df, spec), synthetic=src == SOURCES[2], truth=truth))
    for lab, res in results_of("newsvendor").items():
        with st.expander(f"Results · {lab}", expanded=True):
            show_ranking(res, res.get("truth"))
            s_ = res["stats"]
            c = st.columns(3)
            c[0].metric("Pull-to-center ratio", f"{s_['ptc'][0]:.2f}",
                        help=f"(mean order − mean demand)/(optimal − mean demand): 1 = optimal, 0 = mean demand. "
                             f"95% CI {s_['ptc'][1]:.2f}–{s_['ptc'][2]:.2f}.")
            c[1].metric("Demand-chasing slope", f"{s_['chase'][0]:+.2f}",
                        help=f"Order change per unit of last demand minus order; 95% CI {s_['chase'][1]:+.2f} to "
                             f"{s_['chase'][2]:+.2f}.")
            c[2].metric("Periods with an order change", f"{s_['change'][0]:.0%}", help="How often orders change.")
            download(res["fits"], "newsvendor_fits.csv", "Download fits (CSV)")

with tabs[3]:
    st.markdown("**Data:** time-pressure learning-to-forecast experiments (13 sessions; the same participants "
                "under high and low time pressure). Time pressure manipulates **competence**, not uncertainty.")
    src = choose_source("time")
    df, h = None, 2
    if src == SOURCES[0]:
        df = load_original("time", "ltf_time")
        if df is not None:
            st.caption(f"{df['subject'].nunique()} participants; forecast horizon 2 (each forecast targets the next "
                       "round's price). Rounds without a forecast are dropped.")
    elif src == SOURCES[1]:
        df = load_mapped("time", {"group": ("Market / group", ("group", "market")),
                                  "subject": ("Participant", ("subject", "participant", "id")),
                                  "period": ("Period", ("period", "round")),
                                  "price": ("Realized price", ("price",)),
                                  "forecast": ("Forecast of that period's price", ("forecast", "expect")),
                                  "pressure": ("Time-pressure condition", ("pressure", "time", "treat"))})
        h = st.selectbox("Forecast horizon", [1, 2], index=1, key="emp_time_h", help="2 for asset-pricing designs.")
    else:
        d0, _ = synthetic_forecast_data(n_groups=4)
        df = pd.concat([d0.assign(pressure="high"), d0.assign(pressure="low",
                                                               forecast=d0["forecast"] + np.random.default_rng(1)
                                                               .normal(0, 0.3, len(d0)))], ignore_index=True)
        h = 1
        st.warning("Synthetic demonstration: not evidence about people.", icon="🧪")
    if df is not None and len(df):
        vals = sorted(df["pressure"].astype(str).unique())
        c = st.columns(2)
        hi_v = c[0].selectbox("High time pressure", vals, index=0, key="emp_time_hi", help="Condition value.")
        lo_v = c[1].selectbox("Low time pressure", vals, index=min(1, len(vals) - 1), key="emp_time_lo",
                              help="Condition value.")
        if st.button("Compare conditions", type="primary", key="emp_time_run"):
            with st.spinner("Fitting rules in both conditions"):
                keep("time", "Time pressure (high vs low)" + (" (synthetic)" if src == SOURCES[2] else ""),
                     dict(run_time_pressure(df, int(h), "pressure", hi_v, lo_v), synthetic=src == SOURCES[2]))
    for lab, res in results_of("time").items():
        if res.get("diff"):
            st.metric(f"{lab}: simple or restricted rules best, high − low", f"{res['diff'][0]:+.0%}",
                      help=f"High {res['high']:.0%}, low {res['low']:.0%}; 95% CI {res['diff'][1]:+.0%} to "
                           f"{res['diff'][2]:+.0%}" + (" (paired by participant)." if res.get("paired") else "."))

# ------------------------------------------------------------------------------------------------ verdicts
st.header("5 · Verdicts for this session's analyses", divider="gray")
if not store:
    st.info("Run an analysis above to see the verdicts for the data analyzed in this session.", icon="ℹ️")
else:
    if any(r.get("synthetic") for v in store.values() for r in v.values()):
        st.warning("Some analyses use synthetic demonstration data. Their verdicts show how the protocol works and "
                   "are not evidence about human behavior.", icon="🧪")
    for _, r in evaluate_empirical(store).iterrows():
        kind = {"supported": "support", "not supported": "reject"}.get(r["verdict"], "neutral")
        verdict(kind, f"**{r['id']}** · {r['hypothesis']}  \n{r['verdict'].capitalize()}: {r['result']}")
