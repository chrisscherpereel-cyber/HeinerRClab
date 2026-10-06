import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heiner_abm.calibration import (FORECAST_RULES, QUANTITY_RULES, fit_subjects, prepare_forecasts,
                                    prepare_quantities, summarise, synthetic_cournot_data, synthetic_forecast_data)
from ui.common import CAT, download, hypothesis_card, replication_notice, style

st.title("Calibration: which theory predicts people best?")
st.caption("Each theory's decision rule is fitted to laboratory data and judged by how well it predicts choices it "
           "has not seen. Two experimental formats are supported: learning-to-forecast cobweb markets (the design "
           "of Hommes et al. 2007) and quantity-setting Cournot markets (the design of Huck et al. 1999).")

hypothesis_card(
    "CALIB",
    "For every subject and every rule, the rule's parameters are fitted on the **first half** of that subject's "
    "periods (grid search, squared error) and the rule then predicts the **second half**. Rules are ranked by "
    "out-of-sample error, and each subject is classified by the rule that predicts them best. Rules with more "
    "parameters fit the first half better, but gain nothing out of sample unless they capture real behavior.")
replication_notice("calibration")

st.page_link("app_pages/empirical.py", label="Five public datasets, with a validation protocol fixed in advance, are on "
             "the Empirical validation page", icon="🗄️")
with st.expander("Getting the published data", icon="📂"):
    st.markdown(
        "The published datasets are not bundled with this app: their licenses require a request to the authors or "
        "the archive.\n\n"
        "* **Cournot oligopoly (Huck, Normann & Oechssler 1999):** archived in heiDATA, the Heidelberg research data "
        "repository, [doi:10.11588/data/10012](https://doi.org/10.11588/data/10012) (access on request).\n"
        "* **Cobweb learning-to-forecast (Hommes, Sonnemans, Tuinstra & van de Velden 2007):** available from the "
        "authors (CeNDEF, University of Amsterdam).\n\n"
        "Convert the data to one row per subject and period with the columns below, then upload it. Until then, the "
        "synthetic data (clearly labeled) show that the pipeline recovers known rule types.")
    st.markdown("**Forecasting format:** `group, subject, period, price, forecast`, where *forecast* is the "
                "subject's forecast of that period's price, made before the price was known.  \n"
                "**Cournot format:** `group, subject, period, quantity`, plus the market's inverse demand "
                "P = a − b·Q and unit cost c entered below.")

st.header("1 · Data", divider="gray")
fmt = st.radio("Experimental format", ["Learning-to-forecast (cobweb)", "Quantity setting (Cournot)"],
               horizontal=True, key="cal_fmt",
               help="Forecasting experiments ask subjects for price forecasts; Cournot experiments ask for outputs.")
forecast = fmt.startswith("Learning")
src = st.radio("Source", ["Synthetic data with known rule types", "Upload a CSV"], horizontal=True, key="cal_src",
               help="Synthetic data are simulated subjects who follow known rules with noise. They check the method; "
                    "they are not evidence about people.")
truth = None
extra = {}
if src.startswith("Synthetic"):
    c1, c2 = st.columns(2)
    noise = c1.slider("Decision noise", 0.0, 3.0, 1.0 if not forecast else 0.3, 0.1, key="cal_noise",
                      help="Standard deviation of the noise added to each synthetic subject's rule.")
    seed = c2.number_input("Seed", 0, 9999, 0, key="cal_seed", help="Random seed for the synthetic subjects.")
    if forecast:
        df, truth = synthetic_forecast_data(noise=noise, seed=int(seed))
    else:
        df, truth = synthetic_cournot_data(noise=noise, seed=int(seed))
        extra = dict(a=100.0, b=1.0, c=1.0, n=4)
    st.warning("Synthetic data: simulated subjects with known rules, used to check the method. Not evidence about "
               "human behavior.", icon="🧪")
else:
    up = st.file_uploader("CSV file", type="csv", key="cal_upload",
                          help="One row per subject and period, with the columns listed above.")
    if up is None:
        st.info("Upload a file to continue.", icon="ℹ️")
        st.stop()
    df = pd.read_csv(up)
    need = {"group", "subject", "period"} | ({"price", "forecast"} if forecast else {"quantity"})
    if not need <= set(df.columns):
        st.error(f"The file needs the columns {', '.join(sorted(need))}.")
        st.stop()
    if not forecast:
        c = st.columns(4)
        extra = dict(a=c[0].number_input("Demand intercept a", value=100.0, key="cal_a",
                                         help="Inverse demand P = a − b·Q."),
                     b=c[1].number_input("Demand slope b", value=1.0, min_value=1e-6, key="cal_b",
                                         help="Inverse demand P = a − b·Q."),
                     c=c[2].number_input("Unit cost c", value=1.0, key="cal_c", help="Constant marginal cost."),
                     n=c[3].number_input("Firms per market", value=int(df.groupby("group")["subject"].nunique()
                                                                        .median()), min_value=2, key="cal_n",
                                         help="Number of firms in each market, used for the Cournot–Nash output."))
st.caption(f"{df['subject'].nunique()} subjects in {df['group'].nunique()} groups, "
           f"{int(df.groupby('subject')['period'].count().median())} periods per subject.")

if st.button("Fit and compare rules", type="primary", key="cal_run"):
    with st.spinner("Fitting every rule to every subject"):
        data = prepare_forecasts(df) if forecast else prepare_quantities(df, extra["a"], extra["b"], extra["c"])
        fits = fit_subjects(data, FORECAST_RULES if forecast else QUANTITY_RULES,
                            "forecast" if forecast else "quantity", extra)
        st.session_state["cal_result"] = (fmt, src, fits)
stored = st.session_state.get("cal_result")
if stored is None or stored[:2] != (fmt, src):
    st.stop()
fits = stored[2]
rank, best = summarise(fits)

st.header("2 · Out-of-sample ranking", divider="gray")
fig = go.Figure()
r = rank.sort_values("test_rmse", ascending=False)
fig.add_trace(go.Bar(y=r["name"], x=r["test_rmse"], orientation="h", name="Out of sample (second half)",
                     marker_color=CAT[0]))
fig.add_trace(go.Scatter(y=r["name"], x=r["train_rmse"], mode="markers", name="In sample (first half)",
                         marker=dict(color=CAT[1], size=9, symbol="diamond")))
fig.update_xaxes(title="Root-mean-square prediction error (mean over subjects)")
st.plotly_chart(style(fig, height=80 + 34 * len(r)), width="stretch")
st.dataframe(rank.rename(columns={"name": "Rule", "theory": "Theory", "test_rmse": "Out-of-sample RMSE",
                                  "train_rmse": "In-sample RMSE", "best_for_share": "Best for (share of subjects)"})
             .drop(columns=["rule"]), hide_index=True, width="stretch",
             column_config={"Best for (share of subjects)": st.column_config.ProgressColumn(format="percent",
                                                                                           min_value=0, max_value=1)})

st.header("3 · Classifying subjects", divider="gray")
by_theory = best.groupby("theory").size().rename("subjects").reset_index().sort_values("subjects", ascending=False)
c1, c2 = st.columns([1, 2])
c1.dataframe(by_theory.rename(columns={"theory": "Theory", "subjects": "Subjects best predicted"}),
             hide_index=True, width="stretch")
if truth is not None:
    m = best.merge(truth, on=["group", "subject"])
    acc = float((m["rule"] == m["true_rule"]).mean())
    c2.metric("Recovery of the true rule", f"{acc:.0%}",
              help="Share of synthetic subjects whose best-predicting rule is the rule that generated them. Chance "
                   f"is about {1 / len(FORECAST_RULES if forecast else QUANTITY_RULES):.0%}.")
    c2.dataframe(pd.crosstab(m["true_rule"], m["rule"]), width="stretch")
    c2.caption("Rows: the rule that generated each synthetic subject. Columns: the rule that predicts it best out "
               "of sample. Rules that make similar predictions over a short series are hard to tell apart: the "
               "reliability condition, for example, is easily mistaken for an inaction band or for keeping the "
               "previous decision.")
download(fits, "calibration_fits.csv", "Download all fits (CSV)")
